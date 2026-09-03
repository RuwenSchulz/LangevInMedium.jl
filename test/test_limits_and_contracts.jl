#!/usr/bin/env julia
# ==============================================================================================
# test_limits_and_contracts.jl — the LIMITS and the INPUT CONTRACT.
#
# The 0.2.1 audit covered every function and 0.2.2 the p_z frame. Neither asked what the engine
# does when it is asked for a LIMIT (D_sT → 0, glued to the flow, free streaming) or handed an
# input outside its assumptions (a table that ends before final_time, a non-uniform density grid,
# m ≤ 0). Seven defects were found that way on 2026-09-02, measured, and FIXED in 0.2.3; this file
# is what holds each fix in place. Every number in a comment below is the PRE-FIX measurement, so
# a regression has something specific to fail against.
#
# Fast: N ≤ 20 000 and runs of a few thousand steps; ≈ 40 s. Part of the LIM_FAST set.
# ==============================================================================================
using Test, Random, Statistics, Logging, LangevInMedium

const M_L  = 1.5
const DST_L = 0.11634
const RG_L = collect(0.0:0.5:20.0)
const PG_L = collect(0.05:0.1:8.0)
const DENS_L = ones(length(PG_L), length(RG_L))

"Uniform bath T with a radial flow rising to `v` and flat beyond r = 6 fm."
function flow_box(T, v; t0 = 0.4, tf = 3.0, rmax = 20.0)
    xg = collect(0.0:0.25:rmax); tg = collect(t0:0.1:tf)
    (xg, tg, fill(T, length(xg), length(tg)), [v * min(1.0, r / 6.0) for r in xg, _ in tg])
end
sim(bk, xg, tg, Tf, Vf; kw...) = (Random.seed!(11);
    simulate_ensemble_bulk(bk, RG_L, PG_L, DENS_L, Tf, Vf, (xg, tg); kw...))

@testset "L — limits and input contract" begin

# ── L1: the glued-to-the-flow limit zeroes every extra momentum row ────────────────────────────
# `kernel_set_to_fluid_velocity_*` used to write momentum rows 1 and 2 only, on both backends. With
# `momentum_dimensions = 3` row 3 kept whatever the IC put there and still entered
# E = √(m² + p_⊥² + p_z*²), so the particle that is supposed to BE the fluid element streamed
# slower than it: MEASURED at v = 0.5, T = 0.30, τ ∈ [0.4, 2.4], ⟨v_x⟩ = 0.4645 against the fluid's
# 0.5000, a −7.10 % deficit (CPU and GPU agreeing to six digits), with ⟨p_z*²⟩ = 0.596 GeV² alive at
# the end. The fluid is longitudinally comoving in Milne by construction, so the glued state is
# p_z* = 0 exactly. The old gate could not see it — `test_momentum_dims3.jl` "(R)" uses a ZERO-flow
# box, where the deficit vanishes identically. This one runs at v = 0.5.
@testset "L1 the glued limit reproduces the fluid velocity at any momentum_dimensions" begin
    v = 0.5; N = 10_000
    xg, tg, Tf, Vf = flow_box(0.30, v)
    x0 = zeros(2, N); x0[1, :] .= 8.0; p0 = zeros(2, N)
    for (pdim, kw) in ((2, (;)), (3, (;)), (3, (; pz_init = :comoving)))
        t, mom, pos = sim(CPUBackend(), xg, tg, Tf, Vf; x_init = x0, p_init = p0, N_particles = N,
            Δt = 1e-3, initial_time = 0.4, final_time = 2.4, save_interval = 1.0, m = M_L,
            DsT = DST_L, dimensions = 2, momentum_dimensions = pdim, momentum_langevin = false, kw...)
        vx = mean(pos[end][1, :] .- pos[end-1][1, :]) / (t[end] - t[end-1])
        @test isapprox(vx, v; rtol = 1e-6)                       # glued means glued
        pdim == 3 && @test all(iszero, mom[end][3, :])           # row 3 is zeroed, not carried
    end
end

# ── L2: the three D_sT limits are three different things ───────────────────────────────────────
# `DsT == 0.0` branches into `kernel_set_to_fluid_velocity_*`, i.e. `p = m·γ·v`: the COMOVING limit
# at zero temperature. It is NOT free streaming — three places in the tree said it was, and until
# 2026-09-02 the only thing that free-streamed was a NEGATIVE DsT, by accident, through
# `tau_drag ≤ 0 ⇒ η_D = κ = 0` (now refused, see L3). `collision_mode = :none` is the explicit
# free-streaming path added with the fix. Nor is `DsT = 0` the `DsT → 0⁺` limit: that one
# thermalises WITH the fluid and keeps the Jüttner spread, ⟨p_x⟩ = γv⟨E*⟩ = 1.0350 here against
# m·γ·v = 0.86603 — a 19.5 % gap.
@testset "L2 DsT = 0 (comoving), DsT to 0+ (thermal comoving) and :none (free streaming)" begin
    v = 0.5; N = 20_000; γ = 1 / sqrt(1 - v^2)
    xg, tg, Tf, Vf = flow_box(0.30, v)
    x0 = zeros(2, N); x0[1, :] .= 8.0
    p0 = zeros(2, N); p0[1, :] .= 1.0                            # a definite non-thermal LRF momentum
    px(; kw...) = mean(sim(CPUBackend(), xg, tg, Tf, Vf; x_init = x0, p_init = p0, N_particles = N,
        Δt = 1e-3, initial_time = 0.4, final_time = 2.4, save_interval = 1.0, m = M_L,
        dimensions = 2, kw...)[2][end][1, :])
    free = γ * (1.0 + v * sqrt(1 + M_L^2))                       # the t0 lab boost, never touched again
    @test isapprox(px(DsT = 0.0), M_L * γ * v; rtol = 1e-6)      # cold comoving, exactly
    @test px(DsT = 1e-9) > M_L * γ * v * 1.15                    # DsT to 0+ keeps the thermal spread
    # `:none` free-streams exactly: no drag, no noise, and no boost PAIR either, so not even the
    # per-step γ round-trip contraction (≈2e-7 over 1000 steps) is applied. What remains is the
    # ONE t0 lab boost, which carries the documented γ = 1/√(1−v²+1e-10) regularisation:
    # ½·1e-10/(1−v²) = 6.7e-11 at v = 0.5. Measured 6.67e-11 — that residue and nothing else.
    for dst in (DST_L, 0.0)                                      # :none wins over any DsT
        p_free = px(DsT = dst, collision_mode = :none)
        @test isapprox(p_free, free; rtol = 1e-9)
        @test abs(p_free / free - 1) < 1e-10                     # it IS the γ regularisation
    end
end

# ── L3: inputs that used to degrade silently ───────────────────────────────────────────────────
# `tau_drag` returns 0.0 for any non-positive m, T or DsT, and every kernel reads τ ≤ 0 as
# η_D = κ = 0. So a mistyped mass or a negative DsT produced a FREE-STREAMING run indistinguishable
# from a Langevin run except by its numbers — measured pre-fix, m = 0, m = −1.5 and DsT = −0.1 all
# left ⟨p²⟩ frozen to 1e-9 with nothing said. Both are refused now; DsT = 0 stays legal (L2).
@testset "L3 m <= 0 and DsT < 0 are refused, DsT = 0 is not" begin
    N = 500
    xg, tg, Tf, Vf = flow_box(0.30, 0.0)
    go(; kw...) = sim(CPUBackend(), xg, tg, Tf, Vf; N_particles = N, Δt = 1e-3, initial_time = 0.4,
        final_time = 0.6, save_interval = 0.1, m = M_L, DsT = DST_L, dimensions = 2, kw...)
    @test_throws ErrorException go(m = 0.0)
    @test_throws ErrorException go(m = -M_L)
    @test_throws ErrorException go(m = NaN)
    @test_throws ErrorException go(DsT = -0.1)
    @test_throws ErrorException go(DsT = NaN)
    @test go(DsT = 0.0) isa Tuple                                # the comoving limit stays reachable
end

# ── L4: the background table is checked against the requested window ───────────────────────────
# `interpolate_2d_*` clamps the query into the table — right for a particle leaving the fireball
# rim, wrong for a run that outlives the hydro output: past tgrid[end] the medium freezes at its
# last tabulated slice and the run continues. That behaviour is KEPT (it is occasionally what a
# caller wants) but it is no longer silent, on either backend.
@testset "L4 leaving the tabulated window warns" begin
    N = 500
    xg = collect(0.0:0.25:20.0); tg = collect(0.4:0.1:2.0)
    Tf = [0.45 * (0.4 / τ)^(1 / 3) for r in xg, τ in tg]
    Vf = zeros(length(xg), length(tg))
    x0 = zeros(2, N); x0[1, :] .= 3.0; p0 = zeros(2, N)
    go(tf) = sim(CPUBackend(), xg, tg, Tf, Vf; x_init = x0, p_init = p0, N_particles = N,
        Δt = 1e-3, initial_time = 0.4, final_time = tf, save_interval = (tf - 0.4), m = M_L,
        DsT = DST_L, dimensions = 2)
    @test_logs (:warn, r"leaves the tabulated time axis") match_mode = :any go(6.0)
    # particles finishing outside the RADIAL axis warn too (pre-fix: 16.2 % of an ensemble left an
    # 8 fm table, out to r = 15.2 fm, dragged at the rim T and v the whole way, in silence)
    xs = collect(0.0:0.25:2.0)
    Ts = fill(0.30, length(xs), length(tg)); Vs = zeros(length(xs), length(tg))
    xf = zeros(2, N); xf[1, :] .= 1.5
    pf = zeros(2, N); pf[1, :] .= 3.0                            # 0.89c straight out of a 2 fm table
    @test_logs (:warn, r"OUTSIDE the tabulated radial axis") match_mode = :any sim(CPUBackend(),
        xs, tg, Ts, Vs; x_init = xf, p_init = pf, N_particles = N, Δt = 1e-3, initial_time = 0.4,
        final_time = 1.4, save_interval = 1.0, m = M_L, DsT = DST_L, dimensions = 2)
end

# ── L5: the step count survives the binary representation of the endpoints ─────────────────────
# `steps = floor(Int, (tf − t0)/Δt)`: 1.4 − 0.4 = 0.9999999999999999, so the quotient lands a
# fraction of an ulp below the integer and floor took a whole step off. Usually one step of 10⁻³ fm
# — the damage is that it also broke `steps % save_every == 0`, and `_snapshot_times` then dropped
# the entire trailing save interval and blamed `save_interval`. MEASURED worst case: t0 = 0.4,
# tf = 1.4, Δt = 10⁻³, save = 0.5 kept 501 of 1000 steps, HALF the requested history.
@testset "L5 step count is ulp-tolerant but does not over-snap" begin
    SC = LangevInMedium.SimulateCPU._step_count
    @test SC(0.4, 1.4, 1e-3)  == 1000        # was 999  ⇒ 499 steps of history lost
    @test SC(0.4, 13.0, 1e-4) == 126_000     # was 125999 ⇒ AttractorHydro's portrait window
    @test SC(0.4, 8.0, 1e-3)  == 7600        # was 7599
    @test SC(0.4, 13.0, 5e-3) == 2520        # LP1: was already exact
    @test SC(0.0, 12.0, 1e-3) == 12_000      # and stays exact
    # a genuine shortfall must NOT be snapped up: the tolerance is 64 ulps, not a fudge factor
    @test SC(0.4, 1.4005, 1e-3) == 1000
    @test SC(0.4, 1.4 - 1e-6, 1e-3) == 999
end

# ── L6: the FONLL sampler integrates on the ACTUAL nodes ───────────────────────────────────────
# Its inverse CDF was `cumsum(w) * mean(diff(grid))` — a right-Riemann sum with one constant
# spacing, which is two errors at once: first order even where the grid IS uniform, and simply the
# wrong quadrature where it is not, so refinement could not fix it. Pre-fix, against the exact ⟨p⟩
# of P(p) ∝ p·f(p) on the same range with a FONLL-like shape: −3.14 % at np = 100, −1.08 % at the
# production np = 300, −0.29 % at np = 1200, and −44.6 % on a log-spaced grid (−25.7 % at 160
# points, still −24.7 % at 1600 — non-convergent). Now a cumulative trapezoid on the real nodes.
@testset "L6 sample_particles_from_FONLL is grid-spacing correct" begin
    f(p) = p <= 0 ? 0.0 : (1.0 + (p / 2.1)^2)^(-3.1)             # FONLL-like
    ref = begin
        num = 0.0; den = 0.0; h = 1e-4
        for pp in h:h:10.0; w = pp * f(pp); num += pp * w * h; den += w * h; end
        num / den
    end
    rg = collect(range(0.0, 15.0, length = 60))
    bias(pg, N) = begin
        dens = [f(p) * exp(-r^2 / 18) for p in pg, r in rg]
        Random.seed!(5)
        _, p = LangevInMedium.sample_particles_from_FONLL(rg, pg, dens, N)
        mean(sqrt.(sum(abs2, p; dims = 1))) / ref - 1
    end
    @test abs(bias(collect(range(0.0, 10.0, length = 100)), 200_000)) < 0.01   # was −3.14 %
    @test abs(bias(collect(range(0.0, 10.0, length = 300)), 200_000)) < 0.01   # was −1.08 %
    # the grid it could not do at all: log-spaced, where refinement used to change nothing
    @test abs(bias(vcat(0.0, exp10.(range(log10(0.02), log10(10.0), length = 299))), 200_000)) < 0.01
    # and the underlying quadrature, directly
    ct = LangevInMedium.Utils._cumtrapz
    xs = vcat(0.0, exp10.(range(-2, 0, length = 400)))
    @test isapprox(ct(xs .^ 2, xs)[end], 1 / 3; rtol = 1e-3)      # ∫₀¹x²dx on a log grid
    @test isapprox(ct(fill(1.0, 5), [0.0, 0.1, 0.5, 0.9, 1.0])[end], 1.0; rtol = 1e-14)
end

# ── L7: the RTA/BGK collision probability is exponential ───────────────────────────────────────
# `Pcol` was `clamp(Δt·dil/τn, 0, 1)`, so the survival probability was 1 − Δt/τ_n and the realised
# rate −ln(1 − Δt/τ_n)/Δt: always too fast. MEASURED pre-fix against the ℓ=1 decay of a drifting
# ensemble at T = 0.30 (τ_n = 0.5976 fm), ratio to the nominal 1/τ_n = 1.0013 / 1.0069 / 1.0517 /
# 1.0936 / 1.2198 at Δt = 0.002 / 0.01 / 0.05 / 0.1 / 0.2, each within 1 % of that closed form.
# `−expm1(−x)` is the exact per-step probability at ANY Δt and removes the ceiling on the step size.
@testset "L7 RTA relaxes at 1/tau_n at any step size" begin
    T = 0.30; N = 60_000
    τn = tau_n_main3(T, M_L, DST_L)
    xg = collect(0.0:0.5:200.0); tg = collect(0.0:0.25:2.0)
    Tf = fill(T, length(xg), length(tg)); Vf = zeros(size(Tf))
    x0 = zeros(2, N); x0[1, :] .= 50.0; p0 = zeros(2, N); p0[1, :] .= 1.0
    rate(dt) = begin
        t, mom, _ = sim(CPUBackend(), xg, tg, Tf, Vf; x_init = x0, p_init = p0, N_particles = N,
            Δt = dt, initial_time = 0.0, final_time = 1.0, save_interval = 0.5, m = M_L,
            DsT = DST_L, dimensions = 2, collision_mode = :rta)
        -log(abs(mean(mom[end][1, :]) / mean(mom[1][1, :]))) / (t[end] - t[1])
    end
    # the point of the fix: the LARGE step no longer over-relaxes. 1/√N ≈ 0.4 % here, so 3 % is the
    # honest bar; pre-fix Δt = 0.2 sat at 1.22, seven times outside it.
    for dt in (0.01, 0.05, 0.20)
        @test isapprox(rate(dt) * τn, 1.0; rtol = 0.03)
    end
end

# ── L8: reflecting_boundary is CORRECT (a negative result, pinned) ─────────────────────────────
# A specular wall must leave the uniform disc measure invariant. MEASURED over 20 fm at T = 0.30:
# ⟨r⟩ and ⟨r²⟩ within 0.11 % of the uniform-disc values, no escapes, flat radial profile. Nothing
# was wrong here; it is recorded so a later change to the reflection has something to fail against.
@testset "L8 reflecting_boundary preserves the uniform disc measure" begin
    rmax = 6.0; N = 12_000
    xg = collect(0.0:0.1:rmax); tg = collect(0.0:0.5:8.0)
    Tf = fill(0.30, length(xg), length(tg)); Vf = zeros(size(Tf))
    u = rand(MersenneTwister(21), N); φ = 2π .* rand(MersenneTwister(22), N)
    r0 = rmax .* sqrt.(u)
    x0 = vcat((r0 .* cos.(φ))', (r0 .* sin.(φ))')
    p0 = 0.5 .* randn(MersenneTwister(23), 2, N)
    _, _, pos = sim(CPUBackend(), xg, tg, Tf, Vf; x_init = x0, p_init = p0, N_particles = N,
        Δt = 1e-3, initial_time = 0.0, final_time = 6.0, save_interval = 6.0, m = M_L,
        DsT = DST_L, dimensions = 2, reflecting_boundary = true)
    r = sqrt.(sum(abs2, pos[end]; dims = 1))[:]
    @test maximum(r) <= rmax + 1e-9
    @test isapprox(mean(r), 2rmax / 3; rtol = 0.01)
    @test isapprox(mean(r .^ 2), rmax^2 / 2; rtol = 0.01)
end

# ── L9: what `collision_mode = :none` actually is ──────────────────────────────────────────────
# Free streaming, exactly: momenta constant, positions on straight lines at p/E, and the Bjorken
# redshift still applied when asked because dp_z/dτ = −p_z/τ IS the longitudinal free-streaming law.
@testset "L9 collision_mode = :none is exact free streaming" begin
    N = 2_000; t0 = 0.4; tf = 2.4
    xg, tg, Tf, Vf = flow_box(0.30, 0.5)
    x0 = zeros(2, N); x0[1, :] .= 8.0
    p0 = zeros(2, N); p0[1, :] .= 0.7; p0[2, :] .= 0.3
    t, mom, pos = sim(CPUBackend(), xg, tg, Tf, Vf; x_init = x0, p_init = p0, N_particles = N,
        Δt = 1e-3, initial_time = t0, final_time = tf, save_interval = (tf - t0) / 2, m = M_L,
        DsT = DST_L, dimensions = 2, collision_mode = :none)
    @test mom[end] == mom[1]                                     # momenta EXACTLY constant
    E = sqrt(M_L^2 + mom[1][1, 1]^2 + mom[1][2, 1]^2)
    @test isapprox(pos[end][1, 1] - pos[1][1, 1], (tf - t0) * mom[1][1, 1] / E; rtol = 1e-9)
    # with a p_z row the redshift is the only thing that moves it, and p_z* = 0 stays 0
    _, m3, _ = sim(CPUBackend(), xg, tg, Tf, Vf; x_init = x0, p_init = p0, N_particles = N,
        Δt = 1e-3, initial_time = t0, final_time = tf, save_interval = tf - t0, m = M_L,
        DsT = DST_L, dimensions = 2, momentum_dimensions = 3, pz_init = :comoving,
        bjorken_redshift = true, collision_mode = :none)
    @test all(iszero, m3[end][3, :])
    @test m3[end][1:2, :] == m3[1][1:2, :]
end

end # testset L
