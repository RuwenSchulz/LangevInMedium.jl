# ==============================================================================================
# test_pz_lab_rapidity.jl — the p_z row against a code that actually has z  (2026-09-01)
#
# WHY THIS EXISTS. Every other p_z gate in this package checks the engine against a FORMULA:
# test_momentum_dims3.jl (R) checks ⟨p_z²⟩ ∝ (τ₀/τ)², test_bjorken_redshift_exact.jl checks the
# whole distribution rescales by τ₀/τ. Both would pass just as happily if the frame convention
# itself were wrong, because both are written in the same variables as the code under test.
#
# `momenta[3, :]` is NOT a lab p_z. It is p_z* = m_T sinh(y − η_s), the longitudinal momentum in
# the frame comoving with the Bjorken fluid at the particle's OWN η_s — and the package stores no
# z and no η_s, so nothing in it can detect a mistake in that identification. This file supplies
# the missing instrument: a second, independently written integrator in PLAIN CARTESIAN LAB
# COORDINATES that carries (t, z, p_x, p_y, p_z), reads the fluid rapidity off the particle's own
# position as η_s = artanh(z/t), boosts in and out explicitly, and streams z += (p_z/E)·dt. It
# never uses τ_a/τ_b, never assumes a clock, and never assumes p_z* means anything.
#
# It also pins the two conventions that are easy to get wrong and that no other test touches:
#   * the position kernel's E = √(m² + p_⊥,lab² + p_z*²) is a MIXED-frame energy, and that mixture
#     is exactly the Milne streaming dx_⊥/dτ = p_⊥/E*;
#   * dτ = cosh(η)dt − sinh(η)dz = dt* IDENTICALLY along any worldline, so the Milne step IS the
#     local-rest-frame time and no dilation factor is missing longitudinally. The reference steps
#     in LAB time with dt* = (E*/E)·dt_lab; the engine steps in τ with no factor at all. If the
#     clocks disagreed the two would relax for different amounts of LRF time and C2 would fail.
#
# Gates
#   C1  p_z* mean and width agree with the Cartesian reference within 3 s.e.
#   C2  every decile of p_z* agrees (the SHAPE, not two moments)
#   C3  two-sample Kolmogorov–Smirnov on the full distribution
#   C4  p_z* is independent of η_s — boost invariance, measured rather than assumed
#   P1  pz_init = :comoving puts EXACT zeros in row 3 (and the transverse boost leaves them there)
#   P2  pz_init = :thermal is the default: passing it explicitly is bit-identical to omitting it
#   E1  track_eta_s is a PASSENGER: momenta and positions are bit-identical with it on and off
#   E2  under FREE STREAMING the reconstructed lab rapidity y = η_s + atanh(p_z*/E*) is CONSERVED
#       — a parameter-free identity that pins the log factor, the sign, the energy and the clock
#   E3  the on-the-fly η_s equals a post-hoc integral over the returned snapshot history
#
#   julia --project=Julia Julia/LangevInMedium.jl/test/test_pz_lab_rapidity.jl
# ==============================================================================================
using Test, Random, Statistics, LangevInMedium

const PZM    = 1.5          # charm
const PZT    = 0.350        # static uniform bath: isolates the longitudinal sector
const PZTAU0 = 0.6
const PZTAUF = 4.0
const PZDST  = 0.20
const PZN    = 8_000

pz_taudrag(T) = LangevInMedium.Transport.tau_drag(T, PZM, PZDST)

"One exact-OU step on a 3-vector in the LRF over LRF time `dt` — mirrors kernel_compute_all_forces_cpu!."
@inline function pz_ou_step!(p, T, dt, rng)
    td = pz_taudrag(T); td > 0 || return
    ηD = 1.0 / td; κ = 2.0 * PZM * T / td
    E  = sqrt(PZM^2 + p[1]^2 + p[2]^2 + p[3]^2)
    ηe = ηD * PZM / E
    a  = exp(-ηe * dt)
    sd = sqrt(κ * (1 - a * a) / (2 * ηe))
    @inbounds for d in 1:3
        p[d] = a * p[d] + sd * randn(rng)
    end
end

"""
THE REFERENCE. Cartesian lab frame with an explicit z: η_s is read off the particle's own
position, the boosts are written out, and the LRF time step is dt* = (E*/E)·dt_lab. Each particle
runs on its own lab clock until its own Milne time sqrt(t²−z²) reaches PZTAUF.
"""
function pz_cartesian_reference(; ystar0, eta0, pT, dtlab = 5.0e-3, seed = 20260901)
    rng = MersenneTwister(seed); n = length(ystar0)
    out_pzstar = zeros(n); out_eta = zeros(n); p = zeros(3)
    for i in 1:n
        mT = sqrt(PZM^2 + pT[i]^2)
        y0 = ystar0[i] + eta0[i]                       # lab rapidity at τ₀
        t  = PZTAU0 * cosh(eta0[i]); z = PZTAU0 * sinh(eta0[i])
        φ  = 2π * rand(rng)
        px = pT[i] * cos(φ); py = pT[i] * sin(φ); pz = mT * sinh(y0)
        τ  = PZTAU0
        while τ < PZTAUF
            E   = sqrt(PZM^2 + px^2 + py^2 + pz^2)
            η   = atanh(clamp(z / t, -0.999999, 0.999999))
            ch  = cosh(η); sh = sinh(η)
            pzs = ch * pz - sh * E                     # into the local fluid frame
            Es  = sqrt(PZM^2 + px^2 + py^2 + pzs^2)
            p[1] = px; p[2] = py; p[3] = pzs
            pz_ou_step!(p, PZT, (Es / E) * dtlab, rng) # dt* = (E*/E) dt_lab  ( = dτ )
            px, py, pzs = p[1], p[2], p[3]
            Es  = sqrt(PZM^2 + px^2 + py^2 + pzs^2)
            pz  = ch * pzs + sh * Es                   # back out
            E   = sqrt(PZM^2 + px^2 + py^2 + pz^2)
            z  += (pz / E) * dtlab; t += dtlab
            τ   = sqrt(max(t * t - z * z, 1e-12))
        end
        E = sqrt(PZM^2 + px^2 + py^2 + pz^2)
        η = atanh(clamp(z / t, -0.999999, 0.999999))
        out_pzstar[i] = cosh(η) * pz - sinh(η) * E
        out_eta[i]    = η
    end
    (pzstar = out_pzstar, eta = out_eta)
end

"The engine on the same static bath. Returns whatever `simulate_ensemble_bulk` returns."
function pz_engine(; pz0, pT, dt = 2.0e-3, dsave = PZTAUF - PZTAU0, DsT = PZDST,
                     seed = 20260901, kw...)
    n = length(pT)
    xg = collect(0.0:0.5:60.0); tg = collect(PZTAU0:0.1:(PZTAUF + 0.5))
    Tf = fill(PZT, length(xg), length(tg)); Vf = zeros(length(xg), length(tg))
    rng = MersenneTwister(seed)
    x = zeros(2, n); p = zeros(3, n)
    for i in 1:n
        r = 2.0 + 3.0 * rand(rng); θ = 2π * rand(rng)
        x[1, i] = r * cos(θ); x[2, i] = r * sin(θ)
        φ = 2π * rand(rng)
        p[1, i] = pT[i] * cos(φ); p[2, i] = pT[i] * sin(φ); p[3, i] = pz0[i]
    end
    Random.seed!(4321)
    simulate_ensemble_bulk(CPUBackend(), nothing, nothing, nothing, Tf, Vf, (xg, tg);
        N_particles = n, Δt = dt, initial_time = PZTAU0, final_time = PZTAUF,
        save_interval = dsave, m = PZM, DsT = DsT,
        dimensions = 2, momentum_dimensions = 3, bjorken_redshift = true,
        x_init = x, p_init = p, kw...)
end

"""
The SAMPLER path (no `p_init`): this is the only way in, so it is the only place the p_z
completion — and hence `pz_init` — actually runs.
"""
function pz_sampler_run(; steps_to = PZTAU0 + 0.5, kw...)
    xg = collect(0.0:0.5:60.0); tg = collect(PZTAU0:0.1:(PZTAUF + 0.5))
    Tf = fill(PZT, length(xg), length(tg)); Vf = zeros(length(xg), length(tg))
    rg = collect(0.0:0.5:20.0); pg = collect(range(0.0, 6.0; length = 40))
    dens = [exp(-p) * exp(-r^2 / 32) for p in pg, r in rg]
    Random.seed!(7)
    simulate_ensemble_bulk(CPUBackend(), rg, pg, dens, Tf, Vf, (xg, tg);
        N_particles = 400, Δt = 1e-2, initial_time = PZTAU0, final_time = steps_to,
        save_interval = steps_to - PZTAU0, m = PZM, DsT = PZDST, dimensions = 2,
        momentum_dimensions = 3, bjorken_redshift = true, Tfo = 0.0, kw...)
end

"Kolmogorov–Smirnov two-sample statistic and the 1 % critical value (fixed seeds ⇒ deterministic)."
function pz_ks2(a, b)
    x = sort(a); y = sort(b); na = length(x); nb = length(y)
    i = j = 1; d = 0.0
    while i <= na && j <= nb
        if x[i] <= y[j]; i += 1 else; j += 1 end
        d = max(d, abs(i / na - j / nb))
    end
    (d, 1.63 * sqrt((na + nb) / (na * nb)))
end

"Standard error of the p-quantile (Gaussian-kernel density estimate at the quantile)."
function pz_qse(v, p)
    n = length(v); q = quantile(v, p); h = 1.06 * std(v) * n^(-0.2)
    f = sum(exp.(-0.5 .* ((v .- q) ./ h) .^ 2)) / (n * h * sqrt(2π))
    sqrt(p * (1 - p) / n) / max(f, 1e-12)
end

@testset "p_z is the comoving p_z*, and η_s closes the lab rapidity" begin

    rng  = MersenneTwister(99)
    pT   = [0.5 + 2.5 * rand(rng) for _ in 1:PZN]
    mT   = sqrt.(PZM^2 .+ pT .^ 2)
    # a BOX in y − η_s: nothing thermal, nothing Gaussian, so a wrong frame cannot hide in a shape
    ystar0 = [(-0.8 + 1.6 * rand(rng)) for _ in 1:PZN]
    pz0    = mT .* sinh.(ystar0)
    # η_s spread wide and INDEPENDENT of p_z*: a boost-invariant ensemble, so the reference must
    # return the same p_z* distribution the engine does even though only it knows where things are
    eta0   = [(-1.5 + 3.0 * rand(rng)) for _ in 1:PZN]

    ref = pz_cartesian_reference(ystar0 = ystar0, eta0 = eta0, pT = pT)
    _, mom, _ = pz_engine(pz0 = pz0, pT = pT)
    ez = vec(mom[end][3, :]); rz = ref.pzstar

    @testset "C1 mean and width vs the Cartesian reference" begin
        se_m = sqrt(var(ez) / PZN + var(rz) / PZN)
        @test abs(mean(ez) - mean(rz)) < 3 * se_m
        se_s = std(rz) * sqrt(1 / PZN)
        @test abs(std(ez) - std(rz)) < 3 * se_s
    end

    @testset "C2 every decile" begin
        for p in 0.1:0.1:0.9
            d  = quantile(ez, p) - quantile(rz, p)
            se = sqrt(pz_qse(ez, p)^2 + pz_qse(rz, p)^2)
            @test abs(d) < 3 * se
        end
    end

    @testset "C3 Kolmogorov–Smirnov, whole distribution" begin
        d, dcrit = pz_ks2(ez, rz)
        @test d < dcrit
    end

    @testset "C4 p_z* is independent of η_s (boost invariance, measured)" begin
        lo = ref.pzstar[ref.eta .< -0.7]; hi = ref.pzstar[ref.eta .> 0.7]
        @test min(length(lo), length(hi)) > 500
        @test abs(std(lo) / std(hi) - 1) < 0.06
    end

    @testset "P1 pz_init = :comoving is EXACTLY zero" begin
        # the transverse boost leaves row 3 alone, so exact zeros must survive to the t0 snapshot
        _, mc, _ = pz_sampler_run(pz_init = :comoving)
        @test all(mc[1][3, :] .== 0.0)
        @test !all(mc[end][3, :] .== 0.0)          # ... and the medium then does something to them
        # the shipped default must NOT be zero, or P1 would pass for the wrong reason
        _, mt, _ = pz_sampler_run()
        @test !any(mt[1][3, :] .== 0.0)
        @test_throws ErrorException pz_sampler_run(pz_init = :bogus)
        # :comoving is a statement about the p_z ROW; it is meaningless without one
        @test_throws ErrorException pz_sampler_run(pz_init = :comoving, momentum_dimensions = 2,
                                                  bjorken_redshift = false)
    end

    @testset "P2 pz_init = :thermal is the default, bit-identically" begin
        a = pz_sampler_run(); b = pz_sampler_run(pz_init = :thermal)
        @test a[2][end] == b[2][end]
        @test a[3][end] == b[3][end]
    end

    @testset "E1 track_eta_s is a passenger" begin
        off = pz_engine(pz0 = pz0, pT = pT)
        on  = pz_engine(pz0 = pz0, pT = pT, track_eta_s = true)
        @test length(off) == 3
        @test length(on)  == 4
        @test on[2][end] == off[2][end]            # momenta bit-identical
        @test on[3][end] == off[3][end]            # positions bit-identical
        @test all(on[4][1] .== 0.0)                # η_s(τ0) := 0 by construction
        @test_throws ErrorException pz_engine(pz0 = pz0, pT = pT, track_eta_s = true,
                                              momentum_dimensions = 2)
    end

    @testset "E2 free streaming conserves the reconstructed lab rapidity" begin
        # A huge DsT makes τ_drag ∝ DsT enormous, so η_D → 0 without taking the DsT == 0 branch
        # (which glues momenta to the flow instead of leaving them alone). For a FREE particle the
        # lab rapidity is a constant of the motion: η_s and p_z* both move, and their sum must not.
        #
        # The engine splits the two — the redshift is applied at the START of the step, the η_s
        # increment uses the END-of-step p_z*/E* — so both are consistent FIRST-ORDER rules and y
        # is conserved to O(Δt), not exactly. That is what this gate tests. A wrong sign, a missing
        # log, the wrong energy or the wrong clock each break the identity at O(1) and would not
        # converge at all.
        function drift(dt)
            t, m3, _, eta = pz_engine(pz0 = pz0, pT = pT, DsT = 1.0e12, track_eta_s = true,
                                      dt = dt, dsave = (PZTAUF - PZTAU0) / 4)
            ylab(k) = begin
                mk = m3[k]; pz = @view mk[3, :]
                E  = sqrt.(PZM^2 .+ (@view mk[1, :]) .^ 2 .+ (@view mk[2, :]) .^ 2 .+ pz .^ 2)
                eta[k] .+ atanh.(clamp.(pz ./ E, -0.999999999, 0.999999999))
            end
            y0 = ylab(1)
            (length(t), maximum(abs.(ylab(length(m3)) .- y0)))
        end
        n1, e1 = drift(2.0e-3)
        n2, e2 = drift(1.0e-3)
        @test n1 == 5 && n2 == 5
        @test e1 < 2.0e-3                       # loose: an O(1) mistake misses this by decades
        @test 0.35 < e2 / e1 < 0.70             # first order: halving Δt halves the drift
    end

    @testset "E3 on-the-fly η_s equals the post-hoc integral" begin
        # 0.1 fm divides the 3.4 fm evolution at Δt = 2e-3 exactly (34 intervals): no dropped tail
        t, m3, _, eta = pz_engine(pz0 = pz0, pT = pT, track_eta_s = true, dsave = 0.1)
        acc = zeros(PZN)
        rate(k) = begin
            mk = m3[k]; pz = @view mk[3, :]
            E  = sqrt.(PZM^2 .+ (@view mk[1, :]) .^ 2 .+ (@view mk[2, :]) .^ 2 .+ pz .^ 2)
            (pz ./ E) ./ t[k]
        end
        fprev = rate(1)
        for k in 2:length(m3)
            fnow = rate(k)
            acc .+= 0.5 * (t[k] - t[k - 1]) .* (fprev .+ fnow)
            fprev = fnow
        end
        # trapezoid over 40 snapshots vs the exact per-step sum: they agree to the snapshot
        # resolution, which is the whole reason the accumulator lives inside the engine.
        @test abs(mean(acc) - mean(eta[end])) < 2e-3
        @test abs(std(acc) / std(eta[end]) - 1) < 0.02
    end
end
