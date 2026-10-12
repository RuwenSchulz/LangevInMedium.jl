# ==============================================================================================
# test_adaptive_dt.jl — the variable time step (2026-10-12).
#
# Every kernel reads time only as `t0 + step·Δt` (END of the step) or `t0 + (step−1)·Δt` (START),
# so the drivers run a variable step as the call (step = 1, t0 = t_k, Δt = t_{k+1} − t_k). These
# checks pin that plumbing and the grid the builder makes:
#
#   V1  a `time_grid` that is the uniform grid reproduces the fixed-Δt run to float rounding —
#       same seed, same RNG draws, every kernel on the path (boosts, OU, positions, Bjorken
#       redshift, η_s). Any kernel handed the wrong (step, Δt, t0) would differ at O(1).
#   V2  the adaptive grid: starts/ends on the window, lands EXACTLY on every snapshot, respects
#       η_D·h ≤ drag_tol in the hottest cell, dt_min ≤ h ≤ dt_max, and grows as the medium cools.
#   V3  free streaming + Bjorken redshift on a strongly non-uniform grid: p_z·τ is conserved to
#       rounding (the redshift telescopes only if each step uses ITS OWN start time).
#   V4  a stationary bath: the adaptive run holds the Jüttner ⟨p²⟩ at the accuracy the bench
#       predicts for its η_D·h, and a deliberately coarse tolerance shows the bias it should —
#       the test can tell a good step from a bad one.
#   V5  bad input is refused: a grid that misses a snapshot, does not span the window, or is
#       not increasing; time_grid together with adaptive_dt.
#
#   julia --project=Julia Julia/LangevInMedium.jl/test/test_adaptive_dt.jl
# ==============================================================================================
using Test, Random, Statistics, LinearAlgebra, QuadGK, LangevInMedium

const M, DST = 1.5, 0.11634
const HBARC = LangevInMedium.GevInvTofm

"Cooling Gaussian fireball with radial flow (the corpus' flow_bg, longer)."
function flow_bg(; t0 = 0.4, tf = 3.0)
    xg = collect(0.0:0.25:40.0); tg = collect(t0:0.1:(t0 + tf + 0.5))
    Tf = [max(0.12, 0.45 * exp(-r^2 / 60) * (t0 / τ)^(1 / 3)) for r in xg, τ in tg]
    Vf = [0.6 * r / (r + 4.0) * min(1.0, τ / 2.0) for r in xg, τ in tg]
    xg, tg, Tf, Vf
end

const RG = collect(0.0:0.5:20.0); const PG = collect(0.05:0.1:8.0)
const DENS = ones(length(PG), length(RG))

run(xg, tg, Tf, Vf, x0, p0; seed, kw...) = (Random.seed!(seed);
    simulate_ensemble_bulk(CPUBackend(), RG, PG, DENS, Tf, Vf, (xg, tg);
        x_init = x0, p_init = p0, N_particles = size(x0, 2), m = M, DsT = DST, dimensions = 2, kw...))

@testset "V1 a uniform time_grid reproduces the fixed-Δt run" begin
    xg, tg, Tf, Vf = flow_bg(; tf = 0.6)
    rng = MersenneTwister(7); N = 2000
    x0 = 3.0 .* randn(rng, 2, N); p0 = 0.8 .* randn(rng, 2, N)
    t0, tf, dt, si = 0.4, 1.0, 2e-3, 0.2
    nsteps = round(Int, (tf - t0) / dt)
    for kw in ((;),
               (; momentum_dimensions = 3, bjorken_redshift = true, track_eta_s = true),
               (; momentum_dimensions = 3, bjorken_redshift = true, bjorken_frame = :lab, proper_time_kicks = true),
               (; collision_mode = :rta),
               (; relativistic = false, position_diffusion = true))
        a = run(xg, tg, Tf, Vf, x0, p0; seed = 11, Δt = dt, initial_time = t0, final_time = tf,
                save_interval = si, kw...)
        b = run(xg, tg, Tf, Vf, x0, p0; seed = 11, initial_time = t0, final_time = tf,
                save_interval = si, time_grid = collect(range(t0, tf; length = nsteps + 1)), kw...)
        @test length(a[1]) == length(b[1])
        @test maximum(abs.(collect(a[1]) .- b[1])) < 1e-12
        for k in 2:length(a)                       # momenta, positions (and η_s)
            err = maximum(maximum(abs.(a[k][s] .- b[k][s])) for s in eachindex(a[k]))
            @test err < 1e-8
        end
    end
end

@testset "V2 the adaptive grid" begin
    xg, tg, Tf, Vf = flow_bg(; tf = 6.0)
    t0, tf, si, tol = 0.4, 6.4, 0.5, 0.01
    grid, ss = adaptive_time_grid(Tf, tg; initial_time = t0, final_time = tf, save_interval = si,
                                  m = M, DsT = DST, drag_tol = tol, dt_max = 0.1)
    h = diff(grid)
    @test grid[1] == t0 && grid[end] == tf
    @test all(>(0), h)
    @test all(h .<= 0.1 * (1 + 1e-12))
    # snapshots: exactly on t0 + k·si (k = 1…11), then tf (= t0 + 12·si)
    @test grid[ss .+ 1] == vcat([t0 + k * si for k in 1:11], tf)
    # η_D·h ≤ tol at the hottest cell of the slices the step touches (the T floor 0.12 is cold)
    ηD(T) = (T^2 / (M * DST)) / HBARC
    for k in eachindex(h)
        j = clamp(searchsortedlast(tg, grid[k]), 1, length(tg) - 1)
        @test ηD(maximum(Tf[:, j])) * h[k] <= tol * (1 + 1e-9)
    end
    # it grows as the medium cools: late steps ≫ early steps, and far fewer than a fixed 1e-3
    @test h[end÷2] > 2 * h[1]
    nfix = round(Int, (tf - t0) / 1e-3)
    @info "V2: adaptive grid" steps = length(h) fixed_1e3 = nfix h_first = h[1] h_last = h[end-1]
    @test length(h) < nfix / 2

    # drag inactive (free streaming / comoving) ⇒ only dt_max
    g2, _ = adaptive_time_grid(Tf, tg; initial_time = t0, final_time = tf, save_interval = si,
                               m = M, DsT = DST, drag_active = false, dt_max = 0.05)
    @test maximum(diff(g2)) <= 0.05 * (1 + 1e-12)
    @test length(g2) - 1 == 120

    # save_interval that does not divide the window: the last snapshot is final_time, nothing dropped
    g3, s3 = adaptive_time_grid(Tf, tg; initial_time = t0, final_time = 1.75, save_interval = 0.5,
                                m = M, DsT = DST)
    @test g3[s3 .+ 1] ≈ [0.9, 1.4, 1.75] atol = 1e-14
end

@testset "V3 free streaming + Bjorken redshift on a non-uniform grid: p_z·τ exact" begin
    xg, tg, Tf, Vf = flow_bg(; tf = 3.0)
    rng = MersenneTwister(3); N = 500
    x0 = 2.0 .* randn(rng, 2, N); p0 = randn(rng, 2, N)
    t0, tf = 0.4, 3.4
    # geometric grid (τ_{k+1}/τ_k = const) with the snapshot times t0 + k·0.5 inserted
    g = sort(unique(vcat(exp.(range(log(t0), log(tf); length = 60)), [t0 + 0.5k for k in 1:6])))
    g[end] = tf
    out = run(xg, tg, Tf, Vf, x0, p0; seed = 1, initial_time = t0, final_time = tf, save_interval = 0.5,
              time_grid = g, collision_mode = :none, momentum_dimensions = 3, bjorken_redshift = true,
              pz_init = :thermal)
    tp, P = out[1], out[2]
    pz0 = P[1][3, :]
    for (k, τ) in enumerate(tp)
        @test maximum(abs.(P[k][3, :] .* τ .- pz0 .* t0)) < 1e-12
    end
end

@testset "V4 stationary bath: the adaptive step holds the Jüttner ⟨p²⟩" begin
    # box at T = 0.3: η_D = 2.61 fm⁻¹. The bench measured the 2-row stationary-⟨p²⟩ bias as
    # 53 %·(ηΔt)^1.56: ≈ 0.04 % at η h = 0.01 and ≈ 3.4 % at 0.2.
    T = 0.3
    xg = collect(0.0:0.5:400.0); tg = collect(0.0:0.25:12.0)
    Tf = fill(T, length(xg), length(tg)); Vf = zeros(length(xg), length(tg))
    rng = MersenneTwister(5); N = 40_000
    # start AT the stationary 2-D Jüttner (rejection), so the whole run is stationary
    pmax = 14 * sqrt(M * T + T^2); w(p) = p * exp(-(sqrt(p^2 + M^2) - M) / T)
    wmax = maximum(w, range(1e-6, pmax; length = 4000))
    p0 = zeros(2, N)
    for i in 1:N
        p = 0.0
        while true
            p = pmax * rand(rng); rand(rng) * wmax <= w(p) && break
        end
        φ = 2π * rand(rng); p0[:, i] .= p .* (cos(φ), sin(φ))
    end
    x0 = zeros(2, N)
    # the EXACT 2-D Jüttner ⟨p²⟩, not the sample's (whose own noise is ≈ 0.5 %)
    ref = quadgk(p -> p^2 * w(p), 0, pmax)[1] / quadgk(w, 0, pmax)[1]
    function stationary_p2(tol)
        out = run(xg, tg, Tf, Vf, x0, p0; seed = 2, initial_time = 0.0, final_time = 10.0,
                  save_interval = 0.5, adaptive_dt = true, adaptive_drag_tol = tol, dt_max = 1.0)
        mean(mean(sum(abs2, out[2][k]; dims = 1)) for k in 5:length(out[1]))   # skip the first 2 fm
    end
    good = stationary_p2(0.01) / ref - 1
    bad  = stationary_p2(0.2)  / ref - 1
    @info "V4: stationary ⟨p²⟩ bias" good_pct = 100good bad_pct = 100bad
    @test abs(good) < 0.005            # 16 snapshots × 40k particles: noise on the mean ≈ 0.15 %
    @test bad < -0.015                 # the coarse step shows its (negative) bias
end

@testset "V5 bad input is refused" begin
    xg, tg, Tf, Vf = flow_bg(; tf = 1.0)
    x0 = zeros(2, 10); p0 = ones(2, 10)
    base = (; seed = 1, initial_time = 0.4, final_time = 1.4, save_interval = 0.5)
    @test_throws ErrorException run(xg, tg, Tf, Vf, x0, p0; base..., time_grid = [0.4, 0.8, 1.4])   # misses 0.9
    @test_throws ErrorException run(xg, tg, Tf, Vf, x0, p0; base..., time_grid = [0.4, 0.9, 1.3])   # short of 1.4
    @test_throws ErrorException run(xg, tg, Tf, Vf, x0, p0; base..., time_grid = [0.4, 0.9, 0.9, 1.4])
    @test_throws ErrorException run(xg, tg, Tf, Vf, x0, p0; base..., time_grid = [0.4, 0.9, 1.4], adaptive_dt = true)
    # and the accepted version of the first
    out = run(xg, tg, Tf, Vf, x0, p0; base..., time_grid = [0.4, 0.8, 0.9, 1.4])
    @test out[1] == [0.4, 0.9, 1.4]
end

# ── V6 the GPU driver takes the same steps ────────────────────────────────────────────────────────
# CURAND is seedable (`CUDA.seed!`), so the GPU twin of V1 is exact too. Skipped without a GPU.
const HAVE_GPU = get(ENV, "LIM_NOGPU", "0") != "1" && try
    @eval using CUDA
    Base.invokelatest(() -> CUDA.functional())
catch
    false
end
if HAVE_GPU
    @testset "V6 GPU: uniform time_grid ≡ fixed Δt; adaptive grid matches the CPU's" begin
        xg, tg, Tf, Vf = flow_bg(; tf = 0.6)
        rng = MersenneTwister(9); N = 4000
        x0 = 3.0 .* randn(rng, 2, N); p0 = 0.8 .* randn(rng, 2, N)
        t0, tf, dt, si = 0.4, 1.0, 2e-3, 0.2
        nsteps = round(Int, (tf - t0) / dt)
        gpu(; seed, kw...) = (Random.seed!(seed); Base.invokelatest(CUDA.seed!, seed);
            Base.invokelatest(simulate_ensemble_bulk, GPUBackend(), RG, PG, DENS, Tf, Vf, (xg, tg);
                x_init = x0, p_init = p0, N_particles = N, m = M, DsT = DST, dimensions = 2,
                initial_time = t0, final_time = tf, save_interval = si, kw...))
        for kw in ((;), (; momentum_dimensions = 3, bjorken_redshift = true, track_eta_s = true),
                   (; collision_mode = :rta))
            a = gpu(; seed = 4, Δt = dt, kw...)
            b = gpu(; seed = 4, time_grid = collect(range(t0, tf; length = nsteps + 1)), kw...)
            @test maximum(abs.(collect(a[1]) .- b[1])) < 1e-12
            for k in 2:length(a)
                @test maximum(maximum(abs.(a[k][s] .- b[k][s])) for s in eachindex(a[k])) < 1e-8
            end
        end
        # adaptive: identical time axis on both backends; moments agree within the ensemble noise
        c = run(xg, tg, Tf, Vf, x0, p0; seed = 5, initial_time = t0, final_time = tf,
                save_interval = si, adaptive_dt = true)
        g = gpu(; seed = 5, adaptive_dt = true)
        @test c[1] == g[1]
        p2c = mean(sum(abs2, c[2][end]; dims = 1)); p2g = mean(sum(abs2, g[2][end]; dims = 1))
        @test abs(p2g / p2c - 1) < 0.03
        # the freeze-out latch runs on the variable step and books every particle that crosses
        fo = gpu(; seed = 6, adaptive_dt = true, freezeout_capture = true, Tfo = 0.30)
        @test all(isfinite, fo.tau) && all(isfinite, fo.mom)
        @test all(t -> t == 0 || t0 <= t <= tf, fo.tau)
    end
end
