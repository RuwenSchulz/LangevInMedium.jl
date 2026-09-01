#!/usr/bin/env julia
# ==============================================================================
# LangevInMedium.jl unit tests — the transport-kernel physics primitives that the
# heavy-quark Langevin integrator (and its FluiduM/FiVo comparison) rest on:
#
#   * the matched relaxation time  τ_n = D_s z K₃/K₂  (Transport.tau_n_main3),
#     incl. the z>50 catastrophic-cancellation fast path and the NR limit;
#   * the diffusion-coefficient prescription effective_DsT (const / linear);
#   * the Einstein / fluctuation-dissipation closure  κ = 2 M T η_D  used by the
#     CPU kernel (kernels_cpu.jl: η_D = 1/τ_n, κ = 2MT/τ_n), via the exact OU
#     propagator's stationary variance ⟨p²⟩ → κ/(2η_D) = M T.
#
# These were previously proven only inside LangevinPaper1's harness; they belong
# with the library so `Pkg.test()` on LangevInMedium exercises them directly.
#
#   julia --project=Julia Julia/LangevInMedium.jl/test/runtests.jl              # everything (≈ 10 min CPU, + GPU gates if CUDA works)
#   LIM_FAST=1 julia --project=Julia Julia/LangevInMedium.jl/test/runtests.jl   # transport + unit + time-convention (≈ 30 s)
#   julia --project=Julia Julia/LangevInMedium.jl/test/regression_corpus.jl     # bit-identity corpus (separate; see its header)
# ==============================================================================

using Test
using LangevInMedium          # exports tau_n_main3, effective_DsT, fmGeV
using Bessels
using Statistics
using Random

# Exp-scaled Bessel K (the e^{-z} cancels in any ratio / linear combination at fixed z).
Kx(ν, z) = Bessels.besselkx(ν, z)

# Paper reference: τ_n = (DsT/T)·z·K₃(z)/K₂(z) / fmGeV  [fm], z = M/T.
tau_n_reference(T, M, DsT) = (DsT / T) * (M / T) * (Kx(3, M / T) / Kx(2, M / T)) / fmGeV

const M_CHARM = 1.5
const DST     = 0.11634

@testset "LangevInMedium transport kernels" begin

    @testset "Bessel matching identity 2K₁-3K₃+K₅ = (48/z²)K₃" begin
        for z in (0.5, 1.0, 2.0, 5.0, 9.6, 20.0, 40.0)
            @test isapprox(2Kx(1, z) - 3Kx(3, z) + Kx(5, z), (48 / z^2) * Kx(3, z); rtol = 1e-9)
        end
    end

    @testset "tau_n_main3 ≡ D_s z K₃/K₂ and positive" begin
        for T in (0.6, 0.5, 0.4, 0.3, 0.25, 0.2, 0.156, 0.12, 0.1)
            @test isapprox(tau_n_main3(T, M_CHARM, DST), tau_n_reference(T, M_CHARM, DST); rtol = 1e-8)
            @test tau_n_main3(T, M_CHARM, DST) > 0
        end
        # invalid inputs return 0 (guard the early-out branch)
        @test tau_n_main3(-1.0, M_CHARM, DST) == 0.0
        @test tau_n_main3(0.3, M_CHARM, 0.0) == 0.0
        @test tau_n_main3(0.3, M_CHARM, -1.0) == 0.0
    end

    @testset "z>50 fast path is correct and continuous across z=50" begin
        T_below = M_CHARM / 49.99   # general Bessel branch
        T_above = M_CHARM / 50.01   # fast-path branch
        @test isapprox(tau_n_main3(T_below, M_CHARM, DST), tau_n_reference(T_below, M_CHARM, DST); rtol = 1e-8)
        @test isapprox(tau_n_main3(T_above, M_CHARM, DST), tau_n_reference(T_above, M_CHARM, DST); rtol = 1e-6)
        @test isapprox(tau_n_main3(T_below, M_CHARM, DST), tau_n_main3(T_above, M_CHARM, DST); rtol = 1e-3)
    end

    @testset "non-relativistic limit τ_n → M D_s/T (K₃/K₂ → 1)" begin
        ratio(T) = tau_n_main3(T, M_CHARM, DST) * T^2 * fmGeV / (DST * M_CHARM)
        @test ratio(0.05) > 1.0
        @test ratio(0.05) < ratio(0.20)
        @test isapprox(ratio(0.02), 1.0; atol = 0.05)   # z = 75: within 5% of NR limit
    end

    @testset "effective_DsT prescription (quadratic ⇒ T-independent drag)" begin
        Tref = 0.30
        for T in (0.12, 0.2, 0.3, 0.45)
            @test isapprox(effective_DsT(T, DST; DsT_quad = true, DsT_Tref = Tref), DST * (T / Tref)^2; rtol = 1e-12)
            # the drag time built from it is constant in T: m·DsT_eff/T² = m·DsT/Tref²
            @test isapprox(tau_drag(T, M_CHARM, effective_DsT(T, DST; DsT_quad = true, DsT_Tref = Tref)),
                           tau_drag(Tref, M_CHARM, DST); rtol = 1e-12)
        end
        @test_throws ErrorException effective_DsT(0.3, DST; DsT_quad = true, DsT_Tref = 0.0)
        @test_throws ErrorException effective_DsT(0.3, DST; DsT_quad = true, DsT_Tref = 0.3, DsT_linear = true)
        # and through the spline builder: flat to the grid's linear-interpolation floor
        _, _, vals = build_tau_drag_spline(M_CHARM, DST; Tmin = 0.12, Tmax = 0.5, nT = 64, DsT_quad = true, DsT_Tref = Tref)
        @test maximum(vals) / minimum(vals) < 1 + 1e-9
    end

    @testset "snapshot time axis matches the snapshots taken" begin
        ST = LangevInMedium.SimulateCPU._snapshot_times
        # divisible: the historical range, bit for bit
        @test ST(0.0, 8.0, 2e-3, 4000, 250, 16) === range(0.0, 8.0, length = 17)
        # not divisible: 5295 steps, save every 662 ⇒ 7 saves at k·1.324, NOT k·10.59/7
        tp = @test_logs (:warn, r"does not divide") ST(0.0, 10.59, 2e-3, 5295, 662, 7)
        @test length(tp) == 8 && isapprox(step(tp), 662 * 2e-3; rtol = 1e-12) && tp[1] == 0.0
    end

    @testset "effective_DsT prescription (const / linear)" begin
        # constant mode is T-independent
        @test effective_DsT(0.3, DST) == DST
        @test effective_DsT(0.6, DST) == DST
        # linear mode: DsT(T) = slope·max(T,Tfo) + offset, floored at Tfo
        slope, offset, Tfo = 1.765, -0.159, 0.156
        @test isapprox(effective_DsT(0.30, DST; DsT_linear=true), slope*0.30 + offset; rtol=1e-12)
        @test isapprox(effective_DsT(0.10, DST; DsT_linear=true), slope*Tfo + offset; rtol=1e-12)  # floored
        @test isapprox(effective_DsT(Tfo,  DST; DsT_linear=true), slope*Tfo + offset; rtol=1e-12)
    end

    @testset "tau_drag is the DRAG, tau_n_main3 is the CURRENT time (ratio = K₃/K₂)" begin
        # 🔴 REGRESSION GUARD for the 2026-08-02 bug. `build_tau_n_spline` fed tau_n_main3 in as
        # 1/η_D, so the realised D_s was K₃/K₂ (1.26-1.74×) larger than the DsT label. Nothing in
        # this suite tested the drag, which is why it survived. These are the missing assertions.
        for T in (0.156, 0.20, 0.30, 0.40), dst in (0.116, 0.371)
            z  = M_CHARM / T
            τd = tau_drag(T, M_CHARM, dst)
            τn = tau_n_main3(T, M_CHARM, dst)
            # the drag IS the Einstein relation, nothing else
            @test isapprox(τd, M_CHARM * dst / T^2 / fmGeV; rtol = 1e-12)
            # a medium built from this drag realises the D_sT it is LABELLED with
            @test isapprox((T / (M_CHARM * (1/τd))) * T * fmGeV, dst; rtol = 1e-12)
            # the current time is the derived consequence, larger by exactly K₃/K₂
            @test isapprox(τn / τd, Kx(3, z) / Kx(2, z); rtol = 1e-12)
            @test τn > τd
        end
        # the two splines must differ by exactly K₃/K₂ pointwise
        Tmin, Tmax, nT = 0.10, 0.50, 128
        _,  _, dvals = build_tau_drag_spline(M_CHARM, DST; Tmin=Tmin, Tmax=Tmax, nT=nT)
        _,  _, nvals = build_taun_current_spline(M_CHARM, DST; Tmin=Tmin, Tmax=Tmax, nT=nT)
        dT = (Tmax - Tmin) / (nT - 1)
        for i in 1:nT
            z = M_CHARM / (Tmin + (i-1)*dT)
            @test isapprox(nvals[i] / dvals[i], Kx(3, z) / Kx(2, z); rtol = 1e-12)
        end
    end

    @testset "Einstein / FDR closure: κ = 2MTη_D ⟹ OU variance → MT" begin
        # The kernel defines η_D = 1/τ_drag, κ = 2MT/τ_drag, so κ/(2η_D) = MT exactly.
        T  = 0.3
        τn = tau_drag(T, M_CHARM, DST)
        ηD = 1.0 / τn
        κ  = 2 * M_CHARM * T / τn
        @test isapprox(κ / (2ηD), M_CHARM * T; rtol = 1e-12)

        # Exact OU propagator (kernels_cpu.jl, non-relativistic η_eff=η_D):
        #   p ← a·p + √Δt·noise_pref·ξ,  a=e^{-η_D Δt},  noise_pref=√κ·√((1-a²)/(2 η_D Δt)).
        # Its stationary variance is κ/(2η_D) = MT.
        Δt = 0.02 * τn
        a  = exp(-ηD * Δt)
        noise_pref = sqrt(κ) * sqrt((1 - a^2) / (2 * ηD * Δt))
        Random.seed!(20260615)
        Npart, Nsteps = 40_000, 800
        p = zeros(Npart)
        for _ in 1:Nsteps
            @. p = a * p + sqrt(Δt) * noise_pref * randn()
        end
        @test isapprox(var(p), M_CHARM * T; rtol = 0.05)
    end
end

# ── engine smoke tests through the public entry point (seconds) ────────────────────────────────
@testset "entry-point contract" begin
    xg = collect(0.0:0.5:60.0); tg = collect(0.0:0.5:3.0)
    Tf = fill(0.3, length(xg), length(tg)); Vf = zeros(length(xg), length(tg))
    rg = collect(0.0:0.5:20.0); pg = collect(0.05:0.1:8.0); dens = ones(length(pg), length(rg))
    N = 2_000; Random.seed!(1); x0 = randn(2, N); p0 = 0.8 .* randn(2, N)
    run(; kw...) = simulate_ensemble_bulk(CPUBackend(), rg, pg, dens, Tf, Vf, (xg, tg); x_init = x0, p_init = p0,
        N_particles = N, Δt = 1e-2, final_time = 0.2, save_interval = 0.1, m = 1.5, DsT = 0.2, dimensions = 2, Tfo = 0.0, kw...)
    t, mom, pos = run()
    @test length(t) == 3 && size(mom[end]) == (2, N) && size(pos[end]) == (2, N)
    t, mom, _ = run(collision_mode = :rta)                      # the BGK path runs and relaxes
    @test size(mom[end]) == (2, N) && all(isfinite, mom[end])
    t, mom, _ = run(momentum_dimensions = 3)
    @test size(mom[end]) == (3, N)
    @test_throws ErrorException run(collision_mode = :bogus)
    @test_throws ErrorException run(integrator_mode = 1)        # GPU-only scheme, refused on the CPU
    @test_throws DimensionMismatch LangevInMedium.KernelsCPU.kernel_rta_collision_cpu!(Tf, xg, tg, zeros(2, 4), zeros(2, 4),
        1e-2, 1.5, 4, 1, 0.0, 0.2; tau_Tmin = 0.1, tau_invdT = 1.0, tau_vals = [1.0, 1.0], dimensions = 3)
    # the sampler path (no x_init): both spatial modes, antithetic pairs are exact mirrors
    Random.seed!(2)
    _, mom, pos = simulate_ensemble_bulk(CPUBackend(), rg, pg, dens, Tf, Vf, (xg, tg); N_particles = N, Δt = 1e-2,
        final_time = 0.1, save_interval = 0.1, m = 1.5, DsT = 0.2, dimensions = 2, Tfo = 0.0, antithetic_momenta = true)
    @test all(mom[1][:, 2:2:end] .== -mom[1][:, 1:2:end-1])
    @test all(sqrt.(sum(abs2, pos[1]; dims = 1)) .<= rg[end] + 1e-12)
end

# Each suite below is evaluated in its OWN module. They are standalone scripts as well as members
# of this suite, so several of them define top-level constants with the same names (`M`, `NP`, the
# background grids) and different values; sharing `Main` would be an "invalid redefinition of
# constant" error the moment two of them met. `Base.include(m, path)` keeps them apart and needs no
# change to the files themselves.
function run_suite(f)
    m = Module(Symbol("LIM_", replace(f, ".jl" => "")))
    Core.eval(m, :(using Test))
    Base.include(m, joinpath(@__DIR__, f))
end

# ── unit coverage of the primitives (seconds; part of the fast set) ─────────────────────────────
# Every function the older suites never touched: the interpolant, the spline evaluator, both boost
# kernels, the two Jüttner samplers, the FONLL sampler's fidelity, LV_TAUN_SCALE, the box path.
run_suite("test_kernel_units.jl")
# WHEN each kernel reads the background — end-of-step for the lookups, start-of-step for the
# Bjorken redshift. Two conventions on purpose; pinned so the mixture stays deliberate.
run_suite("test_time_convention.jl")

if get(ENV, "LIM_FAST", "0") == "1"
    @info "LIM_FAST=1: skipping the engine gates (relativistic switch, momentum_dims3, CPU/GPU kernel parity, GPU-only paths)"
else
    # ── the relativistic switch must actually switch ────────────────────────────────
    # Added after the flag was found to be parsed by the drivers, recorded in the output
    # metadata, and never handed to the solver: two runs differing only in it came out
    # bit-identical.  See the file header for what each assertion guards.
    include(joinpath(@__DIR__, "test_relativistic_switch.jl"))
    include(joinpath(@__DIR__, "test_momentum_dims3.jl"))
    # ── the p_z FRAME, against a code that has z ────────────────────────────────────
    # Every other p_z gate is written in the same variables as the engine, so none of them can
    # detect a mistake in what row 3 *means*. This one integrates the same physics in plain
    # Cartesian lab coordinates with an explicit z and compares distributions. It also gates the
    # two 2026-09-01 additions: pz_init (:thermal | :comoving) and the η_s accumulator.
    run_suite("test_pz_lab_rapidity.jl")
    # ── CPU ↔ GPU, deterministically ────────────────────────────────────────────────
    # The ONLY exact comparison of the two backends: same inputs, same injected noise, per particle,
    # at 1e-12 or tighter. Everything else in the tree compares ensemble moments at 3 %, which is a
    # far weaker instrument than the 0.2.0 bug history warrants. Skips itself without CUDA.
    run_suite("test_kernel_parity.jl")
    # ── the GPU-only features production depends on ─────────────────────────────────
    # freezeout_capture (48 uses in Projects/) and integrator_mode = 1 (20 uses): both untested
    # until 2026-08-31, and the second does not do what it says. Skips itself without CUDA.
    run_suite("test_gpu_only_paths.jl")
end
