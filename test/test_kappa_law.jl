#!/usr/bin/env julia
# ==============================================================================
# test_kappa_law.jl — gates for κ_L(p, T) ≠ κ_T(p, T) (`kappa_law`, src/kappa_law.jl; 2026-10-24).
#
#   K1  the Green–Kubo ℓ = 1 solver: R = 1 for constant κ (d = 2, 3; three T), converged in K
#   K2  Jüttner stationary under two non-constant laws (falling isotropic; anisotropic κ_L ≠ κ_T),
#       ⟨p²⟩ and ⟨E⟩ against the exact Jüttner — and the negative control: the same step with the
#       naive drift κ_L/(2TE) (no −∂_j B_ij) visibly leaves it
#   K3  a constant law (g = 1, norm = :none) reproduces the shipped constant-κ kernel particle by
#       particle (to rounding) through the full driver on a flowing fireball, proper_time_kicks off and on
#   K4  the measured spatial diffusion (MSD slope, static box) equals D_sT/T under :green_kubo and
#       does not under :rest — the normalisation is what the label says
#   K5  every combination the law would be ignored in is refused
#   K6  (CUDA only) the GPU kernel equals the CPU kernel to rounding on the same inputs
#
#   julia --project=Julia Julia/LangevInMedium.jl/test/test_kappa_law.jl
# ==============================================================================
ENV["LIM_QUIET"] = "1"
using Test, Random, Statistics, Bessels
const HAVE_CUDA = try                     # before `using LangevInMedium`, so the Requires hook fires
    @eval using CUDA
    Base.invokelatest(() -> CUDA.functional())
catch
    false
end
using LangevInMedium
const KC = LangevInMedium.KernelsCPU
const KL = LangevInMedium.KappaLaws

const M  = 1.5
const T0 = 0.3
const DST = 0.2

Econ(p) = sqrt(M^2 + p^2)
const CONST = KappaLaw((p, T) -> 1.0)
"falling isotropic law (KD's T-matrix-like shape): κ falls by 3 from rest to high p on a 2 GeV scale"
fall(p, T) = 1 / 3 + (2 / 3) / (1 + (p / 2.0)^2)
"anisotropic, κ_L rising faster than κ_T (the pQCD ordering)"
gLa(p, T) = 1 + 1.0 * p^2 / M^2
gTa(p, T) = 1 + 0.2 * p^2 / M^2

function juttner3(N, T; rng = Random.default_rng())
    P = zeros(3, N)
    for i in 1:N
        ps = KC._draw_juttner_pstar_lib(M, T, 3)
        g = randn(rng, 3); g ./= sqrt(sum(abs2, g))
        P[:, i] .= ps .* g
    end
    return P
end

Kx(n, z) = Bessels.besselkx(n, z)
juttner_p2(T) = 3M * T * Kx(3, M / T) / Kx(2, M / T)
juttner_E(T)  = 3T + M * Kx(1, M / T) / Kx(2, M / T)

# --- K2 helper: the momentum step alone in a uniform box (kernels called directly) -------------
function box_run(table, N, steps, Δt; seed = 7)
    Random.seed!(seed)
    P = juttner3(N, T0)
    X = fill(1.0, 2, N)
    tg = [0.0, 100.0]; xg = [0.0, 100.0]; Tf = fill(T0, 2, 2)
    τmin, τinv, τvals = build_tau_drag_spline(M, DST; Tmin = T0, Tmax = T0 + eps(), nT = 1024)
    ξ = zeros(3, N); det = zeros(3, N); sto = zeros(3, N)
    for s in 1:steps
        randn!(ξ)
        KC.kernel_compute_forces_kappa_cpu!(Tf, xg, tg, P, X, ξ, det, sto, Δt, M, 3, N, s, 0.0;
            tau_Tmin = τmin, tau_invdT = τinv, tau_vals = τvals, ktab = table)
        KC.kernel_update_momenta_LRF_cpu!(P, det, sto, Δt, 3, N)
    end
    p2 = vec(sum(abs2, P; dims = 1))
    return mean(p2), std(p2) / sqrt(N), mean(sqrt.(p2 .+ M^2)), std(sqrt.(p2 .+ M^2)) / sqrt(N)
end

# --- a small flowing fireball for the driver-level gates -----------------------------------------
xg = collect(range(0.0, 12.0; length = 61)); tg = collect(range(0.4, 3.0; length = 27))
Tfb = [max(0.16, 0.42 * exp(-x^2 / 40) * (0.4 / t)^(1 / 3)) for x in xg, t in tg]
Vfb = [0.6 * tanh(x / 6) * (1 - exp(-(t - 0.4))) for x in xg, t in tg]

@testset "kappa_law" begin

@testset "K1 Green–Kubo ratio: R = 1 for constant κ, converged" begin
    for d in (2, 3), T in (0.16, 0.3, 0.5)
        R = kappa_green_kubo_ratio((p, T) -> 1.0, (p, T) -> 1.0, M, T; d = d)
        @test abs(R - 1) < 2e-5
    end
    Ra = kappa_green_kubo_ratio(gLa, gTa, M, T0; d = 3, K = 2000)
    Rb = kappa_green_kubo_ratio(gLa, gTa, M, T0; d = 3, K = 8000)
    @test abs(Ra / Rb - 1) < 1e-4
    Rf = kappa_green_kubo_ratio(fall, fall, M, T0; d = 3)
    @info "K1" R_fall = Rf R_aniso = Rb
    @test Rf > 1.03            # falling κ ⇒ larger D_s than κ(0) says — K4 needs this to discriminate
end

@testset "K2 Jüttner stationary; the naive drift is not" begin
    N = 200_000; Δt = 0.005; steps = 1000          # η_DΔt ≈ 0.008; 5 fm ≈ 7.6 drag times
    p2J = juttner_p2(T0); EJ = juttner_E(T0)
    for (name, law) in (("falling", KappaLaw(fall)), ("anisotropic", KappaLaw(gLa, gTa)))
        tab = build_kappa_table(law, M, 3; Tmin = T0, Tmax = T0)
        p2, σp2, E, σE = box_run(tab, N, steps, Δt)
        @info "K2 $name (Itô drift)" p2_rel = p2 / p2J - 1 sigma = σp2 / p2J E_rel = E / EJ - 1
        @test abs(p2 / p2J - 1) < 4σp2 / p2J + 2e-3   # 2e-3: the O(ηΔt) step bias allowance
        @test abs(E / EJ - 1) < 4σE / EJ + 5e-4
        tabn = build_kappa_table(law, M, 3; Tmin = T0, Tmax = T0, drift = :naive)
        p2n, σn, _, _ = box_run(tabn, N, steps, Δt)
        @info "K2 $name (naive drift — control)" p2_rel = p2n / p2J - 1 sigma = σn / p2J
        @test abs(p2n / p2J - 1) > 10 * (σn / p2J + 2e-3)
    end
end

@testset "K3 constant law ≡ the shipped kernel, particle by particle (driver, flowing)" begin
    N = 4000
    Random.seed!(11)
    x0 = 2.5 .* randn(2, N); p0 = juttner3(N, 0.3)
    for ptk in (false, true)
        kw = (N_particles = N, Δt = 0.002, initial_time = 0.4, final_time = 1.4, save_interval = 1.0,
              m = M, DsT = 0.11634, dimensions = 2, momentum_dimensions = 3, x_init = x0, p_init = p0,
              proper_time_kicks = ptk)
        Random.seed!(5); _, P0, X0 = simulate_ensemble_bulk(CPUBackend(), nothing, nothing, nothing, Tfb, Vfb, (xg, tg); kw...)
        Random.seed!(5); _, P1, X1 = simulate_ensemble_bulk(CPUBackend(), nothing, nothing, nothing, Tfb, Vfb, (xg, tg);
                                                             kw..., kappa_law = KappaLaw((p, T) -> 1.0; norm = :none))
        Random.seed!(5); _, P2, _ = simulate_ensemble_bulk(CPUBackend(), nothing, nothing, nothing, Tfb, Vfb, (xg, tg);
                                                            kw..., kappa_law = CONST)
        dp1 = maximum(abs, P1[end] .- P0[end]); dx1 = maximum(abs, X1[end] .- X0[end])
        dp2 = maximum(abs, P2[end] .- P0[end])
        @info "K3 proper_time_kicks=$ptk" max_dp_none = dp1 max_dx_none = dx1 max_dp_gk = dp2
        @test dp1 < 1e-10         # rounding: Λ ≡ 1 is exact on the table
        @test dx1 < 1e-10
        @test dp2 < 1e-4          # :green_kubo adds the solver's R − 1 (≲ 2e-5, K1)
    end
end

@testset "K4 measured D_s = D_sT/T under :green_kubo, not under :rest" begin
    N = 100_000
    xgb = [0.0, 5000.0]; tgb = [0.0, 100.0]
    Tb = fill(T0, 2, 2); Vb = zeros(2, 2)
    Dtarget = DST / T0 / fmGeV                         # fm
    out = Dict{Symbol, Float64}()
    for norm in (:green_kubo, :rest)
        Random.seed!(21)
        x0 = fill(2500.0, 2, N); x0[2, :] .= 0.0       # r > 0 so the boosts see a direction
        p0 = juttner3(N, T0)
        _, _, X = simulate_ensemble_bulk(CPUBackend(), nothing, nothing, nothing, Tb, Vb, (xgb, tgb);
            N_particles = N, Δt = 0.005, initial_time = 0.0, final_time = 12.0, save_interval = 6.0,
            m = M, DsT = DST, dimensions = 2, momentum_dimensions = 3, x_init = x0, p_init = p0,
            kappa_law = KappaLaw(fall; norm = norm))
        msd(k) = mean(sum(abs2, X[k] .- X[1]; dims = 1)) / 2      # per component
        D = (msd(3) - msd(2)) / (2 * 6.0)
        out[norm] = D / Dtarget
    end
    Rf = kappa_green_kubo_ratio(fall, fall, M, T0; d = 3)
    @info "K4 D_meas/(D_sT/T)" green_kubo = out[:green_kubo] rest = out[:rest] predicted_rest = Rf
    @test abs(out[:green_kubo] - 1) < 0.02
    @test abs(out[:rest] / Rf - 1) < 0.02
    @test abs(out[:rest] - 1) > 0.03
end

@testset "K5 refusals" begin
    law = KappaLaw(gLa, gTa)
    base = (N_particles = 10, Δt = 0.01, initial_time = 0.4, final_time = 0.5, save_interval = 0.1,
            m = M, DsT = 0.2, x_init = ones(2, 10), p_init = zeros(2, 10), kappa_law = law)
    call(; kw...) = simulate_ensemble_bulk(CPUBackend(), nothing, nothing, nothing, Tfb, Vfb, (xg, tg); base..., kw...)
    @test_throws ErrorException call(dimensions = 1, p_init = zeros(1, 10))
    @test_throws ErrorException call(dimensions = 2, relativistic = false)
    @test_throws ErrorException call(dimensions = 2, collision_mode = :rta)
    @test_throws ErrorException call(dimensions = 2, collision_mode = :none)
    @test_throws ErrorException call(dimensions = 2, DsT = 0.0)
    @test_throws ErrorException call(dimensions = 2, momentum_langevin = false)
    @test_throws ErrorException call(dimensions = 2, adaptive_dt = true)
    @test_throws ErrorException KappaLaw(gLa; norm = :foo)
    # κ_L(0) ≠ κ_T(0) is not a law
    @test_throws ErrorException build_kappa_table(KappaLaw((p, T) -> 1.0, (p, T) -> 2.0), M, 3; Tmin = 0.2, Tmax = 0.3)
    @test call(dimensions = 2) isa Tuple                     # and the legal call runs
end

end

if HAVE_CUDA
@testset "K6 GPU kernel ≡ CPU kernel on the same inputs" begin
    KG = LangevInMedium.Simulate.KernelsGPU
    N = 20_000
    Random.seed!(3)
    P = vcat(4 .* randn(2, N), randn(1, N)); X = 5 .* rand(2, N) .+ 0.5; ξ = randn(3, N)
    τmin, τinv, τvals = build_tau_drag_spline(M, 0.11634; Tmin = minimum(Tfb), Tmax = maximum(Tfb), nT = 1024)
    tab = build_kappa_table(KappaLaw(gLa, gTa), M, 3; Tmin = minimum(Tfb), Tmax = maximum(Tfb))
    for ptk in (false, true)
        dc = zeros(3, N); sc = zeros(3, N)
        KC.kernel_compute_forces_kappa_cpu!(Tfb, xg, tg, P, X, ξ, dc, sc, 0.002, M, 3, N, 100, 0.4;
            tau_Tmin = τmin, tau_invdT = τinv, tau_vals = τvals, ktab = tab,
            proper_time_kicks = ptk, Vfield = Vfb)
        dg = CUDA.zeros(Float64, 3, N); sg = CUDA.zeros(Float64, 3, N)
        CUDA.@sync @cuda threads = 256 blocks = cld(N, 256) KG.kernel_compute_forces_kappa_gpu!(
            CuArray(Tfb), CuArray(xg), CuArray(tg), CuArray(P), CuArray(X), CuArray(ξ), dg, sg,
            0.002, M, 3, N, 100, 0.4, τmin, τinv, CuArray(τvals),
            CuArray(tab.vals), tab.invdp, tab.np, tab.Tmin, tab.invdT, tab.nT,
            ptk, CuArray(Vfb), nothing, nothing, nothing)
        e = max(maximum(abs, Array(dg) .- dc), maximum(abs, Array(sg) .- sc))
        @info "K6 proper_time_kicks=$ptk" max_abs_diff = e
        @test e < 1e-12
    end
end
else
    @warn "K6 skipped: CUDA not functional — the GPU kernel is NOT checked"
end
