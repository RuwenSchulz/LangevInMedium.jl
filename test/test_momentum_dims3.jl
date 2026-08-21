# ==============================================================================================
# test_momentum_dims3.jl — a 2-D transverse plane carrying THREE momentum components.
#
# WHAT IS AT STAKE. The hydrodynamic side matches τ_n = D_s z K₃/K₂ in the 3-D Jüttner measure;
# an engine evolving two momentum components relaxes to the 2-D Jüttner, whose first-moment
# relaxation rate λ₁ η_D differs from K₂/K₃ η_D by 5–12 % over z = 3.5–10. `momentum_dimensions = 3`
# is supposed to remove that offset. These gates establish, with the SHIPPED kernels:
#
#   (Q) the identity λ₁(3D) ≡ ⟨(M/E)p²⟩/⟨p²⟩ = K₂/K₃ by quadrature, and that λ₁(2D) is NOT it;
#   (S) the initial p_z draw is the exact thermal conditional (isotropy against a 3-D Jüttner);
#   (E) the 3-D run equilibrates to the 3-D Jüttner (⟨p²⟩ and the ensemble λ₁ = K₂/K₃), while the
#       2-D run — same engine, same drag — gives λ₁(2D): the offset is the dimension, nothing else;
#   (D) the ℓ=1 current of a boosted Jüttner decays at λ₁ η_D — the rate the hydro τ_n encodes;
#   (R) the Bjorken redshift telescopes exactly: ⟨p_z²⟩(τ) = ⟨p_z²⟩(τ₀)(τ₀/τ)² under free streaming;
#   (M) the spatial diffusion coefficient measured from the particles' own MSD is the D_s handed in
#       (the physical-units test that the drag convention cannot fake);
#   (G) the GPU path agrees with the CPU path on (E) and (M).
#
#   julia --project=Julia Julia/LangevInMedium.jl/test/test_momentum_dims3.jl
# ==============================================================================================
using Test, Random, Statistics, LinearAlgebra, QuadGK, Bessels, LangevInMedium
const U = LangevInMedium.Utils
const HBARC = 0.1973269804

# ── reference quantities by quadrature ───────────────────────────────────────────────────────────
"⟨g(p)⟩ in the d-dimensional Jüttner at (M,T): measure p^(d-1) e^{-E/T}."
function jmean(g, M, T, d)
    w(p) = p^(d - 1) * exp(-(sqrt(p^2 + M^2) - M) / T)
    num, _ = quadgk(p -> g(p) * w(p), 0, Inf; rtol = 1e-10)
    den, _ = quadgk(w, 0, Inf; rtol = 1e-10)
    num / den
end
lam1(M, T, d) = jmean(p -> (M / sqrt(p^2 + M^2)) * p^2, M, T, d) / jmean(p -> p^2, M, T, d)
p2eq(M, T, d) = jmean(p -> p^2, M, T, d)
K2K3(z) = Bessels.besselkx(2, z) / Bessels.besselkx(3, z)

"Isotropic d-dimensional Jüttner sample (rejection), (d, N)."
function juttner_sample(rng, M, T, d, N)
    pmax = 14 * sqrt(M * T + T^2)
    w(p) = p^(d - 1) * exp(-(sqrt(p^2 + M^2) - M) / T)
    wmax = maximum(w, range(1e-6, pmax; length = 4000))
    P = zeros(d, N)
    for i in 1:N
        p = 0.0
        while true
            p = pmax * rand(rng); rand(rng) * wmax <= w(p) && break
        end
        n = randn(rng, d); n ./= norm(n)
        P[:, i] .= p .* n
    end
    P
end

# ── the static box through the PUBLIC entry point (the real kernels, all of them) ───────────────
"""
Uniform bath at T, zero flow, particles handed in via x_init/p_init (so the engine's only freedom
is the dynamics). Returns (times, mom_hist, pos_hist).
"""
function box_run(backend; M, T, DsT, N, dt, tfinal, save, x0, p0, pdim, t0 = 0.0,
                 redshift = false, momentum_langevin = true, seed = 1)
    Random.seed!(seed)
    xgrid = collect(0.0:0.5:400.0); tgrid = collect(t0:1.0:(t0 + tfinal + 2))
    Tf = fill(T, length(xgrid), length(tgrid)); Vf = zeros(length(xgrid), length(tgrid))
    dens = ones(length(xgrid), 10)            # unused: x_init/p_init bypass the sampler
    simulate_ensemble_bulk(backend, xgrid, collect(range(0, 10; length = 10)), dens, Tf, Vf,
        (xgrid, tgrid); x_init = x0, p_init = p0, N_particles = N, Δt = dt,
        initial_time = t0, final_time = t0 + tfinal, save_interval = save, m = M, DsT = DsT,
        dimensions = 2, momentum_dimensions = pdim, bjorken_redshift = redshift,
        position_diffusion = false, momentum_langevin = momentum_langevin,
        reflecting_boundary = false, Tfo = 0.0)
end

"Ensemble estimators on one snapshot: ⟨p²⟩, λ₁ = ⟨(M/E)p²⟩/⟨p²⟩, ⟨p_x⟩, ⟨(M/E)p_x⟩."
function est(mom, M)
    p2 = vec(sum(abs2, mom; dims = 1)); E = sqrt.(M^2 .+ p2)
    (p2 = mean(p2), lam = mean((M ./ E) .* p2) / mean(p2),
     jx = mean(view(mom, 1, :)), jxw = mean((M ./ E) .* view(mom, 1, :)))
end

const M_, T_ = 1.5, 0.30          # z = 5: λ₁(2D)/λ₁(3D) = 1.09, a clean 9 % to see
const DST_ = 0.11634
const ETA_ = (T_^2 / (M_ * DST_)) / HBARC     # fm⁻¹, the Einstein drag the kernel builds
const DS_FM = (DST_ / T_) * HBARC             # D_s = D_sT/T [GeV⁻¹] → fm (×ħc)

@testset "momentum_dimensions = 3" begin

    @testset "(Q) λ₁ identity: 3-D is K₂/K₃, 2-D is not" begin
        for z in (3.48, 5.0, 7.0, 9.62)
            T = M_ / z
            @test isapprox(lam1(M_, T, 3), K2K3(z); rtol = 1e-9)
            ratio = lam1(M_, T, 2) / lam1(M_, T, 3)
            @test 1.04 < ratio < 1.13          # the 5–12 % the paper is about
        end
        println("  (Q) λ₁(2D)/λ₁(3D) at z=3.48,5,7,9.62: ",
                join([string(round(lam1(M_, M_ / z, 2) / lam1(M_, M_ / z, 3); digits = 4)) for z in (3.48, 5.0, 7.0, 9.62)], ", "))
    end

    @testset "(S) p_z conditional draw is the thermal one" begin
        rng = MersenneTwister(7)
        # fixed m_T: ⟨p_z²⟩ against quadrature of exp(-√(m_T²+p_z²)/T)
        for (mT, T) in ((1.6, 0.3), (2.5, 0.45), (1.5, 0.156))
            s = [U.sample_pz_conditional_juttner(mT, T; rng = rng) for _ in 1:200_000]
            num, _ = quadgk(pz -> pz^2 * exp(-(sqrt(mT^2 + pz^2) - mT) / T), -Inf, Inf; rtol = 1e-10)
            den, _ = quadgk(pz -> exp(-(sqrt(mT^2 + pz^2) - mT) / T), -Inf, Inf; rtol = 1e-10)
            @test isapprox(mean(s .^ 2), num / den; rtol = 0.02)
            @test abs(mean(s)) < 4 * std(s) / sqrt(length(s))
        end
        # joint: transverse rows of a 3-D Jüttner + the conditional p_z must be ISOTROPIC
        P3 = juttner_sample(rng, M_, T_, 3, 200_000)
        x  = zeros(2, size(P3, 2))                     # positions only feed T_of_r (uniform here)
        P  = U.append_thermal_pz(P3[1:2, :], x, M_, r -> T_; rng = rng)
        @test isapprox(mean(P[3, :] .^ 2), mean(P3[1, :] .^ 2); rtol = 0.02)
        @test isapprox(mean(P[3, :] .^ 2), mean(P[1, :] .^ 2); rtol = 0.02)
        # antithetic mirroring is exact
        Pa = U.append_thermal_pz(P3[1:2, 1:1000], x[:, 1:1000], M_, r -> T_; antithetic = true, rng = rng)
        @test all(Pa[3, 2:2:end] .== -Pa[3, 1:2:end-1])
    end

    # One 3-D and one 2-D run from a hot isotropic start, both through the CPU engine.
    N_E = 150_000; dt_E = 2e-3; tfin_E = 4.0           # t_final·η_D ≈ 10 relaxation times
    rng = MersenneTwister(11)
    x0 = zeros(2, N_E)
    p0_3 = 0.9 .* randn(rng, 3, N_E); p0_2 = p0_3[1:2, :]
    t3, m3, _ = box_run(CPUBackend(); M = M_, T = T_, DsT = DST_, N = N_E, dt = dt_E, tfinal = tfin_E,
                        save = 0.5, x0 = x0, p0 = p0_3, pdim = 3)
    t2, m2, _ = box_run(CPUBackend(); M = M_, T = T_, DsT = DST_, N = N_E, dt = dt_E, tfinal = tfin_E,
                        save = 0.5, x0 = x0, p0 = p0_2, pdim = 2)
    e3 = est(m3[end], M_); e2 = est(m2[end], M_)

    @testset "(E) equilibria: 3-D run is the 3-D Jüttner, 2-D run the 2-D one" begin
        @test size(m3[end], 1) == 3 && size(m2[end], 1) == 2
        @test isapprox(e3.p2, p2eq(M_, T_, 3); rtol = 0.015)
        @test isapprox(e2.p2, p2eq(M_, T_, 2); rtol = 0.015)
        @test isapprox(e3.lam, K2K3(M_ / T_); rtol = 0.005)         # ensemble λ₁ = K₂/K₃
        @test isapprox(e2.lam, lam1(M_, T_, 2); rtol = 0.005)       # ensemble λ₁ = λ₁(2D)
        @test e2.lam / e3.lam > 1.06                                # and they differ, by the 9 %
        # isotropy of the 3-D run, p_z included
        @test isapprox(mean(m3[end][3, :] .^ 2), mean(m3[end][1, :] .^ 2); rtol = 0.02)
        println("  (E) ⟨p²⟩ 3D ", round(e3.p2; digits = 4), " (ref ", round(p2eq(M_, T_, 3); digits = 4),
                ")   2D ", round(e2.p2; digits = 4), " (ref ", round(p2eq(M_, T_, 2); digits = 4), ")")
        println("  (E) λ₁  3D ", round(e3.lam; digits = 4), " (K₂/K₃ ", round(K2K3(M_ / T_); digits = 4),
                ")   2D ", round(e2.lam; digits = 4), " (λ₁(2D) ", round(lam1(M_, T_, 2); digits = 4), ")")
    end

    @testset "(D) ℓ=1 decay of a boosted Jüttner: rate = λ₁ η_D, 3-D vs 2-D" begin
        # exact boosted Jüttner: sample in the rest frame, Lorentz-boost every particle by v along x.
        # f(p) is a scalar and the particles are the same set, so this IS f = exp(−u·p/T).
        # v small: the reference λ₁ is LINEAR response (δf ∝ p_x f_eq); v=0.15 already shifts the
        # ensemble ratio by 1.5 % through the O(v²) shape, so the gate works at v=0.05.
        v = 0.05; γ = 1 / sqrt(1 - v^2); N_D = 400_000
        rates = Dict{Int,Float64}(); refs = Dict{Int,Float64}()
        for d in (3, 2)
            P = juttner_sample(MersenneTwister(100 + d), M_, T_, d, N_D)
            for i in 1:N_D
                E = sqrt(M_^2 + sum(abs2, view(P, :, i))); P[1, i] = γ * (P[1, i] + v * E)
            end
            tt, mm, _ = box_run(CPUBackend(); M = M_, T = T_, DsT = DST_, N = N_D, dt = 1e-3,
                                tfinal = 0.30, save = 0.05, x0 = zeros(2, N_D), p0 = P, pdim = d, seed = 5 + d)
            jx = [est(m, M_).jx for m in mm]; tt = collect(tt)
            # exact moment equation d⟨p_x⟩/dt = −η_D⟨(M/E)p_x⟩ ⇒ the instantaneous rate on the ensemble
            r_inst = ETA_ * est(mm[1], M_).jxw / est(mm[1], M_).jx
            # and the measured slope of ln⟨p_x⟩ over the window (η_D·t ≤ 0.8)
            A = hcat(ones(length(tt)), tt); c = A \ log.(jx)
            rates[d] = -c[2]; refs[d] = (d == 3 ? K2K3(M_ / T_) : lam1(M_, T_, d)) * ETA_
            @test isapprox(r_inst, refs[d]; rtol = 0.012)     # the IC realises the mode shape
            @test isapprox(rates[d], refs[d]; rtol = 0.04)    # the engine decays it at that rate
        end
        @test rates[2] / rates[3] > 1.05                          # the dimension offset is dynamical
        println("  (D) ℓ=1 rate/η_D  3D ", round(rates[3] / ETA_; digits = 4), " (K₂/K₃ ", round(K2K3(M_ / T_); digits = 4),
                ")   2D ", round(rates[2] / ETA_; digits = 4), " (λ₁(2D) ", round(lam1(M_, T_, 2); digits = 4), ")")
    end

    @testset "(R) Bjorken redshift telescopes exactly under free streaming" begin
        # momentum_langevin=false ⇒ the transverse rows are glued to the (zero) flow and p_z evolves
        # ONLY by the redshift, so ∏ τ_a/τ_b = τ₀/τ to machine precision.
        N_R = 20_000; t0 = 0.4; tf = 2.0; dt = 1e-3
        P = hcat([vcat(0.3 .* randn(2), 0.5 .* randn(1)) for _ in 1:N_R]...)
        tt, mm, _ = box_run(CPUBackend(); M = M_, T = T_, DsT = DST_, N = N_R, dt = dt, tfinal = tf,
                            save = 0.4, x0 = 4.0 .* ones(2, N_R), p0 = P, pdim = 3, t0 = t0,
                            redshift = true, momentum_langevin = false)
        for (k, τ) in enumerate(tt)
            @test isapprox(mean(mm[k][3, :] .^ 2), mean(P[3, :] .^ 2) * (t0 / τ)^2; rtol = 1e-9)
        end
        # and the contract refuses the meaningless combinations
        @test_throws ErrorException U.check_momentum_dims(2, 3, false, true, 0.0)
        @test_throws ErrorException U.check_momentum_dims(2, 4, false, false, 0.4)
        @test_throws ErrorException U.check_momentum_dims(3, 2, false, false, 0.4)
    end

    @testset "(M) spatial diffusion: MSD slope = 4 D_s (3-D momenta, 2-D plane)" begin
        N_M = 100_000; tf = 8.0
        P = juttner_sample(MersenneTwister(3), M_, T_, 3, N_M)     # start IN equilibrium
        tt, _, xx = box_run(CPUBackend(); M = M_, T = T_, DsT = DST_, N = N_M, dt = 2e-3, tfinal = tf,
                            save = 0.5, x0 = 50.0 .* ones(2, N_M), p0 = P, pdim = 3, seed = 9)
        msd = [mean(sum(abs2, x .- xx[1]; dims = 1)) for x in xx]; tt = collect(tt)
        sel = tt .>= 4.0                                            # ≫ τ_drag = 0.38 fm: diffusive
        c = hcat(ones(sum(sel)), tt[sel]) \ msd[sel]
        Ds_meas = c[2] / 4                                          # 2 spatial dimensions
        @test isapprox(Ds_meas, DS_FM; rtol = 0.03)
        println("  (M) D_s measured/nominal = ", round(Ds_meas / DS_FM; digits = 4))
    end

    @testset "(G) GPU path agrees with CPU" begin
        cuda_ok = try
            @eval using CUDA
            Base.invokelatest(() -> CUDA.functional())
        catch
            false
        end
        if !cuda_ok
            @warn "CUDA not functional — GPU agreement gate skipped"
        else
            tg, mg, xg = Base.invokelatest(box_run, GPUBackend(); M = M_, T = T_, DsT = DST_, N = N_E, dt = dt_E,
                                           tfinal = tfin_E, save = 0.5, x0 = x0, p0 = p0_3, pdim = 3)
            eg = est(mg[end], M_)
            @test size(mg[end], 1) == 3
            @test isapprox(eg.p2, e3.p2; rtol = 0.02)
            @test isapprox(eg.lam, K2K3(M_ / T_); rtol = 0.005)
            @test isapprox(mean(mg[end][3, :] .^ 2), mean(mg[end][1, :] .^ 2); rtol = 0.02)
            # redshift on the device: same telescoping
            N_R = 20_000; t0 = 0.4
            P = hcat([vcat(0.3 .* randn(2), 0.5 .* randn(1)) for _ in 1:N_R]...)
            tr, mr, _ = Base.invokelatest(box_run, GPUBackend(); M = M_, T = T_, DsT = DST_, N = N_R, dt = 1e-3,
                                          tfinal = 2.0, save = 0.4, x0 = 4.0 .* ones(2, N_R), p0 = P, pdim = 3,
                                          t0 = t0, redshift = true, momentum_langevin = false)
            for (k, τ) in enumerate(tr)
                @test isapprox(mean(mr[k][3, :] .^ 2), mean(P[3, :] .^ 2) * (t0 / τ)^2; rtol = 1e-9)
            end
            println("  (G) GPU ⟨p²⟩ ", round(eg.p2; digits = 4), "  λ₁ ", round(eg.lam; digits = 4))
        end
    end
end
