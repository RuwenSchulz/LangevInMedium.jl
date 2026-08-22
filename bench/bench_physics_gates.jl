#!/usr/bin/env julia
# ==============================================================================================
# bench_physics_gates.jl — the engine against quantities in PHYSICAL UNITS.
#
# The 2026-08-02 drag/current bug (realised D_s = K₃/K₂ × the label, 1.26–1.74×) survived every
# agreement test in the tree because τ_n cancels in a comparison of two implementations of the same
# formula; it fell the moment the particles' own MSD was held against D_s = (D_sT/T)·ħc. These
# gates are of that kind — each one pins the engine to a number it cannot fake:
#
#   (a) MSD slope = 2·d·D_s at three z = M/T, CPU and GPU         (the drag convention)
#   (b) the Jüttner TAIL f(p > 3 GeV) at the Poisson floor          (⟨p²⟩ alone missed a 16.5 % tail error)
#   (c) ℓ=1 current decay rate = η_D·K₂/K₃ (3 rows) and λ₁(2D)·η_D (2 rows)
#   (d) the exact-OU propagator has NO Δt bias for the Galilean drag; the pre-point relativistic
#       drag's O(Δt) bias on ⟨p²⟩ stays within a 1.5 % budget up to ηΔt = 0.1 (production: 0.003)
#   (e) Galilean twin: the full MSD(t) curve ⟨Δx²⟩ = 2d(T/Mη)[t − (1−e^{−ηt})/η], not only its slope
#   (f) DsT_quad: the realised drag rate is the SAME at T = 0.2 and 0.4 (const D_sT would give ×4)
#
#   julia --project=Julia Julia/LangevInMedium.jl/bench/bench_physics_gates.jl     (LIM_NOGPU=1 to skip the GPU)
# ==============================================================================================
include(joinpath(@__DIR__, "bench_common.jl"))
using QuadGK, Bessels
const HAVE_GPU = gpu_available()
const M, DST = 1.5, 0.11634
eta_drag(T) = (T^2 / (M * DST)) / HBARC             # fm⁻¹ — Einstein drag the kernel builds
ds_fm(T) = (DST / T) * HBARC                        # fm — D_s = D_sT/T [GeV⁻¹] → fm
K2K3(z) = Bessels.besselkx(2, z) / Bessels.besselkx(3, z)
lam1(T, d) = jmean(p -> (M / sqrt(p^2 + M^2)) * p^2, M, T, d) / jmean(p -> p^2, M, T, d)

function box_run(backend, T; N, dt, tfinal, save, x0, p0, seed, kw...)
    xg, tg, Tf, Vf = box_fields(T; tf = tfinal)
    run_fields(backend, xg, tg, Tf, Vf; M, DsT = DST, N, dt, tfinal, save, x0, p0, seed, kw...)
end
msd(xx) = [mean(sum(abs2, x .- xx[1]; dims = 1)) for x in xx]

println("── (a) MSD slope = 2·d·D_s ──")
for T in (0.45, 0.30, 0.156), backend in (HAVE_GPU ? (CPUBackend(), GPUBackend()) : (CPUBackend(),))
    N = 60_000; tf = 12 / eta_drag(T) + 6.0                       # ≫ τ_drag, then a diffusive window
    P = juttner_sample(MersenneTwister(3), M, T, 3, N)
    tt, _, xx = box_run(backend, T; N, dt = 2e-3, tfinal = tf, save = tf / 8, x0 = 100.0 .* ones(2, N), p0 = P,
                        seed = 9, momentum_dimensions = 3)
    tt = collect(tt); m = msd(xx); sel = tt .>= tf / 2
    c = hcat(ones(sum(sel)), tt[sel]) \ m[sel]
    ratio = (c[2] / 4) / ds_fm(T)
    gate!(isapprox(ratio, 1.0; atol = 0.03), "(a) $(nameof(typeof(backend))) z=$(fmt(M/T; d=2)): D_s measured/nominal = $(fmt(ratio))")
end

println("── (b) Jüttner tail ──")
let T = 0.30, N = 300_000, pcut = 3.0
    P = juttner_sample(MersenneTwister(5), M, T, 2, N)           # start IN equilibrium: any drift is dynamical
    tf = 10 / eta_drag(T)
    _, mm, _ = box_run(CPUBackend(), T; N, dt = 2e-3, tfinal = tf, save = tf / 2, x0 = zeros(2, N), p0 = P, seed = 17)
    frac(m) = count(>(pcut^2), vec(sum(abs2, m; dims = 1))) / size(m, 2)
    f_ref = jmean(p -> p > pcut ? 1.0 : 0.0, M, T, 2)
    f0, f1 = frac(mm[1]), frac(mm[end]); σ = sqrt(f_ref / N)
    gate!(abs(f0 - f_ref) < 4σ, "(b) IC tail fraction p>$(pcut): $(fmt(f0; d=5)) vs Jüttner $(fmt(f_ref; d=5)) (σ=$(fmt(σ; d=5)))")
    gate!(abs(f1 - f_ref) < 4σ, "(b) tail after 10 τ_drag: $(fmt(f1; d=5)) vs $(fmt(f_ref; d=5)) — $(fmt(abs(f1-f_ref)/σ; d=1))σ")
end

println("── (c) ℓ=1 decay rate ──")
# Two estimators of the rate at which a boosted Jüttner's current ⟨p_x⟩ decays:
#   · the INSTANTANEOUS rate the ensemble realises, η_D⟨(M/E)p_x⟩/⟨p_x⟩ (exact moment equation) — a
#     ratio of two sums over the same particles, noise ≈ 0.2 % at N = 10⁶; it starts at λ₁ and drifts
#     down by ≈ 1 % over ηt ≤ 0.8 because δf ∝ p_x f_eq is not an eigenmode (slow high-E particles);
#   · the FITTED slope of ln⟨p_x⟩ — carries the accumulated kick noise, ±2.4 % at N = 10⁶ (seed scan:
#     0.683, 0.703, 0.688, 0.715 around λ₁ = 0.698). Gate it at the noise floor, not tighter.
let T = 0.30, N = 1_000_000, v = 0.05
    γ = 1 / sqrt(1 - v^2)
    for d in (3, 2)
        P = juttner_sample(MersenneTwister(100 + d), M, T, d, N)
        for i in 1:N
            E = sqrt(M^2 + sum(abs2, view(P, :, i))); P[1, i] = γ * (P[1, i] + v * E)
        end
        tt, mm, _ = box_run(CPUBackend(), T; N, dt = 1e-3, tfinal = 0.30, save = 0.05, x0 = zeros(2, N), p0 = P,
                            seed = 5 + d, momentum_dimensions = d)
        jx = [mean(m[1, :]) for m in mm]; tt = collect(tt)
        c = hcat(ones(length(tt)), tt) \ log.(jx)
        lam = d == 3 ? K2K3(M / T) : lam1(T, d)
        rinst = [(E1 = sqrt.(M^2 .+ vec(sum(abs2, m; dims = 1))); mean((M ./ E1) .* m[1, :]) / mean(m[1, :])) for m in mm]
        println("    $(d) rows: instantaneous rate/η_D over the window ", join(fmt.(rinst[1:2:end]; d = 4), " → "),
                "   fitted $(fmt(-c[2]/eta_drag(T)))   λ₁ = $(fmt(lam))")
        gate!(isapprox(rinst[1], lam; rtol = 0.01), "(c) $(d) rows: the ensemble realises λ₁ at t=0 ($(fmt(rinst[1])) vs $(fmt(lam)))")
        gate!(all(rinst .< lam * 1.01) && rinst[end] > lam * 0.97, "(c) $(d) rows: instantaneous rate stays within [−3 %, +1 %] of λ₁ over ηt ≤ 0.8")
        gate!(isapprox(-c[2], lam * eta_drag(T); rtol = 0.06), "(c) $(d) rows: fitted rate/η_D = $(fmt(-c[2]/eta_drag(T))) within 6 % of λ₁ (noise floor)")
    end
end

println("── (d) Δt bias of the momentum propagator ──")
let T = 0.30, N = 150_000
    η = eta_drag(T); tf = 8 / η
    for rel in (false, true)
        bias = Float64[]
        for dt in (0.04, 0.02, 0.01)
            rng = MersenneTwister(31)
            p0 = rel ? juttner_sample(rng, M, T, 2, N) : sqrt(M * T) .* randn(rng, 2, N)
            _, mm, _ = box_run(CPUBackend(), T; N, dt, tfinal = tf, save = tf / 2, x0 = zeros(2, N), p0, seed = 33, relativistic = rel)
            ref = rel ? jmean(p -> p^2, M, T, 2) : 2 * M * T
            push!(bias, mean(sum(abs2, mm[end]; dims = 1)) / ref - 1)
        end
        sem = sqrt(2 / N)                                          # relative SEM of ⟨p²⟩ (Gaussian-ish)
        println("    relativistic=$rel  ⟨p²⟩/ref − 1 at ηΔt = ", join(fmt.(η .* (0.04, 0.02, 0.01); d = 3), ", "), " : ",
                join(fmt.(bias; d = 4), ", "), "  (SEM $(fmt(sem; d=4)))")
        if rel
            # the pre-point drag η_D·M/E is evaluated at the start of the step ⇒ an O(ηΔt) bias on ⟨p²⟩.
            # Measured ≈ −1 % at ηΔt = 0.1 (N = 1.5·10⁵ resolves 0.4 %); at production ηΔt ≈ 3·10⁻³ it is
            # below anything the ensembles can see. This is a budget gate, not a scaling measurement.
            gate!(all(abs.(bias) .< 0.015), "(d) relativistic pre-point drag: |⟨p²⟩ bias| < 1.5 % for ηΔt ≤ $(fmt(η*0.04; d=2))")
        else
            gate!(all(abs.(bias) .< 3sem), "(d) Galilean exact-OU: no Δt bias at any step (all within 3 SEM)")
        end
    end
end

println("── (e) Galilean twin: MSD(t) exact at all t ──")
let T = 0.30, N = 150_000
    η = eta_drag(T); tf = 6 / η; dt = 0.005 / η
    p0 = sqrt(M * T) .* randn(MersenneTwister(41), 2, N)          # Maxwell start ⇒ the OU covariance is exact
    tt, _, xx = box_run(CPUBackend(), T; N, dt, tfinal = tf, save = tf / 12, x0 = 100.0 .* ones(2, N), p0, seed = 43,
                        relativistic = false)
    tt = collect(tt); m = msd(xx)
    # per spatial dim ⟨Δx²⟩ = 2(T/M)/η² · [ηt − 1 + e^{−ηt}]  (⟨v²⟩ = T/M in c = 1, η in fm⁻¹); two dims
    exact = [2 * 2 * (T / M) / η^2 * (η * t - 1 + exp(-η * t)) for t in tt]
    worst = maximum(abs.(m[2:end] ./ exact[2:end] .- 1))
    gate!(worst < 0.015, "(e) MSD(t)/exact − 1 over 12 samples, worst $(fmt(worst))  (ballistic→diffusive, ηt ∈ [0.5, 6])")
end

println("── (f) DsT_quad makes the drag T-independent ──")
let N = 300_000, Tref = 0.30
    rates = Float64[]
    η_ref = eta_drag(Tref)
    for T in (0.20, 0.40)
        p0 = sqrt(M * T) .* randn(MersenneTwister(51), 2, N); p0[1, :] .+= 0.8      # Maxwell + a 0.8 GeV kick
        tf = 1.0 / η_ref                                                              # one realised drag time
        tt, mm, _ = box_run(CPUBackend(), T; N, dt = 1e-3, tfinal = tf, save = tf / 6, x0 = zeros(2, N), p0, seed = 53,
                            relativistic = false, DsT_quad = true, DsT_Tref = Tref)
        jx = [mean(m[1, :]) for m in mm]; tt = collect(tt)
        push!(rates, -(hcat(ones(length(tt)), tt) \ log.(jx))[2])
    end
    println("    realised η_D at T=0.2, 0.4: ", join(fmt.(rates; d = 4), ", "), " fm⁻¹  (η_D(Tref=0.3) = $(fmt(η_ref; d=4)); const-D_sT would give ratio 4)")
    gate!(isapprox(rates[1], η_ref; rtol = 0.03) && isapprox(rates[2], η_ref; rtol = 0.03), "(f) DsT_quad: η_D(0.2) = η_D(0.4) = η_D(Tref) within 3 %")
end

finish!("bench_physics_gates")
