#!/usr/bin/env julia
# ==============================================================================================
# 01 — THERMALISATION IN A UNIFORM BATH
#
# The smallest complete setup: a box at fixed T with no flow, charm quarks started far from
# equilibrium, and the one question the engine has to get right — does the ensemble relax to the
# Jüttner distribution at the rate the Einstein relation prescribes?
#
# Everything here has a closed form, which is why this is the first example: if a change to the
# engine breaks something, it breaks HERE first and visibly.
#
#   julia --project=Julia Julia/LangevInMedium.jl/examples/01_uniform_bath.jl
#   LIM_NOPLOT=1 ...     numbers only
# ==============================================================================================
include(joinpath(@__DIR__, "example_common.jl"))
using QuadGK

const M   = 1.5        # charm mass [GeV]
const T   = 0.30       # bath temperature [GeV]
const DST = 0.11634    # the D_s·T label — see below for what it means dynamically
const N   = 100_000

# ── what D_sT actually sets ────────────────────────────────────────────────────────────────────
# The engine takes ONE transport coefficient. Everything else follows from the Einstein relation:
#     τ_drag = m·D_sT/T²   [fm]        η_D = 1/τ_drag          κ = 2mT·η_D
# and the diffusion-current relaxation time τ_n = τ_drag·K₃/K₂ is then a DERIVED quantity, not a
# second input. Do not build the drag from `tau_n_main3` — that applies K₃/K₂ once too often and
# inflates the realised D_s by 1.26–1.74× (it was the state of every product before 2026-08-02).
τ_drag = tau_drag(T, M, DST)
η_D    = 1 / τ_drag
τ_n    = tau_n_main3(T, M, DST)
D_s    = (DST / T) * HBARC                       # [fm], the spatial diffusion coefficient
@printf("bath: T = %.2f GeV, m = %.1f GeV, z = m/T = %.2f, D_sT = %.5f\n", T, M, M / T, DST)
@printf("  τ_drag = %.4f fm   η_D = %.4f fm⁻¹   τ_n = τ_drag·K₃/K₂ = %.4f fm   D_s = %.5f fm\n\n",
        τ_drag, η_D, τ_n, D_s)

xg, tg, Tf, Vf = uniform_bath(T; τf = 26 * τ_drag)   # the table must outlast the run

# ── the initial condition ──────────────────────────────────────────────────────────────────────
# `x_init`/`p_init` inject particles directly and bypass the sampler — the right choice whenever
# you want to control the IC exactly. ⚠ `p_init` is a LOCAL-REST-FRAME momentum: the engine applies
# the t0 lab boost itself. Here the bath does not flow, so the two frames coincide.
# All particles start with |p| = 2.5 GeV in +x: a δ-function far from equilibrium, and a pure ℓ=1
# (current) perturbation, so both the ISOTROPISATION and the CURRENT decay are visible.
x0 = zeros(2, N)
p0 = zeros(2, N); p0[1, :] .= 2.5

tf   = 25 * τ_drag                               # long enough for a clean diffusive window
rg   = collect(0.0:0.5:20.0)                     # unused (x_init bypasses the sampler)
pg   = collect(range(0.0, 10.0; length = 300))
dens = ones(length(pg), length(rg))

Random.seed!(1)                                  # the CPU path is bit-reproducible under this
t, mom, pos = simulate_ensemble_bulk(CPUBackend(), rg, pg, dens, Tf, Vf, (xg, tg);
    x_init      = x0,
    p_init      = p0,
    N_particles = N,
    Δt          = 0.01 * τ_drag,                 # η_DΔt = 0.01 ⇒ the propagator bias is ≈ 0.1 %
    initial_time = 0.0,
    final_time  = tf,
    save_interval = tf / 30,
    m           = M,
    DsT         = DST,
    dimensions  = 2,                             # transverse plane
    momentum_dimensions = 3,                     # ⚠ but THREE momentum rows — see below
    Tfo         = 0.0)

# ── why momentum_dimensions = 3 on a 2-D plane ─────────────────────────────────────────────────
# `dimensions` sets the POSITIONS. If the momenta are 2-D too, the ensemble relaxes to the 2-D
# Jüttner, whose ℓ=1 rate λ₁η_D differs from the 3-D K₂/K₃·η_D by 5–12 % over z = 3.5–10 — and the
# hydro coefficients (τ_n = D_s z K₃/K₂, κ) are matched in the 3-D theory. Three momentum rows on a
# two-dimensional plane is the combination that removes that convention offset.
t = collect(t)
p2  = [mean(sum(abs2, m; dims = 1)) for m in mom]
px  = [mean(view(m, 1, :)) for m in mom]
p2eq = juttner_mean(p -> p^2, M, T, 3)
K2K3 = Bessels.besselkx(2, M / T) / Bessels.besselkx(3, M / T)

# closed forms
#   ⟨p²⟩ relaxes towards the Jüttner value; the current ⟨p_x⟩ decays at exactly λ₁η_D = (K₂/K₃)η_D
sel = px .> 0.10 * px[1]
c   = hcat(ones(sum(sel)), t[sel]) \ log.(px[sel])
rate_fit = -c[2]
# TWO estimators of the same rate, and the difference between them is the point of this block.
#   · the FITTED slope of ln⟨p_x⟩ accumulates the kick noise: its floor is ≈ 2.4 %·√(10⁶/N), so at
#     N = 10⁵ it is ±7.6 % and a 5 % miss means nothing;
#   · the INSTANTANEOUS estimator is the exact moment equation d⟨p_x⟩/dt = −η_D⟨(m/E)p_x⟩, i.e.
#     rate(t) = η_D⟨(m/E)p_x⟩/⟨p_x⟩. It is a ratio of two sums over the SAME particles, so the noise
#     largely cancels — but it equals λ₁ ONLY once δf has relaxed onto the ℓ=1 eigenmode.
# From this δ-function IC it starts at η_D·m/E(p₀) — every particle has the same energy, so the
# weight ⟨m/E⟩ is just m/E(p₀), far from its thermal value — and climbs to λ₁η_D as the ensemble
# spreads. λ₁ is an ASYMPTOTIC eigenvalue, not the instantaneous rate of an arbitrary initial state.
rate_inst(k) = begin
    Ek = sqrt.(M^2 .+ vec(sum(abs2, mom[k]; dims = 1)))
    η_D * mean((M ./ Ek) .* view(mom[k], 1, :)) / mean(view(mom[k], 1, :))
end
floor_pct = 2.4 * sqrt(1e6 / N)

println("── relaxation ──")
@printf("  ⟨p²⟩:  %.4f → %.4f GeV²   (Jüttner ⟨p²⟩ = %.4f, off by %+.2f %%)\n",
        p2[1], p2[end], p2eq, 100 * (p2[end] / p2eq - 1))
println("  ℓ=1 instantaneous rate η_D⟨(m/E)p_x⟩/⟨p_x⟩, approaching λ₁η_D = $(round(K2K3 * η_D; digits = 4)) fm⁻¹:")
for k in (1, 2, 3, 5, 8)
    @printf("      t/τ_drag = %5.2f   rate = %.4f fm⁻¹  (%+6.2f %% of λ₁η_D)\n",
            t[k] / τ_drag, rate_inst(k), 100 * (rate_inst(k) / (K2K3 * η_D) - 1))
end
@printf("  ℓ=1 rate, fitted slope of ln⟨p_x⟩ over the window = %.4f fm⁻¹  (%+.1f %%, noise floor ±%.1f %% at N = %d)\n",
        rate_fit, 100 * (rate_fit / (K2K3 * η_D) - 1), floor_pct, N)

# ── the spatial sector: MSD slope = 2·d·D_s ────────────────────────────────────────────────────
msd = [mean(sum(abs2, x .- pos[1]; dims = 1)) for x in pos]
dif = t .>= 12 * τ_drag                          # ≫ τ_drag: well past the ballistic stage
cd  = hcat(ones(sum(dif)), t[dif]) \ msd[dif]
@printf("  MSD slope / 4 = %.5f fm   vs D_s = D_sT/T·ħc = %.5f fm   (%.2f %%)\n\n",
        cd[2] / 4, D_s, 100 * (cd[2] / 4 / D_s - 1))

if plots_on()
    pmag = vec(sqrt.(sum(abs2, mom[end]; dims = 1)))
    ref  = vec(sqrt.(sum(abs2, juttner_sample(MersenneTwister(7), M, T, 3, N); dims = 1)))
    edges = range(0, 4.5; length = 60)
    c1, h1 = hist(pmag, edges); _, h2 = hist(ref, edges)
    pa = plot(t ./ τ_drag, p2; m = :circle, c = :steelblue, xlabel = "t / τ_drag",
              ylabel = "⟨p²⟩ [GeV²]", label = "engine", title = "isotropisation")
    hline!(pa, [p2eq]; ls = :dash, c = :black, label = "Jüttner ⟨p²⟩")
    pb = plot(t ./ τ_drag, max.(px, 1e-4); m = :circle, c = :firebrick, yscale = :log10,
              xlabel = "t / τ_drag", ylabel = "⟨p_x⟩ [GeV]", label = "engine",
              title = "ℓ=1 current decay")
    plot!(pb, t ./ τ_drag, px[1] .* exp.(-K2K3 * η_D .* t); ls = :dash, c = :black,
          label = "exp(−(K₂/K₃)η_D t)")
    pc = plot(c1, max.(h1, 1e-4); m = :circle, c = :steelblue, yscale = :log10,
              xlabel = "|p| [GeV]", ylabel = "density", label = "engine, final",
              title = "equilibrium shape", ylims = (1e-3, 3))
    plot!(pc, c1, max.(h2, 1e-4); ls = :dash, c = :black, label = "exact Jüttner")
    pd = plot(t, msd; m = :circle, c = :seagreen, xlabel = "t [fm]", ylabel = "⟨Δx²⟩ [fm²]",
              label = "engine", title = "ballistic → diffusive")
    plot!(pd, t[dif], cd[1] .+ cd[2] .* t[dif]; ls = :dash, c = :black,
          label = @sprintf("fit slope/4 = %.4f fm (D_s = %.4f)", cd[2] / 4, D_s))
    savefig_ex(plot(pa, pb, pc, pd; layout = (2, 2), size = (1150, 820)), "01_uniform_bath.png")
end
