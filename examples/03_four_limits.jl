#!/usr/bin/env julia
# ==============================================================================================
# 03 — THE FOUR LIMITS, ON ONE BACKGROUND
#
# The engine can be asked for four qualitatively different things, and three of them were confused
# with each other in this repository until 2026-09-02. Run them side by side and the distinctions
# stop being abstract:
#
#   :langevin, D_sT > 0     the physics — drag + noise, relaxing towards the boosted Jüttner
#   :rta,      D_sT > 0     BGK: the same τ_n, a different collision operator
#   D_sT = 0                the COMOVING limit — every quark handed p = m·γ(r)·v(r) EXACTLY, with
#                           no thermal spread at all. A cold, perfect blast wave.
#   :none                   FREE STREAMING — no drag, no noise, momenta exactly constant.
#
# ⚠ THE TRAP THIS EXAMPLE EXISTS FOR. `D_sT = 0` is NOT free streaming; it is the opposite limit.
# Nor is it the `D_sT → 0⁺` limit, which thermalises WITH the fluid and keeps the Jüttner width —
# a ~20 % difference in ⟨p_x⟩ on the flow plateau below. Before 0.2.3 the only thing that actually
# free-streamed was a NEGATIVE D_sT, by accident, and that is now refused. Ask for free streaming
# with `collision_mode = :none`.
#
#   julia --project=Julia Julia/LangevInMedium.jl/examples/03_four_limits.jl
# ==============================================================================================
include(joinpath(@__DIR__, "example_common.jl"))

const M   = 1.5
const DST = 0.11634
const N   = 100_000
const τ0, τf = 0.4, 6.0
const V    = 0.5                      # flow plateau, so the closed forms are one-liners

# a bath at fixed T with a radial flow that saturates at V beyond r = 6 fm
xg = collect(0.0:0.25:20.0); tg = collect(τ0:0.1:(τf + 0.5))
Tf = fill(0.30, length(xg), length(tg))
Vf = [V * min(1.0, r / 6.0) for r in xg, _ in tg]

# every particle starts on the plateau at r = 8 fm with 1 GeV of LRF momentum along +x
x0 = zeros(2, N); x0[1, :] .= 8.0
p0 = zeros(2, N); p0[1, :] .= 1.0
rg = collect(0.0:0.5:20.0); pg = collect(range(0.0, 10.0; length = 300))
dens = ones(length(pg), length(rg))

run(; kw...) = (Random.seed!(3); simulate_ensemble_bulk(CPUBackend(), rg, pg, dens, Tf, Vf, (xg, tg);
    x_init = x0, p_init = p0, N_particles = N, Δt = 1e-3, initial_time = τ0, final_time = τf,
    save_interval = (τf - τ0) / 20, m = M, dimensions = 2, Tfo = 0.0, kw...))

# ⚠ "D_sT → 0⁺" is taken at 10⁻³, not 10⁻⁹, and the reason is worth knowing. τ_drag = m·D_sT/T²,
# so at 10⁻⁹ the attenuation exponent η_eff·Δt is ≈ 10⁴: the exact-OU step then has a = e^{−η_effΔt}
# numerically 0 and degenerates into a Gaussian REDRAW with variance T·E* at the PREVIOUS step's
# energy. That fixed point is close to the Jüttner but is not it (≈ 2 % in ⟨E*⟩ here), so the closed
# form below would miss for a reason that has nothing to do with the limit being taken. At 10⁻³ the
# drag time is 3·10⁻³ fm, η_eff·Δt ≈ 0.3 with Δt = 10⁻³, and the step is resolved.
cases = [
    ("Langevin  (D_sT = $DST)",     (; DsT = DST)),
    ("RTA/BGK   (D_sT = $DST)",     (; DsT = DST, collision_mode = :rta)),
    ("comoving  (D_sT = 0)",        (; DsT = 0.0)),
    ("D_sT → 0⁺ (D_sT = 1e-3)",     (; DsT = 1e-3)),
    ("free streaming (:none)",      (; DsT = DST, collision_mode = :none)),
]

# the closed forms on the plateau
γ    = 1 / sqrt(1 - V^2)
p_comov = M * γ * V                                    # cold comoving: p = m·γ·v exactly
p_free  = γ * (1.0 + V * sqrt(1 + M^2))                # the t0 lab boost, never touched again
Ttot = 0.30
# ⚠ d = 2: these runs carry TWO momentum rows (the default when `momentum_dimensions` is not
# given), so the equilibrium they relax to is the 2-D Jüttner and ⟨E*⟩ must be its mean energy.
# Using the 3-D value here would over-predict by 13 % — the same 2-D/3-D convention offset that
# `momentum_dimensions = 3` exists to remove (see example 01).
p_thermal_comov = γ * V * juttner_mean(p -> sqrt(p^2 + M^2), M, Ttot, 2)   # γv⟨E*⟩

println("closed forms on the v = $V plateau:")
@printf("  cold comoving  m·γ·v            = %.5f GeV\n", p_comov)
@printf("  thermal comoving γ·v·⟨E*⟩       = %.5f GeV   (+%.1f %% over the cold one)\n",
        p_thermal_comov, 100 * (p_thermal_comov / p_comov - 1))
@printf("  free streaming (boosted IC)     = %.5f GeV\n\n", p_free)

results = Any[]
@printf("%-26s  %10s  %10s  %10s  %12s\n", "case", "⟨p_x⟩", "⟨|p|⟩", "sd(p_x)", "⟨v_x⟩ meas")
for (label, kw) in cases
    t, mom, pos = run(; kw...)
    t = collect(t)
    pxs = view(mom[end], 1, :)
    vx  = mean(pos[end][1, :] .- pos[end-1][1, :]) / (t[end] - t[end-1])
    @printf("%-26s  %10.5f  %10.5f  %10.5f  %12.5f\n",
            label, mean(pxs), mean(sqrt.(sum(abs2, mom[end]; dims = 1))), std(pxs), vx)
    push!(results, (label, t, [mean(view(m, 1, :)) for m in mom], mom[end]))
end

println("""
read the table:
  · comoving sits exactly on m·γ·v with sd(p_x) = 0 — a COLD blast wave, no thermal width;
  · D_sT → 0⁺ sits on γ·v·⟨E*⟩ instead — the number printed above the table — and HAS the thermal
    width. Same "limit",
    different answer, because the width does not vanish as the drag time does;
  · free streaming keeps the boosted initial momentum forever and moves FASTER than the fluid;
  · Langevin and RTA both relax towards the boosted Jüttner and differ only in how — same τ_n,
    different collision operator, which is the comparison the papers are about.""")

if plots_on()
    pa = plot(; xlabel = "τ [fm]", ylabel = "⟨p_x⟩ [GeV]", title = "the four limits")
    cols = [:steelblue, :seagreen, :black, :darkorange, :firebrick]
    for (k, (label, t, px, _)) in enumerate(results)
        plot!(pa, t, px; m = :circle, c = cols[k], label = label)
    end
    hline!(pa, [p_comov]; ls = :dash, c = :gray, label = "m·γ·v")
    hline!(pa, [p_free];  ls = :dot,  c = :gray, label = "boosted IC")
    edges = range(0.0, 4.0; length = 45)
    pb = plot(; xlabel = "|p| [GeV]", ylabel = "density", yscale = :log10,
              title = "final |p| distributions", ylims = (1e-3, 30))
    for (k, (label, _, _, mf)) in enumerate(results)
        c, h = hist(vec(sqrt.(sum(abs2, mf; dims = 1))), edges)
        plot!(pb, c, max.(h, 1e-4); c = cols[k], label = label)
    end
    savefig_ex(plot(pa, pb; layout = (1, 2), size = (1200, 460)), "03_four_limits.png")
end
