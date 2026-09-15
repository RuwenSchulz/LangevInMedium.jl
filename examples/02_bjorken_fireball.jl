#!/usr/bin/env julia
# ==============================================================================================
# 02 — A HEAVY-ION RUN: cooling fireball, FONLL initial condition, freeze-out spectrum
#
# The production shape of a real run, with a synthetic background so it needs no hydro output. What
# it demonstrates:
#   · handing the engine a tabulated `T(r, τ)`, `v(r, τ)` and an initial phase-space density,
#     and letting IT sample the particles (the path every driver in the tree takes);
#   · the temperature-dependent D_sT prescriptions;
#   · reading a freeze-out spectrum off the histories, and what "freeze-out" costs you if you do
#     it from snapshots rather than on the fly (example 05 shows the GPU's on-the-fly latch);
#   · the RADIAL FLOW picking the spectrum up — the physics the whole apparatus exists for.
#
#   julia --project=Julia Julia/LangevInMedium.jl/examples/02_bjorken_fireball.jl
# ==============================================================================================
include(joinpath(@__DIR__, "example_common.jl"))

const M   = 1.5
const DST = 0.11634
const TFO = 0.156          # freeze-out temperature [GeV]
const N   = 200_000
const τ0, τf = 0.4, 15.0

xg, tg, Tf, Vf = bjorken_fireball(; τ0 = τ0, τf = τf)
@printf("background: r ∈ [0, %.0f] fm (%d nodes), τ ∈ [%.1f, %.1f] fm (%d nodes)\n",
        last(xg), length(xg), τ0, τf, length(tg))
@printf("            T(0, τ₀) = %.3f GeV → T(0, τ_f) = %.3f GeV,  v_max = %.2f c\n",
        Tf[1, 1], Tf[1, end], maximum(Vf))

# ── the initial condition, on grids the sampler can integrate ──────────────────────────────────
# The engine samples `N_particles` from `density[p_index, r_index]` on `(r_grid, p_grid)`:
# positions by inverse CDF of `P(r) ∝ r·n(r)` (or by rejection in the disc, see below), momenta
# from the conditional `P(p|r) ∝ p·f(r, p)` at the nearest tabulated r, with isotropic azimuth.
# ⚠ Both quadratures are trapezoids on the nodes YOU hand it (0.2.3). A uniform grid is what every
# driver builds and what the resolution below assumes: np = 300 over [0, 10] costs ≈ 0.1 % on the
# sampled ⟨p_T⟩; np = 100 would cost ≈ 3 %.
r_grid = collect(range(0.0, 20.0; length = 150))
p_grid = collect(range(0.0, 10.0; length = 300))
density = fonll_density(r_grid, p_grid; σ = 3.0)

# `cartesian_spatial_sampling = true` rejection-samples (x, y) in the disc against n(r) — no r → 0
# grid artefact, but the acceptance is the fireball's area fraction of the disc (a few % for a
# narrow profile in a 20 fm disc), and it is host-side and serial. `false` is the polar inverse CDF.
CART = false

# ── the run ────────────────────────────────────────────────────────────────────────────────────
# `DsT_linear` makes D_sT(T) = slope·max(T, Tfo) + offset — the lattice-QCD-shaped law. The
# alternative prescriptions are a constant D_sT (the default) and `DsT_quad` (D_sT ∝ T², i.e. a
# T-INDEPENDENT drag time, the one member of the family for which the non-relativistic solvable
# class closes on an inhomogeneous T(r)).
Random.seed!(2)
t, mom, pos = simulate_ensemble_bulk(CPUBackend(), r_grid, p_grid, density, Tf, Vf, (xg, tg);
    N_particles   = N,
    Δt            = 5e-3,          # η_DΔt ≈ 0.01 at T = 0.3; see example 01 for what that costs
    initial_time  = τ0,
    final_time    = τf,
    save_interval = 0.4,
    m             = M,
    DsT           = DST,
    DsT_linear    = true, DsT_slope = 1.765, DsT_offset = -0.159, Tfo = TFO,
    dimensions    = 2,
    momentum_dimensions = 3,       # 3-D momenta on the 2-D plane (example 01 explains why)
    bjorken_redshift = true,       # dp_z/dτ = −p_z/τ: the longitudinal work a 2-D run omits
    proper_time_kicks = true,      # kick per the PARTICLE's proper time, not the lab step
    cartesian_spatial_sampling = CART,
    reflecting_boundary = false)
t = collect(t)

# ── why proper_time_kicks matters on a FLOWING background ──────────────────────────────────────
# With the undilated lab-Δt kick the stationary state on a flowing background is f_J/(γ(1+v·v_r))
# rather than the boosted Jüttner, which shows up as a spurious inward diffusion current
# ν^r/n ≈ −γv⟨v_r²⟩. `proper_time_kicks = true` runs the OU attenuation over Δt* = Δt·E*/E_lab.
# It defaults to FALSE so that pre-2026-08 products stay bit-identical; new work should set it.

# ── freeze-out from the snapshots ──────────────────────────────────────────────────────────────
# The honest version: a particle freezes out the first time its local T drops below Tfo. Reading
# that off snapshots resolves it to the SAVE cadence (0.4 fm here), not to Δt. Example 05 uses the
# GPU's on-the-fly latch, which resolves it to Δt and stores only N-length arrays.
Tof(r, τ) = LangevInMedium.KernelsCPU.interpolate_2d_cpu(xg, tg, Tf, r, τ)
frozen = falses(N); pT_fo = zeros(N); τ_fo = zeros(N); r_fo = zeros(N)
for k in eachindex(t), i in 1:N
    frozen[i] && continue
    r = hypot(pos[k][1, i], pos[k][2, i])
    if Tof(r, t[k]) <= TFO
        frozen[i] = true
        pT_fo[i]  = hypot(mom[k][1, i], mom[k][2, i])
        τ_fo[i]   = t[k]; r_fo[i] = r
    end
end
nf = count(frozen)
pT_init = vec(sqrt.(sum(abs2, view(mom[1], 1:2, :); dims = 1)))

@printf("\nfroze out: %d / %d (%.1f %%) by τ = %.1f fm;  ⟨τ_fo⟩ = %.2f fm, ⟨r_fo⟩ = %.2f fm\n",
        nf, N, 100nf / N, τf, mean(τ_fo[frozen]), mean(r_fo[frozen]))
@printf("⟨p_T⟩:  initial %.4f GeV  →  freeze-out %.4f GeV   (%+.1f %%, the radial flow)\n",
        mean(pT_init), mean(pT_fo[frozen]), 100 * (mean(pT_fo[frozen]) / mean(pT_init) - 1))
@printf("⟨p_T²⟩: initial %.4f      →  freeze-out %.4f\n",
        mean(pT_init .^ 2), mean(pT_fo[frozen] .^ 2))

# a nuclear-modification-shaped ratio: freeze-out over initial, same particles
edges = range(0.0, 6.0; length = 31)
c, h_i = hist(pT_init, edges); _, h_f = hist(pT_fo[frozen], edges)
println("\n  p_T [GeV]   spectrum ratio (freeze-out / initial)")
for k in 1:3:length(c)
    h_i[k] > 0 && @printf("   %5.2f       %.3f\n", c[k], h_f[k] / h_i[k])
end

if plots_on()
    pa = plot(c, max.(h_i, 1e-6); m = :circle, c = :gray, yscale = :log10, xlabel = "p_T [GeV]",
              ylabel = "1/N dN/dp_T", label = "initial (FONLL-like)", title = "charm p_T spectrum",
              ylims = (1e-5, 3))
    plot!(pa, c, max.(h_f, 1e-6); m = :square, c = :firebrick, label = "at freeze-out")
    pb = plot(c, h_f ./ max.(h_i, 1e-12); m = :circle, c = :steelblue, xlabel = "p_T [GeV]",
              ylabel = "ratio", label = "", title = "freeze-out / initial")
    hline!(pb, [1.0]; ls = :dash, c = :black, label = "")
    pc = histogram(τ_fo[frozen]; bins = 30, c = :seagreen, xlabel = "τ_fo [fm]",
                   ylabel = "particles", label = "", title = "freeze-out time")
    pd = plot(t, [mean(sqrt.(sum(abs2, x; dims = 1))) for x in pos]; m = :circle, c = :darkorange,
              xlabel = "τ [fm]", ylabel = "⟨r⟩ [fm]", label = "", title = "radial expansion")
    savefig_ex(plot(pa, pb, pc, pd; layout = (2, 2), size = (1150, 820)), "02_bjorken_fireball.png")
end
