#!/usr/bin/env julia
# ==============================================================================================
# bench_semianalytic.jl — the engine against CLOSED FORMS, with plots.
#
# WHERE THIS SITS. `Projects/SemiAnalyticBenchmarks/bench_langevin_ou.jl` (B10) already pins the
# Galilean ⟨p²⟩ relaxation, the 1/√N Monte-Carlo rate and the momentum autocorrelation. It lives in
# the repo, drives the FP solvers alongside the engine, and tests the MOMENTUM sector only. This
# one lives in the PACKAGE (so it travels with the engine and needs nothing but the monorepo
# environment) and takes the five closed forms B10 does not:
#
#   (S1) the full Uhlenbeck–Ornstein PHASE-SPACE covariance — σ_pp, σ_xx and, the sharp one,
#        the position–momentum cross-correlation σ_xp = (T/η)(1 − e^{−ηt}). Nothing in the tree
#        tests σ_xp, and it is exactly where an operator-split error between the momentum update
#        and the position update would show up. Across the ballistic→diffusive crossover.
#   (S2) the EXACT BGK moment law. In a uniform bath a particle survives to t with probability
#        e^{−t/τ_n} and is otherwise equilibrium-distributed, so for ANY observable g
#             ⟨g⟩(t) = e^{−t/τ_n}⟨g⟩₀ + (1 − e^{−t/τ_n})⟨g⟩_eq          (exact, all t)
#        — the whole relaxation curve, not just its rate. Checked at four Δt spanning 100×, which
#        is a gate on the 0.2.3 `−expm1` fix: with the old linearised Δt/τ_n the largest step
#        relaxed 22 % too fast and this curve was simply wrong.
#   (S3) free streaming WITH the Bjorken redshift, which has a closed form worth having:
#        p_z(τ) = p_z(τ₀)τ₀/τ exactly, and since E depends on the redshifting p_z,
#             x_⊥(τ) = x_⊥(τ₀) + (p_⊥/m_T²)·[√(m_T²τ² + a²) − √(m_T²τ₀² + a²)],  a = p_z(τ₀)τ₀
#        The engine is first order here by construction (the redshift reads the START of the step,
#        the position update the END), so this block measures the ORDER, not exactness.
#   (S4) the relativistic equilibrium SHAPE, not its moments: the |p| histogram against the exact
#        Jüttner p^{d−1}e^{−(E−m)/T} with a two-sample KS statistic, in 2 and 3 momentum rows.
#   (S5) the comoving blast wave (`DsT = 0`), which is exact PER PARTICLE: every quark carries
#        p = m·γ(r)·v(r) at its own radius, so the predicted momentum can be checked against the
#        measured one particle by particle rather than through an ensemble moment.
#
# Figures: bench/results/figures/semianalytic_*.png (one panel per block, plus a summary sheet).
#
#   julia --project=Julia Julia/LangevInMedium.jl/bench/bench_semianalytic.jl
#   LIM_BENCH_QUICK=1 ...       smaller ensembles (≈ ⅓ the wall time)
#   LIM_NOPLOT=1 ...            numbers only, no Plots dependency
#
# Accuracy, not speed: every target is a closed form, so machine load cannot move any number here.
# ==============================================================================================
include(joinpath(@__DIR__, "bench_common.jl"))
using QuadGK, Bessels, Printf

const QUICK  = get(ENV, "LIM_BENCH_QUICK", "0") == "1"
const NOPLOT = get(ENV, "LIM_NOPLOT", "0") == "1"
const FIGDIR = joinpath(RESULTS, "figures")
NOPLOT || mkpath(FIGDIR)

if !NOPLOT
    ENV["GKSwstype"] = get(ENV, "GKSwstype", "100")
    using Plots
    gr()
    default(; fontfamily = "sans-serif", framestyle = :box, grid = true, legend = :best,
            dpi = 150, lw = 2, ms = 4, size = (560, 420),
        left_margin = 6Plots.mm, bottom_margin = 6Plots.mm, top_margin = 3Plots.mm)
end
const PANELS = Any[]
stash!(p) = (NOPLOT || push!(PANELS, p); p)

const M, DST = 1.5, 0.11634
const TBATH  = 0.30
eta_drag(T) = (T^2 / (M * DST)) / HBARC        # fm⁻¹
const ETA  = eta_drag(TBATH)
const TAUD = 1 / ETA

println("="^94)
println("LangevInMedium — semi-analytic benchmarks   (m = $M GeV, T = $TBATH GeV, D_sT = $DST)")
println("η_D = $(fmt(ETA; d=4)) fm⁻¹, τ_drag = $(fmt(TAUD; d=4)) fm" * (QUICK ? "   [QUICK]" : ""))
println("="^94)

# ══════════════════════════════════════════════════════════════════════════════════════════════
# (S1) Uhlenbeck–Ornstein phase-space covariance — σ_pp, σ_xx, σ_xp
# ══════════════════════════════════════════════════════════════════════════════════════════════
# The Galilean branch (`relativistic = false`) IS the classic OU process: dp = −ηp dt + √κ dW with
# κ = 2mTη, and dx = p/m dt. From a THERMAL initial momentum at x = 0 the exact per-component
# moments are (Chandrasekhar 1943, Rev. Mod. Phys. 15, 1 §II):
#     σ_pp(t) = mT                                            (stationary from the start)
#     σ_xx(t) = 2(T/m)/η² · (ηt − 1 + e^{−ηt})                 ballistic t², diffusive 2D_s t
#     σ_xp(t) = (T/η)(1 − e^{−ηt})                             the cross-correlation
# σ_xp is the one to watch: it is built from the momentum update and the position update TOGETHER,
# so an operator-split or an off-by-one-step error shows there first, while σ_pp and σ_xx can each
# look right on their own.
println("\n── (S1) Uhlenbeck–Ornstein phase-space covariance (Galilean, exact) ──")
let N = QUICK ? 100_000 : 400_000
    dt = 0.004 * TAUD
    tf = 1440 * dt                      # 24 saves × 60 steps, so save_interval divides exactly
    p0 = sqrt(M * TBATH) .* randn(MersenneTwister(11), 2, N)       # thermal (Maxwell) IC
    x0 = zeros(2, N)
    xg, tg, Tf, Vf = box_fields(TBATH; tf = tf)
    tt, mm, xx = run_fields(CPUBackend(), xg, tg, Tf, Vf; M, DsT = DST, N, dt, tfinal = tf,
        save = tf / 24, x0 = x0, p0 = p0, seed = 12, relativistic = false)
    tt = collect(tt) .- first(tt)
    # per-component ensemble covariances, averaged over the two Cartesian directions
    spp = [mean(vec(mm[k] .^ 2)) for k in eachindex(tt)]
    sxx = [mean(vec(xx[k] .^ 2)) for k in eachindex(tt)]
    sxp = [mean(vec(xx[k] .* mm[k])) for k in eachindex(tt)]
    e_pp = fill(M * TBATH, length(tt))
    e_xx = [2 * (TBATH / M) / ETA^2 * (ETA * t - 1 + exp(-ETA * t)) for t in tt]
    e_xp = [(TBATH / ETA) * (1 - exp(-ETA * t)) for t in tt]
    rel(a, b) = maximum(abs.(a[2:end] ./ b[2:end] .- 1))
    sem = sqrt(2 / (2N))
    @printf("  σ_pp: worst |meas/exact − 1| = %.4f  (stationary mT = %.5f GeV²; SEM %.4f)\n",
            rel(spp, e_pp), M * TBATH, sem)
    @printf("  σ_xx: worst |meas/exact − 1| = %.4f  over ηt ∈ [%.2f, %.1f] (ballistic → diffusive)\n",
            rel(sxx, e_xx), ETA * tt[2], ETA * tt[end])
    @printf("  σ_xp: worst |meas/exact − 1| = %.4f  ← the cross-correlation nothing else tests\n",
            rel(sxp, e_xp))
    gate!(rel(spp, e_pp) < 0.02, "(S1) σ_pp = mT at every t")
    gate!(rel(sxx, e_xx) < 0.02, "(S1) σ_xx = 2(T/m)/η²·(ηt − 1 + e^{−ηt}) across the crossover")
    gate!(rel(sxp, e_xp) < 0.03, "(S1) σ_xp = (T/η)(1 − e^{−ηt})")
    if !NOPLOT
        ηt = ETA .* tt
        p = plot(ηt, sxx ./ e_xx[end]; m = :circle, c = :steelblue, label = "σ_xx  measured",
                 xlabel = "η_D t", ylabel = "covariance / its own late value",
                 title = "(S1) OU phase space: exact vs engine")
        plot!(p, ηt, e_xx ./ e_xx[end]; ls = :dash, c = :black, label = "σ_xx  exact")
        plot!(p, ηt, sxp ./ e_xp[end]; m = :square, c = :firebrick, label = "σ_xp  measured")
        plot!(p, ηt, e_xp ./ e_xp[end]; ls = :dashdot, c = :black, label = "σ_xp  exact")
        plot!(p, ηt, spp ./ e_pp[end]; m = :diamond, c = :seagreen, label = "σ_pp  measured")
        hline!(p, [1.0]; ls = :dot, c = :gray, label = "σ_pp  exact (= mT)")
        savefig(stash!(p), joinpath(FIGDIR, "semianalytic_S1_ou_covariance.png"))
    end
end

# ══════════════════════════════════════════════════════════════════════════════════════════════
# (S2) the exact BGK moment law, at four step sizes spanning 100×
# ══════════════════════════════════════════════════════════════════════════════════════════════
# A BGK/RTA particle in a uniform bath either has not collided (probability e^{−t/τ_n}, so it still
# carries its initial momentum) or has, in which case it is Jüttner-distributed — and stays so
# under further collisions. Hence for ANY observable g, exactly and at all t,
#       ⟨g⟩(t) = e^{−t/τ_n}⟨g⟩₀ + (1 − e^{−t/τ_n})⟨g⟩_eq.
# This is a much stronger statement than "the ℓ=1 mode decays at 1/τ_n": it fixes the entire curve.
# It also only holds if the PER-STEP collision probability is exact — with the linearised Δt/τ_n
# the engine carried before 0.2.3, the Δt = 0.2 curve relaxed 22 % too fast.
println("\n── (S2) exact BGK moment law ⟨g⟩(t) = e^{−t/τ_n}⟨g⟩₀ + (1−e^{−t/τ_n})⟨g⟩_eq ──")
let N = QUICK ? 100_000 : 300_000
    τn   = tau_n_main3(TBATH, M, DST)
    p2eq = jmean(p -> p^2, M, TBATH, 2)
    tf   = 3.0
    x0 = zeros(2, N); x0[1, :] .= 50.0
    p0 = zeros(2, N); p0[1, :] .= 2.5                     # far from equilibrium, and ℓ=1 loaded
    p2_0 = 2.5^2
    curves = Any[]
    worst = 0.0
    println("    Δt      Δt/τ_n    worst |⟨p²⟩ meas/exact − 1|   worst |⟨p_x⟩ meas/exact − 1|")
    for dt in (QUICK ? (0.002, 0.05, 0.2) : (0.002, 0.02, 0.1, 0.2))
        xg, tg, Tf, Vf = box_fields(TBATH; tf = tf + 0.5)
        # the save cadence must be a multiple of Δt AND at least Δt — at Δt = 0.2 a tf/30 = 0.1
        # cadence is shorter than one step and the driver (rightly) refuses.
        save = max(round(tf / 30 / dt) * dt, dt)
        tt, mm, _ = run_fields(CPUBackend(), xg, tg, Tf, Vf; M, DsT = DST, N, dt, tfinal = tf,
            save = save, x0 = x0, p0 = p0, seed = 21, collision_mode = :rta)
        tt = collect(tt) .- first(tt)
        surv = exp.(-tt ./ τn)
        m_p2 = [mean(sum(abs2, m; dims = 1)) for m in mm]
        m_px = [mean(view(m, 1, :)) for m in mm]
        e_p2 = surv .* p2_0 .+ (1 .- surv) .* p2eq
        e_px = surv .* 2.5                                # ⟨p_x⟩_eq = 0
        w2 = maximum(abs.(m_p2 ./ e_p2 .- 1))
        wx = maximum(abs.(m_px[surv .> 0.05] ./ e_px[surv .> 0.05] .- 1))
        worst = max(worst, w2)
        @printf("  %7.4f  %8.4f   %24.4f   %27.4f\n", dt, dt / τn, w2, wx)
        push!(curves, (dt, tt, m_p2, e_p2))
    end
    gate!(worst < 0.03, "(S2) the whole ⟨p²⟩(t) curve follows the exact BGK law at every Δt (3 %)")
    if !NOPLOT
        cols = [:steelblue, :firebrick, :seagreen, :darkorange]
        # the exact law drawn ONCE as a wide pale band, the engines as markers on top of it
        tt_ref = range(0, tf; length = 200)
        e_ref  = [exp(-t / τn) * p2_0 + (1 - exp(-t / τn)) * p2eq for t in tt_ref]
        pa = plot(tt_ref, e_ref; lw = 7, c = :gray85,
                  label = "exact  e^{−t/τ_n}⟨p²⟩₀ + (1−e^{−t/τ_n})⟨p²⟩_eq",
                  xlabel = "t [fm]", ylabel = "⟨p²⟩ [GeV²]", title = "(S2) BGK: the exact moment law")
        for (i, (dt, tt, m_p2, _)) in enumerate(curves)
            scatter!(pa, tt, m_p2; m = :circle, ms = 4, c = cols[mod1(i, 4)], label = "engine Δt = $dt")
        end
        hline!(pa, [p2eq]; ls = :dot, c = :black, label = "Jüttner ⟨p²⟩")
        # the residual is where the fix is visible: pre-0.2.3 the Δt = 0.2 curve sat ~20 % low here
        pb = plot(; xlabel = "t [fm]", ylabel = "⟨p²⟩ engine / exact − 1",
                  title = "(S2) residual — no Δt trend left", ylims = (-0.06, 0.06))
        for (i, (dt, tt, m_p2, e_p2)) in enumerate(curves)
            plot!(pb, tt, m_p2 ./ e_p2 .- 1; m = :circle, c = cols[mod1(i, 4)], label = "Δt = $dt")
        end
        hline!(pb, [0.0]; ls = :dash, c = :black, label = "")
        hspan!(pb, [-0.03, 0.03]; c = :gray90, alpha = 0.5, label = "gate band (3 %)")
        savefig(stash!(plot(pa, pb; layout = (1, 2), size = (1150, 430))),
                joinpath(FIGDIR, "semianalytic_S2_bgk_moment_law.png"))
    end
end

# ══════════════════════════════════════════════════════════════════════════════════════════════
# (S3) free streaming with the Bjorken redshift — closed form, and the scheme's order
# ══════════════════════════════════════════════════════════════════════════════════════════════
# `collision_mode = :none` with `bjorken_redshift = true` is integrable: p_⊥ is constant, p_z*
# telescopes to p_z(τ₀)τ₀/τ, and the transverse position follows from dx/dτ = p_⊥/E(τ) with
# E(τ)² = m_T² + (p_z(τ₀)τ₀/τ)²:
#       x(τ) = x(τ₀) + (p_⊥/m_T²)·[√(m_T²τ² + a²) − √(m_T²τ₀² + a²)],   a = p_z(τ₀)·τ₀
# The engine is first order here by construction — the redshift kernel reads the START of the step
# (which is what makes it the exact free-streaming solution over the interval) while the position
# update reads the END — so the right question is the ORDER, not the residual.
println("\n── (S3) free streaming + Bjorken redshift: closed form and convergence order ──")
let N = 20_000, τ0 = 0.4, τf = 6.0, pT = 1.2
    mT = sqrt(M^2 + pT^2)
    xg = collect(0.0:0.25:60.0); tg = collect(τ0:0.1:(τf + 0.5))
    Tf = fill(TBATH, length(xg), length(tg)); Vf = zeros(length(xg), length(tg))
    x0 = zeros(2, N); x0[1, :] .= 8.0
    p0 = zeros(2, N); p0[1, :] .= pT
    rg = collect(0.0:0.5:20.0); pg = collect(0.05:0.1:8.0); dens = ones(length(pg), length(rg))
    go(dt) = (Random.seed!(3); simulate_ensemble_bulk(CPUBackend(), rg, pg, dens, Tf, Vf, (xg, tg);
        x_init = x0, p_init = p0, N_particles = N, Δt = dt, initial_time = τ0, final_time = τf,
        save_interval = (τf - τ0) / 20, m = M, DsT = DST, dimensions = 2, momentum_dimensions = 3,
        pz_init = :thermal, bjorken_redshift = true, collision_mode = :none, Tfo = 0.0))

    dts  = QUICK ? (8e-3, 2e-3) : (8e-3, 4e-3, 2e-3, 1e-3)
    errs = Float64[]
    local tt, xm, xe, pz2, e_pz2
    for dt in dts
        tt_, mm_, xx_ = go(dt)
        tt = collect(tt_)
        # every particle has the same p_⊥ but its OWN p_z(τ₀), drawn by the thermal completion.
        # The single-particle closed form is exact, so the ensemble prediction is its average over
        # the SAMPLED p_z(τ₀) — no distributional assumption enters.
        a_i = view(mm_[1], 3, :) .* τ0
        xe  = [mean(8.0 .+ (pT / mT^2) .* (sqrt.(mT^2 * τ^2 .+ a_i .^ 2) .- sqrt.(mT^2 * τ0^2 .+ a_i .^ 2)))
               for τ in tt]
        xm  = [mean(view(x, 1, :)) for x in xx_]
        push!(errs, maximum(abs.(xm .- xe)))
        pz2   = [mean(view(m, 3, :) .^ 2) for m in mm_]
        e_pz2 = pz2[1] .* (τ0 ./ tt) .^ 2
    end
    ord = log(errs[1] / errs[end]) / log(dts[1] / dts[end])
    @printf("  ⟨x_⊥(τ)⟩ vs the per-particle closed form: max |Δx| = %s fm\n",
            join([fmt(e; d = 7) for e in errs], ", "))
    @printf("      at Δt = %s  ⇒  convergence order %.2f (the scheme is first order by construction)\n",
            join(string.(dts), ", "), ord)
    gate!(0.7 < ord < 1.4, "(S3) x_⊥(τ) converges to the closed form at first order (measured $(fmt(ord; d=2)))")
    wz = maximum(abs.(pz2 ./ e_pz2 .- 1))
    @printf("  ⟨p_z*²⟩(τ) vs the exact (τ₀/τ)²: worst |meas/exact − 1| = %.3e over τ ∈ [%.1f, %.1f]\n",
            wz, τ0, τf)
    gate!(wz < 1e-9, "(S3) ⟨p_z*²⟩ ∝ 1/τ² telescopes exactly under free streaming")
    if !NOPLOT
        pa = plot(tt, pz2 ./ pz2[1]; m = :circle, c = :steelblue, yscale = :log10,
                  xlabel = "τ [fm]", ylabel = "⟨p_z*²⟩(τ) / ⟨p_z*²⟩(τ₀)",
                  label = "engine", title = "(S3) Bjorken redshift, free streaming")
        plot!(pa, tt, (τ0 ./ tt) .^ 2; ls = :dash, c = :black, label = "exact (τ₀/τ)²")
        pb = plot(tt, xm; m = :circle, c = :firebrick, xlabel = "τ [fm]", ylabel = "⟨x_⊥⟩ [fm]",
                  label = "engine", title = "(S3) transverse streaming with a redshifting E")
        plot!(pb, tt, xe; ls = :dash, c = :black, label = "per-particle closed form")
        plot!(pb, tt, 8.0 .+ (pT / mT) .* (tt .- τ0); ls = :dot, c = :gray,
              label = "p_z* = 0 branch (x₀ + p_⊥τ/m_T)")
        savefig(stash!(plot(pa, pb; layout = (1, 2), size = (1100, 420))),
                joinpath(FIGDIR, "semianalytic_S3_free_streaming.png"))
    end
end

# ══════════════════════════════════════════════════════════════════════════════════════════════
# (S4) the equilibrium SHAPE, not its moments
# ══════════════════════════════════════════════════════════════════════════════════════════════
# ⟨p²⟩ agreeing with the Jüttner value does not say the distribution IS Jüttner — the 0.2.1 audit
# added a tail gate for exactly that reason (a 16.5 % tail error hid under a correct ⟨p²⟩). Here the
# whole |p| distribution is compared to P(p) ∝ p^{d−1}e^{−(E−m)/T} by a two-sample KS statistic
# against an independent rejection sample, in 2 and 3 momentum rows.
println("\n── (S4) equilibrium shape: |p| distribution vs the exact Jüttner (KS) ──")
let N = QUICK ? 60_000 : 200_000
    ksres = Any[]
    for d in (2, 3)
        P0 = juttner_sample(MersenneTwister(50 + d), M, TBATH, d, N)
        p0 = d == 3 ? P0[1:2, :] : P0
        tf = 12 * TAUD
        xg, tg, Tf, Vf = box_fields(TBATH; tf = tf)
        _, mm, _ = run_fields(CPUBackend(), xg, tg, Tf, Vf; M, DsT = DST, N, dt = 0.01 * TAUD,
            tfinal = tf, save = tf / 2, x0 = zeros(2, N), p0 = p0, seed = 55 + d,
            momentum_dimensions = d)
        meas = sort(vec(sqrt.(sum(abs2, mm[end]; dims = 1))))
        ref  = sort(vec(sqrt.(sum(abs2, juttner_sample(MersenneTwister(900 + d), M, TBATH, d, N); dims = 1))))
        # two-sample KS: sup |F_meas − F_ref|
        i = j = 1; ks = 0.0
        while i <= length(meas) && j <= length(ref)
            meas[i] <= ref[j] ? (i += 1) : (j += 1)
            ks = max(ks, abs(i / length(meas) - j / length(ref)))
        end
        crit = 1.36 * sqrt(2 / N)                     # 95 % two-sample critical value
        @printf("  %d momentum rows: KS = %.5f   (95 %% critical %.5f at N = %d)  %s\n",
                d, ks, crit, N, ks < crit ? "consistent" : "DIFFERENT")
        gate!(ks < 2 * crit, "(S4) $d rows: |p| distribution is the exact Jüttner (KS $(fmt(ks; d=5)) < $(fmt(2crit; d=5)))")
        push!(ksres, (d, meas, ref, ks))
    end
    if !NOPLOT
        p = plot(; xlabel = "|p| [GeV]", ylabel = "normalised density", yscale = :log10,
                 title = "(S4) equilibrium shape vs exact Jüttner", ylims = (1e-4, 3))
        for (k, (d, meas, ref, ks)) in enumerate(ksres)
            edges = range(0, 4.5; length = 60)
            h(v) = begin
                c = zeros(length(edges) - 1)
                for x in v; b = searchsortedlast(edges, x); 1 <= b <= length(c) && (c[b] += 1); end
                c ./ (sum(c) * step(edges))
            end
            ctr = (edges[1:end-1] .+ edges[2:end]) ./ 2
            plot!(p, ctr, max.(h(meas), 1e-5); m = :circle, c = k == 1 ? :steelblue : :firebrick,
                  label = "engine, $d rows")
            plot!(p, ctr, max.(h(ref), 1e-5); ls = :dash, c = :black,
                  label = k == 1 ? "exact Jüttner p^{d−1}e^{−(E−m)/T}" : "")
        end
        savefig(stash!(p), joinpath(FIGDIR, "semianalytic_S4_juttner_shape.png"))
    end
end

# ══════════════════════════════════════════════════════════════════════════════════════════════
# (S5) the comoving blast wave — exact PER PARTICLE
# ══════════════════════════════════════════════════════════════════════════════════════════════
# `DsT = 0` glues every quark to the flow, so it does not merely reproduce an ensemble average: at
# its own radius each particle must carry exactly p = m·γ(r)·v(r) along r̂. That makes the check
# per-particle, which is far sharper than any moment — and it is the limit three places in the tree
# used to call "free streaming" (it is not; `collision_mode = :none` is, and both are plotted here).
println("\n── (S5) comoving blast wave: p = m·γ(r)·v(r), per particle ──")
let N = QUICK ? 20_000 : 60_000
    vmax = 0.6
    xg = collect(0.0:0.25:20.0); tg = collect(0.4:0.1:4.0)
    Tf = fill(TBATH, length(xg), length(tg))
    vprof(r) = vmax * min(1.0, r / 8.0)
    Vf = [vprof(r) for r in xg, _ in tg]
    φ = 2π .* rand(MersenneTwister(71), N); rr = 2.0 .+ 6.0 .* rand(MersenneTwister(72), N)
    x0 = vcat((rr .* cos.(φ))', (rr .* sin.(φ))')
    p0 = 0.5 .* randn(MersenneTwister(73), 2, N)
    rg = collect(0.0:0.5:20.0); pg = collect(0.05:0.1:8.0)
    go(; Δt = 1e-3, kw...) = (Random.seed!(9); simulate_ensemble_bulk(CPUBackend(), rg, pg,
        ones(length(pg), length(rg)), Tf, Vf, (xg, tg); x_init = x0, p_init = p0,
        N_particles = N, Δt = Δt, initial_time = 0.4, final_time = 3.4, save_interval = 3.0,
        m = M, dimensions = 2, Tfo = 0.0, kw...))
    # ⚠ THE COMPARISON CARRIES A ONE-STEP LAG, AND IT IS PHYSICS, NOT SLOP. Within a step the glue
    # kernel writes p = m·γ(r)v(r) at the particle's CURRENT radius and the position kernel then
    # streams it; the snapshot therefore pairs a momentum set at r_before with a position r_after,
    # Δr = v·Δt. So the residual is m·d(γv)/dr·v·Δt = O(Δt) by construction and the right statement
    # is its ORDER, not its size. (At Δt = 10⁻³, v = 0.5 and dv/dr = 0.075 fm⁻¹ that predicts
    # ≈ 8·10⁻⁵ GeV, which is what comes out.)
    local meas, pred, mf
    worsts = Float64[]; dts = QUICK ? (2e-3, 5e-4) : (4e-3, 2e-3, 1e-3, 5e-4)
    for dt in dts
        _, mc, xc = go(DsT = 0.0, Δt = dt)
        r_end = vec(sqrt.(sum(abs2, xc[end]; dims = 1)))
        v_end = vprof.(min.(r_end, last(xg)))
        pred  = M .* v_end ./ sqrt.(1 .- v_end .^ 2)
        meas  = vec(sqrt.(sum(abs2, mc[end]; dims = 1)))
        push!(worsts, maximum(abs.(meas .- pred)))
    end
    _, mfh, _ = go(DsT = DST, collision_mode = :none)            # free streaming, for the contrast
    mf = mfh[end]
    ord = log(worsts[1] / worsts[end]) / log(dts[1] / dts[end])
    @printf("  comoving: worst per-particle |p_meas − m·γ(r)v(r)| = %s GeV\n",
            join([fmt(w; d = 7) for w in worsts], ", "))
    @printf("      at Δt = %s  ⇒  order %.2f (the one-step lag; it is NOT a modelling error)\n",
            join(string.(dts), ", "), ord)
    @printf("  ⟨|p|⟩ comoving = %.5f GeV   vs free streaming (:none) = %.5f GeV   (ratio %.3f)\n",
            mean(meas), mean(vec(sqrt.(sum(abs2, mf; dims = 1)))),
            mean(meas) / mean(vec(sqrt.(sum(abs2, mf; dims = 1)))))
    gate!(0.7 < ord < 1.4 && worsts[end] < 1e-4,
          "(S5) the glued momentum is m·γ(r)·v(r) up to the O(Δt) streaming lag (order $(fmt(ord; d=2)))")
    if !NOPLOT
        edges = range(0, 3.0; length = 50); ctr = (edges[1:end-1] .+ edges[2:end]) ./ 2
        h(v) = begin
            c = zeros(length(edges) - 1)
            for x in v; b = searchsortedlast(edges, x); 1 <= b <= length(c) && (c[b] += 1); end
            c ./ (sum(c) * step(edges))
        end
        p = plot(ctr, max.(h(meas), 1e-4); m = :circle, c = :steelblue, yscale = :log10,
                 xlabel = "|p| [GeV]", ylabel = "normalised density",
                 label = "comoving (DsT = 0)", title = "(S5) the two limits that are NOT each other",
                 ylims = (1e-3, 10))
        plot!(p, ctr, max.(h(pred), 1e-4); ls = :dash, c = :black, label = "exact m·γ(r)v(r)")
        plot!(p, ctr, max.(h(vec(sqrt.(sum(abs2, mf; dims = 1)))), 1e-4); m = :square,
              c = :firebrick, label = "free streaming (collision_mode = :none)")
        savefig(stash!(p), joinpath(FIGDIR, "semianalytic_S5_blastwave_vs_free.png"))
    end
end

if !NOPLOT && !isempty(PANELS)
    sheet = plot(PANELS...; layout = (length(PANELS), 1), size = (1100, 420 * length(PANELS)))
    savefig(sheet, joinpath(FIGDIR, "semianalytic_summary.png"))
    println("\n  figures → $(FIGDIR)")
end

finish!("bench_semianalytic")
