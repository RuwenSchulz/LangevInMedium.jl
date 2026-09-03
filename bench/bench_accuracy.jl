#!/usr/bin/env julia
# ==============================================================================================
# bench_accuracy.jl — the Δt ACCURACY BUDGET: how wrong is the engine at a given step size?
#
# WHY THIS IS NOT bench_physics_gates (d). That gate pins the propagator's Δt bias inside a 1.5 %
# budget at three step sizes and says so in its own comment: "This is a budget gate, not a scaling
# measurement." At N = 1.5·10⁵ its SEM is 0.37 % and the three readings (−1.23 %, −0.47 %, −0.84 %)
# do not resolve a slope — so the tree knows the bias EXISTS, and has never measured its ORDER or
# its COEFFICIENT. Neither has it ever answered the question a production run actually asks:
#
#     "at Δt = 10⁻³ fm on a T ≈ 0.3 GeV medium, how wrong are ⟨p²⟩, D_s and the ℓ=1 rate?"
#
# HOW THE NOISE IS BEATEN DOWN. Every quantity here is a STATIONARY-STATE average, so it is
# averaged over TIME as well as over particles: with snapshots one drag time apart over a run of
# `NTAU` drag times, N_eff ≈ N·NTAU and the SEM falls as 1/√(N·NTAU) at a cost that is the same
# either way (cost ∝ N·T_run/Δt, N_eff ∝ N·T_run — the ratio is Δt regardless of how the budget is
# split). That is what makes an O(ηΔt) bias measurable down to ηΔt ≈ 0.01, where it is ≈ 0.1 %.
#
# WHAT IS MEASURED
#   (A) stationary ⟨p²⟩ vs the exact d-dimensional Jüttner moment  — relativistic, 2 and 3 rows.
#       The Galilean branch is the CONTROL: its propagator is exact at any Δt, so a nonzero slope
#       there would mean the measurement, not the engine, is biased.
#   (B) realised D_s from the MSD slope vs D_s = D_sT/T·ħc         — the position/momentum split.
#   (C) realised ℓ=1 (diffusion-current) rate vs λ₁η_D             — reported with its noise floor.
#   (D) the RTA/BGK realised rate vs the nominal 1/τ_n at five step sizes. Until 2026-09-02 the
#       per-step collision probability was the LINEARISED Δt/τ_n, so the realised rate was
#       −ln(1 − Δt/τ_n)/Δt and the RTA carried a step-size ceiling; it is `−expm1(−Δt/τ_n)` now
#       (0.2.3), and this block is what says the Δt trend is gone rather than merely smaller.
#
# Each block fits log|bias| against log Δt and prints the fitted order and coefficient, so a
# required accuracy converts into a required Δt:  |bias| ≈ C·(ηΔt)^order.
#
#   julia --project=Julia Julia/LangevInMedium.jl/bench/bench_accuracy.jl
#   LIM_BENCH_QUICK=1 ...      fewer step sizes / smaller ensembles (≈ ¼ the wall time)
#
# Wall-clock independent: this bench measures ACCURACY, not speed, so unlike bench_throughput.jl
# it is valid on a loaded machine.
# ==============================================================================================
include(joinpath(@__DIR__, "bench_common.jl"))
using QuadGK, Bessels, Dates

const QUICK = get(ENV, "LIM_BENCH_QUICK", "0") == "1"
const M, DST = 1.5, 0.11634
const TBATH  = 0.30

eta_drag(T) = (T^2 / (M * DST)) / HBARC          # fm⁻¹, the Einstein drag the kernel builds
ds_fm(T)    = (DST / T) * HBARC                  # fm, D_s = D_sT/T [GeV⁻¹] → fm
K2K3(z)     = Bessels.besselkx(2, z) / Bessels.besselkx(3, z)
lam1(T, d)  = jmean(p -> (M / sqrt(p^2 + M^2)) * p^2, M, T, d) / jmean(p -> p^2, M, T, d)

const ETA  = eta_drag(TBATH)
const TAUD = 1 / ETA
const ETADT = QUICK ? (0.16, 0.08, 0.04) : (0.16, 0.08, 0.04, 0.02, 0.01)

"Fit log|y| = a + order·log(x); returns (order, C) with |y| ≈ C·x^order."
function fit_order(x, y)
    a = hcat(ones(length(x)), log.(x)) \ log.(abs.(y))
    (a[2], exp(a[1]))
end

"""
Stationary-state run in a uniform bath: start FROM the exact Jüttner (independent rejection
sampler), discard the first `burn` drag times, and return the snapshots after it. Sampling one
drag time apart keeps the snapshots close to independent, so N_eff ≈ N × (number kept).
"""
function stationary_run(T, dt; N, ntau, burn, pdim, rel, seed, kw...)
    tf = ntau * TAUD
    P = rel ? juttner_sample(MersenneTwister(seed), M, T, pdim, N) :
              sqrt(M * T) .* randn(MersenneTwister(seed), pdim, N)
    xg, tg, Tf, Vf = box_fields(T; tf = tf)
    # save every drag time; the sampler's p_init must have `dimensions` rows, so hand it the
    # transverse pair and let the engine complete p_z when pdim = 3 is asked for.
    p0 = pdim == 3 ? P[1:2, :] : P
    tt, mm, xx = run_fields(CPUBackend(), xg, tg, Tf, Vf; M, DsT = DST, N, dt, tfinal = tf,
        save = TAUD, x0 = zeros(2, N), p0 = p0, seed = seed + 1, relativistic = rel,
        momentum_dimensions = pdim, kw...)
    keep = findall(>=(first(tt) + burn * TAUD), collect(tt))
    tt, mm, xx, keep
end

println("LangevInMedium — Δt accuracy budget   ($(Dates.format(now(), "yyyy-mm-dd HH:MM")), Julia $(VERSION))")
println("bath T = $TBATH GeV, m = $M GeV, D_sT = $DST  ⇒  η_D = $(fmt(ETA; d=4)) fm⁻¹, τ_drag = $(fmt(TAUD; d=4)) fm")
println("production runs use Δt ≈ 1e-3…5e-3 fm, i.e. ηΔt ≈ $(fmt(ETA*1e-3; d=5))…$(fmt(ETA*5e-3; d=4))\n")

# ── (A) stationary ⟨p²⟩ ────────────────────────────────────────────────────────────────────────
# ⚠ READ THE COLUMNS AS A SLOPE, NOT AS FIVE ABSOLUTE NUMBERS. Every Δt is started from the SAME
# seeded Jüttner draw, so the IC's own finite-N deviation from the exact moment (≈ 1 SEM) is COMMON
# to all five columns: it cancels in the Δt-dependence, which is what is being measured, and it
# shifts the whole row by a constant. The Galilean control makes that visible — its five readings
# are +0.03…+0.12 %, one-signed and Δt-independent, which is exactly that shared offset and not a
# propagator bias. Subtract the control's level before quoting an absolute relativistic bias.
println("── (A) stationary ⟨p²⟩ vs the exact Jüttner moment ──")
let N = QUICK ? 20_000 : 50_000, ntau = QUICK ? 40 : 120, burn = 5
    for (rel, pdim) in ((true, 2), (true, 3), (false, 2))
        ref  = rel ? jmean(p -> p^2, M, TBATH, pdim) : pdim * M * TBATH
        bias = Float64[]
        for x in ETADT
            dt = x / ETA
            _, mm, _, keep = stationary_run(TBATH, dt; N, ntau, burn, pdim, rel, seed = 101)
            vals = [mean(sum(abs2, mm[k]; dims = 1)) for k in keep]
            push!(bias, mean(vals) / ref - 1)
        end
        neff = N * (QUICK ? 35 : 115)
        sem  = sqrt(2 / neff)
        ord, C = fit_order(collect(ETADT), bias)
        lbl = rel ? "relativistic, $pdim rows" : "GALILEAN CONTROL, $pdim rows"
        println("  $lbl   ref ⟨p²⟩ = $(fmt(ref; d=5)) GeV²,  SEM $(fmt(100sem; d=3)) %")
        println("    ηΔt   ", join([lpad(fmt(x; d = 3), 9) for x in ETADT]))
        println("    bias% ", join([lpad(fmt(100b; d = 3), 9) for b in bias]))
        if rel
            gate!(all(<(0), bias) && ord > 0.7,
                  "(A) $lbl: bias is one-signed and scales as (ηΔt)^$(fmt(ord; d=2))  ⇒  |bias| ≈ $(fmt(100C; d=2)) %·(ηΔt)^$(fmt(ord; d=2))")
            println("    ⇒ at production ηΔt = $(fmt(ETA*1e-3; d=5)) this extrapolates to $(fmt(100C*(ETA*1e-3)^ord; d=5)) %")
        else
            gate!(all(b -> abs(b) < 3sem, bias),
                  "(A) $lbl: exact at every Δt (all $(length(bias)) within 3 SEM = $(fmt(300sem; d=3)) %) — the measurement itself is unbiased")
        end
        println()
    end
end

# ── (B) realised D_s ───────────────────────────────────────────────────────────────────────────
println("── (B) realised D_s from the MSD slope vs D_s = D_sT/T ──")
let N = QUICK ? 20_000 : 40_000, ntau = QUICK ? 40 : 80
    ref = ds_fm(TBATH)
    bias = Float64[]
    for x in ETADT
        dt = x / ETA
        tt, _, xx, _ = stationary_run(TBATH, dt; N, ntau, burn = 0, pdim = 3, rel = true, seed = 211)
        tt = collect(tt)
        msd = [mean(sum(abs2, x .- xx[1]; dims = 1)) for x in xx]
        sel = tt .>= first(tt) + 0.4 * (last(tt) - first(tt))     # diffusive window only
        c = hcat(ones(sum(sel)), tt[sel]) \ msd[sel]
        push!(bias, (c[2] / 4) / ref - 1)                          # 2 spatial dims ⇒ slope = 4 D_s
    end
    ord, C = fit_order(collect(ETADT), bias)
    println("  D_s reference = $(fmt(ref; d=5)) fm   (2 spatial rows, 3 momentum rows)")
    println("    ηΔt   ", join([lpad(fmt(x; d = 3), 9) for x in ETADT]))
    println("    bias% ", join([lpad(fmt(100b; d = 3), 9) for b in bias]))
    gate!(abs(bias[end]) < 0.02, "(B) realised D_s within 2 % at the smallest step (ηΔt = $(fmt(last(ETADT); d=3)))")
    # ⚠ The MSD slope's own scatter (a fit over ~½ the snapshots of a random walk) is percent-level
    # here, so only the LARGEST step separates from noise: the fit below is an UPPER BOUND on the
    # position-update bias, not a measured law like (A)'s. What it does establish is that D_s is
    # unbiased to ≲1 % everywhere at ηΔt ≤ 0.08 — i.e. the O(ηΔt) bias (A) sees in ⟨p²⟩ does NOT
    # propagate into the transport coefficient at any step size anyone would use.
    println("    fitted |bias| ≲ $(fmt(100C; d=2)) %·(ηΔt)^$(fmt(ord; d=2))  (noise-limited; see the note)\n")
end

# ── (C) realised ℓ=1 rate ──────────────────────────────────────────────────────────────────────
println("── (C) realised ℓ=1 (diffusion-current) rate vs λ₁·η_D ──")
let N = QUICK ? 100_000 : 400_000, v = 0.05
    γ = 1 / sqrt(1 - v^2)
    lam = K2K3(M / TBATH)
    rates = Float64[]
    for x in ETADT
        dt = x / ETA
        P = juttner_sample(MersenneTwister(301), M, TBATH, 3, N)
        for i in 1:N
            E = sqrt(M^2 + sum(abs2, view(P, :, i))); P[1, i] = γ * (P[1, i] + v * E)
        end
        xg, tg, Tf, Vf = box_fields(TBATH; tf = 0.35)
        tt, mm, _ = run_fields(CPUBackend(), xg, tg, Tf, Vf; M, DsT = DST, N, dt,
            tfinal = 0.30, save = 0.05, x0 = zeros(2, N), p0 = P[1:2, :], seed = 303,
            momentum_dimensions = 3)
        jx = [mean(m[1, :]) for m in mm]; tt = collect(tt)
        c = hcat(ones(length(tt)), tt) \ log.(jx)
        push!(rates, -c[2] / ETA)
    end
    sem = 2.4 / sqrt(N / 1e6) / 100          # the bench_physics_gates (c) seed-scan floor, scaled
    println("  λ₁ = K₂/K₃(z = $(fmt(M/TBATH; d=2))) = $(fmt(lam; d=5));  fitted-rate noise floor ≈ ±$(fmt(100sem; d=2)) %")
    println("    ηΔt        ", join([lpad(fmt(x; d = 3), 9) for x in ETADT]))
    println("    rate/η_D   ", join([lpad(fmt(r; d = 4), 9) for r in rates]))
    println("    vs λ₁ (%)  ", join([lpad(fmt(100 * (r / lam - 1); d = 2), 9) for r in rates]))
    gate!(abs(rates[end] / lam - 1) < 3sem,
          "(C) the ℓ=1 rate is λ₁·η_D at the smallest step, within the $(fmt(300sem; d=1)) % noise floor")
    println()
end

# ── (D) the RTA/BGK step is exact in Δt ────────────────────────────────────────────────────────
println("── (D) RTA/BGK: the realised rate is 1/τ_n at any step size ──")
let N = QUICK ? 100_000 : 300_000
    τn = tau_n_main3(TBATH, M, DST)
    println("  τ_n = $(fmt(τn; d=5)) fm.  A BGK step that re-draws with probability p per Δt realises")
    println("  a rate −ln(1 − p)/Δt, so the per-step probability must be p = 1 − e^{−Δt/τ_n} for that")
    println("  to be 1/τ_n. It was the LINEARISED Δt/τ_n until 2026-09-02, which relaxed the ensemble")
    println("  by −ln(1 − Δt/τ_n)/(Δt/τ_n) too fast — 1.0517 at Δt/τ_n = 0.084, 1.2198 at 0.33.")
    println("    Δt        Δt/τ_n     measured/nominal    OLD linearised law    was measured")
    ok = true
    olds = Dict(0.002 => 1.0013, 0.01 => 1.0069, 0.05 => 1.0517, 0.10 => 1.0936, 0.20 => 1.2198)
    for dt in (QUICK ? (0.01, 0.05, 0.20) : (0.002, 0.01, 0.05, 0.10, 0.20))
        x0 = zeros(2, N); x0[1, :] .= 50.0
        p0 = zeros(2, N); p0[1, :] .= 1.0                    # a pure ℓ=1 (drift) perturbation
        xg, tg, Tf, Vf = box_fields(TBATH; tf = 1.2)
        tt, mm, _ = run_fields(CPUBackend(), xg, tg, Tf, Vf; M, DsT = DST, N, dt,
            tfinal = 1.0, save = 0.5, x0 = x0, p0 = p0, seed = 401, collision_mode = :rta)
        tt = collect(tt)
        rate = -log(abs(mean(mm[end][1, :]) / mean(mm[1][1, :]))) / (last(tt) - first(tt))
        old = -log(1 - min(dt / τn, 1 - 1e-15)) / dt
        @printf("   %7.4f  %8.4f   %16.4f  %20.4f  %13s\n", dt, dt / τn, rate * τn, old * τn,
                haskey(olds, dt) ? fmt(olds[dt]; d = 4) : "—")
        ok &= isapprox(rate * τn, 1.0; rtol = 0.03)          # 1/√N ≈ 0.2 %; 3 % is the honest bar
    end
    gate!(ok, "(D) the realised BGK rate is 1/τ_n at every step size, with no Δt trend")
    println("    ⇒ the step-size ceiling the RTA used to carry is gone: `−expm1(−Δt/τ_n)` is the")
    println("      exact per-step probability, so Δt is now limited by the STREAMING error only.")
    println()
end

finish!("bench_accuracy")
