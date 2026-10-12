#!/usr/bin/env julia
# ==============================================================================================
# bench_adaptive.jl — what the variable time step buys, and what it costs, on a production-shaped run.
#
# The question: on a cooling fireball, does `adaptive_dt = true` reproduce the fixed-Δt answer at
# fewer steps, and is the variable step actually better than a UNIFORM step of the same cost?
#
# Setup: the examples' Bjorken fireball (T₀ = 0.5 GeV at τ₀ = 0.4 fm, Gaussian σ = 4 fm, radial flow
# to 0.65), LP1-like charm (m = 1.5, D_sT = 0.11634), 3 momentum rows with the Bjorken redshift on
# the lab momenta, FONLL-shaped initial condition. ONE initial condition, drawn on the host and
# handed to every run, so the runs differ only by their step and their noise.
#
# Runs: fixed Δt = 2.5e-4 (the REFERENCE), fixed 1e-3 (production), adaptive at four drag
# tolerances, and a fixed Δt with the SAME step count as adaptive tol = 0.01 (the cost-matched
# control). For every run and snapshot: ⟨p_T⟩, ⟨p_T²⟩, ⟨p·r̂⟩ (the radial flow the quarks picked
# up) and ⟨r²⟩, as % deviations from the reference, each with the 1σ noise of the difference.
#
#   julia --project=Julia Julia/LangevInMedium.jl/bench/bench_adaptive.jl
#   LIM_NOGPU=1 ... runs on the CPU at N = 2·10⁵ (slower, noisier)
# ==============================================================================================
include(joinpath(@__DIR__, "bench_common.jl"))
using Dates

const M, DST = 1.5, 0.11634
const τ0, τf = 0.4, 10.4
const SAVE = 2.0
const GPU = gpu_available()
const N = GPU ? 2_000_000 : 200_000
const backend = GPU ? GPUBackend() : CPUBackend()

function fireball(; T0 = 0.50, σ = 4.0, vmax = 0.65, T_floor = 0.05, rmax = 26.0, τ_rise = 2.0, nr = 209)
    xg = collect(range(0.0, rmax; length = nr)); tg = collect(τ0:0.05:(τf + 0.5))
    Tf = [max(T_floor, T0 * (τ0 / τ)^(1 / 3) * exp(-r^2 / (2σ^2))) for r in xg, τ in tg]
    Vf = [vmax * tanh(r / σ) * min(1.0, τ / τ_rise) for r in xg, τ in tg]
    xg, tg, Tf, Vf
end
const XG, TG, TF, VF = fireball()

# the initial condition: FONLL-shaped p_T (1 + (p/2.1)²)^−3.1 at fixed r-profile, isotropic azimuth,
# LOCAL-REST-FRAME momenta (the engine boosts them); drawn once
function initial_condition(N; seed = 2026)
    rng = MersenneTwister(seed)
    x = 3.0 .* randn(rng, 2, N)
    pmax = 12.0; w(p) = p * (1 + (p / 2.1)^2)^(-3.1); wmax = maximum(w, range(0, pmax; length = 4000))
    p = zeros(2, N)
    for i in 1:N
        q = 0.0
        while true
            q = pmax * rand(rng); rand(rng) * wmax <= w(q) && break
        end
        φ = 2π * rand(rng); p[:, i] .= q .* (cos(φ), sin(φ))
    end
    x, p
end
const X0, P0 = initial_condition(N)

function observe(P, X)
    pT = sqrt.(P[1, :] .^ 2 .+ P[2, :] .^ 2)
    r = sqrt.(X[1, :] .^ 2 .+ X[2, :] .^ 2)
    pr = (P[1, :] .* X[1, :] .+ P[2, :] .* X[2, :]) ./ max.(r, 1e-12)
    (; pT = pT, pT2 = pT .^ 2, pr = pr, r2 = r .^ 2)
end

function run(label; kw...)
    Random.seed!(1); GPU && Base.invokelatest(CUDA.seed!, 1)
    t = @elapsed out = Base.invokelatest(simulate_ensemble_bulk, backend,
        collect(0.0:0.5:20.0), collect(0.05:0.1:8.0), ones(80, 41), TF, VF, (XG, TG);
        x_init = X0, p_init = P0, N_particles = N, m = M, DsT = DST, dimensions = 2,
        momentum_dimensions = 3, bjorken_redshift = true, bjorken_frame = :lab, pz_init = :comoving,
        initial_time = τ0, final_time = τf, save_interval = SAVE, Tfo = 0.0, kw...)
    (; label, time = t, tp = collect(out[1]), obs = [observe(out[2][k], out[3][k]) for k in eachindex(out[1])])
end

steps_of(kw) = haskey(kw, :Δt) ? round(Int, (τf - τ0) / kw[:Δt]) :
    length(adaptive_time_grid(TF, TG; initial_time = τ0, final_time = τf, save_interval = SAVE,
                              m = M, DsT = DST, drag_tol = get(kw, :adaptive_drag_tol, 0.01))[1]) - 1

const n_adapt = steps_of((; adaptive_dt = true, adaptive_drag_tol = 0.01))
const dt_matched = (τf - τ0) / n_adapt
# the cost-matched uniform step must land on the snapshots; round to the nearest divisor of SAVE
const dt_ctrl = SAVE / round(SAVE / dt_matched)

const RUNS = [
    ("fixed Δt=2.5e-4 (ref)", (; Δt = 2.5e-4)),
    ("fixed Δt=1e-3 (prod)",  (; Δt = 1e-3)),
    ("adaptive tol=0.005",    (; adaptive_dt = true, adaptive_drag_tol = 0.005)),
    ("adaptive tol=0.01",     (; adaptive_dt = true, adaptive_drag_tol = 0.01)),
    ("adaptive tol=0.02",     (; adaptive_dt = true, adaptive_drag_tol = 0.02)),
    ("adaptive tol=0.04",     (; adaptive_dt = true, adaptive_drag_tol = 0.04)),
    (@sprintf("fixed Δt=%.4g (cost-matched)", dt_ctrl), (; Δt = dt_ctrl)),
]

@printf("bench_adaptive — %s, %s, N = %d\n", Dates.format(now(), "yyyy-mm-dd HH:MM"), GPU ? "GPU" : "CPU", N)
@printf("fireball τ %.1f → %.1f fm, T₀ = %.2f GeV; m = %.1f, D_sT = %.5f; η_D(T₀) = %.2f fm⁻¹\n",
        τ0, τf, TF[1, 1], M, DST, (TF[1, 1]^2 / (M * DST)) / HBARC)
results = [run(l; kw...) for (l, kw) in RUNS]
ref = results[1]
for (res, (l, kw)) in zip(results, RUNS)
    @printf("\n%-30s steps %6d   wall %6.1f s\n", l, steps_of(kw), res.time)
    res === ref && continue
    @printf("   τ [fm]   ⟨p_T⟩ %%       ⟨p_T²⟩ %%      ⟨p·r̂⟩ %%      ⟨r²⟩ %%       (deviation from ref ± 1σ noise)\n")
    for k in 2:length(ref.tp)
        @assert abs(res.tp[k] - ref.tp[k]) < 1e-9
        @printf("   %5.1f ", ref.tp[k])
        for f in (:pT, :pT2, :pr, :r2)
            a = getfield(res.obs[k], f); b = getfield(ref.obs[k], f)
            d = 100 * (mean(a) / mean(b) - 1)
            σ = 100 * sqrt(var(a) / N + var(b) / N) / abs(mean(b))
            @printf("  %+6.3f±%5.3f", d, σ)
        end
        println()
    end
end

# ── gate: the DEFAULT tolerance must reproduce the reference within 3σ on every observable and
# snapshot. σ treats the two runs as independent; they share the initial condition, so it is
# conservative. The flag lives in a Ref: a bare `ok = false` in a top-level loop would make a new
# local and the gate would print PASS after its failures (CLAUDE.md, the soft-scope trap).
const OK = Ref(true)
function gate(res, ref)
    for k in 2:length(ref.tp), f in (:pT, :pT2, :pr, :r2)
        a = getfield(res.obs[k], f); b = getfield(ref.obs[k], f)
        d = mean(a) / mean(b) - 1; σ = sqrt(var(a) / N + var(b) / N) / abs(mean(b))
        if abs(d) > 3σ
            @printf("FAIL %s: %s at τ = %.1f deviates %+.3f %% (3σ = %.3f %%)\n", res.label, f, ref.tp[k], 100d, 300σ)
            OK[] = false
        end
    end
end
gate(results[findfirst(r -> r.label == "adaptive tol=0.01", results)], ref)
n_prod = steps_of((; Δt = 1e-3))
@printf("\ndefault adaptive: %d steps vs %d at the production Δt = 1e-3 (%.1f× fewer)\n", n_adapt, n_prod, n_prod / n_adapt)
println(OK[] ? "[PASS] bench_adaptive: the default tolerance reproduces the reference within 3σ" :
               "[FAIL] bench_adaptive")
OK[] || exit(1)
