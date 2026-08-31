#!/usr/bin/env julia
# ==============================================================================================
# bench_gpu_parity.jl — CPU ↔ GPU agreement, including the inputs that separate the two.
#
# ⚠ THIS IS THE STATISTICAL HALF OF THE CPU/GPU COMPARISON. Since 2026-08-31 the DETERMINISTIC
# half lives in `test/test_kernel_parity.jl`, which drives each kernel pair with the same inputs
# AND the same injected noise and compares per particle at 1e-12 or tighter. Prefer that file when
# asking "do the two backends compute the same thing"; this one answers the different and still
# necessary question "do two full RUNS, with different RNG streams, land in the same place".
#
# The GPU kernels were hand-transliterated from the CPU ones and drifted: the CPU interpolant
# clamps the query point to the tabulated domain, the CPU boosts clamp |v| < 1 and the CPU
# force/diffusion kernels clamp T ≥ 0 — the GPU versions did none of these (see CHANGELOG
# 0.2.0). On a well-behaved background the two paths agree to the statistical floor; these
# ADVERSARIAL cases put particles outside the grid, hand in a field cell with |v| > 1 and a
# T = 0 region, and ask both backends for a finite, mutually consistent answer.
#
#   julia --project=Julia Julia/LangevInMedium.jl/bench/bench_gpu_parity.jl
# ==============================================================================================
include(joinpath(@__DIR__, "bench_common.jl"))
using QuadGK
gpu_available() || (println("CUDA not functional — nothing to compare"); exit(0))

const M, DST, N = 1.5, 0.11634, 50_000
# CURAND is unseedable, so a GPU run is one sample of the ensemble: at N = 50 000 the run-to-run
# spread of ⟨p²⟩ is 1.5 % (8 repeats, measured 2026-08-22) while the GPU mean sits 0.66 % from the
# CPU. The default tolerance is the scatter, not the agreement — tighten it only by averaging runs.
moms(mom, pos) = (p2 = mean(sum(abs2, mom[end]; dims = 1)), x2 = mean(sum(abs2, pos[end]; dims = 1)),
                  r = mean(sqrt.(sum(abs2, pos[end]; dims = 1))))
function compare(label, xg, tg, Tf, Vf; rtol = 0.03, kw...)
    tc, mc, xc = run_fields(CPUBackend(), xg, tg, Tf, Vf; M, DsT = DST, N, kw...)
    tg_, mg, xg_ = run_fields(GPUBackend(), xg, tg, Tf, Vf; M, DsT = DST, N, kw...)
    gate!(collect(tc) == collect(tg_), "$label: CPU and GPU report the same snapshot times")
    nf_c = nonfinite(mc[end]) + nonfinite(xc[end]); nf_g = nonfinite(mg[end]) + nonfinite(xg_[end])
    a = moms(mc, xc); b = moms(mg, xg_)
    @printf("    %-34s CPU ⟨p²⟩=%.4f ⟨x²⟩=%.2f  GPU ⟨p²⟩=%.4f ⟨x²⟩=%.2f  non-finite CPU/GPU %d/%d\n",
            label, a.p2, a.x2, b.p2, b.x2, nf_c, nf_g)
    gate!(nf_c == 0, "$label: CPU result finite")
    gate!(nf_g == 0, "$label: GPU result finite")
    okm = nf_g == 0 && all(isapprox.(values(a), values(b); rtol = rtol))
    gate!(okm, "$label: GPU moments within $(rtol) of CPU")
end

println("── nominal backgrounds ──")
let
    xg = collect(0.0:0.25:40.0); tg = collect(0.4:0.1:2.0)
    Tf = [max(0.12, 0.45 * exp(-r^2 / 60) * (0.4 / τ)^(1 / 3)) for r in xg, τ in tg]
    Vf = [0.6 * r / (r + 4.0) * min(1.0, τ / 2.0) for r in xg, τ in tg]
    rng = MersenneTwister(1)
    x0 = 3.0 .* randn(rng, 2, N); p0 = juttner_sample(rng, M, 0.4, 2, N)
    compare("flow, 2 momentum rows", xg, tg, Tf, Vf; dt = 2e-3, tfinal = 1.0, save = 0.25, x0, p0, seed = 11)
    compare("flow, p_z + Bjorken redshift", xg, tg, Tf, Vf; dt = 2e-3, tfinal = 1.0, save = 0.25, x0, p0, seed = 12,
            momentum_dimensions = 3, bjorken_redshift = true)
    compare("flow, Galilean", xg, tg, Tf, Vf; dt = 2e-3, tfinal = 1.0, save = 0.25, x0, p0, seed = 13, relativistic = false)
    compare("flow, RTA", xg, tg, Tf, Vf; dt = 2e-3, tfinal = 1.0, save = 0.25, x0, p0, seed = 14, collision_mode = :rta)
end

println("── adversarial: particles leave the tabulated (r, τ) domain ──")
let
    # T falls linearly in r AND τ and the run outlasts the time table: extrapolating the table gives
    # T < 0. The CPU interpolant clamps the query point to the last tabulated value; an unclamped
    # one extrapolates to negative T ⇒ κ < 0 ⇒ √κ = NaN in the force kernel (and, with position
    # diffusion on, D < 0 silently switches the diffusion off instead of flooring T).
    xg = collect(0.0:0.5:10.0); tg = collect(0.0:0.25:0.5)
    Tf = [0.45 - 0.030 * r - 0.2 * τ for r in xg, τ in tg]       # 0.05 GeV at (r=10, τ=0.5); negative beyond
    Vf = zeros(length(xg), length(tg))
    rng = MersenneTwister(2)
    x0 = 9.0 .* ones(2, N) ./ sqrt(2)
    p0 = 2.5 .* ones(2, N) ./ sqrt(2) .+ 0.3 .* randn(rng, 2, N)   # at r = 9, heading outward at ~0.9c
    # NOTE: tfinal/save = 2.5/1.25 is deliberately NOT step-divisible (floor(2.5/Δt) = 1250 steps,
    # save_every = round(1.25/Δt) = 625 ⇒ 124 steps trail off the history). The `save_interval does
    # not divide the evolution` warning below is expected and is itself under test: both backends
    # must drop the same steps and report the same snapshot times.
    compare("outside grid in r and τ", xg, tg, Tf, Vf; dt = 2e-3, tfinal = 2.5, save = 1.25, x0, p0, seed = 21)
    compare("outside grid, position_diffusion", xg, tg, Tf, Vf; dt = 2e-3, tfinal = 2.5, save = 1.25, x0, p0, seed = 22,
            position_diffusion = true, rtol = 0.06)
end

println("── adversarial: a flow-field cell with |v| > 1 ──")
let
    # v = 1.02 in the outer cells (a hydro table with a superluminal glitch). The CPU clamps
    # v to √(1−1e−12) — γ ≈ 10⁶, absurd but finite and the same on both backends once clamped;
    # unclamped, 1−v²+1e−10 < 0 and γ is NaN. The mean p² is dominated by the γ ≈ 10⁶ outliers,
    # so the agreement gate here is the MEDIAN p², which the glitch cells do not reach.
    xg = collect(0.0:0.5:20.0); tg = collect(0.0:0.25:2.0)
    Tf = fill(0.30, length(xg), length(tg))
    Vf = [r > 12 ? 1.02 : 0.08 * r for r in xg, τ in tg]
    rng = MersenneTwister(3)
    x0 = 6.0 .* randn(rng, 2, N); p0 = juttner_sample(rng, M, 0.3, 2, N)
    tc, mc, xc = run_fields(CPUBackend(), xg, tg, Tf, Vf; M, DsT = DST, N, dt = 2e-3, tfinal = 1.0, save = 0.25, x0, p0, seed = 31)
    tg_, mg, xg_ = run_fields(GPUBackend(), xg, tg, Tf, Vf; M, DsT = DST, N, dt = 2e-3, tfinal = 1.0, save = 0.25, x0, p0, seed = 31)
    medc = median(vec(sum(abs2, mc[end]; dims = 1))); medg = median(vec(sum(abs2, mg[end]; dims = 1)))
    @printf("    %-34s CPU median p²=%.4f  GPU median p²=%.4f  non-finite CPU/GPU %d/%d\n",
            "|v| = 1.02 cells", medc, medg, nonfinite(mc[end]) + nonfinite(xc[end]), nonfinite(mg[end]) + nonfinite(xg_[end]))
    gate!(nonfinite(mc[end]) + nonfinite(xc[end]) == 0, "|v| = 1.02 cells: CPU result finite")
    gate!(nonfinite(mg[end]) + nonfinite(xg_[end]) == 0, "|v| = 1.02 cells: GPU result finite")
    gate!(isfinite(medg) && isapprox(medc, medg; rtol = 0.03), "|v| = 1.02 cells: GPU median p² within 0.03 of CPU")
end

println("── GPU-only: freeze-out capture on a flowing background ──")
let
    # No CPU twin exists, so this is a self-consistency gate: every particle must latch exactly
    # once, inside the run, with a finite state. The QUANTITATIVE check (the crossing booked to
    # 2.7e-15 against a host-computed reference) is `test/test_gpu_only_paths.jl` F1-F4.
    # the run must outlast the CENTRE's cooling: T(r=0, τ) = 0.45(0.4/τ)^{1/3} reaches Tfo = 0.155
    # only at τ ≈ 9.8 fm, so a 7.4 fm run leaves ~10 % of the ensemble legitimately unfrozen.
    xg = collect(0.0:0.25:40.0); tg = collect(0.4:0.1:12.0)
    Tf = [max(0.10, 0.45*exp(-r^2/60)*(0.4/τ)^(1/3)) for r in xg, τ in tg]
    Vf = [0.6*r/(r + 4.0)*min(1.0, τ/2.0) for r in xg, τ in tg]
    rng = MersenneTwister(41)
    x0 = 3.0 .* randn(rng, 2, N); p0 = juttner_sample(rng, M, 0.4, 2, N)
    Random.seed!(41)
    rg = collect(0.0:0.5:20.0); pg = collect(0.05:0.1:8.0); dens = ones(length(pg), length(rg))
    fo = Base.invokelatest(simulate_ensemble_bulk, GPUBackend(), rg, pg, dens, Tf, Vf, (xg, tg);
        x_init = x0, p_init = p0, N_particles = N, Δt = 2e-3, initial_time = 0.4,
        final_time = 11.4, save_interval = 1.0, m = M, DsT = DST, dimensions = 2,
        Tfo = 0.155, freezeout_capture = true, freezeout_interp = true)
    frac = count(==(1.0), fo.flag)/N
    @printf("    %-34s flagged %.3f  τ_fo ∈ [%.3f, %.3f]  non-finite %d\n",
            "freezeout_capture", frac, minimum(fo.tau[fo.flag .== 1]), maximum(fo.tau[fo.flag .== 1]),
            nonfinite(fo.pos) + nonfinite(fo.mom) + nonfinite(fo.tau))
    gate!(frac == 1.0, "freezeout_capture: EVERY particle latches once the run outlasts the centre's cooling ($(fmt(100frac; d=1)) %)")
    gate!(nonfinite(fo.pos) + nonfinite(fo.mom) + nonfinite(fo.tau) == 0, "freezeout_capture: all booked states finite")
    gate!(all(0.4 .<= fo.tau[fo.flag .== 1] .<= 11.4), "freezeout_capture: every booked τ lies inside the run")
    gate!(size(fo.pos) == (2, N) && size(fo.mom) == (2, N), "freezeout_capture: returns (2, N) position and momentum")
end

# NOTE (corrected 2026-08-31): a field with T = 0 (or T < 0) cells is rejected by
# `_build_time_spline` ("Tmin must be > 0") on BOTH backends before any kernel runs, so no DRIVER
# can reach T = 0 — the statement below is right about reachability. It was wrong to conclude that
# the two backends therefore agree there: driven directly, they do not. At T = 0 the CPU RTA
# sampler returns |p*| = 0 and ZEROES the momentum, while the GPU inverse-CDF table clamps to its
# coldest column and returns a finite thermal |p*|. Recorded as a known divergence in
# `test/test_kernel_parity.jl` @testset "D1a".

finish!("bench_gpu_parity")
