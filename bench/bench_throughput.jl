#!/usr/bin/env julia
# ==============================================================================================
# bench_throughput.jl — where the wall clock goes.
#
# Particle-steps per second for the CPU and GPU paths over the production kwarg axes, plus the
# three HOST-side phases that the campaign notes blame for most of a batch (the FONLL rejection
# sampler, the per-step Gaussian draw, the snapshot save). No BenchmarkTools: min of 3 `@elapsed`
# after one warm-up. Writes a markdown table to bench/results/throughput_<host>_<date>.md.
#
#   julia --project=Julia Julia/LangevInMedium.jl/bench/bench_throughput.jl           (LIM_NOGPU=1 to skip the GPU)
#   LIM_BENCH_QUICK=1 ...                                                               (N ≤ 1e5 only)
# ==============================================================================================
include(joinpath(@__DIR__, "bench_common.jl"))
using Dates
const HAVE_GPU = gpu_available()
const QUICK = get(ENV, "LIM_BENCH_QUICK", "0") == "1"
const M, DST = 1.5, 0.11634
const STEPS = (100, 400)          # two step counts ⇒ marginal cost per particle-step and fixed per-call overhead
const NS = QUICK ? (10_000, 100_000) : (10_000, 100_000, 1_000_000)

function flow_fields(; t0 = 0.4, tf = 1.0)
    xg = collect(0.0:0.25:40.0); tg = collect(t0:0.1:(t0 + tf + 0.5))
    Tf = [max(0.12, 0.45 * exp(-r^2 / 60) * (t0 / τ)^(1 / 3)) for r in xg, τ in tg]
    Vf = [0.6 * r / (r + 4.0) * min(1.0, τ / 2.0) for r in xg, τ in tg]
    xg, tg, Tf, Vf
end
const FIELDS = flow_fields()

"min-of-3 wall time of f() after one warm-up call"
function best3(f)
    f(); minimum(@elapsed(f()) for _ in 1:3)
end

rows = String[]
push!(rows, "| backend | N | momentum rows | relativistic | mode | wall [s] (steps) | wall [s] (steps) | marginal ns/particle-step | fixed per call [s] |")
push!(rows, "|---|---|---|---|---|---|---|---|---|")
println("── engine throughput ($(STEPS[1]) and $(STEPS[2]) steps, dt=2e-3, flowing background, 2 snapshots) ──")
for backend in (HAVE_GPU ? (CPUBackend(), GPUBackend()) : (CPUBackend(),)), N in NS,
    pdim in (2, 3), rel in (true, false), mode in (:langevin, :rta)
    backend isa CPUBackend && N == 1_000_000 && mode == :rta && continue   # CPU rejection sampler: minutes; not informative
    rng = MersenneTwister(1); x0 = 3.0 .* randn(rng, 2, N); p0 = 0.8 .* randn(rng, 2, N)
    f(steps) = () -> run_fields(backend, FIELDS...; M, DsT = DST, N, dt = 2e-3, tfinal = steps * 2e-3, save = steps * 1e-3,
                                x0, p0, seed = 2, momentum_dimensions = pdim, relativistic = rel, collision_mode = mode)
    st = (backend isa CPUBackend && N >= 1_000_000) ? (25, 100) : STEPS   # CPU at 10⁶: 200 ns × N × steps
    t1 = best3(f(st[1])); t2 = best3(f(st[2]))
    marg = (t2 - t1) / ((st[2] - st[1]) * N)                # s per particle-step
    fixed = max(t1 - st[1] * N * marg, 0.0)
    @printf("  %-10s N=%-8d pdim=%d rel=%-5s %-8s %7.3f / %7.3f s   marginal %6.1f ns/particle-step   fixed %5.2f s\n",
            nameof(typeof(backend)), N, pdim, rel, mode, t1, t2, 1e9 * marg, fixed)
    push!(rows, "| $(nameof(typeof(backend))) | $N | $pdim | $rel | $mode | $(fmt(t1; d=3)) ($(st[1])) | $(fmt(t2; d=3)) ($(st[2])) | $(fmt(1e9*marg; d=1)) | $(fmt(fixed; d=2)) |")
end

println("── host-side phases ──")
host = String[]
push!(host, "| phase | N | wall [s] | note |"); push!(host, "|---|---|---|---|")
let N = 100_000
    rg = collect(0.0:0.5:20.0); pg = collect(0.05:0.1:8.0)
    dens = [p * exp(-sqrt(p^2 + M^2) / 0.5) * exp(-r^2 / 30) for p in pg, r in rg]       # σ_r ≈ 3.9 fm in a 20 fm disc
    for cart in (true, false)
        t = best3(() -> (Random.seed!(1); LangevInMedium.sample_particles_from_FONLL(rg, pg, dens, N; cartesian_spatial_sampling = cart)))
        @printf("  FONLL sampler cartesian=%-5s N=%d  %7.3f s   (%.2f µs/particle)\n", cart, N, t, 1e6 * t / N)
        push!(host, "| FONLL sampler, cartesian=$cart | $N | $(fmt(t; d=3)) | $(fmt(1e6*t/N; d=2)) µs/particle; Gaussian σ_r≈3.9 fm in r≤20 fm |")
    end
    # rejection acceptance of the Cartesian sampler at production geometry: fireball σ ≈ 3.9 fm, rmax = 20 fm
    for pdim in (2, 3)
        ξ = zeros(pdim, N)
        t = best3(() -> randn!(ξ))
        @printf("  randn! %d×%d per step  %7.4f s\n", pdim, N, t)
        push!(host, "| randn! ($pdim × N) per step | $N | $(fmt(t; d=4)) | CPU path draws this every step |")
    end
    H = zeros(2, N, 10); A = randn(2, N)
    t = best3(() -> (H[:, :, 5] .= A))
    push!(host, "| snapshot save (2 × N) | $N | $(fmt(t; d=4)) | per saved snapshot |")
    @printf("  snapshot copy 2×%d  %7.4f s\n", N, t)
    if HAVE_GPU
        t = best3(() -> Base.invokelatest(() -> (d = CUDA.randn(Float64, 2, N); CUDA.synchronize(); d)))
        push!(host, "| CUDA.randn 2 × N | $N | $(fmt(t; d=5)) | device draw per step |")
        @printf("  CUDA.randn 2×%d  %7.5f s\n", N, t)
        Hd = CUDA.zeros(Float64, 2, N * 10)
        t = best3(() -> Base.invokelatest(() -> (h = Array(Hd); h)))
        push!(host, "| device→host copy (2 × 10N) | $N | $(fmt(t; d=4)) | whole history download |")
        @printf("  D2H copy 2×%d  %7.4f s\n", 10N, t)
    end
end

out = joinpath(RESULTS, "throughput_$(gethostname())_$(Dates.format(now(), "yyyy-mm-dd")).md")
open(out, "w") do io
    println(io, "# LangevInMedium throughput — $(gethostname()), $(Dates.format(now(), "yyyy-mm-dd HH:MM"))\n")
    println(io, "Julia $(VERSION), threads $(Threads.nthreads()), CPU $(Sys.cpu_info()[1].model)",
            HAVE_GPU ? ", GPU $(Base.invokelatest(() -> CUDA.name(CUDA.device())))" : ", no GPU", "\n")
    println(io, "Engine: $(STEPS[1]) and $(STEPS[2]) steps at dt = 2e-3 on a flowing Gaussian fireball, two snapshots, particles injected (no sampler). ",
            "Marginal cost = (t₄₀₀ − t₁₀₀)/(300·N); fixed = per-call overhead (allocation, spline build, upload/download, cleanup).\n")
    foreach(r -> println(io, r), rows); println(io); println(io, "## Host-side phases\n"); foreach(r -> println(io, r), host)
end
println("\nwritten: $out")
