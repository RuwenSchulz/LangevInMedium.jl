#!/usr/bin/env julia
# ==============================================================================================
# regression_corpus.jl — the bit-identity gate for LangevInMedium.jl.
#
# Ten small, seeded runs through the PUBLIC entry point `simulate_ensemble_bulk`, spanning the
# kwarg surface that production consumers use (default, Galilean, 3 momentum rows + Bjorken
# redshift, DsT_quad, RTA, position diffusion + reflecting wall, the FONLL sampler in both its
# spatial modes, the radial dimensions=1 mode, a flowing background). For each run the CPU
# momentum and position histories are SHA-256 hashed; a cleanup that claims to be bit-identical
# must reproduce every hash. The GPU path is unseedable (CURAND), so it is held to the CPU
# ensemble moments at `GPU_RTOL`, plus one exact check: the sampler runs on the host in both
# paths, so the t0 snapshot BEFORE the lab boost is the same matrix — the post-boost first
# snapshot must agree to float rounding.
#
#   write the baseline:   LIM_CORPUS_WRITE=1 julia --project=Julia Julia/LangevInMedium.jl/test/regression_corpus.jl
#   check against it:                        julia --project=Julia Julia/LangevInMedium.jl/test/regression_corpus.jl
#   skip the GPU half:    LIM_CORPUS_NOGPU=1 ...
#
# Baseline file: test/regression_corpus_baseline.txt (plain text, committed). Regenerate it ONLY
# for a deliberate, documented change of the default dynamics — and say so in CHANGELOG.md.
# ==============================================================================================
using Random, Statistics, SHA, Printf, LinearAlgebra
using LangevInMedium

const BASE = joinpath(@__DIR__, "regression_corpus_baseline.txt")
const WRITE = get(ENV, "LIM_CORPUS_WRITE", "0") == "1"
const NOGPU = get(ENV, "LIM_CORPUS_NOGPU", "0") == "1"
const GPU_RTOL = 0.03     # unseeded CURAND: run-to-run scatter of ⟨p²⟩ at N=20k is ≈1 %; 2 % brushed the gate once
const GPU_PX_ATOL = 0.02     # ⟨p_x⟩ [GeV] — a zero-mean quantity, N=20k ⇒ SEM ≈ 0.007

LangevInMedium.LV_TAUN_SCALE[] == 1.0 || error("LV_TAUN_SCALE = $(LangevInMedium.LV_TAUN_SCALE[]) ≠ 1: the corpus is defined at scale 1")

# ── backgrounds ──────────────────────────────────────────────────────────────────────────────────
"Uniform bath T, zero flow. Grid wide enough that diffusing particles never leave it."
function box_bg(T; t0 = 0.0, tf = 1.0)
    xg = collect(0.0:0.5:200.0); tg = collect(t0:0.25:(t0 + tf + 1))
    xg, tg, fill(T, length(xg), length(tg)), zeros(length(xg), length(tg))
end
"Cooling Gaussian fireball with a radial flow that saturates at 0.6: T(r,τ), v(r,τ)."
function flow_bg(; t0 = 0.4, tf = 1.0)
    xg = collect(0.0:0.25:40.0); tg = collect(t0:0.1:(t0 + tf + 0.5))
    Tf = [max(0.12, 0.45 * exp(-r^2 / 60) * (t0 / τ)^(1 / 3)) for r in xg, τ in tg]
    Vf = [0.6 * r / (r + 4.0) * min(1.0, τ / 2.0) for r in xg, τ in tg]
    xg, tg, Tf, Vf
end
"Np × Nr FONLL-like density: Gaussian in r, p·exp(-m_T/T_init) in p (what the sampler expects)."
function fonll_density(rg, pg; m = 1.5, Tini = 0.5)
    [p * exp(-sqrt(p^2 + m^2) / Tini) * exp(-r^2 / 30) for p in pg, r in rg]
end

const M = 1.5; const DST = 0.11634; const N = 20_000; const DT = 2e-3; const SAVE = 0.2

# ── the corpus ───────────────────────────────────────────────────────────────────────────────────
# name => (background thunk, kwargs, seed). x_init/p_init are drawn inside `run` from the seed
# unless the case says `:sampler`, in which case the FONLL sampler is exercised.
const CASES = [
    ("default_box",      () -> box_bg(0.30; tf = 0.6),  (;), 101),
    ("galilean_flow",    () -> flow_bg(; tf = 0.6),      (; relativistic = false), 102),
    ("pdim3_redshift",   () -> flow_bg(; tf = 0.6),      (; momentum_dimensions = 3, bjorken_redshift = true), 103),
    ("dst_quad_flow",    () -> flow_bg(; tf = 0.6),      (; DsT_quad = true, DsT_Tref = 0.30), 104),
    ("dst_linear_flow",  () -> flow_bg(; tf = 0.6),      (; DsT_linear = true), 105),
    ("rta_flow",         () -> flow_bg(; tf = 0.6),      (; collision_mode = :rta), 106),
    ("posdiff_reflect",  () -> box_bg(0.25; tf = 0.6),   (; position_diffusion = true, reflecting_boundary = true), 107),
    ("sampler_cart",     () -> flow_bg(; tf = 0.4),      (; cartesian_spatial_sampling = true, sampler = true), 108),
    ("sampler_polar",    () -> flow_bg(; tf = 0.4),      (; cartesian_spatial_sampling = false, sampler = true), 109),
    ("radial_dim1",      () -> flow_bg(; tf = 0.4),      (; dimensions = 1, sampler = true, antithetic_momenta = true), 110),
]

function run(backend, bg, kw, seed)
    xg, tg, Tf, Vf = bg()
    t0 = first(tg); tf = t0 + (kw == (;) || !haskey(kw, :sampler) ? 0.6 : 0.4)
    kw = Base.structdiff(kw, (; sampler = nothing))            # strip our own marker
    dims = get(kw, :dimensions, 2)
    Random.seed!(seed)
    rg = collect(0.0:0.5:20.0); pg = collect(0.05:0.1:8.0)
    dens = fonll_density(rg, pg)
    ic = haskey(kw, :cartesian_spatial_sampling) || dims == 1 ? (;) :
         (; x_init = 3.0 .* randn(2, N), p_init = 0.8 .* randn(2, N))
    simulate_ensemble_bulk(backend, rg, pg, dens, Tf, Vf, (xg, tg);
        N_particles = N, Δt = DT, initial_time = t0, final_time = tf, save_interval = SAVE,
        m = M, DsT = DST, Tfo = 0.0, dimensions = dims, ic..., kw...)
end

sha(A) = bytes2hex(sha256(reinterpret(UInt8, vec(collect(Float64, A)))))
function moments(mom, pos)
    m = mom[end]; x = pos[end]
    (p2 = mean(sum(abs2, m; dims = 1)), px = mean(m[1, :]),
     pz2 = size(m, 1) == 3 ? mean(m[3, :] .^ 2) : 0.0,
     x2 = mean(sum(abs2, x; dims = 1)), r = mean(sqrt.(sum(abs2, x; dims = 1))))
end
fmtm(mo) = join((@sprintf("%.10e", v) for v in values(mo)), " ")

# ── CPU: hashes ──────────────────────────────────────────────────────────────────────────────────
lines = String[]; cpu = Dict{String,Any}()
for (name, bg, kw, seed) in CASES
    t = @elapsed (tt, mom, pos) = run(CPUBackend(), bg, kw, seed)
    mo = moments(mom, pos)
    cpu[name] = (; mom, pos, mo)
    push!(lines, "$name cpu $(sha(cat(mom...; dims = 3))) $(sha(cat(pos...; dims = 3))) $(fmtm(mo))")
    @printf("  %-18s cpu %6.2fs  ⟨p²⟩=%.5f ⟨x²⟩=%.4f\n", name, t, mo.p2, mo.x2)
end

# ── GPU: moments + first-snapshot agreement ──────────────────────────────────────────────────────
gpu_ok = !NOGPU && (try; @eval using CUDA; Base.invokelatest(() -> CUDA.functional()); catch; false; end)
gpu = Dict{String,Any}()
if gpu_ok
    for (name, bg, kw, seed) in CASES
        t = @elapsed (tt, mom, pos) = Base.invokelatest(run, GPUBackend(), bg, kw, seed)
        mo = moments(mom, pos)
        d1 = maximum(abs, mom[1] .- cpu[name].mom[1]) + maximum(abs, pos[1] .- cpu[name].pos[1])
        gpu[name] = (; mo, d1)
        push!(lines, "$name gpu $(@sprintf("%.3e", d1)) $(fmtm(mo))")
        @printf("  %-18s gpu %6.2fs  ⟨p²⟩=%.5f ⟨x²⟩=%.4f  |snap1−cpu|=%.2e\n", name, t, mo.p2, mo.x2, d1)
    end
else
    println("  (GPU half skipped)")
end

# ── write or check ───────────────────────────────────────────────────────────────────────────────
if WRITE
    open(BASE, "w") do io
        println(io, "# LangevInMedium regression corpus — julia $(VERSION), N=$N, dt=$DT; see regression_corpus.jl")
        foreach(l -> println(io, l), lines)
    end
    println("baseline written: $BASE ($(length(lines)) lines)")
else
    isfile(BASE) || error("no baseline at $BASE — run with LIM_CORPUS_WRITE=1 first")
    base = Dict{Tuple{String,String},Vector{String}}()
    for l in eachline(BASE)
        startswith(l, "#") && continue
        f = split(l); base[(f[1], f[2])] = f[3:end]
    end
    global nfail = 0
    for (name, _, _, _) in CASES
        b = base[(name, "cpu")]; c = cpu[name]
        hm, hp = sha(cat(c.mom...; dims = 3)), sha(cat(c.pos...; dims = 3))
        ok = (b[1] == hm) && (b[2] == hp)
        println(ok ? "  PASS " : "  FAIL ", name, " cpu bit-identical", ok ? "" : "   (momenta $(b[1]==hm), positions $(b[2]==hp))")
        global nfail += !ok
        if gpu_ok
            bm = parse.(Float64, b[3:end]); gm = collect(values(gpu[name].mo))
            # fields: p2 px pz2 x2 r — px is a mean that sits at zero (absolute tolerance), the rest relative
            rel = [i == 2 ? abs(gm[i] - bm[i]) / GPU_PX_ATOL : abs(gm[i] - bm[i]) / max(abs(bm[i]), 1e-12) / GPU_RTOL for i in eachindex(bm)]
            okg = all(rel .< 1) && gpu[name].d1 < 1e-6
            println(okg ? "  PASS " : "  FAIL ", name, " gpu moments within $(GPU_RTOL) of CPU baseline, snapshot-1 exact",
                    okg ? "" : "   (worst tolerance fraction " * @sprintf("%.2f", maximum(rel)) * ", d1 " * @sprintf("%.2e", gpu[name].d1) * ")")
            global nfail += !okg
        end
    end
    println(nfail == 0 ? "\n[PASS] regression corpus: all $(length(CASES)) cases" : "\n[FAIL] regression corpus: $nfail failure(s)")
    nfail == 0 || exit(1)
end
