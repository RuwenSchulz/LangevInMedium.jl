# ==============================================================================================
# bitid_production_paths.jl — bit-identity over the switches the PRODUCTION cells run (2026-09-24).
#
# `regression_corpus.jl` spans the kwarg surface up to 0.2.3, but not `proper_time_kicks`,
# `bjorken_frame = :lab`, the cold-comoving (DsT == 0) kernel or the GPU freeze-out latch — the
# switches every LP1 / O+O / AH / KD cell runs with. This hashes seeded CPU AND GPU runs through
# them on a RADIAL background (the only kind any DPM cell uses). GPU runs are bit-reproducible
# under `CUDA.seed!` (+ `Random.seed!` for the host-side draws); checked twice in a row first.
#
# Written for the 2+1D flow-direction fix (CHANGELOG "Unreleased / Fixed"; IDEAS I-64): the
# baseline was generated at c1ca501, BEFORE the edit, and the fixed tree reproduces all 21 hashes.
# That is the evidence behind the DPM attestation c1ca501 <-> the fix commit.
#
#   write:  BITID_MODE=write julia --project=Julia Julia/LangevInMedium.jl/test/bitid_production_paths.jl
#   check:                   julia --project=Julia Julia/LangevInMedium.jl/test/bitid_production_paths.jl
#
# ⚠ The GPU hashes are specific to the device, driver and CUDA.jl build (written on an RTX 5070,
# CUDA.jl 5.11.2, Julia 1.12.6). On another GPU, or after a toolchain update, the GPU lines will
# not match for reasons that are not the engine: regenerate at the OLD commit first, then check
# the new one. The CPU lines are portable across machines with the same Julia version.
# ==============================================================================================
using CUDA, LangevInMedium, Random, SHA, Printf

const OUT = joinpath(@__DIR__, "bitid_production_paths_baseline.txt")
const MODE = get(ENV, "BITID_MODE", "check")
const N = 8_000

function flow_bg(; t0 = 0.4, tf = 1.0)
    xg = collect(0.0:0.25:40.0); tg = collect(t0:0.1:(t0 + tf + 0.5))
    Tf = [max(0.12, 0.45 * exp(-r^2 / 60) * (t0 / τ)^(1 / 3)) for r in xg, τ in tg]
    Vf = [0.6 * r / (r + 4.0) * min(1.0, τ / 2.0) for r in xg, τ in tg]
    xg, tg, Tf, Vf
end

const PROD = (; momentum_dimensions = 3, bjorken_redshift = true, bjorken_frame = :lab, proper_time_kicks = true)
const CASES = [
    ("prod_ou_lab",       PROD, 0.11634),
    ("prod_ou_lrf",       merge(PROD, (; bjorken_frame = :lrf)), 0.11634),
    ("prod_linear",       merge(PROD, (; DsT_linear = true)), 0.11634),
    ("ptk_pdim2",         (; proper_time_kicks = true), 0.11634),
    ("prod_rta",          merge(PROD, (; collision_mode = :rta)), 0.11634),
    ("rta_ptk_pdim2",     (; collision_mode = :rta, proper_time_kicks = true), 0.11634),
    ("cold_comoving",     (; momentum_dimensions = 3, bjorken_redshift = true, bjorken_frame = :lab), 0.0),
    ("cold_comoving_gal", (; relativistic = false), 0.0),
    ("free_streaming",    merge(PROD, (; collision_mode = :none)), 0.11634),
    ("ptk_off_default",   (;), 0.11634),
]

digest(xs...) = bytes2hex(sha256(reduce(vcat, [reinterpret(UInt8, vec(Float64.(x))) for x in xs])))

function run_case(backend, kw, DsT, seed; fo = false)
    xg, tg, Tf, Vf = flow_bg()
    Random.seed!(seed); backend isa GPUBackend && CUDA.seed!(seed)
    pdim = get(kw, :momentum_dimensions, 2)
    x0 = 3.0 .* randn(2, N); p0 = 0.8 .* randn(pdim, N)
    extra = fo ? (; freezeout_capture = true, Tfo = 0.2) : (;)
    res = simulate_ensemble_bulk(backend, collect(0.0:0.5:20.0), collect(0.05:0.1:8.0), zeros(80, 41),
        Tf, Vf, (xg, tg); N_particles = N, Δt = 2e-3, initial_time = 0.4, final_time = 1.0,
        save_interval = 0.2, m = 1.5, DsT = DsT, dimensions = 2, x_init = x0, p_init = p0, kw..., extra...)
    fo && return digest(res...)
    t, mom, pos = res[1], res[2], res[3]
    digest(t, mom..., pos...)
end

lines = String[]
for (bk, bname) in ((CPUBackend(), "cpu"), (GPUBackend(), "gpu"))
    for (k, (name, kw, DsT)) in enumerate(CASES)
        bname == "gpu" && get(kw, :collision_mode, :langevin) ∉ (:langevin, :rta, :none) && continue
        h = try run_case(bk, kw, DsT, 1000 + k) catch e; "ERROR " * sprint(showerror, e)[1:min(end, 120)] end
        push!(lines, "$bname $name $h")
    end
end
# the GPU freeze-out latch on the production configuration
push!(lines, "gpu prod_ou_lab_focapture " * run_case(GPUBackend(), PROD, 0.11634, 2001; fo = true))

if MODE == "write"
    open(OUT, "w") do io
        println(io, "# bitid baseline, unix time ", round(Int, time()), ", LangevInMedium HEAD pre-fix (c1ca501)")
        foreach(l -> println(io, l), lines)
    end
    foreach(println, lines); println("wrote ", OUT)
else
    base = Dict{String,String}()
    for l in eachline(OUT)
        startswith(l, "#") && continue
        a = split(l, ' '; limit = 3); base[a[1] * " " * a[2]] = a[3]
    end
    nfail = 0
    for l in lines
        a = split(l, ' '; limit = 3); key = a[1] * " " * a[2]
        ok = get(base, key, "MISSING") == a[3] && !startswith(a[3], "ERROR")
        ok || (nfail += 1)
        @printf("%-4s %-28s %s\n", ok ? "OK" : "DIFF", key, ok ? a[3][1:16] : "base " * get(base, key, "MISSING")[1:min(end, 16)] * "  now " * a[3][1:min(end, 16)])
    end
    println(nfail == 0 ? "BIT-IDENTICAL: all $(length(lines)) hashes match the pre-fix baseline" : "FAIL: $nfail of $(length(lines)) differ")
    exit(nfail == 0 ? 0 : 1)
end
