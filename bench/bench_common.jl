# Shared helpers for the LangevInMedium bench suite. `include`d by the bench_*.jl scripts.
using Random, Statistics, LinearAlgebra, Printf
using LangevInMedium

LangevInMedium.LV_TAUN_SCALE[] == 1.0 || error("LV_TAUN_SCALE = $(LangevInMedium.LV_TAUN_SCALE[]) ≠ 1 — benches are defined at scale 1")

const HBARC = LangevInMedium.GevInvTofm          # the package's own ħc, not a private copy
const RESULTS = joinpath(@__DIR__, "results")
mkpath(RESULTS)

"Try to load CUDA; return true iff the GPU path is usable."
function gpu_available()
    get(ENV, "LIM_NOGPU", "0") == "1" && return false
    try
        @eval using CUDA
        Base.invokelatest(() -> CUDA.functional())
    catch
        false
    end
end

"Uniform bath at T with zero flow, wide enough that nothing leaves it."
function box_fields(T; t0 = 0.0, tf = 1.0, rmax = 400.0)
    xg = collect(0.0:0.5:rmax); tg = collect(t0:0.25:(t0 + tf + 1))
    xg, tg, fill(T, length(xg), length(tg)), zeros(length(xg), length(tg))
end

"""
One run through the PUBLIC entry point on a prescribed (T, v) background with injected particles.
Returns (times, momenta_snapshots, position_snapshots). Always seeded.
"""
function run_fields(backend, xg, tg, Tf, Vf; M, DsT, N, dt, tfinal, save, x0, p0, t0 = first(tg),
                    seed = 1, kw...)
    Random.seed!(seed)
    rg = collect(0.0:0.5:20.0); pg = collect(0.05:0.1:8.0); dens = ones(length(pg), length(rg))
    Base.invokelatest(simulate_ensemble_bulk, backend, rg, pg, dens, Tf, Vf, (xg, tg);
        x_init = x0, p_init = p0, N_particles = N, Δt = dt, initial_time = t0,
        final_time = t0 + tfinal, save_interval = save, m = M, DsT = DsT, dimensions = 2,
        Tfo = 0.0, kw...)
end

"Isotropic d-dimensional Jüttner sample (rejection), shape (d, N)."
function juttner_sample(rng, M, T, d, N)
    pmax = 14 * sqrt(M * T + T^2)
    w(p) = p^(d - 1) * exp(-(sqrt(p^2 + M^2) - M) / T)
    wmax = maximum(w, range(1e-6, pmax; length = 4000))
    P = zeros(d, N)
    for i in 1:N
        p = 0.0
        while true
            p = pmax * rand(rng); rand(rng) * wmax <= w(p) && break
        end
        n = randn(rng, d); n ./= norm(n); P[:, i] .= p .* n
    end
    P
end

"⟨g(p)⟩ in the d-dimensional Jüttner at (M,T) by quadrature."
function jmean(g, M, T, d)
    w(p) = p^(d - 1) * exp(-(sqrt(p^2 + M^2) - M) / T)
    num, _ = quadgk(p -> g(p) * w(p), 0, Inf; rtol = 1e-10)
    den, _ = quadgk(w, 0, Inf; rtol = 1e-10)
    num / den
end

# PASS/FAIL bookkeeping — top-level globals, mutated through functions only (soft-scope trap).
const GATES = Tuple{Bool,String}[]
function gate!(ok::Bool, label::AbstractString)
    push!(GATES, (ok, String(label)))
    println(ok ? "  PASS " : "  FAIL ", label)
    ok
end
function finish!(name)
    nf = count(!first, GATES)
    println(nf == 0 ? "\n[PASS] $name: $(length(GATES)) gates" : "\n[FAIL] $name: $nf of $(length(GATES)) gates failed")
    nf == 0 || exit(1)
end
nonfinite(A) = count(!isfinite, A)
fmt(x; d = 4) = @sprintf("%.*f", d, x)
