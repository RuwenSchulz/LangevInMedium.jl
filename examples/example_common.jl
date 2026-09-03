# ==============================================================================================
# example_common.jl — shared background builders for the examples. `include`d by each of them.
#
# Nothing here is part of the package: it is the boilerplate a driver would otherwise carry, kept
# in one place so the examples themselves are only about the ENGINE. Two backgrounds:
#
#   uniform_bath(T)     a box at fixed T with no flow — the setting every closed form lives in
#   bjorken_fireball()  a cooling, radially expanding fireball with a freeze-out floor — the shape
#                       of a real hydro output, without needing one
#
# Both return `(xgrid, tgrid, T_field, v_field)` in the layout the engine wants:
# `T_field[i, j] = T(xgrid[i], tgrid[j])` in GeV, `v_field` the radial flow velocity in units of c.
# ==============================================================================================
using LangevInMedium, Random, Statistics, Printf, QuadGK, Bessels

const HBARC = LangevInMedium.GevInvTofm      # the package's own ħc, never a private copy

"Uniform bath at `T` GeV with zero flow, wide enough in r and τ that nothing leaves the table."
function uniform_bath(T; τ0 = 0.0, τf = 10.0, rmax = 300.0)
    xg = collect(0.0:0.5:rmax)
    tg = collect(τ0:0.25:(τf + 1.0))
    (xg, tg, fill(T, length(xg), length(tg)), zeros(length(xg), length(tg)))
end

"""
    bjorken_fireball(; T0, τ0, τf, σ, vmax, Tfo, rmax)

A Bjorken-like cooling fireball: `T(r, τ) = max(T_floor, T0·(τ0/τ)^{1/3}·exp(−r²/2σ²))` with a radial
flow that saturates, `v(r, τ) = vmax·tanh(r/σ)·min(1, τ/τ_rise)`. Not a hydro solution — it is the
right SHAPE (ideal Bjorken cooling, a Gaussian transverse profile, flow building up over ≈ 2 fm)
for an example that must run in seconds. Swap in a real `T`/`v` table and nothing else changes.

⚠ `T_floor` is the temperature the table falls to OUTSIDE the fireball, and it must sit strictly
BELOW the freeze-out temperature a driver is going to look for. Flooring at `Tfo` itself puts the
entire outside region exactly ON the freeze-out isotherm, so `T <= Tfo` is true everywhere there
and every quark produced outside the hot core "freezes out" at τ₀ — which is arguably right, but it
also makes a `<` test and a `<=` test disagree, and it made two of these examples report ⟨τ_fo⟩ of
2.5 fm and 8.2 fm for the same physics. Keep the floor cold.

`rmax` is deliberately well past the fireball: the engine CLAMPS the background at the table edge,
so a particle that leaves the table keeps being dragged at the rim `T` and `v` forever. It warns
when that happens (0.2.3) — if you see that warning, widen `rmax` rather than ignoring it.
"""
function bjorken_fireball(; T0 = 0.50, τ0 = 0.4, τf = 15.0, σ = 4.0, vmax = 0.65,
                            T_floor = 0.05, rmax = 26.0, τ_rise = 2.0, nr = 209)
    xg = collect(range(0.0, rmax; length = nr))
    tg = collect(τ0:0.05:τf)
    Tf = [max(T_floor, T0 * (τ0 / τ)^(1 / 3) * exp(-r^2 / (2σ^2))) for r in xg, τ in tg]
    Vf = [vmax * tanh(r / σ) * min(1.0, τ / τ_rise) for r in xg, τ in tg]
    (xg, tg, Tf, Vf)
end

"""
    fonll_density(r_grid, p_grid; σ = 3.0, n = 3.1, p_ref = 2.1)

A separable initial phase-space density `f[p_index, r_index]` of the shape the engine's sampler
expects: a steeply falling charm `p_T` spectrum `(1 + (p/p_ref)²)^{−n}` times a Gaussian transverse
profile. **Use a UNIFORM p grid** — the sampler integrates on the nodes you hand it (trapezoid
since 0.2.3), and a uniform grid is what every driver in the tree builds.
"""
function fonll_density(r_grid, p_grid; σ = 3.0, n = 3.1, p_ref = 2.1)
    [(1 + (p / p_ref)^2)^(-n) * exp(-r^2 / (2σ^2)) for p in p_grid, r in r_grid]
end

"Isotropic d-dimensional Jüttner sample (rejection), shape (d, N) — an INDEPENDENT reference."
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
        nvec = randn(rng, d); nvec ./= sqrt(sum(abs2, nvec)); P[:, i] .= p .* nvec
    end
    P
end

"⟨g(p)⟩ in the d-dimensional Jüttner at (M, T) by quadrature — the exact target."
function juttner_mean(g, M, T, d)
    w(p) = p^(d - 1) * exp(-(sqrt(p^2 + M^2) - M) / T)
    num, _ = quadgk(p -> g(p) * w(p), 0, Inf; rtol = 1e-10)
    den, _ = quadgk(w, 0, Inf; rtol = 1e-10)
    num / den
end

"Normalised histogram of `v` on `edges`; returns (centres, density)."
function hist(v, edges)
    c = zeros(length(edges) - 1)
    for x in v
        b = searchsortedlast(edges, x)
        1 <= b <= length(c) && (c[b] += 1)
    end
    ((edges[1:end-1] .+ edges[2:end]) ./ 2, c ./ max(sum(c) * step(edges), eps()))
end

"Set up Plots once, honouring LIM_NOPLOT=1. Returns true if plotting is on."
function plots_on()
    get(ENV, "LIM_NOPLOT", "0") == "1" && return false
    ENV["GKSwstype"] = get(ENV, "GKSwstype", "100")
    @eval Main begin
        using Plots
        gr()
        default(; fontfamily = "sans-serif", framestyle = :box, grid = true, legend = :best,
                dpi = 150, lw = 2, ms = 4, size = (620, 440),
                left_margin = 6Plots.mm, bottom_margin = 6Plots.mm, top_margin = 3Plots.mm)
    end
    true
end

const EXFIG = joinpath(@__DIR__, "figures")
savefig_ex(p, name) = (mkpath(EXFIG); Main.savefig(p, joinpath(EXFIG, name));
                       println("  figure → examples/figures/$name"))
