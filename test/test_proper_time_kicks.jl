#=
   test_proper_time_kicks.jl — the "boost case" gate for the proper-time kick fix (2026-08-25).

   A uniformly flowing medium in equilibrium must carry ZERO Landau diffusion current for any
   LRF-isotropic f.  The undilated lab-Δt kick violates this: its stationary state is
   f_J/(γ(1+v v_r)), measured 2026-08-04 as ν^r/n = −0.071 at u^r = 0.5, T = 300 MeV, charm
   (AttractorMomentum FIGURE_REGISTRY "boost case"; quadrature −0.075).  With
   `proper_time_kicks = true` the same configuration must give ν^r/n ≈ 0.

   Gates:  P1  flag OFF reproduces the recorded artifact (−0.075 ± 20%)
           P2  flag ON kills it (|ν^r/n| < 0.012)
           P3  the LRF momentum variance is T to 3% in BOTH modes (the fix must not touch widths)

   Run: julia --project=Julia Julia/LangevInMedium.jl/test/test_proper_time_kicks.jl
=#

import Pkg; Pkg.activate(normpath(joinpath(@__DIR__, "..", "..")); io = devnull)
using LangevInMedium, Random, Statistics, Printf

const M = 1.5; const T0 = 0.300; const UR = 0.5
const V = UR / sqrt(1 + UR^2); const γV = sqrt(1 + UR^2)
const DST = 0.01                       # drag time ≈ 0.03 fm/c ⇒ fully stationary long before t_f
const N = 400_000

function sample_lrf_juttner(rng, N)    # 2-D Jüttner, rejection in |k|
    p = zeros(2, N); kmax = 6.0
    fmax = maximum(k * exp(-(sqrt(k^2 + M^2) - M) / T0) for k in 0:0.01:kmax)
    i = 1
    while i <= N
        k = kmax * rand(rng); φ = 2π * rand(rng)
        if rand(rng) * fmax < k * exp(-(sqrt(k^2 + M^2) - M) / T0)
            p[1, i] = k * cos(φ); p[2, i] = k * sin(φ); i += 1
        end
    end
    p
end

function run_case(; proper::Bool)
    rng = MersenneTwister(20260825)
    xgrid = collect(0.0:0.5:30.0); tgrid = collect(0.0:0.1:2.0)
    Tf = fill(T0, length(xgrid), length(tgrid)); Vf = fill(V, length(xgrid), length(tgrid))
    # annulus r ∈ [3, 5], uniform in area
    x = zeros(2, N)
    for i in 1:N
        r = sqrt(9 + 16 * rand(rng)); φ = 2π * rand(rng)
        x[1, i] = r * cos(φ); x[2, i] = r * sin(φ)
    end
    p = sample_lrf_juttner(rng, N)
    Random.seed!(4242)
    _, moms, poss = simulate_ensemble_bulk(CPUBackend(), nothing, nothing, nothing,
        Tf, Vf, (xgrid, tgrid);
        N_particles = N, Δt = 0.002, initial_time = 0.0, final_time = 1.0,
        save_interval = 0.5, m = M, DsT = DST, dimensions = 2,
        x_init = x, p_init = p, proper_time_kicks = proper)
    pm = moms[end]; xm = poss[end]
    # the production estimator on one global bin: Jτ ∝ Σ1, J^r ∝ Σ (p·r̂)/E_lab
    s_v = 0.0; nn = 0
    for i in 1:N
        r = sqrt(xm[1, i]^2 + xm[2, i]^2); r < 1e-9 && continue
        E = sqrt(pm[1, i]^2 + pm[2, i]^2 + M^2)
        s_v += (pm[1, i] * xm[1, i] + pm[2, i] * xm[2, i]) / (E * r)
        nn += 1
    end
    vbar = s_v / nn
    # exact landau_project! algebra: n = u^τ⟨1⟩ − u^r⟨v⟩ ; ν^r = ⟨v⟩ − n·u^r
    ur = γV * V
    nL2 = γV - ur * vbar
    νr = vbar - nL2 * ur
    # LRF width: deboost each momentum with the fluid boost and measure the radial variance
    s2 = 0.0
    for i in 1:N
        r = sqrt(xm[1, i]^2 + xm[2, i]^2); r < 1e-9 && continue
        rx, ry = xm[1, i] / r, xm[2, i] / r
        pr = pm[1, i] * rx + pm[2, i] * ry; pt = -pm[1, i] * ry + pm[2, i] * rx
        E = sqrt(pm[1, i]^2 + pm[2, i]^2 + M^2)
        prs = γV * (pr - V * E)
        s2 += (prs^2 + pt^2) / 2
    end
    (; w = νr / nL2, Teff = s2 / nn / M)   # nonrel width proxy ⟨k²⟩/2 per dof ≈ M·T_eff
end

function main()
    println("boost-case gate: T = $(T0), u^r = $(UR) (v = $(round(V, digits = 4))), DsT = $(DST), N = $N")
    off = run_case(proper = false)
    on  = run_case(proper = true)
    ok = true
    g(nm, c, det) = (println("  ", nm, "  ", c ? "PASS" : "FAIL", "  ", det); ok &= c)
    g("P1 lab-Δt kick reproduces the artifact", abs(off.w - (-0.075)) < 0.015,
      @sprintf("ν^r/n = %+.4f (recorded −0.071, quadrature −0.075)", off.w))
    g("P2 proper-time kick kills it", abs(on.w) < 0.012, @sprintf("ν^r/n = %+.4f", on.w))
    g("P3 widths untouched", abs(off.Teff / on.Teff - 1) < 0.03,
      @sprintf("⟨k²⟩/2M: off %.4f  on %.4f", off.Teff, on.Teff))
    println(ok ? "ALL GATES PASS" : "GATE FAILURE")
    exit(ok ? 0 : 1)
end

main()
