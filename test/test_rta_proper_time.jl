#=
   test_rta_proper_time.jl — the RATE gate for proper-time kicks in the BGK / RTA collision.
   Written 2026-08-29; REWRITTEN 2026-08-30 after the estimator, not the operator, turned out to be
   what the first version was measuring.  See "THE RETIRED ANOMALY" at the bottom.

   Why this is NOT the OU gate (test_proper_time_kicks.jl).  That gate measures a STATIONARY-STATE
   artifact: the undilated OU kick relaxes to f_J/(γ(1+v v_r)) and so carries a spurious Landau
   current.  BGK has no such artifact — every collision re-draws an ISOTROPIC LRF Jüttner, and on a
   uniform flow the LRF momentum is frozen between collisions, so the stationary state is the exact
   Jüttner either way.  What the dilation changes in BGK is the RATE, which is the whole point of the
   operator: τ_n is a comoving relaxation time, so the elapsed time that divides it must be comoving.

   THE OBSERVABLE, and why it is a FIXED axis.  Seed a common momentum offset k0 along the LAB x
   axis on top of a Jüttner and watch ⟨p_x⟩.  A pure BGK step keeps a particle's momentum with
   probability (1−P) and re-draws it isotropically with probability P, and an isotropic draw has
   ⟨p_x⟩ = 0 whatever the flow does, by azimuthal symmetry of the annulus.  So

        ⟨p_x⟩(t) / ⟨p_x⟩(0)  =  (1 − P)^n      exactly,   P = Δt·dil/τ_n,  n = t/Δt

   and P2 below CHECKS that the estimator is clean rather than assuming it: a collisionless control
   must leave ⟨p_x⟩ flat.  It does, to 1e-5.

   🔴 DO NOT PROJECT ON THE RADIAL DIRECTION.  ⟨k*_r⟩ taken against each particle's CURRENT r̂ has an
   ADDITIVE FLOOR: free streaming converts the annulus's finite extent into radial momentum
   anisotropy, so the collided population — re-isotropized and then streaming again — keeps
   regenerating a positive ⟨k*_r⟩ that the decaying signal falls into.  That is what the first
   version of this gate measured. See the bottom.

   Gates:  P1  v = 0 ⇒ the flag is EXACTLY inert (bit-identical ensembles, same seed).
               `dil` is the literal 1.0 there and `Δt*1.0/τn == Δt/τn` in floating point.
           P2  the estimator is clean: a collisionless control leaves ⟨p_x⟩ flat.
           P3  flag OFF ⇒ ⟨p_x⟩ decays as (1 − Δt/τ_n)^n. THE OPERATOR IS CORRECT.
           P4  flag ON, heavy limit ⇒ the effective rate is 1/γ of it. THE DILATION IS CORRECT.
           P5  the EQUILIBRIUM LRF width is unchanged by the flag (a rate fix must not move widths).

   Run: julia --project=Julia Julia/LangevInMedium.jl/test/test_rta_proper_time.jl
=#

import Pkg; Pkg.activate(normpath(joinpath(@__DIR__, "..", "..")); io = devnull)
using LangevInMedium, Random, Statistics, Printf
using SpecialFunctions: besselk

const M  = 1.5
const UR = 0.5
const V  = UR / sqrt(1 + UR^2)
const γV = sqrt(1 + UR^2)
const N  = 200_000
const DT = 0.002
const TF = 0.30

# The heavy bath: T ≪ M ⇒ k*/E* → 0 ⇒ dil → 1/γ INDEPENDENT of momentum, which is what makes P4's
# prediction a closed form rather than a quadrature. DsT is then chosen to put τ_n back at the same
# ≈0.25 fm (τ_n ∝ DsT·M/T²), because the gate is about the dilation and the rate it divides is free.
const T_HEAVY   = 0.010
const DST_HEAVY = 8.4e-5
# The production-scale bath, for P5's widths.
const T0  = 0.300
const DST = 0.05

taun(T, dst) = (dst / T) * (M / T) * besselk(3, M / T) / besselk(2, M / T) * 0.1973269804

"""
    run_case(; v, dst, T, k0, tf, proper, n) -> (d, err, width)

Evolve a BGK ensemble on a uniform bath with uniform radial flow `v` and return the decay of the
FIXED-AXIS mean ⟨p_x⟩, its statistical error, and the LRF width ⟨k*²⟩/2M.

🔴 `p_init` is an LRF momentum by the engine's contract (`simulate_cpu.jl` completes p_z there and
then calls `kernel_boost_to_lab_frame_cpu!` BEFORE writing snapshot 1). Do not pre-boost it.
"""
function run_case(; v, dst, T, k0, tf = TF, proper::Bool, n::Int = N)
    rng = MersenneTwister(20260830)
    xgrid = collect(0.0:0.5:80.0); tgrid = collect(0.0:0.05:(tf + 0.5))
    Tf = fill(T, length(xgrid), length(tgrid)); Vf = fill(v, length(xgrid), length(tgrid))
    x = zeros(2, n); p = zeros(2, n)
    kmax = max(6.0 * sqrt(2 * M * T), 0.05)
    fmax = maximum(k * exp(-(sqrt(k^2 + M^2) - M) / T) for k in 0:(kmax/2000):kmax)
    for i in 1:n
        r = sqrt(9 + 16 * rand(rng)); θ = 2π * rand(rng)      # annulus r ∈ [3,5], uniform in area
        x[1, i] = r * cos(θ); x[2, i] = r * sin(θ)
        k = 0.0
        while true
            k = kmax * rand(rng)
            rand(rng) * fmax < k * exp(-(sqrt(k^2 + M^2) - M) / T) && break
        end
        φ = 2π * rand(rng)
        p[1, i] = k * cos(φ) + k0                              # the offset is on the FIXED x axis
        p[2, i] = k * sin(φ)
    end
    Random.seed!(2718)
    _, moms, poss = simulate_ensemble_bulk(CPUBackend(), nothing, nothing, nothing,
        Tf, Vf, (xgrid, tgrid);
        N_particles = n, Δt = DT, initial_time = 0.0, final_time = tf, save_interval = tf,
        m = M, DsT = dst, dimensions = 2, collision_mode = :rta,
        x_init = x, p_init = p, proper_time_kicks = proper)
    px0 = sum(@view moms[1][1, :]) / n
    px1 = sum(@view moms[end][1, :]) / n
    err = std(@view moms[end][1, :]) / sqrt(n) / max(abs(px0), 1e-12)
    # LRF width, for P5 only
    γ = 1 / sqrt(1 - v * v); s2 = 0.0; cnt = 0
    pm = moms[end]; xm = poss[end]
    for i in 1:n
        r = sqrt(xm[1, i]^2 + xm[2, i]^2); r < 1e-9 && continue
        rx, ry = xm[1, i] / r, xm[2, i] / r
        pr = pm[1, i] * rx + pm[2, i] * ry; pt = -pm[1, i] * ry + pm[2, i] * rx
        E = sqrt(pm[1, i]^2 + pm[2, i]^2 + M^2)
        prs = γ * (pr - V_of(v) * E)
        s2 += (prs^2 + pt^2) / 2; cnt += 1
    end
    (d = px1 / px0, err = err, width = s2 / cnt / M, mom = moms[end])
end
V_of(v) = v

function main()
    n = round(Int, TF / DT)
    tn_h = taun(T_HEAVY, DST_HEAVY); P_h = DT / tn_h
    @printf("BGK proper-time gate: u^r = %.2f (v = %.4f, γ = %.4f), Δt = %.4f, n = %d, N = %d\n",
            UR, V, γV, DT, n, N)
    @printf("  heavy bath T = %.3f, DsT = %.2e ⇒ τ_n = %.4f fm, P = Δt/τ_n = %.5f\n",
            T_HEAVY, DST_HEAVY, tn_h, P_h)
    ok = Ref(true)                 # ⚠ Ref: `ok = false` inside a loop would make a NEW local
    g(nm, c, det) = (println("  ", c ? "PASS" : "FAIL", "  ", rpad(nm, 44), det); ok[] &= c)

    # P1 — v = 0: the flag must be EXACTLY inert
    z_off = run_case(v = 0.0, dst = DST, T = T0, k0 = 0.30, tf = 0.05, proper = false, n = 20_000)
    z_on  = run_case(v = 0.0, dst = DST, T = T0, k0 = 0.30, tf = 0.05, proper = true,  n = 20_000)
    g("P1 v=0 ⇒ flag exactly inert", z_off.mom == z_on.mom,
      "ensembles bit-identical: $(z_off.mom == z_on.mom)")

    # P2 — the estimator itself: collisionless (τ_n ≫ TF) must leave ⟨p_x⟩ flat
    ctl = run_case(v = V, dst = 50.0, T = T_HEAVY, k0 = 0.30, proper = false)
    g("P2 estimator clean (collisionless control)", abs(ctl.d - 1) < 0.005,
      @sprintf("⟨p_x⟩ decay %.5f ± %.5f — no additive floor", ctl.d, ctl.err))

    # P3 — the OPERATOR: flag off, the decay is the plain Bernoulli survival
    off = run_case(v = V, dst = DST_HEAVY, T = T_HEAVY, k0 = 0.30, proper = false)
    p_off = (1 - P_h)^n
    g("P3 OFF = (1−Δt/τ_n)^n  [the operator]", abs(off.d / p_off - 1) < 0.02,
      @sprintf("measured %.5f ± %.5f   predicted %.5f   ratio %.4f", off.d, off.err, p_off, off.d/p_off))

    # P4 — the DILATION: heavy limit ⇒ dil → 1/γ uniformly ⇒ the rate is exactly 1/γ of it
    on = run_case(v = V, dst = DST_HEAVY, T = T_HEAVY, k0 = 0.30, proper = true)
    p_on = (1 - P_h / γV)^n
    ratio = log(on.d) / log(off.d)
    g("P4 ON  = (1−Δt/(γτ_n))^n  [the dilation]", abs(ratio * γV - 1) < 0.01,
      @sprintf("rate ratio %.4f vs 1/γ %.4f (%+.2f%%);  decay %.5f vs predicted %.5f",
               ratio, 1/γV, 100*(ratio*γV - 1), on.d, p_on))

    # P5 — equilibrium widths, seeded AT equilibrium so no transient contaminates them
    e_off = run_case(v = V, dst = DST, T = T0, k0 = 0.0, proper = false)
    e_on  = run_case(v = V, dst = DST, T = T0, k0 = 0.0, proper = true)
    g("P5 equilibrium widths untouched", abs(e_off.width / e_on.width - 1) < 0.02,
      @sprintf("⟨k²⟩/2M: off %.4f  on %.4f  (T = %.3f)", e_off.width, e_on.width, T0))

    println("""
── THE RETIRED ANOMALY (2026-08-29 → resolved 2026-08-30) ────────────────────────────────────────
The first version of this gate projected on the CURRENT r̂ and reported that a BGK ensemble's ℓ=1
mode decays 12-18% slower than its own nominal 1/τ_n — flow-independently (rate·τ_n = 0.864 / 0.867 /
0.868 / 0.868 at u^r = 0 / 0.25 / 0.5 / 0.8) and not vanishing as Δt → 0 (1.18 / 1.11 / 1.13 at
Δt = 2e-3 / 1e-3 / 5e-4). It was the ESTIMATOR, and all three features are explained by that:
  · flow-independent, because it is geometry, not the boost — it is there at u^r = 0, where nothing
    is boosted at all and only the projection depends on r̂;
  · Δt-independent, because it is not a discretisation error;
  · NOT removable by a control, because it is ADDITIVE, not multiplicative. Measured: a collisionless
    control on the radial estimator grows ⟨k*_r⟩ by 7.2% out of nothing, and dividing by it leaves a
    residual that GROWS with the collision rate (1.05 at P = 0.0078, 1.64 at P = 0.0195) — the
    signature of a floor the decaying signal falls into, not of a biased rate.
The mechanism: free streaming converts the annulus's finite spatial extent into radial momentum
anisotropy, and the collided population — re-isotropized, then streaming again — keeps regenerating
it. On the fixed x axis, protected by the annulus's azimuthal symmetry, the control is flat to 1e-5
and the survival matches (1−Δt/τ_n)^n across a factor 5 in P and 2 in n. The operator was never
wrong. Convergence of P4 as the perturbation is taken into the heavy limit (v·k0/E* → 0):
    k0 = 2.00 (0.358) → −1.79% | 0.60 (0.166) → −0.68% | 0.30 (0.088) → −0.04%
─────────────────────────────────────────────────────────────────────────────────────────────────""")
    println(ok[] ? "ALL GATES PASS" : "GATE FAILURE")
    exit(ok[] ? 0 : 1)
end

main()
