#=
   test_rta_proper_time.jl — the RATE gate for proper-time kicks in the BGK / RTA collision
   (2026-08-29).

   Why this is NOT the OU gate (test_proper_time_kicks.jl).  That gate measures a STATIONARY-STATE
   artifact: the undilated OU kick relaxes to f_J/(γ(1+v v_r)) and so carries a spurious Landau
   current.  BGK has no such artifact — every collision re-draws an ISOTROPIC LRF Jüttner, and on a
   uniform flow the LRF momentum is constant between collisions, so the stationary state is the exact
   Jüttner whether or not Δt is dilated.  What the dilation changes in BGK is the RATE, which is the
   whole point of the operator: `τ_n` is a comoving relaxation time, so the elapsed time that divides
   it must be comoving too.  Undilated, the ℓ=1 mode decays at 1/τ_n per LAB time — too fast by the
   time dilation.

   Construction.  Uniform T and uniform radial flow.  Seed a common radial momentum offset on top of
   a Jüttner.  A pure BGK step keeps ⟨k*_r⟩ with probability (1−P) and resets it to an isotropic draw
   (mean 0) with probability P, so ⟨k*_r⟩ decays geometrically at rate P/Δt, read off the first and
   last snapshot.

   ⚠ NO SINGLE DECAY RATE EXISTS, AND THAT IS PHYSICS, NOT A BUG.  `dil = 1/(γ(1+v k*_r/E*))`
   depends on the particle's OWN k*_r: outward-moving particles (k*_r > 0, aligned with the flow)
   have the larger E_lab, hence the smaller dil, hence collide LESS.  The surviving ℓ=1 excess is
   therefore exactly the under-collided population, the decay is NOT a single exponential, and any
   ensemble-averaged dilation under-predicts how slowly it goes (0.62 measured against ⟨dil⟩ = 0.92).

   So the gate does not fit a rate — it predicts the WHOLE decay, exactly.  For BGK there are no
   forces between collisions and the flow is uniform, so an LRF momentum is frozen until its particle
   collides, after which it contributes 0 to ⟨k*_r⟩ on average.  Each particle therefore survives
   independently with probability (1−p)^n, p = Δt·dil(k*)/τ_n, and

        ⟨k*_r⟩(t) / ⟨k*_r⟩(0)  =  ⟨ k*_r (1 − Δt·dil(k*)/τ_n)^{t/Δt} ⟩ / ⟨ k*_r ⟩

   over the seeded distribution — a quadrature this file does independently, from the dilation
   FORMULA rather than from the code that implements it.  With the flag off, dil ≡ 1 and it collapses
   to the plain exponential.

   Gates:  P1  v = 0 ⇒ the flag is EXACTLY inert (bit-identical ensembles, same seed).
               `dil` is the literal 1.0 there and `Δt*1.0/τn == Δt/τn` in floating point.
           P2  the flag SLOWS the decay, and by more than the estimator's own scatter.
           P3  the two survival predictions are far apart, so the direction is resolvable.
           P4  the EQUILIBRIUM LRF width (seeded at k0 = 0) is unchanged by the flag.

   🔴 WHAT THIS GATE DOES NOT ESTABLISH, AND WHY IT IS REPORTED RATHER THAN ASSERTED.
   The survival quadrature above is printed next to the measurement, but it is NOT gated, because
   the measurement misses it ALREADY WITH THE FLAG OFF — where the prediction is the plain
   (1−Δt/τ_n)^n and the code path is byte-for-byte the pre-change one. Measured 2026-08-30:

       u^r  = 0.00  0.25  0.50  0.80   ⇒  rate·τ_n = 0.864 0.867 0.868 0.868   (flow-INDEPENDENT)
       Δt   = 2e-3  1e-3  5e-4         ⇒  decay/prediction = 1.18  1.11  1.13   (does NOT vanish)

   So the ℓ=1 mode of a BGK ensemble decays 12-18% SLOWER than its own nominal 1/τ_n, the effect is
   independent of the flow and does not go away as Δt → 0, and it therefore belongs to the shipped
   operator (or to this estimator), NOT to the dilation added here. A known contamination of the
   estimator is kinematic focusing: ⟨k*_r⟩ is taken against each particle's CURRENT r̂, and a
   free-streaming particle's trajectory becomes more radial as r grows, which lifts the
   un-collided population's k*_r and slows the apparent decay. That is O(few %) at this displacement
   and does not obviously account for 15%. UNRESOLVED — do not quote an RTA relaxation time against
   τ_n until it is.

   Run: julia --project=Julia Julia/LangevInMedium.jl/test/test_rta_proper_time.jl
=#

import Pkg; Pkg.activate(normpath(joinpath(@__DIR__, "..", "..")); io = devnull)
using LangevInMedium, Random, Statistics, Printf
using SpecialFunctions: besselk

const M   = 1.5
const T0  = 0.300         # production-scale bath
const T_HEAVY = 0.010     # T ≪ M ⇒ k*/E* → 0 ⇒ dil → 1/γ independent of momentum
const UR  = 0.5
const V   = UR / sqrt(1 + UR^2)
const γV  = sqrt(1 + UR^2)
const DST = 0.05          # τ_n ≈ 0.26 fm at T0 — several decays inside the window
# τ_n = D_s z K₃/K₂ ∝ DsT·M/T², so the production DsT at T_HEAVY would give τ_n ≈ 150 fm and a
# 225k-step run.  Pick DsT_HEAVY to land τ_n at the SAME ≈0.25 fm instead: the gate is about the
# DILATION, and the rate it divides is free.
const DST_HEAVY = 8.4e-5
const N   = 200_000
const DT  = 0.002
const TF  = 0.30          # short: keep every particle on the grid and the flow uniform

"""
Jüttner in the LOCAL REST FRAME with a common radial offset k0 (the ℓ=1 seed).

🔴 DO NOT BOOST THIS TO THE LAB. `p_init` is an LRF momentum by the engine's contract —
`simulate_cpu.jl` says so at the p_z completion ("sampled momenta are LRF momenta at this point, the
lab boost follows below") and then calls `kernel_boost_to_lab_frame_cpu!` BEFORE writing snapshot 1.
Boosting here too makes the ensemble a DOUBLE boost: measured ⟨k*_r⟩(0) = +0.92 = γv⟨E*⟩ instead of
k0, a spurious ℓ=1 mode of its own that BGK then relaxes, contaminating every rate read off it.
"""
function seed_momenta(rng, x, k0, T, N, v)
    p = zeros(2, N); kmax = max(6.0 * sqrt(2 * M * T), 0.05)
    fmax = maximum(k * exp(-(sqrt(k^2 + M^2) - M) / T) for k in 0:(kmax/2000):kmax)
    for i in 1:N
        k = 0.0
        while true
            k = kmax * rand(rng)
            rand(rng) * fmax < k * exp(-(sqrt(k^2 + M^2) - M) / T) && break
        end
        φ = 2π * rand(rng)
        r = sqrt(x[1, i]^2 + x[2, i]^2); rx, ry = x[1, i] / r, x[2, i] / r
        kr = k * cos(φ) + k0; kt = k * sin(φ)          # LRF, offset along r̂ — the engine boosts
        p[1, i] = kr * rx - kt * ry
        p[2, i] = kr * ry + kt * rx
    end
    p
end

function positions(rng, N)
    x = zeros(2, N)
    for i in 1:N
        r = sqrt(9 + 16 * rand(rng)); φ = 2π * rand(rng)   # annulus r ∈ [3,5], uniform in area
        x[1, i] = r * cos(φ); x[2, i] = r * sin(φ)
    end
    x
end

"⟨k*_r⟩ of an ensemble, in the LOCAL REST FRAME of a medium flowing at v."
function mean_kr_lrf(pm, xm, v)
    γ = 1 / sqrt(1 - v * v); s = 0.0; n = 0
    for i in 1:size(pm, 2)
        r = sqrt(xm[1, i]^2 + xm[2, i]^2); r < 1e-9 && continue
        rx, ry = xm[1, i] / r, xm[2, i] / r
        pr = pm[1, i] * rx + pm[2, i] * ry
        E  = sqrt(pm[1, i]^2 + pm[2, i]^2 + M^2)
        s += γ * (pr - v * E); n += 1
    end
    s / n
end

function lrf_width(pm, xm, v)
    γ = 1 / sqrt(1 - v * v); s = 0.0; n = 0
    for i in 1:size(pm, 2)
        r = sqrt(xm[1, i]^2 + xm[2, i]^2); r < 1e-9 && continue
        rx, ry = xm[1, i] / r, xm[2, i] / r
        pr = pm[1, i] * rx + pm[2, i] * ry; pt = -pm[1, i] * ry + pm[2, i] * rx
        E  = sqrt(pm[1, i]^2 + pm[2, i]^2 + M^2)
        prs = γ * (pr - v * E)
        s += (prs^2 + pt^2) / 2; n += 1
    end
    s / n / M
end

"""
    run_case(; proper, v, k0) -> (rate, width, moms, poss)

Evolve the BGK ensemble and read off the geometric decay rate of ⟨k*_r⟩ from the first and last
snapshot: ⟨k*_r⟩(t) = k0·exp(−rate·t) exactly for a pure BGK.
"""
function run_case(; proper::Bool, v::Float64, T::Float64 = T0, k0::Float64 = 0.30,
                  tf::Float64 = TF, dst::Float64 = DST, n::Int = N)
    rng = MersenneTwister(20260829)
    xgrid = collect(0.0:0.5:40.0); tgrid = collect(0.0:0.05:(tf + 0.5))
    Tf = fill(T, length(xgrid), length(tgrid)); Vf = fill(v, length(xgrid), length(tgrid))
    x = positions(rng, n); p = seed_momenta(rng, x, k0, T, n, v)
    Random.seed!(90210)
    _, moms, poss = simulate_ensemble_bulk(CPUBackend(), nothing, nothing, nothing,
        Tf, Vf, (xgrid, tgrid);
        N_particles = n, Δt = DT, initial_time = 0.0, final_time = tf,
        save_interval = tf, m = M, DsT = dst, dimensions = 2,
        collision_mode = :rta, x_init = x, p_init = p, proper_time_kicks = proper)
    # 🔴 normalise on the MEASURED initial LRF mean, never on k0: p_init is a LAB momentum, so the
    # boost to the rest frame changes it, and assuming k0 puts an additive offset in every rate.
    k_0   = mean_kr_lrf(moms[1],   poss[1],   v)
    k_end = mean_kr_lrf(moms[end], poss[end], v)
    (decay = k_end / k_0,
     rate = -log(max(abs(k_end), 1e-12) / abs(k_0)) / tf,
     width = lrf_width(moms[end], poss[end], v),
     mom = moms[end])
end

"""
    predicted_decay(v, T, k0, tf; dilate) -> ⟨k*_r⟩(tf)/⟨k*_r⟩(0)

The BGK survival quadrature, over the SAME seeded distribution `seed_momenta` draws: Jüttner in |k|,
uniform in φ, with k0 added to the radial component. τ_n is written out here from the closed form
(D_s z K₃/K₂, ×ħc) so this prediction shares no code with the engine under test.
"""
function predicted_decay(v, T, k0, tf; dilate::Bool)
    γ = 1 / sqrt(1 - v * v)
    z = M / T
    τn = (DST_OF[] / T) * z * besselk(3, z) / besselk(2, z) * 0.1973269804
    nstep = round(Int, tf / DT)
    kmax = max(6.0 * sqrt(2 * M * T), 0.05)
    num = 0.0; den = 0.0
    for k in (kmax/4000):(kmax/2000):kmax, φ in (π/512):(π/256):π
        w = k * exp(-(sqrt(k^2 + M^2) - M) / T)                 # 2-D measure k dk, pre-offset
        for sgn in (1.0, -1.0)
            kr = sgn * k * cos(φ) + k0; kt = k * sin(φ)
            Es = sqrt(kr^2 + kt^2 + M^2)
            dil = dilate ? 1.0 / (γ * (1 + v * kr / Es)) : 1.0
            pcol = clamp(DT * dil / τn, 0.0, 1.0)
            num += w * kr * (1 - pcol)^nstep
            den += w * kr
        end
    end
    num / den
end
const DST_OF = Ref(DST)          # the coupling the current case runs at

function main()
    @printf("BGK proper-time gate: u^r = %.2f (v = %.4f, γ = %.4f), DsT = %.3f, Δt = %.4f, N = %d\n",
            UR, V, γV, DST, DT, N)
    ok = Ref(true)                 # ⚠ Ref: `ok = false` inside a loop would make a NEW local
    g(nm, c, det) = (println("  ", c ? "PASS" : "FAIL", "  ", rpad(nm, 44), det); ok[] &= c)

    # P1 — v = 0: the flag must be EXACTLY inert
    z_off = run_case(proper = false, v = 0.0, n = 20_000, tf = 0.05)
    z_on  = run_case(proper = true,  v = 0.0, n = 20_000, tf = 0.05)
    g("P1 v=0 ⇒ flag exactly inert", z_off.mom == z_on.mom,
      "ensembles bit-identical: $(z_off.mom == z_on.mom)")

    # P2/P3 — direction and separability. The quadrature is PRINTED, not gated: see the header.
    DST_OF[] = DST
    off = run_case(proper = false, v = V)
    on  = run_case(proper = true,  v = V)
    p_off = predicted_decay(V, T0, 0.30, TF; dilate = false)
    p_on  = predicted_decay(V, T0, 0.30, TF; dilate = true)
    g("P2 the flag SLOWS the decay", on.decay > off.decay * 1.10,
      @sprintf("decay off %.4f → on %.4f  (+%.1f%%)", off.decay, on.decay, 100*(on.decay/off.decay - 1)))
    g("P3 predictions separable", abs(p_on / p_off - 1) > 0.30,
      @sprintf("survival quadrature: off %.4f  on %.4f  (ratio %.3f)", p_off, p_on, p_on / p_off))
    @printf("  ── reported, NOT gated ──  measured/predicted:  off %.3f   on %.3f\n",
            off.decay / p_off, on.decay / p_on)
    println("     both miss, and the OFF miss is on the UNCHANGED code path — see the header.")

    # P4 — equilibrium widths, seeded AT equilibrium so the k0 transient cannot contaminate them
    e_off = run_case(proper = false, v = V, k0 = 0.0)
    e_on  = run_case(proper = true,  v = V, k0 = 0.0)
    g("P4 equilibrium widths untouched", abs(e_off.width / e_on.width - 1) < 0.02,
      @sprintf("⟨k²⟩/2M: off %.4f  on %.4f  (T = %.3f)", e_off.width, e_on.width, T0))

    println(ok[] ? "ALL GATES PASS" : "GATE FAILURE")
    exit(ok[] ? 0 : 1)
end

main()
