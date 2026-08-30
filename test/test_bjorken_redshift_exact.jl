#=
   test_bjorken_redshift_exact.jl — the Bjorken redshift against the EXACT free-streaming solution,
   on a distribution it cannot fake (2026-08-30).

   WHY THIS EXISTS. `test_momentum_dims3.jl` gate (R) checks that ⟨p_z²⟩ telescopes as (τ₀/τ)². That
   is the implementation against the INTENDED FORMULA — it would pass just as happily if the formula
   itself were wrong, and it constrains only the second moment of a thermal ensemble. RS asked, quite
   reasonably, how sure we are: the redshift and the proper-time kick are both ~2 weeks old and both
   now sit under published numbers, and `proper_time_kicks` alone is worth 29% of the spectrum at
   3-4 GeV. So this file tries to FALSIFY the redshift instead of confirming it.

   THE PHYSICS, IN TWO LINES OF CARTESIAN KINEMATICS (no Milne algebra needed). For a FREE particle
   the lab E and p_z are constants. Boosting to the local longitudinally-comoving frame (rapidity
   η_s, with τ cosh η_s = t and τ sinh η_s = z):

        τ·p_z*  =  τ(cosh η_s·p_z − sinh η_s·E)  =  t·p_z − z·E
        d/dt (t p_z − z E)  =  p_z − (dz/dt)·E  =  p_z − (p_z/E)·E  =  0

   so τ·p_z* is EXACTLY conserved along any free worldline, and p_z*(τ) = p_z*(τ₀)·τ₀/τ with no
   approximation. For a boost-invariant ensemble that lifts to the whole distribution:

        f(p_⊥, p_z, τ)  =  f₀(p_⊥, p_z·τ/τ₀)          — a pure RESCALING of the p_z axis.

   THE TEST. Free-stream a DELIBERATELY NON-THERMAL p_z distribution (a bimodal ±p₀ plus a broad
   uniform pedestal — nothing Gaussian, nothing thermal, and asymmetric so a sign flip shows) and
   demand the whole distribution rescale by exactly τ₀/τ. A wrong POWER of τ, a wrong SIGN, a
   redshift applied to the wrong momentum row, or one applied in the wrong frame each break a
   different gate below; a formula that merely reproduced ⟨p_z²⟩ ∝ τ^{-2} on a thermal ensemble would
   still fail R3 and R4.

   Gates:  R1  ⟨p_z^k⟩ scales as (τ₀/τ)^k for k = 1, 2, 3, 4 — pins the POWER, not just the variance
           R2  the transverse sector is UNTOUCHED: ⟨p_⊥²⟩ ratio = 1 — pins the ROW
           R3  every DECILE of p_z scales by the same τ₀/τ — pins the whole SHAPE, not two moments
           R4  the signed structure survives: the bimodal peaks stay at ∓, none reflected
           R5  it is the exact solution, not an O(Δt) one: halving Δt does not move the answer

   Run: julia --project=Julia Julia/LangevInMedium.jl/test/test_bjorken_redshift_exact.jl
=#

import Pkg; Pkg.activate(normpath(joinpath(@__DIR__, "..", "..")); io = devnull)
using LangevInMedium, Random, Statistics, Printf

const M    = 1.5
const TAU0 = 0.4
const TAU1 = 3.2            # τ₀/τ = 1/8 — a big lever arm, so a wrong power is unmissable
const N    = 200_000
const T0   = 0.300
# Drag must be OFF: the redshift is a FREE-STREAMING statement. τ_drag = M·DsT/T² ∝ DsT, so a huge
# DsT makes η_D = 1/τ_drag negligible (τ_drag ≈ 4e6 fm here) without taking the DsT = 0 branch,
# which glues momenta to the fluid instead of leaving them alone.
# 1e12, not 1e6: at 1e6 the residual noise still moves ⟨p_⊥⟩ by ~1e-3, which is far above the
# tolerance R2 needs to be a statement about the ROW rather than about the drag. At 1e12,
# τ_drag ≈ 1.7e13 fm and κ = 2MT/τ_drag ≈ 5e-14, so the transverse sector moves by ~4e-7 over the
# whole window and the residual is a decade below every tolerance here.
const DST_FREE = 1.0e12

"A deliberately non-thermal, asymmetric p_z law: bimodal ±p₀ (60/40 split) on a uniform pedestal."
function seed_pz(rng, n)
    pz = zeros(n)
    for i in 1:n
        u = rand(rng)
        pz[i] = u < 0.36 ? -1.20 + 0.05*randn(rng) :
                u < 0.60 ?  0.80 + 0.05*randn(rng) :
                            -0.5 + 2.5*rand(rng)          # broad pedestal, asymmetric about 0
    end
    pz
end

function run_free(; dt, n = N)
    rng = MersenneTwister(20260830)
    xgrid = collect(0.0:0.5:60.0); tgrid = collect(TAU0:0.05:(TAU1 + 0.5))
    Tf = fill(T0, length(xgrid), length(tgrid)); Vf = zeros(length(xgrid), length(tgrid))
    x = zeros(2, n); p = zeros(3, n)
    pz0 = seed_pz(rng, n)
    for i in 1:n
        r = sqrt(9 + 16*rand(rng)); θ = 2π*rand(rng)
        x[1, i] = r*cos(θ); x[2, i] = r*sin(θ)
        φ = 2π*rand(rng); kt = 0.4 + 0.3*rand(rng)          # transverse: fixed, must not move
        p[1, i] = kt*cos(φ); p[2, i] = kt*sin(φ); p[3, i] = pz0[i]
    end
    Random.seed!(1234)
    _, moms, _ = simulate_ensemble_bulk(CPUBackend(), nothing, nothing, nothing,
        Tf, Vf, (xgrid, tgrid);
        N_particles = n, Δt = dt, initial_time = TAU0, final_time = TAU1, save_interval = TAU1 - TAU0,
        m = M, DsT = DST_FREE, dimensions = 2, momentum_dimensions = 3, bjorken_redshift = true,
        x_init = x, p_init = p)
    (a = moms[1], b = moms[end])
end

function main()
    R = TAU0 / TAU1
    @printf("Bjorken redshift vs the exact free-streaming solution: τ %.2f → %.2f, τ₀/τ = %.5f, N = %d\n",
            TAU0, TAU1, R, N)
    ok = Ref(true)                  # ⚠ Ref: `ok = false` in a loop would make a NEW local
    g(nm, c, det) = (println("  ", c ? "PASS" : "FAIL", "  ", rpad(nm, 40), det); ok[] &= c)

    s = run_free(dt = 1.0e-3)
    z0 = @view s.a[3, :]; z1 = @view s.b[3, :]
    pt0 = [sqrt(s.a[1,i]^2 + s.a[2,i]^2) for i in 1:N]
    pt1 = [sqrt(s.b[1,i]^2 + s.b[2,i]^2) for i in 1:N]

    # R1 — the POWER of τ, from four moments at once
    worst = 0.0; det = String[]
    for k in 1:4
        m0 = mean(abs.(z0) .^ k); m1 = mean(abs.(z1) .^ k)
        dev = abs((m1/m0) / R^k - 1); worst = max(worst, dev)
        push!(det, @sprintf("k=%d %.4f", k, (m1/m0)/R^k))
    end
    g("R1 ⟨|p_z|^k⟩ ∝ (τ₀/τ)^k, k=1..4", worst < 0.002, join(det, "  ") * @sprintf("  (worst %.2e)", worst))

    # R2 — the ROW: the transverse sector must not move at all
    # tolerance 1e-5, not 0: the drag is negligible, not absent (see DST_FREE). What R2 pins is that
    # the redshift touches the p_z ROW and no other — a factor applied to the wrong row would show
    # here as ~τ₀/τ = 0.125, four orders above this.
    g("R2 transverse untouched", abs(mean(pt1)/mean(pt0) - 1) < 1e-5,
      @sprintf("⟨p_⊥⟩ ratio %.12f", mean(pt1)/mean(pt0)))

    # R3 — the SHAPE: every decile rescales by the same factor
    q0 = quantile(collect(z0), 0.1:0.1:0.9); q1 = quantile(collect(z1), 0.1:0.1:0.9)
    dq = maximum(abs.((q1 ./ q0) ./ R .- 1))
    g("R3 every decile scales by τ₀/τ", dq < 0.002,
      @sprintf("worst decile deviation %.2e", dq))

    # R4 — the SIGN structure: no reflection, and the 60/40 asymmetry survives
    frac0 = count(<(0), z0)/N; frac1 = count(<(0), z1)/N
    g("R4 signed structure preserved", abs(frac1 - frac0) < 1e-9 && frac0 > 0.3,
      @sprintf("fraction p_z<0: %.6f → %.6f", frac0, frac1))

    # R5 — EXACT, not O(Δt): halving the step must not move it
    s2 = run_free(dt = 5.0e-4, n = 50_000)
    r1 = mean(abs.(@view s.b[3, 1:50_000])) / mean(abs.(@view s.a[3, 1:50_000]))
    r2 = mean(abs.(@view s2.b[3, :])) / mean(abs.(@view s2.a[3, :]))
    g("R5 Δt-independent (exact telescoping)", abs(r2/r1 - 1) < 1e-6,
      @sprintf("Δt=1e-3 %.10f   Δt=5e-4 %.10f", r1, r2))

    println(ok[] ? "ALL GATES PASS" : "GATE FAILURE")
    exit(ok[] ? 0 : 1)
end

main()
