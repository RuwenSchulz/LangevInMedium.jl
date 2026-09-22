#=
   test_bjorken_frame.jl — WHERE the Bjorken redshift acts, with a transverse flow (2026-09-22).

   THE DEFECT THIS PINS. With `bjorken_redshift = true` the engine scaled p_z ← p_z τ_a/τ_b on the
   REST-FRAME momenta, between the boost to the rest frame and the boost back. p_z is invariant under
   the transverse boost, so the old comment called the two orderings "the same operation" — but E* is
   not, and the back-boost reads E*: every step the LAB transverse momentum fell by γ·v·δE*. Free
   streaming conserves the lab p_⊥ exactly (the η_s-comoving frame's transverse momentum is a
   constant of the motion). In the fluid frame the old ordering is a spurious force −u^r p_z²/(τE*);
   its first-order current, −n D_s κ₂ u^r/τ with κ₂ ≈ 0.9, was the "missing inertia" of
   AttractorHydro I-43 (fitted −0.96 … −1.02 of it, prediction −1, on the stored portraits).

   `test_bjorken_redshift_exact.jl` could not see this: its background has NO flow, where the two
   orderings coincide. This file drives a uniform radial flow v = 0.5 with the collisions switched off
   in effect (D_sT = 1e18 ⇒ τ_drag ≈ 3e18 fm), so the boost pair is exercised and nothing but the
   redshift can move a momentum.

   Gates:  F1  bjorken_frame = :lab conserves every particle's LAB p_⊥ (free streaming)
           F2  bjorken_frame = :lrf (the shipped default) does NOT: ⟨Δp_⊥⟩ < 0, and it vanishes when
               p_z = 0 — the defect, pinned so a silent change of the default shows
           F3  the p_z row is scaled identically by both orderings
           F4  with no flow the two orderings agree

   Run: julia --project=Julia Julia/LangevInMedium.jl/test/test_bjorken_frame.jl
=#
import Pkg; Pkg.activate(normpath(joinpath(@__DIR__, "..", "..")); io = devnull)
using LangevInMedium, Random, Statistics, Printf, Test

const M_BF   = 1.5
const TAU0_BF = 0.4
const TAU1_BF = 2.4
const N_BF   = 20_000
const DST_FREE_BF = 1.0e18     # τ_drag ≈ 3e18 fm: residual kicks ~1e-11 per step (1e12 left ~1e-6)

function run_frame(frame::Symbol; v = 0.5, pz_scale = 1.0)
    rng = MersenneTwister(20260922)
    xgrid = collect(0.0:0.5:60.0); tgrid = collect(TAU0_BF:0.05:(TAU1_BF + 0.5))
    Tf = fill(0.3, length(xgrid), length(tgrid)); Vf = fill(v, length(xgrid), length(tgrid))
    x = zeros(2, N_BF); p = zeros(3, N_BF)
    for i in 1:N_BF
        r = 3.0 + 2.0 * rand(rng); θ = 2π * rand(rng)
        x[1, i] = r * cos(θ); x[2, i] = r * sin(θ)
        φ = 2π * rand(rng); kt = 0.4 + 0.3 * rand(rng)
        p[1, i] = kt * cos(φ); p[2, i] = kt * sin(φ)
        p[3, i] = pz_scale * (rand(rng) < 0.5 ? -1.0 : 1.0) * (0.5 + rand(rng))
    end
    Random.seed!(4321)
    _, moms, _ = simulate_ensemble_bulk(CPUBackend(), nothing, nothing, nothing, Tf, Vf, (xgrid, tgrid);
        N_particles = N_BF, Δt = 1.0e-3, initial_time = TAU0_BF, final_time = TAU1_BF,
        save_interval = TAU1_BF - TAU0_BF, m = M_BF, DsT = DST_FREE_BF, dimensions = 2,
        momentum_dimensions = 3, bjorken_redshift = true, bjorken_frame = frame, x_init = x, p_init = p)
    a, b = moms[1], moms[end]
    pt(m, i) = sqrt(m[1, i]^2 + m[2, i]^2)
    (dpt = [pt(b, i) - pt(a, i) for i in 1:N_BF], pt0 = mean(pt(a, i) for i in 1:N_BF),
     pz = b[3, :], pz0 = a[3, :])
end

@testset "Bjorken redshift frame (bjorken_frame)" begin
    lab = run_frame(:lab); lrf = run_frame(:lrf)
    @printf("  flow v = 0.5, τ %.1f → %.1f, free streaming: ⟨Δp_⊥⟩ = %+.3e (:lab)  %+.3e (:lrf)  of ⟨p_⊥⟩ = %.3f GeV\n",
            TAU0_BF, TAU1_BF, mean(lab.dpt), mean(lrf.dpt), lab.pt0)
    # F1 — the lab transverse momentum is a constant of free streaming. The floor is NOT zero: the
    # boost kernels use γ = 1/√(1 − v² + 1e-10), so each rest↔lab round trip is inexact at ~1e-10 and
    # 2000 steps leave ~4e-7 GeV (measured, identical for both orderings and with p_z = 0). 1e-5 sits
    # an order above that floor and four below the defect (0.16 GeV here).
    @test maximum(abs, lab.dpt) < 1e-5
    # F2 — the shipped ordering drains it (sign and size), and only through p_z
    @test mean(lrf.dpt) < -1e-2
    lrf0 = run_frame(:lrf; pz_scale = 0.0)
    @test maximum(abs, lrf0.dpt) < 1e-5
    # F3 — p_z is scaled identically; the orderings differ only in what the back-boost does with E*
    @test maximum(abs, lab.pz .- lrf.pz) < 1e-9
    @test isapprox(mean(abs.(lab.pz)) / mean(abs.(lab.pz0)), TAU0_BF / TAU1_BF; rtol = 1e-6)
    # F4 — without flow the orderings coincide
    a0 = run_frame(:lab; v = 0.0); b0 = run_frame(:lrf; v = 0.0)
    @test maximum(abs, a0.dpt .- b0.dpt) < 1e-9
    @test_throws ErrorException run_frame(:bogus)
end
