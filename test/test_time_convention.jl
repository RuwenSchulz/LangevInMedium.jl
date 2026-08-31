# ==============================================================================================
# test_time_convention.jl — WHEN each kernel reads the background (2026-08-31 audit).
#
# The engine uses TWO different step-time conventions, and nothing said so or checked it:
#
#   · every kernel that LOOKS UP the background — both boosts, the force kernel, the position
#     update, the RTA collision, the freeze-out latch — evaluates it at `t₀ + step·Δt`, the END of
#     the step, while the state it is applied to is the state at `t₀ + (step−1)·Δt`;
#   · `kernel_bjorken_redshift_*` uses `τ_a = t₀ + (step−1)·Δt`, the START of the step, and steps
#     over `[τ_a, τ_a + Δt]`.
#
# The second is not an inconsistency to be repaired: `p_z ← p_z·τ_a/(τ_a+Δt)` is the EXACT free
# streaming solution over the interval the particle actually traverses, and it is exact only with
# the start-of-step τ. The first is an ordinary O(Δt) choice. Together they are consistent to
# O(Δt), which is the scheme's order anyway — but the mixture should be on the record and pinned,
# not rediscovered by the next person who compares two conventions at the sub-percent level.
#
#   julia --project=Julia Julia/LangevInMedium.jl/test/test_time_convention.jl
# ==============================================================================================
using Test, Random, Statistics, LangevInMedium
const KC = LangevInMedium.KernelsCPU

const M, DST, N = 1.5, 0.11634, 512
const XG = collect(0.0:0.25:20.0)
const TG = collect(0.0:0.05:4.0)
# both fields vary STRONGLY in τ, so reading the wrong step is a large, unmistakable error
const VF = [0.02*r/(r + 3.0) + 0.10*τ for r in XG, τ in TG]
const TF = [0.20 + 0.01*r + 0.05*τ    for r in XG, τ in TG]

ring(seed) = (rng = MersenneTwister(seed);
              φ = 2π .* rand(rng, N);
              (vcat((5.0 .* cos.(φ))', (5.0 .* sin.(φ))'), 0.7 .* randn(rng, 2, N)))

@testset "T1 the boosts read the background at t₀ + step·Δt (END of the step)" begin
    # The GALILEAN branch is exactly invertible — p∥ ← p∥ − m·v — so the v the kernel used can be
    # RECOVERED from the output and compared with the field at a specific time. No tolerance games.
    pos, mom = ring(1)
    Δt, t0 = 1e-2, 0.30
    for step in (1, 5, 37)
        m2 = copy(mom)
        KC.kernel_boost_to_rest_frame_cpu!(m2, pos, XG, TG, VF, M, N, step, Δt, t0; relativistic = false)
        for i in 1:N
            r = hypot(pos[1, i], pos[2, i])
            v_used = (mom[1, i] - m2[1, i]) * r / (M * pos[1, i])
            @test isapprox(v_used, KC.interpolate_2d_cpu(XG, TG, VF, r, t0 + step*Δt); rtol = 1e-11)
            # and it is NOT the start-of-step value (the fields vary enough that this is decisive)
            @test !isapprox(v_used, KC.interpolate_2d_cpu(XG, TG, VF, r, t0 + (step-1)*Δt); rtol = 1e-4)
        end
    end
    # step = 0 is the INITIAL lab boost the drivers do before the loop: it must read exactly t₀
    m3 = copy(mom)
    KC.kernel_boost_to_lab_frame_cpu!(m3, pos, XG, TG, VF, M, N, 0, Δt, t0; relativistic = false)
    for i in 1:N
        r = hypot(pos[1, i], pos[2, i])
        v_used = (m3[1, i] - mom[1, i]) * r / (M * pos[1, i])
        @test isapprox(v_used, KC.interpolate_2d_cpu(XG, TG, VF, r, t0); rtol = 1e-11)
    end
end

@testset "T2 the force kernel reads T at t₀ + step·Δt" begin
    # η_D is written out per particle, and η_D = 1/τ_drag(T), so the T the kernel used is directly
    # observable — no inversion needed.
    pos, mom = ring(2)
    Δt, t0 = 1e-2, 0.30
    T0s, invdT, vals = build_tau_drag_spline(M, DST; Tmin = minimum(TF), Tmax = maximum(TF), nT = 4096)
    for step in (1, 11)
        ηv = zeros(N); det = zeros(2, N); sto = zeros(2, N)
        KC.kernel_compute_all_forces_cpu!(TF, XG, TG, copy(mom), pos, zeros(N), zeros(2, N),
            ηv, zeros(N), zeros(N), randn(MersenneTwister(3), 2, N), det, sto,
            Δt, M, zeros(2, N), 2, N, step, t0, DST;
            tau_Tmin = T0s, tau_invdT = invdT, tau_vals = vals)
        for i in 1:N
            r = hypot(pos[1, i], pos[2, i])
            Tend = KC.interpolate_2d_cpu(XG, TG, TF, r, t0 + step*Δt)
            @test isapprox(ηv[i], 1/eval_tau_n_spline(Tend, T0s, invdT, vals); rtol = 1e-11)
        end
    end
end

@testset "T3 the Bjorken redshift uses the START of the step — and that is what makes it exact" begin
    # τ_a = t₀ + (step−1)Δt, so step `k` maps [t₀+(k−1)Δt, t₀+kΔt]. Chaining the whole run
    # telescopes to τ₀/τ_final EXACTLY. Had it used the end-of-step τ (like every other kernel),
    # the product would telescope to (τ₀+Δt)/(τ_final+Δt) — a different, wrong answer.
    Δt, t0, nsteps = 1e-3, 0.4, 800
    mom = ones(3, 8); pz0 = copy(mom[3, :])
    for step in 1:nsteps
        KC.kernel_bjorken_redshift_cpu!(mom, 3, step, Δt, t0, 8)
    end
    τf = t0 + nsteps*Δt
    @test maximum(abs, mom[3, :] ./ pz0 .- t0/τf) < 1e-12          # the exact free-streaming law
    # the end-of-step alternative gives a measurably different number — so the choice is not cosmetic
    wrong = prod((t0 + k*Δt)/(t0 + (k+1)*Δt) for k in 1:nsteps)
    @test abs(wrong - t0/τf) > 1e-6
    # a single step is exactly τ_a/(τ_a+Δt) with τ_a the START time
    m1 = ones(3, 2); KC.kernel_bjorken_redshift_cpu!(m1, 3, 7, Δt, t0, 2)
    τa = t0 + 6*Δt
    @test all(isapprox.(m1[3, :], τa/(τa + Δt); rtol = 1e-14))
end

@testset "T4 the driver is first-order consistent in Δt" begin
    # Deterministic probe: `momentum_langevin = false` glues the particles to the flow, so the whole
    # trajectory is a fixed function of the background with no RNG anywhere. Refining Δt must then
    # converge at FIRST order — the order of the end-of-step background sampling in T1. (A scheme
    # reading the background at the wrong step is still first order, so this does not identify the
    # convention; T1/T2 do that. This checks the two conventions do not fight each other into
    # something inconsistent.)
    rg = collect(0.0:0.5:20.0); pg = collect(0.05:0.1:8.0); dens = ones(length(pg), length(rg))
    x0 = 4.0 .* randn(MersenneTwister(9), 2, 2_000); p0 = 0.5 .* randn(MersenneTwister(10), 2, 2_000)
    run(dt) = (t, m, x) = simulate_ensemble_bulk(CPUBackend(), rg, pg, dens, TF, VF, (XG, TG);
        x_init = x0, p_init = p0, N_particles = 2_000, Δt = dt, initial_time = 0.5,
        final_time = 2.5, save_interval = 2.0, m = M, DsT = DST, dimensions = 2, Tfo = 0.0,
        momentum_langevin = false)[3][end]
    ref = run(1.25e-4)
    e1 = maximum(abs, run(4e-3) .- ref)
    e2 = maximum(abs, run(2e-3) .- ref)
    e3 = maximum(abs, run(1e-3) .- ref)
    @test e1 > e2 > e3 > 0
    @test isapprox(e1/e2, 2.0; rtol = 0.3)          # first order
    @test isapprox(e2/e3, 2.0; rtol = 0.3)
    @info "T4 |x(Δt) − x(Δt→0)| at Δt = 4e-3, 2e-3, 1e-3: $(round.((e1, e2, e3); sigdigits = 3)); ratios $(round(e1/e2; digits = 2)), $(round(e2/e3; digits = 2))"
end
