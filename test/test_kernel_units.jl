# ==============================================================================================
# test_kernel_units.jl — the functions the suite never touched, each against an INDEPENDENT
# construction rather than against a restatement of its own formula (2026-08-31 audit).
#
# WHAT WAS MISSING. Before this file, `runtests.jl` covered the transport times, `effective_DsT`,
# the FDR closure and the entry-point contract; `test_momentum_dims3.jl` and the bench gates covered
# the ensemble-level physics. Nothing exercised the interpolant, the spline EVALUATOR, either boost
# kernel, the two Jüttner samplers, the FONLL sampler's fidelity, `LV_TAUN_SCALE`, or the
# homogeneous-box entry point. Two of those had defects (see U2 and the CHANGELOG).
#
# THE RULE FOLLOWED HERE. A test that re-derives the kernel's own algebra passes whatever the
# kernel does. So each block below is anchored on something the kernel cannot fake:
#   · the interpolant on data whose exact value is known analytically;
#   · the boosts on Lorentz INVARIANTS and on the 4-vector boost matrix, never on the kernel's
#     own parallel/perpendicular decomposition;
#   · the samplers on χ² / KS against the distribution they are supposed to be drawing;
#   · the spline evaluator against the closed-form transport time it is a spline OF.
#
#   julia --project=Julia Julia/LangevInMedium.jl/test/test_kernel_units.jl
# ==============================================================================================
using Test, Random, Statistics, LinearAlgebra, QuadGK, LangevInMedium

const KC = LangevInMedium.KernelsCPU
const TR = LangevInMedium.Transport
const UT = LangevInMedium.Utils

# ⚠ EVERY relativistic kernel builds its Lorentz factor as `1/sqrt(1 - v^2 + 1e-10)`, not
# `1/sqrt(1 - v^2)` — a guard against a table cell with 1 - v^2 <= 0. Reference values below are
# therefore built from γ_reg, and the size and consequences of the regularisation are pinned
# separately in U3b (bias) and U3c (the round trip it stops being an involution).
γ_reg(v) = 1.0/sqrt(1.0 - v*v + 1e-10)

@testset "U1 interpolate_2d_cpu" begin
    # EXACT on a bilinear field: v(x,y) = a + b·x + c·y + d·x·y is reproduced with NO error by
    # bilinear interpolation, so any deviation is an indexing or weighting bug, not discretisation.
    x = collect(0.0:0.5:3.0); y = collect(-1.0:0.25:1.0)
    a, b, c, d = 0.7, -1.3, 2.1, 0.9
    V = [a + b*xi + c*yi + d*xi*yi for xi in x, yi in y]
    exact(xi, yi) = a + b*xi + c*yi + d*xi*yi
    Random.seed!(7)
    for _ in 1:400
        xi = 3.0*rand(); yi = -1.0 + 2.0*rand()
        @test isapprox(KC.interpolate_2d_cpu(x, y, V, xi, yi), exact(xi, yi); rtol = 1e-12, atol = 1e-12)
    end
    # nodes are hit exactly
    for (i, xi) in enumerate(x), (j, yi) in enumerate(y)
        @test isapprox(KC.interpolate_2d_cpu(x, y, V, xi, yi), V[i, j]; rtol = 1e-12, atol = 1e-12)
    end
    # CLAMPED outside the table, never extrapolated — this is the 0.2.0 GPU fix's CPU reference,
    # and the property the whole "a particle past the rim reads the edge value" contract rests on.
    for (xi, yi) in ((-5.0, 0.0), (99.0, 0.0), (1.0, -9.0), (1.0, 9.0), (-5.0, 9.0))
        @test isapprox(KC.interpolate_2d_cpu(x, y, V, xi, yi),
                       exact(clamp(xi, first(x), last(x)), clamp(yi, first(y), last(y))); rtol = 1e-12)
    end
    # non-uniform axis (the interpolant must not assume constant spacing)
    xnu = [0.0, 0.1, 0.7, 0.75, 3.0]
    Vnu = [a + b*xi + c*yi + d*xi*yi for xi in xnu, yi in y]
    for _ in 1:200
        xi = 3.0*rand(); yi = -1.0 + 2.0*rand()
        @test isapprox(KC.interpolate_2d_cpu(xnu, y, Vnu, xi, yi), exact(xi, yi); rtol = 1e-12, atol = 1e-12)
    end
    # degenerate cell (x1 == x0) returns the node value instead of dividing by zero
    xd = [1.0, 1.0, 2.0]; Vd = [1.0 2.0; 1.0 2.0; 3.0 4.0]
    @test isfinite(KC.interpolate_2d_cpu(xd, y[1:2], Vd, 1.0, y[1]))
end

@testset "U2 eval_tau_n_spline — accuracy in range, and the EXTRAPOLATION defect" begin
    M, DsT, Tmin, Tmax = 1.5, 0.11634, 0.12, 0.50
    T0, invdT, vals = build_tau_drag_spline(M, DsT; Tmin, Tmax, nT = 1024)
    # in range: a linear spline of a smooth convex function, error set by the grid
    worst = maximum(abs(eval_tau_n_spline(T, T0, invdT, vals) / tau_drag(T, M, DsT) - 1)
                    for T in range(Tmin, Tmax; length = 997))
    @test worst < 1e-5
    # the endpoints are hit exactly
    @test eval_tau_n_spline(Tmin, T0, invdT, vals) == vals[1]
    @test isapprox(eval_tau_n_spline(Tmax, T0, invdT, vals), vals[end]; rtol = 1e-12)
    @test eval_tau_n_spline(NaN, T0, invdT, vals) == 0.0
    @test eval_tau_n_spline(0.3, 0.0, 1.0, [2.5]) == 2.5          # degenerate table

    # ── the defect (2026-08-31), pinned as expected-fail ─────────────────────────────────────────
    # Outside [Tmin, Tmax] the cell INDEX is clamped but the interpolation WEIGHT is not, so this
    # extrapolates a convex 1/T² function and the extension crosses zero. A negative time is not a
    # NaN anyone would notice: every kernel guards `τ > 0 ? 1/τ : 0`, so the particle silently loses
    # BOTH drag and noise. The one-line fix is `t = clamp(u - (i-1), 0.0, 1.0)`; it is not applied
    # because it moves the last bits of 6 of the 10 regression-corpus hashes (⟨p²⟩ and ⟨x²⟩ stay
    # identical to 17 digits — it is a sub-ulp reordering, not a physics change). See the long
    # docstring on `Transport.eval_tau_n_spline` and CHANGELOG 0.2.1 "found, not fixed".
    #
    # @test_broken, not @test_skip, ON PURPOSE: if someone applies the clamp these turn
    # "Unexpectedly passed" and point the reader straight at this comment.
    @test_broken eval_tau_n_spline(0.90, T0, invdT, vals) > 0
    @test_broken eval_tau_n_spline(0.90, T0, invdT, vals) == vals[end]
    @test_broken eval_tau_n_spline(0.05, T0, invdT, vals) == vals[1]
    # what it does TODAY, measured — so the defect's size is recorded, not just its existence
    @test eval_tau_n_spline(0.90, T0, invdT, vals) < 0                       # ← −0.083 fm
    @test isapprox(eval_tau_n_spline(0.05, T0, invdT, vals), 5.168; atol = 0.01)
    @test tau_drag(0.05, M, DsT) / eval_tau_n_spline(0.05, T0, invdT, vals) > 2.6   # 2.7× off
    # the far-out query is what a kernel would turn into "no drag, no noise":
    @test (τ = eval_tau_n_spline(0.90, T0, invdT, vals); (τ > 0.0 && isfinite(τ)) ? 1/τ : 0.0) == 0.0
end

# ── boost kernels ────────────────────────────────────────────────────────────────────────────────
"Set up N particles on a ring of radius r with a uniform radial flow v; returns (mom, pos, args)."
function boost_setup(; N = 500, v = 0.45, r = 4.0, M = 1.5, pdim = 2, seed = 3)
    rng = MersenneTwister(seed)
    φ = 2π .* rand(rng, N)
    pos = vcat((r .* cos.(φ))', (r .* sin.(φ))')
    mom = 0.9 .* randn(rng, pdim, N)
    xg = collect(0.0:0.5:20.0); tg = collect(0.0:0.5:5.0)
    Vf = fill(v, length(xg), length(tg))
    (mom, pos, xg, tg, Vf, M, N, v)
end

@testset "U3 boosts — invariants, round trip, and the 4-vector boost matrix" begin
    # ⚠ THE KERNEL'S γ IS REGULARISED: `γ = 1/sqrt(1 - v² + 1e-10)`, not `1/sqrt(1 - v²)`. So the
    # boost is not EXACTLY a Lorentz transformation — γ comes out low by ≈ ½·1e-10/(1−v²), i.e.
    # 6.3e-11 relative at v = 0.45 and 2.6e-10 at v = 0.9. That is far below anything the physics
    # can see, but it is a real property of the kernel and it sets the tolerance below; a test
    # written against the exact γ fails at 1e-11 (it did, which is how this got documented).
    # The regularisation is quantified explicitly in U3b. Both backends carry the same expression.
    for pdim in (2, 3), rel in (true, false)
        mom, pos, xg, tg, Vf, M, N, v = boost_setup(; pdim = pdim)
        p_lab = copy(mom)
        KC.kernel_boost_to_rest_frame_cpu!(mom, pos, xg, tg, Vf, M, N, 1, 0.1, 0.0; relativistic = rel)
        p_lrf = copy(mom)
        KC.kernel_boost_to_lab_frame_cpu!(mom, pos, xg, tg, Vf, M, N, 1, 0.1, 0.0; relativistic = rel)
        # LAB -> LRF -> LAB. EXACT in the Galilean branch (p∥ ∓ m·v is its own inverse). NOT exact
        # in the relativistic one: Λ(v)Λ(−v) = 1 requires γ²(1−v²) = 1, and the regularised γ gives
        # γ²(1−v²) = (1−v²)/(1−v²+1e-10) = 1 − 1e-10/(1−v²) instead. See U3c for the measurement.
        if rel
            @test maximum(abs, mom .- p_lab) < 1e-8
        else
            @test maximum(abs, mom .- p_lab) < 1e-14
        end

        if rel
            γ = γ_reg(v)
            worst_shell = 0.0
            for i in 1:N
                r = hypot(pos[1, i], pos[2, i]); n = (pos[1, i]/r, pos[2, i]/r)
                E = sqrt(M^2 + sum(abs2, view(p_lab, :, i)))
                ppar = p_lab[1, i]*n[1] + p_lab[2, i]*n[2]
                # Λ(v) acting on the 4-vector (E, p∥) — built here from the boost matrix, NOT from
                # the kernel's parallel/perpendicular decomposition, so the STRUCTURE of the
                # transformation (sign, E-dependence, which components move) is independently checked.
                ppar2 = γ*(ppar - v*E)
                E2    = γ*(E - v*ppar)
                @test isapprox(p_lrf[1, i], p_lab[1, i] + (ppar2 - ppar)*n[1]; rtol = 1e-12, atol = 1e-13)
                @test isapprox(p_lrf[2, i], p_lab[2, i] + (ppar2 - ppar)*n[2]; rtol = 1e-12, atol = 1e-13)
                # THE MASS SHELL: E'² − |p'|² = M². Not a tautology — E2 is built by the boost
                # matrix above and |p'| by the kernel, so this fails if either loses a γ or a v·E.
                worst_shell = max(worst_shell, abs(E2^2 - sum(abs2, view(p_lrf, :, i)) - M^2))
            end
            # the residual is set by the 1e-10 γ regularisation, not by the algebra
            @test worst_shell < 1e-7
            # p_z IS INVARIANT under the TRANSVERSE boost — the whole premise of pdim = 3
            pdim == 3 && @test p_lrf[3, :] == p_lab[3, :]
        else
            # Galilean: exactly p∥ − m·v along r̂. No γ, no E, no regularisation ⇒ exact to 1e-15.
            for i in 1:N
                r = hypot(pos[1, i], pos[2, i])
                @test isapprox(p_lrf[1, i], p_lab[1, i] - M*v*pos[1, i]/r; rtol = 1e-14, atol = 1e-15)
                @test isapprox(p_lrf[2, i], p_lab[2, i] - M*v*pos[2, i]/r; rtol = 1e-14, atol = 1e-15)
            end
            pdim == 3 && @test p_lrf[3, :] == p_lab[3, :]
        end
    end
end

@testset "U3b the γ regularisation, quantified" begin
    # `+1e-10` inside the square root is a guard against 1−v² ≤ 0 from a glitchy table cell. Pinned
    # here so its size is on record and a change to it is visible:
    #   · at production flow (|v| ≤ 0.8) it biases γ LOW by ≤ 1.4e-10 relative — invisible;
    #   · it also OVERRIDES the |v| clamp. The kernels clamp |v| ≤ √(1−1e-12), which alone would
    #     admit γ ≈ 1e6; with the epsilon, γ saturates at ≈ 1e5 instead. The two guards disagree by
    #     an order of magnitude, and the epsilon is the one that binds.
    γ_exact(v) = 1.0/sqrt(1.0 - v*v)
    for v in (0.0, 0.2, 0.45, 0.8)
        @test abs(γ_reg(v)/γ_exact(v) - 1) < 1.5e-10
        @test γ_reg(v) <= γ_exact(v)                      # always biased low, never high
    end
    vmax = sqrt(1.0 - 1e-12)
    @test 0.9e5 < γ_reg(vmax) < 1.1e5                     # the epsilon binds, not the clamp
    @test γ_exact(vmax) > 9e5                             # what the clamp alone would have allowed
end

@testset "U3c the boost round trip is not an identity — and by exactly the predicted amount" begin
    # A CONSEQUENCE of the +1e-10 in γ, not an independent defect: one lab→LRF→lab round trip
    # multiplies the momentum by ≈ γ²(1−v²) = 1 − 1e-10/(1−v²), so the error must GROW with the
    # flow like 1/(1−v²). The driver does one round trip PER STEP, so this is a slow systematic
    # contraction; at a production 5 800 steps and v = 0.6 it accumulates to ≈ 9e-7 relative, which
    # is why it has never mattered. Pinned so that it cannot grow unnoticed.
    drift = Float64[]
    for v in (0.30, 0.80)
        mom, pos, xg, tg, Vf, M, N, _ = boost_setup(; N = 400, v = v)
        Vf = fill(v, size(Vf))
        p0 = copy(mom)
        KC.kernel_boost_to_rest_frame_cpu!(mom, pos, xg, tg, Vf, M, N, 1, 0.1, 0.0)
        KC.kernel_boost_to_lab_frame_cpu!(mom, pos, xg, tg, Vf, M, N, 1, 0.1, 0.0)
        push!(drift, maximum(abs, mom .- p0) / maximum(abs, p0))
    end
    @test all(0 .< drift .< 1e-8)                       # non-zero: the round trip really is not exact
    # the ratio must track (1−0.30²)/(1−0.80²) = 2.53 — the SIGNATURE of the 1/(1−v²) law, which a
    # generic rounding artefact would not have
    @test 1.5 < drift[2]/drift[1] < 4.5
    # Galilean: an exact involution, to the last bit
    mom, pos, xg, tg, Vf, M, N, _ = boost_setup(; N = 400, v = 0.8)
    Vf = fill(0.8, size(Vf)); p0 = copy(mom)
    KC.kernel_boost_to_rest_frame_cpu!(mom, pos, xg, tg, Vf, M, N, 1, 0.1, 0.0; relativistic = false)
    KC.kernel_boost_to_lab_frame_cpu!(mom, pos, xg, tg, Vf, M, N, 1, 0.1, 0.0; relativistic = false)
    @test maximum(abs, mom .- p0) < 1e-15
end

@testset "U4 boosts — a particle comoving with the fluid is at rest in the LRF" begin
    # The physical statement, checked WITHOUT reference to the kernel's algebra: give each particle
    # exactly the fluid's own momentum m·γ·v·r̂ and demand the rest-frame momentum vanish.
    mom, pos, xg, tg, Vf, M, N, v = boost_setup(; N = 200)
    γ = γ_reg(v)
    for i in 1:N
        r = hypot(pos[1, i], pos[2, i])
        mom[1, i] = M*γ*v*pos[1, i]/r
        mom[2, i] = M*γ*v*pos[2, i]/r
    end
    KC.kernel_boost_to_rest_frame_cpu!(mom, pos, xg, tg, Vf, M, N, 1, 0.1, 0.0)
    @test maximum(abs, mom) < 1e-10
    # and `kernel_set_to_fluid_velocity_cpu!` is what produces that momentum in the first place
    mom2 = randn(2, N)
    KC.kernel_set_to_fluid_velocity_cpu!(mom2, pos, xg, tg, Vf, M, N, 1, 0.1, 0.0)
    for i in 1:N
        r = hypot(pos[1, i], pos[2, i])
        @test isapprox(mom2[1, i], M*γ*v*pos[1, i]/r; rtol = 1e-12)
        @test isapprox(mom2[2, i], M*γ*v*pos[2, i]/r; rtol = 1e-12)
    end
    mom3 = randn(2, N)
    KC.kernel_set_to_fluid_velocity_cpu!(mom3, pos, xg, tg, Vf, M, N, 1, 0.1, 0.0; relativistic = false)
    @test isapprox(hypot(mom3[1, 1], mom3[2, 1]), M*v; rtol = 1e-12)   # m·v, no γ
end

@testset "U5 Bjorken redshift kernel — exact telescoping and the τ ≤ 0 guard" begin
    N = 64; Δt = 1e-3; τ0 = 0.4; nsteps = 500
    mom = ones(3, N); mom[3, :] .= collect(range(-2.0, 2.0; length = N))
    pz0 = copy(mom[3, :])
    for step in 1:nsteps
        KC.kernel_bjorken_redshift_cpu!(mom, 3, step, Δt, τ0, N)
    end
    # p_z(τ) = p_z(τ₀)·τ₀/τ EXACTLY (the product telescopes), and rows 1,2 are untouched
    @test maximum(abs, mom[3, :] ./ pz0 .- τ0/(τ0 + nsteps*Δt)) < 1e-12
    @test all(mom[1, :] .== 1.0) && all(mom[2, :] .== 1.0)
    # τ_a ≤ 0 is a no-op, not a division by zero
    m2 = ones(3, 4); KC.kernel_bjorken_redshift_cpu!(m2, 3, 1, Δt, 0.0, 4)
    @test all(m2[3, :] .== 1.0)
end

# ── the Jüttner samplers ─────────────────────────────────────────────────────────────────────────
"Analytic P(p) ∝ p^(d-1) e^{-(E-m)/T} normalised on [0, pmax]."
function juttner_cdf(M, T, d, p)
    w(q) = q^(d - 1) * exp(-(sqrt(q^2 + M^2) - M)/T)
    num, _ = quadgk(w, 0.0, p; rtol = 1e-10)
    den, _ = quadgk(w, 0.0, Inf; rtol = 1e-10)
    num/den
end

@testset "U6 _draw_juttner_pstar_lib draws the Jüttner (χ², d = 1, 2, 3)" begin
    M, T, Nsamp = 1.5, 0.30, 60_000
    for d in (1, 2, 3)
        Random.seed!(1000 + d)
        s = [KC._draw_juttner_pstar_lib(M, T, d) for _ in 1:Nsamp]
        @test all(>=(0.0), s)
        # χ² over deciles of the ANALYTIC distribution ⇒ 10 equiprobable bins by construction
        edges = [0.0]
        for q in 0.1:0.1:0.9
            lo, hi = 0.0, 16*sqrt(M*T) + 16*T
            for _ in 1:80
                mid = (lo + hi)/2
                juttner_cdf(M, T, d, mid) < q ? (lo = mid) : (hi = mid)
            end
            push!(edges, (lo + hi)/2)
        end
        push!(edges, Inf)
        obs = [count(p -> edges[k] <= p < edges[k+1], s) for k in 1:10]
        expd = Nsamp/10
        χ2 = sum((o - expd)^2/expd for o in obs)
        @test χ2 < 27.9              # 9 dof, p = 0.001
        # the envelope must actually dominate the target (a rejection sampler with wmax < max w
        # silently truncates the peak)
        a = d - 1
        Epk = (a*T + sqrt(a^2*T^2 + 4M^2))/2
        ppk = sqrt(max(Epk^2 - M^2, 1e-12))
        gpk = (ppk^a)*exp(-(Epk - M)/T)
        @test all(((p^a)*exp(-(sqrt(p^2 + M^2) - M)/T) <= 1.3*gpk + 1e-15)
                  for p in range(0, 16*sqrt(M*T) + 16*T; length = 4000))
    end
    @test KC._draw_juttner_pstar_lib(M, 0.0, 2) == 0.0     # T ≤ 0 ⇒ no momentum
end

@testset "U7 build_juttner_invcdf agrees with the rejection sampler it replaces on the GPU" begin
    # THIS IS THE AGREEMENT PRODUCTION RESTS ON and nothing measured it: the CPU RTA path rejection-
    # samples |p*|, the GPU RTA path reads this table. If they differ, `collision_mode = :rta` means
    # two different things on the two backends.
    M, Nsamp = 1.5, 40_000
    for d in (1, 2, 3), T in (0.16, 0.30, 0.45)
        inv, nU, nT, Tmin_t, invdT_t = TR.build_juttner_invcdf(M, d; Tmin = 0.12, Tmax = 0.50)
        @test length(inv) == nU*nT
        @test all(isfinite, inv) && all(>=(0.0), inv)
        # the table is monotone in u at fixed T (it is an inverse CDF)
        jT = clamp(Int(floor((T - Tmin_t)*invdT_t)) + 1, 1, nT)
        col = inv[(jT-1)*nU + 1 : jT*nU]
        @test all(diff(col) .>= -1e-12)
        # host-side twin of the device bilinear lookup
        function lookup(u, Tq)
            fu = clamp(u, 0, 1)*(nU - 1); iu = clamp(Int(floor(fu)), 0, nU - 2); tu = fu - iu
            fT = (Tq - Tmin_t)*invdT_t;   jj = clamp(Int(floor(fT)), 0, nT - 2); tT = clamp(fT - jj, 0, 1)
            c0 = inv[jj*nU + iu + 1]*(1 - tu) + inv[jj*nU + iu + 2]*tu
            c1 = inv[(jj+1)*nU + iu + 1]*(1 - tu) + inv[(jj+1)*nU + iu + 2]*tu
            c0*(1 - tT) + c1*tT
        end
        Random.seed!(4242)
        tab = [lookup(rand(), T) for _ in 1:Nsamp]
        rej = [KC._draw_juttner_pstar_lib(M, T, d) for _ in 1:Nsamp]
        # two-sample Kolmogorov–Smirnov
        st = sort(tab); sr = sort(rej); allp = sort(vcat(st, sr))
        F(v, s) = searchsortedlast(s, v)/length(s)
        ks = maximum(abs(F(v, st) - F(v, sr)) for v in allp)
        @test ks < 1.63*sqrt(2/Nsamp)          # two-sided, α = 0.01
        # and the moments the physics actually uses
        @test isapprox(mean(tab), mean(rej); rtol = 0.02)
        @test isapprox(mean(abs2, tab), mean(abs2, rej); rtol = 0.04)
    end
end

@testset "U8 sample_pz_conditional_juttner / append_thermal_pz" begin
    # exact conditional f(p_z) ∝ exp(−√(m_T²+p_z²)/T); both envelope branches (a > 1.05 Gaussian-in-y
    # and a ≤ 1.05 Laplace) are exercised, including immediately either side of the branch cut.
    for (mT, T) in ((1.5, 0.30), (2.5, 0.16), (1.05*0.4 + 1e-6, 0.4), (0.95*0.4, 0.4), (0.5, 0.9))
        Random.seed!(hash((mT, T)) % 10^6)
        s = [UT.sample_pz_conditional_juttner(mT, T) for _ in 1:40_000]
        @test all(isfinite, s)
        w(pz) = exp(-sqrt(mT^2 + pz^2)/T)
        den, _ = quadgk(w, -Inf, Inf; rtol = 1e-10)
        m2, _ = quadgk(pz -> pz^2*w(pz), -Inf, Inf; rtol = 1e-10)
        m4, _ = quadgk(pz -> pz^4*w(pz), -Inf, Inf; rtol = 1e-10)
        @test isapprox(mean(s), 0.0; atol = 6*std(s)/sqrt(length(s)))   # symmetric
        @test isapprox(mean(abs2, s), m2/den; rtol = 0.04)
        @test isapprox(mean(x -> x^4, s), m4/den; rtol = 0.10)          # 4th moment: the tail too
    end
    @test UT.sample_pz_conditional_juttner(1.5, 0.0) == 0.0
    # append_thermal_pz: rows 1-2 copied verbatim, row 3 drawn at the LOCAL temperature
    N = 4_000; mom = 0.5 .* randn(MersenneTwister(9), 2, N); pos = 2.0 .* randn(MersenneTwister(10), 2, N)
    out = UT.append_thermal_pz(mom, pos, 1.5, r -> 0.30)
    @test size(out) == (3, N) && out[1:2, :] == mom
    @test isapprox(mean(out[3, :]), 0.0; atol = 6*std(out[3, :])/sqrt(N))
    # antithetic mode mirrors p_z across the (2i−1, 2i) pairs
    outa = UT.append_thermal_pz(mom, pos, 1.5, r -> 0.30; antithetic = true)
    @test all(outa[3, 2:2:end] .== -outa[3, 1:2:end-1])
    @test_throws ErrorException UT.append_thermal_pz(zeros(3, 4), zeros(2, 4), 1.5, r -> 0.3)
end

@testset "U9 sample_particles_from_FONLL reproduces the density it was handed" begin
    # FIDELITY, not just "it returns the right shape": bin the sampled particles and compare against
    # the tabulated density in BOTH spatial modes.
    M = 1.5; rg = collect(0.0:0.25:12.0); pg = collect(0.05:0.05:6.0)
    dens = [p*exp(-sqrt(p^2 + M^2)/0.5)*exp(-r^2/18) for p in pg, r in rg]   # (Np, Nr)
    N = 120_000
    for cart in (true, false)
        Random.seed!(cart ? 11 : 12)
        x, p = LangevInMedium.sample_particles_from_FONLL(rg, pg, dens, N; cartesian_spatial_sampling = cart)
        @test size(x) == (2, N) && size(p) == (2, N)
        r = vec(sqrt.(sum(abs2, x; dims = 1)))
        @test maximum(r) <= last(rg) + 1e-9
        # radial profile: P(r) ∝ r·n(r) in BOTH modes (the Cartesian mode gets the r from the area
        # element, the polar mode from the explicit Jacobian) — they must agree with each other
        # and with the table.
        n_r = vec(sum(dens .* pg; dims = 1))
        wref = n_r .* rg; wref ./= sum(wref)
        edges = rg
        h = zeros(length(edges) - 1)
        for ri in r
            k = clamp(searchsortedlast(edges, ri), 1, length(h)); h[k] += 1
        end
        h ./= sum(h)
        ref = [(wref[k] + wref[k+1])/2 for k in 1:length(h)]; ref ./= sum(ref)
        @test sum(abs, h .- ref) < 0.06                      # total-variation distance
        # momentum: isotropic azimuth, and |p| matching the p-marginal of the table
        φ = atan.(p[2, :], p[1, :])
        @test isapprox(mean(cos.(φ)), 0.0; atol = 0.02) && isapprox(mean(sin.(φ)), 0.0; atol = 0.02)
        pm = vec(sqrt.(sum(abs2, p; dims = 1)))
        wp = vec(sum(dens .* rg'; dims = 2)) .* pg; wp ./= sum(wp)
        @test isapprox(mean(pm), sum(wp .* pg); rtol = 0.05)
    end
    # an (Nr, Np) table is transposed automatically; a table of neither shape is an error
    Random.seed!(13)
    x1, p1 = LangevInMedium.sample_particles_from_FONLL(rg, pg, permutedims(dens), 500)
    @test size(x1) == (2, 500)
    @test_throws ErrorException LangevInMedium.sample_particles_from_FONLL(rg, pg, ones(3, 5), 10)
end

@testset "U10 LV_TAUN_SCALE rescales BOTH splines" begin
    # The docstring's whole point: it is a Ref populated in __init__, not a const parsed at
    # precompile time (which would bake the value into the .ji and silently ignore the env var).
    @test LangevInMedium.LV_TAUN_SCALE isa Ref{Float64}
    M, DsT = 1.5, 0.11634
    _, _, d1 = build_tau_drag_spline(M, DsT; Tmin = 0.12, Tmax = 0.5, nT = 32)
    _, _, n1 = build_taun_current_spline(M, DsT; Tmin = 0.12, Tmax = 0.5, nT = 32)
    old = LangevInMedium.LV_TAUN_SCALE[]
    try
        LangevInMedium.LV_TAUN_SCALE[] = 1/6
        _, _, d2 = build_tau_drag_spline(M, DsT; Tmin = 0.12, Tmax = 0.5, nT = 32)
        _, _, n2 = build_taun_current_spline(M, DsT; Tmin = 0.12, Tmax = 0.5, nT = 32)
        @test all(isapprox.(d2, d1 ./ 6; rtol = 1e-12))
        @test all(isapprox.(n2, n1 ./ 6; rtol = 1e-12))    # BOTH — it moves D_s, not just the rate
    finally
        LangevInMedium.LV_TAUN_SCALE[] = old
    end
    @test LangevInMedium.LV_TAUN_SCALE[] == old
end

@testset "U11 check_momentum_dims contract" begin
    @test UT.check_momentum_dims(2, 2, false, false, 0.0) === nothing
    @test UT.check_momentum_dims(2, 3, false, false, 0.0) === nothing
    @test UT.check_momentum_dims(2, 3, false, true,  0.4) === nothing
    @test_throws ErrorException UT.check_momentum_dims(3, 2, false, false, 0.0)
    @test_throws ErrorException UT.check_momentum_dims(2, 3, true,  false, 0.0)   # radial + p_z
    @test_throws ErrorException UT.check_momentum_dims(2, 2, false, true,  0.4)   # redshift needs pdim 3
    @test_throws ErrorException UT.check_momentum_dims(2, 3, false, true,  0.0)   # needs τ₀ > 0
end

@testset "U12 momentum update and snapshot kernels" begin
    N = 100; d = 3; Δt = 4e-3
    mom = randn(MersenneTwister(1), d, N); det = randn(MersenneTwister(2), d, N); sto = randn(MersenneTwister(3), d, N)
    # association matters: the kernel computes p + (det + √Δt·sto), so the reference must too
    want = mom .+ (det .+ sqrt(Δt) .* sto)
    KC.kernel_update_momenta_LRF_cpu!(mom, det, sto, Δt, d, N)
    @test mom == want
    H = zeros(d, N, 3); KC.kernel_save_momenta_cpu!(H, mom, 2, N); @test H[:, :, 2] == mom
    P = zeros(2, N, 3); pos = randn(2, N); KC.kernel_save_positions_cpu!(P, pos, 3, N); @test P[:, :, 3] == pos
    # the scalar snapshot kernel behind the homogeneous-box |p| history
    col = zeros(N); snap = abs.(randn(N))
    KC.kernel_save_snapshot_cpu!(col, snap, N); @test col == snap
    col2 = zeros(N); KC.kernel_save_snapshot_cpu!(col2, snap, 3); @test col2[1:3] == snap[1:3] && all(iszero, col2[4:end])
end

@testset "U13 homogeneous-box entry point against the EXACT discrete OU solution" begin
    # The toy path: no positions, a fixed κ = 2.5T³ that ignores DsT, η_D = κ/(2Tm), and the EULER
    # update p ← (1 − ηΔt)p + √(κΔt)·ξ (not the exact-OU propagator the field-driven path uses).
    # For that scheme ⟨p²⟩ obeys a closed recursion, so the whole relaxation CURVE can be predicted
    # exactly — a much sharper test than checking the endpoint, and it pins η_D, κ AND the update.
    T, M, d = 0.35, 1.5, 3
    κ  = 2.5*T^3
    ηD = κ/(2*T*M)
    Δt, p0 = 0.05, 3.0
    @test isapprox(1/ηD, 9.79; atol = 0.05)          # the relaxation time this toy actually has
    tf, save = 60.0, 5.0                             # ≈ 6 relaxation times
    Random.seed!(77)
    t, hist = simulate_ensemble_bulk(CPUBackend(), T; N_particles = 20_000, Δt = Δt,
        final_time = tf, save_interval = save, m = M, dimensions = d, p0 = p0,
        initial_condition = "delta")
    @test length(t) == length(hist)
    @test all(isapprox(h, p0; rtol = 1e-12) for h in hist[1])     # "delta": every |p| = p0 exactly
    # exact discrete recursion: ⟨p²⟩ₙ = a^{2n}p0² + d·κΔt·(1−a^{2n})/(1−a²),  a = 1 − ηΔt
    a = 1 - ηD*Δt
    nstep = round(Int, save/Δt)
    for (k, h) in enumerate(hist)
        n = (k - 1)*nstep
        want = a^(2n)*p0^2 + d*κ*Δt*(1 - a^(2n))/(1 - a^2)
        @test isapprox(mean(abs2, h), want; rtol = 0.03)
    end
    # and the k→∞ limit of that recursion is the Maxwell equipartition d·mT, up to the Euler
    # O(ηΔt) bias 1/(1 − ηΔt/2) — assert BOTH so the toy's own discretisation is on record
    @test isapprox(d*κ*Δt/(1 - a^2), d*M*T/(1 - ηD*Δt/2); rtol = 1e-12)
    @test isapprox(mean(abs2, hist[end]), d*M*T; rtol = 0.03)
    @test all(isfinite, hist[end])
    Random.seed!(78)
    _, hb = simulate_ensemble_bulk(CPUBackend(), T; N_particles = 2_000, Δt = 2e-3, final_time = 0.2,
        save_interval = 0.1, m = M, dimensions = d, initial_condition = "bimodal")
    @test all(isfinite, hb[end])
    @test_throws ErrorException simulate_ensemble_bulk(CPUBackend(), T; N_particles = 10, Δt = 2e-3,
        final_time = 0.1, save_interval = 0.05, initial_condition = "nonsense")
end

@testset "U14 the retired samplers are gone" begin
    # 2026-08-31: three zero-consumer samplers moved to Julia/Projects/trash/ (two of them buggy —
    # see the header of that file). This pins the removal so they cannot silently reappear.
    for f in (:sample_particles_from_density, :sample_initial_particles_from_pdf!,
              :sample_initial_particles_at_origin!)
        @test !isdefined(LangevInMedium, f)
        @test !isdefined(LangevInMedium.Utils, f)
    end
    @test isdefined(LangevInMedium.Utils, :sample_initial_particles_at_origin_no_position!)  # still live
end
