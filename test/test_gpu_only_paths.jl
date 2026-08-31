# ==============================================================================================
# test_gpu_only_paths.jl — the two GPU-only features that production uses and nothing tested
# (2026-08-31 audit).
#
#   `freezeout_capture`  — 48 uses across Julia/Projects/. It replaces the trajectory history with
#                          a per-particle latch of the T = Tfo crossing, and every Cooper–Frye
#                          spectrum in the programme is built from what it books. Zero tests.
#   `integrator_mode=1`  — 20 uses. The README claims it "removes the O(ε) term" of the pre-point
#                          relativistic drag. Nothing checked that it does anything at all.
#
# THE TRICK THAT MAKES FREEZE-OUT TESTABLE. Give the background a temperature that depends ONLY on
# τ. Then the crossing time is the same for every particle no matter where it is or how it moves,
# and it can be computed on the host to machine precision from the SAME interpolated field the
# kernel reads — so `fo_tau` has an exact reference rather than a statistical one. Because the
# kernel sees a bilinearly interpolated T, that reference is the crossing of the PIECEWISE-LINEAR
# T, which is what `freezeout_interp = true` claims to find exactly.
#
#   julia --project=Julia Julia/LangevInMedium.jl/test/test_gpu_only_paths.jl
# ==============================================================================================
ENV["LIM_QUIET"] = "1"
using Test, Random, Statistics, LinearAlgebra

const HAVE_CUDA = try
    @eval using CUDA
    Base.invokelatest(() -> CUDA.functional())
catch
    false
end
using LangevInMedium

if !HAVE_CUDA
    @warn "CUDA not functional — the GPU-only paths (freezeout_capture, integrator_mode=1) are NOT covered"
else
    const KC = LangevInMedium.KernelsCPU
    const M, NP = 1.5, 100_000

    # T(τ) alone: Bjorken cooling, no radial dependence ⇒ one crossing time for the whole ensemble
    const TAU0, TFO = 0.6, 0.155
    const XGF = collect(0.0:0.5:60.0)
    const TGF = collect(TAU0:0.05:12.0)
    # T₀ = 0.30 so that the Bjorken law crosses Tfo at τ_c ≈ 4.35 fm, comfortably inside the table
    # (0.45 would cross at 14.7 fm, past its end).
    const TFF = [0.30*(TAU0/τ)^(1/3) for r in XGF, τ in TGF]
    const VFF = zeros(length(XGF), length(TGF))

    "Crossing of the PIECEWISE-LINEAR interpolated T — the field the kernel actually sees."
    function tau_cross(Tfo)
        f(τ) = KC.interpolate_2d_cpu(XGF, TGF, TFF, 0.0, τ) - Tfo
        lo, hi = TAU0, last(TGF)
        f(lo) > 0 || error("T(τ₀) already below Tfo"); f(hi) < 0 || error("never crosses")
        for _ in 1:200
            mid = (lo + hi)/2
            f(mid) > 0 ? (lo = mid) : (hi = mid)
        end
        (lo + hi)/2
    end
    const TAUC = tau_cross(TFO)

    function fo_run(; dt, interp, tf = 7.0, N = NP, Tfo = TFO, seed = 5, DsT = 1e6)
        Random.seed!(seed)
        rg = collect(0.0:0.5:20.0); pg = collect(0.05:0.1:8.0); dens = ones(length(pg), length(rg))
        x0 = 2.0 .* randn(2, N); p0 = 0.8 .* randn(2, N)
        Base.invokelatest(simulate_ensemble_bulk, GPUBackend(), rg, pg, dens, TFF, VFF, (XGF, TGF);
            x_init = x0, p_init = p0, N_particles = N, Δt = dt, initial_time = TAU0,
            final_time = TAU0 + tf, save_interval = 1.0, m = M, DsT = DsT, dimensions = 2,
            Tfo = Tfo, freezeout_capture = true, freezeout_interp = interp)
    end

    @testset "F1 freeze-out capture books the exact T = Tfo crossing" begin
        @test 3.0 < TAUC < 6.0
        dt = 2e-3
        fo = fo_run(; dt = dt, interp = true)
        @test keys(fo) == (:pos, :mom, :tau, :flag)
        @test size(fo.pos) == (2, NP) && size(fo.mom) == (2, NP) && length(fo.tau) == NP
        @test all(==(1.0), fo.flag)                       # every particle crosses on this background
        @test all(isfinite, fo.pos) && all(isfinite, fo.mom)
        # T(τ) has no radial dependence, so the crossing is the SAME for every particle and equal
        # to the host-computed crossing of the interpolated field. `interp = true` must find it to
        # far better than a step.
        @test maximum(abs, fo.tau .- TAUC) < 1e-6
        @info "F1 interp=true: |τ_fo − τ_c| max = $(maximum(abs, fo.tau .- TAUC)) (Δt = $dt, τ_c = $(round(TAUC; digits = 6)))"
    end

    @testset "F2 freezeout_interp removes the booking lag — the raw mode's lag is PREDICTED exactly" begin
        # Because T depends on τ alone, EVERY particle crosses on the same step. That removes the
        # averaging one would normally rely on — the raw lag is not "≈ Δt/2 on average", it is one
        # deterministic offset — but it makes that offset exactly predictable, which is a sharper
        # statement: the raw mode must book the FIRST SAMPLED TIME BELOW Tfo, and nothing else.
        "first τ = τ₀ + n·Δt at which the interpolated T drops below Tfo — what raw mode must book"
        function first_below(dt)
            n = 1
            while KC.interpolate_2d_cpu(XGF, TGF, TFF, 0.0, TAU0 + n*dt) >= TFO
                n += 1
                n > 10^7 && error("no crossing")
            end
            TAU0 + n*dt
        end
        rawerr = Float64[]; interperr = Float64[]
        for dt in (8e-3, 4e-3, 1e-3)
            raw = fo_run(; dt = dt, interp = false, N = 20_000)
            itp = fo_run(; dt = dt, interp = true,  N = 20_000)
            want = first_below(dt)
            @test all(==(1.0), raw.flag) && all(==(1.0), itp.flag)
            # the raw booking is the sampled step, to the last bit, for every particle
            @test maximum(abs, raw.tau .- want) < 1e-12
            # ... and that step sits within one Δt above the true crossing, never below it
            @test 0 < want - TAUC <= dt + 1e-12
            push!(rawerr, want - TAUC)
            push!(interperr, maximum(abs, itp.tau .- TAUC))
        end
        # The interpolated mode is better at EVERY step, by orders of magnitude, and its residual
        # does NOT track Δt: it is the curvature of T across a tgrid node (the τ table is spaced
        # 0.05 fm, so the two bracketing samples occasionally straddle a kink, and linear
        # interpolation in T is exact only within a cell). The raw error is O(Δt) by construction.
        @test all(interperr .< 1e-4)
        @test all(interperr .< 0.2 .* rawerr)
        @test rawerr[1] > rawerr[3]                       # the raw lag shrinks with Δt
        @info "F2 raw lag (τ_booked − τ_c) at Δt = 8e-3, 4e-3, 1e-3: $(round.(rawerr; sigdigits = 3));  interp residual: $(round.(interperr; sigdigits = 3))"
    end

    @testset "F3 the latch fires once and the run does not stop at it" begin
        # Running 4 fm/c longer past the crossing must not move anything that was booked: the
        # particles keep propagating (documented behaviour) but `fo_flag` gates the write.
        a = fo_run(; dt = 4e-3, interp = true, tf = 5.5,  N = 40_000, seed = 9)
        b = fo_run(; dt = 4e-3, interp = true, tf = 9.0, N = 40_000, seed = 9)
        @test all(==(1.0), a.flag) && all(==(1.0), b.flag)
        @test maximum(abs, a.tau .- b.tau) < 1e-9
        # positions are re-sampled from the same seed, so they must agree to the noise the (tiny)
        # drag adds — the point is that the LATER run did not overwrite the booking with a later state
        @test isapprox(mean(sqrt.(sum(abs2, a.pos; dims = 1))), mean(sqrt.(sum(abs2, b.pos; dims = 1))); rtol = 1e-3)
        # a particle that never gets cold enough is left unflagged rather than booked at t_final
        cold = fo_run(; dt = 4e-3, interp = true, tf = 1.0, N = 20_000, Tfo = 0.01)
        @test all(==(0.0), cold.flag)
        @test all(==(0.0), cold.tau)
    end

    @testset "F4 the booked position is the trajectory at the crossing" begin
        # With a drag time of ~10⁶ fm the particles free-stream, so x(τ_c) = x₀ + (p/E)(τ_c − τ₀)
        # is known in closed form. This checks the kernel interpolates POSITION to the crossing too,
        # not just the time.
        Random.seed!(31)
        N = 40_000
        rg = collect(0.0:0.5:20.0); pg = collect(0.05:0.1:8.0); dens = ones(length(pg), length(rg))
        x0 = 2.0 .* randn(2, N); p0 = 0.8 .* randn(2, N)
        fo = Base.invokelatest(simulate_ensemble_bulk, GPUBackend(), rg, pg, dens, TFF, VFF, (XGF, TGF);
            x_init = x0, p_init = p0, N_particles = N, Δt = 2e-3, initial_time = TAU0,
            final_time = TAU0 + 7.0, save_interval = 1.0, m = M, DsT = 1e8, dimensions = 2,
            Tfo = TFO, freezeout_capture = true, freezeout_interp = true)
        E = sqrt.(M^2 .+ vec(sum(abs2, p0; dims = 1)))
        want = x0 .+ (p0 ./ E') .* (TAUC - TAU0)
        @test maximum(abs, fo.pos .- want) < 5e-3          # free streaming + a residual O(Δt) kick
        @test maximum(abs, fo.mom .- p0) < 5e-3            # momenta essentially untouched
        @info "F4 |x_fo − x₀ − v(τ_c−τ₀)| max = $(round(maximum(abs, fo.pos .- want); sigdigits = 3))"
    end

    @testset "F5 integrator_mode = 1 MAKES THE Δt BIAS WORSE — it does not remove it" begin
        # 🔴🔴 THE DOCUMENTED CLAIM IS FALSE. `integrator_mode = 1` is described in the README as
        # "drift-midpoint drag (O(ε²))" and in the kernel as removing the O(ε) term of the
        # pre-point drag. Measured at N = 10⁶ (SEM 0.14 %) on a uniform bath, against the 2-D
        # Jüttner ⟨p²⟩ the engine cannot fake:
        #
        #     ηΔt      mode 0      mode 1     ratio
        #     0.05    −1.03 %     −1.74 %      1.70
        #     0.10    −1.36 %     −3.49 %      2.57
        #     0.20    −3.23 %     −6.53 %      2.02
        #     0.30    −4.26 %     −9.19 %      2.16
        #
        # Mode 1 roughly DOUBLES the bias, at every step, with the same sign, and still scales
        # linearly in Δt — so it is O(ε), not O(ε²).
        #
        # WHY. The predictor is a NOISE-FREE drag half-step, p_mid = exp(−η_eff Δt/2)·p, so it can
        # only SHRINK |p|. Hence E_mid < E always, hence η_eff = η_D·M/E_mid > the pre-point value
        # ALWAYS: the correction is one-sided by construction. A true midpoint would have the noise
        # pushing ⟨p²⟩ up as much as the drag pushes it down — that is what stationarity means — so
        # the correct midpoint energy is ≈ the pre-point energy. Dropping the noise (which the
        # kernel comment does deliberately, to stop η_eff correlating with ξ) trades a correlation
        # error for a systematic drift of the same order and the same sign as the error it targets.
        #
        # NOT A PRODUCTION INCIDENT: every driver in Julia/Projects/ maps its "integrator" option
        # with `== "mid" ? 1 : 0` and no recipe passes "mid", so every Langevin product ever made
        # used mode 0. Reported, not fixed — a fix changes what mode 1 computes. See CHANGELOG 0.2.1.
        using QuadGK
        T = 0.30
        function jm(g)
            n, _ = quadgk(p -> g(p)*p*exp(-(sqrt(p^2 + M^2) - M)/T), 0, Inf; rtol = 1e-10)
            d, _ = quadgk(p ->      p*exp(-(sqrt(p^2 + M^2) - M)/T), 0, Inf; rtol = 1e-10)
            n/d
        end
        p2ref = jm(p -> p^2)
        DsT = 0.11634
        η = (T^2/(M*DsT))/GevInvTofm                 # fm⁻¹
        xg = collect(0.0:0.5:400.0); tg = collect(0.0:0.25:(14/η + 2))
        Tf = fill(T, length(xg), length(tg)); Vf = zeros(length(xg), length(tg))
        rg = collect(0.0:0.5:20.0); pg = collect(0.05:0.1:8.0); dens = ones(length(pg), length(rg))
        N = 400_000
        sem = sqrt(2/N)
        bias = Dict{Tuple{Float64,Int},Float64}()
        for ed in (0.10, 0.30), mode in (0, 1)
            dt = ed/η
            Random.seed!(41)
            p0 = sqrt(M*T) .* randn(2, N)
            _, mom, _ = Base.invokelatest(simulate_ensemble_bulk, GPUBackend(), rg, pg, dens, Tf, Vf, (xg, tg);
                x_init = zeros(2, N), p_init = p0, N_particles = N, Δt = dt, initial_time = 0.0,
                final_time = 12/η, save_interval = 6/η, m = M, DsT = DsT, dimensions = 2,
                Tfo = 0.0, integrator_mode = mode)
            bias[(ed, mode)] = mean(sum(abs2, mom[end]; dims = 1))/p2ref - 1
        end
        @info "F5 ⟨p²⟩/Jüttner − 1  (SEM $(round(sem; sigdigits=2))):  " *
              "ηΔt=0.10 mode0 $(round(bias[(0.10,0)]; sigdigits=3)) mode1 $(round(bias[(0.10,1)]; sigdigits=3));  " *
              "ηΔt=0.30 mode0 $(round(bias[(0.30,0)]; sigdigits=3)) mode1 $(round(bias[(0.30,1)]; sigdigits=3))"
        for ed in (0.10, 0.30)
            b0, b1 = bias[(ed, 0)], bias[(ed, 1)]
            @test b0 < -4sem                        # the pre-point bias is real and negative
            @test b1 < -4sem                        # so is mode 1's
            @test abs(b1) > abs(b0) + 4sem          # and mode 1's is BIGGER — the finding
            @test 1.3 < b1/b0 < 3.5                 # by roughly a factor of two
            # what the README promises, pinned as expected-fail so a real fix turns this red
            @test_broken abs(b1) < abs(b0)
        end
        # the CPU still refuses the mode it does not implement
        @test_throws ErrorException simulate_ensemble_bulk(CPUBackend(), rg, pg, dens, Tf, Vf, (xg, tg);
            N_particles = 10, Δt = 1e-2, final_time = 0.1, save_interval = 0.1, integrator_mode = 1)
    end
end
