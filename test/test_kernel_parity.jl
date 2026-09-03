# ==============================================================================================
# test_kernel_parity.jl — DETERMINISTIC CPU ↔ GPU parity, kernel by kernel (2026-08-31 audit).
#
# WHY THIS EXISTS. Until now the only thing holding the two backends together was an ENSEMBLE
# comparison through the full driver at 3 % (`bench_gpu_parity.jl`, `regression_corpus.jl`). That
# is a weak instrument: CURAND is unseedable, so a real per-particle disagreement has to exceed the
# Monte-Carlo scatter of ⟨p²⟩ at N = 50 000 (≈1.5 %) before any gate notices. The 0.2.0 bugs — the
# GPU interpolant extrapolating past the table, the GPU boosts not clamping |v| < 1 — were found
# only because they produced NaNs; a 5 % silent drift would have sailed through.
#
# The GPU kernels take their randomness as PRE-GENERATED ARRAYS (ξ, u_collide, u_sample,
# dir_gauss). So the two backends can be driven with the SAME inputs AND THE SAME NOISE and
# compared per particle. That is what this file does, and it makes the comparison exact instead of
# statistical: every gate below is at 1e-12 relative or tighter, and several are bit-for-bit.
#
# Any kernel-level divergence that is REAL and intended is recorded here as such (see D1), not
# quietly reconciled — the point is that the difference is a decision on the record, not a drift.
#
#   julia --project=Julia Julia/LangevInMedium.jl/test/test_kernel_parity.jl
# ==============================================================================================
ENV["LIM_QUIET"] = "1"
using Test, Random, Statistics, LinearAlgebra, Printf

const HAVE_CUDA = try
    @eval using CUDA
    Base.invokelatest(() -> CUDA.functional())
catch
    false
end
using LangevInMedium

if !HAVE_CUDA
    @warn "CUDA not functional — CPU/GPU kernel parity SKIPPED (this is the only gate that compares the backends exactly)"
else
    const KC = LangevInMedium.KernelsCPU
    const KG = LangevInMedium.Simulate.KernelsGPU
    const TR = LangevInMedium.Transport

    launch(f, N, args...) = (@cuda threads=256 blocks=cld(N, 256) f(args...); CUDA.synchronize())

    # ⚠ WHY NOTHING WITH A MULTIPLY-ADD IN IT IS COMPARED WITH `==`.
    # The NVPTX backend CONTRACTS `a*b + c*d` into fused multiply-adds; the host does not. So even
    # a character-for-character identical expression — `v00*(1-xd) + v10*xd`, the bilinear
    # interpolant — returns results one ulp apart on the two backends, by construction and with no
    # bug involved. Exact equality is therefore reserved for kernels with nothing to contract (a
    # pure multiply, a copy); everything else is held at a few ulps, which is still ~13 orders of
    # magnitude tighter than the 3 % ensemble gate this file exists to replace.
    const ULP = 8e-16                      # ≈ 4 ulps of Float64
    ulpeq(A, B) = maximum(abs.(A .- B) ./ max.(abs.(A), abs.(B), 1e-300)) <= ULP
    """
    Worst absolute deviation between a host array and a device array, and that deviation relative
    to the ARRAY'S OWN SCALE. Deliberately not a per-element relative error: a momentum component
    is a difference of larger numbers and routinely lands within an ulp of zero, where a pointwise
    ratio diverges while nothing is wrong. Scale-relative is the meaningful statement.
    """
    function dev(A, dA)
        B = Array(dA)
        aerr = maximum(abs, A .- B)
        (aerr, aerr / max(maximum(abs, A), 1e-300))
    end

    # ── a background both backends see identically ───────────────────────────────────────────────
    const XG = collect(0.0:0.25:20.0)
    const TG = collect(0.4:0.1:2.4)
    const TF = [max(0.12, 0.45*exp(-r^2/60)*(0.4/τ)^(1/3)) for r in XG, τ in TG]
    const VF = [0.6*r/(r + 4.0)*min(1.0, τ/2.0) for r in XG, τ in TG]
    const V2 = [0.05*r/(r + 6.0) for r in XG, τ in TG]
    const M, DST, NP = 1.5, 0.11634, 4096
    const d_XG, d_TG, d_TF, d_VF, d_V2 = CuArray(XG), CuArray(TG), CuArray(TF), CuArray(VF), CuArray(V2)

    @testset "P1 interpolate_2d — pointwise, including off-table and degenerate cells" begin
        # a kernel that just evaluates the interpolant at a list of query points
        function probe!(out, x, y, v, qx, qy, n)
            i = (blockIdx().x - 1)*blockDim().x + threadIdx().x
            i <= n && (@inbounds out[i] = KG.interpolate_2d_cuda(x, y, v, qx[i], qy[i]))
            return
        end
        Random.seed!(5)
        nq = 20_000
        # deliberately spill far outside the table in BOTH axes and both directions
        qx = vcat(range(-30.0, 60.0; length = nq ÷ 2), 20.0 .* rand(nq ÷ 2))
        qy = vcat(range(-5.0, 12.0; length = nq ÷ 2), 0.4 .+ 2.0 .* rand(nq ÷ 2))
        # ... and land exactly on nodes, where an off-by-one in the cell search shows up
        qx[1:length(XG)] .= XG; qy[1:length(XG)] .= TG[1]
        for (nm, vals) in (("T", TF), ("v", VF))
            dvals = CuArray(vals)
            cpu = [KC.interpolate_2d_cpu(XG, TG, vals, qx[i], qy[i]) for i in 1:nq]
            dout = CUDA.zeros(Float64, nq)
            launch(probe!, nq, dout, d_XG, d_TG, dvals, CuArray(qx), CuArray(qy), nq)
            @test ulpeq(Array(dout), cpu)     # same clamp, same cell, same weights (FMA aside)
            @test all(isfinite, Array(dout))
            nm == "v" && @test all(abs.(Array(dout)) .<= 1.0)
        end
        # a NON-UNIFORM axis — the binary search must not assume constant spacing
        xnu = [0.0, 0.05, 0.4, 3.0, 3.1, 9.0, 20.0]
        Tnu = [0.2 + 0.01*r + 0.02*τ for r in xnu, τ in TG]
        cpu = [KC.interpolate_2d_cpu(xnu, TG, Tnu, qx[i], qy[i]) for i in 1:nq]
        dout = CUDA.zeros(Float64, nq)
        launch(probe!, nq, dout, CuArray(xnu), d_TG, CuArray(Tnu), CuArray(qx), CuArray(qy), nq)
        @test ulpeq(Array(dout), cpu)
    end

    @testset "P2 transport-time spline evaluators — exact in range, and what happens outside" begin
        function probe!(out, Ts, Tmin, invdT, vals, n)
            i = (blockIdx().x - 1)*blockDim().x + threadIdx().x
            i <= n && (@inbounds out[i] = KG._eval_time_spline_cuda(Ts[i], Tmin, invdT, vals))
            return
        end
        for build in (build_tau_drag_spline, build_taun_current_spline)
            T0, invdT, vals = build(M, DST; Tmin = 0.12, Tmax = 0.50, nT = 1024)
            Ts = vcat(collect(range(-1.0, 3.0; length = 4000)),
                      [0.12 + (k-1)*(0.50-0.12)/1023 for k in 1:1024])   # every node, exactly
            cpu = [eval_tau_n_spline(T, T0, invdT, vals) for T in Ts]
            dout = CUDA.zeros(Float64, length(Ts))
            launch(probe!, length(Ts), dout, CuArray(Ts), T0, invdT, CuArray(vals), length(Ts))
            gpu = Array(dout)
            inr = findall(T -> 0.12 <= T <= 0.50, Ts)
            out = findall(T -> T < 0.12 || T > 0.50, Ts)
            @test ulpeq(gpu[inr], cpu[inr])            # in range: sub-ulp, measured 2.2e-16
            # OUTSIDE the range both backends extrapolate (the defect documented on
            # `Transport.eval_tau_n_spline`), and `(1-t)y0 + t·y1` with |t| ~ 10³–10⁴ AMPLIFIES the
            # one-ulp FMA difference by exactly that factor: measured worst 4.4e-11 at T = 0.749,
            # where the extrapolated "time" is already on its way through zero. So the unclamped
            # weight costs CPU/GPU parity as well as correctness — a second argument for the clamp.
            rel = maximum(abs.(gpu[out] .- cpu[out]) ./ max.(abs.(cpu[out]), abs.(gpu[out]), 1e-300))
            @test rel < 1e-8
            @test rel > ULP                            # i.e. it really is worse than in range
        end
    end

    # ── shared per-particle state ────────────────────────────────────────────────────────────────
    function state(; pdim = 2, seed = 11, spread = 6.0)
        rng = MersenneTwister(seed)
        pos = spread .* randn(rng, 2, NP)
        pos[:, 1] .= 0.0                       # r < eps: the "leave it alone" branch
        pos[:, 2] .= 1e3                       # far outside the table
        mom = 0.9 .* randn(rng, pdim, NP)
        mom[:, 3] .= 0.0                       # p = 0: the p̂ fallback branch
        (mom, pos)
    end

    @testset "P3 boosts — every switch combination, per particle" begin
        for pdim in (2, 3), rel in (true, false), usev2 in (false, true)
            mom, pos = state(; pdim = pdim)
            dmom, dpos = CuArray(mom), CuArray(pos)
            for (cpuk, gpuk) in ((KC.kernel_boost_to_rest_frame_cpu!, KG.kernel_boost_to_rest_frame_gpu!),
                                 (KC.kernel_boost_to_lab_frame_cpu!,  KG.kernel_boost_to_lab_frame_gpu!))
                cpuk(mom, pos, XG, TG, VF, M, NP, 7, 2e-3, 0.4;
                     radial_mode = false, V2Evolution = usev2 ? V2 : nothing, psi2 = 0.3, relativistic = rel)
                launch(gpuk, NP, dmom, dpos, d_XG, d_TG, d_VF, M, NP, 7, 2e-3, 0.4,
                       false, usev2, d_V2, 0.3, rel)
                a, r = dev(mom, dmom)          # measured: 1-2 ulps (4.4e-16) in every combination
                @test a < 1e-13 && r < 1e-13
            end
        end
        # radial mode (dimensions = 1)
        mom = 0.9 .* randn(MersenneTwister(12), 1, NP); pos = abs.(6.0 .* randn(MersenneTwister(13), 1, NP))
        dmom, dpos = CuArray(mom), CuArray(pos)
        for rel in (true, false)
            KC.kernel_boost_to_rest_frame_cpu!(mom, pos, XG, TG, VF, M, NP, 7, 2e-3, 0.4; radial_mode = true, relativistic = rel)
            launch(KG.kernel_boost_to_rest_frame_gpu!, NP, dmom, dpos, d_XG, d_TG, d_VF, M, NP, 7, 2e-3, 0.4, true, false, d_V2, 0.0, rel)
            a, _ = dev(mom, dmom); @test a < 1e-13
        end
    end

    @testset "P4 forces kernel — relativistic × proper_time_kicks × radial, SAME noise" begin
        for pdim in (2, 3), rel in (true, false), ptk in (false, true)
            mom, pos = state(; pdim = pdim, seed = 21)
            ξ = randn(MersenneTwister(31), pdim, NP)
            rd = randn(MersenneTwister(32), pdim, NP); rd ./= sqrt.(sum(abs2, rd; dims = 1))
            T0, invdT, vals = build_tau_drag_spline(M, DST; Tmin = minimum(TF), Tmax = maximum(TF), nT = 1024)
            det = zeros(pdim, NP); sto = zeros(pdim, NP)
            pm = zeros(NP); pu = zeros(pdim, NP); ηv = zeros(NP); kLv = zeros(NP); kTv = zeros(NP)
            KC.kernel_compute_all_forces_cpu!(TF, XG, TG, mom, pos, pm, pu, ηv, kLv, kTv, ξ, det, sto,
                2e-3, M, rd, pdim, NP, 7, 0.4, DST; tau_Tmin = T0, tau_invdT = invdT, tau_vals = vals,
                radial_mode = false, relativistic = rel, proper_time_kicks = ptk, Vfield = VF)
            ddet = CUDA.zeros(Float64, pdim, NP); dsto = CUDA.zeros(Float64, pdim, NP)
            dpm = CUDA.zeros(Float64, NP); dpu = CUDA.zeros(Float64, pdim, NP)
            dη = CUDA.zeros(Float64, NP); dkL = CUDA.zeros(Float64, NP); dkT = CUDA.zeros(Float64, NP)
            launch(KG.kernel_compute_all_forces_gpu!, NP, CuArray(TF), d_XG, d_TG, CuArray(mom), CuArray(pos),
                dpm, dpu, dη, dkL, dkT, CuArray(ξ), ddet, dsto, 2e-3, M, CuArray(rd), pdim, NP, 7, 0.4, DST,
                T0, invdT, CuArray(vals), false, Int32(0), rel, ptk, d_VF)
            _, rd_ = dev(det, ddet); _, rs = dev(sto, dsto)
            # Judged RELATIVE TO EACH TERM'S OWN SCALE: the deterministic term is O(0.02) GeV while
            # the noise amplitude is √κ ≈ 9 GeV/√fm, so a common absolute threshold would compare
            # two quantities three decades apart. Measured worst, over all eight combinations:
            # det 3.4e-16 abs on a 0.015 scale, sto 1.6e-13 abs on a 10 scale — both ≈1.6e-14
            # relative. The residual is `exp(-η_eff·Δt)` differing by an ulp between host libm and
            # the device, amplified through (a-1)·p and √((1-a²)/(2η_eff Δt)).
            @test rd_ < 1e-12 && rs < 1e-12
            _, rη = dev(ηv, dη); @test rη < 1e-12
        end
    end

    @testset "P5 momentum update, Bjorken redshift, save kernels" begin
        pdim = 3
        mom = randn(MersenneTwister(41), pdim, NP)
        det = randn(MersenneTwister(42), pdim, NP); sto = randn(MersenneTwister(43), pdim, NP)
        dmom = CuArray(mom)
        KC.kernel_update_momenta_LRF_cpu!(mom, det, sto, 2e-3, pdim, NP)
        launch(KG.kernel_update_momenta_LRF_gpu!, NP, dmom, CuArray(det), CuArray(sto), 2e-3, pdim, NP)
        # `det + √Δt·sto` IS contractible, so this is ulp-level rather than exact — and judged
        # against the array scale, since an updated momentum component routinely lands near zero.
        _, r = dev(mom, dmom); @test r < 1e-15
        # redshift: a pure multiply — nothing to contract — so it must agree to the LAST BIT, and
        # over 200 chained steps, which is where a one-ulp-per-step drift would show
        m2 = randn(MersenneTwister(44), pdim, NP); dm2 = CuArray(m2)
        for step in 1:200
            KC.kernel_bjorken_redshift_cpu!(m2, 3, step, 2e-3, 0.4, NP)
            launch(KG.kernel_bjorken_redshift_gpu!, NP, dm2, 3, step, 2e-3, 0.4, NP)
        end
        @test Array(dm2) == m2
        # saves
        H = zeros(pdim, NP, 3); dH = CUDA.zeros(Float64, pdim, NP*3)
        KC.kernel_save_momenta_cpu!(H, m2, 2, NP)
        launch(KG.kernel_save_momenta_gpu!, NP, dH, dm2, 2, NP, pdim)
        @test Array(dH)[:, NP+1:2NP] == H[:, :, 2]
        # the position twin: the device history is a flat (dims, N·(saves+1)) buffer, so the slice
        # arithmetic `(save_idx-1)*N` is part of what is under test, not just the copy
        _, pos = state(; pdim = pdim, seed = 45)
        Pc = zeros(2, NP, 3); dP = CUDA.zeros(Float64, 2, NP*3)
        KC.kernel_save_positions_cpu!(Pc, pos, 3, NP)
        launch(KG.kernel_save_positions_gpu!, NP, dP, CuArray(pos), 3, NP, 2)
        @test Array(dP)[:, 2NP+1:3NP] == Pc[:, :, 3]
        @test all(iszero, Array(dP)[:, 1:2NP])          # earlier slots untouched
    end

    @testset "P6 position update — streaming, diffusion, reflection" begin
        for pdim in (2, 3), rel in (true, false), diff in (false, true), refl in (false, true)
            mom, pos = state(; pdim = pdim, seed = 51, spread = 14.0)   # some particles past the rim
            ξp = randn(MersenneTwister(52), 2, NP)
            dmom, dpos = CuArray(mom), CuArray(pos)
            # The CPU draws its diffusion noise INSIDE the kernel from the global RNG while the GPU
            # takes a pre-generated array, so the two cannot be compared draw for draw with
            # diffusion ON. It is still covered — deterministically — by the ZERO-NOISE block
            # below, which isolates everything about the diffusion branch except the Gaussian.
            diff && continue
            KC.kernel_update_positions_cpu!(pos, mom, M, 2e-3, NP, 7, 0.4, XG, TG, TF, DST;
                dimensions = 2, momentum_dimensions = pdim, radial_mode = false,
                position_diffusion = false, reflecting_boundary = refl, relativistic = rel)
            launch(KG.kernel_update_positions_gpu!, NP, dpos, dmom, M, 2e-3, NP, 7, 0.4,
                d_XG, d_TG, CuArray(TF), DST, 2, false, false, refl, CuArray(ξp), rel, pdim)
            ap, _ = dev(pos, dpos); am, _ = dev(mom, dmom)
            @test ap < 1e-13 && am < 1e-13
        end
        # ── position_diffusion, deterministically ────────────────────────────────────────────────
        # Hand the GPU a ZERO noise array. In the Cartesian branch the only diffusion term is
        # σ·ξ, so the result must then be bit-comparable to the CPU with diffusion OFF — which
        # tests that the GPU computes D = DsT/T(r,τ)/ħc, gates on D > 0 and applies the noise
        # PURELY multiplicatively (an additive term or a wrong D would survive ξ = 0 and show up).
        for pdim in (2, 3)
            mom, pos = state(; pdim = pdim, seed = 55, spread = 10.0)
            posC = copy(pos); momC = copy(mom)
            dmom, dpos = CuArray(mom), CuArray(pos)
            KC.kernel_update_positions_cpu!(posC, momC, M, 2e-3, NP, 7, 0.4, XG, TG, TF, DST;
                dimensions = 2, momentum_dimensions = pdim, position_diffusion = false)
            launch(KG.kernel_update_positions_gpu!, NP, dpos, dmom, M, 2e-3, NP, 7, 0.4,
                d_XG, d_TG, CuArray(TF), DST, 2, false, true, false,
                CUDA.zeros(Float64, 2, NP), true, pdim)
            a, _ = dev(posC, dpos); @test a < 1e-13
        end
        # radial: with ξ = 0 the diffusion branch still contributes the geometric drift (D/r)·Δt,
        # so the GPU must differ from a no-diffusion CPU run by EXACTLY that, computed here from
        # the published formula (D = DsT/T/ħc) rather than from the kernel.
        let
            mom = randn(MersenneTwister(56), 1, NP)
            pos = 0.5 .+ abs.(4.0 .* randn(MersenneTwister(57), 1, NP))
            posC = copy(pos); momC = copy(mom)
            dmom, dpos = CuArray(copy(mom)), CuArray(copy(pos))
            KC.kernel_update_positions_cpu!(posC, momC, M, 2e-3, NP, 7, 0.4, XG, TG, TF, DST;
                dimensions = 1, momentum_dimensions = 1, radial_mode = true, position_diffusion = false)
            launch(KG.kernel_update_positions_gpu!, NP, dpos, dmom, M, 2e-3, NP, 7, 0.4,
                d_XG, d_TG, CuArray(TF), DST, 1, true, true, false,
                CUDA.zeros(Float64, 1, NP), true, 1)
            gpu = Array(dpos)
            r_axis_eps = max(1e-12, 0.5*abs(XG[2] - XG[1]))
            worst = 0.0
            for i in 1:NP
                r_safe = max(abs(pos[1, i]), r_axis_eps)
                T = KC.interpolate_2d_cpu(XG, TG, TF, r_safe, 7*2e-3 + 0.4)
                D = (DST/max(T, eps()))/(1/LangevInMedium.GevInvTofm)
                worst = max(worst, abs(gpu[1, i] - (posC[1, i] + (D/r_safe)*2e-3)))
            end
            @test worst < 1e-13
        end

        # radial mode, including the r < 0 reflection
        mom = randn(MersenneTwister(53), 1, NP); pos = 0.02 .* randn(MersenneTwister(54), 1, NP)
        dmom, dpos = CuArray(mom), CuArray(pos)
        KC.kernel_update_positions_cpu!(pos, mom, M, 2e-3, NP, 7, 0.4, XG, TG, TF, DST;
            dimensions = 1, momentum_dimensions = 1, radial_mode = true, reflecting_boundary = true)
        launch(KG.kernel_update_positions_gpu!, NP, dpos, dmom, M, 2e-3, NP, 7, 0.4,
            d_XG, d_TG, CuArray(TF), DST, 1, true, false, true, CUDA.zeros(Float64, 1, NP), true, 1)
        ap, _ = dev(pos, dpos); @test ap < 1e-13
    end

    @testset "P7 set_to_fluid_velocity" begin
        for rel in (true, false)
            mom, pos = state(; seed = 61)
            dmom, dpos = CuArray(mom), CuArray(pos)
            KC.kernel_set_to_fluid_velocity_cpu!(mom, pos, XG, TG, VF, M, NP, 7, 2e-3, 0.4; relativistic = rel)
            launch(KG.kernel_set_to_fluid_velocity_gpu!, NP, dmom, dpos, d_XG, d_TG, d_VF, M, NP, 7, 2e-3, 0.4, false, rel)
            a, _ = dev(mom, dmom); @test a < 1e-13
        end
    end

    @testset "P8 RTA/BGK — same collision decision, same |p*|, same direction" begin
        # The two paths draw |p*| differently BY DESIGN (CPU: rejection; GPU: inverse-CDF table), so
        # they cannot be compared draw for draw. What CAN be compared exactly is everything else:
        # the collision DECISION (Pcol) and the direction, given the same uniforms and Gaussians.
        # U7 in test_kernel_units.jl is the distributional check on the two |p*| samplers.
        pdim = 2
        mom, pos = state(; pdim = pdim, seed = 71)
        T0, invdT, tvals = build_taun_current_spline(M, DST; Tmin = minimum(TF), Tmax = maximum(TF), nT = 1024)
        icdf, nU, nT, itmin, iinv = TR.build_juttner_invcdf(M, pdim; Tmin = minimum(TF), Tmax = maximum(TF))
        uc = rand(MersenneTwister(72), NP)
        us = rand(MersenneTwister(73), NP)
        dg = randn(MersenneTwister(74), pdim, NP)
        for ptk in (false, true)
            dmom = CuArray(copy(mom)); dpos = CuArray(pos)
            launch(KG.kernel_rta_collision_gpu!, NP, dmom, dpos, d_XG, d_TG, CuArray(TF),
                2e-3, M, NP, 7, 0.4, DST, T0, invdT, CuArray(tvals),
                CuArray(icdf), nU, nT, itmin, iinv, pdim, false,
                CuArray(uc), CuArray(us), CuArray(dg), ptk, d_VF)
            out = Array(dmom)
            # Pcol recomputed here from the published formula — an independent reimplementation
            ncol_expected = 0
            for i in 1:NP
                r = hypot(pos[1, i], pos[2, i])
                T = max(KC.interpolate_2d_cpu(XG, TG, TF, r, 7*2e-3 + 0.4), 0.0)
                τn = (DST > 0 && T > 0) ? eval_tau_n_spline(T, T0, invdT, tvals) : 0.0
                dil = 1.0
                if ptk
                    v = clamp(KC.interpolate_2d_cpu(XG, TG, VF, r, 7*2e-3 + 0.4), -0.999999, 0.999999)
                    γv = 1/sqrt(1 - v*v)
                    kr = r > 1e-12 ? (mom[1, i]*pos[1, i] + mom[2, i]*pos[2, i])/r : 0.0
                    Es = sqrt(sum(abs2, view(mom, :, i)) + M^2)
                    dil = 1/max(γv*(1 + v*kr/Es), 1e-6)
                end
                Pcol = (τn > 0 && isfinite(τn)) ? clamp(2e-3*dil/τn, 0.0, 1.0) : 1.0
                collided = uc[i] < Pcol
                collided && (ncol_expected += 1)
                if !collided
                    @test out[:, i] == mom[:, i]              # untouched, bit for bit
                else
                    # direction must be dir_gauss normalised — |p*| aside, the ANGLE is exact
                    n = hypot(dg[1, i], dg[2, i]); pn = hypot(out[1, i], out[2, i])
                    if n > 1e-8 && pn > 1e-12
                        @test isapprox(out[1, i]/pn, dg[1, i]/n; atol = 1e-12)
                        @test isapprox(out[2, i]/pn, dg[2, i]/n; atol = 1e-12)
                    end
                end
            end
            @test ncol_expected > 0
        end
    end

    @testset "P9 verbose = true prints the device banner instead of crashing" begin
        # `print_cuda_status` runs only under `verbose = true`; it was moved off the default path in
        # 0.2.0 and has had no coverage since, so a typo in it would only surface for whoever turned
        # the flag on. This is a smoke test: the run must complete and return the normal shape.
        xg = collect(0.0:0.5:30.0); tg = collect(0.4:0.25:2.0)
        Tf = fill(0.30, length(xg), length(tg)); Vf = zeros(length(xg), length(tg))
        rg = collect(0.0:0.5:20.0); pg = collect(0.05:0.1:8.0); dens = ones(length(pg), length(rg))
        t, mom, pos = simulate_ensemble_bulk(GPUBackend(), rg, pg, dens, Tf, Vf, (xg, tg);
            x_init = zeros(2, 256), p_init = 0.5 .* randn(2, 256), N_particles = 256, Δt = 1e-2,
            initial_time = 0.4, final_time = 0.6, save_interval = 0.1, m = M, DsT = DST,
            dimensions = 2, Tfo = 0.0, verbose = true)
        @test size(mom[end]) == (2, 256) && size(pos[end]) == (2, 256)
        @test all(isfinite, mom[end]) && all(isfinite, pos[end])
    end

    @testset "P10 2-D BACKGROUND — CPU/GPU parity of the trilinear path" begin
        # The 2-D background (T[x,y,t] plus a (u^x, u^y) vector pair) was added CPU-first and the
        # GPU kernels mirror it. The CPU tests prove the physics reduces to the shipped radial path;
        # THIS proves the two backends implement that same physics, per particle, with the same
        # noise -- the only instrument here that can see a silent GPU-only divergence, which is
        # exactly the class the 0.2.0 interpolant and clamp bugs belonged to.
        YG = collect(-20.0:0.5:20.0)
        XG2 = collect(-20.0:0.5:20.0)
        T3  = [max(0.12, 0.45*exp(-(x^2+y^2)/60)*(0.4/τ)^(1/3)) for x in XG2, y in YG, τ in TG]
        # a flow that is nowhere parallel to r-hat, so the direction logic is exercised and not
        # accidentally satisfied by the radial special case
        UX  = [0.55*tanh(hypot(x,y)/5)*(x/max(hypot(x,y),1e-9)) - 0.18*(y/20.0) for x in XG2, y in YG, τ in TG]
        UY  = [0.55*tanh(hypot(x,y)/5)*(y/max(hypot(x,y),1e-9)) + 0.18*(x/20.0) for x in XG2, y in YG, τ in TG]
        d_XG2, d_YG, d_T3, d_UX, d_UY = CuArray(XG2), CuArray(YG), CuArray(T3), CuArray(UX), CuArray(UY)

        @testset "the trilinear lookup itself" begin
            function probe3!(out, x, y, t, v, qx, qy, qt, n)
                i = (blockIdx().x - 1)*blockDim().x + threadIdx().x
                i <= n && (@inbounds out[i] = KG.interpolate_3d_cuda(x, y, t, v, qx[i], qy[i], qt[i]))
                return
            end
            Random.seed!(91)
            nq = 20_000
            # spill outside the table on every axis, and land exactly on nodes
            qx = vcat(range(-60.0, 60.0; length = nq ÷ 2), 40.0 .* rand(nq ÷ 2) .- 20.0)
            qy = vcat(range(-60.0, 60.0; length = nq ÷ 2), 40.0 .* rand(nq ÷ 2) .- 20.0)
            qt = vcat(range(-3.0, 9.0;   length = nq ÷ 2), 0.4 .+ 2.0 .* rand(nq ÷ 2))
            qx[1:length(XG2)] .= XG2; qy[1:length(XG2)] .= YG[1]; qt[1:length(XG2)] .= TG[1]
            # 🔑 T is strictly positive, so a pointwise relative error is meaningful and `ulpeq`
            # applies. u^x and u^y are SIGNED and cross zero, where `|Δ|/max(|a|,|b|)` is
            # meaningless -- measured, it reports 31 ulps for a 7e-15 absolute difference on a
            # field of order 0.5. That is the same pathology this file's own `dev` docstring
            # records for momentum components, so the vector components are compared the same way:
            # deviation against the ARRAY'S OWN SCALE.
            for (nm, vals) in (("T", T3), ("ux", UX), ("uy", UY))
                dv  = CuArray(vals)
                cpu = [KC.interpolate_3d_cpu(XG2, YG, TG, vals, qx[i], qy[i], qt[i]) for i in 1:nq]
                dout = CUDA.zeros(Float64, nq)
                launch(probe3!, nq, dout, d_XG2, d_YG, d_TG, dv,
                       CuArray(qx), CuArray(qy), CuArray(qt), nq)
                got = Array(dout)
                if nm == "T"
                    @test ulpeq(got, cpu)
                else
                    @test maximum(abs.(got .- cpu)) <= 8e-16 * maximum(abs, cpu)
                end
                @test all(isfinite, got)
            end
            # NON-UNIFORM axes: the device binary search must not assume constant spacing
            xnu = [-20.0, -19.5, -3.0, 0.0, 0.2, 7.0, 20.0]
            ynu = [-20.0, -6.0, -0.1, 0.0, 4.0, 20.0]
            Tnu = [1.0 + 0.01x - 0.02y + 0.03τ for x in xnu, y in ynu, τ in TG]   # strictly positive: ulpeq applies
            cpu = [KC.interpolate_3d_cpu(xnu, ynu, TG, Tnu, qx[i], qy[i], qt[i]) for i in 1:nq]
            dout = CUDA.zeros(Float64, nq)
            launch(probe3!, nq, dout, CuArray(xnu), CuArray(ynu), d_TG, CuArray(Tnu),
                   CuArray(qx), CuArray(qy), CuArray(qt), nq)
            @test ulpeq(Array(dout), cpu)
        end

        @testset "boosts on a 2-D background, every switch combination" begin
            for pdim in (2, 3), rel in (true, false)
                mom, pos = state(; pdim = pdim, seed = 77)
                dmom, dpos = CuArray(mom), CuArray(pos)
                for (cpuk, gpuk) in ((KC.kernel_boost_to_rest_frame_cpu!, KG.kernel_boost_to_rest_frame_gpu!),
                                     (KC.kernel_boost_to_lab_frame_cpu!,  KG.kernel_boost_to_lab_frame_gpu!))
                    cpuk(mom, pos, XG2, TG, nothing, M, NP, 7, 2e-3, 0.4;
                         radial_mode = false, relativistic = rel,
                         ygrid = YG, VxField = UX, VyField = UY)
                    launch(gpuk, NP, dmom, dpos, d_XG2, d_TG, d_VF, M, NP, 7, 2e-3, 0.4,
                           false, false, d_V2, 0.0, rel, d_YG, d_UX, d_UY)
                    a, r = dev(mom, dmom)
                    @test a < 1e-13 && r < 1e-13
                end
            end
        end

        @testset "forces kernel on a 2-D background, SAME noise" begin
            # Mirrors P4 exactly, with the background swapped for the 2-D tables. Same
            # pre-generated noise on both sides, so this is a per-particle comparison and not a
            # statistical one.
            for rel in (true, false), ptk in (false, true)
                mom, pos = state(; pdim = 3, seed = 33)
                ξ  = randn(MersenneTwister(44), 3, NP)
                rd = randn(MersenneTwister(55), 3, NP)
                pm, pu = zeros(NP), zeros(3, NP)
                ηv, kLv, kTv = zeros(NP), zeros(NP), zeros(NP)
                det, sto = zeros(3, NP), zeros(3, NP)
                T0s, invdT, vals = TR.build_tau_drag_spline(M, DST; Tmin = 0.05, Tmax = 0.8, nT = 1024)
                KC.kernel_compute_all_forces_cpu!(T3, XG2, TG, mom, pos, pm, pu, ηv, kLv, kTv,
                    ξ, det, sto, 2e-3, M, rd, 3, NP, 7, 0.4, DST;
                    tau_Tmin = T0s, tau_invdT = invdT, tau_vals = vals,
                    radial_mode = false, relativistic = rel, proper_time_kicks = ptk,
                    Vfield = nothing, ygrid = YG, VxField = UX, VyField = UY)
                ddet = CUDA.zeros(Float64, 3, NP); dsto = CUDA.zeros(Float64, 3, NP)
                dpm = CUDA.zeros(Float64, NP); dpu = CUDA.zeros(Float64, 3, NP)
                dη = CUDA.zeros(Float64, NP); dkL = CUDA.zeros(Float64, NP); dkT = CUDA.zeros(Float64, NP)
                launch(KG.kernel_compute_all_forces_gpu!, NP, d_T3, d_XG2, d_TG,
                    CuArray(mom), CuArray(pos), dpm, dpu, dη, dkL, dkT, CuArray(ξ), ddet, dsto,
                    2e-3, M, CuArray(rd), 3, NP, 7, 0.4, DST, T0s, invdT, CuArray(vals),
                    false, Int32(0), rel, ptk, d_VF, d_YG, d_UX, d_UY)
                # P4 (the radial twin of this test) asserts the RELATIVE deviation at 1e-12 and
                # not the absolute one, because `dev`'s absolute value scales with the field: the
                # stochastic terms are the largest of these arrays and land at 1.7e-13 absolute
                # while sitting at 1.6e-14 relative. Same convention here, so the 2-D gate is no
                # weaker than the radial one it mirrors.
                for (h, d) in ((det, ddet), (sto, dsto), (ηv, dη), (kLv, dkL), (kTv, dkT))
                    _, r_ = dev(h, d)
                    @test r_ < 1e-12
                end
            end
        end
    end

    @testset "D1 KNOWN DIVERGENCES — recorded, not reconciled" begin
        # ── D1a. RTA at T → 0. ──────────────────────────────────────────────────────────────────
        # CPU `_draw_juttner_pstar_lib` returns 0.0 for T <= 0, so a collision ZEROES the momentum.
        # GPU `_juttner_invcdf_lookup` clamps the query to the table's Tmin edge and returns a
        # FINITE THERMAL |p*|. Same input, qualitatively different output: at the fireball rim the
        # CPU brings particles to rest and the GPU thermalises them at the table's coldest bin.
        # Not reconciled here because either choice moves numbers (see CHANGELOG 0.2.1). The
        # background T table is required to be strictly positive by `_build_time_spline`, so this
        # is reachable only where the interpolated T underflows to 0 — but it IS reachable there.
        m0 = ones(2, 8); p0 = fill(50.0, 2, 8)          # far outside a table that is 0 everywhere
        xg0 = collect(0.0:0.5:5.0); tg0 = collect(0.0:0.5:1.0)
        Tz = zeros(length(xg0), length(tg0))
        Random.seed!(1)
        KC.kernel_rta_collision_cpu!(Tz, xg0, tg0, m0, p0, 1e-3, M, 8, 1, 0.0, 0.2;
            tau_Tmin = 0.1, tau_invdT = 1.0, tau_vals = [1.0, 1.0], dimensions = 2)
        @test all(iszero, m0)                            # CPU: momentum zeroed at T = 0
        icdf, nU, nT, itmin, iinv = TR.build_juttner_invcdf(M, 2; Tmin = 0.12, Tmax = 0.50)
        # the GPU table has no zero-T entry to return: its coldest column is a real thermal draw
        @test maximum(icdf[1:nU]) > 0.1                  # GPU: a finite |p*| at the table edge
        @info "D1a CPU zeroes p at T=0; GPU returns a thermal |p*| at the table's Tmin edge (max |p*| = $(round(maximum(icdf[1:nU]); digits = 3)) GeV)"

        # ── D1b. Δt <= 0 in the forces kernel. ─────────────────────────────────────────────────
        # CPU branches on `ηD > 0` alone and sets noise_pref = 0; GPU branches on
        # `ηD > 0 && Δt > 0` and falls through to the `else` arm, which writes sto = kT·ξ. Not
        # physical (Δt > 0 always in the driver) but the two are not the same function.
        pdim = 2; mom, pos = state(; seed = 81)
        ξ = randn(MersenneTwister(82), pdim, NP); rdv = randn(MersenneTwister(83), pdim, NP)
        rdv ./= sqrt.(sum(abs2, rdv; dims = 1))
        T0, invdT, vals = build_tau_drag_spline(M, DST; Tmin = minimum(TF), Tmax = maximum(TF), nT = 1024)
        det = zeros(pdim, NP); sto = zeros(pdim, NP)
        KC.kernel_compute_all_forces_cpu!(TF, XG, TG, mom, pos, zeros(NP), zeros(pdim, NP),
            zeros(NP), zeros(NP), zeros(NP), ξ, det, sto, 0.0, M, rdv, pdim, NP, 7, 0.4, DST;
            tau_Tmin = T0, tau_invdT = invdT, tau_vals = vals)
        ddet = CUDA.zeros(Float64, pdim, NP); dsto = CUDA.zeros(Float64, pdim, NP)
        launch(KG.kernel_compute_all_forces_gpu!, NP, CuArray(TF), d_XG, d_TG, CuArray(mom), CuArray(pos),
            CUDA.zeros(Float64, NP), CUDA.zeros(Float64, pdim, NP), CUDA.zeros(Float64, NP),
            CUDA.zeros(Float64, NP), CUDA.zeros(Float64, NP), CuArray(ξ), ddet, dsto,
            0.0, M, CuArray(rdv), pdim, NP, 7, 0.4, DST, T0, invdT, CuArray(vals),
            false, Int32(0), true, false, d_VF)
        @test all(iszero, det) && all(iszero, sto)              # CPU: both terms zero at Δt = 0
        @test maximum(abs, Array(dsto)) > 0                     # GPU: writes kT·ξ instead
        @info "D1b at Δt = 0 the CPU writes zero noise, the GPU writes kT·ξ (max |sto| = $(round(maximum(abs, Array(dsto)); digits = 4)))"
    end
end
