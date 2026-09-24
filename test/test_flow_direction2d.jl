#=
   test_flow_direction2d.jl — the flow DIRECTION on a 2-D (x, y, τ) background (2026-09-24).

   THE DEFECTS THIS PINS. Two kernels took the flow to point along r̂. On a radial (r, τ) table that
   is exact, so no gate built on one could see it; on a 2-D background the flow points along its own
   direction û = (v^x, v^y)/|v|, which departs from r̂ by a median 22° (p90 105°) on the Pb+Pb 20–30 %
   event and 12° on the O+O average, in cells above T_fo weighted by the charm density.

     1. `proper_time_kicks` dilated the kick by 1/(γ(1 + |v|·k·r̂/E*)) instead of 1/(γ(1 + v⃗·k⃗/E*)),
        at all four sites (CPU and GPU, OU and RTA). Measured before the fix on an azimuthal flow,
        |v| = 0.5, T = 0.3 GeV: ⟨p·û⟩ 1.1763 against the boosted Jüttner's 1.3529 (the correction
        lost entirely: the kicks-off value is 1.1750), and a spurious outward ⟨p·r̂⟩ = +0.149 GeV.
     2. The GPU cold-comoving kernel (literal DsT == 0) wrote m·γ·v along (x/r, y/r); the CPU wrote it
        along û. Measured before the fix on a uniform x-flow: GPU ⟨p_x⟩ = −0.0009 against 0.86603.

   Both fixes touch only the 2-D-background branch: the radial path is textually unchanged and its
   outputs are bit-identical (regression corpus and a 21-hash CPU+GPU harness over the production
   switches, 2026-09-24).

   Gates:  F1  cold comoving on a uniform x-flow: every particle carries p = m·γ·v·x̂ (the engine's
               regularised γ), on the CPU and on the GPU, per particle to 1e-12
           F2  proper_time_kicks, OU: an azimuthal flow reaches the boosted-Jüttner ⟨p·û⟩ as a radial
               flow of the same |v| does, and its ⟨p·r̂⟩ vanishes (CPU; GPU when available)
           F3  the same for collision_mode = :rta

   Run: julia --project=Julia Julia/LangevInMedium.jl/test/test_flow_direction2d.jl
=#
import Pkg; Pkg.activate(normpath(joinpath(@__DIR__, "..", "..")); io = devnull)
using Test, Printf, Random, Statistics

# CUDA is optional: the GPU halves below are skipped without a device, and say so.
const HAVE_CUDA_FD = try
    @eval using CUDA
    Base.invokelatest(() -> CUDA.functional())
catch
    false
end
using LangevInMedium

const M_FD  = 1.5
const T_FD  = 0.3
const V_FD  = 0.5
const R0_FD = 12.0            # start far from the axis: the flow's curvature/divergence ~ v/r is small

"Uniform T; the flow is `:radial` (v r̂), `:azimuthal` (v φ̂) or `:uniform_x` (v x̂), |v| = V_FD."
function box_fd(flow::Symbol; L = 24.0, nx = 161)
    xs = collect(range(-L, L; length = nx)); ys = copy(xs); ts = collect(range(1.0, 12.0; length = 3))
    Tg = fill(T_FD, nx, nx, length(ts))
    vx = zeros(nx, nx, length(ts)); vy = zeros(nx, nx, length(ts))
    for k in eachindex(ts), j in 1:nx, i in 1:nx
        x, y = xs[i], ys[j]; r = hypot(x, y)
        s = V_FD * min(1.0, r / 2)                    # regular at the origin
        if flow === :uniform_x
            vx[i, j, k] = V_FD
        elseif r > 1e-12
            if flow === :radial
                vx[i, j, k] = s * x / r; vy[i, j, k] = s * y / r
            else
                vx[i, j, k] = -s * y / r; vy[i, j, k] = s * x / r
            end
        end
    end
    xs, ys, ts, Tg, vx, vy
end

"The boosted Jüttner ⟨p·û⟩ counted at fixed lab time (weight E/E*), by rejection sampling in 3-D."
function boosted_juttner_along(; N = 2_000_000, rng = Xoshiro(3))
    γ = 1 / sqrt(1 - V_FD^2); sw = 0.0; swp = 0.0; n = 0
    while n < N
        k = -T_FD * log(rand(rng) * rand(rng) * rand(rng))     # proposal Gamma(3, T) ∝ k² e^{−k/T}
        E = sqrt(k^2 + M_FD^2)
        rand(rng) < exp(-(E - k) / T_FD) || continue           # target k² e^{−E/T}
        kx = k * (2rand(rng) - 1)
        w = (E + V_FD * kx) / E
        sw += w; swp += w * γ * (kx + V_FD * E); n += 1
    end
    swp / sw
end

"Run to stationarity (≈ 9 drag times) and return ⟨p·û⟩, ⟨p·r̂⟩ (azimuthal only) and their SEMs."
function stationary_fd(backend, flow; mode = :langevin, N = 40_000, seed = 77)
    xs, ys, ts, Tg, vx, vy = box_fd(flow)
    Random.seed!(seed); HAVE_CUDA_FD && backend isa GPUBackend && CUDA.seed!(seed)
    φ = 2π .* rand(N)
    pos = vcat((R0_FD .* cos.(φ))', (R0_FD .* sin.(φ))'); mom = 0.3 .* randn(3, N)
    _, momh, posh = simulate_ensemble_bulk(backend, nothing, nothing, nothing, Tg, (vx, vy), (xs, ys, ts);
        N_particles = N, Δt = 0.005, initial_time = 1.0, final_time = 9.0, save_interval = 8.0,
        m = M_FD, DsT = 0.01, dimensions = 2, momentum_dimensions = 3, collision_mode = mode,
        x_init = pos, p_init = mom, proper_time_kicks = true)
    P = momh[end]; X = posh[end]
    along = zeros(N); outward = zeros(N)
    for a in 1:N
        x, y = X[1, a], X[2, a]; r = hypot(x, y)
        ux, uy = flow === :radial ? (x / r, y / r) : (-y / r, x / r)
        along[a] = P[1, a] * ux + P[2, a] * uy
        outward[a] = P[1, a] * x / r + P[2, a] * y / r
    end
    (along = mean(along), se_along = std(along) / sqrt(N),
     outward = mean(outward), se_out = std(outward) / sqrt(N))
end

@testset "flow direction on a 2-D background" begin
    W = boosted_juttner_along()

    @testset "F1 cold comoving imprints m·γ·v along the FLOW, not along r̂" begin
        xs, ys, ts, Tg, vx, vy = box_fd(:uniform_x)
        pxe = M_FD * V_FD / sqrt(1 - V_FD^2 + 1e-10)           # the kernels' regularised γ
        backends = HAVE_CUDA_FD ? (CPUBackend(), GPUBackend()) : (CPUBackend(),)
        HAVE_CUDA_FD || @info "F1: no CUDA device — GPU half skipped"
        for bk in backends
            Random.seed!(5); N = 5_000
            φ = 2π .* rand(N); pos = vcat((5 .* cos.(φ))', (5 .* sin.(φ))'); mom = 0.3 .* randn(3, N)
            _, momh, _ = simulate_ensemble_bulk(bk, nothing, nothing, nothing, Tg, (vx, vy), (xs, ys, ts);
                N_particles = N, Δt = 0.01, initial_time = 1.0, final_time = 1.05, save_interval = 0.05,
                m = M_FD, DsT = 0.0, dimensions = 2, momentum_dimensions = 3, x_init = pos, p_init = mom)
            P = momh[end]
            @printf("  F1 %-10s max|p_x − mγv| = %.2e   max|p_y| = %.2e   max|p_z| = %.2e\n",
                    nameof(typeof(bk)), maximum(abs.(P[1, :] .- pxe)), maximum(abs.(P[2, :])), maximum(abs.(P[3, :])))
            @test maximum(abs.(P[1, :] .- pxe)) < 1e-12
            @test maximum(abs.(P[2, :])) < 1e-12
            @test maximum(abs.(P[3, :])) == 0.0
        end
    end

    for (gate, mode) in (("F2", :langevin), ("F3", :rta))
        @testset "$gate proper_time_kicks ($mode): azimuthal flow = radial flow = boosted Jüttner" begin
            backends = HAVE_CUDA_FD ? ((CPUBackend(), 40_000), (GPUBackend(), 200_000)) : ((CPUBackend(), 40_000),)
            HAVE_CUDA_FD || @info "$gate: no CUDA device — GPU half skipped"
            for (bk, N) in backends
                rad = stationary_fd(bk, :radial; mode = mode, N = N)
                azi = stationary_fd(bk, :azimuthal; mode = mode, N = N)
                @printf("  %s %-10s ⟨p·û⟩ radial %.4f ± %.4f (%+.2f %%)  azimuthal %.4f ± %.4f (%+.2f %%)  azimuthal ⟨p·r̂⟩ %+.4f ± %.4f   [W = %.4f]\n",
                        gate, nameof(typeof(bk)), rad.along, rad.se_along, 100 * (rad.along / W - 1),
                        azi.along, azi.se_along, 100 * (azi.along / W - 1), azi.outward, azi.se_out, W)
                # The defect put the azimuthal case at −13 % with ⟨p·r̂⟩ = +0.149; the gradient
                # corrections of these two geometries are O(1 %) and differ, so 3 % separates the
                # two cleanly without asserting a precision the setup does not have.
                @test abs(azi.along / W - 1) < 0.03
                @test abs(rad.along / W - 1) < 0.03
                @test abs(azi.outward) < 0.03
            end
        end
    end
end
