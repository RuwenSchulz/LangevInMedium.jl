# =================================================================================================
# test/test_simulate2d.jl — END TO END on a 2-D background.
#
# Step 3. Steps 1 and 2 pinned the lookup (test_background3d.jl) and the boost kernels
# (test_boost2d.jl). This runs the actual driver: `simulate_ensemble_bulk` handed a 3-element
# SpaceTimeGrid, a T[x,y,t] table and the flow as a VECTOR pair (u^x, u^y).
#
# THE TEST THAT MATTERS is the first one. Hand the 2-D path a background that is RADIAL and it must
# reproduce the shipped radial run. Everything published by this package runs through that path, so
# an end-to-end agreement is what says the generalisation did not quietly change the dynamics --
# not the boost alone, but the drag, the noise, the streaming and the p_z completion together.
#
# ⚠ It is a STOCHASTIC comparison. Both runs draw the same random numbers in the same order (same
# seed, same kernels, same call sequence), so the agreement is limited only by the difference
# between the two interpolants -- v(r,t) bilinear versus (u^x,u^y)(x,y,t) trilinear, whose hypot is
# not the interpolant of the magnitude. That is ~1e-3 at usable grids, exactly as test_boost2d
# measured, and it CONVERGES under refinement. So the assertion is on ensemble moments at a
# tolerance set by that, plus a convergence check -- never bit-identity, which is not available.
#
# Run: julia --project=Julia/LangevInMedium.jl Julia/LangevInMedium.jl/test/test_simulate2d.jl
# =================================================================================================

using Test, Printf, Random

const _HERE = @__DIR__
include(joinpath(_HERE, "..", "src", "LangevInMedium.jl"))
using .LangevInMedium

const M    = 1.5
const DsT  = 0.2
const TAU0 = 0.6
const TAUF = 3.0

"""A radial fireball, tabulated as v(r,t) AND as the vector (u^x,u^y)(x,y,t)."""
function backgrounds(; nr = 81, nxy = 161, nt = 21, rmax = 10.0)
    rs = collect(range(0.0, rmax; length = nr))
    ts = collect(range(TAU0, TAUF; length = nt))
    Tf(r, t) = 0.45 * (TAU0/t)^(1/3) * exp(-(r/6.0)^2) + 0.06
    vf(r, t) = 0.55 * tanh(r/4.0) * (t/TAUF)^0.4

    T2 = [Tf(r, t) for r in rs, t in ts]
    V2 = [vf(r, t) for r in rs, t in ts]

    xs = collect(range(-rmax, rmax; length = nxy))
    ys = collect(range(-rmax, rmax; length = nxy))
    T3 = [Tf(hypot(x,y), t) for x in xs, y in ys, t in ts]
    UX = [(r = hypot(x,y); r < 1e-12 ? 0.0 : vf(r,t)*x/r) for x in xs, y in ys, t in ts]
    UY = [(r = hypot(x,y); r < 1e-12 ? 0.0 : vf(r,t)*y/r) for x in xs, y in ys, t in ts]
    return (rs, ts, T2, V2, xs, ys, T3, UX, UY)
end

"""Identical initial particles for both runs, so only the background differs."""
function ic(n; seed = 4242)
    rng = MersenneTwister(seed)
    x = zeros(2, n); p = zeros(2, n)
    for i in 1:n
        r = 4.0*sqrt(rand(rng)); φ = 2π*rand(rng)
        x[1,i] = r*cos(φ); x[2,i] = r*sin(φ)
        p[1,i] = 1.2*randn(rng); p[2,i] = 1.2*randn(rng)
    end
    return x, p
end

function run_case(grid, Tfield, Vfield, x0, p0; kw...)
    Random.seed!(20260903)
    return simulate_ensemble_bulk(CPUBackend(), nothing, nothing, nothing,
        Tfield, Vfield, grid;
        N_particles = size(x0, 2), Δt = 0.004,
        initial_time = TAU0, final_time = TAUF, save_interval = TAUF - TAU0,
        m = M, DsT = DsT, dimensions = 2, momentum_dimensions = 3,
        x_init = x0, p_init = p0, kw...)
end

# the driver returns (time_points, momenta_snapshots, position_snapshots), each snapshot list a
# Vector of matrices -- NOT a 3-D array, which is what I first assumed here.
moments(res) = begin
    mom = res[2][end]; pos = res[3][end]
    (p2 = sum(abs2, mom)/size(mom,2), x2 = sum(abs2, pos)/size(pos,2),
     px = sum(mom[1,:])/size(mom,2))
end

@testset "simulate_ensemble_bulk on a 2-D background" begin

    @testset "1. a RADIAL 2-D background reproduces the shipped radial run" begin
        errs = Float64[]
        for nref in (1, 2)
            rs, ts, T2, V2, xs, ys, T3, UX, UY = backgrounds(; nr = 80*nref+1, nxy = 160*nref+1)
            x0, p0 = ic(4000)
            a = run_case((rs, ts), T2, V2, x0, p0)
            b = run_case((xs, ys, ts), T3, (UX, UY), x0, p0)
            ma, mb = moments(a), moments(b)
            e = max(abs(ma.p2-mb.p2)/ma.p2, abs(ma.x2-mb.x2)/ma.x2)
            push!(errs, e)
            @printf("  refine x%d : <p^2> %.6f vs %.6f   <x^2> %.4f vs %.4f   rel %.3e\n",
                    nref, ma.p2, mb.p2, ma.x2, mb.x2, e)
        end
        @test errs[1] < 5e-3
        @test errs[2] < errs[1]
        @printf("  ⇒ agrees to %.1e and IMPROVES on refinement (%.2fx) => interpolation, not dynamics\n",
                errs[2], errs[1]/errs[2])
    end

    @testset "2. an elliptic flow is NOT the radial answer" begin
        # If the driver silently ignored the vector field, test 1 would still pass.
        rs, ts, T2, V2, xs, ys, T3, UX, UY = backgrounds()
        EX = similar(UX); EY = similar(UY)
        for (i, x) in enumerate(xs), (j, y) in enumerate(ys), (k, t) in enumerate(ts)
            r = hypot(x, y)
            s = r < 1e-12 ? 0.0 : 1.0
            # 30% stronger push along x than along y: a genuine m = 2 flow field
            EX[i,j,k] = s * UX[i,j,k] * 1.30
            EY[i,j,k] = s * UY[i,j,k] * 0.70
        end
        x0, p0 = ic(4000)
        rad = moments(run_case((xs, ys, ts), T3, (UX, UY), x0, p0))
        ell = moments(run_case((xs, ys, ts), T3, (EX, EY), x0, p0))
        @printf("  radial <p^2> %.5f   elliptic <p^2> %.5f   (%.1f%% apart)\n",
                rad.p2, ell.p2, 100*abs(rad.p2-ell.p2)/rad.p2)
        @test abs(rad.p2 - ell.p2)/rad.p2 > 0.01
    end

    @testset "3. the input contract is enforced" begin
        rs, ts, T2, V2, xs, ys, T3, UX, UY = backgrounds()
        x0, p0 = ic(200)
        # 3-element grid but a 2-D temperature table
        @test_throws ErrorException run_case((xs, ys, ts), T2, (UX, UY), x0, p0)
        # 3-element grid but a scalar flow table
        @test_throws ErrorException run_case((xs, ys, ts), T3, V2, x0, p0)
        # grids that do not match the table
        @test_throws ErrorException run_case((xs, ys[1:end-1], ts), T3, (UX, UY), x0, p0)
        # V2Evolution double-counts the anisotropy a 2-D background already carries
        @test_throws ErrorException run_case((xs, ys, ts), T3, (UX, UY), x0, p0;
                                             V2Evolutionn = V2, psi2 = 0.0)
        println("  shape, vector-flow, grid-match and V2 double-count all refused")
    end
end
