# =================================================================================================
# test/test_boost2d.jl — the boost kernels on a 2-D (vector) background.
#
# Step 2. `interpolate_3d_cpu` (test_background3d.jl) supplies T, u^x, u^y at the particle's own
# (x, y, tau); this gives the boost kernels a flow DIRECTION of their own instead of r-hat.
#
# Four things have to hold, and the first is the one that makes the rest believable.
#
#   1. REDUCTION. Handed a purely RADIAL vector field u = v(r,t) r-hat, the 2-D path must reproduce
#      the shipped radial path. This is the only check that ties the new path to eight months of
#      validated results.
#      ⚠ NOT to rounding, and I first asserted that it would be. The two paths interpolate
#      DIFFERENT TABLES -- v(r,t) bilinearly on a radial grid, (u^x,u^y)(x,y,t) trilinearly on a
#      Cartesian one -- and hypot of two interpolated components is not the interpolant of the
#      magnitude. The gap is interpolation error, ~1e-3 at the grids used here, and demanding
#      rounding would be demanding something false. What distinguishes interpolation error from a
#      logic bug is that it CONVERGES: the test refines both grids together and requires the
#      discrepancy to fall. A wrong direction or a wrong sign would sit at a constant.
#
#   2. ROUND TRIP. rest-frame then lab-frame must be the identity, on a genuinely non-radial flow.
#      That is what says the two kernels were generalised CONSISTENTLY; a sign or direction error
#      in one alone would survive test 1 (which uses a radial field) and fail here.
#      ⚠ Exact only to ~1e-10, and that is PRE-EXISTING: the shipped kernels use
#      gamma = 1/sqrt(1 - v^2 + 1e-10), and that regulator makes the forward and inverse boosts
#      not quite reciprocal (gamma^2(1-v^2) = 1 - 1e-10/(1-v^2)). The radial path is measured
#      alongside as the control, and it carries the same error.
#
#   3. IT ACTUALLY DOES SOMETHING NEW. On a rotational flow the 2-D path must DIFFER from the
#      radial path by a large margin. Without this, 1 and 2 are equally satisfied by a kernel that
#      silently ignores the new arguments.
#
#   4. GUARDS. `radial_mode` (a p_r-only momentum row) cannot represent a boost along a non-radial
#      flow and must refuse rather than quietly project.
#
# Run: julia --project=Julia Julia/LangevInMedium.jl/test/test_boost2d.jl
# =================================================================================================

using Test, Printf, Random

# `using` the package, not the source, so every gate in this directory runs in one
# environment (the same one the corpus and the parity gate use).
using LangevInMedium
using LangevInMedium.KernelsCPU: kernel_boost_to_rest_frame_cpu!, kernel_boost_to_lab_frame_cpu!

const M  = 1.5
const T0 = 0.0
const DT = 0.01

"""Grids and a radial flow field, tabulated both as v(r,t) and as the vector (u^x, u^y)(x,y,t)."""
function radial_pair(; vmax = 0.55, nr = 41, nxy = 65)
    rs = collect(range(0.0, 8.0; length = nr))
    ts = collect(range(0.0, 2.0; length = 11))
    vfun(r, t) = vmax * tanh(r/3.0) * (0.6 + 0.2t)
    V2 = [vfun(r, t) for r in rs, t in ts]

    xs = collect(range(-8.0, 8.0; length = nxy))
    ys = collect(range(-8.0, 8.0; length = nxy))
    VX = [(hypot(x,y) < 1e-12 ? 0.0 : vfun(hypot(x,y), t) * x/hypot(x,y)) for x in xs, y in ys, t in ts]
    VY = [(hypot(x,y) < 1e-12 ? 0.0 : vfun(hypot(x,y), t) * y/hypot(x,y)) for x in xs, y in ys, t in ts]
    return (rs, ts, V2, xs, ys, VX, VY)
end

function sample_particles(n; seed = 20260903)
    rng = MersenneTwister(seed)
    pos = zeros(2, n); mom = zeros(3, n)
    for i in 1:n
        # keep well inside the tabulated disc so nothing is clamped
        r = 0.3 + 5.0*rand(rng); φ = 2π*rand(rng)
        pos[1,i] = r*cos(φ); pos[2,i] = r*sin(φ)
        for d in 1:3; mom[d,i] = 2.0*(2rand(rng)-1); end
    end
    return pos, mom
end

@testset "boost kernels on a 2-D background" begin

    rs, ts, V2, xs, ys, VX, VY = radial_pair()

    @testset "1. reduces to the shipped radial path on a radial field" begin
        # Refine BOTH tables together. Interpolation error falls; a logic error would not.
        prev = NaN; errs = Float64[]
        for nref in (1, 2, 4)
            rs2, ts2, V22, xs2, ys2, VX2, VY2 = radial_pair(; nr = 40*nref + 1, nxy = 64*nref + 1)
            pos, mom = sample_particles(3000)
            a = copy(mom); b = copy(mom)
            kernel_boost_to_rest_frame_cpu!(a, pos, rs2, ts2, V22, M, size(pos,2), 3, DT, T0)
            kernel_boost_to_rest_frame_cpu!(b, pos, xs2, ts2, nothing, M, size(pos,2), 3, DT, T0;
                                            ygrid = ys2, VxField = VX2, VyField = VY2)
            e = maximum(abs, a .- b) / maximum(abs, a)
            push!(errs, e)
            @printf("  refine x%d : max |2-D - radial| / scale = %.3e\n", nref, e)
        end
        @test errs[2] < errs[1]
        @test errs[3] < errs[2]
        @printf("  ⇒ converges (%.2fx then %.2fx per halving) => interpolation, not logic\n",
                errs[1]/errs[2], errs[2]/errs[3])

        # and the lab-frame kernel reduces the same way
        rs2, ts2, V22, xs2, ys2, VX2, VY2 = radial_pair(; nr = 161, nxy = 257)
        pos, mom = sample_particles(3000)
        c = copy(mom); d = copy(mom)
        kernel_boost_to_lab_frame_cpu!(c, pos, rs2, ts2, V22, M, size(pos,2), 3, DT, T0)
        kernel_boost_to_lab_frame_cpu!(d, pos, xs2, ts2, nothing, M, size(pos,2), 3, DT, T0;
                                       ygrid = ys2, VxField = VX2, VyField = VY2)
        e2 = maximum(abs, c .- d)/maximum(abs, c)
        @printf("  lab frame at the finest grid: %.3e\n", e2)
        @test e2 < 2e-4
    end

    @testset "2. rest -> lab is the identity on a NON-radial flow" begin
        # A rotational + radial flow: u is nowhere parallel to r-hat, so this exercises the
        # generalisation rather than the radial special case.
        tsr = ts
        UX = [ 0.30*tanh(hypot(x,y)/3) * (x/max(hypot(x,y),1e-9)) - 0.25*(y/8.0) for x in xs, y in ys, t in tsr]
        UY = [ 0.30*tanh(hypot(x,y)/3) * (y/max(hypot(x,y),1e-9)) + 0.25*(x/8.0) for x in xs, y in ys, t in tsr]
        pos, mom = sample_particles(4000; seed = 11)
        p = copy(mom)
        kernel_boost_to_rest_frame_cpu!(p, pos, xs, tsr, nothing, M, size(pos,2), 3, DT, T0;
                                        ygrid = ys, VxField = UX, VyField = UY)
        kernel_boost_to_lab_frame_cpu!(p, pos, xs, tsr, nothing, M, size(pos,2), 3, DT, T0;
                                       ygrid = ys, VxField = UX, VyField = UY)
        worst = maximum(abs, p .- mom); scale = maximum(abs, mom)
        # CONTROL: the shipped radial path round-trips no better -- the gamma regulator, not the
        # generalisation, is what limits this.
        pr = copy(mom)
        kernel_boost_to_rest_frame_cpu!(pr, pos, rs, ts, V2, M, size(pos,2), 3, DT, T0)
        kernel_boost_to_lab_frame_cpu!(pr, pos, rs, ts, V2, M, size(pos,2), 3, DT, T0)
        wr = maximum(abs, pr .- mom)
        @printf("  round trip : 2-D %.3e   radial (control) %.3e   (scale %.2f)\n", worst, wr, scale)
        @test worst < 1e-8*scale
        @test worst < 20*max(wr, 1e-16)      # no worse than the shipped path, to a factor

        @testset "3. and it is NOT the radial path" begin
            # If the kernels quietly ignored VxField/VyField, 1 and 2 would both still pass.
            q = copy(mom)
            kernel_boost_to_rest_frame_cpu!(q, pos, xs, tsr, nothing, M, size(pos,2), 3, DT, T0;
                                            ygrid = ys, VxField = UX, VyField = UY)
            rr = copy(mom)
            kernel_boost_to_rest_frame_cpu!(rr, pos, rs, tsr, V2, M, size(pos,2), 3, DT, T0)
            diff = maximum(abs, q .- rr)
            @printf("  rotational flow vs radial path: max difference = %.3e (scale %.2f)\n", diff, scale)
            @test diff > 0.01*scale
        end
    end

    @testset "4. radial_mode refuses a 2-D background" begin
        pos1 = zeros(1, 4); pos1[1,:] .= [1.0, 2.0, 3.0, 4.0]
        mom1 = ones(1, 4)
        @test_throws ErrorException kernel_boost_to_rest_frame_cpu!(
            mom1, pos1, xs, ts, nothing, M, 4, 3, DT, T0;
            radial_mode = true, ygrid = ys, VxField = VX, VyField = VY)
        println("  radial_mode + 2-D background refuses, as it must")
    end
end
