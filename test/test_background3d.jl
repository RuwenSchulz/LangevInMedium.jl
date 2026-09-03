# =================================================================================================
# test/test_background3d.jl — the (x, y, tau) background lookup.
#
# Step 1 of giving the transport a background that is a genuine function of the TRANSVERSE PLANE
# rather than of the radius. Everything published here runs on `T(r, tau)`, `v_r(r, tau)` with an
# optional ad-hoc m = 2 modulation of |v| along r-hat, which cannot represent flow that is not
# radial and cannot represent an odd harmonic at all.
#
# This file pins the interpolator BEFORE any kernel uses it, because every later claim about a 2-D
# background rests on it:
#
#   1. EXACTNESS. Trilinear interpolation is exact for a trilinear function. Anything else is a
#      bug in the cell location or the weights, and a smooth-field test would hide it behind the
#      interpolation error.
#   2. REDUCTION. On a field with NO y dependence the 3-D lookup must return `interpolate_2d_cpu`
#      to within a rounding. This is the check that matters: it is what lets a 2-D background be
#      validated against the shipped radial path later, and it holds only because the two routines
#      share their clamping and cell-location conventions verbatim.
#      ⚠ NOT bit-identical, and I first asserted that it would be. `c*(1-yd) + c*yd` re-rounds even
#      when the two y-nodes carry the same value, so a few ulp is the strongest claim the
#      arithmetic supports. Asserting bit-equality here would be asserting something false.
#   3. CLAMPING. Outside the table the value is the edge value, never an extrapolation --
#      `interpolate_2d_cpu`'s docstring records that extrapolating produced T < 0 and |v| > 1 once
#      particles left the grid.
#   4. DEGENERATE AXES. A length-1 axis must contribute its node value, not a division by zero.
#
# Run: julia --project=Julia/LangevInMedium.jl Julia/LangevInMedium.jl/test/test_background3d.jl
# =================================================================================================

using Test, Printf, Random

const _HERE = @__DIR__
include(joinpath(_HERE, "..", "src", "LangevInMedium.jl"))
# KernelsCPU is included by the package but not `using`-ed into its namespace, so the
# interpolators are reached through the submodule directly.
using .LangevInMedium.KernelsCPU: interpolate_2d_cpu, interpolate_3d_cpu

@testset "3-D background lookup" begin

    # --- non-uniform axes on purpose: a uniform grid hides an index-vs-position bug ---
    xs = [0.0, 0.7, 1.9, 3.0, 5.5]
    ys = [-2.0, -0.5, 0.25, 1.0, 4.0]
    ts = [0.4, 0.9, 2.1, 3.0]

    @testset "exact on a trilinear function" begin
        # f is trilinear, so interpolation must be EXACT, not merely close.
        f(x, y, t) = 1.3 + 0.7x - 0.4y + 0.9t + 0.11x*y - 0.23x*t + 0.31y*t + 0.017x*y*t
        V = [f(x, y, t) for x in xs, y in ys, t in ts]
        rng = MersenneTwister(20260903); worst = 0.0
        for _ in 1:20_000
            xi = xs[1] + rand(rng)*(xs[end]-xs[1])
            yi = ys[1] + rand(rng)*(ys[end]-ys[1])
            ti = ts[1] + rand(rng)*(ts[end]-ts[1])
            worst = max(worst, abs(interpolate_3d_cpu(xs, ys, ts, V, xi, yi, ti) - f(xi, yi, ti)))
        end
        @printf("  max |interp - exact| on a trilinear field: %.3e\n", worst)
        @test worst < 1e-12
    end

    @testset "reduces to interpolate_2d_cpu when the field has no y dependence" begin
        # THE decisive one: a 2-D background embedded in the 3-D table must be read
        # identically, bit for bit, or nothing downstream can be compared to the
        # shipped radial path.
        g(x, t) = 0.42 + 0.9x - 0.6t + 0.27x*t
        V2 = [g(x, t) for x in xs, t in ts]
        V3 = [g(x, t) for x in xs, _ in ys, t in ts]
        scale = maximum(abs, V2)
        rng = MersenneTwister(7); worst = 0.0; nbit = 0; ntot = 0; worstulp = 0.0
        for _ in 1:20_000
            xi = xs[1] + rand(rng)*(xs[end]-xs[1])
            ti = ts[1] + rand(rng)*(ts[end]-ts[1])
            yi = ys[1] + rand(rng)*(ys[end]-ys[1])
            a = interpolate_3d_cpu(xs, ys, ts, V3, xi, yi, ti)
            b = interpolate_2d_cpu(xs, ts, V2, xi, ti)
            worst = max(worst, abs(a - b)); ntot += 1; a === b && (nbit += 1)
            # 🪤 NOT eps(|b|): this field crosses zero, and normalising by the eps of a
            # near-zero local value reports 1024 "ulp" for a 1.8e-15 absolute difference.
            # The meaningful scale is the FIELD's, not the sample's.
            worstulp = max(worstulp, abs(a - b)/eps(scale))
        end
        @printf("  max |3-D - 2-D| on a y-independent field: %.3e = %.1f ulp of the field scale;  bit-identical on %d/%d\n",
                worst, worstulp, nbit, ntot)
        @test worstulp <= 8.0
    end

    @testset "clamps outside the table, never extrapolates" begin
        f(x, y, t) = 2.0 + x + 2y + 3t
        V = [f(x, y, t) for x in xs, y in ys, t in ts]
        # far outside on every axis and in both directions
        for (xi, yi, ti, xe, ye, te) in (
                (-99.0, -99.0, -99.0, xs[1],   ys[1],   ts[1]),
                (+99.0, +99.0, +99.0, xs[end], ys[end], ts[end]),
                (-99.0, +99.0, 1.5,   xs[1],   ys[end], 1.5),
                (2.0,   -99.0, +99.0, 2.0,     ys[1],   ts[end]))
            got = interpolate_3d_cpu(xs, ys, ts, V, xi, yi, ti)
            @test got ≈ f(xe, ye, te) atol=1e-12
        end
        # and the clamped value is always inside the tabulated range
        lo, hi = minimum(V), maximum(V)
        for _ in 1:2000
            v = interpolate_3d_cpu(xs, ys, ts, V, 50*(2rand()-1), 50*(2rand()-1), 50*(2rand()-1))
            @test lo - 1e-12 <= v <= hi + 1e-12
        end
        println("  clamped values stay within the tabulated range")
    end

    @testset "degenerate axes contribute their node value" begin
        # A single-slice axis is how a t-independent or y-independent background
        # would naturally be handed in; it must not divide by zero.
        V = reshape([1.0, 2.0, 3.0, 4.0], 2, 2, 1)
        for ti in (-5.0, 0.0, 5.0)
            @test interpolate_3d_cpu([0.0,1.0], [0.0,1.0], [7.0], V, 0.5, 0.5, ti) ≈ 2.5
        end
        V2 = reshape([1.0, 3.0], 1, 2, 1)
        @test interpolate_3d_cpu([2.0], [0.0,1.0], [0.0], V2, 99.0, 0.5, 0.0) ≈ 2.0
        println("  length-1 axes handled without a division by zero")
    end
end
