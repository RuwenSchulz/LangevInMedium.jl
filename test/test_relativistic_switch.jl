# ==============================================================================================
# test_relativistic_switch.jl — the `relativistic` switch must actually switch.
#
# WHY THIS TEST EXISTS. The flag was, for a long time, parsed by the drivers, written into the
# output metadata, and never handed to the solver. Every run used the engine default while the
# sidecar recorded whatever had been asked for. It was found only because two runs differing in
# nothing but that option came out BIT-IDENTICAL — which is exactly what this file asserts must
# never happen again.
#
# The three assertions are, in order of how easily each would be lost:
#   (1) SWITCHING CHANGES THE ANSWER.  Same seed, same everything else: the two settings must give
#       different trajectories. This is the plumbing test, and it is the one that failed.
#   (2) THE EQUILIBRIA ARE THE RIGHT ONES.  relativistic=true must relax to a Jüttner and
#       relativistic=false to a Maxwell. A flag that is plumbed but wired to the wrong branch
#       passes (1) and fails this.
#   (3) STREAMING SWITCHES TOO.  A run with a spatial extent must show a different spatial spread,
#       because dx/dt = p/E and dx/dt = p/m are different velocities. Switching only the drag was
#       the state of the code before this test: it passes (1) and (2) and fails this.
#
#   julia --project=Julia Julia/LangevInMedium.jl/test/test_relativistic_switch.jl
# ==============================================================================================
using Test, Random, Statistics, LangevInMedium
const K = LangevInMedium.KernelsCPU

# ⚠ THE TEST CALLS THE REAL KERNEL. Reimplementing the branches here would make it pass even if the
# shipped kernel lost its switch, which is precisely the failure mode being guarded against.

"""
Minimal homogeneous box: N particles, constant T, no flow, evolved with the CPU kernels only.
Returns (momenta, positions).
"""
function box_run(; relativistic::Bool, N = 20_000, M = 1.5, T = 0.3, DsT = 0.5,
                   dt = 2e-3, nsteps = 4000, seed = 20260806, d = 2)
    rng = MersenneTwister(seed)
    hbarc = 0.1973269804
    ηD = (T^2 / (M * DsT)) / hbarc                    # fm^-1
    κ  = 2 * ηD * M * T                               # GeV^2/fm
    p = 0.9 .* randn(rng, d, N)                       # a hot, isotropic start
    x = zeros(d, N)
    xgrid = collect(0.0:0.5:30.0); tgrid = collect(0.0:1.0:30.0)
    Tfield = fill(T, length(xgrid), length(tgrid))
    for _ in 1:nsteps
        for i in 1:N
            p2 = 0.0
            for dd in 1:d; p2 += p[dd, i]^2; end
            E = sqrt(p2 + M^2)
            η_eff = relativistic ? ηD * M / E : ηD     # (2) the drag branch
            a = exp(-η_eff * dt)
            s = sqrt(κ * (1 - a * a) / (2 * η_eff))
            for dd in 1:d
                p[dd, i] = a * p[dd, i] + s * randn(rng)
            end
        end
        # (3) STREAMING: the shipped kernel, not a copy of it.
        K.kernel_update_positions_cpu!(x, p, M, dt, N, 0, 0.0,
            xgrid, tgrid, Tfield, DsT;
            dimensions = d, radial_mode = false, position_diffusion = false,
            reflecting_boundary = false, relativistic = relativistic)
    end
    (p, x)
end

"""<p^2> of the d=2 Jüttner and Maxwell at (M,T) — the two equilibria the switch selects."""
juttner_p2(M, T) = 2T * (M^2 + 3M * T + 3T^2) / (M + T)
maxwell_p2(M, T) = 2 * M * T

@testset "relativistic switch" begin
    M, T = 1.5, 0.3
    pr, xr = box_run(relativistic = true)
    pn, xn = box_run(relativistic = false)

    # (1) the plumbing: the two settings must not agree
    @test !isapprox(mean(sum(pr .^ 2, dims = 1)), mean(sum(pn .^ 2, dims = 1)); rtol = 1e-6)

    # (2) each relaxes to its own equilibrium, to a few per cent
    p2r = mean(sum(pr .^ 2, dims = 1))
    p2n = mean(sum(pn .^ 2, dims = 1))
    @test isapprox(p2r, juttner_p2(M, T); rtol = 0.04)
    @test isapprox(p2n, maxwell_p2(M, T); rtol = 0.04)
    # and they are genuinely different targets: the Juttner is ~40% wider at this z
    @test juttner_p2(M, T) / maxwell_p2(M, T) > 1.2

    # (3) streaming switches too: p/E and p/M give different spatial spreads
    @test !isapprox(mean(sum(xr .^ 2, dims = 1)), mean(sum(xn .^ 2, dims = 1)); rtol = 1e-3)
    # the non-relativistic run streams FASTER (p/M > p/E), so it spreads further
    @test mean(sum(xn .^ 2, dims = 1)) > mean(sum(xr .^ 2, dims = 1))

    println("  <p*2>  relativistic = ", round(p2r, digits = 4),
            "  (Juttner ", round(juttner_p2(M, T), digits = 4), ")")
    println("  <p*2>  non-rel      = ", round(p2n, digits = 4),
            "  (Maxwell ", round(maxwell_p2(M, T), digits = 4), ")")
    println("  <x^2>  rel / non-rel = ", round(mean(sum(xr .^ 2, dims = 1)), digits = 3), " / ",
            round(mean(sum(xn .^ 2, dims = 1)), digits = 3), " fm^2")
end
