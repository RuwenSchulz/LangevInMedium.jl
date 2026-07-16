#!/usr/bin/env julia
# ==============================================================================
# LangevInMedium.jl unit tests — the transport-kernel physics primitives that the
# heavy-quark Langevin integrator (and its FluiduM/FiVo comparison) rest on:
#
#   * the matched relaxation time  τ_n = D_s z K₃/K₂  (Transport.tau_n_main3),
#     incl. the z>50 catastrophic-cancellation fast path and the NR limit;
#   * the diffusion-coefficient prescription effective_DsT (const / linear);
#   * the Einstein / fluctuation-dissipation closure  κ = 2 M T η_D  used by the
#     CPU kernel (kernels_cpu.jl: η_D = 1/τ_n, κ = 2MT/τ_n), via the exact OU
#     propagator's stationary variance ⟨p²⟩ → κ/(2η_D) = M T.
#
# These were previously proven only inside LangevinPaper1's harness; they belong
# with the library so `Pkg.test()` on LangevInMedium exercises them directly.
#
#   julia --project=Julia/LangevInMedium.jl -e 'using Pkg; Pkg.test()'
# ==============================================================================

using Test
using LangevInMedium          # exports tau_n_main3, effective_DsT, fmGeV
using Bessels
using Statistics
using Random

# Exp-scaled Bessel K (the e^{-z} cancels in any ratio / linear combination at fixed z).
Kx(ν, z) = Bessels.besselkx(ν, z)

# Paper reference: τ_n = (DsT/T)·z·K₃(z)/K₂(z) / fmGeV  [fm], z = M/T.
tau_n_reference(T, M, DsT) = (DsT / T) * (M / T) * (Kx(3, M / T) / Kx(2, M / T)) / fmGeV

const M_CHARM = 1.5
const DST     = 0.11634

@testset "LangevInMedium transport kernels" begin

    @testset "Bessel matching identity 2K₁-3K₃+K₅ = (48/z²)K₃" begin
        for z in (0.5, 1.0, 2.0, 5.0, 9.6, 20.0, 40.0)
            @test isapprox(2Kx(1, z) - 3Kx(3, z) + Kx(5, z), (48 / z^2) * Kx(3, z); rtol = 1e-9)
        end
    end

    @testset "tau_n_main3 ≡ D_s z K₃/K₂ and positive" begin
        for T in (0.6, 0.5, 0.4, 0.3, 0.25, 0.2, 0.156, 0.12, 0.1)
            @test isapprox(tau_n_main3(T, M_CHARM, DST), tau_n_reference(T, M_CHARM, DST); rtol = 1e-8)
            @test tau_n_main3(T, M_CHARM, DST) > 0
        end
        # invalid inputs return 0 (guard the early-out branch)
        @test tau_n_main3(-1.0, M_CHARM, DST) == 0.0
        @test tau_n_main3(0.3, M_CHARM, 0.0) == 0.0
        @test tau_n_main3(0.3, M_CHARM, -1.0) == 0.0
    end

    @testset "z>50 fast path is correct and continuous across z=50" begin
        T_below = M_CHARM / 49.99   # general Bessel branch
        T_above = M_CHARM / 50.01   # fast-path branch
        @test isapprox(tau_n_main3(T_below, M_CHARM, DST), tau_n_reference(T_below, M_CHARM, DST); rtol = 1e-8)
        @test isapprox(tau_n_main3(T_above, M_CHARM, DST), tau_n_reference(T_above, M_CHARM, DST); rtol = 1e-6)
        @test isapprox(tau_n_main3(T_below, M_CHARM, DST), tau_n_main3(T_above, M_CHARM, DST); rtol = 1e-3)
    end

    @testset "non-relativistic limit τ_n → M D_s/T (K₃/K₂ → 1)" begin
        ratio(T) = tau_n_main3(T, M_CHARM, DST) * T^2 * fmGeV / (DST * M_CHARM)
        @test ratio(0.05) > 1.0
        @test ratio(0.05) < ratio(0.20)
        @test isapprox(ratio(0.02), 1.0; atol = 0.05)   # z = 75: within 5% of NR limit
    end

    @testset "effective_DsT prescription (const / linear)" begin
        # constant mode is T-independent
        @test effective_DsT(0.3, DST) == DST
        @test effective_DsT(0.6, DST) == DST
        # linear mode: DsT(T) = slope·max(T,Tfo) + offset, floored at Tfo
        slope, offset, Tfo = 1.765, -0.159, 0.156
        @test isapprox(effective_DsT(0.30, DST; DsT_linear=true), slope*0.30 + offset; rtol=1e-12)
        @test isapprox(effective_DsT(0.10, DST; DsT_linear=true), slope*Tfo + offset; rtol=1e-12)  # floored
        @test isapprox(effective_DsT(Tfo,  DST; DsT_linear=true), slope*Tfo + offset; rtol=1e-12)
    end

    @testset "Einstein / FDR closure: κ = 2MTη_D ⟹ OU variance → MT" begin
        # The kernel defines η_D = 1/τ_n, κ = 2MT/τ_n, so κ/(2η_D) = MT exactly.
        T  = 0.3
        τn = tau_n_main3(T, M_CHARM, DST)
        ηD = 1.0 / τn
        κ  = 2 * M_CHARM * T / τn
        @test isapprox(κ / (2ηD), M_CHARM * T; rtol = 1e-12)

        # Exact OU propagator (kernels_cpu.jl, non-relativistic η_eff=η_D):
        #   p ← a·p + √Δt·noise_pref·ξ,  a=e^{-η_D Δt},  noise_pref=√κ·√((1-a²)/(2 η_D Δt)).
        # Its stationary variance is κ/(2η_D) = MT.
        Δt = 0.02 * τn
        a  = exp(-ηD * Δt)
        noise_pref = sqrt(κ) * sqrt((1 - a^2) / (2 * ηD * Δt))
        Random.seed!(20260615)
        Npart, Nsteps = 40_000, 800
        p = zeros(Npart)
        for _ in 1:Nsteps
            @. p = a * p + sqrt(Δt) * noise_pref * randn()
        end
        @test isapprox(var(p), M_CHARM * T; rtol = 0.05)
    end
end
