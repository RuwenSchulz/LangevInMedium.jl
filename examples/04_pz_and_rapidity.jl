#!/usr/bin/env julia
# ==============================================================================================
# 04 — THE LONGITUDINAL SECTOR: p_z*, its initialisation, and the dN/dy kernel
#
# `momenta[3, :]` is not a lab p_z. It is `p_z* = m_T sinh(y − η_s)`, the longitudinal momentum in
# the frame comoving with the Bjorken fluid at the particle's own space-time rapidity. The package
# stores no `z` and no `η_s`, legitimately — η_s is CYCLIC, it never enters the equations of motion.
# This example is about the three consequences:
#
#   1. `pz_init` — how row 3 is filled at τ₀. `:thermal` (shipped) draws the local Jüttner
#      conditional; `:comoving` sets p_z* = 0, which is the PRODUCTION-KINEMATICS answer: a quark
#      created at t = z = 0 and free-streaming to τ₀ arrives at η_s = y exactly.
#   2. `bjorken_redshift` — dp_z*/dτ = −p_z*/τ between kicks, the longitudinal work term a 2-D
#      momentum run simply omits.
#   3. `track_eta_s` — integrate dη_s/dτ = (1/τ)(p_z*/E*) and get the rapidity kernel, from which
#      dN/dy = ρ(η_s) ⊛ P(K) EXACTLY, because K = Δη_s + y*(freeze-out) is independent of η_s(τ₀).
#      One boost-invariant run therefore gives the kernel for ANY production profile.
#
#   julia --project=Julia Julia/LangevInMedium.jl/examples/04_pz_and_rapidity.jl
# ==============================================================================================
include(joinpath(@__DIR__, "example_common.jl"))

const M   = 1.5
const DST = 0.11634
const N   = 100_000
const τ0, τf = 0.4, 12.0

xg, tg, Tf, Vf = bjorken_fireball(; τ0 = τ0, τf = τf)
r_grid = collect(range(0.0, 20.0; length = 150))
p_grid = collect(range(0.0, 10.0; length = 300))
density = fonll_density(r_grid, p_grid; σ = 3.0)

function go(pz_init)
    Random.seed!(4)
    simulate_ensemble_bulk(CPUBackend(), r_grid, p_grid, density, Tf, Vf, (xg, tg);
        N_particles = N, Δt = 5e-3, initial_time = τ0, final_time = τf, save_interval = 0.4,
        m = M, DsT = DST, dimensions = 2,
        # polar inverse-CDF sampling. The Cartesian mode rejection-samples the whole disc and
        # its acceptance is the fireball's AREA FRACTION of it — 3.6 % for this profile in a
        # 26 fm disc, i.e. ~28 proposals per particle, host-side and serial.
        cartesian_spatial_sampling = false,
        momentum_dimensions = 3,       # row 3 exists
        pz_init          = pz_init,    # ...and this is how it is filled
        bjorken_redshift = true,       # ...and this is how it evolves between kicks
        track_eta_s      = true,       # ⇒ a FOURTH returned element: the η_s history
        proper_time_kicks = true, Tfo = 0.156)
end

println("── the two initialisations of row 3 ──")
res = Dict{Symbol,Any}()
for mode in (:thermal, :comoving)
    t, mom, pos, eta = go(mode)        # NOTE the fourth element — only with track_eta_s = true
    t = collect(t)
    pz2  = [mean(view(m, 3, :) .^ 2) for m in mom]
    pT2  = [mean(sum(abs2, view(m, 1:2, :); dims = 1)) for m in mom]
    res[mode] = (t, pz2, pT2, mom, eta)
    @printf("  pz_init = %-9s ⟨p_z*²⟩(τ₀) = %.5f GeV²   ⟨p_z*²⟩/(⟨p_T²⟩/2) at τ₀ = %.3f\n",
            mode, pz2[1], pz2[1] / (pT2[1] / 2))
end
println("""
  `:thermal` asserts the quark is longitudinally EQUILIBRATED at τ₀ while transversally it is not:
  the ratio printed above is well below 1 precisely because the FONLL p_T is much harder than the
  bath, so row 3 starts thermal while rows 1–2 do not. `:comoving` starts at exactly zero — the
  production-kinematics answer, and the internally consistent partner of a non-thermal p_T. They
  are indistinguishable within about a fermi (the drag time at T ≈ 0.5 is ≈ 0.1 fm), which is why
  the choice moves the freeze-out spectrum by well under a percent.""")

t_th, pz2_th, _, mom_th, eta_th = res[:thermal]
t_co, pz2_co, _, mom_co, eta_co = res[:comoving]
@printf("\n  ⟨p_z*²⟩ at τ = %.1f fm:  thermal %.5f   comoving %.5f   (differ by %.1f %%)\n",
        t_th[end], pz2_th[end], pz2_co[end], 100 * abs(pz2_co[end] / pz2_th[end] - 1))

# ── the rapidity kernel ────────────────────────────────────────────────────────────────────────
# η_s is accumulated from ZERO on purpose: the engine returns only the CHANGE, because the
# production value η_s(τ₀) cannot enter the dynamics and so must not enter the run. The kernel is
#       K = Δη_s + y*,      y* = atanh(p_z*/E*) at freeze-out
# and because K is independent of η_s(τ₀) by construction, dN/dy = ρ(η_s) ⊛ P(K) is EXACT: convolve
# the kernel below with whatever production rapidity profile you believe.
for (name, mom, eta) in (("thermal", mom_th, eta_th), ("comoving", mom_co, eta_co))
    mf = mom[end]
    E★ = sqrt.(M^2 .+ vec(sum(abs2, mf; dims = 1)))
    y★ = atanh.(clamp.(view(mf, 3, :) ./ E★, -0.999999, 0.999999))
    K  = eta[end] .+ y★
    @printf("  %-9s kernel: ⟨Δη_s⟩ = %+.5f, sd = %.4f | ⟨y*⟩ = %+.5f, sd = %.4f | ⟨K⟩ = %+.5f, sd(K) = %.4f\n",
            name, mean(eta[end]), std(eta[end]), mean(y★), std(y★), mean(K), std(K))
end
println("""
  ⟨K⟩ ≈ 0 and sd(K) is the rapidity SMEARING the medium applies to a quark produced at η_s: that
  single number is what a dN/dy prediction needs. η_s is a PASSENGER — nothing in the dynamics
  reads it — so `track_eta_s` cannot move any other output (the engine gates that: momenta and
  positions are bit-identical with it on and off).""")

if plots_on()
    pa = plot(t_th, pz2_th; m = :circle, c = :steelblue, xlabel = "τ [fm]",
              ylabel = "⟨p_z*²⟩ [GeV²]", label = "pz_init = :thermal",
              title = "the two initialisations are forgotten")
    plot!(pa, t_co, pz2_co; m = :square, c = :firebrick, label = "pz_init = :comoving")
    edges = range(-2.5, 2.5; length = 60)
    pb = plot(; xlabel = "K = Δη_s + y*", ylabel = "density", title = "the dN/dy kernel P(K)")
    for (name, mom, eta, col) in (("thermal", mom_th, eta_th, :steelblue),
                                  ("comoving", mom_co, eta_co, :firebrick))
        mf = mom[end]
        E★ = sqrt.(M^2 .+ vec(sum(abs2, mf; dims = 1)))
        K  = eta[end] .+ atanh.(clamp.(view(mf, 3, :) ./ E★, -0.999999, 0.999999))
        c, h = hist(K, edges)
        plot!(pb, c, h; c = col, label = name)
    end
    savefig_ex(plot(pa, pb; layout = (1, 2), size = (1200, 460)), "04_pz_and_rapidity.png")
end
