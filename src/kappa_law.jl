module KappaLaws

# === momentum-dependent, anisotropic momentum diffusion: κ_L(p, T) ≠ κ_T(p, T) ===================
# Added 2026-10-24. Until then the engine had one coefficient: κ = 2MT/τ_drag, isotropic and
# p-independent (the `kL == kT == √κ` no-op in the radial branch of kernel_compute_all_forces_cpu!).
#
# The operator (rest frame, d momentum components, gradient form):
#     ∂_t f = ∂_i[ B_ij π ∂_j(f/π) ],   B_ij = (κ_L/2) p̂_i p̂_j + (κ_T/2)(δ_ij − p̂_i p̂_j),   π = e^{−E/T}
# so the Jüttner is stationary for any κ_L(p), κ_T(p). Written as an Itô process,
#     dp_i = −λ(p) p_i dt + [√κ_L p̂_i p̂_j + √κ_T (δ_ij − p̂_i p̂_j)] dW_j,
#     λ(p) = κ_L/(2TE) − κ_L'/(2p) − (d − 1)(κ_L − κ_T)/(2p²)                       (κ' = ∂κ/∂p)
# The last two terms are −∂_j B_ij / p: the spurious drift an Itô step picks up from a
# p-dependent, anisotropic noise. Dropping them (`drift = :naive`, the post-point drag κ_L/(2TE)
# alone) heats or cools the ensemble away from the Jüttner — test_kappa_law.jl K2 measures it.
# Constant κ: λ = κ/(2TE) = η_D M/E, the engine's relativistic drag.
#
# Normalisation. A law is a shape g_X(p, T) (dimensionless); the run's coefficients are
#     κ_X(p, T) = N(T) · κ₀(T) · g_X(p, T),   κ₀ = 2MT/τ_drag(T) = 2T³/D_sT      (the engine's κ)
# and N(T) is fixed by `norm`:
#   :green_kubo (default) — the operator's own Green–Kubo D_s equals D_sT/T, so the D_sT label keeps
#                           its meaning (spatial diffusion) and laws are compared at equal D_s, as
#                           KD does. N = D_GK[g]/(2T²) from the ℓ = 1 solve below.
#   :rest                 — κ_L(0) = κ₀: the drag at rest is the D_sT drag. D_s then differs.
#   :none                 — N = 1: g multiplies κ₀ as given.
#
# Everything the step needs is tabulated once on the host on a (p, T) grid — (N g_L, N g_T, N Λ) with
# the dimensionless drift Λ = 2TE·λ/κ₀ (= 1 for constant κ, so that law is exact on the table; the
# 1/(2TE) is applied exactly in the step) — and read by bilinear lookup on both backends (`kappa_lookup` is shared, not twinned), so the
# CPU and GPU take the same step from the same table.
# ==================================================================================================

using LinearAlgebra

export KappaLaw, KappaTable, build_kappa_table, kappa_green_kubo_ratio, kappa_lookup, kappa_step!

"""
    KappaLaw(gL, gT; norm = :green_kubo, pmax = 30.0, np = 3001, nT = 96)
    KappaLaw(g; kwargs...)                        # isotropic: g_L = g_T = g

Momentum diffusion along (`gL`) and across (`gT`) the heavy quark's rest-frame momentum, each a
callable `(p, T) -> shape` (p = |p*| and T in GeV, dimensionless result, even in p, finite and
positive). Pass it as `kappa_law` to `simulate_ensemble_bulk`; the run's κ_X(p, T) is
`N(T)·(2T³/D_sT)·g_X(p, T)` with N(T) fixed by `norm` (`:green_kubo`, `:rest`, `:none` — see the
module header). `pmax`, `np`: the momentum table (uniform in p; beyond `pmax` the edge value is
used); `nT`: temperature nodes over the background's range. g_L(0) = g_T(0) is required (at rest
there is no direction to be anisotropic about) and checked when the table is built.
"""
struct KappaLaw{FL, FT}
    gL::FL
    gT::FT
    norm::Symbol
    pmax::Float64
    np::Int
    nT::Int
    function KappaLaw(gL::FL, gT::FT; norm::Symbol = :green_kubo, pmax::Real = 30.0,
                      np::Integer = 3001, nT::Integer = 96) where {FL, FT}
        norm in (:green_kubo, :rest, :none) ||
            error("KappaLaw: norm must be :green_kubo, :rest or :none, got $norm")
        pmax > 0 || error("KappaLaw: pmax = $pmax must be positive")
        np >= 8 || error("KappaLaw: np = $np is too coarse (need ≥ 8)")
        nT >= 2 || error("KappaLaw: nT = $nT must be ≥ 2")
        return new{FL, FT}(gL, gT, norm, Float64(pmax), Int(np), Int(nT))
    end
end
KappaLaw(g; kwargs...) = KappaLaw(g, g; kwargs...)

"""
    KappaTable

The host table a run steps with: `vals[1:3, ip, iT] = (N g_L, N g_T, N Λ)` at
`p = (ip − 1)/invdp`, `T = Tmin + (iT − 1)/invdT`, with the dimensionless drift
Λ = 2TE·λ/κ₀ = g_L − TE g_L'/p − (d − 1)TE(g_L − g_T)/p² (λ = κ₀ N Λ/(2TE)). Built by
`build_kappa_table`; `norm` holds N(T) on the T nodes for inspection.
"""
struct KappaTable{A <: AbstractArray{Float64, 3}}
    invdp::Float64
    np::Int
    Tmin::Float64
    invdT::Float64
    nT::Int
    vals::A
    norm::Vector{Float64}
end

# ∂g/∂p by a 5-point stencil, using that g is even in p (g(−q) = g(q)), so it is exact-0 at p = 0.
function _dg(g, p, T)
    h = 1e-3 * max(p, 0.1)
    return (-g(p + 2h, T) + 8g(p + h, T) - 8g(abs(p - h), T) + g(abs(p - 2h), T)) / (12h)
end

"""
    kappa_green_kubo_ratio(gL, gT, m, T; d = 3, K = 4000) -> R

The Green–Kubo spatial diffusion coefficient of the operator with κ_L = g_L, κ_T = g_T, divided by
its constant-κ = 1 value 2T². With κ_X = N κ₀ g_X, the choice N = R gives D_s = 2T²/κ₀ = D_sT/T
exactly — the `:green_kubo` normalisation. R = 1 for constant g = 1.

D_s = ∫₀^∞ dt ⟨v_x(t) v_x(0)⟩ = (1/d)⟨(p/E) φ⟩, with φ(p) p̂_x = (−L†)⁻¹ v_x, i.e. the ℓ = 1 ODE
    (w g_L φ')' − (d − 1) w g_T φ/p² = −2w p/E,   w = p^{d−1} e^{−(E−m)/T},   φ(0) = 0,
solved by second-order finite volumes on p_k = k·h up to (E − m)/T = 50 (zero flux beyond).
Constant g: φ = 2Tp and R = 1 — the gate test_kappa_law.jl K1 checks the solver against that.
"""
function kappa_green_kubo_ratio(gL, gT, m::Real, T::Real; d::Integer = 3, K::Integer = 4000)
    M = Float64(m); Tf = Float64(T)
    (M > 0 && Tf > 0) || error("kappa_green_kubo_ratio: m = $m and T = $T must be positive")
    Emax = M + 50.0 * Tf
    pcut = sqrt(Emax^2 - M^2)
    h = pcut / K
    wfun(p) = (d == 1 ? 1.0 : p^(d - 1)) * exp(-(sqrt(M^2 + p^2) - M) / Tf)
    diag = Vector{Float64}(undef, K)
    off  = Vector{Float64}(undef, K - 1)
    rhs  = Vector{Float64}(undef, K)
    Wm = wfun(0.5h) * gL(0.5h, Tf)                     # flux weight at p_{k−½}
    for k in 1:K
        p  = k * h
        E  = sqrt(M^2 + p^2)
        w  = wfun(p)
        Wp = k < K ? wfun((k + 0.5) * h) * gL((k + 0.5) * h, Tf) : 0.0
        diag[k] = -(Wm + Wp) / h^2 - (d - 1) * w * gT(p, Tf) / p^2
        k < K && (off[k] = Wp / h^2)
        rhs[k] = -2.0 * w * p / E
        Wm = Wp
    end
    φ = SymTridiagonal(diag, off) \ rhs
    num = 0.0; den = 0.5 * wfun(0.0)                   # trapezoid; node 0 has φ = 0
    for k in 1:K
        p = k * h; E = sqrt(M^2 + p^2); w = wfun(p)
        c = k == K ? 0.5 : 1.0
        num += c * w * (p / E) * φ[k]
        den += c * w
    end
    return (num / d) / den / (2.0 * Tf^2)
end

"""
    build_kappa_table(law::KappaLaw, m, d; Tmin, Tmax, drift = :ito) -> KappaTable

Tabulate `law` for mass `m` and `d` momentum components over `[Tmin, Tmax]` (padded by 2 % each
side). `drift = :naive` drops the −∂_j B_ij terms of λ and exists only as the negative control of
test_kappa_law.jl K2 — it does not keep the Jüttner stationary.
"""
function build_kappa_table(law::KappaLaw, m::Real, d::Integer; Tmin::Real, Tmax::Real,
                           drift::Symbol = :ito)
    drift in (:ito, :naive) || error("build_kappa_table: drift must be :ito or :naive")
    d >= 1 || error("build_kappa_table: d = $d")
    M = Float64(m)
    lo = max(Float64(Tmin), 1e-4)
    hi = max(Float64(Tmax), lo * (1.0 + 1e-6))
    span = hi - lo
    lo = max(lo - 0.02 * span - 1e-3 * lo, 1e-4)
    hi = hi + 0.02 * span + 1e-3 * hi
    nT = law.nT; np = law.np
    dT = (hi - lo) / (nT - 1)
    dp = law.pmax / (np - 1)
    vals = Array{Float64, 3}(undef, 3, np, nT)
    Ns = Vector{Float64}(undef, nT)
    gL = law.gL; gT = law.gT
    for j in 1:nT
        T = lo + (j - 1) * dT
        gL0 = gL(0.0, T); gT0 = gT(0.0, T)
        (isfinite(gL0) && gL0 > 0) || error("KappaLaw: g_L(0, T = $T) = $gL0 is not a positive number")
        isapprox(gL0, gT0; rtol = 1e-6) ||
            error("KappaLaw: g_L(0) = $gL0 ≠ g_T(0) = $gT0 at T = $T — at rest κ cannot depend on a direction")
        N = law.norm === :green_kubo ? kappa_green_kubo_ratio(gL, gT, M, T; d = d) :
            law.norm === :rest       ? 1.0 / gL0 : 1.0
        Ns[j] = N
        # Λ = 2TE·λ/κ₀ (closure-local names deliberately distinct from the loop's — the shared-name trap)
        Λ(q) = begin
            TEq = T * sqrt(M^2 + q^2); gLq = gL(q, T)
            drift === :naive ? gLq :
                gLq - TEq * _dg(gL, q, T) / q - (d - 1) * TEq * (gLq - gT(q, T)) / q^2
        end
        for k in 1:np
            p = (k - 1) * dp
            l = gL(p, T); t = gT(p, T)
            (isfinite(l) && isfinite(t) && l > 0 && t > 0) ||
                error("KappaLaw: g_L = $l, g_T = $t at (p, T) = ($p, $T) — both must be positive and finite")
            vals[1, k, j] = N * l
            vals[2, k, j] = N * t
            # Λ is even in p and regular at 0; its formula is 0/0 there, so the p = 0 node takes the
            # p² extrapolation from the next two nodes (exact for α + βp², error O(dp⁴)).
            vals[3, k, j] = N * (k == 1 ? (4Λ(dp) - Λ(2dp)) / 3 : Λ(p))
        end
    end
    return KappaTable(1.0 / dp, np, lo, 1.0 / dT, nT, vals, Ns)
end

# Bilinear (p, T) lookup of the three table entries. Both weights clamped (no extrapolation) and
# no `Int(floor(·))` — it throws on NaN, and this runs inside CUDA kernels too.
@inline function kappa_lookup(vals, invdp::Float64, np::Int, Tmin::Float64, invdT::Float64,
                              nT::Int, p::Float64, T::Float64)
    u = p * invdp
    u = u > 0.0 ? (u < np - 1.0 ? u : np - 1.0) : 0.0
    v = (T - Tmin) * invdT
    v = v > 0.0 ? (v < nT - 1.0 ? v : nT - 1.0) : 0.0
    i = unsafe_trunc(Int, u) + 1; i = i < np ? i : np - 1
    j = unsafe_trunc(Int, v) + 1; j = j < nT ? j : nT - 1
    a = u - (i - 1); b = v - (j - 1)
    @inbounds begin
        gL = (1 - b) * ((1 - a) * vals[1, i, j] + a * vals[1, i + 1, j]) +
             b * ((1 - a) * vals[1, i, j + 1] + a * vals[1, i + 1, j + 1])
        gT = (1 - b) * ((1 - a) * vals[2, i, j] + a * vals[2, i + 1, j]) +
             b * ((1 - a) * vals[2, i, j + 1] + a * vals[2, i + 1, j + 1])
        Λ  = (1 - b) * ((1 - a) * vals[3, i, j] + a * vals[3, i + 1, j]) +
             b * ((1 - a) * vals[3, i, j + 1] + a * vals[3, i + 1, j + 1])
    end
    return gL, gT, Λ
end

"""
    kappa_step!(momenta, ξ, deterministic_terms, stochastic_terms, i, dimensions, κ0, dil, Δt, M, T,
                vals, invdp, np, Tmin, invdT, nT)

One particle's rest-frame step with the tabulated law — shared by the CPU and GPU kernels. The
drift λ and the two noise amplitudes are frozen at the start of the step and integrated as an
exponential (OU) step over the proper step Δt·dil:
    p ← a·p + √(dil·q·Δt) [√κ_T ξ + (√κ_L − √κ_T) p̂(p̂·ξ)],   a = e^{−λΔt·dil},  q = (1 − a²)/(2λΔt·dil)
(the update kernel supplies the √Δt). Constant g = 1 with `norm = :none` is the engine's own
exact-OU step to rounding (Λ ≡ 1 is exact on the table; K3 in test_kappa_law.jl). λ < 0 (a law
rising steeply in p) is handled: q stays positive.
"""
@inline function kappa_step!(momenta, ξ, deterministic_terms, stochastic_terms, i::Int,
                             dimensions::Int, κ0::Float64, dil::Float64, Δt::Float64, M::Float64,
                             T::Float64, vals, invdp::Float64, np::Int, Tmin::Float64,
                             invdT::Float64, nT::Int)
    p2 = 0.0; pξ = 0.0
    for d in 1:dimensions
        @inbounds pd = momenta[d, i]
        p2 += pd * pd
        @inbounds pξ += pd * ξ[d, i]
    end
    p = sqrt(p2)
    gL, gT, Λ = kappa_lookup(vals, invdp, np, Tmin, invdT, nT, p, T)
    λ  = T > 0.0 ? κ0 * Λ / (2.0 * T * sqrt(p2 + M * M)) : 0.0
    sL = sqrt(κ0 * (gL > 0.0 ? gL : 0.0))
    sT = sqrt(κ0 * (gT > 0.0 ? gT : 0.0))
    f  = λ * Δt * dil
    a  = exp(-f)
    q  = abs(f) > 1e-8 ? -expm1(-2.0 * f) / (2.0 * f) : 1.0 - f
    pref = sqrt(dil * q)
    # longitudinal part of the noise along p̂; at p = 0 there is no direction (and κ_L = κ_T)
    cL = p > 0.0 ? (sL - sT) * pξ / p2 : 0.0
    for d in 1:dimensions
        @inbounds pd = momenta[d, i]
        @inbounds deterministic_terms[d, i] = (a - 1.0) * pd
        @inbounds stochastic_terms[d, i] = pref * (sT * ξ[d, i] + cL * pd)
    end
    return nothing
end

end # module KappaLaws
