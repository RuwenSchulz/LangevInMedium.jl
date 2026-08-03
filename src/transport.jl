module Transport

using ..Constants: fmGeV
using Bessels

export tau_n_main3, tau_drag, build_tau_n_spline, eval_tau_n_spline, effective_DsT, build_juttner_invcdf
export LV_TAUN_SCALE

function __init__()
    LV_TAUN_SCALE[] = parse(Float64, get(ENV, "LV_TAUN_SCALE", "1.0"))
    LV_TAUN_SCALE[] == 1.0 ||
        @warn "LangevInMedium: τ_n scaled by LV_TAUN_SCALE — DIAGNOSTIC MODE (this also rescales η_D, κ and hence D_s)" scale=LV_TAUN_SCALE[]
end

const _TINY = 1e-300

"""
    LV_TAUN_SCALE

Diagnostic multiplier on the Langevin diffusion relaxation time τ_n. Default 1.0 = bare
`tau_n_main3`, i.e. unchanged production behaviour. Set `LV_TAUN_SCALE=0.16666666666666666` (1/6)
to impose the ÷g_hq convention that Fluidum's `τ_diffusion_hadron` carries, so both descriptions can
be run under the SAME convention.

⚠️ PHYSICS CAVEAT — this is NOT the same kind of knob as Fluidum's `HQ_TAUN_SCALE`. In hydro, κ and
τ_n are independent (κ from `diffusion_hadron`, τ_n from `τ_diffusion_hadron`), so scaling τ_n
changes the relaxation rate at FIXED D_s. In the Langevin they are locked by the
fluctuation–dissipation relation — `kernels_cpu.jl:236-242` sets η_D = 1/τ_n and κ = 2MT/τ_n from
the same τ_n — so scaling τ_n here rescales BOTH, and therefore moves D_s and the Navier–Stokes
limit as well. A "Langevin with the factor" run is a physically different medium, not merely a
differently-relaxing one. Interpret accordingly.

Implemented as a Ref populated in `__init__` rather than `const X = parse(ENV...)`: the latter is
evaluated at PRECOMPILE time and baked into the .ji, so the env var would be silently ignored in
every later process (this exact mistake produced bit-identical "different" runs in Fluidum).
"""
const LV_TAUN_SCALE = Ref(1.0)

"""
    tau_n_main3(T, m, DsT) -> Float64

Compute the diffusion relaxation time \\tau_n in **fm** using the Bessel-moment
matching \\tau_n = D_s z K_3(z)/K_2(z), with z = m/T and D_s = DsT/T. This is the
single-species form that FluiduM's `τ_diffusion` evaluates and is used on the
Langevin/Fokker–Planck side of the comparison.

Note: the AttractorPaper1 blast-wave spine and its analytic derivation
(`derivations.tex`) use a *different*, also literature-standard matching,
\\tau_n = D_s/c_n^2 with c_n^2 = (T/m) K_2(z)/K_1(z), i.e. \\tau_n ∝ z K_1(z)/K_2(z)
(see `AttractorRadialCommon.hq_taun`). By the recurrence K_3(z) − K_1(z) = (4/z)K_2(z)
the two forms differ by exactly +4 D_s (≈1.5× at T_fo, up to ~2.5× at T=0.4 GeV);
they are distinct relaxation-time matchings, not the same coefficient.

Inputs:
- `T` [GeV]
- `m` [GeV]
- `DsT` dimensionless
"""
@inline function tau_n_main3(T::Real, m::Real, DsT::Real)::Float64
    Tm = Float64(T)
    Mm = Float64(m)
    Ds = Float64(DsT)

    if !isfinite(Tm) || !isfinite(Mm) || !isfinite(Ds) || Tm <= 0.0 || Mm <= 0.0 || Ds <= 0.0
        return 0.0
    end

    z = Mm / Tm
    z <= 0.0 && return 0.0

    # τ_n = D_s z K₃(z)/K₂(z) with D_s = DsT/T (paper Eq. (A.tc-first)).
    # For z ≲ 50 we evaluate the equivalent rank-integral combination
    # (2K₁ − 3K₃ + K₅) = (48/z²)K₃, which matches the original matching algebra.
    # For z ≳ 50 that combination loses all significant digits to catastrophic
    # cancellation (the besselkx values become nearly equal), so we evaluate the
    # algebraically identical K₃/K₂ ratio directly — cancellation-free and
    # continuous with the z<50 branch at z=50.
    K2x = Bessels.besselkx(2, z)
    K3x = Bessels.besselkx(3, z)
    denominator = max(abs(K2x), _TINY)

    if z > 50.0
        ratio_K3K2 = K3x / denominator * sign(K2x == 0 ? 1.0 : K2x)
        τ_GeVinv = (Ds / Tm) * z * ratio_K3K2
        return min(τ_GeVinv / fmGeV, 1e20)
    end

    K1x = Bessels.besselkx(1, z)
    K5x = Bessels.besselkx(5, z)

    numerator = 2 * K1x - 3 * K3x + K5x
    ratio = numerator / denominator * sign(K2x == 0 ? 1.0 : K2x)

    z3_over_Tm = z^3 / Tm
    if !isfinite(z3_over_Tm) || z3_over_Tm > 1e50
        z3_over_Tm = 1e50
    end

    τ_GeVinv = (Ds / 48.0) * z3_over_Tm * ratio
    if !isfinite(τ_GeVinv) || abs(τ_GeVinv) > 1e50
        return 1e50 * sign(τ_GeVinv)
    end

    return τ_GeVinv / fmGeV
end

"""
    tau_drag(T, m, DsT) -> Float64      [fm]

The **drag** time, 1/η_D, i.e. what the Langevin step actually needs. Fixed by the Einstein
relation and nothing else:

    D_s = T/(m η_D)   ⇒   1/η_D = m D_s/T = m·DsT/T²        (×ħc for fm)

🔴 **THIS IS NOT `tau_n_main3`, AND CONFUSING THE TWO WAS A REAL BUG (fixed 2026-08-02).**
`tau_n_main3` = D_s z K₃/K₂ is the **diffusion-current relaxation time** — the τ_n of the
Israel–Stewart current equation, which is what Fluidum's `τ_diffusion_hadron` evaluates. For a
Fokker–Planck process with drag η_D the ℓ=1 mode relaxes at

    λ₁ = η_D · ⟨(m/E)p²⟩/⟨p²⟩ = η_D · K₂(z)/K₃(z)     (an identity in 3-D, verified to 4 digits)

so its current time is automatically `(1/η_D)·K₃/K₂`. Substituting `1/η_D = m DsT/T²` returns
`D_s z K₃/K₂` — Fluidum's τ_n exactly. **So a Langevin built with `tau_drag` reproduces BOTH of
hydro's coefficients: D_s (the Navier–Stokes limit) and τ_n (the relaxation time).** It has only one
free coefficient, and this is the value that makes both agree.

Building the drag from `tau_n_main3` instead — which `build_tau_n_spline` did until 2026-08-02 —
applies K₃/K₂ one time too many. The medium then has **D_s too large by K₃/K₂** (1.27× at z=10,
1.77× at z=3.75; measured directly from the particles' mean-square displacement in
`diag_clock_from_diffusion.jl`) **and a current relaxing K₃/K₂ too slowly**, i.e. it matches neither
hydro quantity while being labelled with D_sT.
"""
@inline function tau_drag(T::Real, m::Real, DsT::Real)::Float64
    Tm = Float64(T); Mm = Float64(m); Ds = Float64(DsT)
    (!isfinite(Tm) || !isfinite(Mm) || !isfinite(Ds) || Tm <= 0.0 || Mm <= 0.0 || Ds <= 0.0) &&
        return 0.0
    return Mm * Ds / (Tm^2) / fmGeV      # (GeV^-1) -> fm
end

@inline function effective_DsT(T::Real, DsT::Real; DsT_linear::Bool=false, DsT_slope::Real=1.765, DsT_offset::Real=-0.159, Tfo::Real=0.156)::Float64
    if !DsT_linear
        return Float64(DsT)
    end
    Tf = max(Float64(T), Float64(Tfo))
    return Float64(DsT_slope) * Tf + Float64(DsT_offset)
end

"""
    build_tau_n_spline(m, DsT; Tmin, Tmax, nT) -> (Tmin, invdT, tau_vals)

Build a fast **uniform-grid linear spline** for \\tau_n(T) (in fm):

- `Tmin`, `Tmax` in GeV
- `nT` number of grid points

Returns:
- `Tmin::Float64`
- `invdT::Float64` (where `dT = 1/invdT`)
- `tau_vals::Vector{Float64}` length `nT`
"""
function build_tau_n_spline(
    m::Real,
    DsT::Real;
    Tmin::Real,
    Tmax::Real,
    nT::Integer = 400,
    DsT_linear::Bool=false,
    DsT_slope::Real=1.765,
    DsT_offset::Real=-0.159,
    Tfo::Real=0.156,
)
    Tmin_f = Float64(Tmin)
    Tmax_f = Float64(Tmax)
    n = Int(nT)
    n < 2 && error("nT must be >= 2")
    Tmin_f > 0.0 || error("Tmin must be > 0")
    Tmax_f > Tmin_f || error("Tmax must be > Tmin")

    dT = (Tmax_f - Tmin_f) / (n - 1)
    invdT = 1.0 / dT

    tau_vals = Vector{Float64}(undef, n)
    @inbounds for i in 1:n
        Ti = Tmin_f + (i - 1) * dT
        DsT_eff = effective_DsT(Ti, DsT; DsT_linear=DsT_linear, DsT_slope=DsT_slope, DsT_offset=DsT_offset, Tfo=Tfo)
        # 🔴 THE DRAG, not the current relaxation time. This line called `tau_n_main3` until
        # 2026-08-02, which made the realised D_s too large by K₃/K₂ — see `tau_drag`.
        tau_vals[i] = tau_drag(Ti, m, DsT_eff) * LV_TAUN_SCALE[]
    end
    return Tmin_f, invdT, tau_vals
end

"""
    eval_tau_n_spline(T, Tmin, invdT, tau_vals) -> Float64

Evaluate the uniform-grid linear spline produced by `build_tau_n_spline`.
"""
@inline function eval_tau_n_spline(
    T::Real,
    Tmin::Float64,
    invdT::Float64,
    tau_vals::AbstractVector{<:Real},
)::Float64
    n = length(tau_vals)
    n < 2 && return Float64(first(tau_vals))

    x = Float64(T)
    if !isfinite(x)
        return 0.0
    end

    u = (x - Tmin) * invdT
    i = Int(floor(u)) + 1
    i = clamp(i, 1, n - 1)
    t = u - (i - 1)

    y0 = Float64(tau_vals[i])
    y1 = Float64(tau_vals[i + 1])
    return (1.0 - t) * y0 + t * y1
end

"""
    build_juttner_invcdf(m, dimensions; Tmin, Tmax, nT, nU, np)
        -> (invCDF::Vector{Float64}, nU::Int, nT::Int, Ttab_min::Float64, Ttab_invdT::Float64)

Precompute a HOST-side inverse-CDF table for the isotropic local-rest-frame Jüttner
momentum-magnitude distribution

    P(p*) ∝ p*^(d-1) · exp(-(E-m)/T),   E = sqrt(m² + p*²),   d = `dimensions`,

the EXACT same shape sampled by rejection in `_draw_juttner_pstar_lib` (kernels_cpu.jl).
This lets a GPU thread draw `p*` by a constant-work table lookup instead of a variable-length
rejection loop (divergence-free).

Layout: flat column-major, `invCDF[(jT-1)*nU + iu]` = p*(u_iu ; T_jT), with
- u_iu = (iu-1)/(nU-1)            uniform on [0,1],          iu = 1..nU
- T_jT = Ttab_min + (jT-1)/invdT  uniform on [Tmin,Tmax],    jT = 1..nT

For each T the CDF F(p*) = ∫₀^{p*}P / ∫₀^{∞}P is built by cumulative-trapezoid on a fine p*
grid capped at `pmax = 16√(mT) + 16T` (same cap as the CPU rejection sampler), then inverted
onto the uniform u-grid by a monotone two-pointer sweep. Upload the returned vector as a
CuArray and index it in `kernel_rta_collision_gpu!` with a bilinear interpolation in (u,T).
"""
function build_juttner_invcdf(
    m::Real, dimensions::Integer;
    Tmin::Real, Tmax::Real,
    nT::Integer = 256, nU::Integer = 2048, np::Integer = 8192,
    quantile_cap::Real = 1.0e-6,
)
    M  = Float64(m)
    d  = Int(dimensions)
    a  = d - 1
    nTi = Int(nT); nUi = Int(nU); npi = Int(np)
    nTi >= 2 || error("nT must be >= 2")
    nUi >= 2 || error("nU must be >= 2")
    npi >= 2 || error("np must be >= 2")

    # Clamp/pad the temperature range so the table safely brackets every T the medium can hand
    # the kernel (bilinear lookups clamp to the edges, so a small margin is enough).
    Tmin_f = max(Float64(Tmin), 1e-4)
    Tmax_f = max(Float64(Tmax), Tmin_f * (1.0 + 1e-6))
    span   = Tmax_f - Tmin_f
    Tmin_f = max(Tmin_f - 0.02 * span, 1e-4)
    Tmax_f = Tmax_f + 0.02 * span

    dT    = (Tmax_f - Tmin_f) / (nTi - 1)
    invdT = 1.0 / dT

    invCDF = Vector{Float64}(undef, nUi * nTi)

    pgrid = Vector{Float64}(undef, npi)
    cdf   = Vector{Float64}(undef, npi)

    @inbounds for jT in 1:nTi
        T    = Tmin_f + (jT - 1) * dT
        pmax = 16.0 * sqrt(M * T) + 16.0 * T
        dp   = pmax / (npi - 1)

        # --- fine p* grid, PDF, and cumulative-trapezoid CDF ---
        pgrid[1] = 0.0
        # PDF value w(p) = p^a exp(-(E-m)/T); a==0 ⇒ p^0 = 1 (incl. p=0).
        wprev = (a == 0) ? 1.0 : 0.0        # p=0: p^a = 0 for a>=1, 1 for a=0
        cdf[1] = 0.0
        for k in 2:npi
            p = (k - 1) * dp
            pgrid[k] = p
            E = sqrt(M * M + p * p)
            w = (a == 0) ? exp(-(E - M) / T) : (p^a) * exp(-(E - M) / T)
            cdf[k] = cdf[k - 1] + 0.5 * (wprev + w) * dp
            wprev = w
        end
        norm = cdf[npi]
        invnorm = norm > 0.0 ? 1.0 / norm : 0.0
        for k in 1:npi
            cdf[k] *= invnorm
        end
        cdf[npi] = 1.0   # guard against roundoff so u=1 maps to pmax

        # High-quantile cap: `pmax` (=16√(mT)+16T) is a *generous* rejection envelope bound whose
        # far tail carries exp(-(E-m)/T)≈0 weight. The CPU rejection sampler visits it with the
        # correct (vanishing) probability, but linear inverse-CDF interpolation across the final
        # u-bin would smear a 1/nU fraction of draws all the way up to pmax, inflating ⟨p²⟩. Clamp
        # the table to the p where the CDF first reaches 1-quantile_cap (a real ~1e-6 quantile),
        # matching the CPU sampler's effective support to well within MC noise.
        p_hi = pgrid[npi]
        thr = 1.0 - Float64(quantile_cap)
        for k in 1:npi
            if cdf[k] >= thr
                p_hi = pgrid[k]
                break
            end
        end

        # --- invert F(p)=u onto the uniform u-grid by a monotone two-pointer sweep ---
        base = (jT - 1) * nUi
        k = 1
        for iu in 1:nUi
            u = (iu - 1) / (nUi - 1)
            while k < npi && cdf[k + 1] < u
                k += 1
            end
            c0 = cdf[k]; c1 = cdf[k + 1]
            p0 = pgrid[k]; p1 = pgrid[k + 1]
            frac = (c1 > c0) ? (u - c0) / (c1 - c0) : 0.0
            frac = frac < 0.0 ? 0.0 : (frac > 1.0 ? 1.0 : frac)
            pval = p0 + frac * (p1 - p0)
            invCDF[base + iu] = pval < p_hi ? pval : p_hi
        end
    end

    return invCDF, nUi, nTi, Tmin_f, invdT
end

end # module Transport
