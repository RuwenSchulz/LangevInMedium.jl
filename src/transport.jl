module Transport

using ..Constants: fmGeV
using Bessels

export tau_n_main3, build_tau_n_spline, eval_tau_n_spline

const _TINY = 1e-300

"""
    tau_n_main3(T, m, DsT) -> Float64

Compute the diffusion relaxation time \\tau_n in **fm** using the same logic as
`diff_tauN_bg` (generic EOS path) in `FiVoHydro.jl/main3.jl`.

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

    if z > 50.0
        τ_GeVinv = (Ds / 48.0) * (Mm^2 / (Tm^2 + 1e-10))
        τ_fm = τ_GeVinv / fmGeV
        return min(τ_fm, 1e20)
    end

    K1x = Bessels.besselkx(1, z)
    K2x = Bessels.besselkx(2, z)
    K3x = Bessels.besselkx(3, z)
    K5x = Bessels.besselkx(5, z)

    numerator = 2 * K1x - 3 * K3x + K5x
    denominator = max(abs(K2x), _TINY)
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
        tau_vals[i] = tau_n_main3(Ti, m, DsT)
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

end # module Transport
