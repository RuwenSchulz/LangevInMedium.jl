module Utils

using Interpolations, Random, Statistics, LinearAlgebra
using Distributions: Normal, Truncated, MixtureModel
using ..Constants

export sample_initial_particles_at_origin_no_position!
export sample_particles_from_FONLL
export append_thermal_pz, append_comoving_pz, append_pz, sample_pz_conditional_juttner, check_momentum_dims

"""
    _cumtrapz(y, x) -> Vector{Float64}

Cumulative trapezoid of `y` over the ACTUAL nodes `x`: `out[1] = 0`, `out[k] = ∫_{x₁}^{x_k} y dx`.

🔴 FIXED 2026-09-02 — this replaces `cumsum(y) * mean(diff(x))` throughout the FONLL sampler.
That form is a right-Riemann sum with ONE constant spacing, which is two separate errors: it is
only first order even on a uniform grid, and on a non-uniform grid it is simply the wrong
quadrature, so refinement cannot fix it. Measured against the exact ⟨p⟩ of `P(p) ∝ p·f(p)` with a
FONLL-like `(1+(p/2.1)²)^(−3.1)` (`test_limits_and_contracts.jl` L6):

| p grid over [0, 10] | old (Riemann) | new (trapezoid) |
|---|---|---|
| uniform, np = 100 | −3.14 % | +0.05 % |
| uniform, np = 300 (production) | **−1.08 %** | **−0.02 %** |
| uniform, np = 1200 | −0.29 % | −0.03 % |
| log-spaced, 300 | **−44.6 %** | −0.01 % |

⚠ THIS MOVES EVERY FONLL INITIAL CONDITION IN THE TREE by ≈ 1 % in ⟨p_T⟩ (the production grid is
np = 300, pmax = 10). It is a bias on the IC, so it partly cancels in a ratio that carries the same
IC top and bottom. Six of the ten `regression_corpus.jl` hashes were regenerated for it.
"""
function _cumtrapz(y::AbstractVector, x::AbstractVector)
    n = length(y)
    n == length(x) || error("_cumtrapz: length mismatch $(length(y)) vs $(length(x))")
    out = zeros(Float64, n)
    @inbounds for k in 2:n
        out[k] = out[k-1] + 0.5 * (Float64(y[k]) + Float64(y[k-1])) * (Float64(x[k]) - Float64(x[k-1]))
    end
    return out
end

"Trapezoid quadrature weights on the nodes `x`: `∫f dx ≈ Σ w[i] f[i]`. Non-uniform safe."
function _trapz_weights(x::AbstractVector)
    n = length(x)
    w = zeros(Float64, n)
    n < 2 && return w
    @inbounds for k in 1:n-1
        h = 0.5 * (Float64(x[k+1]) - Float64(x[k]))
        w[k] += h; w[k+1] += h
    end
    return w
end

"Normalise a cumulative array into a strictly increasing CDF on [0, 1] (or return `nothing`)."
function _to_cdf!(c::Vector{Float64})
    n = length(c)
    (n >= 2 && isfinite(c[n]) && c[n] > 0) || return nothing
    c[1] = 0.0
    @inbounds for k in 2:n
        c[k] = max(c[k], c[k-1] + eps(Float64))
    end
    c ./= c[n]
    c[n] = 1.0
    return c
end

"""
    sample_particles_from_FONLL(r_grid, p_grid, f, N; cartesian_spatial_sampling=false) -> (x, p)

Draw `N` particles from a tabulated initial phase-space density `f[p_index, r_index]` (an
`(Nr, Np)` table is transposed automatically). Returns lab positions `x` `(2, N)` and
transverse momenta `p` `(2, N)` with isotropic azimuth; |p| is drawn from the conditional
`P(p | r) ∝ p·f(r, p)` by inverse CDF on the nearest tabulated `r`.

Spatial sampling: `cartesian_spatial_sampling = true` rejection-samples `(x, y)` uniformly in the
disc `r ≤ r_grid[end]` against `n(r) = ∫p f dp` — no `r → 0` grid artefacts, but the acceptance is
the fireball's area fraction of the disc (a few % for a Pb+Pb profile in a 20 fm disc); `false`
inverse-CDF samples `P(r) ∝ r·n(r)` on the grid. Uses the global RNG (`Random.seed!`).
"""
function sample_particles_from_FONLL(r_grid, p_grid, f_HQ_init_FONLL, N_samples::Int;
                                     n_cdf_points=500,
                                     cartesian_spatial_sampling::Bool=false)

    # 1. Normalize full PDF: P(r,p) ∝ r * p * f(r,p)
    # Expect f_HQ_init_FONLL to be indexed as f[p_index, r_index] (Np × Nr).
    # If user passes Nr × Np, transpose it for robustness.
    Nr = length(r_grid)
    Np = length(p_grid)
    f = f_HQ_init_FONLL
    if !(size(f, 1) == Np && size(f, 2) == Nr)
        if size(f, 1) == Nr && size(f, 2) == Np
            f = permutedims(f)
        else
            error("sample_particles_from_FONLL: f_HQ_init_FONLL has size $(size(f)), expected ($Np, $Nr) (p×r) or ($Nr, $Np) (r×p).")
        end
    end

    # 🔴 2026-09-02: every quadrature below is a TRAPEZOID ON THE ACTUAL NODES (`_cumtrapz` /
    # `_trapz_weights`). It used to be `cumsum(·) * mean(diff(grid))`, a right-Riemann sum on an
    # assumed-uniform grid — first order where the grid IS uniform and simply wrong where it is not.
    # See the `_cumtrapz` docstring for the measured table.
    r_nodes = collect(Float64, r_grid)
    p_nodes = collect(Float64, p_grid)
    wp = _trapz_weights(p_nodes)          # ∫dp weights, used for every marginal over p

    # --- Precompute conditional CDFs for p|r (needed by both sampling modes) ---
    # Build P(p|r) ∝ p * f(r,p) for each r
    P_p_given_r = @. p_grid .* f  # shape (Np, Nr), unnormalized
    inverse_cdf_p_given_r = Vector{Any}(undef, Nr)
    for i in 1:Nr
        p_pdf = P_p_given_r[:, i]
        cdf_p = _to_cdf!(_cumtrapz(p_pdf, p_nodes))
        if cdf_p === nothing
            inverse_cdf_p_given_r[i] = _ -> 0.0
        else
            inverse_cdf_p_given_r[i] = LinearInterpolation(cdf_p, p_nodes, extrapolation_bc=Flat())
        end
    end

    x_matrix = zeros(2, N_samples)
    p_matrix = zeros(2, N_samples)

    if cartesian_spatial_sampling
        # =====================================================================
        # Cartesian (x,y) rejection sampling — no grid artifacts at r→0
        # =====================================================================
        # P(x,y) ∝ n_spatial(√(x²+y²))  (no geometric r factor in Cartesian)

        rmax = Float64(last(r_grid))

        # Marginal spatial density on radial grid: n_spatial(r) = ∫ p f(r,p) dp  (trapezoid in p)
        f_spatial = vec(P_p_given_r' * wp)               # length Nr

        # Interpolate onto arbitrary r
        r_grid_vec = r_nodes
        f_spatial_interp = LinearInterpolation(r_grid_vec, f_spatial, extrapolation_bc=0.0)

        # Envelope for rejection: maximum of n_spatial(r) over the grid
        f_max = maximum(f_spatial) + eps()

        n_accepted = 0
        n_trials   = 0
        @info "Cartesian rejection sampling: rmax=$(rmax) fm, f_max=$(f_max)"

        while n_accepted < N_samples
            n_trials += 1
            # Uniform proposal in the square [-rmax, rmax]²
            x = rmax * (2*rand() - 1)
            y = rmax * (2*rand() - 1)
            r = sqrt(x^2 + y^2)

            # Reject points outside the disk or by density ratio
            if r > rmax
                continue
            end
            if rand() * f_max > f_spatial_interp(r)
                continue
            end

            n_accepted += 1
            x_matrix[:, n_accepted] .= (x, y)

            # --- Sample p|r using the conditional CDF ---
            r = clamp(r, first(r_grid), last(r_grid))
            j = searchsortedlast(r_grid_vec, r)
            j = clamp(j, 1, Nr)
            p_mag = inverse_cdf_p_given_r[j](rand())
            φp = 2π * rand()
            p_matrix[:, n_accepted] .= (p_mag * cos(φp), p_mag * sin(φp))
        end

        @info "Rejection sampling done: $(n_accepted) accepted / $(n_trials) trials " *
              "(efficiency = $(round(100*n_accepted/n_trials; digits=1))%)"

    else
        # =====================================================================
        # Original polar (r,φ) spatial sampling
        # =====================================================================
        P_rp = @. r_grid' .* p_grid .* f  # shape (Np, Nr)

        # Marginal PDF for r: integrate over p with the trapezoid weights, then a cumulative
        # trapezoid over the r nodes. (The overall normalisation cancels in `_to_cdf!`, so the
        # separate Z is gone; the zero-distribution guard it carried is kept below.)
        P_r = vec(P_rp' * wp)
        cdf_r = _to_cdf!(_cumtrapz(P_r, r_nodes))
        cdf_r === nothing && error("Distribution normalization is zero.")
        inverse_cdf_r = LinearInterpolation(cdf_r, r_nodes, extrapolation_bc=Flat())

        for i in 1:N_samples
            # Sample r, φ
            r = inverse_cdf_r(rand())
            r = clamp(r, first(r_grid), last(r_grid))
            φ = 2π * rand()
            x_matrix[:, i] .= (r * cos(φ), r * sin(φ))

            # Sample p|r, φ_p
            j = searchsortedlast(r_grid, r)
            j = clamp(j, 1, length(r_grid))
            p_mag = inverse_cdf_p_given_r[j](rand())
            φp = 2π * rand()
            p_matrix[:, i] .= (p_mag * cos(φp), p_mag * sin(φp))
        end
    end

    return x_matrix, p_matrix
end


function sample_initial_particles_at_origin_no_position!(initial_condition,
    p0, dimensions, N_particles)


    function sample_bimodal_p_vectors(dimensions, N_particles;
        μ1=1.0, μ2=2.0, σ=0.2,
        weight1=0.5, pmin=0.1, pmax=5.0)

        # Bimodal magnitude distribution
        d1 = Truncated(Normal(μ1, σ), pmin, pmax)
        d2 = Truncated(Normal(μ2, σ), pmin, pmax)
        mix = MixtureModel([d1, d2], [weight1, 1 - weight1])
        p_mags = rand(mix, N_particles)

        # Sample random directions
        momenta = zeros(Float64, dimensions, N_particles)
        for i in 1:N_particles
        dir = randn(dimensions)
        dir ./= norm(dir)  # normalize to unit vector
        momenta[:, i] .= p_mags[i] * dir
        end

        return momenta
    end


    if initial_condition == "delta"
        rand_dirs = randn(Float64, dimensions, N_particles)
        # Normalize columns (L2 norm across each particle's vector)
        norms = sqrt.(sum(rand_dirs .^ 2, dims=1))
        rand_dirs ./= norms  # Broadcasted division to normalize
        momenta = zeros(Float64, dimensions, N_particles)
        momenta .= p0 .* rand_dirs
    elseif initial_condition == "bimodal"
        momenta = sample_bimodal_p_vectors(dimensions, N_particles)
  
    else 
        error("Unknown initial condition: $initial_condition")
    end


    return momenta
end

# ═══════════════════════════════════════════════════════════════════════════════════════════════
# THREE MOMENTUM COMPONENTS ON A TWO-DIMENSIONAL TRANSVERSE PLANE  (2026-08-21)
#
# WHY. The engine's `dimensions` couples the spatial and the momentum dimensionality, so every
# midrapidity run (positions in the transverse plane) evolved a TWO-component momentum. Drag and
# noise are isotropic, so the equilibrium such a run relaxes to is the 2-D Jüttner on the measure
# p dp dφ — a different kinetic theory from the 3-D one in which the hydrodynamic coefficients
# (τ_n = D_s z K₃/K₂, κ) are matched. The drag η_D = T/(M D_s) is dimension-independent and was
# always right; the DERIVED current-relaxation rate is λ₁ η_D with
#     λ₁ = ⟨(M/E) p²⟩/⟨p²⟩  =  K₂/K₃        in 3-D (exact identity)
#                            ≠  K₂/K₃        in 2-D (5–12 % larger over z = 3.5–10),
# so a 2-D ensemble compared against a 3-D τ_n carried a convention offset of that size
# (AttractorHydro §setup, "5–9 % over this background"). `momentum_dimensions = 3` removes it:
# a third momentum component lives on every particle, invariant under the transverse flow boost,
# entering E* (drag, streaming, Cooper–Frye) and the isotropic kicks. Positions stay 2-D.
#
# INITIAL p_z. Given a sampled transverse momentum p_T, the thermal conditional at the local
# temperature is  f(p_z | p_T) ∝ exp(−√(m_T² + p_z²)/T),  m_T = √(m² + p_T²)  — exact for a
# Jüttner initial state, and the natural completion of a FONLL/boosted transverse IC (midrapidity).
# ═══════════════════════════════════════════════════════════════════════════════════════════════

"""
    sample_pz_conditional_juttner(mT, T; rng) -> p_z

One exact draw from f(p_z) ∝ exp(−√(m_T²+p_z²)/T). In the rapidity variable p_z = m_T sinh y the
density is cosh(y)·exp(−a cosh y), a = m_T/T, and a Gaussian N(0, c/a) is a valid rejection
envelope with acceptance probability cosh(y)·exp(−a(cosh y − 1) + a y²/(2c)) ≤ 1 whenever
2a(1 − 1/c) ≥ 2, i.e. c ≥ a/(a−1): the log-derivative tanh y − a(1−1/c) y is then ≤ 0 for y ≥ 0,
so the ratio peaks at y=0 where it equals one. Acceptance is ≈ 1/√c ≈ 70 % at c = 2.
For a ≤ 1.05 (a bath hotter than the mass, not a heavy-quark regime) a wide Laplace envelope in
p_z is used instead, valid since √(m_T²+p_z²) ≥ (m_T+|p_z|)/√2.
"""
function sample_pz_conditional_juttner(mT::Float64, T::Float64; rng::AbstractRNG = Random.default_rng())
    (T > 0.0 && mT > 0.0) || return 0.0
    a = mT / T
    if a > 1.05
        c = max(2.0, a / (a - 1.0))
        σ = sqrt(c / a)
        @inbounds for _ in 1:100_000
            y = σ * randn(rng)
            ch = cosh(y)
            acc = ch * exp(-a * (ch - 1.0) + a * y * y / (2.0 * c))
            rand(rng) <= acc && return mT * sinh(y)
        end
        return 0.0
    else
        # Laplace envelope g ∝ exp(−|p_z|/(√2 T)) ≥ exp(−√(m_T²+p_z²)/T)·exp(m_T/(√2T)) — loose but safe.
        b = sqrt(2.0) * T
        @inbounds for _ in 1:1_000_000
            pz = -b * log(rand(rng)) * (rand(rng) < 0.5 ? -1.0 : 1.0)
            acc = exp(-(sqrt(mT * mT + pz * pz) - (mT + abs(pz)) / sqrt(2.0)) / T)
            rand(rng) <= acc && return pz
        end
        return 0.0
    end
end

"""
    append_thermal_pz(momenta, positions, m, T_of_r; antithetic=false, rng) -> momenta3

Return a (3, N) copy of the (2, N) transverse `momenta` with a third row p_z drawn from the
thermal conditional at the particle's local temperature `T_of_r(r)`, r = |x_⊥|. The input
momenta must be LOCAL-REST-FRAME momenta (this is called before the initial lab boost).
`antithetic = true` mirrors p_z across the (2i−1, 2i) pairs the samplers build, keeping the
pairs exact reflections in all three components.
"""
function append_thermal_pz(momenta::AbstractMatrix, positions::AbstractMatrix, m::Real, T_of_r;
                           antithetic::Bool = false, rng::AbstractRNG = Random.default_rng(),
                           # On a 2-D background the local temperature is NOT a function of the
                           # radius, so `T_of_r` cannot express it. Supplying `T_of_xy(x, y)`
                           # overrides it; absent, the radial path is bit-identical.
                           T_of_xy = nothing)
    size(momenta, 1) == 2 || error("append_thermal_pz: expected 2 transverse momentum rows, got $(size(momenta, 1))")
    N = size(momenta, 2)
    out = zeros(Float64, 3, N)
    @inbounds out[1:2, :] .= momenta
    @inbounds for i in 1:N
        if antithetic && iseven(i) && i > 1
            out[3, i] = -out[3, i - 1]
            continue
        end
        r2 = 0.0
        for d in 1:size(positions, 1); r2 += Float64(positions[d, i])^2; end
        T  = T_of_xy === nothing ?
                max(Float64(T_of_r(sqrt(r2))), 0.0) :
                max(Float64(T_of_xy(Float64(positions[1, i]),
                                    size(positions, 1) >= 2 ? Float64(positions[2, i]) : 0.0)), 0.0)
        mT = sqrt(Float64(m)^2 + Float64(momenta[1, i])^2 + Float64(momenta[2, i])^2)
        out[3, i] = sample_pz_conditional_juttner(mT, T; rng = rng)
    end
    return out
end

"""
    append_comoving_pz(momenta) -> momenta3

Return a (3, N) copy of the (2, N) transverse `momenta` with a third row of EXACT ZEROS.

This is not "neglecting p_z": it is the production kinematics. A quark created in the hard
scattering at t = z = 0 and free-streaming to the hydro start time arrives at

    t = τ₀ cosh y,   z = τ₀ sinh y   ⇒   η_s = artanh(z/t) = y   (exactly)

so its rapidity relative to the fluid at its own position, y − η_s, is exactly zero, and
p_z* = m_T sinh(y − η_s) = 0. Two consequences worth knowing:

  * it is independent of τ₀ — a free-streaming quark is comoving at EVERY time, not just at τ₀,
    because the Bjorken field v = z/t and a free worldline z = v t are the same relation;
  * the whole production rapidity spectrum then lives in η_s, not in p_z*. dN/dy is recovered as
    ρ(η_s) ⊛ P(K) with the kernel from `track_eta_s` — see `kernel_accumulate_eta_s_cpu!`.

It is also the internally consistent partner of a non-thermal FONLL p_T: `:thermal` asserts the
quark is longitudinally equilibrated at τ₀ while transversally it is not, and nothing does that.
"""
function append_comoving_pz(momenta::AbstractMatrix)
    size(momenta, 1) == 2 || error("append_comoving_pz: expected 2 transverse momentum rows, got $(size(momenta, 1))")
    out = zeros(Float64, 3, size(momenta, 2))
    @inbounds out[1:2, :] .= momenta
    return out
end

"""
    append_pz(mode, momenta, positions, m, T_of_r; antithetic=false, rng) -> momenta3

The p_z completion, selected by `mode`:

  `:thermal`  — the thermal conditional at the local T(r, τ₀) (`append_thermal_pz`). The SHIPPED
                DEFAULT, and what every product generated before 2026-09-01 carries.
  `:comoving` — p_z* = 0 (`append_comoving_pz`), the free-streaming initial condition.

Measured cost of switching (2026-09-01, LP1 `pbpb_const_fonll` config, real Pb+Pb background):
freeze-out p_T spectrum < 1 %, final dN/dy 2 %, rapidity kernel 6 %. The two become
indistinguishable by τ ≈ 1 fm — at T = 0.58 the drag time is ≈ 0.10 fm, so the initial p_z is
forgotten almost immediately. `:comoving` is the defensible choice; the default stays `:thermal`
so existing products remain bit-identical.
"""
function append_pz(mode::Symbol, momenta::AbstractMatrix, positions::AbstractMatrix, m::Real, T_of_r;
                   antithetic::Bool = false, rng::AbstractRNG = Random.default_rng(),
                   T_of_xy = nothing)
    if mode === :thermal
        return append_thermal_pz(momenta, positions, m, T_of_r; antithetic = antithetic, rng = rng,
                                 T_of_xy = T_of_xy)
    elseif mode === :comoving
        return append_comoving_pz(momenta)
    else
        error("append_pz: pz_init=$(repr(mode)) is not supported (only :thermal and :comoving).")
    end
end

"""
    check_momentum_dims(dimensions, pdim, radial_mode, bjorken_redshift, initial_time)

The contract: `pdim == dimensions` is the historical (bit-identical) engine; the only other
supported combination is a 2-D transverse plane carrying 3 momentum components. The Bjorken
redshift dp_z/dτ = −p_z/τ is the longitudinal free-streaming of a boost-invariant system and is
only meaningful for that combination, with a positive initial Milne time.
"""
function check_momentum_dims(dimensions::Int, pdim::Int, radial_mode::Bool, bjorken_redshift::Bool, initial_time::Real;
                             pz_init::Symbol = :thermal, track_eta_s::Bool = false)
    if pdim != dimensions
        (dimensions == 2 && pdim == 3 && !radial_mode) ||
            error("momentum_dimensions=$pdim with dimensions=$dimensions is not supported: " *
                  "only (dimensions=2, momentum_dimensions=3) differs from the coupled default.")
    end
    if bjorken_redshift
        (dimensions == 2 && pdim == 3) ||
            error("bjorken_redshift requires dimensions=2 and momentum_dimensions=3 (p_z is row 3).")
        initial_time > 0 || error("bjorken_redshift needs initial_time > 0 (Milne τ; dp_z/dτ = −p_z/τ).")
    end
    (pz_init === :thermal || pz_init === :comoving) ||
        error("pz_init=$(repr(pz_init)) is not supported (only :thermal and :comoving).")
    if pz_init === :comoving && pdim < 3
        error("pz_init=:comoving is a statement about the p_z row; it needs momentum_dimensions=3.")
    end
    if track_eta_s
        pdim >= 3 || error("track_eta_s needs momentum_dimensions=3 (η_s is driven by row 3).")
        initial_time > 0 || error("track_eta_s needs initial_time > 0 (dη_s/dτ = (1/τ)(p_z*/E*)).")
    end
    return nothing
end

end # module Utils
