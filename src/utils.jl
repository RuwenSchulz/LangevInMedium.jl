module Utils

using Interpolations, QuadGK, Random, Statistics, LinearAlgebra
using Distributions: Uniform, Normal, Truncated, MixtureModel
using ..Constants

export sample_initial_particles_from_pdf!
export sample_initial_particles_at_origin!
export sample_initial_particles_at_origin_no_position!
export sample_particles_from_density
export sample_particles_from_FONLL
export append_thermal_pz, sample_pz_conditional_juttner, check_momentum_dims

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

    dr = mean(diff(collect(r_grid)))
    dp = mean(diff(collect(p_grid)))

    # --- Precompute conditional CDFs for p|r (needed by both sampling modes) ---
    # Build P(p|r) ∝ p * f(r,p) for each r
    P_p_given_r = @. p_grid .* f  # shape (Np, Nr), unnormalized
    inverse_cdf_p_given_r = Vector{Any}(undef, Nr)
    for i in 1:Nr
        p_pdf = P_p_given_r[:, i]
        if sum(p_pdf) > 0
            cdf_p = cumsum(p_pdf) * dp
            cdf_p[1] = 0.0
            for k in 2:length(cdf_p)
                cdf_p[k] = max(cdf_p[k], cdf_p[k-1] + eps(Float64))
            end
            cdf_p ./= cdf_p[end]
            cdf_p[end] = 1.0
            inverse_cdf_p_given_r[i] = LinearInterpolation(cdf_p, p_grid, extrapolation_bc=Flat())
        else
            inverse_cdf_p_given_r[i] = _ -> 0.0
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

        # Marginal spatial density on radial grid: n_spatial(r) = ∫ p f(r,p) dp
        f_spatial = vec(sum(P_p_given_r, dims=1)) * dp   # length Nr

        # Interpolate onto arbitrary r
        r_grid_vec = collect(Float64, r_grid)
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

        # total normalization
        Z = sum(P_rp) * dr * dp
        if Z == 0
            error("Distribution normalization is zero.")
        end
        P_rp ./= Z

        # Marginal PDF for r
        P_r = sum(P_rp, dims=1)[:] * dp  # integrate over p
        cdf_r = cumsum(P_r) * dr
        if !isempty(cdf_r)
            cdf_r[1] = 0.0
            for i in 2:length(cdf_r)
                cdf_r[i] = max(cdf_r[i], cdf_r[i-1] + eps(Float64))
            end
            cdf_r ./= cdf_r[end]
            cdf_r[end] = 1.0
        end
        inverse_cdf_r = LinearInterpolation(cdf_r, r_grid, extrapolation_bc=Flat())

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


function sample_particles_from_density(r_values, n_rt, N_samples::Int, T_interp,nur_interp, fugacity_interp;
                                       n_cdf_points=1000, rmax=10.0, t0=0.0,
                                       mode::Symbol = :density, m=1.5)

    # helper: sample |p| from relativistic 2D MB at local T
    sample_p_mag = function (T_local)
        # simple robust cutoff (same as your MB2D branch)
        pmax = 15 * max(T_local, eps())
        p_grid = range(0, pmax, length=800)
        dp = step(p_grid)
        # f(p) ∝ p * exp(-sqrt(p^2 + m^2)/T)
        f_p = @. p_grid * exp(-sqrt(p_grid^2 + m^2)/max(T_local, eps()))
        Z = sum(f_p) * dp
        if Z == 0.0
            return 0.0
        end
        f_p ./= Z
        cdf_p = cumsum(f_p) * dp
        cdf_p[end] = 1.0
        inverse_cdf_p = LinearInterpolation(cdf_p, p_grid, extrapolation_bc=Line())
        return inverse_cdf_p(rand())
    end

    if mode == :density
        # ------------------------------
        # 1) Spatial sampling from user-supplied n_rt(r)
        # ------------------------------
        interp = LinearInterpolation(r_values, n_rt, extrapolation_bc=0.0)
        norm, _ = quadgk(r -> r * interp(r), 0, rmax)

        # Precompute CDF for r
        r_cdf = range(0, rmax, length=n_cdf_points)
        cdf_values = zeros(n_cdf_points)
        for i in 2:n_cdf_points
            result, _ = quadgk(r′ -> r′ * interp(r′) / norm, 0, r_cdf[i])
            cdf_values[i] = result
        end

        # Enforce monotonicity
        for i in 2:n_cdf_points
            if cdf_values[i] <= cdf_values[i-1]
                cdf_values[i] = cdf_values[i-1] + 1e-9
            end
        end
        cdf_values[end] = 1.0
        inverse_cdf = LinearInterpolation(cdf_values, collect(r_cdf), extrapolation_bc=Line())

     

        # ------------------------------
        # 2) Sample positions and relativistic MB momenta
        # ------------------------------
        x_matrix = zeros(2, N_samples)
        p_matrix = zeros(2, N_samples)

        for i in 1:N_samples
            # position from n_rt
            r = inverse_cdf(rand())
            φ = 2π * rand()
            x_matrix[1,i] = r * cos(φ)
            x_matrix[2,i] = r * sin(φ)

            # momentum from SAME relativistic MB as MB2D
            T_local = T_interp(r, t0)
            p_mag = sample_p_mag(T_local)
            φp = 2π * rand()
            p_matrix[1,i] = p_mag * cos(φp)
            p_matrix[2,i] = p_mag * sin(φp)
        end

    elseif mode == :MB2D
        # ------------------------------
        # 1) Spatial sampling from thermal MB radial density
        #     n_MB(r) ∝ r * ∫ p exp(-sqrt(p^2+m^2)/T(r)) dp
        # ------------------------------
        n_MB = zeros(length(r_values))
        for (j, r) in enumerate(r_values)
            T_local = T_interp(r, t0)
            pmax = 15 * max(T_local, eps())
            p_grid = range(0, pmax, length=800)
            dp = step(p_grid)
            f_p = @. p_grid * exp(-sqrt(p_grid^2 + m^2)/max(T_local, eps()))
            n_MB[j] = sum(f_p) * dp * r  # include 2D Jacobian r
        end

        # Normalize and build CDF over r_values
        total = sum(n_MB)
        if total == 0.0
            error("Thermal radial weights are zero; check T_interp and parameters.")
        end
        n_MB ./= total
        cdf_values = cumsum(n_MB)
        cdf_values ./= cdf_values[end]
        inverse_cdf = LinearInterpolation(cdf_values, collect(r_values), extrapolation_bc=Line())

        # ------------------------------
        # 2) Sample positions and SAME relativistic MB momenta
        # ------------------------------
        x_matrix = zeros(2, N_samples)
        p_matrix = zeros(2, N_samples)

        for i in 1:N_samples
            # position from thermal n_MB(r)
            r = inverse_cdf(rand())
            φ = 2π * rand()
            x_matrix[1,i] = r * cos(φ)
            x_matrix[2,i] = r * sin(φ)

            # momentum from SAME relativistic MB
            T_local = T_interp(r, t0)
            p_mag = sample_p_mag(T_local)
            φp = 2π * rand()
            p_matrix[1,i] = p_mag * cos(φp)
            p_matrix[2,i] = p_mag * sin(φp)
        end

    else
        error("Invalid mode. Use :density or :MB2D")
    end

    return x_matrix, p_matrix
end



function sample_initial_particles_from_pdf!(
    m, dim, N_particles,
    t, T_profile, ur_profile, mu_profile,
    x_range::Tuple{Float64, Float64}, nbins::Int    
    )
    positions = zeros(dim, N_particles)
    momenta = zeros(dim, N_particles)

    # Discretize radial domain
    x_edges = range(x_range[1], x_range[2], length=nbins + 1)
    dx = step(x_edges)
    x_centers = (x_edges[1:end-1] .+ x_edges[2:end]) ./ 2

    # Compute normalized PDF over radial positions
    T_vals  = T_profile.(x_centers, t)
    ur_vals = ur_profile.(x_centers, t)
    mu_vals = mu_profile.(x_centers, t)
    γ_vals  = sqrt.(1 .+ ur_vals .^ 2)

    n_boltz = T_vals .^ (3/2) ./ γ_vals .* exp.((mu_vals .- m .* γ_vals) ./ T_vals)
    pdf_vals = n_boltz ./ sum(n_boltz) ./ dx  
    # Compute unnormalized PDF
    #pdf_vals = n_boltz .* x_centers         # Include 2D volume element (r * dr)
    max_pdf = maximum(pdf_vals)             # For rejection threshold

    # Rejection sampling
    range_sampler = Uniform(x_range[1], x_range[2])
    sampled_r = Float64[]

    while length(sampled_r) < N_particles
        r_try = rand(range_sampler)
        idx = searchsortedfirst(x_centers, r_try)
        p = idx <= length(pdf_vals) ? pdf_vals[idx] : 0.0
        if rand() < p / max_pdf
            push!(sampled_r, r_try)
        end
    end

    # Assign positions and thermal momenta
    for i in 1:N_particles
        r = sampled_r[i]
        T = T_profile(r, t)
        σ = sqrt(m * T)
        positions[:, i] .= r                  # Uniform radial position
        momenta[:, i] .= abs.(σ .* randn(dim))      # Gaussian-distributed thermal momentum
    end

    return positions, momenta
end

function sample_initial_particles_at_origin!(
    m, dim, N_particles,
    t, T_profile)
    positions = zeros(dim, N_particles)
    momenta = zeros(dim, N_particles)

    T = T_profile(0.0, t)
    σ = sqrt(m * T)

    for i in 1:N_particles
        positions[:, i] .= 0.0
        momenta[:, i] .= σ .* randn(dim)
    end

    return positions, momenta
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
                           antithetic::Bool = false, rng::AbstractRNG = Random.default_rng())
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
        T  = max(Float64(T_of_r(sqrt(r2))), 0.0)
        mT = sqrt(Float64(m)^2 + Float64(momenta[1, i])^2 + Float64(momenta[2, i])^2)
        out[3, i] = sample_pz_conditional_juttner(mT, T; rng = rng)
    end
    return out
end

"""
    check_momentum_dims(dimensions, pdim, radial_mode, bjorken_redshift, initial_time)

The contract: `pdim == dimensions` is the historical (bit-identical) engine; the only other
supported combination is a 2-D transverse plane carrying 3 momentum components. The Bjorken
redshift dp_z/dτ = −p_z/τ is the longitudinal free-streaming of a boost-invariant system and is
only meaningful for that combination, with a positive initial Milne time.
"""
function check_momentum_dims(dimensions::Int, pdim::Int, radial_mode::Bool, bjorken_redshift::Bool, initial_time::Real)
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
    return nothing
end

end # module Utils
