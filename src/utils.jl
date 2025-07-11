module Utils

# === Imports ===
using StaticArrays
using Plots
using Bessels
using Interpolations
using Statistics
using LinearAlgebra
using Distributions
using DataStructures
using LaTeXStrings
using StatsBase: Histogram, fit
using Statistics: mean, std
using Printf

using ..Constants

# === Exports ===
export sample_initial_particles_from_pdf!
export sample_initial_particles_milne!
export sample_MB_r_p_pairs!
export MB_distribution
export sample_initial_particles_at_origin!
export sample_initial_particles_at_origin_no_position!
export n_rt
export plot_n_rt_comparison_hydro_langevin
export sample_phase_space
export sample_phase_space2

# === Function Definitions ===

"""
    sample_initial_particles_from_pdf!(m, dim, N_particles, t, T_profile, ur_profile, mu_profile, x_range, nbins)

Sample initial particle positions and momenta from a Boltzmann distribution based on hydrodynamic profiles.

# Arguments
- `m`: Particle mass in GeV.
- `dim`: Number of spatial dimensions.
- `N_particles`: Total number of particles to sample.
- `t`: Current time in simulation.
- `T_profile`, `ur_profile`, `mu_profile`: Functions of (r, t) giving temperature, radial velocity, and chemical potential.
- `x_range`: Tuple indicating radial domain (min, max).
- `nbins`: Number of spatial bins for constructing the distribution.

# Returns
- `(positions, momenta)`: Arrays of size `(dim, N_particles)` for sampled positions and momenta.
"""
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


"""
    sample_initial_particles_milne!(m, dim, N_particles, τ, T_profile, ur_profile, mu_profile, x_range)

Sample initial particle positions and momenta in Milne coordinates (τ, r) from a Boltzmann distribution.

# Arguments
- `m`: Particle mass in GeV.
- `dim`: Number of spatial dimensions (should be 2: τ, r).
- `N_particles`: Number of particles to sample.
- `τ`: Proper time (Milne time) of initialization.
- `T_profile`, `ur_profile`, `mu_profile`: Functions of (r, τ).
- `x_range`: Tuple (min_r, max_r).
- `nbins`: Number of bins for inverse-CDF sampling.

# Returns
- `(positions, momenta)`: Arrays of shape `(2, N_particles)` for positions (τ, r) and momenta (p^τ, p^r).
"""
function sample_initial_particles_milne!(
    m, dim::Int, N_particles::Int,
    τ::Float64, T_profile, ur_profile, mu_profile, x_range::Tuple{Float64, Float64}, nbins::Int
    )
    @assert dim == 2 "Milne sampling requires dim = 2 (τ, r)"
    positions = zeros(dim, N_particles)
    momenta = zeros(dim, N_particles)

    # Discretize radial domain
    x_edges = range(x_range[1], x_range[2], length=nbins + 1)
    dx = step(x_edges)
    x_centers = (x_edges[1:end-1] .+ x_edges[2:end]) ./ 2

    # Compute normalized PDF over radial positions
    T_vals  = T_profile.(x_centers, τ)
    ur_vals = ur_profile.(x_centers, τ)
    mu_vals = mu_profile.(x_centers, τ)
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

    # Sampling loop
    for i in 1:N_particles
        r = sampled_r[i]
        T = T_profile(r, τ)
        σ = sqrt(m * T)
        pr = σ * randn(dim-1) 
        pτ = sqrt(m^2 + dot(pr, pr))
        positions[:, i] .= (τ, r)
        momenta[:, i]   .= (pτ, abs.(pr[1]))
    end

    return positions, momenta
end 

function sample_phase_space2(N_particles::Int, r_grid::Vector{Float64}, t::Float64,
                            m, T_profile, mu_profile, dimss)

    # 1. Particle density profile n(r, t) × r × t (with Bessel function)
    function nrt_free_cham_rt(r, t)
        T = T_profile(r, t)
        z = exp(mu_profile(r, t))  # fugacity
        ex = z * exp(-m / T)
        b2 = Bessels.besselkx(2, m / T)
        deg = 6  # spin x2, color x3
        density = deg * (m^2 * T / (2π^2) * ex * b2) * fmGeV^3
        return density * r * t
    end

    # 2. Normalize n(r,t) × r × t to get probability weights
    function normalize_nrt_discrete(r_grid::Vector{Float64}, t::Float64)
        vals = [nrt_free_cham_rt(r, t) for r in r_grid]
        vals .= max.(vals, 0.0)  # ensure non-negative
        dr = r_grid[2] - r_grid[1]
        norm = sum(vals) * dr
        return vals ./ norm
    end

    # 3. Sample positions using inverse CDF
    function sample_positions_cdf(N_particles, r_grid, t)
        weights = normalize_nrt_discrete(r_grid, t)
        cdf = cumsum(weights)
        cdf ./= cdf[end]  # normalize
        samples = rand(N_particles)
        indices = searchsortedfirst.(Ref(cdf), samples)
        return r_grid[indices]
    end

    # 4. Count how many particles fall into each r_bin
    function count_particles_in_1D_grid(sampled_positions::Vector{Float64}, r_grid::Vector{Float64})
        Δr = r_grid[2] - r_grid[1]
        edges = vcat(r_grid .- Δr/2, r_grid[end] + Δr/2)
        counts = zeros(Int, length(r_grid))
        for r in sampled_positions
            bin = searchsortedfirst(edges, r) - 1
            if bin ≥ 1 && bin ≤ length(counts)
                counts[bin] += 1
            end
        end
        return counts
    end

    # 5. Faster momentum sampling using pre-tabulated thermal spectrum
    function sample_momenta_vectorized(N, T, m)
        E_p(p) = sqrt(p^2 + m^2)
        f_p(p) = p.^2 .* exp.(-E_p.(p) ./ T)

        # Tabulate PDF and CDF
        p_vals = range(0, stop=10*T, length=1000)
        pdf_vals = f_p(p_vals)
        pdf_vals ./= sum(pdf_vals)
        cdf_vals = cumsum(pdf_vals)
        cdf_vals ./= cdf_vals[end]

        # Sample using inverse CDF
        rands = rand(N)
        indices = searchsortedfirst.(Ref(cdf_vals), rands)
        return p_vals[indices]
    end

    # 6. Sample momenta for each radial bin
    function sample_momenta_at_each_ri(N_particles_at_ri::Vector{Int}, r_grid::Vector{Float64})
        momenta_by_r = Vector{Vector{Float64}}(undef, length(r_grid))
        for (i, r) in enumerate(r_grid)
            T = T_profile(r, t)
            N = N_particles_at_ri[i]
            if N > 0
                momenta_by_r[i] = sample_momenta_vectorized(N, T, m)
            else
                momenta_by_r[i] = Float64[]
            end
        end
        return momenta_by_r
    end

    # 7. Flatten phase space arrays
    function flatten_phase_space(momenta_at_ri::Vector{Vector{Float64}}, r_grid::Vector{Float64})
        positions = Float64[]
        momenta = Float64[]
        for (i, r) in enumerate(r_grid)
            for p in momenta_at_ri[i]
                push!(positions, r)
                push!(momenta, p)
            end
        end
        return positions, momenta
    end

    # === Execute Steps ===
    sampled_positions = sample_positions_cdf(N_particles, r_grid, t)
    N_particles_at_ri = count_particles_in_1D_grid(sampled_positions, r_grid)
    momenta_by_r = sample_momenta_at_each_ri(N_particles_at_ri, r_grid)
    positions, momenta = flatten_phase_space(momenta_by_r, r_grid)

    # 8. Construct output arrays
    pos = zeros(dimss, N_particles)
    mom = zeros(dimss, N_particles)
    pos[1, :] .= positions
    mom[1, :] .= momenta

    return pos, mom
end

function sample_phase_space(N_particles::Int, r_grid::Vector{Float64}, t::Float64,m,T_profile,mu_profile,dimss)
    # 1. Rejection sample positions
    function nrt_free_cham_rt(r, t)
        T = T_profile(r, t)
        z = exp(mu_profile(r, t))  # fugacity
        ex = z * exp(-m / T)               # this is z * e^{-m/T}
        b2 = Bessels.besselkx(2, m / T)
        deg = 6  # spin x2, color x3
        density = deg * (m^2 * T / (2 * π^2) * ex * b2) * fmGeV^3
        return density* r *t
    end

    function normalize_nrt_discrete(r_grid::Vector{Float64}, t::Float64)
        vals = [nrt_free_cham_rt(r, t) for r in r_grid]
        dr = r_grid[2] - r_grid[1]
        norm = sum(vals) * dr  # discrete approximation of ∫ n(r) dr
        normalized_vals = vals ./ norm
        return normalized_vals
    end


    function rejection_sample_positions(N_particles, r_grid::Vector{Float64}, t)
        weights = normalize_nrt_discrete(r_grid,t)#[nrt_free_cham_rt(r, t) for r in r_grid]
        for i in eachindex(weights)
            if !isfinite(weights[i]) || weights[i] < 0
                weights[i] = 0.0
            end
        end
        f_max = maximum(weights)
        if f_max == 0
            error("All weights are zero or invalid. Cannot sample.")
        end
        r_min, r_max = extrema(r_grid)
        sampled_r = Float64[]
        while length(sampled_r) < N_particles
            r_try = rand() * (r_max - r_min) + r_min
            y_try = rand() * f_max
            f_r = nrt_free_cham_rt(r_try, t)
            if isfinite(f_r) && f_r ≥ y_try
                push!(sampled_r, r_try)
            end
        end
        return sampled_r
    end

    function count_particles_in_1D_grid(sampled_positions::Vector{Float64}, r_grid::Vector{Float64})
        Δr = r_grid[2] - r_grid[1]
        edges = vcat(r_grid .- Δr/2, r_grid[end] + Δr/2)
        counts = zeros(Int, length(r_grid))
        for r in sampled_positions
            bin = searchsortedfirst(edges, r) - 1
            if bin ≥ 1 && bin ≤ length(counts)
                counts[bin] += 1
            end
        end
        return counts
    end

    function sample_momenta_at_each_ri(N_particles_at_ri::Vector{Int}, r_grid::Vector{Float64})
       
        momenta_by_r = Vector{Vector{Float64}}(undef, length(r_grid))
        for (i, r) in enumerate(r_grid)
            T = T_profile(r, t)
            N = N_particles_at_ri[i]
            sampled = Float64[]
            E_p(p) = sqrt(p^2 + m^2)
            f_p(p) = p^2 * exp(-E_p(p) / T)
            p_max = 10.0
            f_max = f_p(sqrt(2 * m * T))
            while length(sampled) < N
                p_try = rand() * p_max
                y_try = rand() * f_max
                if y_try < f_p(p_try)
                    push!(sampled, p_try)
                end
            end
            momenta_by_r[i] = sampled
        end
        return momenta_by_r
    end

    function flatten_phase_space(momenta_at_ri::Vector{Vector{Float64}}, r_grid::Vector{Float64})
        positions = Float64[]
        momenta = Float64[]
        for (i, r) in enumerate(r_grid)
            for p in momenta_at_ri[i]
                push!(positions, r)
                push!(momenta, p)
            end
        end
        return positions, momenta
    end

    # === Execute Steps ===
    sampled_positions = rejection_sample_positions(N_particles, r_grid, t)
    N_particles_at_ri = count_particles_in_1D_grid(sampled_positions, r_grid)
    momenta_by_r = sample_momenta_at_each_ri(N_particles_at_ri, r_grid)
    positions, momenta = flatten_phase_space(momenta_by_r, r_grid)
    pos = zeros(dimss, N_particles)
    mom = zeros(dimss, N_particles)
    pos[1,:] .= positions
    mom[1,:] .= momenta
    return pos, mom
end



"""
    sample_initial_particles_at_origin!(...)

Initialize all particles at the origin with thermal Gaussian momentum distribution.

Useful for controlled experiments or comparisons.

# Arguments
Same as `sample_initial_particles_from_pdf!`.

# Returns
- `(positions, momenta)`: All positions set to 0, momenta sampled thermally.
"""
function sample_initial_particles_at_origin!(
    m, dim, N_particles,
    t, T_profile, ur_profile, mu_profile,
    x_range::Tuple{Float64, Float64}, nbins::Int
)
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

"""
    n_rt(r, τ, T_profile, mu_profile; m=1.5)

Compute the local particle density `n(r, τ)` × τ at a radial position `r` and time `τ`,
assuming local equilibrium Boltzmann distribution.

# Arguments
- `r`: Radial coordinate.
- `τ`: Proper time.
- `T_profile`, `mu_profile`: Functions of `(r, τ)`.
- `m`: Mass of the particle in GeV (default 1.5 GeV).

# Returns
- `n(r, τ) * τ`: Radial density multiplied by proper time, integrated over azimuthal angle.
"""
function n_rt(r, τ, T_profile, mu_profile; m=1.5)
    integrand(r) = begin
        T = T_profile(r, τ)
        fug = mu_profile(r, τ)
        z = m / T

        b2 = besselkx(2, z)
        b1 = besselk1x(z)
        b3 = b1 + 4 / z * b2

        ex = exp(fug - z)
        deg = 6  # degrees of freedom: spin × color

        n = deg * (m^2 * T / (2 * π^2)) * ex * b2 * Constants.fmGeV^3  # [fm⁻³]
        return 2π * n  # Azimuthal integral
    end

    return integrand(r) * τ
end

"""
    plot_n_rt_comparison_hydro_langevin(...)

Generate an animated comparison of particle density profiles `n(r, t)` between Langevin simulation and hydrodynamic models.

# Arguments
- `n_rts`: List of 2D arrays (r × t) for hydrodynamic model densities.
- `r_vals`: Radial grid points.
- `pos_hist`: List of particle positions over time (dim × N array).
- `times`: Vector of time points.
- `frames`, `fps`: Controls for animation.
- `filename`: Output file for the GIF.
- `colors`: List of line colors for each model.
- `nbins_x`: Histogram bin count.
- `x_range`: Optional range for histogram x-axis.
- `dims`: Spatial dimension to plot (1 or 2).

# Output
- Saves an animated GIF comparing Langevin and hydro models.
"""
function plot_n_rt_comparison_hydro_langevin(
    n_rts, r_vals, pos_hist, times;
    frames=30, fps=5,
    filename="fx_t_langevin.gif",
    title="Langevin on Hydro-Background",
    colors=["green", "black", "red"],
    nbins_x=200, x_range=nothing,
    dims = 2
)
    var_names = [L"\mathrm{n_{Ideal}(r,t)}", L"\mathrm{n_{MIS}(r,t)}"]
    steps = max(1, length(times) ÷ frames)
    idx = 1:steps:length(times)

    pos_hist = pos_hist[idx]
    times = times[idx]

    # Absolute values to ensure non-negative radii
    pos_hist = [abs.(x) for x in pos_hist]

    # Determine plotting range
    xmin = minimum([minimum(x[dims, :]) for x in pos_hist])
    xmax = maximum([maximum(x[dims, :]) for x in pos_hist])
    x_range = x_range === nothing ? (xmin, xmax) : x_range

    edges = range(x_range[1], x_range[2], length=nbins_x + 1).+ 1e-3
    centers = (edges[1:end-1] .+ edges[2:end]) ./ 2
    dx = step(edges)

    dr = r_vals[2] - r_vals[1]

    # Create animation
    anim = @animate for (x, t) in zip(pos_hist, times)
        h = fit(Histogram, x[dims, :], edges)
        fx = h.weights ./ sum(h.weights) ./ dx  # Normalize

        plot(centers, fx, lw=3, label=L"\mathrm{n_{Langevin}(r,t)}",
            xlabel=L"\mathrm{r [fm]}", ylabel=L"\mathrm{n(r,t) [1/fm^2]}",
            ylims=(0, 0.8), xlims=(0, 10), size=(800, 800),
            legendfontsize=18, guidefontsize=16, tickfontsize=14,
            labelfontsize=18, titlesize=20, title=title)

        # Add hydrodynamic profiles
        for (j, n_rt) in enumerate(n_rts)
            n = n_rt[:, findfirst(==(t), times)]

            n_norm = n ./ (sum(n) * dr)
            plot!(r_vals, n_norm, lw=3, color=colors[j], label=var_names[j])
        end

        # Add time label
        plot!([NaN], [NaN], label=@sprintf("t = %.2f", t), color=:white, legend=:topright)
    end

    gif(anim, filename, fps=fps)
end

end # module Utils
