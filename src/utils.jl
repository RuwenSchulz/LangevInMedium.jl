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
using QuadGK

using ..Constants

# === Exports ===
export sample_initial_particles_from_pdf!
export sample_initial_particles_milne!
export sample_initial_particles_at_origin!
export sample_initial_particles_at_origin_no_position!
export compute_MIS_distribution
export sample_particles_from_density


function sample_particles_from_density(r_values, n_rt, N_samples::Int, T_interp; 
                                       n_cdf_points=1000, rmax=10.0, t0=0.0)
    # Create interpolation for density
    interp = LinearInterpolation(r_values, n_rt, extrapolation_bc=0.0)
    
    # Compute normalization constant
    norm, _ = quadgk(r -> r * interp(r), 0, rmax)
    
    # Pre-compute CDF for inverse transform sampling
    r_cdf = range(0, rmax, length=n_cdf_points)
    cdf_values = zeros(n_cdf_points)
    
    for i in 2:n_cdf_points
        result, _ = quadgk(r′ -> r′ * interp(r′) / norm, 0, r_cdf[i])
        cdf_values[i] = result
    end
    
    # Ensure CDF is strictly monotonic
    for i in 2:n_cdf_points
        if cdf_values[i] <= cdf_values[i-1]
            cdf_values[i] = cdf_values[i-1] + 1e-9
        end
    end
    cdf_values[end] = 1.0
    
    # Create inverse CDF interpolation
    inverse_cdf = LinearInterpolation(cdf_values, collect(r_cdf), extrapolation_bc=Line())
    
    # Initialize matrices [dim, N_samples]
    x_matrix = zeros(2, N_samples)  # [x, y]
    p_matrix = zeros(2, N_samples)  # [px, py]
    
    for i in 1:N_samples
        # Sample position
        u = rand()
        r = inverse_cdf(u)
        φ = 2π * rand()
        x_matrix[1, i] = r * cos(φ)  # x
        x_matrix[2, i] = r * sin(φ)  # y
        
        # Evaluate temperature at this position
        T_local = T_interp(r, t0)
        
        # Sample momentum from 2D Maxwell-Boltzmann distribution
        p_matrix[1, i] = sqrt(T_local) * randn()  # px
        p_matrix[2, i] = sqrt(T_local) * randn()  # py
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

end # module Utils
