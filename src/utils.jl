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
export sample_phase_space2
export compute_MIS_distribution
export sample_phase_space3
export sample_phase_space4
export sample_heavy_quarks


function sample_heavy_quarks(
    σ_r::Vector{Float64},          # τ·n(r,τ) = 2D density in x-y plane [fm^-2]
    N_particles::Int,
    r_grid::Vector{Float64},
    τ::Float64,
    m::Float64,
    T_profile,
    dimss::Int;
    refine_small_r::Bool=true)     # NEW: option to refine grid
    
    @assert length(σ_r) == length(r_grid) "weights and grid must match"
    @assert length(r_grid) ≥ 2
    


    function sample_positions_cdf_stable(transverse_density::AbstractVector{<:Real},
                                         r_grid::AbstractVector, N::Int)
        n_bins = length(r_grid) - 1
        dr = diff(r_grid)
        
        # Compute weights for each BIN
        weights = zeros(Float64, n_bins)
        
        for i in 1:n_bins
            r_lo = r_grid[i]
            r_hi = r_grid[i+1]
            σ_lo = transverse_density[i]
            σ_hi = transverse_density[i+1]
            
            r_mid = 0.5 * (r_lo + r_hi)
            σ_mid = 0.5 * (σ_lo + σ_hi)
            
            weights[i] = r_mid * σ_mid * dr[i]
        end
        
        weights = max.(weights, 0.0)
        weights[.!isfinite.(weights)] .= 0.0
        
        total = sum(weights)
        if total ≤ 0
            return fill(r_grid[1], N)
        end
        
        cdf = cumsum(weights) ./ total
        
        samples = rand(N)
        bin_idx = searchsortedfirst.(Ref(cdf), samples)
        bin_idx = clamp.(bin_idx, 1, n_bins)
        
        rs = zeros(Float64, N)
        for k in 1:N
            i = bin_idx[k]
            r_lo = r_grid[i]
            r_hi = r_grid[i+1]
            rs[k] = r_lo + rand() * (r_hi - r_lo)
        end
        
        return rs
    end

    function sample_pr_MB_at_positions(rs::Vector{Float64})
        N = length(rs)
        pr = Vector{Float64}(undef, N)
        @inbounds for i in 1:N
            Tloc = T_profile(rs[i], τ)
            if Tloc <= 0
                pr[i] = 0.0
            else
                σ = sqrt(m * Tloc)
                pr[i] = σ * randn()
            end
        end
        return pr
    end

    rs = sample_positions_cdf_stable(_σ_r, _r_grid, N_particles)
    prs = sample_pr_MB_at_positions(rs)

    pos = zeros(dimss, N_particles)
    mom = zeros(dimss, N_particles)
    pos[1, :] .= rs
    return pos, mom
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
