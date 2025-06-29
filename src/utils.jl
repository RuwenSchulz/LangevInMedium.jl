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
export sample_initial_particles_at_origin!
export n_rt
export plot_n_rt_comparison_hydro_langevin

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
    pdf_vals = n_boltz ./ sum(n_boltz) ./ dx  # Normalize to form a probability distribution

    # Construct inverse CDF for sampling
    cdf_vals = cumsum(pdf_vals) .* dx
    cdf_vals[end] = 1.0  # Ensure normalization
    inv_cdf = LinearInterpolation(cdf_vals, x_centers, extrapolation_bc=Flat())

    # Sample particle positions and thermal momenta
    for i in 1:N_particles
        r = inv_cdf(rand())                 # Sample radius from inverse CDF
        T = T_profile(r, t)
        σ = sqrt(m * T)
        positions[:, i] .= r                # Uniform radial position across all spatial dimensions
        momenta[:, i] .= σ .* randn(dim)    # Gaussian-distributed thermal momentum
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
    τ::Float64, T_profile, ur_profile, mu_profile, r_grid::Vector{Float64}
    )
    r_grid = r_grid[1:end].+1e-10
    @assert dim == 2 "Milne sampling requires dim = 2 (τ, r)"
    dr = r_grid[2] - r_grid[1]  # uniform spacing assumed

    positions = zeros(dim, N_particles)
    momenta   = zeros(dim, N_particles)

    # Evaluate hydro profiles at bin centers
    T_vals  = T_profile.(r_grid, τ)
    ur_vals = ur_profile.(r_grid, τ)
    mu_vals = mu_profile.(r_grid, τ)

    γ_vals = sqrt.(1 .+ ur_vals .^ 2)

    # Compute PDF ∝ local equilibrium density
    n_boltz = T_vals .^ (3/2) ./ γ_vals .* exp.((mu_vals .- m .* γ_vals) ./ T_vals)
    pdf_vals = n_boltz ./ sum(n_boltz) ./ dr  # Normalize

    # Build CDF
    cdf_vals = cumsum(pdf_vals) .* dr
    cdf_vals ./= cdf_vals[end]  # enforce final CDF = 1.0

    # Ensure CDF is strictly increasing for interpolation
    Δcdf = diff(cdf_vals)
    valid = findall(Δcdf .> 0.0)
    valid = [valid; lastindex(cdf_vals)]  # include last point

    cdf_clean = cdf_vals[valid]
    r_clean = r_grid[valid]

    inv_cdf = LinearInterpolation(cdf_clean, r_clean, extrapolation_bc=Flat())

    # Sampling loop
    for i in 1:N_particles
        r = inv_cdf(rand())
        #println(r)
        T = T_profile(r, τ)

        pr = sqrt(m * T) * randn()
        pτ = sqrt(m^2 + pr^2)

        positions[:, i] .= (τ, r)
        momenta[:, i]   .= (pτ, pr)
    end

    return positions, momenta
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

# Output
- Saves an animated GIF comparing Langevin and hydro models.
"""
function plot_n_rt_comparison_hydro_langevin(
    n_rts, r_vals, pos_hist, times;
    frames=30, fps=5,
    filename="fx_t_langevin.gif",
    title="Langevin on Hydro-Background",
    colors=["green", "black", "red"],
    nbins_x=200, x_range=nothing
)
    var_names = [L"\mathrm{n_{Ideal}(r,t)}", L"\mathrm{n_{MIS}(r,t)}"]
    steps = max(1, length(times) ÷ frames)
    idx = 1:steps:length(times)

    pos_hist = pos_hist[idx]
    times = times[idx]

    # Absolute values to ensure non-negative radii
    pos_hist = [abs.(x) for x in pos_hist]

    # Determine plotting range
    xmin = minimum([minimum(x[2, :]) for x in pos_hist])
    xmax = maximum([maximum(x[2, :]) for x in pos_hist])
    x_range = x_range === nothing ? (xmin, xmax) : x_range

    edges = range(x_range[1], x_range[2], length=nbins_x + 1).+ 1e-3
    centers = (edges[1:end-1] .+ edges[2:end]) ./ 2
    dx = step(edges)

    dr = r_vals[2] - r_vals[1]

    # Create animation
    anim = @animate for (x, t) in zip(pos_hist, times)
        h = fit(Histogram, x[2, :], edges)
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
