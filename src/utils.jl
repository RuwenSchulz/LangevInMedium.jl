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
export sample_initial_particles_at_origin!
export n_rt
export plot_n_rt_comparison_hydro_langevin

"""
    sample_initial_particles_from_pdf!(m, dim, N_particles, t, T_profile, ur_profile, mu_profile, x_range, nbins)

Sample particle positions and momenta from a hydrodynamic Boltzmann distribution.

# Arguments
- `m`: Mass of the particle [GeV].
- `dim`: Number of spatial dimensions.
- `N_particles`: Number of particles to sample.
- `t`: Current simulation time.
- `T_profile`, `ur_profile`, `mu_profile`: Hydrodynamic profiles (functions of `r`, `t`).
- `x_range`: `(min, max)` radial position range.
- `nbins`: Number of bins for discretizing the space.

# Returns
- `(positions, momenta)`: Arrays of shape `(dim, N_particles)`.
"""
function sample_initial_particles_from_pdf!(
    m, dim, N_particles,
    t, T_profile, ur_profile, mu_profile,
    x_range::Tuple{Float64, Float64}, nbins::Int
    )
    positions = zeros(dim, N_particles)
    momenta = zeros(dim, N_particles)

    # Step 1: Discretize x (radial) domain
    x_edges = range(x_range[1], x_range[2], length=nbins + 1)
    dx = step(x_edges)
    x_centers = (x_edges[1:end-1] .+ x_edges[2:end]) ./ 2

    # Step 2: Compute the normalized PDF for radial distribution
    T_vals  = T_profile.(x_centers, t)
    ur_vals = ur_profile.(x_centers, t)
    mu_vals = mu_profile.(x_centers, t)
    γ_vals  = sqrt.(1 .+ ur_vals .^ 2)

    n_boltz = T_vals .^ (3/2) ./ γ_vals .* exp.((mu_vals .- m .* γ_vals) ./ T_vals)
    pdf_vals = n_boltz ./ sum(n_boltz) ./ dx  # normalize

    # Step 3: Build interpolated inverse CDF
    cdf_vals = cumsum(pdf_vals) .* dx
    cdf_vals[end] = 1.0  # fix rounding errors
    inv_cdf = LinearInterpolation(cdf_vals, x_centers, extrapolation_bc=Flat())

    # Step 4: Sample positions and momenta
    for i in 1:N_particles
        r = inv_cdf(rand())                 # Sample r from inverse CDF
        T = T_profile(r, t)
        σ = sqrt(m * T)
        positions[:, i] .= r                # All dims get same r
        momenta[:, i] .= σ .* randn(dim)    # Gaussian thermal momentum
    end

    return positions, momenta
end

function sample_initial_particles_at_origin!(
    m, dim, N_particles,
    t, T_profile, ur_profile, mu_profile,
    x_range::Tuple{Float64, Float64}, nbins::Int
    )
    positions = zeros(dim, N_particles)
    momenta = zeros(dim, N_particles)

    # Use temperature and chemical potential at r = 0
    T = T_profile(0.0, t)
    σ = sqrt(m * T)

    for i in 1:N_particles
        positions[:, i] .= r              # Set all to r = 0
        momenta[:, i] .= σ .* randn(dim)     # Gaussian thermal momentum
    end

    return positions, momenta
end


"""
    n_rt(r, τ, T_profile, mu_profile; m=1.5)

Compute the radial particle density `n(r, τ)` at time `τ` for a given temperature
and chemical potential profile, using an equilibrium Boltzmann distribution.

# Arguments
- `r`: Radial coordinate.
- `τ`: Time (proper time).
- `T_profile`, `mu_profile`: Functions of `(r, τ)`.
- `m`: Mass of the particle [GeV].

# Returns
- The value of `n(r, τ)` times `τ`, integrated over azimuth.
"""
function n_rt(r, τ, T_profile, mu_profile; m=1.5)
    integrand(r) = begin
        T = T_profile(r, τ)
        fug = mu_profile(r, τ)  # not divided by T
        z = m / T

        b2 = besselkx(2, z)
        b1 = besselk1x(z)
        b3 = b1 + 4 / z * b2

        ex = exp(fug - z)
        deg = 6  # degeneracy: spin × color
        n = deg * (m^2 * T / (2 * π^2)) * ex * b2 * Constants.fmGeV^3  # number density in fm⁻³

        return 2π * n
    end

    return integrand(r) * τ
end


"""
    plot_n_rt_comparison_hydro_langevin(n_rts, r_vals, pos_hist, times; ...)

Animate the comparison between Langevin-computed and hydrodynamic reference
density profiles `n(r, t)`.

# Arguments
- `n_rts`: Vector of `n(r,t)` reference arrays (e.g., ideal, MIS).
- `r_vals`: Radial grid points.
- `pos_hist`: Vector of particle position arrays over time.
- `times`: Time points.
- `frames`, `fps`: Animation settings.
- `filename`: Output GIF filename.
- `colors`, `nbins_x`, `x_range`: Plot controls.
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

    # Filter out unphysical particles (r < 0.001)
    #pos_hist = [x[:, x[1, :] .≥ 0.001] for x in pos_hist]

    pos_hist = [abs.(x) for x in pos_hist]
    xmin = minimum([minimum(x[1, :]) for x in pos_hist])
    xmax = maximum([maximum(x[1, :]) for x in pos_hist])
    #println("x range: [$xmin, $xmax]")
    x_range = x_range === nothing ? (xmin, xmax) : x_range

    edges = range(x_range[1], x_range[2], length=nbins_x + 1)
    centers = (edges[1:end-1] .+ edges[2:end]) ./ 2
    dx = step(edges)
    dr = r_vals[2] - r_vals[1]

    anim = @animate for (x, t) in zip(pos_hist, times)
        h = fit(Histogram, x[1, :], edges)
        fx = h.weights ./ sum(h.weights) ./ dx

        plot(centers, fx, lw=3, label=L"\mathrm{n_{Langevin}(r,t)}",
            xlabel=L"\mathrm{r [fm]}", ylabel=L"\mathrm{n(r,t) [1/fm^2]}",
            ylims=(0, 0.8), xlims=(0, 10), size=(800, 800),
            legendfontsize=18, guidefontsize=16, tickfontsize=14,
            labelfontsize=18, titlesize=20, title=title)

        for (j, n_rt) in enumerate(n_rts)
            n = n_rt[:, findfirst(==(t), times)]
            n_norm = n ./ (sum(n) * dr)
            plot!(r_vals, n_norm, lw=3, color=colors[j], label=var_names[j])
        end

        plot!([NaN], [NaN], label=@sprintf("t = %.2f", t), color=:white, legend=:topright)
    end

    gif(anim, filename, fps=fps)
end

end # module
