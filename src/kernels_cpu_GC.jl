module KernelsCPU_GC

# === Exported Symbols ===
export kernel_save_snapshot_cpu!,
       kernel_save_momenta_cpu!,
       kernel_save_positions_cpu!,
       interpolate_2d_cpu,
       kernel_compute_all_forces_general_coords_cpu!,
       kernel_update_positions_general_coords_cpu!,
       kernel_update_momenta_LRF_general_coords_cpu!,
       kernel_boost_to_lab_frame_general_coords_cpu!,
       kernel_boost_to_rest_frame_general_coords_cpu!,
       kernel_save_positions_general_coords_gpu!,
       kernel_set_to_fluid_velocity_cpu!

using LinearAlgebra

# ============================================================================
# Interpolation
# ============================================================================

"""
    interpolate_2d_cpu(x, y, values, xi, yi)

Bilinear interpolation on a 2D grid of `values` defined by coordinate vectors `x`, `y`.

# Arguments
- `x`, `y`: 1D vectors defining the spatial and temporal grid.
- `values`: Matrix of size (length(x), length(y)).
- `xi`, `yi`: Coordinates to interpolate at.

# Returns
- Interpolated value at `(xi, yi)`.
"""
function interpolate_2d_cpu(x, y, values, xi, yi)
    i = searchsortedlast(x, xi)
    j = searchsortedlast(y, yi)

    i = clamp(i, 1, length(x) - 1)
    j = clamp(j, 1, length(y) - 1)

    x0, x1 = x[i], x[i + 1]
    y0, y1 = y[j], y[j + 1]

    v00, v10 = values[i, j], values[i + 1, j]
    v01, v11 = values[i, j + 1], values[i + 1, j + 1]

    xd = (xi - x0) / (x1 - x0)
    yd = (yi - y0) / (y1 - y0)

    c0 = v00 * (1 - xd) + v10 * xd
    c1 = v01 * (1 - xd) + v11 * xd

    return c0 * (1 - yd) + c1 * yd
end

# ============================================================================
# Boost Kernels
# ============================================================================


function kernel_boost_to_rest_frame_general_coords_cpu!(
    momenta, positions, xgrid, tgrid, VelocityEvolution,
    m, N, step, Δt, t0
    )
    for i in 1:N
        r = abs(positions[2, i])

        # Local fluid velocity in r-direction
        v = interpolate_2d_cpu(xgrid, tgrid, VelocityEvolution, r, step * Δt + t0)
        γ = 1.0 / sqrt(1 - v^2 + 1e-10)

        pτ = momenta[1, i]
        pr = momenta[2, i]

        # Lorentz boost into LRF
        momenta[1, i] = γ * (pτ - v * pr)  # p^τ in LRF
        momenta[2, i] = γ * (pr - v * pτ)  # p^r in LRF
    end
end


function kernel_boost_to_lab_frame_general_coords_cpu!(
    momenta, positions, xgrid, tgrid, VelocityEvolution,
    m, N, step, Δt, t0
    )
    for i in 1:N
        r = abs(positions[2, i])

        v = interpolate_2d_cpu(xgrid, tgrid, VelocityEvolution, r, step * Δt + t0)
        γ = 1.0 / sqrt(1 - v^2 + 1e-8)

        pτ = momenta[1, i]
        pr = momenta[2, i]

        # Inverse Lorentz boost back to lab frame
        momenta[1, i] = γ * (pτ + v * pr)
        momenta[2, i] = γ * (pr + v * pτ)
    end
end


function compute_christoffel(position::AbstractVector)
    Γ = zeros(2, 2, 2)  # Γ^μ_{νρ} in (τ, r)
    return Γ
end

function kernel_compute_all_forces_general_coords_cpu!(
    Tfield, xgrid, tgrid,
    momenta, positions, p_mags, p_units,
    ηD_vals, kL_vals, kT_vals,
    ξ, deterministic_terms, stochastic_terms,
    Δt, m, random_directions,
    dimensions, N, step, t0, DsT
    )
    for i in 1:N
        # Compute particle momentum magnitude
        p = sqrt(sum(momenta[2:end, i] .^ 2))
        p_mags[i] = p

        # Compute unit momentum vectors (with fallback if zero momentum)
        for d in 2:dimensions
            p_units[d, i] = p < eps() ? random_directions[d, i] : momenta[d, i] / p
        end

        # Interpolate temperature from space-time field
        T = interpolate_2d_cpu(xgrid, tgrid, Tfield, abs(positions[2, i]), step * Δt + t0)

        # Compute transport coefficients
        # Here, DsT is interpreted as the dimensionless quantity D_s * T.
        # (Einstein relation, non-relativistic approximation):
        #   η_D = T^2 / (m * DsT),   κ = 2 T^3 / DsT
        M = m
        ηD = T^2 / (M * DsT)
        κ  = 2 * T^3 / DsT
        kL, kT = sqrt(κ), sqrt(κ)

        ηD_vals[i], kL_vals[i], kT_vals[i] = ηD, kL, kT

        # Compute forces
        for d in 2:dimensions
            # Christoffel geometric drift term
            Γ = compute_christoffel(positions[:, i])  # Returns Γ^μ_{νρ}
            p0 = sqrt(m^2 + sum(momenta[d, i] .^ 2))  # p^τ ≈ relativistic energy in LRF

            geo_term = 0.0
            for ν in 1:dimensions, ρ in 1:dimensions
                geo_term += -Γ[d, ν, ρ] * momenta[ν, i] * momenta[ρ, i] / p0
            end
            geo_term *= Δt

            # Langevin deterministic + geometric terms
            det_term = -ηD * momenta[d, i] * Δt + geo_term

            # Langevin stochastic term
            sto_term = 0.0
            for j in 2:dimensions
                sto_term += (kL - kT) * p_units[d, i] * p_units[j, i] * ξ[j, i] +
                            kT * (d == j ? 1.0 : 0.0) * ξ[j, i]
            end

            # Store computed forces
            deterministic_terms[d, i] = det_term
            stochastic_terms[d, i] = sto_term
        end
    end
end

function kernel_compute_all_forces_general_coords_cpu!(
    T::Float64,
    momenta, positions, p_mags, p_units,
    ηD_vals, kL_vals, kT_vals,
    ξ, deterministic_terms, stochastic_terms,
    Δt, m, random_directions,
    dimensions, N, step, t0, DsT
    )
    for i in 1:N
        # Compute particle momentum magnitude
        p = sqrt(sum(momenta[:, i] .^ 2))
        p_mags[i] = p

        # Compute unit momentum vectors (with fallback if zero momentum)
        for d in 1:dimensions
            p_units[d, i] = p < eps() ? random_directions[d, i] : momenta[d, i] / p
        end

        # Interpolate temperature from space-time field
        

        # Compute transport coefficients (see docstring above)
        M = m
        ηD = T^2 / (M * DsT)
        κ  = 2 * T^3 / DsT
        kL, kT = sqrt(κ), sqrt(κ)

        ηD_vals[i], kL_vals[i], kT_vals[i] = ηD, kL, kT

        # Compute forces
        for d in 1:dimensions
            # Christoffel geometric drift term
            Γ = compute_christoffel(positions[:, i])  # Returns Γ^μ_{νρ}
            p0 = sqrt(m^2 + sum(momenta[:, i] .^ 2))  # p^τ ≈ relativistic energy in LRF

            geo_term = 0.0
            for ν in 1:dimensions, ρ in 1:dimensions
                geo_term += -Γ[d, ν, ρ] * momenta[ν, i] * momenta[ρ, i] / p0
            end
            geo_term *= Δt

            # Langevin deterministic + geometric terms
            det_term = -ηD * momenta[d, i] * Δt + geo_term

            # Langevin stochastic term
            sto_term = 0.0
            for j in 1:dimensions
                sto_term += (kL - kT) * p_units[d, i] * p_units[j, i] * ξ[j, i] +
                            kT * (d == j ? 1.0 : 0.0) * ξ[j, i]
            end

            # Store computed forces
            deterministic_terms[d, i] = det_term
            stochastic_terms[d, i] = sto_term
        end
    end
end



function kernel_update_momenta_LRF_general_coords_cpu!(
    momenta, deterministic_terms, stochastic_terms,
    Δt, dimensions, N,m
 )
    for i in 1:N
        for d in 2:dimensions
            momenta[d, i] += deterministic_terms[d, i] + sqrt(Δt) * stochastic_terms[d, i]
        end
        # Recalculate p^0 to satisfy mass-shell constraint
        p_spatial_sq = sum(momenta[2:dimensions, i] .^ 2)
        momenta[1, i] = sqrt(m^2 + p_spatial_sq)
    end
end


function kernel_update_positions_general_coords_cpu!(positions, momenta, m, Δt, N)
    for i in 1:N
        # Relativistic energy in flat spacetime
        E = momenta[1, i]
        for μ in 1:size(positions, 1)
            positions[μ, i] += Δt * momenta[μ, i] / E
        end
        #if positions[2, i] < 0
        #    positions[2, i] = -10.
        #    momenta[2, i] = 0.0
        #    p_spatial_sq = sum(momenta[2, i] .^ 2)
        #    momenta[1, i] = sqrt(m^2 + p_spatial_sq)
        #end

    end
end

function kernel_save_snapshot_cpu!(
    history_col, snapshot::Vector{Float64}, N::Int
)
    @inbounds for i in 1:N
        history_col[i] = snapshot[i]
    end
end

function kernel_save_momenta_cpu!(
    momenta_history, current_momentum, save_idx::Int, N::Int
)
    D = size(current_momentum, 1)
    @inbounds for i in 1:N
        @simd for d in 1:D
            momenta_history[d, i, save_idx] = current_momentum[d, i]
        end
    end
    return nothing
end

function kernel_save_positions_cpu!(
    position_history, current_positions, save_idx::Int, N::Int
)
    D = size(current_positions, 1)
    @inbounds for i in 1:N
        @simd for d in 1:D
            position_history[d, i, save_idx] = current_positions[d, i]
        end
    end
    return nothing
end

function kernel_save_positions_general_coords_gpu!(
    position_history, current_positions, save_idx::Int, N::Int
)
    for i in 1:N
        #for d in 1:size(current_positions, 1)
            position_history[1, i, save_idx] = current_positions[2, i]
        #end
    end
end

function kernel_set_to_fluid_velocity_cpu!(
    momenta::Array{Float64,2},
    positions::Array{Float64,2},
    xgrid, tgrid, VelocityEvolution,
    m::Float64,
    N::Int,
    step::Int,
    Δt::Float64,
    t0::Float64
)
    @inbounds for i in 1:N
        # Particle coordinates
        x = positions[1, i]
        y = positions[2, i]

        # Radial position
        r = sqrt(x^2 + y^2)
        if r < eps()       # avoid dividing by zero
            continue
        end

        # Local radial fluid velocity (scalar)
        v = interpolate_2d_cpu(xgrid, tgrid, VelocityEvolution, r, step * Δt + t0)

        # Lorentz factor
        γ = 1.0 / sqrt(1.0 - v^2 + 1e-10)

        # Set momentum along radial direction
        momenta[1, i] = m * γ * v * (x / r)
        momenta[2, i] = m * γ * v * (y / r)
    end
    return nothing
end



end # module KernelsCPU_GC
