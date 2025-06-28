module KernelsCPU

export kernel_boost_to_rest_frame_cpu!,
       kernel_boost_to_lab_frame_cpu!,
       kernel_compute_all_forces_cpu!,
       kernel_update_momenta_LRF_cpu!,
       kernel_update_positions_cpu!,
       kernel_save_snapshot_cpu!,
       kernel_save_positions_cpu!,
       interpolate_2d_cpu

using LinearAlgebra

"""
    interpolate_2d_cpu(x, y, values, xi, yi)

Performs bilinear interpolation on a 2D grid of values.

# Arguments
- `x`, `y`: Vectors defining the grid.
- `values`: 2D array of values on the grid.
- `xi`, `yi`: Target coordinates to interpolate at.

# Returns
- Interpolated value at `(xi, yi)`.
"""
function interpolate_2d_cpu(x, y, values, xi, yi)
    i = searchsortedlast(x, xi)
    j = searchsortedlast(y, yi)

    # Clamp indices to valid interpolation range
    i = min(max(i, 1), length(x) - 1)
    j = min(max(j, 1), length(y) - 1)

    # Get grid bounds
    x0, x1 = x[i], x[i+1]
    y0, y1 = y[j], y[j+1]

    # Get surrounding values
    v00 = values[i, j]
    v10 = values[i+1, j]
    v01 = values[i, j+1]
    v11 = values[i+1, j+1]

    # Interpolate along x and y
    xd = (xi - x0) / (x1 - x0)
    yd = (yi - y0) / (y1 - y0)

    c0 = v00 * (1 - xd) + v10 * xd
    c1 = v01 * (1 - xd) + v11 * xd

    return c0 * (1 - yd) + c1 * yd
end

"""
    kernel_boost_to_rest_frame_cpu!(...)

Applies a Lorentz boost to the rest frame using interpolated velocity.

This modifies the particle momenta in-place.

# Arguments
- `momenta`: Momentum array (2×N).
- `positions`: Position array.
- `xgrid`, `tgrid`, `VelocityEvolution`: Grid data for velocity field.
- `m`: Particle mass.
- `N`: Number of particles.
- `step`, `Δt`, `t0`: Time evolution parameters.
"""
function kernel_boost_to_rest_frame_cpu!(momenta, positions, xgrid, tgrid, VelocityEvolution, m, N, step, Δt, t0)
    for i in 1:N
        # Get interpolated local velocity
        v = zeros(2)
        v[1] = sign(positions[1, i]) * interpolate_2d_cpu(xgrid, tgrid, VelocityEvolution, abs(positions[1, i]), step * Δt + t0)

        # Compute energy
        E = sqrt(sum(momenta[:, i].^2) + m^2)
        v2 = sum(v .^ 2)
        γ = 1.0 / sqrt(1.0 - v2 + 1e-8)

        # Apply boost
        for j in 1:size(momenta, 1)
            βj = -v[j]
            p_j = momenta[j, i]
            momenta[j, i] = γ * (p_j - βj * E)
        end
    end
end

"""
    kernel_boost_to_lab_frame_cpu!(momenta, positions, xgrid, tgrid, VelocityEvolution, m, N, step, Δt, t0)

Applies a Lorentz boost to transform momenta from the local rest frame (LRF) back to the lab frame,
using position- and time-dependent velocity fields. This modifies the momenta in-place.

# Arguments
- `momenta`: 2×N array of particle momenta.
- `positions`: 2×N array of particle positions.
- `xgrid`, `tgrid`: Grid vectors for spatial and temporal interpolation.
- `VelocityEvolution`: 2D array of flow velocities over space and time.
- `m`: Particle mass.
- `N`: Number of particles.
- `step`, `Δt`, `t0`: Time evolution parameters.
"""
function kernel_boost_to_lab_frame_cpu!(momenta, positions, xgrid, tgrid, VelocityEvolution, m, N, step, Δt, t0)
    for i in 1:N
        v = zeros(2)
        v[1] = sign(positions[1, i]) * interpolate_2d_cpu(xgrid, tgrid, VelocityEvolution, abs(positions[1, i]), step * Δt + t0)

        E = sqrt(sum(momenta[:, i].^2) + m^2)
        γ = 1.0 / sqrt(1.0 - sum(v .^ 2) + 1e-8)

        for d in 1:size(momenta, 1)
            βj = v[d]
            p_j = momenta[d, i]
            momenta[d, i] = γ * (p_j - βj * E)
        end
    end
end


"""
    kernel_compute_all_forces_cpu!(...)

Computes drag and stochastic forces for all particles.

# Arguments
- `Tfield`: Temperature field.
- `xgrid`, `tgrid`: Grid vectors.
- `momenta`, `positions`: Particle momenta and positions.
- `p_mags`, `p_units`: Output arrays for momentum magnitudes and unit vectors.
- `ηD_vals`, `kL_vals`, `kT_vals`: Drag and diffusion coefficients (outputs).
- `ξ`: Random noise terms.
- `deterministic_terms`, `stochastic_terms`: Output force components.
- `Δt`, `m`, `random_directions`: Miscellaneous simulation data.
- `dimensions`, `N`, `step`, `t0`: Simulation size and timing info.
"""
function kernel_compute_all_forces_cpu!(
    Tfield, xgrid, tgrid,
    momenta, positions, p_mags, p_units,
    ηD_vals, kL_vals, kT_vals,
    ξ, deterministic_terms, stochastic_terms,
    Δt, m, random_directions,
    dimensions, N, step, t0
)
    for i in 1:N
        p = sqrt(sum(momenta[:, i].^2))
        p_mags[i] = p

        # Normalize or assign fallback direction
        for d in 1:dimensions
            p_units[d, i] = p < eps() ? random_directions[d, i] : momenta[d, i] / p
        end

        # Interpolate temperature at current position/time
        T = interpolate_2d_cpu(xgrid, tgrid, Tfield, abs(positions[1, i]), step * Δt + t0)

        # Compute transport coefficients
        DsT = 0.2 * T
        M = 1.5
        ηD = T^2 / (M * DsT)
        κ = 2 * T^3 / DsT
        kL = sqrt(κ)
        kT = sqrt(κ)

        ηD_vals[i] = ηD
        kL_vals[i] = kL
        kT_vals[i] = kT

        for d in 1:dimensions
            det_term = -ηD * momenta[d, i] * Δt
            sto_term = 0.0
            for j in 1:dimensions
                p_ip = p_units[d, i]
                p_jp = p_units[j, i]
                sto_term += (kL - kT) * p_ip * p_jp * ξ[j, i] + kT * (d == j ? 1.0 : 0.0) * ξ[j, i]
            end
            deterministic_terms[d, i] = det_term
            stochastic_terms[d, i] = sto_term
        end
    end
end

"""
    kernel_update_momenta_LRF_cpu!(...)

Updates momenta in the local rest frame using computed forces.

# Arguments
- `momenta`: Momentum array.
- `deterministic_terms`, `stochastic_terms`: Force terms.
- `Δt`: Time step.
- `dimensions`, `N`: Number of dimensions and particles.
"""
function kernel_update_momenta_LRF_cpu!(momenta, deterministic_terms, stochastic_terms, Δt, dimensions, N)
    for i in 1:N
        for d in 1:dimensions
            Δp = deterministic_terms[d, i] + sqrt(Δt) * stochastic_terms[d, i]
            momenta[d, i] += Δp
        end
    end
end

"""
    kernel_update_positions_cpu!(...)

Advances particle positions using momentum and mass.

# Arguments
- `positions`, `momenta`: Particle state.
- `m`: Particle mass.
- `Δt`: Time step.
- `N`: Number of particles.
"""
function kernel_update_positions_cpu!(positions, momenta, m, Δt, N)
    for i in 1:N
        positions[1, i] += Δt * momenta[1, i] / m
    end
end

"""
    kernel_save_snapshot_cpu!(history_col, snapshot, N)

Copies the current snapshot into a history column.

# Arguments
- `history_col`: Destination array.
- `snapshot`: Source data.
- `N`: Number of particles.
"""
function kernel_save_snapshot_cpu!(history_col::Vector{Float64}, snapshot::Vector{Float64}, N::Int)
    @inbounds for i in 1:N
        history_col[i] = snapshot[i]
    end
end

"""
    kernel_save_positions_cpu!(position_history, current_positions, save_idx, N)

Stores the current positions in a 3D history array.

# Arguments
- `position_history`: 3D history array.
- `current_positions`: Current 2D positions.
- `save_idx`: Time index.
- `N`: Number of particles.
"""
function kernel_save_positions_cpu!(position_history, current_positions, save_idx, N)
    for i in 1:N
        for d in 1:size(current_positions, 1)
            position_history[d, i, save_idx] = current_positions[d, i]
        end
    end
end

end # module KernelsCPU
