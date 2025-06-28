module KernelsCPU

# === Exported Symbols ===
export kernel_boost_to_rest_frame_cpu!,
       kernel_boost_to_lab_frame_cpu!,
       kernel_compute_all_forces_cpu!,
       kernel_update_momenta_LRF_cpu!,
       kernel_update_positions_cpu!,
       kernel_save_snapshot_cpu!,
       kernel_save_positions_cpu!,
       interpolate_2d_cpu

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

"""
    kernel_boost_to_rest_frame_cpu!(...)

Apply Lorentz boost to transform particle momenta to the local rest frame (LRF).
"""
function kernel_boost_to_rest_frame_cpu!(
    momenta, positions, xgrid, tgrid, VelocityEvolution,
    m, N, step, Δt, t0
)
    for i in 1:N
        v = zeros(2)
        v[1] = sign(positions[1, i]) * interpolate_2d_cpu(xgrid, tgrid, VelocityEvolution, abs(positions[1, i]), step * Δt + t0)
        E = sqrt(sum(momenta[:, i] .^ 2) + m^2)
        γ = 1.0 / sqrt(1.0 - sum(v .^ 2) + 1e-8)

        for j in 1:size(momenta, 1)
            βj = -v[j]
            momenta[j, i] = γ * (momenta[j, i] - βj * E)
        end
    end
end

"""
    kernel_boost_to_lab_frame_cpu!(...)

Inverse Lorentz boost to transform momenta from LRF back to the lab frame.
"""
function kernel_boost_to_lab_frame_cpu!(
    momenta, positions, xgrid, tgrid, VelocityEvolution,
    m, N, step, Δt, t0
)
    for i in 1:N
        v = zeros(2)
        v[1] = sign(positions[1, i]) * interpolate_2d_cpu(xgrid, tgrid, VelocityEvolution, abs(positions[1, i]), step * Δt + t0)
        E = sqrt(sum(momenta[:, i] .^ 2) + m^2)
        γ = 1.0 / sqrt(1.0 - sum(v .^ 2) + 1e-8)

        for j in 1:size(momenta, 1)
            βj = v[j]
            momenta[j, i] = γ * (momenta[j, i] - βj * E)
        end
    end
end

# ============================================================================
# Langevin Force Calculation
# ============================================================================

"""
    kernel_compute_all_forces_cpu!(...)

Computes deterministic (drag) and stochastic forces acting on particles in the LRF.
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
        p = sqrt(sum(momenta[:, i] .^ 2))
        p_mags[i] = p

        for d in 1:dimensions
            p_units[d, i] = p < eps() ? random_directions[d, i] : momenta[d, i] / p
        end

        T = interpolate_2d_cpu(xgrid, tgrid, Tfield, abs(positions[1, i]), step * Δt + t0)
        DsT = 0.2 * T
        M = 1.5
        ηD = T^2 / (M * DsT)
        κ = 2 * T^3 / DsT
        kL, kT = sqrt(κ), sqrt(κ)

        ηD_vals[i], kL_vals[i], kT_vals[i] = ηD, kL, kT

        for d in 1:dimensions
            det_term = -ηD * momenta[d, i] * Δt
            sto_term = 0.0
            for j in 1:dimensions
                sto_term += (kL - kT) * p_units[d, i] * p_units[j, i] * ξ[j, i] +
                            kT * (d == j ? 1.0 : 0.0) * ξ[j, i]
            end
            deterministic_terms[d, i] = det_term
            stochastic_terms[d, i] = sto_term
        end
    end
end

# ============================================================================
# Momentum and Position Updates
# ============================================================================

"""
    kernel_update_momenta_LRF_cpu!(...)

Update momenta using Langevin forces in the LRF.
"""
function kernel_update_momenta_LRF_cpu!(
    momenta, deterministic_terms, stochastic_terms,
    Δt, dimensions, N
)
    for i in 1:N
        for d in 1:dimensions
            momenta[d, i] += deterministic_terms[d, i] + sqrt(Δt) * stochastic_terms[d, i]
        end
    end
end

"""
    kernel_update_positions_cpu!(...)

Update positions from momenta assuming free motion.
"""
function kernel_update_positions_cpu!(positions, momenta, m, Δt, N)
    for i in 1:N
        positions[1, i] += Δt * momenta[1, i] / m
    end
end

# ============================================================================
# Save Snapshots
# ============================================================================

"""
    kernel_save_snapshot_cpu!(history_col, snapshot, N)

Save momentum magnitudes for all particles into history buffer.
"""
function kernel_save_snapshot_cpu!(
    history_col::Vector{Float64}, snapshot::Vector{Float64}, N::Int
)
    @inbounds for i in 1:N
        history_col[i] = snapshot[i]
    end
end

"""
    kernel_save_positions_cpu!(position_history, current_positions, save_idx, N)

Save particle positions into history array.
"""
function kernel_save_positions_cpu!(
    position_history, current_positions, save_idx::Int, N::Int
)
    for i in 1:N
        for d in 1:size(current_positions, 1)
            position_history[d, i, save_idx] = current_positions[d, i]
        end
    end
end

end # module KernelsCPU
