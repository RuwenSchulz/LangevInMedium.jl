module KernelsGPU

using CUDA
using StaticArrays

# Exported functions used externally
export 
       kernel_boost_to_rest_frame_gpu!,
       kernel_boost_to_lab_frame_gpu!,
       kernel_compute_all_forces_gpu!,
       kernel_update_momenta_LRF_gpu!,
       kernel_update_positions_gpu!,
       kernel_save_snapshot_gpu!,
       kernel_save_positions_gpu!,
       interpolate_2d_cuda

# ----------------------------------------------------------------------
# GPU-compatible bilinear interpolation over a 2D grid
# ----------------------------------------------------------------------

"""
    interpolate_2d_cuda(x::Vector, y::Vector, values::Matrix, xi, yi)

Performs bilinear interpolation of `values` at point `(xi, yi)` based on 1D grids `x`, `y`.
Assumes `values[i, j]` corresponds to `(x[i], y[j])`.

# Returns
Interpolated value at `(xi, yi)`.
"""
@inline function interpolate_2d_cuda(x, y, values, xi, yi)
    # Replace while loops with simple for-loops with fixed upper bounds
    i = 1
    for k in 1:length(x)-1
        if x[k+1] <= xi
            i += 1
        else
            break
        end
    end

    j = 1
    for k in 1:length(y)-1
        if y[k+1] <= yi
            j += 1
        else
            break
        end
    end

    # Clamp i, j to valid ranges manually (no dynamic checks)
    i = ifelse(i < 1, 1, ifelse(i >= length(x), length(x)-1, i))
    j = ifelse(j < 1, 1, ifelse(j >= length(y), length(y)-1, j))

    # Manually inline access
    x0 = x[i]
    x1 = x[i+1]
    y0 = y[j]
    y1 = y[j+1]

    v00 = values[i, j]
    v10 = values[i+1, j]
    v01 = values[i, j+1]
    v11 = values[i+1, j+1]

    xd = (xi - x0) / (x1 - x0 + 1e-8)
    yd = (yi - y0) / (y1 - y0 + 1e-8)

    c0 = v00 * (1 - xd) + v10 * xd
    c1 = v01 * (1 - xd) + v11 * xd

    return c0 * (1 - yd) + c1 * yd
end


# ----------------------------------------------------------------------
# CUDA Kernels for Langevin Dynamics
# ----------------------------------------------------------------------

"""
    kernel_boost_to_rest_frame_gpu!

Applies Lorentz boost from lab frame to local rest frame (LRF) for all particles.
"""
@inline function kernel_boost_to_rest_frame_gpu!(momenta, positions, xgrid, tgrid, VelocityEvolution, m::Float64, N_particles::Int, steps, Δt, initial_time)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= N_particles
        v = MVector{2, Float64}(0.0, 0.0)
        v[1] =  interpolate_2d_cuda(xgrid, tgrid, VelocityEvolution, abs(positions[1, i]), steps * Δt + initial_time)
#sign(positions[1, i]) *
        p_norm = 0.0
        for j in 1:size(momenta, 1)
            p_norm += momenta[j, i]^2
        end
        E = sqrt(p_norm + m^2)

        v2 = sum(v .^ 2)
        γ = 1.0 / sqrt(1.0 - v2 + 1e-8)

        for j in 1:size(momenta, 1)
            β_j = -v[j]
            p_j = momenta[j, i]
            @inbounds momenta[j, i] = γ * (p_j - β_j * E)
        end
    end
    return
end

"""
    kernel_boost_to_lab_frame_gpu!

Inverse Lorentz boost: transforms momenta from local rest frame (LRF) back to the lab frame.
"""
@inline function kernel_boost_to_lab_frame_gpu!(momenta, positions, xgrid, tgrid, VelocityEvolution, m::Float64, N_particles::Int, steps, Δt, initial_time)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= N_particles
        v = MVector{2, Float64}(0.0, 0.0)
        v[1] =  interpolate_2d_cuda(xgrid, tgrid, VelocityEvolution, abs(positions[1, i]), steps * Δt + initial_time)
#sign(positions[1, i]) *
        p_norm = 0.0
        for d in 1:size(momenta, 1)
            @inbounds p_norm += momenta[d, i]^2
        end

        E = sqrt(p_norm + m^2)
        v2 = 0.0
        for j in 1:2
            v2 += v[j]^2
        end
        γ = 1.0 / sqrt(1.0 - v2 + 1e-8)

        for d in 1:size(momenta, 1)
            β_j = v[d]
            p_j = momenta[d, i]
            @inbounds momenta[d, i] = γ * (p_j - β_j * E)
        end
    end
    return
end

"""
    kernel_compute_all_forces_gpu!

Computes drag and stochastic forces for Langevin evolution in LRF.
Stores deterministic and stochastic force components per particle.
"""
@inline function kernel_compute_all_forces_gpu!(
    TemperatureEvolution, xgrid, tgrid,
    momenta, positions, p_mags, p_units,
    ηD_vals, kL_vals, kT_vals,
    ξ, deterministic_terms, stochastic_terms,
    Δt, m, random_directions,
    dimensions, N_particles, steps, initial_time
)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= N_particles
        # === Compute momentum magnitude (p) and unit vector ===
        p_sq = 0.0
        for d in 1:dimensions
            @inbounds p_sq += momenta[d, i]^2
        end
        p = sqrt(p_sq)
        @inbounds p_mags[i] = p

        for d in 1:dimensions
            @inbounds p_units[d, i] = p < eps(Float64) ? random_directions[d, i] : momenta[d, i] / p
        end

        # === Interpolate temperature ===
        T = interpolate_2d_cuda(xgrid, tgrid, TemperatureEvolution, abs(positions[1, i]), steps * Δt + initial_time)

        # === Compute transport coefficients ===
        DsT = 0.2 * T
        M = 1.5
        ηD = T^2 / (M * DsT)
        κ = 2 * T^3 / DsT
        kL = sqrt(κ)
        kT = sqrt(κ)

        @inbounds ηD_vals[i] = ηD
        @inbounds kL_vals[i] = kL
        @inbounds kT_vals[i] = kT

        # === Langevin force: deterministic + stochastic ===
        for d in 1:dimensions
            det_term = -ηD * momenta[d, i] * Δt
            sto_term = 0.0
            for j in 1:dimensions
                p_ip = p_units[d, i]
                p_jp = p_units[j, i]
                ξj = ξ[j, i]
                sto_term += (kL - kT) * p_ip * p_jp * ξj + kT * (d == j ? 1.0 : 0.0) * ξj
            end
            @inbounds deterministic_terms[d, i] = det_term
            @inbounds stochastic_terms[d, i] = sto_term
        end
    end
    return
end

"""
    kernel_update_momenta_LRF_gpu!

Updates the momenta of each particle using Langevin forces in the local rest frame.
"""
@inline function kernel_update_momenta_LRF_gpu!(momenta, deterministic_terms, stochastic_terms, Δt, dimensions, N_particles)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= N_particles
        for d in 1:dimensions
            Δp = @inbounds deterministic_terms[d, i] + sqrt(Δt) * stochastic_terms[d, i]
            @inbounds momenta[d, i] += Δp
        end
    end
    return
end

"""
    kernel_update_positions_gpu!

Moves particles forward in space according to their current momenta.
"""
@inline function kernel_update_positions_gpu!(positions::CuDeviceMatrix{Float64}, momenta::CuDeviceMatrix{Float64}, m::Float64, Δt::Float64, N_particles::Int)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if idx <= N_particles
        @inbounds positions[1, idx] += Δt * momenta[1, idx] / m

        if positions[1, idx] < 0
                        positions[1, idx] = -positions[1, idx]
                        momenta[1, idx] = -momenta[1, idx]
        end

    end
    return
end

"""
    kernel_save_snapshot_gpu!

Stores momenta magnitude snapshot at a given time index.
"""
@inline function kernel_save_snapshot_gpu!(history, snapshot, idx, N_particles)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= N_particles
        @inbounds history[i + (idx - 1) * N_particles] = snapshot[i]
    end
    return
end

"""
    kernel_save_positions_gpu!

Stores particle positions at a given save index.
"""
@inline function kernel_save_positions_gpu!(position_history, current_positions, save_idx::Int, N::Int)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= N
        for d in 1:size(current_positions, 1)
            @inbounds position_history[d, i, save_idx] = current_positions[d, i]
        end
    end
    return
end

end # module Kernels
