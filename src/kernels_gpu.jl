module KernelsGPU

using CUDA
using StaticArrays

# === Exported Symbols ===
export 
    kernel_boost_to_rest_frame_gpu!,
    kernel_boost_to_lab_frame_gpu!,
    kernel_compute_all_forces_gpu!,
    kernel_update_momenta_LRF_gpu!,
    kernel_update_positions_gpu!,

    kernel_boost_to_rest_frame_general_coords_gpu!,
    kernel_boost_to_lab_frame_general_coords_gpu!,
    kernel_compute_all_forces_general_coords_gpu!,
    kernel_update_momenta_LRF_general_coords_gpu!,
    kernel_update_positions_general_coords_gpu!,
    kernel_save_positions_general_coords_gpu!,


    kernel_save_snapshot_gpu!,
    kernel_save_positions_gpu!,
    interpolate_2d_cuda

# ============================================================================
# GPU Utility Function: Bilinear Interpolation
# ============================================================================

"""
    interpolate_2d_cuda(x::Vector, y::Vector, values::Matrix, xi, yi)

Bilinear interpolation over a 2D array of `values` defined on grid `(x, y)`.  
Assumes `values[i, j]` corresponds to `(x[i], y[j])`.

# Arguments
- `x`, `y`: 1D grid vectors.
- `values`: 2D matrix over the `(x, y)` grid.
- `xi`, `yi`: Target coordinates.

# Returns
- Interpolated value at `(xi, yi)`.
"""
@inline function interpolate_2d_cuda(x, y, values, xi, yi)
    i = 1
    for k in 1:length(x) - 1
        if x[k + 1] <= xi
            i += 1
        else
            break
        end
    end

    j = 1
    for k in 1:length(y) - 1
        if y[k + 1] <= yi
            j += 1
        else
            break
        end
    end

    i = clamp(i, 1, length(x) - 1)
    j = clamp(j, 1, length(y) - 1)

    x0, x1 = x[i], x[i+1]
    y0, y1 = y[j], y[j+1]

    v00, v10 = values[i, j], values[i+1, j]
    v01, v11 = values[i, j+1], values[i+1, j+1]

    xd = (xi - x0) / (x1 - x0 + 1e-8)
    yd = (yi - y0) / (y1 - y0 + 1e-8)

    c0 = v00 * (1 - xd) + v10 * xd
    c1 = v01 * (1 - xd) + v11 * xd

    return c0 * (1 - yd) + c1 * yd
end

# ============================================================================
# CUDA Kernels for Langevin Evolution
# ============================================================================

"""
    kernel_boost_to_rest_frame_gpu!

Lorentz boost momenta from lab frame to local rest frame (LRF) using interpolated background velocity.
"""
@inline function kernel_boost_to_rest_frame_gpu!(
    momenta, positions, xgrid, tgrid, VelocityEvolution,
    m::Float64, N_particles::Int, steps, Δt, initial_time
)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= N_particles
        v = interpolate_2d_cuda(xgrid, tgrid, VelocityEvolution, abs(positions[1, i]), steps * Δt + initial_time)

        p_norm = 0.0
        for j in 1:size(momenta, 1)
            p_norm += momenta[j, i]^2
        end
        E = sqrt(p_norm + m^2)
        γ = 1.0 / sqrt(1.0 - sum(v .^ 2) + 1e-8)

        for j in 1:size(momenta, 1)
            @inbounds momenta[j, i] = γ * (momenta[j, i] - v * E)
        end
    end
    return 
end

@inline function kernel_boost_to_rest_frame_general_coords_gpu!(
    momenta, positions, xgrid, tgrid, VelocityEvolution,
    m::Float64, N_particles::Int, steps, Δt, initial_time
)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= N_particles
        r = abs(positions[2, i])
        v = interpolate_2d_cuda(xgrid, tgrid, VelocityEvolution, r, steps * Δt + initial_time)

        γ = 1.0 / sqrt(1 - v^2 + 1e-8)

        pτ = momenta[1, i]
        pr = momenta[2, i]

        # Lorentz boost into LRF
        momenta[1, i] = γ * (pτ - v * pr)  # p^τ in LRF
        momenta[2, i] = γ * (pr - v * pτ)  # p^r in LRF
    end
    return 
end

"""
    kernel_boost_to_lab_frame_gpu!

Applies inverse Lorentz boost to momenta, restoring them from LRF to lab frame.
"""
@inline function kernel_boost_to_lab_frame_gpu!(
    momenta, positions, xgrid, tgrid, VelocityEvolution,
    m::Float64, N_particles::Int, steps, Δt, initial_time
)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= N_particles

        v = interpolate_2d_cuda(xgrid, tgrid, VelocityEvolution, abs(positions[1, i]), steps * Δt + initial_time)

        p_norm = 0.0
        for d in 1:size(momenta, 1)
            @inbounds p_norm += momenta[d, i]^2
        end

        E = sqrt(p_norm + m^2)
        γ = 1.0 / sqrt(1.0 - sum(v .^ 2) + 1e-8)

        for d in 1:size(momenta, 1)
            @inbounds momenta[d, i] = γ * (momenta[d, i] + v * E)
        end
    end
    return
end

@inline function kernel_boost_to_lab_frame_general_coords_gpu!(
    momenta, positions, xgrid, tgrid, VelocityEvolution,
    m::Float64, N_particles::Int, steps, Δt, initial_time
)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= N_particles
        r = abs(positions[2, i])
        v = interpolate_2d_cuda(xgrid, tgrid, VelocityEvolution, r, steps * Δt + initial_time)

        γ = 1.0 / sqrt(1 - v^2 + 1e-8)

        pτ = momenta[1, i]
        pr = momenta[2, i]

        # Inverse Lorentz boost back to lab frame
        momenta[1, i] = γ * (pτ + v * pr)
        momenta[2, i] = γ * (pr + v * pτ)
    end
    return
end

"""
    kernel_compute_all_forces_gpu!

Compute Langevin drag and stochastic forces in the local rest frame.
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
        # Compute magnitude and unit direction of momentum
        p_sq = 0.0
        for d in 1:dimensions
            p_sq += momenta[d, i]^2
        end
        p = sqrt(p_sq)
        p_mags[i] = p

        for d in 1:dimensions
            @inbounds p_units[d, i] = p < eps() ? random_directions[d, i] : momenta[d, i] / p
        end

        # Interpolate background temperature
        T = interpolate_2d_cuda(xgrid, tgrid, TemperatureEvolution, abs(positions[1, i]), steps * Δt + initial_time)

        # Transport coefficients
        DsT = 0.2 * T
        M = 1.5
        ηD = T^2 / (M * DsT)
        κ  = 2 * T^3 / DsT
        kL = sqrt(κ)
        kT = sqrt(κ)

        @inbounds ηD_vals[i] = ηD
        @inbounds kL_vals[i] = kL
        @inbounds kT_vals[i] = kT

        for d in 1:dimensions
            det_term = -ηD * momenta[d, i] * Δt
            sto_term = 0.0
            for j in 1:dimensions
                sto_term += (kL - kT) * p_units[d, i] * p_units[j, i] * ξ[j, i] +
                            kT * (d == j ? 1.0 : 0.0) * ξ[j, i]
            end
            @inbounds deterministic_terms[d, i] = det_term
            @inbounds stochastic_terms[d, i] = sto_term
        end
    end
    return
end


@inline function kernel_compute_all_forces_general_coords_gpu!(
    TemperatureEvolution, xgrid, tgrid,
    momenta, positions, p_mags, p_units,
    ηD_vals, kL_vals, kT_vals,
    ξ, deterministic_terms, stochastic_terms,
    Δt, m, random_directions,
    dimensions, N_particles, steps, initial_time
    )
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= N_particles
        # Compute magnitude and unit direction of momentum
        p_sq = 0.0
        for d in 2:dimensions
            p_sq += momenta[d, i]^2
        end
        p = sqrt(p_sq)

        for d in 2:dimensions
            @inbounds p_units[d, i] = p < eps() ? random_directions[d, i] : momenta[d, i] / p
        end

        # Interpolate background temperature
        T = interpolate_2d_cuda(xgrid, tgrid, TemperatureEvolution, abs(positions[2, i]), steps * Δt + initial_time)

        # Transport coefficients
        DsT = 0.2 * T
        M = 1.5
        ηD = T^2 / (M * DsT)
        κ  = 2 * T^3 / DsT
        kL = sqrt(κ)
        kT = sqrt(κ)

        @inbounds ηD_vals[i] = ηD
        @inbounds kL_vals[i] = kL
        @inbounds kT_vals[i] = kT

        function compute_christoffel(position::SVector{2, Float64})
            Γ = @MArray zeros(Float64, 2, 2, 2)
            # Fill Γ[...] as needed
            return Γ
        end


        for d in 2:dimensions
            # Christoffel geometric drift term
            @inbounds begin
                # Extract the position (τ, r) for particle i
                pos = @SVector [positions[1, i], positions[2, i]]
                Γ = compute_christoffel(pos)
                # Now use Γ safely
            end

            p0 = sqrt(m^2 + momenta[2, i] .^ 2)  # p^τ ≈ relativistic energy in LRF

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
            @inbounds deterministic_terms[d, i] = det_term
            @inbounds stochastic_terms[d, i] = sto_term
        end
    end
    return
end

"""
    kernel_update_momenta_LRF_gpu!

Update momenta of each particle in LRF using Langevin forces.
"""
@inline function kernel_update_momenta_LRF_gpu!(
    momenta, deterministic_terms, stochastic_terms,
    Δt, dimensions, N_particles
    )
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= N_particles
        for d in 1:dimensions
            Δp = @inbounds deterministic_terms[d, i] + sqrt(Δt) * stochastic_terms[d, i]
            @inbounds momenta[d, i] += Δp
        end
    end
    return
end

@inline function kernel_update_momenta_LRF_general_coords_gpu!(
    momenta, deterministic_terms, stochastic_terms,
    Δt, dimensions, N_particles,m
    )
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= N_particles
        for d in 2:dimensions
            momenta[d, i] += deterministic_terms[d, i] + sqrt(Δt) * stochastic_terms[d, i]
        end
        # Recalculate p^0 to satisfy mass-shell constraint
        p0 = sqrt(m^2 + momenta[2, i] .^ 2) 
        momenta[1, i] = p0
    end
    return
end

"""
    kernel_update_positions_gpu!

Move particles forward based on momenta. Reflects at r = 0.
"""
@inline function kernel_update_positions_gpu!(positions::CuDeviceMatrix{Float64}, momenta::CuDeviceMatrix{Float64}, m::Float64, Δt::Float64, N_particles::Int)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if idx <= N_particles
        
        p_sq = 0.0

        p_sq += momenta[1, idx]^2

        E = CUDA.sqrt(p_sq + m * m)

        @inbounds positions[1, idx] += Δt * momenta[1, idx] / E
        if positions[1, idx] < 0
            positions[1, idx] = -10.
            momenta[1, idx] = 0.0
        end
    end

    return
end

@inline function kernel_update_positions_general_coords_gpu!(positions::CuDeviceMatrix{Float64}, momenta::CuDeviceMatrix{Float64}, m::Float64, Δt::Float64, N_particles::Int)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if idx <= N_particles
        
        E = momenta[1, idx]
        for μ in 1:size(positions, 1)
            positions[μ, idx] += Δt * momenta[μ, idx] / E
        end
        if positions[2, idx] < 0
            positions[2, idx] = -10.
            momenta[2, idx] = 0.0
            p_spatial_sq = sum(momenta[2, idx] .^ 2)
            momenta[1, idx] = sqrt(m^2 + p_spatial_sq)
        end
    end

    return
end

"""
    kernel_save_snapshot_gpu!

Save 1D momenta magnitudes into a flattened buffer for snapshotting.
"""
@inline function kernel_save_snapshot_gpu!(history, snapshot, idx, N_particles)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= N_particles
        @inbounds history[i + (idx - 1) * N_particles] = snapshot[i]
    end
    return
end

@inline function kernel_save_snapshot_general_coords_gpu!(history, snapshot, idx, N_particles)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= N_particles
        @inbounds history[i + (idx - 1) * N_particles] = snapshot[i]
    end
    return
end

"""
    kernel_save_positions_gpu!

Save particle positions at current time step into history array.
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

@inline function kernel_save_positions_general_coords_gpu!(position_history, current_positions, save_idx::Int, N::Int)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= N
        
        for d in 2:size(current_positions, 1)
            @inbounds position_history[d, i, save_idx] = current_positions[d, i]
        end
        
    end
    return
end

end # module KernelsGPU
