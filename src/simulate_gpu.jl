module SimulateGPU

# === Imports ===
using ProgressMeter
using ..KernelsGPU
using ..Utils
using CUDA

# === Exports ===
export simulate_ensemble_bulk_gpu


function print_cuda_status()
    dev = CUDA.device()
    name = CUDA.name(dev)
    cap = CUDA.capability(dev)
    println("🟢 CUDA status:")
    println("  Device : $name")
    println("  Compute Capability : $cap")
    CUDA.memory_status()
end


function simulate_ensemble_bulk_gpu(
    r_grid_Langevin,p_grid_Langevin,heavy_quark_density,
    TemperatureEvolutionn, VelocityEvolutionn, SpaceTimeGrid;
    N_particles::Int64=10_000,
    Δt::Float64=0.001,
    initial_time::Float64=0.0,
    final_time::Float64=1.0,
    save_interval::Float64=0.1,
    m::Float64=1.0,
    DsT::Float64=0.2,
    dimensions::Int64=3,
)
    CUDA.reclaim()  # Free any unused GPU memory
    print_cuda_status()

    try
        # === Derived parameters ===
        total_time = final_time - initial_time
        steps = floor(Int, total_time / Δt)
        save_every = Int(save_interval / Δt)
        num_saves = Int(steps ÷ save_every)

        threads = 256
        blocks = cld(N_particles, threads)

        # === Upload background grids to GPU ===
        xgridd, ttgrid = SpaceTimeGrid
        xgrid = CuArray(xgridd)
        tgrid = CuArray(ttgrid)
        TemperatureEvolution = CuArray(TemperatureEvolutionn)
        VelocityEvolution = CuArray(VelocityEvolutionn)
        
        x_matrix, p_matrix = sample_particles_from_FONLL(r_grid_Langevin,p_grid_Langevin, heavy_quark_density, N_particles)
        
        if dimensions == 1
            radial_mode = true
        else 
            radial_mode = false
        end

        if radial_mode
            # --- radial reduction ---
            # compute r = sqrt(x^2 + y^2)
            r_samples = sqrt.(x_matrix[1, :].^2 .+ x_matrix[2, :].^2)

            # avoid division by zero
            r_safe = @. ifelse(r_samples < eps(), eps(), r_samples)

            # compute p_r = (x·p_x + y·p_y) / r
            p_r_samples = (x_matrix[1, :] .* p_matrix[1, :] .+
                        x_matrix[2, :] .* p_matrix[2, :]) ./ r_safe

            # reshape to shape [1, N]
            position = reshape(r_samples, 1, :)
            moment   = reshape(p_r_samples, 1, :)

        else
            # --- full Cartesian mode ---
            position = copy(x_matrix)
            moment   = copy(p_matrix)
        end

        positions = CuArray(position)
        momenta = CuArray(moment)

        # === Boost initial momenta into lab frame ===
        # Keep step indexing consistent with the CPU backend (t = t0 at initialization).
        @cuda threads=threads blocks=blocks kernel_boost_to_lab_frame_gpu!(
            momenta, positions, xgrid, tgrid,
            VelocityEvolution, m, N_particles, 0, Δt, initial_time, radial_mode)

        # === Allocate history arrays ===
        momenta_history_gpu = CUDA.zeros(Float64, dimensions, N_particles, num_saves + 1)
        momenta_history_gpu[:,:, 1] .= momenta

        position_history_gpu = CUDA.zeros(Float64, dimensions, N_particles, num_saves + 1)
        position_history_gpu[:, :, 1] .= positions

        # === Allocate working buffers ===
        p_mags              = CUDA.zeros(Float64, N_particles)
        p_units             = CUDA.zeros(Float64, dimensions, N_particles)
        ηD_vals             = CUDA.zeros(Float64, N_particles)
        kL_vals             = CUDA.zeros(Float64, N_particles)
        kT_vals             = CUDA.zeros(Float64, N_particles)
        stochastic_terms    = CUDA.zeros(Float64, dimensions, N_particles)
        deterministic_terms = CUDA.zeros(Float64, dimensions, N_particles)
        ξ_momentum          = CUDA.zeros(Float64, dimensions, N_particles)
        ξ_position          = CUDA.zeros(Float64, dimensions, N_particles)  # NEW: for position diffusion
        random_directions   = CUDA.zeros(Float64, dimensions, N_particles)

        # === Langevin dynamics loop ===
        @showprogress 10 "Running Langevin simulation GPU Cartesian..." for step in 1:steps
            CUDA.randn!(ξ_momentum)  # Sample Gaussian noise at every step
            CUDA.randn!(ξ_position)  # Sample Gaussian noise at every step
            CUDA.randn!(random_directions)  # Sample Gaussian noise at every step

            # Normalize random direction vectors
            random_norms = sqrt.(sum(random_directions .^ 2, dims=1))
            random_directions ./= random_norms

            # Step 1: Transform momenta to local rest frame (LRF)
            @cuda threads=threads blocks=blocks kernel_boost_to_rest_frame_gpu!(
                momenta, positions, xgrid, tgrid, VelocityEvolution,
                m, N_particles, step, Δt, initial_time,radial_mode)

            if DsT == 0.0

                @cuda threads=threads blocks=blocks kernel_set_to_fluid_velocity_gpu!(
                    momenta, positions, xgrid, tgrid,
                    VelocityEvolution, m, N_particles, step, Δt, initial_time,radial_mode)
            else
                # Step 2: Compute deterministic and stochastic terms in LRF
                @cuda threads=threads blocks=blocks kernel_compute_all_forces_gpu!(
                    TemperatureEvolution, xgrid, tgrid, momenta, positions,
                    p_mags, p_units, ηD_vals, kL_vals, kT_vals,
                    ξ_momentum, deterministic_terms, stochastic_terms,
                    Δt, m, random_directions, dimensions, N_particles,
                    step, initial_time,DsT,radial_mode)

                # Step 3: Update momenta in LRF
                @cuda threads=threads blocks=blocks kernel_update_momenta_LRF_gpu!(
                    momenta, deterministic_terms, stochastic_terms,
                    Δt, dimensions, N_particles)

                # Step 4: Boost updated momenta back to lab frame
                @cuda threads=threads blocks=blocks kernel_boost_to_lab_frame_gpu!(
                    momenta, positions, xgrid, tgrid, VelocityEvolution,
                    m, N_particles, step, Δt, initial_time,radial_mode)
            end 
            # Step 5: Update particle positions based on momenta
            # IMPORTANT: pass the current time-step, not the total number of steps.
            @cuda threads=threads blocks=blocks kernel_update_positions_gpu!(
                positions, momenta, m, Δt, N_particles, step, initial_time,
                xgrid, tgrid, TemperatureEvolution, DsT, dimensions, radial_mode, ξ_position)

            # Step 6: Save state if necessary
            if step % save_every == 0
                save_idx = div(step, save_every) + 1
                @cuda threads=threads blocks=blocks kernel_save_momenta_gpu!(
                    momenta_history_gpu, momenta, save_idx, N_particles,dimensions)
                @cuda threads=threads blocks=blocks kernel_save_positions_gpu!(
                    position_history_gpu, positions, save_idx, N_particles,dimensions)
            end
        end

        # === Transfer saved histories back to CPU ===
        time_points = range(initial_time, final_time, length=num_saves + 1)
        momenta_history_vec  = [Array(momenta_history_gpu[:, :, step]) for step in 1:num_saves+1]
        position_history_vec = [Array(position_history_gpu[:, :, step]) for step in 1:num_saves+1]

        # Finalize all CuArrays
        finalize(p_mags)
        finalize(p_units)
        finalize(ηD_vals)
        finalize(kL_vals)
        finalize(kT_vals)
        finalize(stochastic_terms)
        finalize(deterministic_terms)
        finalize(ξ_momentum)
        finalize(ξ_position)
        finalize(random_directions)

        # Clear references
        p_mags = nothing
        p_units = nothing
        ηD_vals = nothing
        kL_vals = nothing
        kT_vals = nothing
        stochastic_terms = nothing
        deterministic_terms = nothing
        ξ_momentum = nothing
        ξ_position = nothing
        random_directions = nothing

        # Aggressive cleanup
        GC.gc(true)
        GC.gc(true)
        CUDA.reclaim()

        return time_points, momenta_history_vec, position_history_vec

    finally
        # Final cleanup: force garbage collection and memory release
        GC.gc()
        CUDA.reclaim()
        GC.gc()
        CUDA.reclaim()
    end
end

end # module SimulateGPU
