module SimulateGPU

using ProgressMeter
using ..KernelsGPU
using ..Utils

export simulate_ensemble_bulk_gpu

function print_cuda_status()
    dev = CUDA.device()
    name = CUDA.name(dev)
    cap = CUDA.capability(dev)
    println("🟢 CUDA status:")
    println("  Device             : $name")
    println("  Compute Capability : $cap")
    println("  Memory Usage       :")
    CUDA.memory_status()
end


function simulate_ensemble_bulk_gpu(T_profile_MIS, ur_profile_MIS, mu_profile_MIS,TemperatureEvolutionn, VelocityEvolutionn, SpaceTimeGrid;
    N_particles::Int64=10_000,
    Δt::Float64=0.001,
    initial_time::Float64=0.0,
    final_time::Float64=1.0,
    save_interval::Float64=0.1,
    m::Float64=1.0,
    dimensions::Int64=3,
    )

    CUDA.reclaim()  # Free GPU memory before starting
    print_cuda_status()
    try
        # === Simulation parameters ===
        total_time = final_time - initial_time
        threads = 256
        blocks = cld(N_particles, threads)
        steps = floor(Int, total_time / Δt)
        save_every = Int(save_interval / Δt)
        num_saves = Int(steps ÷ save_every)

        # === Upload spacetime grid to GPU ===
        xgridd, ttgrid = SpaceTimeGrid
        xgrid = CuArray(xgridd)
        tgrid = CuArray(ttgrid)

        # === Initial conditions ===

        position, moment = sample_initial_particles_from_pdf!(
            m, dimensions, N_particles,
            initial_time, T_profile_MIS, ur_profile_MIS, mu_profile_MIS,
            (0., 10.), 200)


       

        TemperatureEvolution = CuArray(TemperatureEvolutionn)
        VelocityEvolution = CuArray(VelocityEvolutionn)

        # === Boost initial momenta into lab frame ===
        positions = CuArray(position)
        momenta = CuArray(moment)
        @cuda threads = threads blocks = blocks kernel_boost_to_lab_frame_gpu!(momenta, positions, xgrid, tgrid, VelocityEvolution, m, N_particles, 1, Δt, initial_time)

        # === Allocate memory for history ===
        momenta_history_gpu = CUDA.zeros(Float64, N_particles, num_saves + 1)
        p_mags = sqrt.(sum(momenta .^ 2, dims=1))
        momenta_history_gpu[:, 1] .= p_mags[1, :]

        position_history_gpu = CUDA.zeros(Float64, dimensions, N_particles, num_saves + 1)
        position_history_gpu[:, :, 1] .= positions

        # === Working GPU buffers ===
        p_mags = CUDA.zeros(Float64, N_particles)
        p_units = CUDA.zeros(Float64, dimensions, N_particles)
        ηD_vals = CUDA.zeros(Float64, N_particles)
        kL_vals = CUDA.zeros(Float64, N_particles)
        kT_vals = CUDA.zeros(Float64, N_particles)
        stochastic_terms = CUDA.zeros(Float64, dimensions, N_particles)
        deterministic_terms = CUDA.zeros(Float64, dimensions, N_particles)
        ξ = CUDA.zeros(Float64, dimensions, N_particles)
        random_directions = CUDA.randn(Float64, dimensions, N_particles)

        # Normalize random direction vectors (unit length)
        random_norms = sqrt.(sum(random_directions .^ 2, dims=1))
        random_directions ./= random_norms

        # === Main Langevin simulation loop ===
        @showprogress 10 "Running Langevin simulation..." for step in 1:steps
            CUDA.randn!(ξ)  # Gaussian noise at each step

            # Step 1: Transform to local rest frame (LRF)
            @cuda threads = threads blocks = blocks kernel_boost_to_rest_frame_gpu!(
                momenta, positions, xgrid, tgrid, VelocityEvolution, m, N_particles, step, Δt, initial_time)

            # Step 2: Compute deterministic + stochastic forces in LRF
            @cuda threads = threads blocks = blocks kernel_compute_all_forces_gpu!(
                TemperatureEvolution, xgrid, tgrid, momenta, positions,
                p_mags, p_units, ηD_vals, kL_vals, kT_vals,
                ξ, deterministic_terms, stochastic_terms,
                Δt, m, random_directions, dimensions, N_particles, step, initial_time)

            # Step 3: Update momenta in LRF
            @cuda threads = threads blocks = blocks kernel_update_momenta_LRF_gpu!(
                momenta, deterministic_terms, stochastic_terms, Δt, dimensions, N_particles)

            # Step 4: Boost back to lab frame
            @cuda threads = threads blocks = blocks kernel_boost_to_lab_frame_gpu!(
                momenta, positions, xgrid, tgrid, VelocityEvolution, m, N_particles, step, Δt, initial_time)

            # Step 5: Move particles forward in space
            @cuda threads = threads blocks = blocks kernel_update_positions_gpu!(
                positions, momenta, m, Δt, N_particles)

            # Step 6: Save data at specified intervals
            if step % save_every == 0
                save_idx = div(step, save_every) + 1
                @cuda threads = threads blocks = blocks kernel_save_snapshot_gpu!(momenta_history_gpu, momenta, save_idx, N_particles)
                @cuda threads = threads blocks = blocks kernel_save_positions_gpu!(position_history_gpu, positions, save_idx, N_particles)
            end
        end

        # === Convert saved data to CPU memory ===
        time_points = range(initial_time, final_time, length=num_saves + 1)

        momenta_history_vec = [Array(momenta_history_gpu[:, step]) for step in 1:num_saves+1]
        position_history_vec = [Array(position_history_gpu[:, :, step]) for step in 1:num_saves+1]

        return time_points, momenta_history_vec, position_history_vec

    finally

        GC.gc()
        CUDA.reclaim()
        GC.gc()
        CUDA.reclaim()
    end
end

end # module
