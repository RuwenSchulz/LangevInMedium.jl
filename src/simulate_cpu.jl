module SimulateCPU

using ProgressMeter
using ..KernelsCPU
using ..Utils

export simulate_ensemble_bulk_cpu



function simulate_ensemble_bulk_cpu(
    T_profile_MIS, ur_profile_MIS, mu_profile_MIS,
    TemperatureEvolutionn, VelocityEvolutionn, SpaceTimeGrid;
    N_particles::Int=10_000,
    Δt::Float64=0.001,
    initial_time::Float64=0.0,
    final_time::Float64=1.0,
    save_interval::Float64=0.1,
    m::Float64=1.0,
    dimensions::Int=3,
 )
    

    # === Simulation setup ===
    total_time = final_time - initial_time
    steps = floor(Int, total_time / Δt)
    save_every = Int(save_interval / Δt)
    num_saves = div(steps, save_every)

    xgrid, tgrid = SpaceTimeGrid

    # Initial conditions
    position, moment = sample_initial_particles_from_pdf!(
        m, dimensions, N_particles,
        initial_time, T_profile_MIS, ur_profile_MIS, mu_profile_MIS,
        (0., 10.), 200)

    positions = copy(position)
    momenta = copy(moment)

    # Preallocate history arrays
    momenta_history = [zeros(N_particles) for _ in 1:num_saves+1]
    position_history = zeros(Float64, dimensions, N_particles, num_saves + 1)

    #position_history = [zeros(dimensions, N_particles) for _ in 1:num_saves+1]
    momenta_history[1] .= sqrt.(sum(momenta .^ 2, dims=1))[:]
    position_history[:, :, 1] .= positions


    # Preallocate buffers
    p_mags = zeros(N_particles)
    p_units = zeros(dimensions, N_particles)
    ηD_vals = zeros(N_particles)
    kL_vals = zeros(N_particles)
    kT_vals = zeros(N_particles)
    deterministic_terms = zeros(dimensions, N_particles)
    stochastic_terms = zeros(dimensions, N_particles)
    ξ = randn(dimensions, N_particles)
    random_directions = randn(dimensions, N_particles)
    norm_factors = sqrt.(sum(random_directions .^ 2, dims=1))
    random_directions ./= norm_factors

    # === Main loop ===
    @showprogress 10 "Running Langevin CPU simulation..." for step in 1:steps
        ξ .= randn(dimensions, N_particles)

        kernel_boost_to_rest_frame_cpu!(momenta, positions, xgrid, tgrid, VelocityEvolutionn, m, N_particles, step, Δt, initial_time)
        kernel_compute_all_forces_cpu!(
            TemperatureEvolutionn, xgrid, tgrid,
            momenta, positions, p_mags, p_units,
            ηD_vals, kL_vals, kT_vals,
            ξ, deterministic_terms, stochastic_terms,
            Δt, m, random_directions,
            dimensions, N_particles, step, initial_time)

        kernel_update_momenta_LRF_cpu!(momenta, deterministic_terms, stochastic_terms, Δt, dimensions, N_particles)
        kernel_boost_to_lab_frame_cpu!(momenta, positions, xgrid, tgrid, VelocityEvolutionn, m, N_particles, step, Δt, initial_time)
        kernel_update_positions_cpu!(positions, momenta, m, Δt, N_particles)

        if step % save_every == 0
            save_idx = div(step, save_every) + 1
            kernel_save_snapshot_cpu!(momenta_history[save_idx], sqrt.(sum(momenta .^ 2, dims=1))[:], N_particles)
            kernel_save_positions_cpu!(position_history, positions, save_idx, N_particles)
        end
    end
    position_history_vec = [position_history[:, :, i] for i in 1:size(position_history, 3)]

    time_points = range(initial_time, final_time, length=num_saves + 1)
    return time_points, momenta_history, position_history_vec
end

end