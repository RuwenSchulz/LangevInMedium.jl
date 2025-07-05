module SimulateCPUGeneralCoords

using ProgressMeter
using ..KernelsCPU
using ..Utils

export simulate_ensemble_bulk_general_coords_cpu

"""
    simulate_ensemble_bulk_general_coords_cpu(T_profile_MIS, ur_profile_MIS, mu_profile_MIS,
                               TemperatureEvolution, VelocityEvolution, SpaceTimeGrid;
                               N_particles=10_000, Δt=0.001, initial_time=0.0,
                               final_time=1.0, save_interval=0.1, m=1.0)

Run a Langevin simulation of particle motion in a flat spacetime using Milne coordinates (τ, r).
"""
function simulate_ensemble_bulk_general_coords_cpu(
    T_profile_MIS, ur_profile_MIS, mu_profile_MIS,
    TemperatureEvolution, VelocityEvolution, SpaceTimeGrid;
    N_particles::Int = 10_000,
    Δt::Float64 = 0.001,
    initial_time::Float64 = 0.0,
    final_time::Float64 = 1.0,
    save_interval::Float64 = 0.1,
    m::Float64 = 1.0,
    dimensions::Int = 2  # Milne: τ, r
)

    # === Setup ===
    total_time = final_time - initial_time
    steps = floor(Int, total_time / Δt)
    save_every = Int(save_interval / Δt)
    num_saves = div(steps, save_every)

    xgrid, tgrid = SpaceTimeGrid

    # === Initialize Particles ===
    position, moment = sample_initial_particles_milne!(
        m, dimensions, N_particles,initial_time, T_profile_MIS, ur_profile_MIS, mu_profile_MIS,xgrid)





    positions = copy(position)
    momenta = copy(moment)

    # === History Arrays ===
    momenta_history = [zeros(N_particles) for _ in 1:num_saves + 1]
    position_history = zeros(Float64, dimensions, N_particles, num_saves + 1)

    # Store spatial momentum magnitude (in Milne: just pr)
    momenta_history[1] .= abs.(momenta[2, :])
    position_history[:, :, 1] .= positions

    # === Buffers ===
    p_mags              = zeros(N_particles)
    p_units             = zeros(dimensions, N_particles)
    ηD_vals             = zeros(N_particles)
    kL_vals             = zeros(N_particles)
    kT_vals             = zeros(N_particles)
    deterministic_terms = zeros(dimensions, N_particles)
    stochastic_terms    = zeros(dimensions, N_particles)
    ξ                   = randn(dimensions, N_particles)
    random_directions   = randn(dimensions, N_particles)

    # Normalize directions
    norm_factors = sqrt.(sum(random_directions .^ 2, dims=1))
    random_directions ./= norm_factors

    # === Main Time Evolution ===
    @showprogress 10 "Running Langevin CPU simulation in Milne (τ,r)..." for step in 1:steps
        ξ .= randn(dimensions, N_particles)

        # 1. Boost momenta to local rest frame
        kernel_boost_to_rest_frame_general_coords_cpu!(
            momenta, positions, xgrid, tgrid,
            VelocityEvolution, m, N_particles, step, Δt, initial_time)

        # 2. Compute Langevin forces with Christoffel drift
        kernel_compute_all_forces_general_coords_cpu!(
            TemperatureEvolution, xgrid, tgrid,
            momenta, positions, p_mags, p_units,
            ηD_vals, kL_vals, kT_vals,
            ξ, deterministic_terms, stochastic_terms,
            Δt, m, random_directions,
            dimensions, N_particles, step, initial_time)

        # 3. Langevin update of momenta
        kernel_update_momenta_LRF_cpu!(
            momenta, deterministic_terms, stochastic_terms,
            Δt, dimensions, N_particles)

        # 4. Boost back to lab frame
        kernel_boost_to_lab_frame_general_coords_cpu!(
            momenta, positions, xgrid, tgrid,
            VelocityEvolution, m, N_particles, step, Δt, initial_time)

        # 5. Position update in Milne coordinates
        kernel_update_positions_radial_milne!(
            positions, momenta, m, Δt, N_particles)

        # 6. Save snapshots
        if step % save_every == 0
            save_idx = div(step, save_every) + 1
            kernel_save_snapshot_cpu!(
                momenta_history[save_idx],
                sqrt.(sum(momenta .^ 2, dims=1))[:], N_particles)
            kernel_save_positions_cpu!(
                position_history, positions, save_idx, N_particles)
        end
    end

    # === Output Packaging ===
    time_points = range(initial_time, final_time, length = num_saves + 1)
    position_history_vec = [position_history[:, :, i] for i in 1:size(position_history, 3)]

    return time_points, momenta_history, position_history_vec
end

end # module SimulateCPU
