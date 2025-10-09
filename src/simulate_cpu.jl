module SimulateCPU

# === Imports ===
using ProgressMeter
using ..KernelsCPU
using ..Utils

# === Exported Symbols ===
export simulate_ensemble_bulk_cpu

"""
    simulate_ensemble_bulk_cpu(T_profile_MIS, ur_profile_MIS, mu_profile_MIS,
                               TemperatureEvolutionn, VelocityEvolutionn, SpaceTimeGrid;
                               N_particles=10_000, Δt=0.001, initial_time=0.0,
                               final_time=1.0, save_interval=0.1, m=1.0, dimensions=3)

Run a CPU-based Langevin simulation of particle motion in a hydrodynamic background.

# Arguments
- `T_profile_MIS`, `ur_profile_MIS`, `mu_profile_MIS`: Functions returning temperature, velocity, and chemical potential at `(r, t)`.
- `TemperatureEvolutionn`, `VelocityEvolutionn`: 2D arrays encoding evolution of temperature and velocity over space-time.
- `SpaceTimeGrid`: Tuple of spatial and temporal grid arrays `(xgrid, tgrid)`.

# Keyword Arguments
- `N_particles`: Number of particles (default 10,000).
- `Δt`: Time step size.
- `initial_time`, `final_time`: Time range for simulation.
- `save_interval`: Time interval at which to record snapshots.
- `m`: Particle mass.
- `dimensions`: Number of spatial dimensions.
- `DsT`: Diffusion coefficient parameter (default 0.2).

# Returns
- `time_points`: Vector of saved time points.
- `momenta_history`: Vector of momentum magnitudes at each time point.
- `position_history_vec`: Vector of position arrays at each time point.
"""
function simulate_ensemble_bulk_cpu(heavy_quark_density,
    T_profile_MIS, ur_profile_MIS, mu_profile_MIS,
    TemperatureEvolutionn, VelocityEvolutionn, SpaceTimeGrid;
    N_particles::Int = 10_000,
    Δt::Float64 = 0.001,
    initial_time::Float64 = 0.0,
    final_time::Float64 = 1.0,
    save_interval::Float64 = 0.1,
    m::Float64 = 1.0,
    DsT::Float64 = 0.2,
    dimensions::Int = 3,
)

    # === Setup and Preallocation ===
    total_time = final_time - initial_time
    steps = floor(Int, total_time / Δt)
    save_every = Int(save_interval / Δt)
    num_saves = div(steps, save_every)

    xgrid, tgrid = SpaceTimeGrid

    # Initial particle positions and momenta from Boltzmann distribution
    #n_rt = compute_MIS_distribution(xgrid, initial_time,T_profile_MIS,mu_profile_MIS,m)
    #position, moment = sample_heavy_quarks(heavy_quark_density, N_particles, xgrid, initial_time,m,T_profile_MIS,dimensions)

    position, moment = sample_phase_space4(n_rt, N_particles, xgrid, initial_time,
    m, T_profile_MIS, mu_profile_MIS, dimensions)
    
    #position, moment = sample_initial_particles_at_origin!(m, spatial_dimensions, N_particles,t0, T_profile_MIS)



    positions = copy(position)
    momenta = copy(moment)

    

    # History arrays for positions and momenta
    momenta_history = [zeros(N_particles) for _ in 1:num_saves + 1]
    position_history = zeros(Float64, dimensions, N_particles, num_saves + 1)


    kernel_boost_to_lab_frame_cpu!(
    momenta, positions, xgrid, tgrid,
    VelocityEvolutionn, m, N_particles, 0, Δt, initial_time)

    momenta_history[1] .= sqrt.(sum(momenta .^ 2, dims=1))[:]
    position_history[:, :, 1] .= positions

    # Working buffers
    p_mags              = zeros(N_particles)
    p_units             = zeros(dimensions, N_particles)
    ηD_vals             = zeros(N_particles)
    kL_vals             = zeros(N_particles)
    kT_vals             = zeros(N_particles)
    deterministic_terms = zeros(dimensions, N_particles)
    stochastic_terms    = zeros(dimensions, N_particles)
    ξ                   = randn(dimensions, N_particles)
    random_directions   = randn(dimensions, N_particles)

    # Normalize random directions
    norm_factors = sqrt.(sum(random_directions .^ 2, dims=1))
    random_directions ./= norm_factors


    # === Langevin Time Evolution Loop ===
    @showprogress 10 "Running Langevin CPU simulation..." for step in 1:steps
        ξ .= randn(dimensions, N_particles)


        # 1. Boost momenta to local rest frame
        kernel_boost_to_rest_frame_cpu!(
            momenta, positions, xgrid, tgrid,
            VelocityEvolutionn, m, N_particles, step, Δt, initial_time)

        if DsT == 0.0
            kernel_set_to_fluid_velocity_cpu!(
                momenta, positions,  xgrid, tgrid,
                VelocityEvolutionn, m, N_particles, step, Δt, initial_time)
        else 
            # 2. Compute forces in rest frame
            kernel_compute_all_forces_cpu!(
                TemperatureEvolutionn, xgrid, tgrid,
                momenta, positions, p_mags, p_units,
                ηD_vals, kL_vals, kT_vals,
                ξ, deterministic_terms, stochastic_terms,
                Δt, m, random_directions,
                dimensions, N_particles, step, initial_time,DsT)

    
            # 3. Update momenta
            kernel_update_momenta_LRF_cpu!(
                momenta, deterministic_terms, stochastic_terms,
                Δt, dimensions, N_particles)

            # 4. Boost updated momenta back to lab frame
            kernel_boost_to_lab_frame_cpu!(
                momenta, positions, xgrid, tgrid,
                VelocityEvolutionn, m, N_particles, step, Δt, initial_time)
        end 
        # 5. Update positions
        kernel_update_positions_cpu!(positions, momenta, m, Δt, dimensions,N_particles)

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

    # === Final Data Packaging ===
    time_points = range(initial_time, final_time, length = num_saves + 1)
    position_history_vec = [position_history[:, :, i] for i in 1:size(position_history, 3)]

    return time_points, momenta_history, position_history_vec
end

function simulate_ensemble_bulk_cpu(
    T::Float64;
    N_particles::Int = 10_000,
    Δt::Float64 = 0.001,
    initial_time::Float64 = 0.0,
    final_time::Float64 = 1.0,
    save_interval::Float64 = 0.1,
    m::Float64 = 1.0,
    dimensions::Int = 3,
    p0 = 1.0,
    initial_condition = "delta"
)

    # === Setup and Preallocation ===
    total_time = final_time - initial_time
    steps = floor(Int, total_time / Δt)
    save_every = Int(save_interval / Δt)
    num_saves = div(steps, save_every)


    # Initial particle positions and momenta from Boltzmann distribution
    moment = sample_initial_particles_at_origin_no_position!(initial_condition,p0, dimensions, N_particles)

    momenta = copy(moment)

    # History arrays for positions and momenta
    momenta_history = [zeros(N_particles) for _ in 1:num_saves + 1]

    momenta_history[1] .= sqrt.(sum(momenta .^ 2, dims=1))[:]

    # Working buffers
    p_mags              = zeros(N_particles)
    p_units             = zeros(dimensions, N_particles)
    ηD_vals             = zeros(N_particles)
    kL_vals             = zeros(N_particles)
    kT_vals             = zeros(N_particles)
    deterministic_terms = zeros(dimensions, N_particles)
    stochastic_terms    = zeros(dimensions, N_particles)
    ξ                   = randn(dimensions, N_particles)
    random_directions   = randn(dimensions, N_particles)

    # Normalize random directions
    norm_factors = sqrt.(sum(random_directions .^ 2, dims=1))
    random_directions ./= norm_factors


    # === Langevin Time Evolution Loop ===
    @showprogress 10 "Running Langevin CPU simulation..." for step in 1:steps
        ξ .= randn(dimensions, N_particles)

       

        # 2. Compute forces in rest frame
        kernel_compute_all_forces_cpu!(
            T,
            momenta, p_mags, p_units,
            ηD_vals, kL_vals, kT_vals,
            ξ, deterministic_terms, stochastic_terms,
            Δt, m, random_directions,
            dimensions, N_particles, step, initial_time)

        # 3. Update momenta
        kernel_update_momenta_LRF_cpu!(
            momenta, deterministic_terms, stochastic_terms,
            Δt, dimensions, N_particles)

        # 6. Save snapshots
        if step % save_every == 0
            save_idx = div(step, save_every) + 1
            kernel_save_snapshot_cpu!(
                momenta_history[save_idx],
                sqrt.(sum(momenta .^ 2, dims=1))[:], N_particles)
        end

    end

    # === Final Data Packaging ===
    time_points = range(initial_time, final_time, length = num_saves + 1)


    return time_points, momenta_history
end
end # module SimulateCPU
