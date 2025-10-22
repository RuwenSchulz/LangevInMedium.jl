module SimulateCPU

# === Imports ===
using ProgressMeter
using ..KernelsCPU
using ..Utils

# === Exported Symbols ===
export simulate_ensemble_bulk_cpu

function simulate_ensemble_bulk_cpu(
    r_grid_Langevin, heavy_quark_density,
    T_profile_MIS, ur_profile_MIS, mu_profile_MIS,nur_profile_MIS,
    TemperatureEvolutionn, VelocityEvolutionn, SpaceTimeGrid;
    N_particles::Int = 10_000,
    Δt::Float64 = 0.001,
    initial_time::Float64 = 0.0,
    final_time::Float64 = 1.0,
    save_interval::Float64 = 0.1,
    m::Float64 = 1.0,
    DsT::Float64 = 0.2,
    dimensions::Int = 3)

    # === Setup and Preallocation ===
    total_time = final_time - initial_time
    steps = floor(Int, total_time / Δt)
    save_every = Int(save_interval / Δt)
    num_saves = div(steps, save_every)

    xgrid, tgrid = SpaceTimeGrid

    x_matrix, p_matrix = sample_particles_from_density(r_grid_Langevin, heavy_quark_density, N_particles, T_profile_MIS,nur_profile_MIS, mu_profile_MIS;t0 = initial_time,mode=:density,m=m)
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
        positions = reshape(r_samples, 1, :)
        momenta   = reshape(p_r_samples, 1, :)

    else
        # --- full Cartesian mode ---
        positions = copy(x_matrix)
        momenta   = copy(p_matrix)
    end

    momenta_history = zeros(Float64, dimensions, N_particles, num_saves + 1)
    position_history = zeros(Float64, dimensions, N_particles, num_saves + 1)


    kernel_boost_to_lab_frame_cpu!(
    momenta, positions, xgrid, tgrid,
    VelocityEvolutionn, m, N_particles, 0, Δt, initial_time,radial_mode = radial_mode)

    momenta_history[:,:,1] .= momenta
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
            VelocityEvolutionn, m, N_particles, step, Δt, initial_time,radial_mode = radial_mode)

        if DsT == 0.0
            kernel_set_to_fluid_velocity_cpu!(
                momenta, positions,  xgrid, tgrid,
                VelocityEvolutionn, m, N_particles, step, Δt, initial_time,radial_mode = radial_mode)
        else

            # 2. Compute forces in rest frame
            kernel_compute_all_forces_cpu!(
                TemperatureEvolutionn, xgrid, tgrid,
                momenta, positions, p_mags, p_units,
                ηD_vals, kL_vals, kT_vals,
                ξ, deterministic_terms, stochastic_terms,
                Δt, m, random_directions,
                dimensions, N_particles, step, initial_time,DsT,
                radial_mode = radial_mode)
    
            # 3. Update momenta
            kernel_update_momenta_LRF_cpu!(
                momenta, deterministic_terms, stochastic_terms,
                Δt, dimensions, N_particles)

            # 4. Boost updated momenta back to lab frame
            kernel_boost_to_lab_frame_cpu!(
                momenta, positions, xgrid, tgrid,
                VelocityEvolutionn, m, N_particles, step, Δt, initial_time,radial_mode = radial_mode)
        end 
        # 5. Update positions
       
        kernel_update_positions_cpu!(
                    positions, momenta, m, Δt, N_particles,step,initial_time,
                    xgrid,tgrid, TemperatureEvolutionn,DsT;
                    dimensions,
                    radial_mode = radial_mode
                )


        # --- NaN / Inf check ---
        if any(!isfinite, momenta) || any(!isfinite, positions)
            @error "Detected NaN or Inf at step=$step" 
            display("⚠️  Step $step — NaN/Inf detected in simulation state.")
            println("Non-finite in momenta? ", any(!isfinite, momenta))
            println("Non-finite in positions? ", any(!isfinite, positions))
            #println("Non-finite values:")
            #println("momenta = ", momenta)
            #println("positions = ", positions)
            error("Breaking simulation due to NaN/Inf at step $step")
        end

        # 6. Save snapshots
        if step % save_every == 0
            save_idx = div(step, save_every) + 1

            kernel_save_momenta_cpu!(
                            momenta_history,momenta,save_idx, N_particles)

            kernel_save_positions_cpu!(
                position_history, positions, save_idx, N_particles)
        end

    end

    # === Final Data Packaging ===
    time_points = range(initial_time, final_time, length = num_saves + 1)
    position_history_vec = [position_history[:, :, i] for i in 1:size(position_history, 3)]
    momenta_history_vec  = [momenta_history[:, :, i] for i in 1:size(momenta_history, 3)]
    return time_points, momenta_history_vec, position_history_vec
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
