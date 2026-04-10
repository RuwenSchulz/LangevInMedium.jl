module SimulateCPU

# === Imports ===
using ProgressMeter
using ..KernelsCPU
using ..Utils
using ..Transport

# === Exported Symbols ===
export simulate_ensemble_bulk_cpu

function simulate_ensemble_bulk_cpu(
    r_grid_Langevin,p_grid_Langevin, heavy_quark_density,
    TemperatureEvolutionn, VelocityEvolutionn, SpaceTimeGrid;
    N_particles::Int = 10_000,
    Δt::Float64 = 0.001,
    initial_time::Float64 = 0.0,
    final_time::Float64 = 1.0,
    save_interval::Float64 = 0.1,
    m::Float64 = 1.0,
    DsT::Float64 = 0.2,
    DsT_linear::Bool = false,
    DsT_slope::Float64 = 1.765,
    DsT_offset::Float64 = -0.159,
    Tfo::Float64 = 0.156,
    dimensions::Int = 3,
    cartesian_spatial_sampling::Union{Nothing,Bool} = nothing,
    antithetic_momenta::Bool = false,
    position_diffusion::Bool = false,
    momentum_langevin::Bool = true,
    reflecting_boundary::Bool = false)

    # === Setup and Preallocation ===
    total_time = final_time - initial_time
    steps = floor(Int, total_time / Δt)
    save_every = Int(save_interval / Δt)
    num_saves = div(steps, save_every)

    xgrid, tgrid = SpaceTimeGrid

    # For `dimensions == 1` we still evolve a *radial* degree of freedom in the
    # transverse plane. Sampling directly in polar (r,φ) on the grid can create
    # small-r artifacts; instead we default to Cartesian (x,y) sampling and then
    # collapse to r = √(x²+y²) and p_r = p·ê_r.
    do_cartesian_sampling = cartesian_spatial_sampling === nothing ? (dimensions == 1 || dimensions >= 2) : cartesian_spatial_sampling

    # Initial sampling: optionally use antithetic momentum pairs (p and -p) at
    # the same position in the local rest frame. This is a pure variance-reduction
    # technique: it preserves n and J^τ exactly (depends only on counts/|p|) and
    # strongly reduces noise in signed currents like J^r.
    if antithetic_momenta
        N_half = N_particles ÷ 2
        N_rem  = N_particles - 2 * N_half

        x_half, p_half = sample_particles_from_FONLL(r_grid_Langevin, p_grid_Langevin, heavy_quark_density, N_half;
            cartesian_spatial_sampling = do_cartesian_sampling)

        x_matrix = zeros(eltype(x_half), size(x_half, 1), N_particles)
        p_matrix = zeros(eltype(p_half), size(p_half, 1), N_particles)

        @inbounds for i in 1:N_half
            j1 = 2i - 1
            j2 = 2i
            x_matrix[:, j1] .= x_half[:, i]
            x_matrix[:, j2] .= x_half[:, i]
            p_matrix[:, j1] .= p_half[:, i]
            p_matrix[:, j2] .= -p_half[:, i]
        end

        if N_rem == 1
            x1, p1 = sample_particles_from_FONLL(r_grid_Langevin, p_grid_Langevin, heavy_quark_density, 1;
                cartesian_spatial_sampling = do_cartesian_sampling)
            x_matrix[:, end] .= x1[:, 1]
            p_matrix[:, end] .= p1[:, 1]
        end
    else
        x_matrix, p_matrix = sample_particles_from_FONLL(r_grid_Langevin, p_grid_Langevin, heavy_quark_density, N_particles;
            cartesian_spatial_sampling = do_cartesian_sampling)
    end
    
    if dimensions == 1
        radial_mode = true
    else 
        radial_mode = false
    end

    if radial_mode
        # --- radial reduction ---
        # compute r = sqrt(x^2 + y^2)
        r_samples = sqrt.(x_matrix[1, :].^2 .+ x_matrix[2, :].^2)

        # At r≈0 the radial direction ê_r is undefined. Do NOT compute p_r=(x·p)/r
        # with r≈0 (it creates huge/garbage p_r). Instead, for very small r we
        # choose a random unit vector in the transverse plane.
        dr0 = (length(r_grid_Langevin) >= 2) ? abs(float(r_grid_Langevin[2] - r_grid_Langevin[1])) : 0.0
        r_axis_eps = max(1e-12, 0.5 * dr0)

        ex = similar(r_samples)
        ey = similar(r_samples)
        is_regular = r_samples .> r_axis_eps

        @inbounds begin
            ex[is_regular] .= x_matrix[1, is_regular] ./ r_samples[is_regular]
            ey[is_regular] .= x_matrix[2, is_regular] ./ r_samples[is_regular]

            nsmall = count(.!is_regular)
            if nsmall > 0
                rx = randn(nsmall)
                ry = randn(nsmall)
                invn = 1.0 ./ (sqrt.(rx.^2 .+ ry.^2) .+ eps())
                ex[.!is_regular] .= rx .* invn
                ey[.!is_regular] .= ry .* invn
            end
        end

        # p_r = p⃗ · ê_r (finite even at r=0 with the above convention)
        p_r_samples = ex .* p_matrix[1, :] .+ ey .* p_matrix[2, :]

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

    # === Precompute τn(T) spline (main3 logic) ===
    # Only needed when we actually run momentum Langevin with DsT > 0.
    tau_Tmin::Float64 = 0.0
    tau_invdT::Float64 = 1.0
    tau_vals = Float64[0.0, 0.0]
    if momentum_langevin && DsT > 0.0
        Tmin = max(float(minimum(TemperatureEvolutionn)), 0.0)
        Tmax = max(float(maximum(TemperatureEvolutionn)), Tmin + eps(Float64))
        tau_Tmin, tau_invdT, tau_vals = build_tau_n_spline(m, DsT;
            Tmin = Tmin, Tmax = Tmax, nT = 1024,
            DsT_linear = DsT_linear,
            DsT_slope = DsT_slope,
            DsT_offset = DsT_offset,
            Tfo = Tfo)
    end


    # === Langevin Time Evolution Loop ===
    @showprogress 10 "Running Langevin CPU simulation..." for step in 1:steps
        ξ .= randn(dimensions, N_particles)

        # 1. Boost momenta to local rest frame
        kernel_boost_to_rest_frame_cpu!(
            momenta, positions, xgrid, tgrid,
            VelocityEvolutionn, m, N_particles, step, Δt, initial_time,radial_mode = radial_mode)

        if !momentum_langevin || DsT == 0.0
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
                tau_Tmin = tau_Tmin,
                tau_invdT = tau_invdT,
                tau_vals = tau_vals,
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
                    radial_mode = radial_mode,
                    position_diffusion = position_diffusion,
                    reflecting_boundary = reflecting_boundary
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
