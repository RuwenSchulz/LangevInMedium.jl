module SimulateGPU

# === Imports ===
using ProgressMeter
using ..KernelsGPU
using ...Utils
using CUDA
using ...Transport

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
    DsT_linear::Bool=false,
    DsT_slope::Float64=1.765,
    DsT_offset::Float64=-0.159,
    Tfo::Float64=0.156,
    dimensions::Int64=3,
    cartesian_spatial_sampling::Union{Nothing,Bool}=nothing,
    antithetic_momenta::Bool=false,
    position_diffusion::Bool = false,
    momentum_langevin::Bool = true,
    reflecting_boundary::Bool = false,
    collision_mode::Symbol = :langevin,
    x_init::Union{Nothing, AbstractMatrix} = nothing,
    p_init::Union{Nothing, AbstractMatrix} = nothing,
    V2Evolutionn::Union{Nothing, AbstractMatrix} = nothing,
    psi2::Float64 = 0.0,
    freezeout_capture::Bool = false,
    freezeout_interp::Bool = true,
    integrator_mode::Int = 0,   # 0 = pre-point exact-OU (legacy), 1 = drift-midpoint O(ε²)
    relativistic::Bool = true,  # true = drag ·m/E (Jüttner); false = ηD (Maxwell)
)
    CUDA.reclaim()  # Free any unused GPU memory
    print_cuda_status()

    try
        # === Derived parameters ===
        total_time = final_time - initial_time
        steps = floor(Int, total_time / Δt)
        save_every = round(Int, save_interval / Δt)   # robust to binary roundoff (matches CPU path)
        num_saves = steps ÷ save_every
        # ON-THE-FLY FREEZE-OUT: no trajectory history is needed (memory ∝ N, VRAM-safe at any cadence);
        # each particle latches its own state the step it first crosses T=Tfo. Keep a trivial 1-snapshot
        # history so the existing allocation/return plumbing is untouched.
        if freezeout_capture
            num_saves = 1
            save_every = max(steps, 1)
        end

        threads = 256
        blocks = cld(N_particles, threads)

        # === Upload background grids to GPU ===
        xgridd, ttgrid = SpaceTimeGrid
        xgrid = CuArray(xgridd)
        tgrid = CuArray(ttgrid)
        TemperatureEvolution = CuArray(TemperatureEvolutionn)
        VelocityEvolution = CuArray(VelocityEvolutionn)
        # optional elliptic-flow modulation: upload the v2(r,t) field, or a dummy when off
        use_v2 = V2Evolutionn !== nothing
        V2Evolution = use_v2 ? CuArray(V2Evolutionn) : CuArray(zeros(Float64, 1, 1))

        # === Precompute the DRAG spline τ_drag(T) on CPU and upload to GPU ===
        tau_Tmin::Float64 = 0.0
        tau_invdT::Float64 = 1.0
        tau_vals = Float64[0.0, 0.0]
        # RTA/BGK needs the CURRENT time instead — see build_taun_current_spline.
        taun_vals = Float64[0.0, 0.0]
        if momentum_langevin && DsT > 0.0
            Tmin = max(float(minimum(TemperatureEvolutionn)), 0.0)
            Tmax = max(float(maximum(TemperatureEvolutionn)), Tmin + eps(Float64))
            tau_Tmin, tau_invdT, tau_vals = build_tau_drag_spline(m, DsT;
                Tmin = Tmin, Tmax = Tmax, nT = 1024,
                DsT_linear = DsT_linear,
                DsT_slope = DsT_slope,
                DsT_offset = DsT_offset,
                Tfo = Tfo)
            if collision_mode == :rta
                _, _, taun_vals = build_taun_current_spline(m, DsT;
                    Tmin = Tmin, Tmax = Tmax, nT = 1024,
                    DsT_linear = DsT_linear,
                    DsT_slope = DsT_slope,
                    DsT_offset = DsT_offset,
                    Tfo = Tfo)
            end
        end
        tau_vals_d = CuArray(tau_vals)
        taun_vals_d = CuArray(taun_vals)

        # === RTA/BGK: precompute the isotropic Jüttner inverse-CDF table on CPU, upload to GPU ===
        # Divergence-free `p*` sampler for kernel_rta_collision_gpu! (replaces the CPU rejection loop).
        rta_mode = collision_mode == :rta
        invcdf_nU::Int = 2
        invcdf_nT::Int = 2
        invcdf_Tmin::Float64 = 0.0
        invcdf_invdT::Float64 = 1.0
        invcdf_d = CuArray(Float64[0.0, 0.0, 0.0, 0.0])
        if rta_mode && momentum_langevin && DsT > 0.0
            Tmin_ic = max(float(minimum(TemperatureEvolutionn)), 0.0)
            Tmax_ic = max(float(maximum(TemperatureEvolutionn)), Tmin_ic + eps(Float64))
            invcdf_vals, invcdf_nU, invcdf_nT, invcdf_Tmin, invcdf_invdT =
                build_juttner_invcdf(m, dimensions; Tmin = Tmin_ic, Tmax = Tmax_ic,
                                     nT = 256, nU = 2048, np = 8192)
            invcdf_d = CuArray(invcdf_vals)
        end

        do_cartesian_sampling = cartesian_spatial_sampling === nothing ? (dimensions == 1 || dimensions >= 2) : cartesian_spatial_sampling

        # Use pre-sampled particles when provided (e.g., anisotropic radial-boost ICs),
        # otherwise sample from the density matrix with optional antithetic pairs.
        x_matrix, p_matrix = if x_init !== nothing && p_init !== nothing
            Matrix{Float64}(x_init), Matrix{Float64}(p_init)
        elseif antithetic_momenta
            N_half = N_particles ÷ 2
            N_rem = N_particles - 2 * N_half

            x_half, p_half = sample_particles_from_FONLL(
                r_grid_Langevin, p_grid_Langevin, heavy_quark_density, N_half;
                cartesian_spatial_sampling = do_cartesian_sampling,
            )

            x_matrix_local = zeros(eltype(x_half), size(x_half, 1), N_particles)
            p_matrix_local = zeros(eltype(p_half), size(p_half, 1), N_particles)

            @inbounds for i in 1:N_half
                j1 = 2i - 1
                j2 = 2i
                x_matrix_local[:, j1] .= x_half[:, i]
                x_matrix_local[:, j2] .= x_half[:, i]
                p_matrix_local[:, j1] .= p_half[:, i]
                p_matrix_local[:, j2] .= -p_half[:, i]
            end

            if N_rem == 1
                x1, p1 = sample_particles_from_FONLL(
                    r_grid_Langevin, p_grid_Langevin, heavy_quark_density, 1;
                    cartesian_spatial_sampling = do_cartesian_sampling,
                )
                x_matrix_local[:, end] .= x1[:, 1]
                p_matrix_local[:, end] .= p1[:, 1]
            end

            x_matrix_local, p_matrix_local
        else
            sample_particles_from_FONLL(
                r_grid_Langevin, p_grid_Langevin, heavy_quark_density, N_particles;
                cartesian_spatial_sampling = do_cartesian_sampling,
            )
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

            # At r≈0 the radial direction ê_r is undefined. Avoid p_r=(x·p)/r.
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

            # p_r = p⃗ · ê_r
            p_r_samples = ex .* p_matrix[1, :] .+ ey .* p_matrix[2, :]

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
            VelocityEvolution, m, N_particles, 0, Δt, initial_time, radial_mode, use_v2, V2Evolution, psi2,
            relativistic)

        # === Allocate history arrays as 2D to avoid CuDeviceArray{3} in kernels ===
        # Layout: (dimensions, N_particles * (num_saves+1)), stride = N_particles per snapshot
        momenta_history_gpu  = CUDA.zeros(Float64, dimensions, N_particles * (num_saves + 1))
        momenta_history_gpu[:, 1:N_particles] .= momenta

        position_history_gpu = CUDA.zeros(Float64, dimensions, N_particles * (num_saves + 1))
        position_history_gpu[:, 1:N_particles] .= positions

        # === On-the-fly freeze-out capture arrays (N-length; the only thing saved) ===
        fo_pos_gpu  = freezeout_capture ? CUDA.zeros(Float64, dimensions, N_particles) : nothing
        fo_mom_gpu  = freezeout_capture ? CUDA.zeros(Float64, dimensions, N_particles) : nothing
        fo_tau_gpu  = freezeout_capture ? CUDA.zeros(Float64, N_particles) : nothing
        fo_flag_gpu = freezeout_capture ? CUDA.zeros(Float64, N_particles) : nothing
        # last-above cache for exact-crossing interpolation (last_T init 0 ⇒ any step-1 crossing books raw)
        last_pos_gpu = freezeout_capture ? CUDA.zeros(Float64, dimensions, N_particles) : nothing
        last_mom_gpu = freezeout_capture ? CUDA.zeros(Float64, dimensions, N_particles) : nothing
        last_tau_gpu = freezeout_capture ? CUDA.zeros(Float64, N_particles) : nothing
        last_T_gpu   = freezeout_capture ? CUDA.zeros(Float64, N_particles) : nothing

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
        # RTA/BGK pre-generated uniforms (collision decision + inverse-CDF |p*| draw); dir uses random_directions
        u_collide           = rta_mode ? CUDA.zeros(Float64, N_particles) : CUDA.zeros(Float64, 1)
        u_sample            = rta_mode ? CUDA.zeros(Float64, N_particles) : CUDA.zeros(Float64, 1)

        # === Langevin dynamics loop ===
        @showprogress 10 "Running Langevin simulation GPU Cartesian..." for step in 1:steps
            if rta_mode
                # RTA/BGK noise: two uniforms (collision decision + inverse-CDF |p*| draw) and a RAW
                # (un-normalized) Gaussian direction — the kernel normalizes it per particle.
                CUDA.rand!(u_collide)
                CUDA.rand!(u_sample)
                CUDA.randn!(random_directions)
                if position_diffusion
                    CUDA.randn!(ξ_position)
                end
            else
                CUDA.randn!(ξ_momentum)  # OU momentum noise — always needed
                # ξ_position is only consumed by the position-diffusion update; random_directions only by the
                # radial (1D) force kernel as a p≈0 fallback direction. Generating (and normalizing) them every
                # step when their consumer is off is pure waste — skip it. Behaviour-identical for the real
                # consumers; only the CURAND stream offset shifts (irrelevant to an ensemble average).
                if position_diffusion
                    CUDA.randn!(ξ_position)  # Sample Gaussian noise at every step
                end
                if radial_mode
                    CUDA.randn!(random_directions)  # Sample Gaussian noise at every step
                    # Normalize random direction vectors
                    random_norms = sqrt.(sum(random_directions .^ 2, dims=1))
                    random_directions ./= random_norms
                end
            end

            # Step 1: Transform momenta to local rest frame (LRF)
            @cuda threads=threads blocks=blocks kernel_boost_to_rest_frame_gpu!(
                momenta, positions, xgrid, tgrid, VelocityEvolution,
                m, N_particles, step, Δt, initial_time,radial_mode, use_v2, V2Evolution, psi2,
                relativistic)

            if !momentum_langevin || DsT == 0.0

                @cuda threads=threads blocks=blocks kernel_set_to_fluid_velocity_gpu!(
                    momenta, positions, xgrid, tgrid,
                    VelocityEvolution, m, N_particles, step, Δt, initial_time,radial_mode,
                    relativistic)
            elseif rta_mode
                # Boltzmann RTA / BGK: re-draw from the local Jüttner with prob Δt/τn in the LRF,
                # then boost back to lab. Skips the OU force/update kernels entirely.
                # 🔴 τn here is the CURRENT relaxation time (tau_n_main3), NOT the OU drag: BGK
                # relaxes every moment at 1/τ, so matching the OU's ℓ=1 (diffusion-current) decay
                # needs τ_n = τ_drag·K₃/K₂. The drag would relax it 1.26-1.74× too fast.
                @cuda threads=threads blocks=blocks kernel_rta_collision_gpu!(
                    momenta, positions, xgrid, tgrid, TemperatureEvolution,
                    Δt, m, N_particles, step, initial_time, DsT,
                    tau_Tmin, tau_invdT, taun_vals_d,
                    invcdf_d, invcdf_nU, invcdf_nT, invcdf_Tmin, invcdf_invdT,
                    dimensions, radial_mode,
                    u_collide, u_sample, random_directions)

                @cuda threads=threads blocks=blocks kernel_boost_to_lab_frame_gpu!(
                    momenta, positions, xgrid, tgrid, VelocityEvolution,
                    m, N_particles, step, Δt, initial_time,radial_mode, use_v2, V2Evolution, psi2,
                    relativistic)
            else
                # Step 2: Compute deterministic and stochastic terms in LRF
                @cuda threads=threads blocks=blocks kernel_compute_all_forces_gpu!(
                    TemperatureEvolution, xgrid, tgrid, momenta, positions,
                    p_mags, p_units, ηD_vals, kL_vals, kT_vals,
                    ξ_momentum, deterministic_terms, stochastic_terms,
                    Δt, m, random_directions, dimensions, N_particles,
                    step, initial_time, DsT,
                    tau_Tmin, tau_invdT, tau_vals_d,
                    radial_mode, Int32(integrator_mode), relativistic)

                # Step 3: Update momenta in LRF
                @cuda threads=threads blocks=blocks kernel_update_momenta_LRF_gpu!(
                    momenta, deterministic_terms, stochastic_terms,
                    Δt, dimensions, N_particles)

                # Step 4: Boost updated momenta back to lab frame
                @cuda threads=threads blocks=blocks kernel_boost_to_lab_frame_gpu!(
                    momenta, positions, xgrid, tgrid, VelocityEvolution,
                    m, N_particles, step, Δt, initial_time,radial_mode, use_v2, V2Evolution, psi2,
                    relativistic)
            end
            # Step 5: Update particle positions based on momenta
            # IMPORTANT: pass the current time-step, not the total number of steps.
            @cuda threads=threads blocks=blocks kernel_update_positions_gpu!(
                positions, momenta, m, Δt, N_particles, step, initial_time,
                xgrid, tgrid, TemperatureEvolution, DsT, dimensions, radial_mode, position_diffusion, reflecting_boundary, ξ_position,
                relativistic)

            # Step 5b: on-the-fly freeze-out latch (every step ⇒ crossing resolved to Δt; no history stored)
            if freezeout_capture
                @cuda threads=threads blocks=blocks kernel_capture_freezeout_gpu!(
                    fo_pos_gpu, fo_mom_gpu, fo_tau_gpu, fo_flag_gpu,
                    last_pos_gpu, last_mom_gpu, last_tau_gpu, last_T_gpu,
                    positions, momenta, xgrid, tgrid, TemperatureEvolution, Tfo,
                    N_particles, step, Δt, initial_time, dimensions, radial_mode, freezeout_interp)
            end

            # Step 6: Save state if necessary
            if step % save_every == 0
                save_idx = div(step, save_every) + 1
                @cuda threads=threads blocks=blocks kernel_save_momenta_gpu!(
                    momenta_history_gpu, momenta, save_idx, N_particles,dimensions)
                @cuda threads=threads blocks=blocks kernel_save_positions_gpu!(
                    position_history_gpu, positions, save_idx, N_particles,dimensions)
            end
        end

        # === On-the-fly freeze-out: transfer the N-length capture arrays and return (single save) ===
        if freezeout_capture
            fo = (pos=Array(fo_pos_gpu), mom=Array(fo_mom_gpu), tau=Array(fo_tau_gpu), flag=Array(fo_flag_gpu))
            finalize(fo_pos_gpu); finalize(fo_mom_gpu); finalize(fo_tau_gpu); finalize(fo_flag_gpu)
            finalize(last_pos_gpu); finalize(last_mom_gpu); finalize(last_tau_gpu); finalize(last_T_gpu)
            finalize(momenta_history_gpu); finalize(position_history_gpu)
            GC.gc(true); CUDA.reclaim()
            return fo
        end

        # === Transfer saved histories back to CPU ===
        # 2D layout: (dimensions, N_particles * (num_saves+1)) — single cudaMemcpy, slice on CPU.
        time_points = range(initial_time, final_time, length=num_saves + 1)
        full_mom_cpu = Array(momenta_history_gpu)
        full_pos_cpu = Array(position_history_gpu)
        momenta_history_vec  = [full_mom_cpu[:, (s-1)*N_particles+1 : s*N_particles] for s in 1:num_saves+1]
        position_history_vec = [full_pos_cpu[:, (s-1)*N_particles+1 : s*N_particles] for s in 1:num_saves+1]

        # Finalize all CuArrays (large history + position/momentum arrays must be freed explicitly)
        finalize(momenta_history_gpu)
        finalize(position_history_gpu)
        finalize(positions)
        finalize(momenta)
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
        finalize(u_collide)
        finalize(u_sample)
        finalize(invcdf_d)
        finalize(tau_vals_d)

        # Clear references
        momenta_history_gpu = nothing
        position_history_gpu = nothing
        positions = nothing
        momenta = nothing
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
