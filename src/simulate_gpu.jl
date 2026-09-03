module SimulateGPU

# === Imports ===
using ProgressMeter
using ..KernelsGPU
using ...Utils
using ...KernelsCPU: interpolate_2d_cpu, interpolate_3d_cpu   # host-side T lookups for the
                                                              # initial p_z draw (radial / 2-D)
using ...SimulateCPU: _snapshot_times, _step_count, _check_time_window, _warn_escaped_particles,
                      _validate_transport_inputs
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
    DsT_quad::Bool=false,
    DsT_Tref::Float64=0.0,
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
    momentum_dimensions::Int = 0,   # 0 ⇒ = dimensions; 3 with dimensions=2 adds p_z (utils.jl note)
    bjorken_redshift::Bool = false, # dp_z/dτ = −p_z/τ between kicks; needs momentum_dimensions=3
    proper_time_kicks::Bool = false, # OU kick per proper time Δt·E*/E_lab (see kernels_cpu.jl note)
    pz_init::Symbol = :thermal,     # :thermal (shipped) | :comoving (p_z* = 0); see append_pz
    track_eta_s::Bool = false,      # accumulate η_s and return its history as a FOURTH element
    verbose::Bool = false,          # print device name + memory status at entry
)
    CUDA.reclaim()  # Free any unused GPU memory
    verbose && print_cuda_status()

    try
        # === Derived parameters ===
        # 🔴 2026-09-02: identical to the CPU path — refuse the inputs that used to degrade to
        # silent free streaming, warn when the window leaves the table, and count steps with the
        # ulp-tolerant rule instead of a bare floor(). See the helpers in simulate_cpu.jl.
        _validate_transport_inputs(m, DsT)
        _check_time_window(initial_time, final_time, SpaceTimeGrid[end])
        total_time = final_time - initial_time
        steps = _step_count(initial_time, final_time, Δt)
        steps >= 1 || error("simulate_ensemble_bulk: final_time − initial_time = $total_time is shorter than Δt = $Δt (no step to take)")
        save_every = min(round(Int, save_interval / Δt), steps)   # robust to binary roundoff; clamped as on the CPU path
        save_every >= 1 || error("simulate_ensemble_bulk: save_interval = $save_interval is shorter than Δt = $Δt")
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
        # Geometry is dispatched on the DATA, exactly as on the CPU path: a 2-tuple grid with a
        # matrix T is the shipped radial background; a 3-tuple with T[x,y,t] and a (u^x, u^y) pair
        # is a genuine function of the transverse plane. The contract is validated here, on the
        # host, so a shape error is a clear message rather than a kernel launch failure.
        local xgrid, ygrid, tgrid, VelocityEvolution, VxField, VyField
        if length(SpaceTimeGrid) == 3
            xgridd, ygridd, ttgrid = SpaceTimeGrid
            ndims(TemperatureEvolutionn) == 3 || error(
                "simulate_ensemble_bulk(GPU): a 3-element SpaceTimeGrid selects the 2-D background, " *
                "so the temperature table must be T[x, y, t] (got ndims = $(ndims(TemperatureEvolutionn)))")
            (VelocityEvolutionn isa Union{Tuple,AbstractVector} && length(VelocityEvolutionn) == 2) || error(
                "simulate_ensemble_bulk(GPU): the 2-D background needs the flow as a VECTOR, " *
                "i.e. (u_x, u_y) tables of size (nx, ny, nt); got $(typeof(VelocityEvolutionn))")
            V2Evolutionn === nothing || error(
                "simulate_ensemble_bulk(GPU): V2Evolution is an ad-hoc stand-in for exactly the " *
                "anisotropy a 2-D background carries properly; using both would count it twice")
            vx_h, vy_h = VelocityEvolutionn
            size(vx_h) == size(vy_h) == size(TemperatureEvolutionn) || error(
                "simulate_ensemble_bulk(GPU): T, u_x and u_y must share the table shape")
            (length(xgridd), length(ygridd), length(ttgrid)) == size(TemperatureEvolutionn) || error(
                "simulate_ensemble_bulk(GPU): grid lengths do not match the table shape")
            xgrid = CuArray(xgridd); ygrid = CuArray(ygridd); tgrid = CuArray(ttgrid)
            TemperatureEvolution = CuArray(TemperatureEvolutionn)
            VxField = CuArray(vx_h); VyField = CuArray(vy_h)
            # the scalar flow table is absent on this path; the magnitude is derived from the pair
            VelocityEvolution = CuArray(zeros(Float64, 1, 1))
        else
            xgridd, ttgrid = SpaceTimeGrid
            xgrid = CuArray(xgridd)
            ygrid = nothing
            tgrid = CuArray(ttgrid)
            TemperatureEvolution = CuArray(TemperatureEvolutionn)
            VelocityEvolution = CuArray(VelocityEvolutionn)
            VxField = nothing; VyField = nothing
        end
        # optional elliptic-flow modulation: upload the v2(r,t) field, or a dummy when off
        use_v2 = V2Evolutionn !== nothing
        V2Evolution = use_v2 ? CuArray(V2Evolutionn) : CuArray(zeros(Float64, 1, 1))

        # === Precompute the DRAG spline τ_drag(T) on CPU and upload to GPU ===
        tau_Tmin::Float64 = 0.0
        tau_invdT::Float64 = 1.0
        tau_vals = Float64[0.0, 0.0]
        # RTA/BGK needs the CURRENT time instead — see build_taun_current_spline.
        taun_vals = Float64[0.0, 0.0]
        if momentum_langevin && DsT > 0.0 && collision_mode !== :none
            Tmin = max(float(minimum(TemperatureEvolutionn)), 0.0)
            Tmax = max(float(maximum(TemperatureEvolutionn)), Tmin + eps(Float64))
            tau_Tmin, tau_invdT, tau_vals = build_tau_drag_spline(m, DsT;
                Tmin = Tmin, Tmax = Tmax, nT = 1024,
                DsT_linear = DsT_linear,
                DsT_slope = DsT_slope,
                DsT_offset = DsT_offset,
                Tfo = Tfo,
                DsT_quad = DsT_quad,
                DsT_Tref = DsT_Tref)
            if collision_mode == :rta
                _, _, taun_vals = build_taun_current_spline(m, DsT;
                    Tmin = Tmin, Tmax = Tmax, nT = 1024,
                    DsT_linear = DsT_linear,
                    DsT_slope = DsT_slope,
                    DsT_offset = DsT_offset,
                    Tfo = Tfo,
                    DsT_quad = DsT_quad,
                    DsT_Tref = DsT_Tref)
            end
        end
        tau_vals_d = CuArray(tau_vals)
        taun_vals_d = CuArray(taun_vals)

        # === RTA/BGK: precompute the isotropic Jüttner inverse-CDF table on CPU, upload to GPU ===
        # Divergence-free `p*` sampler for kernel_rta_collision_gpu! (replaces the CPU rejection loop).
        rta_mode = collision_mode == :rta
        # `:none` = FREE STREAMING (2026-09-02); the twin of the CPU flag. See simulate_cpu.jl.
        free_stream = collision_mode === :none
        invcdf_nU::Int = 2
        invcdf_nT::Int = 2
        invcdf_Tmin::Float64 = 0.0
        invcdf_invdT::Float64 = 1.0
        invcdf_d = CuArray(Float64[0.0, 0.0, 0.0, 0.0])
        # momentum rows: the RTA |p*| law is p^(pdim−1)e^{−E/T}, so the table must use pdim
        pdim = momentum_dimensions <= 0 ? dimensions : momentum_dimensions
        if rta_mode && momentum_langevin && DsT > 0.0
            Tmin_ic = max(float(minimum(TemperatureEvolutionn)), 0.0)
            Tmax_ic = max(float(maximum(TemperatureEvolutionn)), Tmin_ic + eps(Float64))
            invcdf_vals, invcdf_nU, invcdf_nT, invcdf_Tmin, invcdf_invdT =
                build_juttner_invcdf(m, pdim; Tmin = Tmin_ic, Tmax = Tmax_ic,
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

        # --- momentum dimensionality (p_z on the transverse plane), completed on the host ---
        check_momentum_dims(dimensions, pdim, radial_mode, bjorken_redshift, initial_time;
                            pz_init = pz_init, track_eta_s = track_eta_s)
        track_eta_s && freezeout_capture &&
            error("simulate_ensemble_bulk_gpu: track_eta_s and freezeout_capture cannot be combined (the capture path returns the crossing state, not histories).")
        if size(moment, 1) < pdim
            # On a 2-D background the local temperature is not a function of the radius, so the
            # p_z completion is handed the particle's own (x, y). Mirrors the CPU driver; without
            # it the radial closure indexes a 3-D table with two indices and throws.
            moment = append_pz(pz_init, moment, position, m,
                ygrid === nothing ?
                    (r -> interpolate_2d_cpu(xgridd, ttgrid, TemperatureEvolutionn, r, initial_time)) :
                    (r -> error("unreachable: T_of_xy is supplied on a 2-D background"));
                antithetic = antithetic_momenta && x_init === nothing,
                T_of_xy = ygrid === nothing ? nothing :
                    ((x, y) -> interpolate_3d_cpu(xgridd, ygridd, ttgrid,
                                                  TemperatureEvolutionn, x, y, initial_time)))
        end
        size(moment, 1) == pdim || error("momentum rows $(size(moment, 1)) ≠ momentum_dimensions $pdim")

        positions = CuArray(position)
        momenta = CuArray(moment)

        # === Boost initial momenta into lab frame ===
        # Keep step indexing consistent with the CPU backend (t = t0 at initialization).
        @cuda threads=threads blocks=blocks kernel_boost_to_lab_frame_gpu!(
            momenta, positions, xgrid, tgrid,
            VelocityEvolution, m, N_particles, 0, Δt, initial_time, radial_mode, use_v2, V2Evolution, psi2,
            relativistic, ygrid, VxField, VyField)

        # === Allocate history arrays as 2D to avoid CuDeviceArray{3} in kernels ===
        # Layout: (dimensions, N_particles * (num_saves+1)), stride = N_particles per snapshot
        momenta_history_gpu  = CUDA.zeros(Float64, pdim, N_particles * (num_saves + 1))
        momenta_history_gpu[:, 1:N_particles] .= momenta

        position_history_gpu = CUDA.zeros(Float64, dimensions, N_particles * (num_saves + 1))
        position_history_gpu[:, 1:N_particles] .= positions

        # η_s starts at 0 for every particle ON PURPOSE — the engine accumulates only the CHANGE;
        # η_s(τ0) is applied afterwards by convolution (see kernel_accumulate_eta_s_cpu!).
        eta_s_gpu         = track_eta_s ? CUDA.zeros(Float64, N_particles) : CUDA.zeros(Float64, 1)
        eta_s_history_gpu = track_eta_s ? CUDA.zeros(Float64, N_particles * (num_saves + 1)) : CUDA.zeros(Float64, 1)

        # === On-the-fly freeze-out capture arrays (N-length; the only thing saved) ===
        fo_pos_gpu  = freezeout_capture ? CUDA.zeros(Float64, dimensions, N_particles) : nothing
        fo_mom_gpu  = freezeout_capture ? CUDA.zeros(Float64, pdim, N_particles) : nothing
        fo_tau_gpu  = freezeout_capture ? CUDA.zeros(Float64, N_particles) : nothing
        fo_flag_gpu = freezeout_capture ? CUDA.zeros(Float64, N_particles) : nothing
        # last-above cache for exact-crossing interpolation (last_T init 0 ⇒ any step-1 crossing books raw)
        last_pos_gpu = freezeout_capture ? CUDA.zeros(Float64, dimensions, N_particles) : nothing
        last_mom_gpu = freezeout_capture ? CUDA.zeros(Float64, pdim, N_particles) : nothing
        last_tau_gpu = freezeout_capture ? CUDA.zeros(Float64, N_particles) : nothing
        last_T_gpu   = freezeout_capture ? CUDA.zeros(Float64, N_particles) : nothing

        # === Allocate working buffers ===
        p_mags              = CUDA.zeros(Float64, N_particles)
        p_units             = CUDA.zeros(Float64, pdim, N_particles)
        ηD_vals             = CUDA.zeros(Float64, N_particles)
        kL_vals             = CUDA.zeros(Float64, N_particles)
        kT_vals             = CUDA.zeros(Float64, N_particles)
        stochastic_terms    = CUDA.zeros(Float64, pdim, N_particles)
        deterministic_terms = CUDA.zeros(Float64, pdim, N_particles)
        ξ_momentum          = CUDA.zeros(Float64, pdim, N_particles)
        ξ_position          = CUDA.zeros(Float64, dimensions, N_particles)  # position diffusion (spatial rows)
        random_directions   = CUDA.zeros(Float64, pdim, N_particles)        # momentum-space directions
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
            elseif !free_stream
                CUDA.randn!(ξ_momentum)  # OU momentum noise — always needed (not under `:none`)
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

            # Step 1: Transform momenta to local rest frame (LRF). Skipped under `:none`, where the
            # lab momenta are constant and the boost pair would only apply the γ regularisation's
            # ≈1e-10 contraction per step (see the CPU twin).
            if !free_stream
                @cuda threads=threads blocks=blocks kernel_boost_to_rest_frame_gpu!(
                    momenta, positions, xgrid, tgrid, VelocityEvolution,
                    m, N_particles, step, Δt, initial_time,radial_mode, use_v2, V2Evolution, psi2,
                    relativistic, ygrid, VxField, VyField)
            end

            # Step 1b: Bjorken redshift of p_z in the LRF (invariant under the transverse boost)
            if bjorken_redshift
                @cuda threads=threads blocks=blocks kernel_bjorken_redshift_gpu!(
                    momenta, 3, step, Δt, initial_time, N_particles)
            end

            if free_stream
                # nothing to do: the momenta are already the (constant) lab momenta

            elseif !momentum_langevin || DsT == 0.0

                @cuda threads=threads blocks=blocks kernel_set_to_fluid_velocity_gpu!(
                    momenta, positions, xgrid, tgrid,
                    VelocityEvolution, m, N_particles, step, Δt, initial_time,radial_mode,
                    relativistic, ygrid, VxField, VyField)
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
                    pdim, radial_mode,
                    u_collide, u_sample, random_directions,
                    proper_time_kicks, VelocityEvolution, ygrid, VxField, VyField)

                @cuda threads=threads blocks=blocks kernel_boost_to_lab_frame_gpu!(
                    momenta, positions, xgrid, tgrid, VelocityEvolution,
                    m, N_particles, step, Δt, initial_time,radial_mode, use_v2, V2Evolution, psi2,
                    relativistic, ygrid, VxField, VyField)
            else
                # Step 2: Compute deterministic and stochastic terms in LRF
                @cuda threads=threads blocks=blocks kernel_compute_all_forces_gpu!(
                    TemperatureEvolution, xgrid, tgrid, momenta, positions,
                    p_mags, p_units, ηD_vals, kL_vals, kT_vals,
                    ξ_momentum, deterministic_terms, stochastic_terms,
                    Δt, m, random_directions, pdim, N_particles,
                    step, initial_time, DsT,
                    tau_Tmin, tau_invdT, tau_vals_d,
                    radial_mode, Int32(integrator_mode), relativistic,
                    proper_time_kicks, VelocityEvolution, ygrid, VxField, VyField)

                # Step 3: Update momenta in LRF
                @cuda threads=threads blocks=blocks kernel_update_momenta_LRF_gpu!(
                    momenta, deterministic_terms, stochastic_terms,
                    Δt, pdim, N_particles)

                # Step 4: Boost updated momenta back to lab frame
                @cuda threads=threads blocks=blocks kernel_boost_to_lab_frame_gpu!(
                    momenta, positions, xgrid, tgrid, VelocityEvolution,
                    m, N_particles, step, Δt, initial_time,radial_mode, use_v2, V2Evolution, psi2,
                    relativistic, ygrid, VxField, VyField)
            end
            # Step 5: Update particle positions based on momenta
            # IMPORTANT: pass the current time-step, not the total number of steps.
            @cuda threads=threads blocks=blocks kernel_update_positions_gpu!(
                positions, momenta, m, Δt, N_particles, step, initial_time,
                xgrid, tgrid, TemperatureEvolution, DsT, dimensions, radial_mode, position_diffusion, reflecting_boundary, ξ_position,
                relativistic, pdim, ygrid, VxField, VyField)

            # Step 5a2: spacetime rapidity — the third position row (kernels_cpu.jl note).
            if track_eta_s
                τa_eta = initial_time + (step - 1) * Δt
                if τa_eta > 0
                    @cuda threads=threads blocks=blocks kernel_accumulate_eta_s_gpu!(
                        eta_s_gpu, momenta, m, log((τa_eta + Δt) / τa_eta), N_particles, pdim)
                end
            end

            # Step 5b: on-the-fly freeze-out latch (every step ⇒ crossing resolved to Δt; no history stored)
            if freezeout_capture
                @cuda threads=threads blocks=blocks kernel_capture_freezeout_gpu!(
                    fo_pos_gpu, fo_mom_gpu, fo_tau_gpu, fo_flag_gpu,
                    last_pos_gpu, last_mom_gpu, last_tau_gpu, last_T_gpu,
                    positions, momenta, xgrid, tgrid, TemperatureEvolution, Tfo,
                    N_particles, step, Δt, initial_time, dimensions, radial_mode, freezeout_interp, ygrid, VxField, VyField)
            end

            # Step 6: Save state if necessary
            if step % save_every == 0
                save_idx = div(step, save_every) + 1
                @cuda threads=threads blocks=blocks kernel_save_momenta_gpu!(
                    momenta_history_gpu, momenta, save_idx, N_particles,pdim)
                @cuda threads=threads blocks=blocks kernel_save_positions_gpu!(
                    position_history_gpu, positions, save_idx, N_particles,dimensions)
                if track_eta_s
                    @inbounds eta_s_history_gpu[(save_idx-1)*N_particles+1 : save_idx*N_particles] .= eta_s_gpu
                end
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

        # 🔴 2026-09-02: warn if particles finished outside the tabulated radial axis (CPU twin).
        _warn_escaped_particles(Array(positions), xgridd)

        # === Transfer saved histories back to CPU ===
        # 2D layout: (dimensions, N_particles * (num_saves+1)) — single cudaMemcpy, slice on CPU.
        time_points = _snapshot_times(initial_time, final_time, Δt, steps, save_every, num_saves)   # see simulate_cpu.jl
        full_mom_cpu = Array(momenta_history_gpu)
        full_pos_cpu = Array(position_history_gpu)
        momenta_history_vec  = [full_mom_cpu[:, (s-1)*N_particles+1 : s*N_particles] for s in 1:num_saves+1]
        position_history_vec = [full_pos_cpu[:, (s-1)*N_particles+1 : s*N_particles] for s in 1:num_saves+1]
        eta_s_history_vec = if track_eta_s
            full_eta_cpu = Array(eta_s_history_gpu)
            [full_eta_cpu[(s-1)*N_particles+1 : s*N_particles] for s in 1:num_saves+1]
        else
            nothing
        end
        finalize(eta_s_gpu); finalize(eta_s_history_gpu)

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
        finalize(taun_vals_d)

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

        if track_eta_s
            return time_points, momenta_history_vec, position_history_vec, eta_s_history_vec
        end
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
