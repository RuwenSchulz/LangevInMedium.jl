module SimulateCPU

# === Imports ===
using ProgressMeter
using Random: randn!
using ..KernelsCPU
using ..Utils
using ..Transport

# === Exported Symbols ===
export simulate_ensemble_bulk_cpu, _snapshot_times, _step_count, _check_time_window,
       _warn_escaped_particles, _validate_transport_inputs

"""
    _step_count(t0, tf, Δt) -> Int

Number of steps in `[t0, tf]`, immune to the binary representation of the endpoints.

🔴 FIXED 2026-09-02. This was a bare `floor(Int, (tf − t0)/Δt)`. When the difference is not exactly
representable — `1.4 − 0.4 = 0.9999999999999999` — the quotient lands a fraction of an ulp BELOW the
integer and `floor` takes a whole step off. Usually that is one step of 10⁻³ fm and nobody notices;
the damage is that it also breaks `steps % save_every == 0`, and then `_snapshot_times` drops the
entire trailing save interval and blames `save_interval` for "not dividing the evolution".
MEASURED worst case: t0 = 0.4, tf = 1.4, Δt = 10⁻³, `save_interval` = 0.5 kept 501 of 1000 steps —
HALF the requested history, silently, with a warning pointing at the wrong cause.

The snap tolerance is 64 ulps of the quotient (≈ 9·10⁻¹⁰ at q = 1.26·10⁵), which is ~4 orders above
any representation error and ~3 below any shortfall a caller could mean. ⚠ It DOES change runs whose
window was previously mis-floored: AttractorHydro's portrait (0.4 → 13.0 at Δt = 10⁻⁴) goes
125 999 → 126 000 steps and 1260 → 1261 snapshots. LP1's 12.6 fm and O+O's 7.6 fm divide exactly and
are untouched.
"""
function _step_count(t0::Real, tf::Real, Δt::Real)
    q = (Float64(tf) - Float64(t0)) / Float64(Δt)
    isfinite(q) || return 0
    steps = floor(Int, q)
    if (steps + 1) - q <= 64 * eps(max(abs(q), 1.0))
        steps += 1
    end
    return steps
end

"""
    _check_time_window(t0, tf, tgrid)

Warn when the requested evolution leaves the tabulated time axis.

🔴 ADDED 2026-09-02. `interpolate_2d_*` clamps every query into the table — right for a particle
leaving the fireball rim, wrong for a run that outlives the hydro output. Past `tgrid[end]` the
medium FREEZES at its last tabulated slice and the run continues, with no error and no warning, on
both backends (measured: a table ending at τ = 2 integrated to τ = 6 kept evolving in the τ = 2
medium). A warning, not an error: freezing the medium is occasionally what a caller wants, and
AttractorMomentum's own driver already validates the window before calling.
"""
function _check_time_window(t0::Real, tf::Real, tgrid)
    lo = Float64(first(tgrid)); hi = Float64(last(tgrid))
    tol = 1e-9 * max(1.0, abs(hi - lo))
    if Float64(t0) < lo - tol || Float64(tf) > hi + tol
        @warn "simulate_ensemble_bulk: the requested window leaves the tabulated time axis — the background is CLAMPED (frozen) outside it, silently" requested = (Float64(t0), Float64(tf)) tabulated = (lo, hi)
    end
    return nothing
end

"""
    _warn_escaped_particles(positions, xgrid)

Warn if any particle finished outside the tabulated radial axis, where `T` and `v` are frozen at
their rim values — an effectively infinite fireball. Added 2026-09-02 with `_check_time_window`;
NO `maxlog`: both are once-per-RUN diagnostics, and `maxlog` is keyed by source location, so it
would silence every call after the first in a campaign that drives the engine many times;
measured on a table cut at r = 8 fm, 16.2 % of a 20 000-particle ensemble ended outside it, out to
r = 15.2 fm. One pass over the FINAL positions only, so it costs nothing per step.
"""
function _warn_escaped_particles(positions::AbstractMatrix, xgrid)
    N = size(positions, 2)
    N == 0 && return nothing
    rmax = Float64(last(xgrid))
    rmax > 0 || return nothing
    cnt = 0; rmaxseen = 0.0
    @inbounds for i in 1:N
        r2 = 0.0
        for d in 1:size(positions, 1)
            r2 += Float64(positions[d, i])^2
        end
        r = sqrt(r2)
        r > rmax && (cnt += 1)
        r > rmaxseen && (rmaxseen = r)
    end
    if cnt > 0
        @warn "simulate_ensemble_bulk: particles finished OUTSIDE the tabulated radial axis — they were dragged at the rim T and v, i.e. in an infinite fireball" escaped = cnt of_N = N fraction = round(cnt / N; digits = 4) r_max_reached = round(rmaxseen; digits = 3) xgrid_end = rmax
    end
    return nothing
end

"""
    _validate_transport_inputs(m, DsT)

🔴 ADDED 2026-09-02. `tau_drag` returns 0.0 for any non-positive `m`, `T` or `DsT`, and every kernel
reads `τ ≤ 0` as `η_D = κ = 0`. So a mistyped mass or a negative `D_sT` used to produce a
FREE-STREAMING run indistinguishable from a Langevin run except by its numbers — measured, `m = 0`,
`m = −1.5` and `DsT = −0.1` all left ⟨p²⟩ frozen to 1e-9, with nothing said. Both are now refused.

`DsT == 0` stays legal and is NOT free streaming: it is the comoving limit (`p = m·γ·v`, every
particle handed the fluid's momentum). Ask for free streaming with `collision_mode = :none`.
"""
function _validate_transport_inputs(m::Real, DsT::Real)
    (isfinite(m) && m > 0) ||
        error("simulate_ensemble_bulk: m = $m is not a positive mass. A non-positive mass makes tau_drag return 0, which every kernel reads as η_D = κ = 0 — the run would free-stream silently.")
    (isfinite(DsT) && DsT >= 0) ||
        error("simulate_ensemble_bulk: DsT = $DsT is negative. A negative D_sT makes tau_drag return 0, which every kernel reads as η_D = κ = 0 — the run would free-stream silently. For free streaming pass collision_mode = :none; for the comoving limit pass DsT = 0.")
    return nothing
end

"""
    _snapshot_times(t0, tf, Δt, steps, save_every, num_saves)

Times of the `num_saves + 1` stored snapshots. Snapshot k (k = 0…num_saves) is taken after step
k·save_every, i.e. at t0 + k·save_every·Δt. Returns the historical `range(t0, tf, length)` when
that is exact (steps divisible by save_every) so existing outputs are unchanged; otherwise the
true times, with a warning that the last `steps − num_saves·save_every` steps are not in the history.
"""
function _snapshot_times(t0, tf, Δt, steps, save_every, num_saves)
    if steps % save_every == 0
        return range(t0, tf, length = num_saves + 1)
    end
    dropped = steps - num_saves * save_every
    @warn "save_interval does not divide the evolution: the last $dropped step(s) ($(dropped * Δt) fm) are not in the returned history; time_points reflect the snapshots actually taken" maxlog = 1
    return range(t0, t0 + num_saves * save_every * Δt, length = num_saves + 1)
end

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
    DsT_quad::Bool = false,
    DsT_Tref::Float64 = 0.0,
    dimensions::Int = 3,
    cartesian_spatial_sampling::Union{Nothing,Bool} = nothing,
    antithetic_momenta::Bool = false,
    position_diffusion::Bool = false,
    momentum_langevin::Bool = true,
    reflecting_boundary::Bool = false,
    collision_mode::Symbol = :langevin,
    x_init::Union{Nothing, AbstractMatrix} = nothing,
    p_init::Union{Nothing, AbstractMatrix} = nothing,
    V2Evolutionn::Union{Nothing, AbstractMatrix} = nothing,
    psi2::Float64 = 0.0,
    relativistic::Bool = true,
    # 0 ⇒ momentum rows = `dimensions` (the coupled, bit-identical default). 3 with dimensions=2 adds
    # a longitudinal p_z to the transverse-plane run — see the note above append_thermal_pz in utils.jl.
    momentum_dimensions::Int = 0,
    # dp_z/dτ = −p_z/τ between kicks (Bjorken longitudinal free-streaming); needs momentum_dimensions=3.
    bjorken_redshift::Bool = false,
    # kick per the particle's proper time (Δt* = Δt·E*/E_lab) — removes the lab-simultaneity
    # ν^r artifact on flowing backgrounds; default false = production byte-identical.
    proper_time_kicks::Bool = false,
    # how the p_z row is initialised: :thermal (shipped; the local Jüttner conditional) or
    # :comoving (p_z* = 0, the free-streaming IC). See `append_pz` in utils.jl. Default keeps
    # every existing product bit-identical.
    pz_init::Symbol = :thermal,
    # accumulate the spacetime rapidity η_s (dη_s/dτ = (1/τ)(p_z*/E*)) and return its history as a
    # FOURTH element. η_s is a passenger — nothing reads it — so this cannot move any other number;
    # with it, y_lab = η_s + atanh(p_z*/E*). Needs momentum_dimensions = 3 and initial_time > 0.
    track_eta_s::Bool = false)

    # === Setup and Preallocation ===
    # 🔴 2026-09-02: refuse the inputs that used to degrade to silent free streaming, and say so
    # when the requested window leaves the background table (both are new; see the helpers above).
    _validate_transport_inputs(m, DsT)
    # SpaceTimeGrid[end], not [2]: on a 3-element (x, y, t) grid the second entry is the Y AXIS,
    # and checking the run window against it is silently vacuous whenever the box happens to
    # bracket the times.
    _check_time_window(initial_time, final_time, SpaceTimeGrid[end])
    total_time = final_time - initial_time
    steps = _step_count(initial_time, final_time, Δt)     # ulp-tolerant; was a bare floor()
    steps >= 1 || error("simulate_ensemble_bulk: final_time − initial_time = $total_time is shorter than Δt = $Δt (no step to take)")
    # save_interval == total time can give round(save/Δt) = floor(total/Δt) + 1 ⇒ num_saves = 0 and a
    # crash in `range(..., length = 1)`; clamp so that the last snapshot is always taken.
    save_every = min(round(Int, save_interval / Δt), steps)
    save_every >= 1 || error("simulate_ensemble_bulk: save_interval = $save_interval is shorter than Δt = $Δt")
    num_saves = div(steps, save_every)

    # `:none` = FREE STREAMING (2026-09-02). It exists because the tree had no way to ask for it:
    # `DsT = 0` is the COMOVING limit (p = m·γ·v) and `momentum_langevin = false` is the same limit,
    # while the only thing that actually free-streamed was a NEGATIVE DsT — by accident, through
    # `tau_drag ≤ 0 ⇒ η_D = κ = 0`, which is now refused. Three call sites in the tree asked for
    # "free streaming" and got the comoving limit; see README "The LIMITS and the INPUT CONTRACT".
    free_stream = collision_mode === :none

    # --- BACKGROUND GEOMETRY: dispatched on what was handed in, never on a flag ---
    # A 2-tuple (xgrid, tgrid) with a matrix T is the shipped RADIAL background. A 3-tuple
    # (xgrid, ygrid, tgrid) with a 3-D T and a (u^x, u^y) pair is a genuine function of the
    # transverse PLANE. Dispatching on the data rather than on a keyword means the radial path
    # cannot be reached by accident and the 2-D path cannot be half-selected: a 3-D table with a
    # 2-tuple grid is a shape error, not a silently different physics run.
    local xgrid, ygrid, tgrid, VxField, VyField
    if length(SpaceTimeGrid) == 3
        xgrid, ygrid, tgrid = SpaceTimeGrid
        ndims(TemperatureEvolutionn) == 3 || error(
            "simulate_ensemble_bulk: a 3-element SpaceTimeGrid selects the 2-D background, so the " *
            "temperature table must be T[x, y, t] (got ndims = $(ndims(TemperatureEvolutionn)))")
        (VelocityEvolutionn isa Union{Tuple,AbstractVector} && length(VelocityEvolutionn) == 2) || error(
            "simulate_ensemble_bulk: the 2-D background needs the flow as a VECTOR, i.e. " *
            "(u_x, u_y) tables of size (nx, ny, nt); got $(typeof(VelocityEvolutionn))")
        VxField, VyField = VelocityEvolutionn
        V2Evolutionn === nothing || error(
            "simulate_ensemble_bulk: V2Evolution is an ad-hoc stand-in for exactly the anisotropy " *
            "a 2-D background carries properly; using both would count it twice")
        size(VxField) == size(TemperatureEvolutionn) == size(VyField) || error(
            "simulate_ensemble_bulk: T, u_x and u_y must share the table shape " *
            "$(size(TemperatureEvolutionn)) / $(size(VxField)) / $(size(VyField))")
        (length(xgrid), length(ygrid), length(tgrid)) == size(TemperatureEvolutionn) || error(
            "simulate_ensemble_bulk: grid lengths $((length(xgrid), length(ygrid), length(tgrid))) " *
            "do not match the table shape $(size(TemperatureEvolutionn))")
    else
        xgrid, tgrid = SpaceTimeGrid
        ygrid = nothing; VxField = nothing; VyField = nothing
    end
    # the tables the kernels read for a SCALAR magnitude (temperature always; flow only for
    # proper_time_kicks) plus the vector pair, bundled once so no call site can pass a subset
    bg2d = (; ygrid = ygrid, VxField = VxField, VyField = VyField)
    # kernels that only need the flow MAGNITUDE (proper_time_kicks) take `Vfield`, a matrix.
    # On the 2-D path the magnitude is derived from the vector pair instead, so the scalar
    # table is absent rather than a tuple the kernel signature would reject.
    Vfield_scalar = ygrid === nothing ? VelocityEvolutionn : nothing

    # For `dimensions == 1` we still evolve a *radial* degree of freedom in the
    # transverse plane. Sampling directly in polar (r,φ) on the grid can create
    # small-r artifacts; instead we default to Cartesian (x,y) sampling and then
    # collapse to r = √(x²+y²) and p_r = p·ê_r.
    do_cartesian_sampling = cartesian_spatial_sampling === nothing ? (dimensions == 1 || dimensions >= 2) : cartesian_spatial_sampling

    # Initial sampling: use pre-sampled particles when x_init/p_init are provided
    # (e.g., anisotropic ICs with a radial momentum boost). Otherwise sample from
    # the density matrix, optionally with antithetic pairs for variance reduction.
    if x_init !== nothing && p_init !== nothing
        x_matrix = Matrix{Float64}(x_init)
        p_matrix = Matrix{Float64}(p_init)
    elseif antithetic_momenta
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

    # --- momentum dimensionality (p_z on the transverse plane) ---
    pdim = momentum_dimensions <= 0 ? dimensions : momentum_dimensions
    check_momentum_dims(dimensions, pdim, radial_mode, bjorken_redshift, initial_time;
                        pz_init = pz_init, track_eta_s = track_eta_s)
    if size(momenta, 1) < pdim
        # sampled momenta are LRF momenta at this point (the lab boost follows below), so both
        # completions are statements in the LOCAL REST FRAME: :thermal draws the local Jüttner
        # conditional at T(r, τ0), :comoving sets p_z* = 0 (free-streamed from the production
        # point). See `append_pz`.
        momenta = append_pz(pz_init, momenta, positions, m,
            ygrid === nothing ?
                (r -> interpolate_2d_cpu(xgrid, tgrid, TemperatureEvolutionn, r, initial_time)) :
                (r -> error("unreachable: T_of_xy is supplied on a 2-D background"));
            antithetic = antithetic_momenta && x_init === nothing,
            # on a 2-D background the local temperature is not a function of the radius, so the
            # p_z completion is handed the particle's own (x, y) instead
            T_of_xy = ygrid === nothing ? nothing :
                ((x, y) -> interpolate_3d_cpu(xgrid, ygrid, tgrid, TemperatureEvolutionn,
                                              x, y, initial_time)))
    end
    size(momenta, 1) == pdim || error("momentum rows $(size(momenta, 1)) ≠ momentum_dimensions $pdim")

    momenta_history = zeros(Float64, pdim, N_particles, num_saves + 1)
    position_history = zeros(Float64, dimensions, N_particles, num_saves + 1)
    # η_s starts at 0 for every particle ON PURPOSE: the engine accumulates only the CHANGE, and
    # the production value η_s(τ0) is applied afterwards by convolution (it cannot enter the
    # dynamics, so it must not enter the run).
    eta_s         = track_eta_s ? zeros(Float64, N_particles) : Float64[]
    eta_s_history = track_eta_s ? zeros(Float64, N_particles, num_saves + 1) : zeros(Float64, 0, 0)


    kernel_boost_to_lab_frame_cpu!(
    momenta, positions, xgrid, tgrid,
    VelocityEvolutionn, m, N_particles, 0, Δt, initial_time,radial_mode = radial_mode,
    V2Evolution = V2Evolutionn, psi2 = psi2, relativistic = relativistic, ygrid = bg2d.ygrid, VxField = bg2d.VxField, VyField = bg2d.VyField)

    momenta_history[:,:,1] .= momenta
    position_history[:, :, 1] .= positions

    # Working buffers
    p_mags              = zeros(N_particles)
    p_units             = zeros(pdim, N_particles)
    ηD_vals             = zeros(N_particles)
    kL_vals             = zeros(N_particles)
    kT_vals             = zeros(N_particles)
    deterministic_terms = zeros(pdim, N_particles)
    stochastic_terms    = zeros(pdim, N_particles)
    ξ                   = randn(pdim, N_particles)
    random_directions   = randn(pdim, N_particles)

    # Normalize random directions
    norm_factors = sqrt.(sum(random_directions .^ 2, dims=1))
    random_directions ./= norm_factors

    # === Precompute τn(T) spline (main3 logic) ===
    # Only needed when we actually run momentum Langevin with DsT > 0.
    tau_Tmin::Float64 = 0.0
    tau_invdT::Float64 = 1.0
    tau_vals = Float64[0.0, 0.0]
    # RTA/BGK needs the CURRENT time, not the drag — see build_taun_current_spline.
    taun_vals = Float64[0.0, 0.0]
    if momentum_langevin && DsT > 0.0 && !free_stream
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


    # === Langevin Time Evolution Loop ===
    @showprogress 10 "Running Langevin CPU simulation..." for step in 1:steps
        # ⚡ randn!(ξ) in place — `ξ .= randn(pdim, N)` allocated a fresh N×pdim matrix EVERY step.
        # Same RNG stream, same values, same order ⇒ bit-identical. Skipped under `:none`, whose
        # only consumer of randomness is the initial sampler.
        free_stream || randn!(ξ)

        # 1. Boost momenta to local rest frame.
        # `collision_mode = :none` is FREE STREAMING (added 2026-09-02): no drag, no noise, and no
        # frame change either — the lab momenta are constant, so the boost pair would be a round
        # trip whose only effect is the γ = 1/√(1−v²+1e-10) contraction (≈1e-10 per step). Skipping
        # it makes free streaming exact. Everything below is bit-identical when free_stream = false.
        if !free_stream
            kernel_boost_to_rest_frame_cpu!(
                momenta, positions, xgrid, tgrid,
                VelocityEvolutionn, m, N_particles, step, Δt, initial_time,radial_mode = radial_mode,
                V2Evolution = V2Evolutionn, psi2 = psi2, relativistic = relativistic, ygrid = bg2d.ygrid, VxField = bg2d.VxField, VyField = bg2d.VyField)
        end

        # 1b. Longitudinal redshift of p_z in the LRF (p_z is invariant under the transverse boost,
        # so it is the same operation with or without the boost pair — and dp_z/dτ = −p_z/τ IS the
        # longitudinal free-streaming law, so it belongs in the :none path too).
        if bjorken_redshift
            kernel_bjorken_redshift_cpu!(momenta, 3, step, Δt, initial_time, N_particles)
        end

        if free_stream
            # nothing to do: the momenta are already the (constant) lab momenta

        elseif !momentum_langevin || DsT == 0.0
            kernel_set_to_fluid_velocity_cpu!(
                momenta, positions,  xgrid, tgrid,
                VelocityEvolutionn, m, N_particles, step, Δt, initial_time,radial_mode = radial_mode,
                relativistic = relativistic, ygrid = bg2d.ygrid, VxField = bg2d.VxField, VyField = bg2d.VyField)
        elseif collision_mode == :rta
            # Boltzmann RTA / BGK: re-draw from the local Jüttner with prob Δt/τn.
            # 🔴 τn here is the CURRENT relaxation time (tau_n_main3), NOT the OU drag.
            # BGK relaxes every moment at 1/τ, so matching the OU's ℓ=1 decay rate — the
            # diffusion-current sector the papers compare — requires τ_n = τ_drag·K₃/K₂.
            # Passing the drag would relax the RTA K₃/K₂ (1.26-1.74×) too fast.
            kernel_rta_collision_cpu!(
                TemperatureEvolutionn, xgrid, tgrid,
                momenta, positions,
                Δt, m, N_particles, step, initial_time, DsT;
                tau_Tmin = tau_Tmin, tau_invdT = tau_invdT, tau_vals = taun_vals,
                dimensions = pdim, radial_mode = radial_mode,
                proper_time_kicks = proper_time_kicks, Vfield = Vfield_scalar, ygrid = bg2d.ygrid, VxField = bg2d.VxField, VyField = bg2d.VyField)

            # Boost updated momenta back to lab frame
            kernel_boost_to_lab_frame_cpu!(
                momenta, positions, xgrid, tgrid,
                VelocityEvolutionn, m, N_particles, step, Δt, initial_time,radial_mode = radial_mode,
                V2Evolution = V2Evolutionn, psi2 = psi2, relativistic = relativistic, ygrid = bg2d.ygrid, VxField = bg2d.VxField, VyField = bg2d.VyField)
        else

            # 2. Compute forces in rest frame
            kernel_compute_all_forces_cpu!(
                TemperatureEvolutionn, xgrid, tgrid,
                momenta, positions, p_mags, p_units,
                ηD_vals, kL_vals, kT_vals,
                ξ, deterministic_terms, stochastic_terms,
                Δt, m, random_directions,
                pdim, N_particles, step, initial_time,DsT,
                tau_Tmin = tau_Tmin,
                tau_invdT = tau_invdT,
                tau_vals = tau_vals,
                radial_mode = radial_mode,
                relativistic = relativistic,
                proper_time_kicks = proper_time_kicks,
                Vfield = Vfield_scalar, ygrid = bg2d.ygrid, VxField = bg2d.VxField, VyField = bg2d.VyField)

            # 3. Update momenta
            kernel_update_momenta_LRF_cpu!(
                momenta, deterministic_terms, stochastic_terms,
                Δt, pdim, N_particles)

            # 4. Boost updated momenta back to lab frame
            kernel_boost_to_lab_frame_cpu!(
                momenta, positions, xgrid, tgrid,
                VelocityEvolutionn, m, N_particles, step, Δt, initial_time,radial_mode = radial_mode,
                V2Evolution = V2Evolutionn, psi2 = psi2, relativistic = relativistic, ygrid = bg2d.ygrid, VxField = bg2d.VxField, VyField = bg2d.VyField)
        end
        # 5. Update positions
       
        kernel_update_positions_cpu!(
                    positions, momenta, m, Δt, N_particles,step,initial_time,
                    xgrid,tgrid, TemperatureEvolutionn,DsT;
                    dimensions,
                    momentum_dimensions = pdim,
                    radial_mode = radial_mode,
                    position_diffusion = position_diffusion,
                    reflecting_boundary = reflecting_boundary,
                    relativistic = relativistic, ygrid = bg2d.ygrid, VxField = bg2d.VxField, VyField = bg2d.VyField
                )


        # 5b. Spacetime rapidity: the third position row (see kernel_accumulate_eta_s_cpu!).
        # The 1/τ is integrated exactly across [τ_a, τ_b]; the log is hoisted out of the loop.
        if track_eta_s
            τa_eta = initial_time + (step - 1) * Δt
            if τa_eta > 0
                kernel_accumulate_eta_s_cpu!(eta_s, momenta, m, log((τa_eta + Δt) / τa_eta),
                                             N_particles, pdim)
            end
        end

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

            if track_eta_s
                @inbounds eta_s_history[:, save_idx] .= eta_s
            end
        end

    end

    # 🔴 2026-09-02: say so if particles finished outside the tabulated radial axis, where T and v
    # are frozen at their rim values. One pass over the final positions; nothing per step.
    _warn_escaped_particles(positions, xgrid)

    # === Final Data Packaging ===
    # The saved snapshots sit at t0 + k·save_every·Δt. When `steps` is not a multiple of `save_every`
    # the trailing partial interval is never saved, and claiming `final_time` for the last snapshot
    # stretches the whole axis (a 13 % error in an MSD slope was traced to exactly this, 2026-08-21).
    # Bit-identical to the old `range(t0, final_time, ...)` whenever the old axis was right.
    time_points = _snapshot_times(initial_time, final_time, Δt, steps, save_every, num_saves)
    position_history_vec = [position_history[:, :, i] for i in 1:size(position_history, 3)]
    momenta_history_vec  = [momenta_history[:, :, i] for i in 1:size(momenta_history, 3)]
    if track_eta_s
        eta_s_history_vec = [eta_s_history[:, i] for i in 1:size(eta_s_history, 2)]
        return time_points, momenta_history_vec, position_history_vec, eta_s_history_vec
    end
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
    save_every = round(Int, save_interval / Δt)
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
        randn!(ξ)          # in place; see the note in the field-driven method above

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
    time_points = _snapshot_times(initial_time, final_time, Δt, steps, save_every, num_saves)

    return time_points, momenta_history
end

end # module SimulateCPU
