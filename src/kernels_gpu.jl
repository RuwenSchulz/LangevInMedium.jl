module KernelsGPU

using CUDA
using StaticArrays
using ...Constants: fmGeV

# === Exported Symbols ===
export 
    kernel_boost_to_rest_frame_gpu!,
    kernel_boost_to_lab_frame_gpu!,
    kernel_compute_all_forces_gpu!,
    kernel_update_momenta_LRF_gpu!,
    kernel_update_positions_gpu!,
    kernel_set_to_fluid_velocity_gpu!,
    kernel_save_snapshot_gpu!,
    kernel_save_momenta_gpu!,
    kernel_save_positions_gpu!,
    kernel_capture_freezeout_gpu!,
    kernel_rta_collision_gpu!,
    interpolate_2d_cuda

# ---------------------------------------------------------------------------
# τn(T) spline evaluation (uniform-grid linear interpolation)
# ---------------------------------------------------------------------------
@inline function _eval_tau_n_spline_cuda(T::Float64, Tmin::Float64, invdT::Float64, tau_vals)
    n = length(tau_vals)
    n < 2 && return Float64(tau_vals[1])
    if !isfinite(T)
        return 0.0
    end
    u = (T - Tmin) * invdT
    i = Int(floor(u)) + 1
    i = clamp(i, 1, n - 1)
    t = u - (i - 1)
    y0 = Float64(tau_vals[i])
    y1 = Float64(tau_vals[i + 1])
    return (1.0 - t) * y0 + t * y1
end

# ============================================================================
# GPU Utility Function: Bilinear Interpolation
# ============================================================================

@inline function interpolate_2d_cuda(x, y, values, xi, yi)
    i = 1
    for k in 1:length(x) - 1
        if x[k + 1] <= xi
            i += 1
        else
            break
        end
    end

    j = 1
    for k in 1:length(y) - 1
        if y[k + 1] <= yi
            j += 1
        else
            break
        end
    end

    i = clamp(i, 1, length(x) - 1)
    j = clamp(j, 1, length(y) - 1)

    x0, x1 = x[i], x[i+1]
    y0, y1 = y[j], y[j+1]

    v00, v10 = values[i, j], values[i+1, j]
    v01, v11 = values[i, j+1], values[i+1, j+1]

    xd = (xi - x0) / (x1 - x0 + 1e-8)
    yd = (yi - y0) / (y1 - y0 + 1e-8)

    c0 = v00 * (1 - xd) + v10 * xd
    c1 = v01 * (1 - xd) + v11 * xd

    return c0 * (1 - yd) + c1 * yd
end

# ---------------------------------------------------------------------------
# Jüttner inverse-CDF table lookup (bilinear in (u, T)) — divergence-free
# `p*` draw for the GPU RTA collision. `invCDF` is the FLAT column-major table
# from `Transport.build_juttner_invcdf`: invCDF[(jT-1)*nU + iu] = p*(u_iu; T_jT).
# u ∈ [0,1] (uniform-index axis), T ∈ [Ttab_min, Ttab_min+(nT-1)/invdT].
# ---------------------------------------------------------------------------
@inline function _juttner_invcdf_lookup(
    invCDF, nU::Int, nT::Int, Ttab_min::Float64, Ttab_invdT::Float64,
    u::Float64, T::Float64,
    )
    # fractional u-index in [0, nU-1]
    uu = u < 0.0 ? 0.0 : (u > 1.0 ? 1.0 : u)
    fu = uu * (nU - 1)
    iu = Int(floor(fu))
    iu = iu < 0 ? 0 : (iu > nU - 2 ? nU - 2 : iu)
    tu = fu - iu

    # fractional T-index in [0, nT-1]
    fT = (T - Ttab_min) * Ttab_invdT
    jT = Int(floor(fT))
    jT = jT < 0 ? 0 : (jT > nT - 2 ? nT - 2 : jT)
    tT = fT - jT
    tT = tT < 0.0 ? 0.0 : (tT > 1.0 ? 1.0 : tT)

    # flat indices (1-based); rows = u (nU), cols = T
    b0 = jT * nU
    b1 = (jT + 1) * nU
    @inbounds v00 = invCDF[b0 + iu + 1]
    @inbounds v10 = invCDF[b0 + iu + 2]
    @inbounds v01 = invCDF[b1 + iu + 1]
    @inbounds v11 = invCDF[b1 + iu + 2]

    c0 = v00 * (1.0 - tu) + v10 * tu
    c1 = v01 * (1.0 - tu) + v11 * tu
    return c0 * (1.0 - tT) + c1 * tT
end

# ============================================================================
# CUDA Kernels for Langevin Evolution
# ============================================================================

@inline function kernel_boost_to_rest_frame_gpu!(
    momenta, positions, xgrid, tgrid, VelocityEvolution,
    m::Float64, N_particles::Int, steps, Δt, initial_time,
    radial_mode::Bool, use_v2::Bool, V2Evolution, psi2::Float64
    )
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= N_particles
        # --- compute current proper time ---
        t_now = steps * Δt + initial_time

        # --- compute radius r ---
        r2 = 0.0
        for d in 1:size(positions, 1)
            r2 += positions[d, i]^2
        end
        r = sqrt(r2)
        if r < eps(Float64)
            return
        end

        # --- local radial fluid velocity ---
        v = interpolate_2d_cuda(xgrid, tgrid, VelocityEvolution, r, t_now)
        # optional m=2 azimuthal modulation → elliptic flow v2 (use_v2=false ⇒ unchanged)
        if use_v2 && size(positions, 1) >= 2
            v2m = interpolate_2d_cuda(xgrid, tgrid, V2Evolution, r, t_now)
            φ = atan(positions[2, i], positions[1, i])
            v *= (1.0 + 2.0 * v2m * cos(2.0 * (φ - psi2)))
        end
        v2 = v * v
        γ = 1.0 / sqrt(1.0 - v2 + 1e-10)

        # --- particle energy ---
        p2 = 0.0
        for d in 1:size(momenta, 1)
            p2 += momenta[d, i]^2
        end
        E = sqrt(p2 + m^2)

        if radial_mode
            # ==================================
            # 🟣 Radial mode (1D momentum)
            # ==================================
            @inbounds momenta[1, i] = γ * (momenta[1, i] - v * E)

        else
            # ==================================
            # 🟢 Cartesian mode (2D or 3D)
            # ==================================
            # Extract coordinates
            x = positions[1, i]
            y = positions[2, i]
            rhatx, rhaty = x / r, y / r

            # Decompose momentum into parallel/perpendicular
            px, py = momenta[1, i], momenta[2, i]
            p_parallel = px * rhatx + py * rhaty
            p_perp_x = px - p_parallel * rhatx
            p_perp_y = py - p_parallel * rhaty

            # Lorentz boost along radial direction
            p_parallel_new = γ * (p_parallel - v * E)

            # Reconstruct boosted momentum components
            @inbounds momenta[1, i] = p_parallel_new * rhatx + p_perp_x
            @inbounds momenta[2, i] = p_parallel_new * rhaty + p_perp_y
        end
    end
    return
end

@inline function kernel_boost_to_lab_frame_gpu!(
    momenta, positions, xgrid, tgrid, VelocityEvolution,
    m::Float64, N_particles::Int, steps, Δt, initial_time,
    radial_mode::Bool, use_v2::Bool, V2Evolution, psi2::Float64
    )
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= N_particles
        # --- compute radius ---
        r2 = 0.0
        for d in 1:size(positions, 1)
            r2 += positions[d, i]^2
        end
        r = sqrt(r2)
        if r < eps(Float64)
            return
        end

        # --- local fluid velocity ---
        t_now = steps * Δt + initial_time
        v = interpolate_2d_cuda(xgrid, tgrid, VelocityEvolution, r, t_now)
        # optional m=2 azimuthal modulation (must match the rest-frame boost for an exact round-trip)
        if use_v2 && size(positions, 1) >= 2
            v2m = interpolate_2d_cuda(xgrid, tgrid, V2Evolution, r, t_now)
            φ = atan(positions[2, i], positions[1, i])
            v *= (1.0 + 2.0 * v2m * cos(2.0 * (φ - psi2)))
        end
        v2 = v * v
        γ = 1.0 / sqrt(1.0 - v2 + 1e-10)

        # --- particle energy in LRF ---
        p2 = 0.0
        for d in 1:size(momenta, 1)
            p2 += momenta[d, i]^2
        end
        E = sqrt(p2 + m^2)

        if radial_mode
            # ======================================
            # 🟣 Radial mode (1D evolution)
            # ======================================
            @inbounds momenta[1, i] = γ * (momenta[1, i] + v * E)

        else
            # ======================================
            # 🟢 Cartesian mode (2D, rotation invariant)
            # ======================================
            x = positions[1, i]
            y = positions[2, i]
            rhatx, rhaty = x / r, y / r

            # Decompose momentum along / perpendicular to radial direction
            px, py = momenta[1, i], momenta[2, i]
            p_parallel = px * rhatx + py * rhaty
            p_perp_x = px - p_parallel * rhatx
            p_perp_y = py - p_parallel * rhaty

            # Lorentz boost back to lab frame (sign flip vs. rest frame)
            p_parallel_new = γ * (p_parallel + v * E)

            # Recombine momentum vector
            @inbounds momenta[1, i] = p_parallel_new * rhatx + p_perp_x
            @inbounds momenta[2, i] = p_parallel_new * rhaty + p_perp_y
        end
    end
    return
end

@inline function kernel_compute_all_forces_gpu!(
    TemperatureEvolution, xgrid, tgrid,
    momenta, positions, p_mags, p_units,
    ηD_vals, kL_vals, kT_vals,
    ξ, deterministic_terms, stochastic_terms,
    Δt, m, random_directions,
    dimensions, N_particles, steps, initial_time, DsT,
    tau_Tmin::Float64, tau_invdT::Float64, tau_vals,
    radial_mode::Bool, integrator_mode::Int32,   # 0 = pre-point exact-OU (legacy), 1 = drift-midpoint O(ε²)
    relativistic::Bool                            # true = ·m/E (Jüttner); false = ηD (Maxwell)
    )
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= N_particles
        M = m  # heavy-quark mass (can be parameterized)

        # --- local temperature ---
        r2 = 0.0
        for d in 1:dimensions
            r2 += positions[d, i]^2
        end
        r = sqrt(r2)
        T = interpolate_2d_cuda(xgrid, tgrid, TemperatureEvolution, r, steps * Δt + initial_time)

        # --- transport coefficients ---
        # Enforce main3-style τn(T) via a precomputed spline in fm.
        τn = (DsT > 0.0) ? _eval_tau_n_spline_cuda(Float64(T), tau_Tmin, tau_invdT, tau_vals) : 0.0
        ηD = (τn > 0.0 && isfinite(τn)) ? (1.0 / τn) : 0.0
        κ  = (τn > 0.0 && isfinite(τn)) ? ((2.0 * M * Float64(T)) / τn) : 0.0
        kL = sqrt(κ)
        kT = sqrt(κ)

        @inbounds ηD_vals[i] = ηD
        @inbounds kL_vals[i] = kL
        @inbounds kT_vals[i] = kT

        if radial_mode
            # ==========================================
            # 🟣 Radial / 1D mode: noise projected along p̂
            # ==========================================
            p_sq = 0.0
            for d in 1:dimensions
                p_sq += momenta[d, i]^2
            end
            p = sqrt(p_sq)
            @inbounds p_mags[i] = p

            # Unit vector along momentum (or random if p≈0)
            for d in 1:dimensions
                @inbounds p_units[d, i] = p < eps(Float64) ? random_directions[d, i] :
                                                              momenta[d, i] / p
            end

            # Relativistic drag: η_eff = ηD * m/E_LRF for Jüttner equilibrium.
            E_LRF_r = sqrt(p_sq + M^2)
            η_eff_r = relativistic ? ηD * M / E_LRF_r : ηD   # non-rel: momentum-independent drag ⇒ Maxwell

            for d in 1:dimensions
                det_term = -η_eff_r * momenta[d, i] * Δt

                sto_term = 0.0
                for j in 1:dimensions
                    sto_term += (kL - kT) * p_units[d, i] * p_units[j, i] * ξ[j, i] +
                                kT * (d == j ? 1.0 : 0.0) * ξ[j, i]
                end

                @inbounds deterministic_terms[d, i] = det_term
                @inbounds stochastic_terms[d, i]    = sto_term
            end

        else
            # ==========================================
            # 🟢 Cartesian mode: exact OU propagator
            # ==========================================
            # Relativistic: η_eff = ηD * m/E_LRF so equilibrium is Jüttner ∝ exp(-E/T).
            # κ = 2mT ηD unchanged; only the attenuation a = exp(-η_eff Δt) changes.
            if ηD > 0.0 && Δt > 0.0
                p2_lrf = 0.0
                for d in 1:dimensions
                    p2_lrf += momenta[d, i]^2
                end
                E_LRF   = sqrt(p2_lrf + M^2)
                η_eff   = relativistic ? ηD * M / E_LRF : ηD    # pre-point drag (rel: ·m/E ⇒ Jüttner; non-rel: ηD ⇒ Maxwell)
                if integrator_mode == Int32(1)
                    # DRIFT-MIDPOINT (O(ε²)): NOISE-FREE deterministic half-step to a midpoint, re-evaluate
                    # η_eff there, then apply the full exact-OU step with the midpoint coefficients. The
                    # predictor must NOT include the noise (else η_eff correlates with ξ ⇒ spurious drift);
                    # noise is additive (κ momentum-independent) so there is no Milstein noise term.
                    a_half = exp(-η_eff * (0.5 * Δt))
                    p2_mid = 0.0
                    for d in 1:dimensions
                        pm = a_half * momenta[d, i]
                        p2_mid += pm * pm
                    end
                    E_mid = sqrt(p2_mid + M^2)
                    η_eff = relativistic ? ηD * M / E_mid : ηD   # midpoint drag (overwrites pre-point)
                end
                a       = exp(-η_eff * Δt)
                noise_pref = kT * sqrt((1.0 - a * a) / (2.0 * η_eff * Δt))   # matched FDR a↔variance
                for d in 1:dimensions
                    @inbounds deterministic_terms[d, i] = (a - 1.0) * momenta[d, i]
                    @inbounds stochastic_terms[d, i]    = noise_pref * ξ[d, i]
                end
            else
                for d in 1:dimensions
                    @inbounds deterministic_terms[d, i] = 0.0
                    @inbounds stochastic_terms[d, i]    = kT * ξ[d, i]
                end
            end
        end
    end
    return
end

@inline function kernel_update_momenta_LRF_gpu!(
    momenta, deterministic_terms, stochastic_terms,
    Δt, dimensions, N_particles
    )
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= N_particles
        for d in 1:dimensions
            Δp = @inbounds deterministic_terms[d, i] + sqrt(Δt) * stochastic_terms[d, i]
            @inbounds momenta[d, i] += Δp
        end
    end
    return
end

@inline function kernel_update_positions_gpu!(
    positions, 
    momenta, 
    m::Float64, 
    Δt::Float64, 
    N_particles::Int,
    steps::Int,
    initial_time::Float64,
    xgrid,
    tgrid,
    Tfield,
    DsT::Float64,
    dimensions::Int,
    radial_mode::Bool,
    position_diffusion::Bool,
    reflecting_boundary::Bool,
    random_normals  # pre-generated random numbers for diffusion
    )
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if idx <= N_particles
        r_max = length(xgrid) >= 1 ? xgrid[end] : 0.0
        dr0 = (length(xgrid) >= 2) ? CUDA.abs(xgrid[2] - xgrid[1]) : 0.0
        r_axis_eps = CUDA.max(1e-12, 0.5 * dr0)
        # Compute energy
        E2 = m * m
        for d in 1:dimensions
            E2 += momenta[d, idx]^2
        end
        E = CUDA.sqrt(E2)

        if radial_mode
            r = positions[1, idx]
            pr = momenta[1, idx]

            r_abs  = CUDA.abs(r)
            r_safe = (r_abs < r_axis_eps) ? r_axis_eps : r_abs

            # deterministic motion
            dr = (pr / E) * Δt

            if position_diffusion
                T = interpolate_2d_cuda(xgrid, tgrid, Tfield, r_safe, steps * Δt + initial_time)
                # D_s = DsT/T in GeV^-1; convert to fm for x-update with Δt in fm.
                D = (DsT / T) / fmGeV
                if D > 0.0
                    ξ = random_normals[1, idx]
                    dr += (D / r_safe) * Δt + CUDA.sqrt(2.0 * D * Δt) * ξ
                end
            end

            # reflection boundary
            r_new = r + dr
            if r_new < 0.0
                r_new = -r_new
                @inbounds momenta[1, idx] = -momenta[1, idx]
            end

            if reflecting_boundary && r_max > 0.0 && r_new > r_max
                r_new = 2.0 * r_max - r_new
                @inbounds momenta[1, idx] = -momenta[1, idx]
                if r_new < 0.0
                    r_new = -r_new
                    @inbounds momenta[1, idx] = -momenta[1, idx]
                end
            end

            @inbounds positions[1, idx] = r_new

        else
            # --- full Cartesian update ---
            for d in 1:dimensions
                @inbounds positions[d, idx] += Δt * momenta[d, idx] / E
            end

            if position_diffusion
                # Spatial diffusion in Cartesian coordinates (consistent with r-mode):
                # x_d -> x_d + sqrt(2 D dt) * ξ_d, with D = DsT / T(r,τ)
                r2 = 0.0
                for d in 1:dimensions
                    r2 += positions[d, idx]^2
                end
                r = CUDA.sqrt(r2)
                T = interpolate_2d_cuda(xgrid, tgrid, Tfield, r, steps * Δt + initial_time)
                # D_s = DsT/T in GeV^-1; convert to fm for x-update with Δt in fm.
                D = (DsT / T) / fmGeV
                if D > 0.0
                    σ = CUDA.sqrt(2.0 * D * Δt)
                    for d in 1:dimensions
                        @inbounds positions[d, idx] += σ * random_normals[d, idx]
                    end
                end
            end

            if reflecting_boundary && r_max > 0.0
                r2 = 0.0
                for d in 1:dimensions
                    r2 += positions[d, idx]^2
                end
                r_now = CUDA.sqrt(r2)
                if r_now > r_max
                    invr = 1.0 / CUDA.max(r_now, r_axis_eps)

                    ppar = 0.0
                    for d in 1:dimensions
                        ppar += momenta[d, idx] * (positions[d, idx] * invr)
                    end
                    for d in 1:dimensions
                        @inbounds momenta[d, idx] -= 2.0 * ppar * (positions[d, idx] * invr)
                    end

                    scale = (2.0 * r_max - r_now) * invr
                    for d in 1:dimensions
                        @inbounds positions[d, idx] *= scale
                    end
                end
            end
        end
    end
    return
end

@inline function kernel_save_momenta_gpu!(
    momenta_history::CuDeviceMatrix{Float64},
    current_momentum,
    save_idx::Int,
    N_particles::Int,
    dimensions::Int
    )
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if idx <= N_particles
        col_offset = (save_idx - 1) * N_particles
        for d in 1:dimensions
            @inbounds momenta_history[d, col_offset + idx] = current_momentum[d, idx]
        end
    end
    return
end

@inline function kernel_save_positions_gpu!(
    position_history::CuDeviceMatrix{Float64},
    current_positions,
    save_idx::Int,
    N_particles::Int,
    dimensions::Int
    )
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if idx <= N_particles
        col_offset = (save_idx - 1) * N_particles
        for d in 1:dimensions
            @inbounds position_history[d, col_offset + idx] = current_positions[d, idx]
        end
    end
    return
end

# On-the-fly freeze-out capture: for each particle NOT yet frozen (fo_flag==0), evaluate the local
# temperature at the current (r,τ). While above Tfo, cache the state in last_*. The step it first drops
# below Tfo, INTERPOLATE position/momentum/time linearly in T between the last-above and first-below states
# to the exact T=Tfo crossing (interp=true; removes the O(Δt) booking lag ⇒ Δτ_save→0), or book the raw
# first-below snapshot (interp=false). No history stored ⇒ memory ∝ N, VRAM-safe at any cadence / resolution.
@inline function kernel_capture_freezeout_gpu!(
    fo_pos, fo_mom, fo_tau, fo_flag, last_pos, last_mom, last_tau, last_T,
    positions, momenta, xgrid, tgrid, Tfield, Tfo::Float64,
    N_particles::Int, steps::Int, Δt::Float64, initial_time::Float64,
    dimensions::Int, radial_mode::Bool, interp::Bool
    )
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if idx <= N_particles
        @inbounds if fo_flag[idx] == 0.0
            r = 0.0
            if radial_mode
                @inbounds r = CUDA.abs(positions[1, idx])
            else
                r2 = 0.0
                for d in 1:dimensions
                    @inbounds r2 += positions[d, idx]^2
                end
                r = CUDA.sqrt(r2)
            end
            τ = steps * Δt + initial_time
            T = interpolate_2d_cuda(xgrid, tgrid, Tfield, r, τ)
            if T < Tfo
                Tl = last_T[idx]
                # frac=1 ⇒ current (raw first-below); frac∈[0,1] ⇒ interpolated to exact T=Tfo
                frac = (interp && Tl > Tfo && Tl > T) ? (Tl - Tfo) / (Tl - T) : 1.0
                for d in 1:dimensions
                    @inbounds fo_pos[d, idx] = last_pos[d, idx] + frac * (positions[d, idx] - last_pos[d, idx])
                    @inbounds fo_mom[d, idx] = last_mom[d, idx] + frac * (momenta[d, idx] - last_mom[d, idx])
                end
                @inbounds fo_tau[idx] = last_tau[idx] + frac * (τ - last_tau[idx])
                @inbounds fo_flag[idx] = 1.0
            else
                # still above Tfo: cache this state as the last-above reference for interpolation
                for d in 1:dimensions
                    @inbounds last_pos[d, idx] = positions[d, idx]
                    @inbounds last_mom[d, idx] = momenta[d, idx]
                end
                @inbounds last_tau[idx] = τ
                @inbounds last_T[idx] = T
            end
        end
    end
    return
end

# === Boltzmann RTA / BGK collision (GPU) — divergence-free drop-in for the OU force+update ===
# For each particle in the LOCAL REST FRAME: with probability Pcol = clamp(Δt/τn(T_local), 0, 1)
# (τn ≤ 0 ⇒ Pcol=1, the DsT→0 instantaneous-equilibration limit), re-draw the momentum magnitude
# from the isotropic local Jüttner via a constant-work inverse-CDF table lookup, then assign an
# isotropic direction; otherwise leave the momentum unchanged. This is the exact GPU analogue of
# `kernel_rta_collision_cpu!`: same τn spline, same Pcol formula, same P(p*) ∝ p*^(d-1)exp(-(E-m)/T)
# shape — but every thread does identical work (no rejection loop ⇒ no warp divergence).
# Uses PRE-GENERATED random arrays (fill with CUDA.rand!/CUDA.randn! in the sim loop):
#   u_collide[i] ~ U(0,1)   collision decision
#   u_sample[i] ~ U(0,1)    inverse-CDF draw of |p*|
#   dir_gauss[d,i] ~ N(0,1) isotropic direction (radial_mode: sign only)
# Call BETWEEN kernel_boost_to_rest_frame_gpu! and kernel_boost_to_lab_frame_gpu!.
@inline function kernel_rta_collision_gpu!(
    momenta, positions, xgrid, tgrid, Tfield,
    Δt::Float64, m::Float64, N_particles::Int, steps, initial_time::Float64, DsT::Float64,
    tau_Tmin::Float64, tau_invdT::Float64, tau_vals,
    invCDF, nU::Int, nT::Int, Ttab_min::Float64, Ttab_invdT::Float64,
    dimensions::Int, radial_mode::Bool,
    u_collide, u_sample, dir_gauss,
    )
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= N_particles
        M = m
        # --- local radius and temperature (matches the CPU RTA kernel) ---
        r = 0.0
        if radial_mode
            @inbounds r = CUDA.abs(positions[1, i])
        else
            r2 = 0.0
            for d in 1:dimensions
                @inbounds r2 += positions[d, i]^2
            end
            r = CUDA.sqrt(r2)
        end
        T = interpolate_2d_cuda(xgrid, tgrid, Tfield, r, steps * Δt + initial_time)
        Tpos = T > 0.0 ? T : 0.0

        τn = (DsT > 0.0 && Tpos > 0.0) ? _eval_tau_n_spline_cuda(Tpos, tau_Tmin, tau_invdT, tau_vals) : 0.0
        Pcol = (τn > 0.0 && isfinite(τn)) ? clamp(Δt / τn, 0.0, 1.0) : 1.0

        @inbounds uc = u_collide[i]
        if uc < Pcol
            @inbounds us = u_sample[i]
            pstar = _juttner_invcdf_lookup(invCDF, nU, nT, Ttab_min, Ttab_invdT, us, Tpos)
            if radial_mode
                # signed radial magnitude; fair ± coin from a Gaussian's sign
                @inbounds g1 = dir_gauss[1, i]
                s = g1 < 0.0 ? -1.0 : 1.0
                @inbounds momenta[1, i] = s * pstar
            else
                nrm2 = 0.0
                for d in 1:dimensions
                    @inbounds g = dir_gauss[d, i]
                    nrm2 += g * g
                end
                inv = pstar / CUDA.sqrt(nrm2 > 1e-300 ? nrm2 : 1e-300)
                for d in 1:dimensions
                    @inbounds momenta[d, i] = dir_gauss[d, i] * inv
                end
            end
        end
    end
    return
end

@inline function kernel_set_to_fluid_velocity_gpu!(
    momenta,
    positions,
    xgrid,
    tgrid,
    VelocityEvolution,
    m::Float64,
    N_particles::Int,
    steps::Int,
    Δt::Float64,
    initial_time::Float64,
    radial_mode::Bool
    )
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if idx <= N_particles
        # --- Compute position & radius ---
        if radial_mode
            # In radial mode, positions has shape [1, N]
            r = positions[1, idx]
        else
            # In Cartesian mode, positions has shape [2, N]
            x = positions[1, idx]
            y = positions[2, idx]
            r = CUDA.sqrt(x * x + y * y)
        end

        if r < eps(Float64)  # avoid divide-by-zero
            return
        end

        # --- Interpolate fluid velocity from (r, τ) field ---
        τ_now = steps * Δt + initial_time
        v = interpolate_2d_cuda(xgrid, tgrid, VelocityEvolution, r, τ_now)

        # --- Lorentz factor ---
        γ = 1.0 / CUDA.sqrt(1.0 - v * v + 1e-10)

        if radial_mode
            # =====================================
            # 🟣 Radial mode (1D)
            # =====================================
            @inbounds momenta[1, idx] = m * γ * v

        else
            # =====================================
            # 🟢 Cartesian mode (2D, rotation invariant)
            # =====================================
            # Set momentum along local radial direction
            @inbounds momenta[1, idx] = m * γ * v * (x / r)
            @inbounds momenta[2, idx] = m * γ * v * (y / r)
        end
    end
    return
end

end # module KernelsGPU
