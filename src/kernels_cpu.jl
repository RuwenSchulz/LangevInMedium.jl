module KernelsCPU

using LinearAlgebra
using ..Constants: fmGeV
using ..Transport: eval_tau_n_spline

# === Exported Symbols ===
export interpolate_2d_cpu,
       kernel_boost_to_rest_frame_cpu!,
       kernel_boost_to_lab_frame_cpu!,
       kernel_compute_all_forces_cpu!,
       kernel_update_momenta_LRF_cpu!,
       kernel_update_positions_cpu!,
       kernel_save_snapshot_cpu!,
       kernel_save_momenta_cpu!,
       kernel_save_positions_cpu!,
       kernel_set_to_fluid_velocity_cpu!


function interpolate_2d_cpu(x, y, values, xi, yi)
    # Clamp query points to the tabulated domain.
    # The old code clamped *indices* but still extrapolated when xi/yi were
    # outside the grid, which can create unphysical values (e.g. |v|>1) and
    # lead to NaNs in Lorentz factors.
    xi = clamp(xi, first(x), last(x))
    yi = clamp(yi, first(y), last(y))

    i = searchsortedlast(x, xi)
    j = searchsortedlast(y, yi)

    i = clamp(i, 1, length(x) - 1)
    j = clamp(j, 1, length(y) - 1)

    x0, x1 = x[i], x[i + 1]
    y0, y1 = y[j], y[j + 1]

    v00, v10 = values[i, j], values[i + 1, j]
    v01, v11 = values[i, j + 1], values[i + 1, j + 1]

    dx = (x1 - x0)
    dy = (y1 - y0)
    xd = abs(dx) < eps() ? 0.0 : (xi - x0) / dx
    yd = abs(dy) < eps() ? 0.0 : (yi - y0) / dy

    c0 = v00 * (1 - xd) + v10 * xd
    c1 = v01 * (1 - xd) + v11 * xd

    return c0 * (1 - yd) + c1 * yd
end

function kernel_boost_to_rest_frame_cpu!(
    momenta, positions, xgrid, tgrid, VelocityEvolution,
    m, N, step, Δt, t0;
    radial_mode::Bool = false
    )
    @inbounds for i in 1:N
        # --- compute current proper time ---
        t_now = step * Δt + t0

        # --- compute radius r ---
        r2 = 0.0
        for d in 1:size(positions, 1)
            r2 += positions[d, i]^2
        end
        r = sqrt(r2)
        if r < eps()
            continue
        end

        # --- local radial fluid velocity ---
        v = interpolate_2d_cpu(xgrid, tgrid, VelocityEvolution, r, t_now)
        # Guard against superluminal values due to numerical noise.
        vmax = sqrt(1.0 - 1e-12)
        v = clamp(v, -vmax, vmax)
        v2 = v * v
        γ  = 1.0 / sqrt(1.0 - v2 + 1e-10)

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
            momenta[1, i] = γ * (momenta[1, i] - v * E)

        else
            # ==================================
            # 🟢 Cartesian mode (2D or 3D)
            # ==================================
            if size(positions, 1) < 2
                error("Cartesian boost requires at least 2D positions")
            end

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
            momenta[1, i] = p_parallel_new * rhatx + p_perp_x
            momenta[2, i] = p_parallel_new * rhaty + p_perp_y
        end
    end
    return nothing
end

function kernel_boost_to_lab_frame_cpu!(
    momenta, positions, xgrid, tgrid, VelocityEvolution,
    m, N, step, Δt, t0;
    radial_mode::Bool = false
    )
    @inbounds for i in 1:N
        # --- compute radius ---
        r2 = 0.0
        for d in 1:size(positions, 1)
            r2 += positions[d, i]^2
        end
        r = sqrt(r2)
        if r < eps()
            continue
        end

        # --- local fluid velocity ---
        t_now = step * Δt + t0
        v = interpolate_2d_cpu(xgrid, tgrid, VelocityEvolution, r, t_now)
        vmax = sqrt(1.0 - 1e-12)
        v = clamp(v, -vmax, vmax)
        v2 = v * v
        γ  = 1.0 / sqrt(1.0 - v2 + 1e-10)

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
            momenta[1, i] = γ * (momenta[1, i] + v * E)

        else
            # ======================================
            # 🟢 Cartesian mode (2D, rotation invariant)
            # ======================================
            if size(positions, 1) < 2
                error("Cartesian boost requires at least 2D positions")
            end

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
            momenta[1, i] = p_parallel_new * rhatx + p_perp_x
            momenta[2, i] = p_parallel_new * rhaty + p_perp_y
        end
    end
    return nothing
end

function kernel_compute_all_forces_cpu!(
    Tfield, xgrid, tgrid,
    momenta, positions,
    p_mags, p_units,       # used only in radial mode
    ηD_vals, kL_vals, kT_vals,
    ξ, deterministic_terms, stochastic_terms,
    Δt, m, random_directions,
    dimensions, N, step, t0, DsT;
    tau_Tmin::Float64,
    tau_invdT::Float64,
    tau_vals::AbstractVector{<:Real},
    radial_mode::Bool = false
 )
    M = m  # heavy-quark mass (can be parameterized)

    @inbounds for i in 1:N
        # --- local temperature ---
        r = sqrt(sum(positions[:, i].^2))
        T = interpolate_2d_cpu(xgrid, tgrid, Tfield, r, step * Δt + t0)
        # Temperature should be non-negative; clamp to avoid NaNs from T<0.
        T = max(float(T), 0.0)

        # Guard against invalid transport inputs.
        DsT_safe = DsT <= 0 ? 0.0 : DsT

        # --- transport coefficients ---
        # We enforce Langevin to use the same τn(T,m,DsT) logic as FiVoHydro.jl/main3.jl.
        # τn is provided as a precomputed spline in fm.
        τn = DsT_safe == 0.0 ? 0.0 : eval_tau_n_spline(T, tau_Tmin, tau_invdT, tau_vals)

        # Define drag/diffusion consistently via τn:
        #   ηD = 1/τn  [1/fm]
        #   κ  = 2 m T / τn  [GeV^2/fm]
        ηD = (τn > 0.0 && isfinite(τn)) ? (1.0 / τn) : 0.0
        κ  = (τn > 0.0 && isfinite(τn)) ? ((2.0 * M * T) / τn) : 0.0
        kL = sqrt(κ)
        kT = sqrt(κ)

        ηD_vals[i], kL_vals[i], kT_vals[i] = ηD, kL, kT

        if radial_mode
            # ==========================================
            # 🟣 Radial / 1D mode: noise projected along p̂
            # ==========================================
            p = sqrt(sum(momenta[:, i].^2))
            p_mags[i] = p

            # Unit vector along momentum (or random if p≈0)
            for d in 1:dimensions
                p_units[d, i] = p < eps() ? random_directions[d, i] :
                                              momenta[d, i] / p
            end

            for d in 1:dimensions
                # Deterministic drag
                det_term = -ηD * momenta[d, i] * Δt

                # Stochastic force with projection
                sto_term = 0.0
                for j in 1:dimensions
                    sto_term += (kL - kT) * p_units[d, i] * p_units[j, i] * ξ[j, i] +
                                kT * (d == j ? 1.0 : 0.0) * ξ[j, i]
                end

                deterministic_terms[d, i] = det_term
                stochastic_terms[d, i]    = sto_term
            end

        else
            # ==========================================
            # 🟢 Cartesian mode: independent px, py
            # ==========================================
            # Numerically stable discretization (exact OU for constant coefficients):
            #   dp = -ηD p dt + kT dW
            # Exact: p_{n+1} = a p_n + kT * sqrt((1-a^2)/(2ηD)) * ξ
            # where a = exp(-ηD dt). We encode this into the existing
            # (deterministic_terms + sqrt(dt)*stochastic_terms) update.
            if ηD > 0
                a = exp(-ηD * Δt)
                # Noise prefactor to be multiplied by sqrt(Δt) in the updater.
                noise_pref = (Δt > 0) ? (kT * sqrt((1 - a * a) / (2 * ηD * Δt))) : 0.0
                for d in 1:dimensions
                    deterministic_terms[d, i] = (a - 1) * momenta[d, i]
                    stochastic_terms[d, i]    = noise_pref * ξ[d, i]
                end
            else
                for d in 1:dimensions
                    deterministic_terms[d, i] = 0.0
                    stochastic_terms[d, i]    = kT * ξ[d, i]
                end
            end
        end
    end

    return nothing
end

function kernel_compute_all_forces_cpu!(
    T,
    momenta, p_mags, p_units,
    ηD_vals, kL_vals, kT_vals,
    ξ, deterministic_terms, stochastic_terms,
    Δt, m, random_directions,
    dimensions, N, step, t0
    )
    for i in 1:N
        # Compute particle momentum magnitude
        p = sqrt(sum(momenta[:, i] .^ 2))
        p_mags[i] = p

        # Compute unit momentum vectors (with fallback if zero momentum)
        for d in 1:dimensions
            p_units[d, i] = p < eps() ? random_directions[d, i] : momenta[d, i] / p
        end

        # Interpolate temperature from space-time field

        # Compute transport coefficients

        κ = 2.5 * T^3
        ηD = κ/(2 * T * m)
        kL, kT = sqrt(κ), sqrt(κ)

        ηD_vals[i], kL_vals[i], kT_vals[i] = ηD, kL, kT

        # Compute force components
        for d in 1:dimensions
            # Langevin deterministic drag term
            det_term = -ηD * momenta[d, i] * Δt

            # Langevin stochastic term
            sto_term = 0.0
            for j in 1:dimensions
                sto_term += ((kL - kT) * p_units[d, i] * p_units[j, i] * ξ[j, i] +
                            kT * (d == j ? 1.0 : 0.0) * ξ[j, i]) 
            end

            # Store computed forces
            deterministic_terms[d, i] = det_term
            stochastic_terms[d, i] = sto_term
        end
    end
end

function kernel_update_momenta_LRF_cpu!(
    momenta, deterministic_terms, stochastic_terms,
    Δt, dimensions, N
    )
    sqrtΔt = sqrt(Δt)
    @inbounds for i in 1:N
        for d in 1:dimensions
            momenta[d, i] += deterministic_terms[d, i] + sqrtΔt * stochastic_terms[d, i]
        end
    end
    return nothing
end

function kernel_update_positions_cpu!(
    positions, momenta, m, Δt, N,step,t0,
    xgrid,tgrid, Tfield,DsT;
    dimensions::Int = 2,
    radial_mode::Bool = false,
    position_diffusion::Bool = false,
    reflecting_boundary::Bool = false,
    )
    # Grid-based axis cutoff for the geometric drift term D/r.
    # Using machine eps() here can create enormous dr near r=0.
    r_axis_eps = if length(xgrid) >= 2
        max(1e-12, 0.5 * abs(float(xgrid[2] - xgrid[1])))
    else
        1e-6
    end

    r_max = length(xgrid) >= 1 ? float(last(xgrid)) : 0.0
    @inbounds for i in 1:N
        # Compute energy
        E2 = m^2
        for d in 1:dimensions
            E2 += momenta[d, i]^2
        end
        E = sqrt(E2)

        if radial_mode
            r = positions[1, i]
            pr = momenta[1, i]

            r_abs  = abs(r)
            r_safe = (r_abs < r_axis_eps) ? r_axis_eps : r_abs

            # deterministic motion
            dr = (pr / E) * Δt

            # Optional coordinate-space diffusion (overdamped Brownian motion).
            # Default is OFF because the underdamped Langevin already produces spatial
            # diffusion through momentum kicks + advection; enabling this typically
            # double-counts diffusion compared to hydro.
            if position_diffusion
                T = interpolate_2d_cpu(xgrid, tgrid, Tfield, r_safe, step * Δt + t0)
                # D_s = DsT/T is in GeV^-1; convert to fm for x-update with Δt in fm.
                D = (DsT / max(float(T), eps())) / fmGeV
                if D > 0
                    ξ = randn()
                    dr += (D / r_safe) * Δt + sqrt(2 * D * Δt) * ξ
                end
            end

            # reflection boundary
            r_new = r + dr
            if r_new < 0
                r_new = -r_new
                momenta[1, i] = -momenta[1, i]
            end

            if reflecting_boundary && r_max > 0.0 && r_new > r_max
                r_new = 2 * r_max - r_new
                momenta[1, i] = -momenta[1, i]
                if r_new < 0.0
                    r_new = -r_new
                    momenta[1, i] = -momenta[1, i]
                end
            end

            positions[1, i] = r_new

        else
            # --- full Cartesian update ---
            for d in 1:dimensions
                positions[d, i] += Δt * momenta[d, i] / E
            end

            # Optional coordinate-space diffusion (see note above).
            if position_diffusion
                r2 = 0.0
                for d in 1:dimensions
                    r2 += positions[d, i]^2
                end
                r_now = sqrt(r2)
                T = interpolate_2d_cpu(xgrid, tgrid, Tfield, r_now, step * Δt + t0)
                # D_s = DsT/T is in GeV^-1; convert to fm for x-update with Δt in fm.
                D = (DsT / max(float(T), eps())) / fmGeV
                if D > 0
                    σ = sqrt(2 * D * Δt)
                    for d in 1:dimensions
                        positions[d, i] += σ * randn()
                    end
                end
            end

            if reflecting_boundary && r_max > 0.0
                r2 = 0.0
                for d in 1:dimensions
                    r2 += positions[d, i]^2
                end
                r_now = sqrt(r2)
                if r_now > r_max
                    # Unit radial vector (before reflection)
                    invr = 1.0 / max(r_now, r_axis_eps)

                    # Reflect momentum: flip radial component, keep tangential.
                    ppar = 0.0
                    for d in 1:dimensions
                        ppar += momenta[d, i] * (positions[d, i] * invr)
                    end
                    for d in 1:dimensions
                        momenta[d, i] -= 2 * ppar * (positions[d, i] * invr)
                    end

                    # Reflect position across the circle r=r_max.
                    scale = (2 * r_max - r_now) * invr
                    for d in 1:dimensions
                        positions[d, i] *= scale
                    end
                end
            end
        end
    end
end

function kernel_save_snapshot_cpu!(
    history_col, snapshot::Vector{Float64}, N::Int
    )
    @inbounds for i in 1:N
        history_col[i] = snapshot[i]
    end
end

function kernel_save_momenta_cpu!(
    momenta_history, current_momentum, save_idx::Int, N::Int
    )
    D = size(current_momentum, 1)
    @inbounds for i in 1:N
        @simd for d in 1:D
            momenta_history[d, i, save_idx] = current_momentum[d, i]
        end
    end
    return nothing
end

function kernel_save_positions_cpu!(
    position_history, current_positions, save_idx::Int, N::Int
    )
    D = size(current_positions, 1)
    @inbounds for i in 1:N
        @simd for d in 1:D
            position_history[d, i, save_idx] = current_positions[d, i]
        end
    end
    return nothing
end

function kernel_set_to_fluid_velocity_cpu!(
    momenta::Array{Float64,2},
    positions::Array{Float64,2},
    xgrid, tgrid, VelocityEvolution,
    m::Float64,
    N::Int,
    step::Int,
    Δt::Float64,
    t0::Float64;
    radial_mode::Bool = false
    )
    @inbounds for i in 1:N
        # --- Compute position & radius ---
        if radial_mode
            # In radial mode, positions has shape [1, N]
            r = positions[1, i]
        else
            # In Cartesian mode, positions has shape [2, N]
            x = positions[1, i]
            y = positions[2, i]
            r = sqrt(x^2 + y^2)
        end

        if r < eps()  # avoid divide-by-zero
            continue
        end

        # --- Interpolate fluid velocity from (r, τ) field ---
        τ_now = step * Δt + t0
        v = interpolate_2d_cpu(xgrid, tgrid, VelocityEvolution, r, τ_now)
        vmax = sqrt(1.0 - 1e-12)
        v = clamp(v, -vmax, vmax)

        # --- Lorentz factor ---
        γ = 1.0 / sqrt(1.0 - v^2 + 1e-10)

        if radial_mode
            # =====================================
            # 🟣 Radial mode (1D)
            # =====================================
            momenta[1, i] = m * γ * v

        else
            # =====================================
            # 🟢 Cartesian mode (2D, rotation invariant)
            # =====================================
            # Set momentum along local radial direction
            momenta[1, i] = m * γ * v * (x / r)
            momenta[2, i] = m * γ * v * (y / r)
        end
    end
    return nothing
end

end # module KernelsCPU
