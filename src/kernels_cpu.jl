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
       kernel_bjorken_redshift_cpu!,
       kernel_rta_collision_cpu!,
       kernel_update_positions_cpu!,
       kernel_save_snapshot_cpu!,
       kernel_save_momenta_cpu!,
       kernel_save_positions_cpu!,
       kernel_set_to_fluid_velocity_cpu!


"""
    interpolate_2d_cpu

Bilinear lookup of a tabulated field `values[i, j]` on `(x[i], y[j])` at `(xi, yi)`. The query point is CLAMPED to the table (edge value outside it) — extrapolating gave `T < 0` and `|v| > 1` once particles left the grid. Degenerate cells (`x1 == x0`) return the node value.
"""
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

"""
    kernel_boost_to_rest_frame_cpu!

Boost every particle's momentum from the lab into the local fluid rest frame at its own `(r, t_now = step·Δt + t0)`: a Lorentz boost along r̂ by the interpolated radial flow `v` (clamped below 1), or, with `relativistic = false`, the Galilean `p∥ −= m·v`. `radial_mode`: `momenta` holds `p_r` only. `V2Evolution`/`psi2` modulate `v` azimuthally. Particles at `r < eps` are left alone (r̂ undefined).
"""
function kernel_boost_to_rest_frame_cpu!(
    momenta, positions, xgrid, tgrid, VelocityEvolution,
    m, N, step, Δt, t0;
    radial_mode::Bool = false,
    V2Evolution = nothing, psi2::Float64 = 0.0,
    # ⚠ relativistic = false makes the frame change GALILEAN: p*∥ = p∥ − m·v with v the flow field
    # read as a VELOCITY (no γ, no E, no clamp — the non-relativistic class puts no bound on H·r).
    # This is what makes `relativistic = false` the actual exactly solvable process rather than the
    # O(T/M) hybrid documented above kernel_boost_to_lab_frame_cpu!. Default true is bit-identical
    # to the old behaviour.
    relativistic::Bool = true
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
        # optional m=2 azimuthal modulation of the radial-flow magnitude (elliptic flow v2):
        # v→v·(1+2 v2(r,t) cos2(φ−Ψ₂)).  Default (V2Evolution=nothing) ⇒ byte-identical radial flow.
        if V2Evolution !== nothing && size(positions, 1) >= 2
            v2m = interpolate_2d_cpu(xgrid, tgrid, V2Evolution, r, t_now)
            φ = atan(positions[2, i], positions[1, i])
            v *= (1.0 + 2.0 * v2m * cos(2.0 * (φ - psi2)))
        end

        if !relativistic
            # Galilean: subtract the flow momentum m·v along r̂ and be done.
            if radial_mode
                momenta[1, i] -= m * v
            else
                x = positions[1, i]; y = positions[2, i]
                momenta[1, i] -= m * v * (x / r)
                momenta[2, i] -= m * v * (y / r)
            end
            continue
        end

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

# ⚠ HISTORY: until 2026-08-15 the boosts were LORENTZ BOOSTS FOR EVERY RUN — `relativistic=false`
# switched only the streaming velocity and the kick, making a "non-relativistic" run a KINEMATIC
# HYBRID at O(T/M): boosting back after an LRF kick hands the particle a mean momentum
# gamma*v*<E> ≈ M*v*(1 + T/M), so the cloud was advected ~T/M faster than the flow (measured: its
# sigma_xx outgrew the exact Galilean covariance solution by 21% by tau = 10 fm/c on the
# stages4_cf3nr_t6 background, T/M = 0.12), and the recorded plateau landed between the Galilean
# and boosted readings of the exact class. The `relativistic` kwarg on both boost kernels now fixes
# this: false ⇒ Galilean p∥ ∓ m·v. Every NR product generated BEFORE this date carries the hybrid
# and must not be compared to the solvable class at the sub-percent level below z ~ 15.
"""
    kernel_boost_to_lab_frame_cpu!

Inverse of `kernel_boost_to_rest_frame_cpu!` at the same `(r, t_now)`: rest-frame momenta back to the lab. Called once at t0 on the sampled (rest-frame) initial momenta and after every momentum update. Same switches (`relativistic`, `radial_mode`, `V2Evolution`).
"""
function kernel_boost_to_lab_frame_cpu!(
    momenta, positions, xgrid, tgrid, VelocityEvolution,
    m, N, step, Δt, t0;
    radial_mode::Bool = false,
    V2Evolution = nothing, psi2::Float64 = 0.0,
    relativistic::Bool = true          # false ⇒ Galilean: p∥ += m·v, see rest-frame kernel
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
        # optional m=2 azimuthal modulation (elliptic flow); must MATCH the rest-frame boost so the
        # round-trip is exact when no diffusion.  Default (nothing) ⇒ unchanged radial flow.
        if V2Evolution !== nothing && size(positions, 1) >= 2
            v2m = interpolate_2d_cpu(xgrid, tgrid, V2Evolution, r, t_now)
            φ = atan(positions[2, i], positions[1, i])
            v *= (1.0 + 2.0 * v2m * cos(2.0 * (φ - psi2)))
        end

        if !relativistic
            if radial_mode
                momenta[1, i] += m * v
            else
                x = positions[1, i]; y = positions[2, i]
                momenta[1, i] += m * v * (x / r)
                momenta[2, i] += m * v * (y / r)
            end
            continue
        end

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

"""
    kernel_compute_all_forces_cpu!

Per-particle drag and noise of the exact Ornstein–Uhlenbeck step in the rest frame. Reads `T(r, t)`, the drag time `τ_drag(T)` from the spline `(tau_Tmin, tau_invdT, tau_vals)` built by `build_tau_drag_spline` (NOT the current time), sets `η_D = 1/τ_drag`, `κ = 2MT/τ_drag`, and with `η_eff = η_D·M/E` (`relativistic`) or `η_D` writes `deterministic_terms = (e^{−η_eff Δt} − 1)·p` and `stochastic_terms = √κ·√((1−a²)/(2η_eff Δt))·ξ`, so that `kernel_update_momenta_LRF_cpu!` realises the stationary variance `MT` at any Δt. `radial_mode` uses the Euler form with noise projected along p̂.
"""
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
    radial_mode::Bool = false,
    relativistic::Bool = true,
    # 2026-08-25: kick per the particle's PROPER time. The undilated lab-Δt kick makes the
    # stationary state on a flowing background f_J/(γ(1+v v_r)) instead of the boosted Jüttner —
    # a spurious inward ν^r/n ≈ −γ v ⟨v_r²⟩ (the 08-04 "boost case" −0.071 at u^r = 0.5; see
    # AttractorMomentum FIGURE_REGISTRY and Tex/MaxEntHydro README §engine simultaneity
    # convention). With `proper_time_kicks = true` the OU attenuation runs over
    # Δt* = Δt·E*/E_lab = Δt/(γ(1+v·k_r/E*)) — the exact proper-time step — and the exact-OU
    # noise keeps the stationary variance MT at any Δt. Streaming and boosts are untouched
    # (geometry is per lab time). Default false = production byte-identical.
    proper_time_kicks::Bool = false,
    Vfield::Union{Nothing,AbstractMatrix} = nothing
 )
    M = m  # heavy-quark mass (can be parameterized)
    proper_time_kicks && Vfield === nothing &&
        error("kernel_compute_all_forces_cpu!: proper_time_kicks needs the velocity field")

    @inbounds for i in 1:N
        # --- local temperature ---
        r = sqrt(sum(positions[:, i].^2))
        T = interpolate_2d_cpu(xgrid, tgrid, Tfield, r, step * Δt + t0)
        # Temperature should be non-negative; clamp to avoid NaNs from T<0.
        T = max(float(T), 0.0)

        # Guard against invalid transport inputs.
        DsT_safe = DsT <= 0 ? 0.0 : DsT

        # --- transport coefficients ---
        # 🔴 This is the DRAG time τ_drag = M·DsT/T² (Transport.tau_drag), supplied as a
        # precomputed spline in fm by `build_tau_drag_spline`. It is NOT `tau_n_main3`.
        # Until 2026-08-02 this comment claimed the spline held the matched τ_n = D_s z K₃/K₂
        # "the same relaxation time as the MIS-hydro side" — and it did, which is precisely the
        # bug: τ_n is the diffusion-CURRENT time, and using it as 1/η_D applied K₃/K₂ once too
        # often, so the realised D_s was K₃/K₂ (1.26-1.74×) larger than the DsT label.
        # The hydro τ_n is a DERIVED consequence here, not an input: with this drag the ℓ=1
        # mode decays at η_D·K₂/K₃, i.e. exactly Fluidum's τ_diffusion_hadron. See `tau_drag`.
        τ_drag = DsT_safe == 0.0 ? 0.0 : eval_tau_n_spline(T, tau_Tmin, tau_invdT, tau_vals)

        # Einstein / fluctuation-dissipation, both from the same drag:
        #   ηD = 1/τ_drag        [1/fm]      ⇒  D_s = T/(M ηD) = DsT/T
        #   κ  = 2 M T / τ_drag  [GeV^2/fm]  ⇒  D_s = 2T²/κ    (Eq. 11 of 2205.07692)
        ηD = (τ_drag > 0.0 && isfinite(τ_drag)) ? (1.0 / τ_drag) : 0.0
        κ  = (τ_drag > 0.0 && isfinite(τ_drag)) ? ((2.0 * M * T) / τ_drag) : 0.0
        kL = sqrt(κ)
        kT = sqrt(κ)

        ηD_vals[i], kL_vals[i], kT_vals[i] = ηD, kL, kT

        # proper-time dilation of the kick: Δt* = Δt·E*/E_lab = Δt/(γ(1+v·k_r/E*)).
        # k_r is the LRF radial momentum (p_z ⊥ the transverse boost), E* the full LRF energy.
        dil = 1.0
        if proper_time_kicks
            v = interpolate_2d_cpu(xgrid, tgrid, Vfield, r, step * Δt + t0)
            v = clamp(float(v), -0.999999, 0.999999)
            γv = 1.0 / sqrt(1.0 - v * v)
            kr = 0.0
            for d in 1:size(positions, 1)
                kr += momenta[d, i] * positions[d, i]
            end
            kr /= max(r, 1e-12)
            p2_all = 0.0
            for d in 1:dimensions
                p2_all += momenta[d, i]^2
            end
            Estar = sqrt(p2_all + M^2)
            dil = 1.0 / max(γv * (1.0 + v * kr / Estar), 1e-6)
        end

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

            # Relativistic drag: multiply ηD by m/E_LRF so the FP equation
            # ∂f/∂t = ∂/∂p[(κ/2T)(p/E)f + (κ/2)∂f/∂p] gives Jüttner equilibrium.
            # κ = 2mT ηD is unchanged; only the attenuation factor changes.
            E_LRF_r = sqrt(p^2 + M^2)   # p = p_mags[i] is already computed above
            η_eff_r = relativistic ? ηD * M / E_LRF_r : ηD   # non-rel: momentum-independent drag ⇒ Maxwell

            for d in 1:dimensions
                det_term = -η_eff_r * momenta[d, i] * (Δt * dil)

                sto_term = 0.0
                for j in 1:dimensions
                    sto_term += (kL - kT) * p_units[d, i] * p_units[j, i] * ξ[j, i] +
                                kT * (d == j ? 1.0 : 0.0) * ξ[j, i]
                end

                deterministic_terms[d, i] = det_term
                # Euler form: noise variance scales with the proper step ⇒ amplitude × √dil
                # (the update kernel multiplies by √Δt, the LAB step).
                stochastic_terms[d, i]    = sto_term * sqrt(dil)
            end

        else
            # ==========================================
            # 🟢 Cartesian mode: independent px, py
            # ==========================================
            # Exact OU propagator with relativistic drag η_eff = ηD * m/E_LRF.
            # κ = 2mT ηD is unchanged; only the attenuation a = exp(-η_eff dt) changes.
            # This gives Jüttner ∝ exp(-E/T) as the equilibrium (vs Maxwell-Boltzmann
            # for the plain ηD version).
            if ηD > 0
                p2_lrf = 0.0
                for d in 1:dimensions
                    p2_lrf += momenta[d, i]^2
                end
                E_LRF   = sqrt(p2_lrf + M^2)
                η_eff   = relativistic ? ηD * M / E_LRF : ηD   # non-rel: momentum-independent drag ⇒ Maxwell
                # attenuation over the PROPER step Δt·dil; the noise prefactor keeps the LAB Δt in
                # its denominator because the update kernel multiplies by √Δt(lab) — together they
                # realise the exact-OU stationary variance MT for any Δt and any dil.
                a       = exp(-η_eff * Δt * dil)
                noise_pref = (Δt > 0 && η_eff > 0) ?
                                 (kT * sqrt((1 - a * a) / (2 * η_eff * Δt))) : 0.0
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

"""
    kernel_update_momenta_LRF_cpu!

`p += deterministic + √Δt·stochastic` for the `dimensions` momentum rows — the second half of the OU step prepared by `kernel_compute_all_forces_cpu!`.
"""
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

# === Bjorken redshift of the longitudinal momentum (boost-invariant background, u^η = 0) ===
# A particle of rapidity y at space-time rapidity η carries p_z' = m_T sinh(y−η) in the local
# longitudinally-comoving frame, and dη/dτ = tanh(y−η)/τ along its free path, so
#     dp_z'/dτ = −m_T cosh(y−η)·tanh(y−η)/τ = −p_z'/τ        (EXACT, no small-angle step)
# between kicks. The midrapidity ensemble is statistically the ensemble co-moving with one cell,
# so the rule is applied to every particle's p_z row in the LRF: p_z ← p_z·τ_a/τ_b over the step
# τ_a = t0+(step−1)Δt → τ_b = τ_a+Δt. This is the longitudinal work term of θ = 1/τ + ∇_⊥·u_⊥
# that a 2-D momentum run omits. Operator-split from the OU kick at O(Δt²).
"""
    kernel_bjorken_redshift_cpu!

Longitudinal free-streaming of a boost-invariant system between kicks: `p_z ← p_z·τ_a/(τ_a + Δt)` on row `pz_row`, with `τ_a = t0 + (step−1)Δt`. Exact telescoping `⟨p_z²⟩ ∝ 1/τ²` (tested).
"""
function kernel_bjorken_redshift_cpu!(momenta, pz_row::Int, step, Δt, t0, N)
    τa = t0 + (step - 1) * Δt
    τa > 0 || return nothing
    f = τa / (τa + Δt)
    @inbounds for i in 1:N
        momenta[pz_row, i] *= f
    end
    return nothing
end

# === Boltzmann RTA / BGK collision (drop-in replacement for the OU force+update) ===
# Isotropic local-rest-frame Jüttner magnitude sampler P(p*) ∝ p*^(d-1) exp(-(E-m)/T),
# by rejection with an exact-mode envelope.  d = number of momentum components
# (d=1 radial, d=2 the diagnostic's transverse plane, d=3 full).  Mirrors the engine's
# `_draw_juttner_pstar` (minimal_langevin_from_splines.jl) but generalized in d.
@inline function _draw_juttner_pstar_lib(m::Float64, T::Float64, d::Int)
    T <= 0.0 && return 0.0
    a = d - 1
    # exact mode of p^a exp(-(E-m)/T): a T E = p² = E²-m² ⇒ E = (aT + √(a²T²+4m²))/2
    E_peak = (a * T + sqrt(a * a * T * T + 4.0 * m * m)) / 2.0
    p_peak = sqrt(max(E_peak * E_peak - m * m, 1e-12))
    gpeak  = (p_peak^a) * exp(-(E_peak - m) / T)          # a==0 ⇒ p_peak=0, gpeak=1 (0^0=1)
    wmax   = 1.3 * max(gpeak, 1e-300)
    pmax   = 16.0 * sqrt(m * T) + 16.0 * T
    @inbounds for _ in 1:100000
        p = pmax * rand()
        E = sqrt(m * m + p * p)
        w = (p^a) * exp(-(E - m) / T)
        rand() * wmax <= w && return p
    end
    return p_peak
end

# RTA/BGK momentum update in the LRF: with probability Pcol = clamp(Δt/τn(T_local), 0, 1)
# re-draw the momentum from an isotropic Jüttner at the local T; otherwise leave it unchanged.
#
# 🔴 `tau_vals` MUST be the CURRENT-time spline (`build_taun_current_spline`, = tau_n_main3),
# NOT the drag spline the OU kernel takes. BGK relaxes EVERY non-equilibrium moment at 1/τ,
# whereas the OU's ℓ=1 mode decays at η_D·K₂/K₃ = 1/τ_n. So "RTA and Langevin differ ONLY in
# the collision operator" — the intent here — holds in the diffusion-current sector iff the RTA
# is driven by τ_n. Before 2026-08-02 both kernels shared one spline that happened to hold τ_n:
# the RTA was right and the OU was wrong. Fixing the OU inverted that, so the two splines are
# now built and passed separately. Feeding the drag here relaxes the RTA K₃/K₂ too fast.
# (BGK cannot match every moment of the FP operator at once; ℓ=1 is the deliberate choice.)
#
# Operates on rest-frame momenta — call it between kernel_boost_to_rest_frame_cpu! and
# kernel_boost_to_lab_frame_cpu!.
"""
    kernel_rta_collision_cpu!

Boltzmann RTA / BGK step in the rest frame: with probability `Δt/τ_n(T)` re-draw the momentum from the local isotropic Jüttner (`|p*| ∝ p^{d−1} e^{−(E−m)/T}` by rejection, random direction), else leave it. `tau_vals` MUST be the CURRENT-time spline from `build_taun_current_spline` (BGK relaxes every moment at 1/τ; the OU's ℓ=1 mode decays at `η_D K₂/K₃`, so matching the current sector needs τ_n = τ_drag·K₃/K₂). `dimensions` must equal the momentum rows (checked).
"""
function kernel_rta_collision_cpu!(
    Tfield, xgrid, tgrid,
    momenta, positions,
    Δt, m, N, step, t0, DsT;
    tau_Tmin::Float64,
    tau_invdT::Float64,
    tau_vals::AbstractVector{<:Real},
    dimensions::Int = 2,
    radial_mode::Bool = false,
 )
    M = m
    # the loop below is @inbounds on momenta[d, i] for d ≤ dimensions — make that true by contract
    dimensions == size(momenta, 1) || throw(DimensionMismatch(
        "kernel_rta_collision_cpu!: dimensions=$dimensions but momenta has $(size(momenta, 1)) rows"))
    N <= size(momenta, 2) || throw(DimensionMismatch("kernel_rta_collision_cpu!: N=$N > $(size(momenta, 2)) particles"))
    @inbounds for i in 1:N
        r = sqrt(sum(positions[:, i].^2))
        T = max(float(interpolate_2d_cpu(xgrid, tgrid, Tfield, r, step * Δt + t0)), 0.0)
        τn = (DsT > 0.0 && T > 0.0) ? eval_tau_n_spline(T, tau_Tmin, tau_invdT, tau_vals) : 0.0
        # τn ≤ 0 ⇒ infinitely fast relaxation ⇒ always re-draw (DsT→0 local-equilibrium limit)
        Pcol = (τn > 0.0 && isfinite(τn)) ? clamp(Δt / τn, 0.0, 1.0) : 1.0
        rand() < Pcol || continue
        pstar = _draw_juttner_pstar_lib(M, T, dimensions)
        if radial_mode
            momenta[1, i] = (rand() < 0.5 ? -pstar : pstar)     # signed radial magnitude
        else
            # random isotropic direction in `dimensions` components
            nrm2 = 0.0
            for d in 1:dimensions
                g = randn(); momenta[d, i] = g; nrm2 += g * g
            end
            inv = pstar / sqrt(max(nrm2, 1e-300))
            for d in 1:dimensions
                momenta[d, i] *= inv
            end
        end
    end
    return nothing
end

"""
    kernel_update_positions_cpu!

Stream the positions: `x += Δt·p/E` with `E = √(m² + Σp²)` over all `momentum_dimensions` rows (`relativistic`) or `p/m`. Optional overdamped diffusion `√(2D_sΔt)·ξ` with `D_s = DsT/T(r, t)` (`position_diffusion`; double-counts against the underdamped dynamics) and reflection at `r = xgrid[end]` (`reflecting_boundary`). `radial_mode` reflects at `r = 0` and adds the `D/r` drift.
"""
function kernel_update_positions_cpu!(
    positions, momenta, m, Δt, N,step,t0,
    xgrid,tgrid, Tfield,DsT;
    dimensions::Int = 2,
    # number of MOMENTUM rows (3 on a 2-D plane carrying p_z); only E = √(m²+Σp²) sees it — the
    # streaming, the diffusion and the reflection act on the `dimensions` spatial components.
    momentum_dimensions::Int = dimensions,
    radial_mode::Bool = false,
    position_diffusion::Bool = false,
    reflecting_boundary::Bool = false,
    # ⚠ STREAMING IS PART OF THE KINEMATICS, NOT A DETAIL. dx/dt = p/E is the relativistic
    # velocity; the non-relativistic dynamics streams at p/m. Switching only the DRAG and leaving
    # this relativistic gives a hybrid that is neither, and the difference is O(p²/m²) — the same
    # order as the drag correction it is supposed to be switched off with.
    relativistic::Bool = true,
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
        for d in 1:momentum_dimensions
            E2 += momenta[d, i]^2
        end
        E = relativistic ? sqrt(E2) : m       # streaming velocity: p/E (rel) or p/m (non-rel)

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

"""
    kernel_save_snapshot_cpu!

Copy a per-particle scalar snapshot (used by the homogeneous-box path for |p|).
"""
function kernel_save_snapshot_cpu!(
    history_col, snapshot::Vector{Float64}, N::Int
    )
    @inbounds for i in 1:N
        history_col[i] = snapshot[i]
    end
end

"""
    kernel_save_momenta_cpu!

Store `momenta` into `momenta_history[:, :, save_idx]`.
"""
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

"""
    kernel_save_positions_cpu!

Store `positions` into `position_history[:, :, save_idx]`.
"""
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

"""
    kernel_set_to_fluid_velocity_cpu!

Glue the particle to the flow: set the rest-frame momentum to the fluid's `m·γ·v·r̂` (`relativistic`) or `m·v·r̂`. Used when `momentum_langevin = false` or `DsT = 0` (instantaneous equilibration limit).
"""
function kernel_set_to_fluid_velocity_cpu!(
    momenta::Array{Float64,2},
    positions::Array{Float64,2},
    xgrid, tgrid, VelocityEvolution,
    m::Float64,
    N::Int,
    step::Int,
    Δt::Float64,
    t0::Float64;
    radial_mode::Bool = false,
    relativistic::Bool = true          # false ⇒ imprint m·v, not m·γ·v
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
        if !relativistic
            if radial_mode
                momenta[1, i] = m * v
            else
                x = positions[1, i]; y = positions[2, i]
                momenta[1, i] = m * v * (x / r)
                momenta[2, i] = m * v * (y / r)
            end
            continue
        end

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
