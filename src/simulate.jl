module Simulate

using ..Backends: AbstractBackend, CPUBackend, GPUBackend
using ..Utils
using ..SimulateCPU

export simulate_ensemble_bulk

# ────────────────────────────────────────────────
# `simulate_ensemble_bulk` dispatches on the backend singleton. The CPU methods are defined here;
# the GPU methods are added to this module by simulate_gpu_wrapper.jl from the Requires hook
# in __init__ (i.e. only after `using CUDA`).

"""
    simulate_ensemble_bulk(backend, r_grid, p_grid, density, T_field, v_field, (xgrid, tgrid); kwargs...)

Propagate `N_particles` heavy quarks from `initial_time` to `final_time` in steps `Δt` on the
tabulated background and return `(time_points, momenta, positions)`, where `momenta[k]` is the
`(momentum rows, N)` matrix at `time_points[k]` (likewise `positions`, `(dimensions, N)`).

Background: `T_field[i, j] = T(xgrid[i], tgrid[j])` in GeV, `v_field` the radial flow velocity
(c = 1); both are bilinearly interpolated in `(r, τ)` and clamped to the table at its edges.

Initial condition: `x_init`/`p_init` (`(2, N)` each, lab positions and LOCAL-REST-FRAME momenta;
the initial lab boost is applied inside) or, when absent, `N_particles` samples from
`density[p_index, r_index]` on `(r_grid, p_grid)` via `sample_particles_from_FONLL`
(`cartesian_spatial_sampling`, `antithetic_momenta`).

Keywords (defaults in `simulate_cpu.jl`/`simulate_gpu.jl`):
- `m` quark mass [GeV]; `DsT` the spatial diffusion coefficient `2πT D_s`-style label `D_s·T`
  (dimensionless). The drag is the Einstein relation `1/η_D = tau_drag = m·DsT/T²` — never
  `tau_n_main3`, which is the derived diffusion-CURRENT time (see `tau_drag`).
- `DsT_linear, DsT_slope, DsT_offset, Tfo`: `DsT(T) = slope·max(T, Tfo) + offset`;
  `DsT_quad, DsT_Tref`: `DsT(T) = DsT·(T/Tref)²`, i.e. a T-independent drag time.
- `dimensions`: 2 = transverse plane (x, y); 1 = radial mode (r, p_r only).
  `momentum_dimensions = 3` with `dimensions = 2` adds a longitudinal `p_z` row (thermal
  conditional at t0, invariant under the transverse boost); `bjorken_redshift` then applies
  `dp_z/dτ = −p_z/τ` between kicks (needs `initial_time > 0`).
- `relativistic`: `true` = Jüttner kinematics (drag `·m/E`, streaming `p/E`, Lorentz boosts);
  `false` = the exactly solvable Galilean process (drag `η_D`, streaming `p/m`, `p∥ ∓ m·v`).
- `collision_mode`: `:langevin` (exact-OU step) or `:rta` (BGK re-draw from the local Jüttner
  with probability `Δt/τ_n`, τ_n the CURRENT time).
- `momentum_langevin = false` or `DsT = 0`: particles are glued to the flow.
- `position_diffusion`: extra overdamped `√(2D_sΔt)` kicks on the positions (off: the
  underdamped dynamics already diffuses); `reflecting_boundary`: reflect at `r = xgrid[end]`.
- `V2Evolutionn, psi2`: optional elliptic modulation `v → v(1 + 2v₂cos2(φ−Ψ₂))`.
- `save_interval`: snapshot cadence; if it does not divide the evolution the trailing steps
  are not in the history and `time_points` says so (a warning is issued once).
- GPU only: `freezeout_capture`/`freezeout_interp` (returns a NamedTuple of the T = Tfo crossing
  instead of histories), `integrator_mode = 1` (drift-midpoint drag), `verbose`.

Reproducibility: the CPU path is bit-reproducible under `Random.seed!`; see
`test/regression_corpus.jl`.
"""
function simulate_ensemble_bulk(
    backend::CPUBackend,
    r_grid_Langevin,
    p_grid_Langevin,
    heavy_quark_density,
    TemperatureEvolutionn,
    VelocityEvolutionn,
    SpaceTimeGrid;
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
    integrator_mode::Int = 0,   # the CPU path implements only the pre-point exact-OU step (0); 1 exists on the GPU only
    # `relativistic` switches the KINEMATICS, not just the drag:
    #   drag       η_D = η m/E   (Jüttner equilibrium)   vs   η_D = η        (Maxwell)
    #   streaming  dx/dt = p/E                           vs   dx/dt = p/m
    #   boost      Lorentz                               vs   Galilean p∥ ∓ m·v
    # (The boost switch landed 2026-08-15 — see the HISTORY note above kernel_boost_to_lab_frame_cpu!
    # in kernels_cpu.jl; before that, `false` was a kinematic hybrid at O(T/M) on a flowing
    # background. Post-fix, `false` is the exactly solvable Galilean process.)
    relativistic::Bool = true,    # p_z on the transverse plane (3 with dimensions=2) and its Bjorken redshift — utils.jl note.
    momentum_dimensions::Int = 0,
    bjorken_redshift::Bool = false,
    proper_time_kicks::Bool = false,  # OU kick per proper time Δt·E*/E_lab (see kernels_cpu.jl); false = production
    verbose::Bool = false,        # accepted for signature parity with the GPU path (prints nothing on the CPU)
)
    integrator_mode == 0 ||
        error("simulate_ensemble_bulk(::CPUBackend): integrator_mode=$(integrator_mode) is not implemented on the CPU (only 0, the pre-point exact-OU step); the drift-midpoint variant (1) exists on the GPU path only.")
    (collision_mode == :langevin || collision_mode == :rta) ||
        error("simulate_ensemble_bulk(::CPUBackend): collision_mode=$(collision_mode) is not supported (only :langevin and :rta).")
    return simulate_ensemble_bulk_cpu(r_grid_Langevin,p_grid_Langevin,heavy_quark_density,
        TemperatureEvolutionn, VelocityEvolutionn, SpaceTimeGrid;
        N_particles = N_particles, Δt = Δt,
        initial_time = initial_time, final_time = final_time,
        save_interval = save_interval, m = m, DsT = DsT,
        DsT_linear = DsT_linear, DsT_slope = DsT_slope, DsT_offset = DsT_offset, Tfo = Tfo,
        DsT_quad = DsT_quad, DsT_Tref = DsT_Tref,
        dimensions = dimensions,
        cartesian_spatial_sampling = cartesian_spatial_sampling,
        antithetic_momenta = antithetic_momenta,
        position_diffusion = position_diffusion,
        momentum_langevin = momentum_langevin,
        reflecting_boundary = reflecting_boundary,
        collision_mode = collision_mode,
        x_init = x_init,
        p_init = p_init,
        V2Evolutionn = V2Evolutionn,
        psi2 = psi2,
        relativistic = relativistic,
        momentum_dimensions = momentum_dimensions,
        bjorken_redshift = bjorken_redshift,
        proper_time_kicks = proper_time_kicks,
    )
end

"""
    simulate_ensemble_bulk(::CPUBackend, T; N_particles, Δt, initial_time, final_time, save_interval, m, p0, DsT, initial_condition, dimensions)

Homogeneous box at temperature `T` with no positions: momenta only, started from `"delta"`
(all |p| = p0, isotropic) or `"bimodal"`. Returns `(time_points, |p| snapshots)`. Toy path for
relaxation studies; its force kernel uses a fixed `κ = 2.5 T³` and ignores `DsT`.
"""
function simulate_ensemble_bulk(
    backend::CPUBackend,
    T::Float64;
    N_particles::Int = 10_000,
    Δt::Float64 = 0.001,
    initial_time::Float64 = 0.0,
    final_time::Float64 = 1.0,
    save_interval::Float64 = 0.1,
    m::Float64 = 1.0,
    p0 = 1.0,
    DsT = 0.2,
    initial_condition = "delta",
    dimensions::Int = 3,
)
    return simulate_ensemble_bulk_cpu(
        T;
        N_particles = N_particles, Δt = Δt,
        initial_time = initial_time, final_time = final_time,
        save_interval = save_interval, m = m,dimensions = dimensions,initial_condition = initial_condition, p0 = p0
    )
end

# ────────────────────────────────────────────────
# GPU backend — attached when CUDA is loaded (Requires.jl). Not precompiled: the GPU files are
# parsed at `using CUDA` time. Set ENV["LIM_QUIET"] = "1" to silence the load notices.
using Requires

function __init__()
    @require CUDA="052768ef-5323-5732-b1bb-66c8b64840ba" begin
        using CUDA
        include("simulate_gpu_wrapper.jl")
        get(ENV, "LIM_QUIET", "0") == "1" || @info "LangevInMedium: CUDA loaded — simulate_ensemble_bulk(GPUBackend(), …) is available."
    end
end

# Without `using CUDA` the Requires hook above never runs and the GPU methods do not exist; a bare
# MethodError would send the user to the wrong place. The 7-positional-argument GPU method defined
# by the hook is more specific than this catch-all, so it wins whenever CUDA is loaded.
function simulate_ensemble_bulk(backend::GPUBackend, args...; kwargs...)
    error("""
    simulate_ensemble_bulk(::GPUBackend, ...) was called, but GPU support is not loaded.

    The GPU path is attached at runtime by Requires.jl: run `using CUDA` BEFORE calling into
    LangevInMedium (CUDA.jl must be installed and `CUDA.functional()` must be true). On a machine
    without a usable GPU, pass `CPUBackend()` instead.
    """)
end

end # module Simulate

