# === simulate_gpu_wrapper.jl ===
# Included into `Simulate` by the Requires hook once CUDA is loaded: defines the GPU kernels, the
# GPU driver and the `simulate_ensemble_bulk(::GPUBackend, …)` method.

include("kernels_gpu.jl")
include("simulate_gpu.jl")
using ..Backends: GPUBackend

"""
    simulate_ensemble_bulk(::GPUBackend, r_grid, p_grid, density, T_field, v_field, (xgrid, tgrid); kwargs...)

CUDA twin of the CPU method (same keywords, same algorithm, same return shape) with the GPU-only
extras `freezeout_capture`, `freezeout_interp`, `integrator_mode` and `verbose`. Sampling and the
p_z completion run on the host; the per-step kernels and the snapshot history live on the device
and are downloaded once at the end. Only `:langevin` and `:rta` collision modes.
"""
function Simulate.simulate_ensemble_bulk(
    backend::GPUBackend,
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
    freezeout_capture::Bool = false,
    freezeout_interp::Bool = true,
    integrator_mode::Int = 0,
    relativistic::Bool = true,  # drag: rel ·m/E (Jüttner) vs non-rel ηD (Maxwell)
    momentum_dimensions::Int = 0,   # p_z on the transverse plane (3 with dimensions=2); utils.jl note
    bjorken_redshift::Bool = false,
    proper_time_kicks::Bool = false,
    pz_init::Symbol = :thermal,
    track_eta_s::Bool = false,
    verbose::Bool = false,
)
    (collision_mode == :langevin || collision_mode == :rta || collision_mode == :none) ||
        error("simulate_ensemble_bulk(::GPUBackend): collision_mode=$(collision_mode) is not supported on GPU (only :langevin, :rta and :none).")
    return SimulateGPU.simulate_ensemble_bulk_gpu(
        r_grid_Langevin,p_grid_Langevin,heavy_quark_density,
        TemperatureEvolutionn, VelocityEvolutionn, SpaceTimeGrid;
        N_particles = N_particles, Δt = Δt,
        initial_time = initial_time, final_time = final_time,
        save_interval = save_interval, m = m, DsT = DsT,
        DsT_linear = DsT_linear, DsT_slope = DsT_slope,
        DsT_offset = DsT_offset, Tfo = Tfo,
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
        freezeout_capture = freezeout_capture,
        freezeout_interp = freezeout_interp,
        integrator_mode = integrator_mode,
        relativistic = relativistic,
        momentum_dimensions = momentum_dimensions,
        bjorken_redshift = bjorken_redshift,
        proper_time_kicks = proper_time_kicks,
        pz_init = pz_init,
        track_eta_s = track_eta_s,
        verbose = verbose,
    )
end
