# === simulate_gpu_wrapper.jl ===

# Load required CUDA functionality and modules
#using CUDA

# Include GPU-specific source files
include("kernels_gpu.jl")
include("kernels_gpu_GC.jl")
include("simulate_gpu.jl")
include("simulate_gpu_general_coords.jl")
# Import the GPUBackend type from the backends module
using ..Backends: GPUBackend, GPU_GCBackend


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
    dimensions::Int = 3,
    cartesian_spatial_sampling::Union{Nothing,Bool} = nothing,
    antithetic_momenta::Bool = false,
    position_diffusion::Bool = false,
    momentum_langevin::Bool = true,
    reflecting_boundary::Bool = false,
)
    return SimulateGPU.simulate_ensemble_bulk_gpu(   
        r_grid_Langevin,p_grid_Langevin,heavy_quark_density,
        TemperatureEvolutionn, VelocityEvolutionn, SpaceTimeGrid;
        N_particles = N_particles, Δt = Δt,
        initial_time = initial_time, final_time = final_time,
        save_interval = save_interval, m = m, DsT = DsT,
        DsT_linear = DsT_linear, DsT_slope = DsT_slope,
        DsT_offset = DsT_offset, Tfo = Tfo,
        dimensions = dimensions,
        cartesian_spatial_sampling = cartesian_spatial_sampling,
        antithetic_momenta = antithetic_momenta,
        position_diffusion = position_diffusion,
        momentum_langevin = momentum_langevin,
        reflecting_boundary = reflecting_boundary,
    )
end

function Simulate.simulate_ensemble_bulk(
    backend::GPU_GCBackend,
    T_profile_MIS,
    ur_profile_MIS,
    mu_profile_MIS,
    TemperatureEvolutionn,
    VelocityEvolutionn,
    SpaceTimeGrid;
    N_particles::Int = 10_000,
    Δt::Float64 = 0.001,
    initial_time::Float64 = 0.0,
    final_time::Float64 = 1.0,
    save_interval::Float64 = 0.1,
    m::Float64 = 1.0,
    dimensions::Int = 3,
)
    return SimulateGPUGeneralCoords.simulate_ensemble_bulk_gpu_general_coords(
        T_profile_MIS, ur_profile_MIS, mu_profile_MIS,
        TemperatureEvolutionn, VelocityEvolutionn, SpaceTimeGrid;
        N_particles = N_particles, Δt = Δt,
        initial_time = initial_time, final_time = final_time,
        save_interval = save_interval, m = m, dimensions = dimensions
    )
end

@info "GPU support loaded. Use `simulate_ensemble_bulk(GPUBackend(), ...)` to run GPU-accelerated Langevin dynamics."