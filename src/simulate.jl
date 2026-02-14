module Simulate

using ProgressMeter
using ..Backends: AbstractBackend, CPUBackend, GPUBackend, CPU_GCBackend
using ..Utils
using ..SimulateCPU
using ..SimulateCPUGeneralCoords

export simulate_ensemble_bulk

# ────────────────────────────────────────────────
# Simulation Interface (Backend-dependent dispatch)
#
# This module provides an entry point `simulate_ensemble_bulk` that delegates
# to the appropriate backend (CPU or GPU) based on the type of the first argument.
# GPU support is optional and loaded via Requires.jl only if CUDA is available.

# ────────────────────────────────────────────────
# CPU Backend — always available

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
    dimensions::Int = 3,
    position_diffusion::Bool = false,
    momentum_langevin::Bool = true,
)
    return simulate_ensemble_bulk_cpu(r_grid_Langevin,p_grid_Langevin,heavy_quark_density,
        TemperatureEvolutionn, VelocityEvolutionn, SpaceTimeGrid;
        N_particles = N_particles, Δt = Δt,
        initial_time = initial_time, final_time = final_time,
        save_interval = save_interval, m = m, DsT = DsT, dimensions = dimensions,
        position_diffusion = position_diffusion,
        momentum_langevin = momentum_langevin
    )
end

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

"""
    simulate_ensemble_bulk(::CPU_GCBackend, ...)

Run a bulk Langevin simulation using the CPU backend in general coordinates (e.g., Milne).
Dispatches to `simulate_ensemble_bulk_cpu` with general-coordinate logic.
"""
function simulate_ensemble_bulk(
    backend::CPU_GCBackend,
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
    DsT::Float64 = 0.2,
    dimensions::Int = 2,  # Milne: τ, r
)
    return simulate_ensemble_bulk_general_coords_cpu(
        T_profile_MIS, ur_profile_MIS, mu_profile_MIS,
        TemperatureEvolutionn, VelocityEvolutionn, SpaceTimeGrid;
        N_particles = N_particles, Δt = Δt,
        initial_time = initial_time, final_time = final_time,
        save_interval = save_interval, m = m, DsT = DsT, dimensions = dimensions
    )
end



# ────────────────────────────────────────────────
# GPU Backend (Optional) — loaded only if CUDA.jl is available
using Requires


function __init__()
    @info "In order to use GPU functionality execute: using CUDA (CUDA must be installed for that: Pkg.add(\"CUDA\") )"
    @require CUDA="052768ef-5323-5732-b1bb-66c8b64840ba" begin
        @info "CUDA is available — loading GPU simulation support."
        using CUDA
        # Add your code here, e.g.:
        include("simulate_gpu_wrapper.jl")
    end
end

## Fallback defined *after* the module is closed:
#function Simulate.simulate_ensemble_bulk(
#    backend::GPUBackend,
#    T_profile_MIS,
#    ur_profile_MIS,
#    mu_profile_MIS,
#    TemperatureEvolutionn,
#    VelocityEvolutionn,
#    SpaceTimeGrid;
#    N_particles::Int = 10_000,
#    Δt::Float64 = 0.001,
#    initial_time::Float64 = 0.0,
#    final_time::Float64 = 1.0,
#    save_interval::Float64 = 0.1,
#    m::Float64 = 1.0,
#    dimensions::Int = 3,
#)
#    error("""
#    simulate_ensemble_bulk(::GPUBackend, ...) was called,
#    but GPU support is not available.
#
#    To enable GPU functionality:
#    - Install CUDA.jl: `pkg> add CUDA`
#    - Ensure CUDA.functional() returns true on your system
#    """)
#end


end # module Simulate

