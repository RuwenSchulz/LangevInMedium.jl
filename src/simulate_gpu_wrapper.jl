# === simulate_gpu_wrapper.jl ===

# Load required CUDA functionality and modules
using CUDA

# Include GPU-specific source files
include("kernels_gpu.jl")
include("simulate_gpu.jl")

# Import the GPUBackend type from the backends module
using ..Backends: GPUBackend

# Extend the Simulate namespace with a GPU-specific implementation
"""
    Simulate.simulate_ensemble_bulk(backend::GPUBackend, ...)

Wrapper for running Langevin dynamics on the GPU.

This method dispatches the call to `simulate_ensemble_bulk_gpu`, using GPU-accelerated kernels
to evolve particle ensembles over a hydrodynamic background.

# Arguments
- `backend::GPUBackend`: A tag type to select GPU-based execution.
- `T_profile_MIS`, `ur_profile_MIS`, `mu_profile_MIS`: Hydrodynamic field functions.
- `TemperatureEvolutionn`, `VelocityEvolutionn`: Precomputed spacetime evolution arrays.
- `SpaceTimeGrid`: Tuple of coordinate arrays `(x, t)`.

# Keyword Arguments
- `N_particles`: Number of particles (default 10,000).
- `Δt`: Time step (default 0.001).
- `initial_time`: Start time of the simulation.
- `final_time`: End time of the simulation.
- `save_interval`: Interval between saved frames.
- `m`: Particle mass.
- `dimensions`: Number of spatial dimensions (default 3).

# Returns
- `(time_points, momenta_history_vec, position_history_vec)`: Simulation outputs.
"""
function Simulate.simulate_ensemble_bulk(
    backend::GPUBackend,
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
    return simulate_ensemble_bulk_gpu(
        T_profile_MIS, ur_profile_MIS, mu_profile_MIS,
        TemperatureEvolutionn, VelocityEvolutionn, SpaceTimeGrid;
        N_particles = N_particles, Δt = Δt,
        initial_time = initial_time, final_time = final_time,
        save_interval = save_interval, m = m, dimensions = dimensions
    )
end
