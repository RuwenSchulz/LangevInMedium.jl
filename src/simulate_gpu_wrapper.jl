# simulate_gpu_wrapper.jl



using CUDA
include("kernels_gpu.jl")
include("simulate_gpu.jl")

using ..Backends: GPUBackend

# Now bring in the simulate_ensemble_bulk_gpu method
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
