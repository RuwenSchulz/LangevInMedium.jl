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
    integrator_mode::Int = 0,   # accepted for signature parity with the GPU path; CPU ignores it (pre-point only)
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
)
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

