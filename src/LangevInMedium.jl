"""
    LangevInMedium

Relativistic Langevin dynamics of heavy quarks in an evolving medium. An ensemble of particles
is propagated on a tabulated background `T(r, τ)`, `v_r(r, τ)` (typically a hydro output): each
step boosts the momenta into the local fluid rest frame, applies the exact Ornstein–Uhlenbeck
propagator for the drag `η_D = T²/(M·D_sT)·(M/E)` with the matching Einstein noise `κ = 2MTη_D`,
boosts back and streams the positions with `dx/dt = p/E`. CPU and CUDA backends run the same
algorithm; the GPU path is attached at runtime by `using CUDA` (Requires.jl).

One public entry point, dispatched on the backend singleton:

    simulate_ensemble_bulk(CPUBackend() | GPUBackend(), r_grid, p_grid, density, T_field, v_field, (xgrid, tgrid); kwargs...)
    simulate_ensemble_bulk(CPUBackend(), T; kwargs...)          # homogeneous box, momenta only

returning `(time_points, momenta_snapshots, position_snapshots)`. See `README.md` for the keyword
table, the drag-vs-current distinction (`tau_drag` vs `tau_n_main3`), the `relativistic`,
`momentum_dimensions` and `DsT_*` switches, and the regression/benchmark suites.
"""
module LangevInMedium

include("constants.jl")         # ħc and the GeV⁻¹ ↔ fm conversion
include("backends.jl")          # CPUBackend / GPUBackend singletons
include("utils.jl")             # initial-condition samplers, the p_z completion
include("transport.jl")         # τ_drag, τ_n (current), D_sT prescriptions, Jüttner inverse CDF
include("kernels_cpu.jl")       # the per-step CPU kernels
include("simulate_cpu.jl")      # CPU driver
include("simulate.jl")          # public dispatch + the Requires hook that loads the GPU files

using .Constants
using .Backends
using .Utils
using .Transport
using .Simulate

export simulate_ensemble_bulk
export CPUBackend, GPUBackend
export fmGeV, GevInvTofm
export tau_n_main3, tau_drag, build_tau_drag_spline, build_taun_current_spline, eval_tau_n_spline, effective_DsT
export sample_particles_from_FONLL

end # module LangevInMedium
