module LangevInMedium

using StaticArrays, LinearAlgebra, JLD2, Dierckx

# ─────────────────────────────────────────────────────────────
# LangevInMedium.jl
#
# A framework for simulating Langevin dynamics in arbitrary
# Cartesian dimensions, where particles interact with a
# dynamic medium defined by temperature and flow velocity fields.
#
# Features:
# - Arbitrary dimensionality (1D, 2D, 3D, ...)
# - Background temperature and velocity evolution
# - Ensemble-based simulations
# - Supports CPU and (optionally) GPU backends
# - Modular and extensible
#
# Entry point: `simulate_ensemble_bulk(::CPUBackend/GPUBackend, ...)`
# ─────────────────────────────────────────────────────────────

# Load internal modules
include("constants.jl")         # Physical constants and parameters
include("backends.jl")          # CPU/GPU backend markers
include("utils.jl")             # Utilities: plotting, interpolation, etc.
include("kernels_cpu.jl")       # CPU-side numerical integration kernels
include("simulate_cpu.jl")      # High-level CPU simulation loop
include("simulate_cpu_general_coords.jl")      # High-level CPU simulation loop
include("simulate.jl")          # Unified frontend with conditional GPU logic

# Load submodules
using .Constants
using .Backends
using .Utils
using .Simulate

# Public API
export simulate_ensemble_bulk          # Main simulation function
export n_rt                            # Observable extraction (density vs r, t)
export plot_n_rt_comparison_hydro_langevin  # Plotting helper
export CPUBackend, GPUBackend, CPU_GCBackend          # Backend selectors
export fmGeV, GevInvTofm  # Physical constants
end # module LangevInMedium
