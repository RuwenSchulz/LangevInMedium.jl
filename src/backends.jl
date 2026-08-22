module Backends

export GPUBackend, CPUBackend

"""
    AbstractBackend

Supertype of the backend singletons that select the implementation of `simulate_ensemble_bulk`.
"""
abstract type AbstractBackend end

"""
    CPUBackend <: AbstractBackend

Run on the CPU (single-threaded loops over particles). Always available; seedable through
`Random.seed!` — every draw goes through the task-local default RNG.
"""
struct CPUBackend <: AbstractBackend end

"""
    GPUBackend <: AbstractBackend

Run on a CUDA device. The methods exist only after `using CUDA` (they are attached by a
Requires.jl hook); the device RNG (CURAND) is not seedable, so GPU runs are reproducible in
their ensemble moments, not bit for bit. Extras over the CPU path: on-the-fly freeze-out capture
(`freezeout_capture`) and the drift-midpoint momentum step (`integrator_mode = 1`).
"""
struct GPUBackend <: AbstractBackend end

end # module Backends
