module Backends

# === Exported Symbols ===
export GPUBackend, CPUBackend,CPU_GCBackend, GPU_GCBackend

# === Backend Type Hierarchy ===

"""
    AbstractBackend

Abstract supertype for all simulation backends. Used for dispatching different implementations
(e.g., CPU vs GPU) of the same high-level simulation logic.
"""
abstract type AbstractBackend end

"""
    CPUBackend <: AbstractBackend

Represents execution on a CPU. Pass this to simulation dispatch functions to use CPU kernels.
"""
struct CPUBackend <: AbstractBackend end

"""
    GPUBackend <: AbstractBackend

Represents execution on a GPU. Pass this to simulation dispatch functions to use CUDA kernels.
"""
struct GPUBackend <: AbstractBackend end

"""
    CPU_GCBackend <: AbstractBackend

Represents execution on a CPU using general coordinates (e.g., Milne coordinates).
Pass this to simulation dispatch functions to use general-coordinate CPU kernels.
"""
struct CPU_GCBackend <: AbstractBackend end

"""
    GPU_GCBackend <: AbstractBackend

Represents execution on a GPU using general coordinates (e.g., Milne coordinates).
Pass this to simulation dispatch functions to use general-coordinate GPU kernels.
"""
struct GPU_GCBackend <: AbstractBackend end

end # module Backends
