module Backends

export GPUBackend, CPUBackend

abstract type AbstractBackend end
    struct CPUBackend <: AbstractBackend end
    struct GPUBackend <: AbstractBackend end

end
