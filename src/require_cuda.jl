#using ProgressMeter
#using ..Backends: AbstractBackend , GPUBackend
#using ..Utils
# ────────────────────────────────────────────────
# GPU Backend (Optional) — loaded only if CUDA.jl is available

#This allows seamless CPU/GPU flexibility while keeping dependencies optional.
begin
   #
    if CUDA.functional()
        #using CUDA
        include("simulate_gpu_wrapper.jl")  # Loads simulate_ensemble_bulk(::GPUBackend, ...) overload
        @info "SimulateGPU backend loaded."
    else
        @warn "CUDA.jl is installed, but CUDA.functional() is false. GPU backend will not be available."
    end
end 
