#!/usr/bin/env julia
# ==============================================================================================
# 05 — THE GPU PATH AND ON-THE-FLY FREEZE-OUT (the production pattern)
#
# The shape a real campaign takes. Two things this shows that no other example does:
#
#   · `freezeout_capture = true` (GPU only). Each particle LATCHES its own state the step its local
#     T first crosses Tfo, and the driver returns a NamedTuple `(pos, mom, tau, flag)` instead of
#     histories. Memory is then ∝ N instead of ∝ N·(saves+1) — at N = 10⁶ with 500 snapshots the
#     history alone is 24 GB on the device, which is why the latch exists — and the crossing is
#     resolved to Δt rather than to the save cadence. `freezeout_interp = true` (the default) books
#     the interpolated crossing, measured accurate to 2.7e-15; `false` books the first sampled step
#     below Tfo, which is O(Δt) late and predictably so.
#   · the run does NOT stop at freeze-out. The latch records; the particle keeps going. That is
#     deliberate — a Cooper–Frye surface can be crossed more than once — and worth knowing before
#     you interpret `tau`.
#
# Falls back to the CPU (without the latch) when CUDA is unavailable, so it always runs.
#
#   julia --project=Julia Julia/LangevInMedium.jl/examples/05_gpu_freezeout.jl
# ==============================================================================================
include(joinpath(@__DIR__, "example_common.jl"))

const M   = 1.5
const DST = 0.11634
const TFO = 0.156
const τ0, τf = 0.4, 15.0

have_gpu = try
    @eval using CUDA
    Base.invokelatest(() -> CUDA.functional())
catch
    false
end
N = have_gpu ? 500_000 : 50_000
println(have_gpu ? "CUDA is functional — running the GPU path with the freeze-out latch, N = $N" :
                   "no usable CUDA — falling back to the CPU without the latch, N = $N")

xg, tg, Tf, Vf = bjorken_fireball(; τ0 = τ0, τf = τf)
r_grid = collect(range(0.0, 20.0; length = 150))
p_grid = collect(range(0.0, 10.0; length = 300))
density = fonll_density(r_grid, p_grid; σ = 3.0)

common = (; N_particles = N, Δt = 5e-3, initial_time = τ0, final_time = τf, save_interval = 0.5,
            m = M, DsT = DST, dimensions = 2, momentum_dimensions = 3, bjorken_redshift = true,
            cartesian_spatial_sampling = false,   # polar inverse CDF; the Cartesian mode's
                                                  # acceptance is the fireball's area fraction
            proper_time_kicks = true, Tfo = TFO)

if have_gpu
    # ── the latch ──────────────────────────────────────────────────────────────────────────────
    # Returns a NamedTuple, NOT (t, mom, pos). `flag[i] > 0` means particle i crossed; `tau[i]` is
    # its crossing time, `pos[:, i]` / `mom[:, i]` its state there.
    Random.seed!(5)
    fo = Base.invokelatest(simulate_ensemble_bulk, GPUBackend(), r_grid, p_grid, density,
        Tf, Vf, (xg, tg); freezeout_capture = true, freezeout_interp = true, common...)
    crossed = fo.flag .> 0
    nfo = count(crossed)
    pT  = vec(sqrt.(sum(abs2, view(fo.mom, 1:2, :); dims = 1)))
    r   = vec(sqrt.(sum(abs2, fo.pos; dims = 1)))
    @printf("\nlatched: %d / %d (%.1f %%) crossed T = %.3f GeV by τ = %.1f fm\n",
            nfo, N, 100nfo / N, TFO, τf)
    @printf("  ⟨τ_fo⟩ = %.3f fm (sd %.3f)   ⟨r_fo⟩ = %.3f fm   ⟨p_T⟩ = %.4f GeV   ⟨p_z*⟩² = %.5f\n",
            mean(fo.tau[crossed]), std(fo.tau[crossed]), mean(r[crossed]),
            mean(pT[crossed]), mean(view(fo.mom, 3, crossed) .^ 2))
    # what the latch buys, in bytes
    hist_bytes = 3 * N * (Int(floor((τf - τ0) / 0.5)) + 1) * 8 + 2 * N * (Int(floor((τf - τ0) / 0.5)) + 1) * 8
    latch_bytes = (3 + 2 + 1 + 1) * N * 8
    @printf("  device memory: full history ≈ %.2f GB   vs the latch ≈ %.2f MB   (%.0f×)\n",
            hist_bytes / 2^30, latch_bytes / 2^20, hist_bytes / latch_bytes)
    println("""
  ⚠ the run does NOT stop at the crossing — the latch records and the particle keeps evolving.
    `tau` is the FIRST crossing, which is what a Cooper–Frye surface wants, but a particle that
    re-enters the hot region is still being propagated afterwards.""")
    if plots_on()
        edges = range(0.0, 6.0; length = 31)
        c, h = hist(pT[crossed], edges)
        pa = plot(c, max.(h, 1e-6); m = :circle, c = :firebrick, yscale = :log10,
                  xlabel = "p_T [GeV]", ylabel = "1/N dN/dp_T", label = "at the T = Tfo crossing",
                  title = "freeze-out spectrum (GPU latch)", ylims = (1e-5, 3))
        pb = histogram(fo.tau[crossed]; bins = 40, c = :seagreen, xlabel = "τ_fo [fm]",
                       ylabel = "particles", label = "", title = "crossing time, resolved to Δt")
        pc = scatter(r[crossed][1:200:end], fo.tau[crossed][1:200:end]; ms = 2, c = :steelblue,
                     xlabel = "r_fo [fm]", ylabel = "τ_fo [fm]", label = "",
                     title = "the freeze-out surface")
        savefig_ex(plot(pa, pb, pc; layout = (1, 3), size = (1500, 430)), "05_gpu_freezeout.png")
    end
else
    Random.seed!(5)
    t, mom, pos = simulate_ensemble_bulk(CPUBackend(), r_grid, p_grid, density, Tf, Vf, (xg, tg);
        common...)
    t = collect(t)
    pT = vec(sqrt.(sum(abs2, view(mom[end], 1:2, :); dims = 1)))
    @printf("\nCPU fallback: %d snapshots, ⟨p_T⟩ at τ = %.1f fm is %.4f GeV\n",
            length(t), t[end], mean(pT))
    println("  (the on-the-fly latch is GPU-only; example 02 shows the snapshot-based alternative)")
    if plots_on()
        edges = range(0.0, 6.0; length = 31)
        c, h = hist(pT, edges)
        pa = plot(c, max.(h, 1e-6); m = :circle, c = :firebrick, yscale = :log10,
                  xlabel = "p_T [GeV]", ylabel = "1/N dN/dp_T", label = "final",
                  title = "p_T spectrum (CPU fallback)", ylims = (1e-5, 3))
        savefig_ex(pa, "05_gpu_freezeout.png")
    end
end
