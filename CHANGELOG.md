# Changelog

Entries marked ⚠ changed the default dynamics or the meaning of a label: outputs produced
before them are not comparable to outputs produced after.

## 0.2.0 — 2026-08-22  (benchmark / debug / cleanup pass)

The CPU default dynamics are **bit-identical** to 0.1.0 at commit `30bbbb5` — verified by
`test/regression_corpus.jl` (ten seeded configurations, SHA-256 of every snapshot).

### Fixed
- **GPU interpolant extrapolated beyond the table** (`interpolate_2d_cuda`): the CPU version
  clamps the query point to the tabulated `(r, τ)` domain, the GPU version clamped only the cell
  index. A particle past the rim or a τ beyond the time table read a linearly extrapolated
  T < 0 ⇒ κ < 0 ⇒ √κ = NaN — the whole GPU ensemble went non-finite (200 000/200 000 in the
  adversarial bench) while the CPU stayed finite. Also replaced the `+1e-8` denominator fudge by
  the CPU's degenerate-cell test.
- **GPU boosts did not clamp |v| < 1**: a table cell with `v = 1.02` gave `γ = NaN` on the GPU
  (32 448 NaN particles of 50 000); the CPU clamps to `√(1−10⁻¹²)`. Now identical.
- GPU force kernel: `T ≥ 0` guard and position-diffusion `T ≥ eps` floor, as on the CPU.
- **`time_points` were wrong when `save_interval` did not divide the evolution** (both
  backends): `range(t0, final_time, num_saves+1)` stretched the axis over the dropped trailing
  interval (5295 steps saved every 662 ⇒ snapshots 1.324 fm apart labelled 1.513 fm apart — a 13 %
  error in an MSD slope). `_snapshot_times` now returns the true times and warns once; exactly
  divisible runs keep the historical range bit for bit.
- `save_interval ≥ total time` could give `num_saves = 0` and a crash in `range`; `save_every`
  is clamped to `steps`, and a zero-step run is refused with a message.
- `taun_vals_d` (RTA current spline on the device) was never finalized.
- CPU `simulate_ensemble_bulk` silently ignored `integrator_mode = 1` (drift-midpoint exists on the
  GPU only) and any unknown `collision_mode`; both are now errors.
- `kernel_rta_collision_cpu!` checks its `dimensions` kwarg against the momentum rows (it was
  `@inbounds` on them).
- `simulate_ensemble_bulk(GPUBackend(), …)` without `using CUDA` raises a message that says so
  instead of a bare `MethodError`.
- `print_cuda_status()` no longer runs on every GPU call (`verbose = true` to get it back).

### Removed
- The general-coordinates (Milne) backends `CPU_GCBackend`/`GPU_GCBackend` and their four files
  (1 143 lines): no consumer, no test, `compute_christoffel` returned zeros, `Int(save/Δt)`
  crash, a stale copy of the pre-clamp interpolant under the same exported name. Milne physics is
  served by the main path's `bjorken_redshift` + radial mode.
- `src/require_cuda.jl` (dead loader), `sample_initial_particles_milne!`.
  (`src/data/Fluidum_MIS_HQ.jld2` stays: the package does not read it, but it is the shared Fluidum
  MIS background that `FokkerPlank1D/2D`, `FiVoHydro/main2.jl` and `CompareBoltzmannHQ` load from
  this path.)
- Phantom exports `n_rt`, `plot_n_rt_comparison_hydro_langevin`, `compute_MIS_distribution`
  (none was defined).
- Dependencies with no remaining use: Plots, LaTeXStrings, Measures, DataStructures, Printf,
  SpecialFunctions, Dierckx, JLD2, StaticArrays, KernelAbstractions, StatsBase.

### Added
- `test/regression_corpus.jl` + `regression_corpus_baseline.txt` (bit-identity gate).
- `bench/`: `bench_physics_gates.jl` (15 gates in physical units), `bench_gpu_parity.jl`
  (21 gates, incl. the adversarial inputs above), `bench_throughput.jl`, results under `bench/results/`.
- `LIM_FAST=1` test mode; tests for `DsT_quad`, `_snapshot_times`, the entry-point contract,
  the `:rta` path and the sampler; tests use the package's ħc instead of a private constant.
- Docstrings on every public method and a module docstring; this README and CHANGELOG.
- `sample_particles_from_FONLL` is exported.

### Known / deferred
- The CPU and GPU kernels remain hand-kept twins (~560 duplicated lines); a
  KernelAbstractions unification is the structural fix. `bench_gpu_parity.jl` is the guard meanwhile.
- Requires.jl → package extension (`ext/`) would precompile the GPU path; deferred because it
  changes when `using CUDA` must happen for every consumer.
- No `seed` kwarg (CPU: `Random.seed!`; GPU: CURAND, unseedable).
- `freezeout_capture` returns a different shape than the histories.
- `sample_initial_particles_from_pdf!` uses `searchsortedfirst` on bin centres (off-by-one at
  the first bin); untouched because it changes sampled ICs.
- The pre-point relativistic drag carries an O(ηΔt) bias on ⟨p²⟩ (≈ −1 % at ηΔt = 0.1).

## 0.1.0 — history reconstructed from the in-source notes

- 2026-08-21 ⚠ `momentum_dimensions = 3` (+ `bjorken_redshift`): a longitudinal `p_z` on the
  2-D transverse plane; default 0 is bit-identical to the coupled engine. `DsT_quad`/`DsT_Tref`.
- 2026-08-16 ⚠ `relativistic = false` now switches the **boosts** too (Galilean `p∥ ∓ m·v`);
  before, Galilean runs used Lorentz boosts — a kinematic hybrid at O(T/M) on a flowing background.
- 2026-08-06 ⚠ `relativistic = false` now switches the **streaming** (`p/m`); before, only the drag.
- 2026-08-03 ⚠ RTA uses the **current** time `tau_n_main3`, not the drag (`build_taun_current_spline`);
  `langevin_*_rta` products from before need regenerating for the opposite reason to everything else.
- 2026-08-02 ⚠⚠ **The drag is `tau_drag = m·DsT/T²`, not `tau_n_main3`.** Every product before
  this realised `D_s = K₃/K₂ × label` (1.26–1.74×). Regression guard in `test/runtests.jl`.
- 2026-07-16 ħc standardised to PDG 0.197327 (was 1/5.068); matched τ_n; Jüttner inverse-CDF;
  GPU freeze-out capture; GPU RTA.
