# Changelog

Entries marked ⚠ changed the default dynamics or the meaning of a label: outputs produced
before them are not comparable to outputs produced after.

## 0.2.1 — 2026-08-31  (full audit: tests for every function, exact CPU/GPU parity, hot-loop pass)

The CPU default dynamics are **bit-identical to 0.2.0** — all ten `regression_corpus.jl` hashes
reproduce. Nothing in this release changes a number any product depends on. Three real defects were
found that *would*; they are documented, gated and **left in place** for RS's call rather than
landed (see "Found, not fixed").

### Fixed (all bit-identical, corpus-verified)
- **`kernel_compute_all_forces_cpu!` allocated 160 B per particle per step** — 160 MB/step at
  N = 10⁶, ~930 GB of garbage over a production run — from `sqrt(sum(A[:, i].^2))`, which
  materialises a slice and a broadcast per particle. Replaced by the accumulation loop the boost
  kernels already used (three sites, plus the RTA kernel's two generators). Measured at N = 20 000:
  **3 200 064 → 64 bytes, and 85.3 → 41.0 ns/particle (2.08×)** on what was the dominant CPU kernel
  (the allocation-free boost kernel costs 26 ns for comparison). Bit-identical because a leading
  `+0.0` is exact and `sum` over a 2/3-element vector adds in the same order.
- `ξ .= randn(pdim, N)` allocated a fresh N×pdim matrix every step in both CPU drivers; now
  `randn!(ξ)` in place. Same RNG stream, same values, same order.
- **GPU `interpolate_2d_cuda` did a LINEAR SCAN** of each axis, 4–6× per particle per step: the
  marginal cost grew 2.5 → 3.1 → 4.1 ns/particle-step as the radial grid went 41 → 161 → 641 points
  (production grids are ≈161). Replaced by a binary search, which returns the *same index* as the
  scan and as the CPU's `searchsortedlast` for any sorted axis — not an approximation, and no
  uniform-grid assumption.

### Fixed (documentation only)
- Both force-kernel docstrings claimed the step "realises the stationary variance **MT** at any Δt".
  That is the Galilean statement; with `relativistic = true` the frozen-E OU variance is
  `κ/(2η_eff) = T·E_LRF`, and it is exact only at frozen E — which is *why* the pre-point O(ηΔt)
  bias exists. The docstring asserted the thing the known bias contradicts.
- The GPU force kernel called the **drag** spline `τn` and evaluated it with
  `_eval_tau_n_spline_cuda` — the same one-name-for-two-times that let the 2026-08-02 drag/current
  bug survive on the CPU side. Renamed to `τ_drag` / `_eval_time_spline_cuda` (the old evaluator
  name is kept as an alias) and the CPU's 🔴 note carried across.
- Recorded the `kL == kT` degeneracy: the radial branch's `(kL−kT)p̂p̂ + kTδ` noise projection,
  `p_mags`, `p_units` and `random_directions` collapse identically to `kT·ξ` and cannot change any
  answer. Kept as the place a κ_L ≠ κ_T split would go.

### Added — tests
- **`test_kernel_parity.jl`: deterministic CPU ↔ GPU parity, kernel by kernel.** The only exact
  comparison of the two backends in the tree. The GPU kernels take their randomness as
  pre-generated arrays, so both sides can be driven with the same inputs *and the same noise* and
  compared per particle at 1e-12 or tighter, instead of through a 3 % ensemble moment that could
  not have seen a 1 % drift. Covers the interpolant (pointwise, off-table, non-uniform axes,
  degenerate cells), both spline evaluators, both boosts across `pdim × relativistic × v2 × radial`,
  the force kernel across `relativistic × proper_time_kicks`, the momentum update, the Bjorken
  redshift over 200 chained steps, the position update including diffusion and reflection,
  `set_to_fluid_velocity`, and the RTA collision decision and direction. 8 290 assertions.
  Measured agreement: interpolant and spline sub-ulp in range, boosts 1–2 ulps (4.4e-16), force
  kernel ≈1.6e-14 relative to each term's own scale.
  *Exact equality is not attainable and is not asked for*: the NVPTX backend contracts multiply-adds
  into FMA and the host does not, so any expression containing one differs by an ulp by
  construction. `==` is reserved for kernels with nothing to contract (the redshift's pure multiply,
  the snapshot copies) — and holds there.
- **`test_kernel_units.jl`**: the primitives nothing had touched — `interpolate_2d_cpu` (exact on
  bilinear data, clamped outside, non-uniform and degenerate axes), `eval_tau_n_spline`, both boost
  kernels (against Lorentz *invariants* and the 4-vector boost matrix, never against their own
  decomposition), the Bjorken kernel, `_draw_juttner_pstar_lib` (χ² for d = 1, 2, 3),
  `build_juttner_invcdf` (two-sample KS against the CPU rejection sampler it replaces on the GPU —
  the agreement `collision_mode = :rta` rests on, previously unmeasured), `sample_pz_conditional_juttner`
  and `append_thermal_pz` (moments to 4th order, both envelope branches), `sample_particles_from_FONLL`
  *fidelity* (does it reproduce the density it was handed, in both spatial modes), `LV_TAUN_SCALE`,
  `check_momentum_dims`, and the homogeneous-box entry point against the **exact discrete OU
  recursion** rather than just its endpoint.
- **`test_gpu_only_paths.jl`**: `freezeout_capture` (48 uses in `Projects/`) and
  `integrator_mode = 1` (20 uses) — neither had any test. Freeze-out is checked on a background
  whose temperature depends on τ alone, so the crossing time is the same for every particle and
  computable on the host to machine precision from the same interpolated field the kernel reads.
  Result: `freezeout_interp = true` books the crossing to **2.7e-15**, while the raw mode books the
  first sampled step below `Tfo` — predicted exactly, and O(Δt) as advertised. Also: the latch fires
  once, the run does not stop at it, a never-crossing particle stays unflagged, and the booked
  position is the free-streaming trajectory at the crossing.
- **`test_time_convention.jl`**: which step time each kernel reads the background at. Every lookup
  (both boosts, forces, positions, RTA, freeze-out) uses `t₀ + step·Δt`, the END of the step;
  `kernel_bjorken_redshift_*` uses `t₀ + (step−1)Δt`, the START — deliberately, because that is what
  makes `p_z ← p_z·τ_a/(τ_a+Δt)` the exact free-streaming solution over the interval traversed (the
  end-of-step alternative telescopes to a measurably different answer, asserted). The mixture is
  consistent to O(Δt), which is the scheme's order; first-order convergence is verified on a
  deterministic probe. Neither convention was documented or asserted before.
- `runtests.jl` now drives all of them, each in its own module (the suites are standalone scripts
  too and define colliding top-level constants). `LIM_FAST=1` runs transport + units +
  time-convention in ≈30 s; the full run adds the engine gates, kernel parity and the GPU-only paths.

### Found, not fixed — each one changes what the engine computes, so it is RS's call

(Note the distinction that matters for item 1: it changes *bits*, not *numbers*. The corpus is a
bit-identity gate, and under this audit's standing rule "nothing regenerates" that was enough to
hold the fix back — but the measurement below says no physics observable moves at all.)
1. **`eval_tau_n_spline` extrapolates outside `[Tmin, Tmax]`.** The cell index is clamped, the
   interpolation weight is not, and both transport times fall like 1/T², so the linear extension of
   a convex falling function crosses zero. On a `[0.12, 0.50]` drag spline (m = 1.5, DsT = 0.11634):
   T = 0.90 returns **−0.083 fm**, T = 0.05 returns 5.168 fm against an exact 13.774 (drag 2.7× too
   strong). The negative branch is the dangerous one — it is not a NaN anyone would notice, because
   every kernel guards `τ > 0 ? 1/τ : 0` and therefore *silently* sets `η_D = κ = 0`.
   *Why not fixed:* `t = clamp(u - (i-1), 0.0, 1.0)` is the whole fix, but it is not bit-neutral.
   The background interpolant returns T a fraction of an ulp outside `[min T, max T]` wherever the
   field is locally flat (a constant-T box, a `max(0.12, …)` floor), and the driver spans the spline
   over exactly that range — so the old code extrapolates by a tiny weight and the clamp changes the
   last bits. Measured on `default_box` (T = 0.30, N = 20 000, 300 steps): per-particle
   max |Δp| = 3.0e-14, max |Δx| = 5.3e-15, and **⟨p²⟩ and ⟨x²⟩ identical to all 17 digits**
   (relative difference exactly 0.0) — it moves no physics number, but it moves 6 of the 10 corpus
   hashes. `posdiff_reflect` (T = 0.25) survives because 0.25 is a power of two and rounds exactly.
   *Also costs parity:* out of range the two backends disagree by up to 4.4e-11 (vs 2.2e-16 in
   range), because `(1−t)y₀ + t·y₁` with |t| ~ 10³–10⁴ amplifies the one-ulp FMA difference by |t|.
   Gated `@test_broken` in `test_kernel_units.jl` U2, and split in `test_kernel_parity.jl` P2.
2. **`integrator_mode = 1` roughly doubles the Δt bias it is documented to remove** — see README
   "Known biases" for the table and the mechanism. Not a production incident: no recipe selects it.
   Gated `@test_broken` in `test_gpu_only_paths.jl` F5.
3. **Two CPU/GPU divergences**, recorded in `test_kernel_parity.jl` `@testset "D1"`:
   *(a)* RTA at T → 0 — the CPU's rejection sampler returns 0 and the momentum is **zeroed**, while
   the GPU's inverse-CDF table clamps to its coldest column and returns a finite thermal |p*|
   (2.75 GeV at the table edge). Qualitatively different limits at the fireball rim.
   *(b)* at Δt ≤ 0 the CPU writes zero noise and the GPU writes `kT·ξ` (unreachable from the
   drivers, but the two are not the same function).

### Verification of this release
Run on 2026-08-31 (Julia 1.12.6, RTX 5070):
- `regression_corpus.jl` — **10/10 CPU hashes bit-identical**. This is the acceptance criterion for
  everything above; the baseline was NOT regenerated.
- `runtests.jl` (full) — 30 testsets, exit 0. The only `Broken` entries are the two deliberate
  expected-fail gates (U2, F5).
- `bench_physics_gates.jl` — **18/18**, and every CPU number reproduces the committed 2026-08-22
  result exactly (D_s measured/nominal 0.9896 / 0.9956 / 0.9975; tail 0.00413 → 0.00423; ℓ=1 rates
  0.6364 and 0.6952; `DsT_quad` η_D 2.6106 / 2.6094). Independent confirmation that the hot-loop
  rewrite changed nothing numerically. GPU columns move within their 3 % gate (unseeded CURAND).
- `bench_gpu_parity.jl` — **31/31** including the new freeze-out case.
- `programme.jl check` — PASS, with the new `engine` entry.
- Public API unchanged: all 12 exports resolve and the full production kwarg surface is accepted.
- **`bench_throughput.jl` was NOT re-run** — two other sessions were running Julia jobs on this
  machine (load average 3.5), and a wall-clock table taken under that load would be misleading, and
  worse, would be written to a dated results file others would trust. The efficiency claims above
  rest instead on the isolated kernel measurement, whose *allocation* figures are load-independent
  and reproduced exactly on a second run under load (64 B, 40.9 ns/particle). Re-run the full table
  on an idle machine to refresh `bench/results/`.

### Other findings, recorded
- **The regression corpus's GPU `px` gate is flaky by construction.** It compares one unseeded GPU
  draw against a *single seeded CPU draw* with `atol = 0.02`, but the GPU run-to-run scatter of
  ⟨p_x⟩ is sd 0.0072 (8 repeats) and the CPU value is itself one draw of the same width — so the
  difference has sd ≈ 0.010 and the gate is a ~2σ test that fails roughly one run in twenty. It
  failed on `default_box` on the first pre-change run of this audit while `p2`, `x2` and `r` agreed
  to −0.54 %, −0.03 % and −0.01 % and snapshot 1 matched to 4.4e-16. Left as is (widening it is a
  judgement call about what the corpus is for) — but `test_kernel_parity.jl` is now the gate that
  actually holds the backends together, and it cannot be flaky.
- `kernel_set_to_fluid_velocity_*` ignores `V2Evolution` on **both** backends while the boosts apply
  it, so `momentum_langevin = false` with an elliptic modulation uses an inconsistent flow field.
  Parity is preserved, so no gate fails; two call sites in `Projects/` pass `V2Evolutionn`.
- In the `!momentum_langevin` branch the driver boosts to the rest frame and then overwrites the
  momentum unconditionally — the boost is dead work (harmless; the kernel writes a lab-frame value).
- **The GPU driver's fixed per-call overhead is ≈0.74–1.0 s** (measured; also visible in
  `bench/results/throughput_*.md` as the "fixed per call" column). It ends every call with
  `GC.gc(true)` twice plus `CUDA.reclaim()`, and then a `finally` block that runs `GC.gc()` and
  `CUDA.reclaim()` twice more — three full collections and three reclaims per invocation. That is
  dead weight for a single long run and real cost for a campaign that calls the driver many times.
  NOT changed: the pattern exists to guarantee the large device history is released between calls,
  and trading it for speed is a memory-safety decision on a machine with 12 GB of VRAM, not a
  cleanup. Worth revisiting deliberately.
- CPU: the per-step `any(!isfinite, momenta) || any(!isfinite, positions)` scan costs 0.019 ms at
  N = 20 000 (≈1 % of the step, and it scales with N). Kept — it is the only thing that turns a
  NaN into an error message instead of a silently ruined run — but it is not free, and the GPU path
  does not do it at all.

### Removed
- `sample_particles_from_density`, `sample_initial_particles_from_pdf!` and
  `sample_initial_particles_at_origin!` — **zero consumers repo-wide** (grep over `*.jl`, `*.md`,
  `*.tex`). Retired to `Julia/Projects/trash/langevinmedium_retired_samplers_2026-08-31.jl` with the
  reasons, per the house rule that wrong turns stay in place and dated. Two of them do not survive
  being read: `sample_initial_particles_from_pdf!` takes `abs.(σ .* randn(dim))`, making every
  momentum component positive (a half-normal with ⟨p⟩ ≠ 0 in every direction, not a thermal
  distribution), and writes the *radius* into every spatial component (`positions[:, i] .= r`, so a
  particle at radius r is placed at |x| = r√2); `sample_particles_from_density` takes two arguments
  it never reads and rebuilds an 800-point grid and a fresh interpolation object per particle.
  `sample_initial_particles_at_origin_no_position!` is a different function, is live, and stays.
- With them went `src/`'s last uses of `QuadGK` and `Distributions.Uniform`. QuadGK stays in
  `[deps]` because the test and bench suites use it heavily; it is now a test-only dependency of
  the package and could move to `[extras]` if anyone wants the runtime dependency list minimal.

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
