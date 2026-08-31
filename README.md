# LangevInMedium.jl

Relativistic Langevin dynamics of heavy quarks in an evolving medium — the particle engine
behind the LangevinPaper1 / O+O / AttractorMomentum / AttractorHydro / KineticAttractor
studies. An ensemble is propagated on a tabulated background `T(r, τ)`, `v_r(r, τ)` (a hydro
output); every step boosts into the local fluid rest frame, applies the **exact
Ornstein–Uhlenbeck propagator** for the drag with the matching Einstein noise, boosts back and
streams the positions. CPU and CUDA backends run the same algorithm.

```julia
using CUDA                      # optional — attaches the GPU backend
using LangevInMedium

t, mom, pos = simulate_ensemble_bulk(GPUBackend(),      # or CPUBackend()
    r_grid, p_grid, f_init,                             # initial density f[p_index, r_index] on (r_grid, p_grid)
    T_field, v_field, (xgrid, tgrid);                   # background tables T[i,j] = T(xgrid[i], tgrid[j]), v likewise
    N_particles = 10^6, Δt = 1e-3, initial_time = 0.4, final_time = 12.0, save_interval = 0.5,
    m = 1.5, DsT = 0.11634, dimensions = 2)
# mom[k] :: (momentum rows, N) at t[k];  pos[k] :: (2, N)
```

Run everything with the monorepo environment: `julia --project=Julia …` (the package's own
environment is not instantiated).

## The one entry point

`simulate_ensemble_bulk(backend, …)` dispatches on the backend singleton:

| method | what it does |
|---|---|
| `(CPUBackend(), r_grid, p_grid, f, T_field, v_field, (xgrid, tgrid); kw...)` | the workhorse, ≈35 call sites in `Projects/` |
| `(GPUBackend(), …same…; kw...)` | CUDA twin; exists only after `using CUDA` |
| `(CPUBackend(), T::Float64; kw...)` | homogeneous box, momenta only (toy: fixed `κ = 2.5T³`, ignores `DsT`) |

Returns `(time_points, momenta_snapshots, position_snapshots)`. `?simulate_ensemble_bulk` has
the full keyword table; the ones that decide the physics:

| keyword | default | meaning |
|---|---|---|
| `m`, `DsT` | 1.0, 0.2 | quark mass [GeV]; `D_s·T` label. The drag is the Einstein relation `1/η_D = tau_drag = m·DsT/T²` |
| `DsT_linear, DsT_slope, DsT_offset, Tfo` | off | `DsT(T) = slope·max(T, Tfo) + offset` |
| `DsT_quad, DsT_Tref` | off | `DsT(T) = DsT·(T/Tref)²` ⇒ a T-independent drag time (the uniform-drag member; the Galilean solvable class closes only here) |
| `dimensions` | 3 (**pass 2**) | 2 = transverse plane (x, y); 1 = radial mode (r, p_r) |
| `momentum_dimensions` | 0 (= `dimensions`) | 3 with `dimensions = 2`: a longitudinal `p_z` row (thermal conditional at t0). 2-D momenta relax to the 2-D Jüttner, whose current rate λ₁η_D differs from the 3-D `K₂/K₃·η_D` by 5–12 % over z = 3.5–10 |
| `bjorken_redshift` | false | `dp_z/dτ = −p_z/τ` between kicks (needs `momentum_dimensions = 3`, `initial_time > 0`) |
| `relativistic` | true | true: Jüttner kinematics — drag `η_D·m/E`, streaming `p/E`, Lorentz boosts. false: the exactly solvable **Galilean** process — drag `η_D`, streaming `p/m`, boosts `p∥ ∓ m·v` |
| `collision_mode` | `:langevin` | `:rta`: BGK re-draw from the local Jüttner with probability `Δt/τ_n`, τ_n the **current** time `tau_n_main3` |
| `x_init, p_init` | sampler | `(2, N)` lab positions and **rest-frame** momenta (the t0 lab boost is applied inside) |
| `cartesian_spatial_sampling`, `antithetic_momenta` | auto, false | sampler mode (disc rejection vs polar inverse-CDF); (p, −p) pairs |
| `position_diffusion`, `reflecting_boundary` | false | extra overdamped position kicks (double-counts vs hydro); reflect at `r = xgrid[end]` |
| `momentum_langevin` | true | false (or `DsT = 0`): particles glued to the flow |
| `V2Evolutionn, psi2` | — | elliptic modulation `v → v(1 + 2v₂cos 2(φ−Ψ₂))` |
| GPU only: `freezeout_capture, freezeout_interp` | false | latch each particle's `T = Tfo` crossing; returns a NamedTuple `(pos, mom, tau, flag)` instead of histories. The run does **not** stop at freeze-out |
| GPU only: `integrator_mode` | 0 | 1 = drift-midpoint drag. 🔴 **Measured to roughly DOUBLE the Δt bias it was meant to remove** — see "Known biases" below. The CPU refuses 1 |
| GPU only: `verbose` | false | print device + memory status at entry |

### Drag time vs current time — read this once

`tau_drag(T, m, DsT) = m·DsT/T²` is what the kernel uses (`η_D = 1/τ_drag`, `κ = 2mT/τ_drag`).
`tau_n_main3(T, m, DsT) = D_s·z·K₃/K₂` is the **diffusion-current** relaxation time — the
Israel–Stewart τ_n that Fluidum's `τ_diffusion_hadron` evaluates. For a Fokker–Planck process
with drag η_D the ℓ=1 mode decays at `η_D·K₂/K₃` (an exact 3-D identity), so a Langevin built
from `tau_drag` reproduces **both** hydro coefficients, D_s and τ_n. Building the drag from
`tau_n_main3` applies K₃/K₂ once too often and inflates the realised D_s by 1.26–1.74× — that
was the state of every product before 2026-08-02. The BGK (`:rta`) path needs τ_n, not the drag
(`build_taun_current_spline`). `test/runtests.jl` guards the ratio pointwise.

`LV_TAUN_SCALE` (env var, default 1) rescales **both** splines — diagnostic only; it moves D_s
too. The bench suite refuses to run under any other value.

### CPU vs GPU

Same algorithm, same kwargs, same return shape. The CPU path is bit-reproducible under
`Random.seed!`; the GPU draws from CURAND and is reproducible in ensemble moments only. The
host does the sampling and the p_z completion on both paths, so the t0 snapshot is identical
to float rounding (the parity bench checks it). The GPU interpolant, boosts and T-guards mirror
the CPU's clamps since 0.2.0 — a particle leaving the table or a `|v| > 1` cell used to NaN the
whole GPU ensemble.

## Tests and benchmarks

```
LIM_FAST=1 julia --project=Julia Julia/LangevInMedium.jl/test/runtests.jl    # transport + unit + time-convention (≈30 s)
           julia --project=Julia Julia/LangevInMedium.jl/test/runtests.jl    # + relativistic switch, momentum_dims3, CPU/GPU kernel parity, GPU-only paths (≈10 min, GPU halves if CUDA works)
           julia --project=Julia Julia/LangevInMedium.jl/test/regression_corpus.jl   # bit-identity vs the committed baseline (CPU hashes, GPU moments)
```

**CPU ↔ GPU is pinned deterministically, not statistically.** `test_kernel_parity.jl` drives each
kernel pair with the same inputs *and the same injected noise arrays* (the GPU kernels take their
randomness pre-generated), so the two backends are compared per particle at 1e-12 or tighter
instead of through a 3 % ensemble moment. Exact equality is not attainable and is not asked for:
the device contracts multiply-adds into FMA and the host does not, so any expression with a
multiply-add differs by an ulp by construction. Measured agreement: the interpolant and the spline
evaluator sub-ulp in range, the boosts 1–2 ulps, the force kernel ≈1.6e-14 relative to each term's
own scale. Two divergences are deliberate and recorded there rather than reconciled (see
`@testset "D1"`).

`test/regression_corpus_baseline.txt` holds SHA-256 hashes of ten seeded runs spanning the
kwarg surface (Galilean, p_z + redshift, DsT_quad/linear, RTA, position diffusion, both sampler
modes, radial mode). A change that keeps the default dynamics must reproduce every hash;
regenerate with `LIM_CORPUS_WRITE=1` **only** for a deliberate change and say so in the CHANGELOG.
Its GPU half is a *statistical* check and its ⟨p_x⟩ gate is a ~2σ test against a single seeded CPU
draw, so it fails roughly one run in twenty on that field alone (CHANGELOG 0.2.1) — the CPU hashes
are the deterministic part, and `LIM_CORPUS_NOGPU=1` runs only those.

The engine is wired into the repo gate as `programme.jl check` → `engine`
(`Julia/Projects/test_langevinmedium_engine.jl`), which runs the deterministic half only:
`LIM_FAST=1 runtests.jl` plus the CPU bit-identity corpus.

`bench/` (results in `bench/results/`):

| script | what it pins |
|---|---|
| `bench_physics_gates.jl` | quantities in physical units the engine cannot fake: MSD slope = 2·d·D_s at three z (CPU and GPU); the Jüttner tail p > 3 GeV at the Poisson floor; the ℓ=1 rate = λ₁η_D for 3 and 2 momentum rows; Δt bias of the propagator (none for Galilean, ≤ 1.5 % at ηΔt ≤ 0.1 relativistic); the Galilean MSD(t) curve at all t; `DsT_quad` ⇒ T-independent drag |
| `bench_gpu_parity.jl` | CPU ↔ GPU **moments** on four nominal backgrounds and on the adversarial inputs (outside the table, `\|v\| > 1` cell) that separated them before 0.2.0, plus a `freezeout_capture` self-consistency case. The *exact* backend comparison is `test/test_kernel_parity.jl`; this one asks the different question "do two full runs with different RNG streams land in the same place" |
| `bench_throughput.jl` | marginal ns per particle-step and fixed per-call overhead for CPU/GPU × N × momentum rows × relativistic × collision mode, plus the host-side phases (sampler, `randn!`, copies) |

All three end in a top-level `[PASS]`/`[FAIL]` line and a non-zero exit on failure.

## Known biases and limits

- **Pre-point relativistic drag**: `η_D·m/E` is evaluated at the start of the step ⇒ an O(ηΔt)
  bias on ⟨p²⟩, ≈ −1 % at ηΔt = 0.1 and below resolution at production ηΔt ≈ 3·10⁻³. The
  Galilean propagator is exact at any Δt.
- 🔴 **`integrator_mode = 1` makes that bias WORSE, not better** (measured 2026-08-31, N = 10⁶,
  SEM 0.14 %, uniform bath against the 2-D Jüttner ⟨p²⟩):

  | ηΔt | mode 0 | mode 1 | ratio |
  |---|---|---|---|
  | 0.05 | −1.03 % | −1.74 % | 1.70 |
  | 0.10 | −1.36 % | −3.49 % | 2.57 |
  | 0.20 | −3.23 % | −6.53 % | 2.02 |
  | 0.30 | −4.26 % | −9.19 % | 2.16 |

  Same sign, roughly double, and still linear in Δt — so it is O(ε), not the advertised O(ε²).
  The predictor is a *noise-free* drag half-step `p_mid = e^{−η_eff Δt/2}·p`, which can only shrink
  |p|; so `E_mid < E` always and the midpoint drag is always *larger* than the pre-point one. A
  genuine midpoint would have the noise raising ⟨p²⟩ as much as the drag lowers it — that is what
  stationarity means — so the correct midpoint energy is ≈ the pre-point one. Dropping the noise
  (deliberately, to stop `η_eff` correlating with ξ) trades a correlation error for a one-sided
  drift of the same order and sign as the error it targets. **No product is affected**: every
  driver maps its `integrator` option with `== "mid" ? 1 : 0` and no recipe passes `"mid"`.
  Gated in `test_gpu_only_paths.jl` `@testset "F5"`; not fixed, because a fix changes what mode 1
  computes.
- 🔴 **`eval_tau_n_spline` extrapolates outside its table** and can return a NEGATIVE time, which
  every kernel reads as `η_D = κ = 0` — silent free streaming with neither drag nor noise. Not
  reachable through the drivers (they span the spline over the whole `T` table and the background
  interpolant clamps into it); reachable by any external caller, and this function is exported.
  Full account, measurements and the one-line fix in its docstring; gated with `@test_broken` in
  `test_kernel_units.jl` `@testset "U2"`.
- **The boosts are not exactly Lorentz**: `γ = 1/√(1 − v² + 1e-10)`. The regularisation biases γ
  low by ≈ ½·1e-10/(1−v²) (1.4e-10 at v = 0.8), and because `γ²(1−v²) ≠ 1` the lab→LRF→lab round
  trip is not an involution — it contracts momenta by ≈ 1e-10/(1−v²) per step, ≈ 9e-7 over a
  5 800-step run at v = 0.6. It also overrides the `|v| ≤ √(1−1e-12)` clamp, capping γ at ≈1e5
  rather than 1e6. Pinned in `test_kernel_units.jl` `@testset "U3b"`/`"U3c"`.
- **2-D momenta** relax to the 2-D Jüttner; compare against 3-D hydro coefficients with
  `momentum_dimensions = 3` or budget the 5–12 % λ₁ offset.
- **`save_interval` should divide the evolution**; otherwise the trailing steps are not in the
  history (the returned `time_points` are the true snapshot times; a warning says how much was dropped).
- A background table with `T ≤ 0` anywhere is refused by the spline builder on both backends.
- The FONLL sampler's acceptance in `cartesian_spatial_sampling = true` mode is the fireball's
  area fraction of the disc — a few % for Pb+Pb in a 20 fm disc; it is host-side and serial.
- GPU: the methods exist only after `using CUDA` (Requires.jl, not precompiled); `:rta` and
  `:langevin` only; the momentum history for `N·(saves+1)` lives on the device until the end.

## Layout

```
src/LangevInMedium.jl     module docstring, includes, exports
src/constants.jl          ħc (PDG 0.197327 GeV·fm) and fmGeV
src/backends.jl           CPUBackend, GPUBackend
src/utils.jl              sample_particles_from_FONLL, the p_z completion (append_thermal_pz), check_momentum_dims
src/transport.jl          tau_drag, tau_n_main3, effective_DsT, the two spline builders, build_juttner_invcdf, LV_TAUN_SCALE
src/kernels_cpu.jl        per-step CPU kernels (boosts, forces, momentum/position updates, RTA, saves)
src/simulate_cpu.jl       CPU driver (+ _snapshot_times)
src/simulate.jl           public dispatch + the Requires hook
src/kernels_gpu.jl        CUDA kernels — line-for-line twins of the CPU ones (keep them in step; bench_gpu_parity.jl is the check)
src/simulate_gpu.jl       GPU driver (freeze-out capture, RTA inverse-CDF table)
src/simulate_gpu_wrapper.jl   the GPU method, included by the hook
src/data/Fluidum_MIS_HQ.jld2  a Fluidum MIS background (23 MB) — NOT read by the package; FokkerPlank1D/2D, FiVoHydro/main2.jl and CompareBoltzmannHQ load it from here
test/                     runtests.jl (drives everything below), regression_corpus.jl (+ baseline)
  test_kernel_units.jl        every primitive against an independent construction (interpolant, spline
                              evaluator, both boosts, the two Jüttner samplers, FONLL fidelity, the box path)
  test_kernel_parity.jl       DETERMINISTIC CPU↔GPU, kernel by kernel, same injected noise, ≤1e-12
  test_gpu_only_paths.jl      freezeout_capture and integrator_mode = 1
  test_time_convention.jl     which step time each kernel reads the background at
  test_relativistic_switch.jl, test_momentum_dims3.jl, test_proper_time_kicks.jl,
  test_rta_proper_time.jl, test_bjorken_redshift_exact.jl
bench/                    bench_common.jl, bench_physics_gates.jl, bench_gpu_parity.jl, bench_throughput.jl, results/
```
