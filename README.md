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
| GPU only: `integrator_mode` | 0 | 1 = drift-midpoint drag (O(ε²)); the CPU refuses 1 |
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
LIM_FAST=1 julia --project=Julia Julia/LangevInMedium.jl/test/runtests.jl    # transport units + entry-point smoke (seconds)
           julia --project=Julia Julia/LangevInMedium.jl/test/runtests.jl    # + relativistic switch + momentum_dims3 gates (≈10 min CPU, GPU gate if CUDA works)
           julia --project=Julia Julia/LangevInMedium.jl/test/regression_corpus.jl   # bit-identity vs the committed baseline (CPU hashes, GPU moments)
```

`test/regression_corpus_baseline.txt` holds SHA-256 hashes of ten seeded runs spanning the
kwarg surface (Galilean, p_z + redshift, DsT_quad/linear, RTA, position diffusion, both sampler
modes, radial mode). A change that keeps the default dynamics must reproduce every hash;
regenerate with `LIM_CORPUS_WRITE=1` **only** for a deliberate change and say so in the CHANGELOG.

`bench/` (results in `bench/results/`):

| script | what it pins |
|---|---|
| `bench_physics_gates.jl` | quantities in physical units the engine cannot fake: MSD slope = 2·d·D_s at three z (CPU and GPU); the Jüttner tail p > 3 GeV at the Poisson floor; the ℓ=1 rate = λ₁η_D for 3 and 2 momentum rows; Δt bias of the propagator (none for Galilean, ≤ 1.5 % at ηΔt ≤ 0.1 relativistic); the Galilean MSD(t) curve at all t; `DsT_quad` ⇒ T-independent drag |
| `bench_gpu_parity.jl` | CPU ↔ GPU moments on four nominal backgrounds and on the adversarial inputs (outside the table, `|v| > 1` cell) that separated them before 0.2.0 |
| `bench_throughput.jl` | marginal ns per particle-step and fixed per-call overhead for CPU/GPU × N × momentum rows × relativistic × collision mode, plus the host-side phases (sampler, `randn!`, copies) |

All three end in a top-level `[PASS]`/`[FAIL]` line and a non-zero exit on failure.

## Known biases and limits

- **Pre-point relativistic drag**: `η_D·m/E` is evaluated at the start of the step ⇒ an O(ηΔt)
  bias on ⟨p²⟩, ≈ −1 % at ηΔt = 0.1 and below resolution at production ηΔt ≈ 3·10⁻³. The
  Galilean propagator is exact at any Δt. (GPU `integrator_mode = 1` removes the O(ε) term.)
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
src/utils.jl              sample_particles_from_FONLL, sample_particles_from_density, the p_z completion (append_thermal_pz), check_momentum_dims
src/transport.jl          tau_drag, tau_n_main3, effective_DsT, the two spline builders, build_juttner_invcdf, LV_TAUN_SCALE
src/kernels_cpu.jl        per-step CPU kernels (boosts, forces, momentum/position updates, RTA, saves)
src/simulate_cpu.jl       CPU driver (+ _snapshot_times)
src/simulate.jl           public dispatch + the Requires hook
src/kernels_gpu.jl        CUDA kernels — line-for-line twins of the CPU ones (keep them in step; bench_gpu_parity.jl is the check)
src/simulate_gpu.jl       GPU driver (freeze-out capture, RTA inverse-CDF table)
src/simulate_gpu_wrapper.jl   the GPU method, included by the hook
src/data/Fluidum_MIS_HQ.jld2  a Fluidum MIS background (23 MB) — NOT read by the package; FokkerPlank1D/2D, FiVoHydro/main2.jl and CompareBoltzmannHQ load it from here
test/                     runtests.jl, test_relativistic_switch.jl, test_momentum_dims3.jl, regression_corpus.jl (+ baseline)
bench/                    bench_common.jl, bench_physics_gates.jl, bench_gpu_parity.jl, bench_throughput.jl, results/
```
