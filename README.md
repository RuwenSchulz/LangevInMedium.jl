<h1 align="center">LangevInMedium.jl</h1>

<p align="center">
  <b>Relativistic Langevin dynamics of heavy quarks in an evolving medium.</b><br>
  <sub>Exact Ornstein–Uhlenbeck propagator · CPU and CUDA</sub>
</p>

<p align="center">
  <img alt="version"  src="https://img.shields.io/badge/version-0.2.4-blue">
  <img alt="license"  src="https://img.shields.io/badge/license-Apache--2.0-blue">
  <img alt="julia"    src="https://img.shields.io/badge/Julia-1.12-9558B2?logo=julia&logoColor=white">
  <img alt="backends" src="https://img.shields.io/badge/backends-CPU%20%2B%20CUDA-76B900">
  <img alt="gates"    src="https://img.shields.io/badge/closed--form%20gates-9%20%2B%2018-brightgreen">
  <a href="https://doi.org/10.5281/zenodo.22791006"><img alt="DOI" src="https://zenodo.org/badge/DOI/10.5281/zenodo.22791006.svg"></a>
</p>

<p align="center">
  <img src="examples/figures/01_uniform_bath.png" alt="Thermalisation in a uniform bath" width="92%">
</p>
<p align="center">
  <sub>
    From <a href="examples/01_uniform_bath.jl"><code>examples/01_uniform_bath.jl</code></a>. The dashed
    lines are closed forms: the Jüttner ⟨p²⟩, the ℓ=1 decay rate <code>(K₂/K₃)·η_D</code>, the Jüttner
    momentum distribution, and <code>D_s = D_sT/T·ħc</code> from the mean-square displacement
    (measured −0.45 %).
  </sub>
</p>

---

## What it computes

An ensemble of heavy quarks on a tabulated hydrodynamic background `T(r, τ)`, `v_r(r, τ)`. Each
step boosts every particle into the local fluid rest frame, applies the exact Ornstein–Uhlenbeck
propagator for the drag with the matching Einstein noise, boosts back, and streams the positions:

```
p*(t+Δt) = a·p*(t) + √(κ (1−a²)/(2η_eff)) · ξ ,     a = e^{−η_eff Δt} ,   ξ ~ N(0, 1)

   η_D = 1/τ_drag ,   τ_drag = m·D_sT/T²  (Einstein)      κ = 2 m T η_D  (fluctuation–dissipation)
   η_eff = η_D·m/E*   (relativistic ⇒ Jüttner equilibrium)     dx/dt = p/E
```

There is no stability limit on `Δt`. In the Galilean case (`relativistic = false`) the step is exact
at any `Δt`. With `relativistic = true` the drag depends on the particle's energy, which is read at
the start of the step. That gives an O(ηΔt) bias, measured in
[`bench/bench_accuracy.jl`](bench/bench_accuracy.jl) as `≈ 12.0 %·(η_DΔt)^0.94`.

One coefficient goes in (`D_sT`) and both hydrodynamic coefficients come out: the Navier–Stokes
`D_s` and the Israel–Stewart current time `τ_n = τ_drag·K₃/K₂`.

The same algorithm runs on the CPU (bit-reproducible under `Random.seed!`) and on CUDA. The two
backends agree per particle to 1e-12.

## Install

```sh
git clone https://github.com/RuwenSchulz/LangevInMedium.jl.git
cd LangevInMedium.jl
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

```julia
using CUDA               # optional, attaches the GPU backend via Requires.jl
using LangevInMedium
```

The examples also need `Plots` (or run them with `LIM_NOPLOT=1`).

## Quick start

```julia
using LangevInMedium, Random

# 1. a background: T_field[i, j] = T(xgrid[i], tgrid[j]) in GeV, v_field the radial flow in units of c
xgrid = collect(range(0.0, 26.0; length = 209))
tgrid = collect(0.4:0.05:15.0)
T_field = [max(0.05, 0.50 * (0.4/τ)^(1/3) * exp(-r^2/32)) for r in xgrid, τ in tgrid]
v_field = [0.65 * tanh(r/4) * min(1.0, τ/2)              for r in xgrid, τ in tgrid]

# 2. an initial phase-space density f[p_index, r_index] on UNIFORM (r_grid, p_grid)
r_grid  = collect(range(0.0, 20.0; length = 150))
p_grid  = collect(range(0.0, 10.0; length = 300))
density = [(1 + (p/2.1)^2)^(-3.1) * exp(-r^2/18) for p in p_grid, r in r_grid]

# 3. run
Random.seed!(1)
t, mom, pos = simulate_ensemble_bulk(CPUBackend(), r_grid, p_grid, density,
    T_field, v_field, (xgrid, tgrid);
    N_particles = 100_000, Δt = 5e-3,
    initial_time = 0.4, final_time = 15.0, save_interval = 0.5,
    m = 1.5, DsT = 0.11634,
    dimensions = 2,                 # positions: the transverse plane
    momentum_dimensions = 3,        # momenta: THREE rows (see "conventions" below)
    bjorken_redshift  = true,       # dp_z/dτ = −p_z/τ between kicks
    proper_time_kicks = true)       # kick per the particle's proper time, not the lab step

# mom[k] :: (momentum rows, N) at t[k];   pos[k] :: (dimensions, N)
```

Swap `CPUBackend()` for `GPUBackend()` and nothing else changes.

## Examples

[`examples/`](examples) holds five runnable setups, smallest first. Each prints the measured numbers
next to the closed form they should match, and writes its figure.

```sh
julia --project=. examples/01_uniform_bath.jl
LIM_NOPLOT=1 julia --project=. examples/03_four_limits.jl     # numbers only, no Plots
```

| | setup | what it shows |
|---|---|---|
| [**01**](examples/01_uniform_bath.jl) `uniform_bath` | a box at fixed `T`, no flow, a δ-function initial momentum | what `D_sT` sets: ⟨p²⟩ → the Jüttner value, the current decays at `(K₂/K₃)η_D`, and the MSD slope is `2·d·D_s` |
| [**02**](examples/02_bjorken_fireball.jl) `bjorken_fireball` | a cooling, expanding fireball; the engine samples a FONLL-shaped density; freeze-out off the snapshots | a realistic run, and the radial flow lifting the `p_T` spectrum (⟨p_T⟩ 1.586 → 1.318 GeV) |
| [**03**](examples/03_four_limits.jl) `four_limits` | `:langevin`, `:rta`, `DsT = 0`, `DsT → 0⁺`, `:none` on one background | the weak-coupling settings are three different limits |
| [**04**](examples/04_pz_and_rapidity.jl) `pz_and_rapidity` | `momentum_dimensions = 3`, both `pz_init` modes, `track_eta_s` | what row 3 means (`p_z* = m_T sinh(y − η_s)`, not a lab `p_z`), and the kernel that makes `dN/dy = ρ(η_s) ⊛ P(K)` exact |
| [**05**](examples/05_gpu_freezeout.jl) `gpu_freezeout` | the GPU path with `freezeout_capture` | memory ∝ `N` instead of `N·(saves+1)`, the crossing resolved to `Δt`; the run does not stop at freeze-out |

### The weak-coupling limits

<p align="center">
  <img src="examples/figures/03_four_limits.png" alt="The collision settings side by side" width="94%">
</p>

Three settings sit near zero coupling, and they are different limits.

- `D_sT = 0` is the **comoving** limit: every quark gets `p = m·γ(r)v(r)` at its own radius, with no
  thermal width. `⟨p_x⟩ = 0.86603 GeV` and `sd(p_x) = 0` exactly (black spike).
- `D_sT → 0⁺` is thermal comoving: the quark keeps the full Jüttner width and lands on
  `γ·v·⟨E*⟩ = 1.068 GeV`, 23 % higher (orange).
- **Free streaming** is `collision_mode = :none` (red spike): no drag, no noise, no frame change. The
  momenta stay at the boosted initial momentum, 2.196 GeV.

`m ≤ 0` and `D_sT < 0` are refused.

## Validation

The engine is checked against closed forms. Two suites, 27 gates, all passing:

```sh
julia --project=. bench/bench_semianalytic.jl   # 9 gates, with plots
julia --project=. bench/bench_physics_gates.jl  # 18 gates, physical units
```

<p align="center">
  <img src="bench/results/figures/semianalytic_S2_bgk_moment_law.png" alt="Exact BGK moment law" width="94%">
</p>

In a uniform bath a BGK particle has either not collided yet (probability `e^{−t/τ_n}`, it still
carries its initial momentum) or has, and is then equilibrium-distributed. So for any observable

```
⟨g⟩(t) = e^{−t/τ_n}·⟨g⟩₀ + (1 − e^{−t/τ_n})·⟨g⟩_eq        exact, at all t
```

Four step sizes spanning 100× fall on that curve with no Δt trend.

| gate | target | measured |
|---|---|---|
| Uhlenbeck–Ornstein covariance | `σ_pp = mT`, `σ_xx = 2(T/m)/η²(ηt−1+e^{−ηt})`, `σ_xp = (T/η)(1−e^{−ηt})` | 0.40 % / 0.21 % / **0.54 %** worst |
| exact BGK moment law | the whole `⟨p²⟩(t)` curve, Δt spanning 100× | 0.5 % worst, no Δt trend |
| Bjorken redshift | `⟨p_z*²⟩ ∝ 1/τ²` telescopes | **2.9e-13** |
| free streaming + redshift | per-particle closed form for `x_⊥(τ)` | converges at order **1.00** |
| equilibrium shape | two-sample KS vs the exact Jüttner, 2 and 3 rows | 0.0019 / 0.0039 (95 % critical 0.0043) |
| comoving blast wave | `p = m·γ(r)v(r)`, per particle, and the residual's shape in `r` | the predicted one-step lag `−m·d(γv)/dr·v·Δt`, order **1.00** |
| MSD slope | `2·d·D_s` at three `z` | 0.9975 / 0.9969 / 0.9975 (CPU) |
| ℓ=1 rate | `η_D·K₂/K₃` (3 rows), `λ₁(2D)·η_D` (2 rows) | 0.6364 vs 0.6403 · 0.6952 vs 0.6977 |

`σ_xp` combines the momentum and the position update, so it is the one that shows an operator-split
error.

<p align="center">
  <img src="bench/results/figures/semianalytic_S1_ou_covariance.png" alt="OU phase-space covariance" width="70%">
</p>

The comoving limit is checked per particle: every quark must carry `p = m·γ(r)v(r)` at its own
radius. The residual is the one-step lag `−m·d(γv)/dr·v·Δt`, and it vanishes beyond `r = 8 fm`, where
the flow profile saturates and `dv/dr = 0`.

<p align="center">
  <img src="bench/results/figures/semianalytic_S5_blastwave_vs_free.png" alt="Comoving blast wave, per particle" width="94%">
</p>

### Step size

[`bench/bench_accuracy.jl`](bench/bench_accuracy.jl) turns a required accuracy into a required step
size. The pre-point relativistic drag gives an O(ηΔt) bias on ⟨p²⟩:

```
|⟨p²⟩ bias| ≈ 12.0 % · (η_D Δt)^0.94      ⇒  0.046 % at the production η_D Δt ≈ 2.6·10⁻³
D_s is unbiased to ≲1 % for η_D Δt ≤ 0.08
```

## Conventions

**1. One transport coefficient.**
`tau_drag(T, m, DsT) = m·DsT/T²` is what the kernel uses (`η_D = 1/τ_drag`, `κ = 2mT/τ_drag`).
`tau_n_main3(T, m, DsT) = D_s·z·K₃/K₂` is the diffusion-current relaxation time, the Israel–Stewart
τ_n (Fluidum's `τ_diffusion_hadron`). For a Fokker–Planck process with drag η_D the ℓ=1 mode decays
at `η_D·K₂/K₃`, so a Langevin built from `tau_drag` reproduces both D_s and τ_n. Building the drag
from `tau_n_main3` instead inflates the realised D_s by 1.26–1.74×. The BGK (`:rta`) path uses τ_n
(`build_taun_current_spline`).

`LV_TAUN_SCALE` (env var, default 1) rescales both splines. It is a diagnostic and moves D_s too;
the bench suite refuses to run with any other value.

**2. `dimensions` sets the positions, `momentum_dimensions` the momenta.**
2-D momenta relax to the 2-D Jüttner, whose ℓ=1 rate differs from the 3-D `K₂/K₃·η_D` by 5–12 % over
`z = 3.5–10`. The hydrodynamic coefficients are the 3-D ones, so use `dimensions = 2,
momentum_dimensions = 3`: three momentum components on the transverse plane.

**3. `D_sT = 0`, `D_sT → 0⁺` and `:none` are three different limits.** See
["The weak-coupling limits"](#the-weak-coupling-limits).

## CPU vs GPU

Same algorithm, same keywords, same return shape. The CPU path is bit-reproducible under
`Random.seed!`; the GPU draws from CURAND and is reproducible in ensemble moments only. The host does
the sampling and the p_z completion on both paths, so the t0 snapshot is identical up to rounding.

## Tests and benchmarks

```
LIM_FAST=1 julia --project=. test/runtests.jl      # transport + unit + time-convention (≈30 s)
           julia --project=. test/runtests.jl      # + relativistic switch, momentum_dims3, CPU/GPU kernel parity, GPU-only paths (≈10 min)
           julia --project=. test/regression_corpus.jl   # bit-identity vs the committed baseline (CPU hashes, GPU moments)
```

`test_kernel_parity.jl` gives each CPU/GPU kernel pair the same inputs and the same noise arrays, so
the backends are compared per particle, at 1e-12 or tighter. The device uses FMA and the host does
not, so they differ at the ulp level: the interpolant and spline evaluator sub-ulp, the boosts 1–2
ulps, the force kernel ≈1.6e-14 relative.

`test/regression_corpus_baseline.txt` holds SHA-256 hashes of ten seeded runs across the keywords.
A change that keeps the dynamics must reproduce every hash. Regenerate with `LIM_CORPUS_WRITE=1` only
for a deliberate change, and note it in the CHANGELOG. The GPU half is statistical (its ⟨p_x⟩ check
fails about one run in twenty); `LIM_CORPUS_NOGPU=1` runs the CPU hashes only.

`bench/` (results in `bench/results/`):

| script | what it checks |
|---|---|
| `bench_physics_gates.jl` | MSD slope = 2·d·D_s at three z (CPU and GPU); the Jüttner tail p > 3 GeV at the Poisson floor; the ℓ=1 rate = λ₁η_D for 3 and 2 momentum rows; the Δt bias (none for Galilean, ≤ 1.5 % at ηΔt ≤ 0.1 relativistic); the Galilean MSD(t) at all t; `DsT_quad` ⇒ T-independent drag |
| `bench_gpu_parity.jl` | CPU ↔ GPU moments on four backgrounds and on inputs outside the table or with `\|v\| > 1`, plus a `freezeout_capture` case. The per-particle comparison is `test/test_kernel_parity.jl` |
| `bench_semianalytic.jl` | closed forms, with plots: the Uhlenbeck–Ornstein phase-space covariance including `σ_xp`; the exact BGK moment law at four Δt spanning 100×; free streaming with the Bjorken redshift; the equilibrium shape by two-sample KS in 2 and 3 momentum rows; the comoving blast wave per particle. Figures in `bench/results/figures/` |
| `bench_accuracy.jl` | the Δt budget: `\|⟨p²⟩ bias\| ≈ 12.0 %·(ηΔt)^0.94`, `D_s` unbiased to ≲1 % for ηΔt ≤ 0.08, and the RTA Δt ceilings |
| `bench_throughput.jl` | ns per particle-step and per-call overhead for CPU/GPU × N × momentum rows × relativistic × collision mode, plus the host-side phases (sampler, `randn!`, copies) |

Each ends in a `[PASS]`/`[FAIL]` line and exits non-zero on failure.

## Reference

<details>
<summary><b>The entry point and the keywords</b> (click to expand)</summary>

### The entry point

`simulate_ensemble_bulk(backend, …)` dispatches on the backend:

| method | what it does |
|---|---|
| `(CPUBackend(), r_grid, p_grid, f, T_field, v_field, (xgrid, tgrid); kw...)` | the main entry |
| `(GPUBackend(), …same…; kw...)` | CUDA version; exists only after `using CUDA` |
| `(CPUBackend(), T::Float64; kw...)` | homogeneous box, momenta only (toy: fixed `κ = 2.5T³`, ignores `DsT`) |

Returns `(time_points, momenta_snapshots, position_snapshots)`. `?simulate_ensemble_bulk` has the full
keyword table; the ones that set the physics:

| keyword | default | meaning |
|---|---|---|
| `m`, `DsT` | 1.0, 0.2 | quark mass [GeV]; `D_s·T`. The drag is the Einstein relation `1/η_D = tau_drag = m·DsT/T²` |
| `DsT_linear, DsT_slope, DsT_offset, Tfo` | off | `DsT(T) = slope·max(T, Tfo) + offset` |
| `DsT_quad, DsT_Tref` | off | `DsT(T) = DsT·(T/Tref)²` ⇒ a T-independent drag time |
| `dimensions` | 3 (**pass 2**) | 2 = transverse plane (x, y); 1 = radial mode (r, p_r) |
| `momentum_dimensions` | 0 (= `dimensions`) | 3 with `dimensions = 2`: a longitudinal `p_z` row (thermal conditional at t0). 2-D momenta relax to the 2-D Jüttner, whose current rate λ₁η_D differs from the 3-D `K₂/K₃·η_D` by 5–12 % over z = 3.5–10 |
| `bjorken_redshift` | false | `dp_z/dτ = −p_z/τ` between kicks (needs `momentum_dimensions = 3`, `initial_time > 0`) |
| `relativistic` | true | true: Jüttner kinematics, drag `η_D·m/E`, streaming `p/E`, Lorentz boosts. false: the exactly solvable Galilean process, drag `η_D`, streaming `p/m`, boosts `p∥ ∓ m·v` |
| `collision_mode` | `:langevin` | `:rta`: BGK re-draw from the local Jüttner with probability `1 − e^{−Δt/τ_n}`, τ_n the current time `tau_n_main3`. `:none`: free streaming, no drag, no noise, no frame change (the Bjorken redshift still applies). `DsT = 0` is the comoving limit, not free streaming |
| `x_init, p_init` | sampler | `(2, N)` lab positions and **rest-frame** momenta (the t0 lab boost is applied inside) |
| `cartesian_spatial_sampling`, `antithetic_momenta` | auto, false | sampler mode (disc rejection vs polar inverse-CDF); (p, −p) pairs |
| `position_diffusion`, `reflecting_boundary` | false | extra overdamped position kicks (double-counts vs hydro); reflect at `r = xgrid[end]` |
| `momentum_langevin` | true | false (or `DsT = 0`): particles move with the flow, `p = m·γ·v`, every momentum row beyond the spatial ones set to zero |
| `V2Evolutionn, psi2` | — | elliptic modulation `v → v(1 + 2v₂cos 2(φ−Ψ₂))` |
| GPU only: `freezeout_capture, freezeout_interp` | false | latch each particle's `T = Tfo` crossing; returns a NamedTuple `(pos, mom, tau, flag)` instead of histories. The run does not stop at freeze-out |
| GPU only: `integrator_mode` | 0 | 1 = drift-midpoint drag. It roughly doubles the Δt bias instead of removing it; the CPU refuses 1 |
| GPU only: `verbose` | false | print device + memory status at entry |

</details>

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
src/kernels_gpu.jl        CUDA kernels, line-for-line twins of the CPU ones (bench_gpu_parity.jl checks them)
src/simulate_gpu.jl       GPU driver (freeze-out capture, RTA inverse-CDF table)
src/simulate_gpu_wrapper.jl   the GPU method, included by the hook
examples/                 five runnable setups + example_common.jl (background builders) and a README
src/data/Fluidum_MIS_HQ.jld2  a Fluidum MIS background (23 MB), not read by the package
test/                     runtests.jl (drives everything below), regression_corpus.jl (+ baseline)
  test_kernel_units.jl        every primitive against an independent construction (interpolant, spline
                              evaluator, both boosts, the two Jüttner samplers, FONLL fidelity, the box path)
  test_kernel_parity.jl       CPU↔GPU, kernel by kernel, same injected noise, ≤1e-12
  test_gpu_only_paths.jl      freezeout_capture and integrator_mode = 1
  test_time_convention.jl     which step time each kernel reads the background at
  test_limits_and_contracts.jl the limits (D_sT → 0, comoving, free streaming) and the input
                              contract (a table that ends before final_time, a non-uniform grid, m ≤ 0)
  test_relativistic_switch.jl, test_momentum_dims3.jl, test_proper_time_kicks.jl,
  test_rta_proper_time.jl, test_bjorken_redshift_exact.jl
bench/                    bench_common.jl, bench_semianalytic.jl (closed forms + plots), bench_accuracy.jl
                          (the Δt budget), bench_physics_gates.jl, bench_gpu_parity.jl, bench_throughput.jl,
                          results/ (+ results/figures/)
```

---

## Licence and citation

© 2026 Ruwen Schulz. Apache License 2.0, see [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE).

If you use this engine in published work, please cite it ([`CITATION.cff`](CITATION.cff), or GitHub's
"Cite this repository" button) and say which version you ran. Changes that move results are listed
in [`CHANGELOG.md`](CHANGELOG.md).

| | DOI |
|---|---|
| all versions (resolves to the newest) | [10.5281/zenodo.22791006](https://doi.org/10.5281/zenodo.22791006) |
| v0.2.4 | [10.5281/zenodo.22791456](https://doi.org/10.5281/zenodo.22791456) |
