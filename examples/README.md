# Examples

Five runnable setups, smallest first. Each is standalone, prints measured numbers next to the
closed form or the expectation they should match, and writes a figure to `examples/figures/`.

```sh
julia --project=Julia Julia/LangevInMedium.jl/examples/01_uniform_bath.jl
LIM_NOPLOT=1 julia --project=Julia Julia/LangevInMedium.jl/examples/01_uniform_bath.jl   # numbers only
```

They share `example_common.jl`, which holds the two background builders (`uniform_bath`,
`bjorken_fireball`), a FONLL-shaped initial density, and an independent Jüttner sampler used as a
reference. None of that is part of the package — it is the boilerplate a driver would otherwise
carry, kept out of the way so each example is only about the engine.

| | what it sets up | what to look at |
|---|---|---|
| `01_uniform_bath.jl` | a box at fixed `T`, no flow, a δ-function initial momentum | ⟨p²⟩ → the Jüttner value, the ℓ=1 current decaying at exactly `(K₂/K₃)η_D`, the MSD slope = `2·d·D_s`, and the final `\|p\|` histogram on top of the exact Jüttner. Everything here has a closed form — if the engine breaks, it breaks here first |
| `02_bjorken_fireball.jl` | a cooling, radially expanding fireball; the engine samples from a FONLL-shaped density; `DsT_linear`; freeze-out read off the snapshots | the radial flow picking the `p_T` spectrum up, and the freeze-out/initial ratio. Also the cost of doing freeze-out from snapshots: it is resolved to the save cadence, not to `Δt` |
| `03_four_limits.jl` | Langevin, RTA, `DsT = 0`, `DsT → 0⁺` and `collision_mode = :none`, on one background | **the example to read if you read one.** `DsT = 0` is the *comoving* limit (`p = m·γ·v`, no thermal width), `DsT → 0⁺` is a different limit again (thermal comoving, ≈ 20 % higher `⟨p_x⟩`), and free streaming is `:none`. Three places in this repository had those confused |
| `04_pz_and_rapidity.jl` | `momentum_dimensions = 3`, both `pz_init` modes, the Bjorken redshift, `track_eta_s` | what row 3 *means* (`p_z* = m_T sinh(y − η_s)`, not a lab `p_z`), how fast the two initialisations are forgotten, and the `dN/dy` kernel `P(K)` that makes `dN/dy = ρ(η_s) ⊛ P(K)` exact |
| `05_gpu_freezeout.jl` | the GPU path with `freezeout_capture` | the production pattern: memory ∝ `N` instead of `N·(saves+1)` (≈ 500× here), the crossing resolved to `Δt`, and the fact that the run does **not** stop at freeze-out. Falls back to the CPU without CUDA |

## Things worth taking from them

- **`p_init` is a LOCAL-REST-FRAME momentum.** The engine applies the `t0` lab boost itself. On a
  flowing background that is a factor of `γ(1 + v·…)`, not a detail.
- **`dimensions` sets the positions, `momentum_dimensions` the momenta.** A 2-D momentum run relaxes
  to the 2-D Jüttner, whose ℓ=1 rate differs from the 3-D `K₂/K₃·η_D` by 5–12 % over `z = 3.5–10` —
  and the hydro coefficients are matched in the 3-D theory. `momentum_dimensions = 3` with
  `dimensions = 2` is the combination that removes that offset.
- **One transport coefficient, not two.** `D_sT` fixes `τ_drag = m·D_sT/T²`, and `τ_n = τ_drag·K₃/K₂`
  is *derived*. Building the drag from `tau_n_main3` applies `K₃/K₂` once too often — it was the
  state of every product in this repository before 2026-08-02.
- **`proper_time_kicks = true` on any flowing background.** The undilated lab-`Δt` kick makes the
  stationary state `f_J/(γ(1+v·v_r))` instead of the boosted Jüttner, i.e. a spurious inward
  diffusion current. It defaults to `false` only so that older products stay bit-identical.
- **The step size.** `η_DΔt ≈ 0.01` costs ≈ 0.1 % on `⟨p²⟩` (the pre-point relativistic drag is
  O(ηΔt)); production runs at `η_DΔt ≈ 3·10⁻³`. `bench/bench_accuracy.jl` turns a required accuracy
  into a required `Δt`.

## The benchmarks these examples lean on

`bench/bench_semianalytic.jl` checks the engine against closed forms with plots — the
Uhlenbeck–Ornstein phase-space covariance including the `σ_xp` cross-correlation, the exact BGK
moment law at four step sizes, free streaming with the Bjorken redshift, the equilibrium *shape* by
KS, and the comoving blast wave per particle. `bench/bench_physics_gates.jl` is the pass/fail
version of the same idea in physical units; `bench/bench_accuracy.jl` is the `Δt` budget.
