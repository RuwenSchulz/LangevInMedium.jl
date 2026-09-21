# LangevInMedium — notes moved out of the README

Moved here from `README.md` on 2026-09-21, unchanged, so the pointers elsewhere that cite them
("Known biases and limits", "The LIMITS and the INPUT CONTRACT") still land somewhere. The
defect ledgers are also in `CHANGELOG.md` under 0.2.3 and 0.2.4.

---

### Known biases and limits

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

#### The LIMITS and the INPUT CONTRACT (found 2026-09-02, all FIXED in 0.2.3)

The 0.2.1 audit covered every function and 0.2.2 the p_z frame. Neither asked what the engine does
when it is asked for a *limit*, or handed an input outside its assumptions. Seven defects came out
of that question. All are fixed; each is gated in `test/test_limits_and_contracts.jl` (41 assertions)
with the pre-fix measurement in the comment, so a regression has something specific to fail against.
⚠ **Three of the seven move numbers** — see "What regenerates" below.

- 🔴🔴 **`sample_particles_from_FONLL` assumed a UNIFORM grid, and was FIRST ORDER even there.**
  Its inverse CDF was `cumsum(w) * mean(diff(grid))` — a right-Riemann sum with one constant
  spacing. Now a cumulative trapezoid on the actual nodes (`Utils._cumtrapz`). Against the exact
  ⟨p⟩ of `P(p) ∝ p·f(p)` on the same range, FONLL-like `(1+(p/2.1)²)^(−3.1)`, 12 independent seeds:

  | p grid over [0, 10] | before | after |
  |---|---|---|
  | uniform, np = 100 | −3.14 % | −0.01 ± 0.05 % |
  | uniform, np = **300 (production)** | **−1.08 %** | **−0.08 ± 0.05 %** |
  | uniform, np = 1200 | −0.29 % | −0.09 ± 0.05 % |
  | log-spaced, 300 | **−44.6 %** | −0.07 ± 0.05 % |

  Two separate errors were in there. On a uniform grid the old rule converged (O(Δp)), so the
  production np = 300 cost ≈ −1 % on ⟨p_T⟩ and −1 % on ⟨p_T²⟩ of every FONLL initial condition —
  partly cancelling in a ratio carrying the same IC top and bottom. On a NON-uniform grid it was
  simply the wrong quadrature and refinement did not help (−25.7 % at 160 log-spaced points, still
  −24.7 % at 1600). The residual is now grid-INDEPENDENT, which is the signature that the
  quadrature is right, and sits at the measurement's own resolution.
- 🔴 **The glued-to-the-flow limit forgot the `p_z` row.** `kernel_set_to_fluid_velocity_*` wrote
  momentum rows 1–2 only, on both backends, so with `momentum_dimensions = 3` row 3 kept whatever
  the IC put there and still entered `E = √(m² + p_⊥² + p_z*²)`: the particle that is supposed to
  *be* the fluid element streamed slower than it. Measured at v = 0.5, T = 0.30, τ ∈ [0.4, 2.4]:
  ⟨v_x⟩ = **0.4645 against the fluid's 0.5000, −7.10 %** (CPU and GPU to six digits), ⟨p_z*²⟩ =
  0.596 GeV² alive at the end. Both kernels now zero every row beyond the spatial ones — the fluid
  is longitudinally comoving in Milne by construction. Measured after: **0.500000, deficit 0.0000 %,
  ⟨p_z*²⟩ = 0**. The old gate could not see it: `test_momentum_dims3.jl` "(R)" uses a ZERO-flow box.
- 🔴 **`D_sT` had three limits and no way to name the third.** `DsT == 0.0` branches into the glue
  kernel: `p = m·γ·v`, the **cold comoving** limit (measured 0.86603 = m·γ·v exactly). `DsT → 0⁺` is
  a *different* limit — thermal comoving, ⟨p_x⟩ = γv⟨E*⟩ = **1.0350**, 19.5 % higher. Free streaming
  was neither, and was reachable only through a **negative** `DsT`, by accident. Three places in the
  tree called `DsT = 0` "free streaming" (this README's trap list, `CLAUDE.md`,
  `Projects/SpectraDiagnostic/plot_dst_sweep.jl`, and `AttractorPaper5/Code/run_langevin_kompost.jl`,
  which *ran* it and labelled the curve). Now **`collision_mode = :none` is free streaming**, on both
  backends: no drag, no noise, and no boost pair either, so the momenta are exactly constant (the
  only residue is the documented γ regularisation in the single t0 lab boost, 6.7e-11 at v = 0.5).
  The Bjorken redshift still applies under `:none`, being the longitudinal free-streaming law.
- 🔴 **`m ≤ 0` and `DsT < 0` degraded SILENTLY to free streaming** — `tau_drag` returns 0.0 for any
  non-positive argument and every kernel reads `τ ≤ 0` as `η_D = κ = 0`, so a mistyped mass gave a
  free-streaming run indistinguishable from a Langevin run except by its numbers (measured: ⟨p²⟩
  frozen to 1e-9, nothing said). Both are now refused with a message that names the alternative.
  `DsT == 0` stays legal — it is the comoving limit, not an error.
- 🔴 **The background table was never checked against the requested window.** `interpolate_2d_*`
  clamps into the table — right for a particle at the rim, wrong for a run that outlives the hydro
  output: past `tgrid[end]` the medium froze at its last slice and the run continued, silently, on
  both backends. The clamping is KEPT (it is occasionally deliberate) and is now announced, by two
  once-per-run warnings: the window check at entry, and an escaped-particle count at exit
  (measured: on a table cut at r = 8 fm, **16.2 %** of a 20 000-particle ensemble finished outside
  it, out to r = 15.2 fm, dragged at the rim `T` and `v` the whole way). Neither carries `maxlog`,
  deliberately: `maxlog` is keyed by source location, so it would silence every call after the
  first in a campaign that drives the engine many times — exactly the silence they exist to break.
- **The step count is no longer `floor`ed on a quotient that is not exactly representable.**
  `1.4 − 0.4 = 0.9999999999999999`, so `q = 999.9999999999999` and a whole step was lost. Usually
  10⁻³ fm; the damage was that it also broke `steps % save_every == 0`, and `_snapshot_times` then
  dropped the entire trailing save interval and blamed `save_interval` for "not dividing the
  evolution". Measured worst case: t0 = 0.4, tf = 1.4, Δt = 10⁻³, `save_interval` = 0.5 kept **501
  of 1000 steps — half the requested history**. `_step_count` snaps within 64 ulps of the quotient,
  which is ~4 orders above any representation error and ~3 below any shortfall a caller could mean
  (gated both ways).
- **The RTA/BGK collision probability is exponential**, `−expm1(−Δt·dil/τ_n)`, not the linearised
  `Δt/τ_n`. The old form made the survival probability `1 − Δt/τ_n` and the realised rate
  `−ln(1 − Δt/τ_n)/Δt`: always too fast. Measured ratio to the nominal `1/τ_n` at
  Δt = 0.002 / 0.01 / 0.05 / 0.1 / 0.2 (τ_n = 0.5976 fm):

  | | 0.002 | 0.01 | 0.05 | 0.1 | 0.2 |
  |---|---|---|---|---|---|
  | before | 1.0013 | 1.0069 | 1.0517 | 1.0936 | **1.2198** |
  | after | 1.0049 | 1.0070 | 0.9970 | 1.0014 | **1.0006** |

  The Δt-dependence is gone (the residual ±0.5 % is the ensemble's own noise at N = 2·10⁵), and with
  it the step-size ceiling the RTA used to carry.
- Two things checked and found **correct**, pinned so a later change has something to fail against:
  `reflecting_boundary` preserves the uniform disc measure (⟨r⟩ and ⟨r²⟩ within 0.11 % over 20 fm,
  no escapes), and `track_eta_s` is an exact passenger (momenta and positions bit-identical with it
  on and off, max |Δ| = **0.000e+00**). The 0.2.1 hot-loop rewrite has held: **0 bytes per
  particle-step** across CPU × N ∈ {2·10⁴, 10⁵} × pdim ∈ {2, 3} × {`:langevin`, `:rta`}.

#### THE DIAGNOSTICS THAT COULD NOT BE HEARD (found 2026-09-15, all FIXED in 0.2.4)

0.2.3 asked what the engine does at a limit. This pass asked the next question — whether the
warnings it added can actually be *heard* in the way the engine is really driven, which is a
campaign of many runs in one session. Three defects, none of which moves a number: all ten corpus
CPU hashes reproduce bit for bit and the GPU moments match.

- 🔴 **The dropped-history warning carried `maxlog = 1`, so only the FIRST affected run in a session
  said anything.** `maxlog` is keyed by source location — exactly the reason the 0.2.3 window and
  escaped-particle warnings were deliberately written without it. Losing history silently is the one
  thing this warning exists to prevent, and it prevented it once per session. Measured: five
  consecutive `_snapshot_times` calls at t0 = 0.4, tf = 1.4, Δt = 10⁻³, `save_interval` = 0.5 each
  returned history only out to **0.9 fm of a requested 1.4 fm — half the evolution — and four of the
  five were silent.** The `maxlog` is gone, and the message now names `requested_final_time` and
  `last_snapshot` so the size of the loss is in the warning itself. Guarded in `runtests.jl`: five
  calls must yield five warnings. The guard was falsified against the old behaviour (it sees one).
- **`_to_cdf!`'s tie-breaking nudge was an absolute `eps(Float64)` and did nothing above a
  cumulative of ≈10³.** `c[k-1] + 2.2e-16` is not representable next to a value of 10³, so `max`
  returned `c[k-1]` and the CDF came out flat exactly where the guard was meant to bite — while the
  docstring promised "strictly increasing". Measured: **149 tied knots at every overall scale ≥ 10³,
  0 at scale 1**, on a FONLL-shaped density with a hard cutoff over 300 nodes. No result moved (the
  ties fall in the zero-density tail and the sampled ⟨p⟩ was bit-identical from scale 1 to 10⁹),
  which is why nothing caught it; a tie in a populated region would not have been harmless. Now
  `nextfloat`, the representable step at any magnitude.
- 🔴 **`bench_physics_gates.jl` gate (b) is labelled "tail after 10 τ_drag" and measured the tail
  after 5.0.** These gates take `tfinal` from a physical time (`10/η_D` = 3.826 fm), so `tfinal/Δt`
  is an arbitrary real, `steps % save_every ≠ 0`, and with `save = tfinal/2` the whole trailing save
  interval went: last snapshot **1.914 fm, exactly half the window**. Gate (a) lost 12.4 % of its
  diffusive window at T = 0.45 and 0.30. **Both gates still passed** — (b) starts in equilibrium, so
  what it measures is stationary — so nothing announced it except the warning that the defect above
  had silenced. `box_run` now snaps `tfinal` to an exact multiple of the save interval; all 18 gates
  still pass, over the windows their labels claim.

#### What regenerates

The corpus says it precisely: **6 of the 10 CPU hashes are unchanged**, and the 4 that moved are
exactly `sampler_cart`, `sampler_polar`, `radial_dim1` (the trapezoid) and `rta_flow` (the
exponential). Nothing leaked into the injected-particle (`x_init`/`p_init`) Langevin path. So:

| fix | what it touches | who |
|---|---|---|
| FONLL trapezoid | every run that lets the engine sample (`heavy_quark_density`, no `x_init`) | **LP1, O+O, AM, KA — every FONLL IC in the tree**; ⟨p_T⟩ of the IC moves ≈ +1 % |
| RTA `−expm1` | `collision_mode = :rta` only | LP1's two RTA cells; < 0.1 % at production Δt |
| step count | windows whose `(tf − t0)/Δt` was mis-floored | **AttractorHydro's portrait** (0.4 → 13.0 at Δt = 10⁻⁴): 125 999 → 126 000 steps, 1260 → 1261 snapshots. LP1's 12.6 fm and O+O's 7.6 fm divide exactly and are untouched |
| glue `p_z`, validation, warnings, `:none` | nothing in production | no driver passes `momentum_langevin = false`, `m ≤ 0` or `DsT < 0`; `:none` is new |


---

## Tests (was in the README)

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
Regenerated once, at 0.2.3, for the four cases the FONLL-trapezoid and RTA-`expm1` fixes move
(`sampler_cart`, `sampler_polar`, `radial_dim1`, `rta_flow`); the other six reproduced and that is
what says the fixes are scoped to the sampler and the BGK step and nothing else.
Its GPU half is a *statistical* check and its ⟨p_x⟩ gate is a ~2σ test against a single seeded CPU
draw, so it fails roughly one run in twenty on that field alone (CHANGELOG 0.2.1) — the CPU hashes
are the deterministic part, and `LIM_CORPUS_NOGPU=1` runs only those.


The engine is wired into the repo gate as `programme.jl check` → `engine`
(`Julia/Projects/test_langevinmedium_engine.jl`), which runs the deterministic half only:
`LIM_FAST=1 runtests.jl` plus the CPU bit-identity corpus.

## The weak-coupling limits (was in the README)

The two zero-coupling limits differ because the order of limits matters. At `D_sT = 0` the noise is
switched off before the drag has anywhere to relax to, so the particle *is* the fluid element. At
`D_sT → 0⁺` drag and noise stay in Einstein balance all the way down, so the equilibrium is the
local Jüttner however small the coefficient gets. `:none` removes the collision term itself, which
is a third thing again.
