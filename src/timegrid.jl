module TimeGrid

# The variable time step (added 2026-10-12). The engine's kernels see time ONLY through
# `t0 + step·Δt` (and the Bjorken/η_s kernels through `t0 + (step−1)·Δt`), so a step of length
# h_k from t_{k−1} is the call `(step = 1, t0 = t_{k−1}, Δt = h_k)` with no kernel touched. The
# drivers therefore need only a vector of step boundaries; this module builds one from the
# background and checks one that a caller hands in.

using ..Transport: tau_drag, effective_DsT

export adaptive_time_grid, check_time_grid

"""
    _drag_rate(T, m, DsT; prescription...) -> η_D [fm⁻¹]

The Einstein drag rate the force kernel builds, `1/tau_drag(T)`, with the same `D_sT(T)`
prescription; 0 where `tau_drag` is 0 (T ≤ 0: no drag, nothing to resolve).
"""
@inline function _drag_rate(T, m, DsT; kw...)
    τ = tau_drag(T, m, effective_DsT(T, DsT; kw...))
    return τ > 0 ? 1 / τ : 0.0
end

"""
    adaptive_time_grid(T_field, tgrid; initial_time, final_time, save_interval, m, DsT,
                       drag_tol = 0.01, bg_tol = 0.01, dt_min = 1e-5, dt_max = 0.1,
                       drag_active = true, DsT_linear, DsT_slope, DsT_offset, Tfo, DsT_quad, DsT_Tref)
        -> (grid, save_steps)

Step boundaries `grid` (t_0 = `initial_time` < t_1 < … < t_n = `final_time`) for one ensemble,
and the step numbers after which a snapshot is taken. The step is global (every particle takes
the same h_k) and depends only on the background and the transport parameters, never on the
particle state, so the grid is the same on the CPU and the GPU and for every seed.

`T_field` is the temperature table with time as its LAST axis (`T[r, τ]` or `T[x, y, τ]`) on
`tgrid`. For each table interval the builder records the hottest drag rate `η_max = max η_D(T)` and
the fastest change `η̇_max = max |∂_τ η_D|` over all cells, and takes at time t

    h = min( drag_tol / η_max,  bg_tol · η_max / η̇_max,  dt_max ),   clamped below at dt_min,

with the maxima over every table interval that the step [t, t + h] touches.

- `drag_tol` bounds `η_D·h` in the hottest cell. The exact-OU step freezes E (hence `η_eff = η_D M/E`)
  over the step; `bench/bench_accuracy.jl` measured the resulting stationary-⟨p²⟩ bias as
  ≈ 12 %·(η_D Δt)^0.94 (3 momentum rows) and ≈ 53 %·(η_D Δt)^1.56 (2 rows), i.e. ≈ 0.16 % at the
  default 0.01. A fixed Δt = 10⁻³ fm has η_D Δt ≈ 0.007 at T = 0.5 GeV (m = 1.5, D_sT = 0.116), so the
  default is close to production accuracy where the medium is hottest and lets the step grow as
  it cools.
- `bg_tol` bounds the change of the drag rate per step, relative to the hottest rate:
  `h·η̇_max ≤ bg_tol·η_max`. The kernels read the background at the END of the step; on a Bjorken
  centre (η_D ∝ T² ∝ τ^{−2/3}) this is `h ≤ 1.5·bg_tol·τ`. Cold cells cannot trigger it.
- `drag_active = false` (free streaming, the comoving limit) drops both rules: free streaming and
  the Bjorken redshift are exact at any step, and only `dt_max` applies.

Not resolved by these rules, deliberately: the change of the flow velocity along a step and the
spatial gradient a particle crosses. Both enter the error only as a product with `η_D·h`, i.e. at
second order once `drag_tol` holds; `dt_max` caps them.

Snapshots: one at every `initial_time + k·save_interval` strictly inside the window, and one at
`final_time`, all hit exactly — the grid lands on them, no history is dropped. When the remaining
distance to a snapshot is between one and two steps it is split in two equal steps, so no sliver
step is taken.
"""
function adaptive_time_grid(T_field::AbstractArray, tgrid::AbstractVector;
                            initial_time::Real, final_time::Real, save_interval::Real,
                            m::Real, DsT::Real,
                            drag_tol::Real = 0.01, bg_tol::Real = 0.01,
                            dt_min::Real = 1e-5, dt_max::Real = 0.1,
                            drag_active::Bool = true,
                            DsT_linear::Bool = false, DsT_slope::Real = 1.765, DsT_offset::Real = -0.159,
                            Tfo::Real = 0.156, DsT_quad::Bool = false, DsT_Tref::Real = 0.0)
    t0 = Float64(initial_time); tf = Float64(final_time); si = Float64(save_interval)
    tf > t0 || error("adaptive_time_grid: final_time = $tf must exceed initial_time = $t0")
    si > 0 || error("adaptive_time_grid: save_interval = $si must be positive")
    (drag_tol > 0 && bg_tol > 0) || error("adaptive_time_grid: drag_tol and bg_tol must be positive")
    0 < dt_min <= dt_max || error("adaptive_time_grid: need 0 < dt_min ≤ dt_max (got $dt_min, $dt_max)")
    nt = size(T_field, ndims(T_field))
    nt == length(tgrid) || error("adaptive_time_grid: the last axis of T_field ($nt) must match tgrid ($(length(tgrid)))")
    tg = Float64.(collect(tgrid))
    issorted(tg) || error("adaptive_time_grid: tgrid must be increasing")

    # ── per-slice hottest rate and per-interval fastest change, over every cell ────────────────
    pres = (; DsT_linear, DsT_slope, DsT_offset, Tfo, DsT_quad, DsT_Tref)
    ηmax_slice = zeros(nt)
    ηdot = zeros(max(nt - 1, 1))
    if drag_active
        η = map(T -> _drag_rate(T, m, DsT; pres...), T_field)
        for j in 1:nt
            ηmax_slice[j] = maximum(selectdim(η, ndims(η), j))
        end
        for j in 1:nt-1
            dτ = tg[j+1] - tg[j]
            dτ > 0 || continue
            a = selectdim(η, ndims(η), j); b = selectdim(η, ndims(η), j + 1)
            ηdot[j] = maximum(abs.(b .- a)) / dτ
        end
    end
    # the table intervals [tg[j], tg[j+1]] overlapping [ta, tb]; outside the table the background
    # is clamped (frozen), so the edge slice holds with zero rate of change.
    function rates(ta, tb)
        ja = clamp(searchsortedlast(tg, ta), 1, max(nt - 1, 1))
        jb = clamp(searchsortedlast(tg, tb), 1, max(nt - 1, 1))
        ηm = 0.0; ηd = 0.0
        for j in ja:jb
            ηm = max(ηm, ηmax_slice[j], ηmax_slice[min(j + 1, nt)])
            if ta < tg[end] && tb > tg[1]
                ηd = max(ηd, ηdot[j])
            end
        end
        return ηm, ηd
    end
    function step_at(t)
        drag_active || return Float64(dt_max)
        h = Float64(dt_max)
        for _ in 1:8                         # the window grows with h, so h only shrinks: converges
            ηm, ηd = rates(t, t + h)
            hnew = Float64(dt_max)
            ηm > 0 && (hnew = min(hnew, drag_tol / ηm))
            ηd > 0 && (hnew = min(hnew, bg_tol * ηm / ηd))
            hnew = max(hnew, Float64(dt_min))
            hnew >= h && break
            h = hnew
        end
        return h
    end

    # ── snapshot targets: every t0 + k·si strictly inside, then tf ────────────────────────────
    tol = 1e-9 * max(1.0, abs(tf - t0))
    targets = Float64[]
    k = 1
    while true
        s = t0 + k * si                      # computed, never accumulated
        s < tf - tol || break
        push!(targets, s); k += 1
    end
    push!(targets, tf)

    grid = [t0]; save_steps = Int[]
    t = t0
    for s in targets
        while true
            h = step_at(t)
            rem = s - t
            if rem <= h * (1 + 1e-12)
                t = s                        # land exactly on the snapshot
            elseif rem < 2h
                t = t + rem / 2
            else
                t = t + h
            end
            push!(grid, t)
            t == s && break
        end
        push!(save_steps, length(grid) - 1)
    end
    return grid, save_steps
end

"""
    check_time_grid(grid, initial_time, final_time, save_interval) -> save_steps

Validate a caller's step boundaries and locate the snapshots on them: the grid must be finite,
strictly increasing, start at `initial_time` and end at `final_time` (to 1e-9 relative), and
contain every `initial_time + k·save_interval` inside the window to the same tolerance — a
snapshot that is not on the grid is an error, never moved to the nearest point.
"""
function check_time_grid(grid::AbstractVector, initial_time::Real, final_time::Real, save_interval::Real)
    g = Float64.(collect(grid))
    length(g) >= 2 || error("time_grid: need at least two points (one step)")
    all(isfinite, g) || error("time_grid: contains a non-finite entry")
    all(>(0), diff(g)) || error("time_grid: must be strictly increasing")
    t0 = Float64(initial_time); tf = Float64(final_time)
    tol = 1e-9 * max(1.0, abs(tf - t0))
    abs(g[1] - t0) <= tol || error("time_grid: starts at $(g[1]), initial_time = $t0")
    abs(g[end] - tf) <= tol || error("time_grid: ends at $(g[end]), final_time = $tf")
    save_interval > 0 || error("time_grid: save_interval = $save_interval must be positive")
    save_steps = Int[]
    k = 1
    while true
        s = t0 + k * save_interval
        s < tf - tol || break
        j = searchsortedfirst(g, s - tol)
        (j <= length(g) && abs(g[j] - s) <= tol) ||
            error("time_grid: the snapshot time $s (initial_time + $k·save_interval) is not a grid point")
        push!(save_steps, j - 1)
        k += 1
    end
    push!(save_steps, length(g) - 1)
    return save_steps
end

end # module TimeGrid
