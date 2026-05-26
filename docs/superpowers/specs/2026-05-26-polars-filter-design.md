# Polars Ships-Sent Deferred Expansion — `70-Polars_filter.py`

**Date:** 2026-05-26
**Context:** Agent 68 reduced `.collect()` from 260 ms to 71 ms per call by moving the `prev_pos_lf` join after a coarse distance pre-filter. The remaining bottleneck is that `ships_sent` is expanded early (in `mine_lf`) before the cross-join with targets, making the cross-join N_src × N_ships_sent × N_targets × N_steps rows. This spec restructures the pipeline to defer ships_sent expansion until after a spatial reachability filter, shrinking the cross-join by the average ships_sent range (50–200×).

---

## Background

### Current Agent 68 pipeline shape

```
mine_lf (ships_sent already expanded)
→ cross_join(df_lf)              # large: ~200k rows
→ filter(step/id)
→ with_columns(dist_tgt_src, fleet_speed, dist_min, dist_prev)
→ coarse filter + sun filter
→ join(prev_pos_lf)
→ swept-pair quadratic
→ collect()
```

### The problem

`mine_lf` explodes ships_sent before the cross-join. A source planet with 100 possible ship counts multiplies the cross-join by 100×. Most of these rows are immediately discarded by the coarse filter, but the expansion cost is already paid.

### The fix

Split the pipeline into three phases:
1. **Phase A** — cross-join at the *planet* level (one row per src planet), apply spatial + sun filter using actual per-tick planet displacement
2. **Ships_sent expansion** — on the small survivor set only
3. **Phase B** — fleet_speed-specific second filter, prev_pos join, swept-pair, angle cone

---

## Scope

File to create: `70-Polars_filter.py` (copy of `68-Polars_optimised.py` with `take_action` restructured)
Profile notebook: `71-Profile_agent70.ipynb`

---

## Design

### New: `planet_disp_lf`

Built from the already-computed `prev_pos_lf`. Gives the actual Euclidean displacement of each planet per tick — correct for all nature types (fix ≈ 0, moving = orbital arc, comet = path step).

```python
planet_disp_lf = (
    df_lf.select(["id", "step", "x", "y"])
    .join(prev_pos_lf, on=["id", "step"], how="left")
    .with_columns(
        ((pl.col("x") - pl.col("x_prev").fill_null(pl.col("x"))).pow(2) +
         (pl.col("y") - pl.col("y_prev").fill_null(pl.col("y"))).pow(2)
        ).sqrt().alias("planet_disp")
    )
    .select(["id", "step", "planet_disp"])
)
```

Size: N_planets × (N_steps+1) ≈ 396 rows max.

### Modified: `mine_base_lf`

Same group_by aggregation as current `mine_lf` but **stops before** `with_columns(int_ranges(...)).explode("ships_sent")`. One row per source planet.

### Restructured: Section 3

#### Phase A — coarse spatial filter (no ships_sent, no prev_pos)

Cross-join at planet level, join the tiny `planet_disp_lf`, apply reachability filter.

**Reachability condition:**
```
dist_tgt_src < (step_diff + 1) * MAX_SPEED + radius_src + PLANET_MARGIN + radius_tgt + planet_disp
```

- `(step_diff + 1) * MAX_SPEED` — max fleet travel distance over the tick (t ∈ [0,1])
- `radius_src + PLANET_MARGIN` — fleet spawn offset from source center
- `radius_tgt` — collision radius of target
- `planet_disp` — actual displacement of target planet during this tick (from simulation data)

**Sun filter** applied here too (doesn't need prev_pos).

```python
coarse_lf = (
    mine_base_lf
    .join(df_lf, how="cross")
    .filter((pl.col("step") > pl.col("step_src")) & (pl.col("id") != pl.col("id_src")))
    .join(planet_disp_lf, on=["id", "step"], how="left")
    .with_columns([
        dist_tgt_src.alias("dist_tgt_src"),
        step_diff.alias("step_diff"),
    ])
    .filter(
        (pl.col("dist_tgt_src") < (pl.col("step_diff") + 1) * MAX_SPEED
         + pl.col("radius_src") + PLANET_MARGIN + pl.col("radius")
         + pl.col("planet_disp").fill_null(0.0))
        & ~crossing_sun
    )
)
```

#### Ships_sent expansion — survivors only

```python
expanded_lf = (
    coarse_lf
    .with_columns(
        pl.int_ranges(
            1,
            pl.col("ships_min") + pl.col("production_src") * nb_steps_sim + 1,
            dtype=pl.Int64,
        ).alias("ships_sent")
    )
    .explode("ships_sent")
)
```

#### Phase B — exact filter + swept-pair + angle cone

```python
pa_lf = (
    expanded_lf
    .with_columns([
        fleet_speed.alias("fleet_speed"),
        dist_min.alias("dist_min"),
        dist_prev.alias("dist_prev"),
    ])
    .filter(
        pl.col("dist_tgt_src") < pl.col("dist_min") + pl.col("fleet_speed")
        + pl.col("radius") + PLANET_MOVEMENT_SLACK
    )
    .join(prev_pos_lf, on=["id", "step"], how="left")
    .with_columns([t1_expr.alias("t1"), t2_expr.alias("t2"), collision.alias("collision")])
    .filter(pl.col("collision"))
    # ... angle cone (t1_eff, t2_eff, p_t1/t2, angle_t1/t2, d_s/f, angle_radius, angle_min/max, angle)
    .sort("step")
)
```

Sections 4 (blocking self-join + collect) and 5 (comet handling + scoring) — **unchanged**.

---

## What does NOT change

- `swept_pair_hit` function
- `interpreter`
- `_simulate`
- Section 4 (blocking self-join)
- Section 5 (comet handling + scoring)
- `nearest_planet_sniper`
- `PLANET_MOVEMENT_SLACK` constant (still used in Phase B exact filter)

---

## Invariants

- Phase A cross-join is N_unique_src_planets × N_target_planets × N_steps (no ships_sent dimension)
- Ships_sent expansion happens exactly once, on Phase A survivors only
- `planet_disp` is non-negative; `fill_null(0.0)` handles the first simulation step (no previous position)
- Phase B filter with `fleet_speed` is a strict subset of Phase A (never admits rows Phase A rejected)
- Collision semantics identical to Agent 68 — only pipeline shape changes
