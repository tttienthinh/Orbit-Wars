# Design: 87-Polars_Simulate10Next_Conqueror_Supplier_fixed.py

**Date:** 2026-06-01  
**Status:** Approved  
**Goal:** Port `84-Simulate10Next_Conqueror_Supplier_fixed.py` to Polars Lazy, preserving the 4-method `StrategyPipeline` class structure, with a single `.collect()` in `_04`.

---

## Architecture

File layout mirrors `84` exactly:

```
GameConfig          ← unchanged (constants class)
PhysicsEngine       ← unchanged (distance, point_to_segment_distance, swept_pair_hit, fleet_speed)
interpreter()       ← unchanged (Python-native game loop, mutates obs)
StrategyPipeline    ← class shape unchanged; method internals converted to Polars
  _01_get_obs_dataframe(obs, step, num_agents)  → (pl.DataFrame, pl.DataFrame)
  _02_get_all_opportunities(df_s, planet_disp, player_id) → pl.LazyFrame
  _03_filter_collision(pa_lf)                   → pl.LazyFrame
  _04_score_and_decide(safe_lf, player_id)      → list
agent()             ← unchanged logic; variable names stay the same
```

The lazy chain is:
```
_02 builds LazyFrame → _03 extends it → _04 calls .collect() once
```

---

## Method Specifications

### `_01_get_obs_dataframe` → `(pl.DataFrame, pl.DataFrame)`

- Build `rows` list in Python exactly as in `84` (game interpreter is Python-native; cannot vectorize).
- Return `pl.DataFrame(rows).sort("step")` instead of `pd.DataFrame`.
- Compute `planet_disp` with a Polars lazy join:
  - Shift `step+1`, join on `(id, step)`, compute `sqrt((x-x_prev)^2 + (y-y_prev)^2)`, collect.
- Returns `(pl.DataFrame, pl.DataFrame)` — both eagerly evaluated; simulation must run to completion.

### `_02_get_all_opportunities` → `pl.LazyFrame`

Inputs: `(pl.DataFrame, pl.DataFrame, int)`.  
Converts inputs to lazy with `.lazy()`, then runs the full chain:

1. **`mine_base_lf`** — `group_by("id").agg(first/min/len).filter(row_count==is_mine and owner==player_id).rename(id→id_src)`
2. **Phase A: coarse cross-join** — `mine_base_lf.join(df_s_lf, how="cross")`, filter `step > step_src and id != id_src`, join `planet_disp_lf`, compute `dist_tgt_src` and `step_diff`.
3. **Sun-crossing filter** — vectorized Polars expressions (same math as `84`, expressed as `pl.Expr`).
4. **Ships-sent expansion** — `pl.int_ranges(1, ships_min + production_src * NB_STEPS_SIM + 1)` then `.explode("ships_sent")`.
5. **Phase B: fleet-speed filter** — compute `fleet_speed`, `dist_min`, `dist_prev`; filter by distance bound.
6. **Prev-pos join** — join shifted `(id, step+1)` frame to get `x_prev, y_prev`.
7. **Swept-pair collision** — full quadratic discriminant in Polars expressions; filter `collision == True`.
8. **Angle geometry** — `t1_eff`, `t2_eff`, `p_t1/t2_xy`, `angle_t1/t2`, `d_s/f_t1/t2`, `angle_radius_t1/t2`, `angle_min`, `angle_max`, `angle` (arctan2 of summed sin/cos).

Returns `pl.LazyFrame`. No `.collect()`.

### `_03_filter_collision` → `pl.LazyFrame`

Input: `pl.LazyFrame` (the `pa_lf` from `_02`).

- Self-join `pa_lf` on `["id_src", "ships_sent"]` with suffix `_obs`.
- Filter: `step_obs < step` and `id_obs != id`.
- Apply cone check: `angle % 2π` falls within `(angle_min_obs, angle_max_obs)` with wrap-around handling.
- `.select(["id_src", "ships_sent", "step", "id"]).unique()` → `blocked_lf`.
- Anti-join `pa_lf` against `blocked_lf` on `["id_src", "ships_sent", "step", "id"]`.
- `.with_columns(pl.col("angle").alias("final_angle"))`.

Returns `pl.LazyFrame`. No `.collect()`.

### `_04_score_and_decide` → `list`

Input: `pl.LazyFrame` (the `safe_lf` from `_03`).

**Single collect at the top:**
```python
attacks_with_angle = safe_lf.collect()
if attacks_with_angle.is_empty():
    return moves
```

Then eager Polars mirrors the `84` scoring logic:

1. **Comet evasion** — filter `nature_src == "comet"`, check offset > 45, emit moves, exclude comet sources.
2. **Top-5 per source** — `sort(["step","ships_sent"]).group_by(["id_src","id"]).first().group_by("id_src").head(5)`.
3. **Supplier/Conqueror classification** — `mine_src_ids = set(attacks_with_angle["id_src"].unique().to_list())`. Join top5 with mine flag; `group_by("id_src").agg(mine_count, total_count)`; `status = "Supplier" if mine_count == total_count else "Conqueror"`.
4. **Conqueror attacks** — filter to conqueror sources; filter `owner != player_id`; compute `ships_needed`; filter to `ships_needed+1 <= ships_sent <= ships_needed+production_src+1`; compute `time_cost`, `total_time_cost`, `score = (total_time_cost - time_cost - step_diff) * production`; sort desc score; one per source; guard `ships_sent <= ships_min`.
5. **`conqueror_needs`** — `group_by("id_src").agg(ship_min, all_need, lowest_need, nb_need)`.
6. **Supplier reinforcement** — filter to supplier sources; join to `conqueror_needs` on `id == id_src`; apply `(lowest_need - ships_min) * 1.5 < ships_sent` and `ships_min * 0.75 < ships_sent < ships_min`; one per source.
7. **Emit** — `pl.concat([attacks_conqueror, attacks_supplier], how="diagonal").select(["id_src","final_angle","ships_sent"]).rows()`.

No additional `.collect()` calls inside `_04`.

---

## Type Signature Table

| Method | Input types | Output type |
|--------|-------------|-------------|
| `_01` | `obs, step: int, num_agents: int` | `(pl.DataFrame, pl.DataFrame)` |
| `_02` | `pl.DataFrame, pl.DataFrame, int` | `pl.LazyFrame` |
| `_03` | `pl.LazyFrame` | `pl.LazyFrame` |
| `_04` | `pl.LazyFrame, int` | `list` |

---

## Implementation Notes

- **Import**: add `import polars as pl`; drop `import numpy as np` and `import pandas as pd` (no longer needed after `_01`).
- **`_01` keeps `import math`** for `math.hypot` in the row-building loop.
- **`pl.int_ranges`** replaces `list(range(...))` + `.explode()`.
- **`pl.arctan2`, `pl.arccos`** replace `np.arctan2`, `np.arccos`.
- **`pl.min_horizontal` / `pl.max_horizontal`** replace `np.minimum` / `np.maximum`.
- **`group_by(..., maintain_order=True)`** matches pandas `groupby(sort=False)` ordering semantics.
- **`how="diagonal"` concat** handles column superset mismatches between `attacks_conqueror` and `attacks_supplier`.
- **`agent()`** variable names: `df_s, planet_disp` (pl.DataFrame), `pa_lf, safe_lf` (pl.LazyFrame), `moves` (list).

---

## What Does Not Change

- `GameConfig`, `PhysicsEngine`, `interpreter` — verbatim from `84`.
- `agent()` control flow — verbatim; only local variable types change.
- The Conqueror/Supplier scoring logic and thresholds.
- The `angular_velocity`-based orbital position formula.
