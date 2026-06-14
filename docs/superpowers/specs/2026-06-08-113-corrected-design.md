# 113-Polars_GNN_Corrected — Design Spec

## Problem

`108-Simulate20Next_GNN_Polars.py` passes `pa` (mine-owned planets only) to `_05_get_GNN` for
reaches edges. Reaches should connect **all** planet snapshots, not just those reachable from the
player's planets. This makes the graph much sparser and misses cross-board connectivity.

## Goal

Create `113-Polars_GNN_Corrected.py` — a corrected copy of `108` — where:
- `reaches` edges are built from ALL planets using a fixed `SHIPS_LIST = [4, 16, 64, 256]`
- `_05_get_GNN` signature changes to `(df_s, reach_df, safe_attacks)`, dropping `pa`

## Approach (selected)

Approach A — add `_02_get_reach` alongside existing methods. No changes to `_02_get_all_opportunities`.

## Changes vs. `108`

### 1. New constant

```python
SHIPS_LIST = [4, 16, 64, 256]
```

### 2. New method: `StrategyPipeline._02_get_reach`

```python
@staticmethod
def _02_get_reach(
    df_s: pl.DataFrame,
    planet_disp: pl.DataFrame,
    ships_list: list[int] = SHIPS_LIST,
) -> pl.LazyFrame:
```

Same geometry pipeline as `_02_get_all_opportunities`, except:
- **Sources**: ALL planet first-snapshots (`group_by("id").agg(pl.first(...))` — no owner filter)
- **ships_sent**: discrete values from `ships_list` (not `int_ranges`)
  - `pl.lit(pl.Series(ships_list, dtype=pl.Int64)).alias("ships_sent")` → explode
- Returns the same column schema as `_02_get_all_opportunities` so `_03_filter_collision` works unchanged

Column schema required by `_03_filter_collision`:
`id_src, step_src, x_src, y_src, radius_src, ships_min, production_src, nature_src, owner_src,
id, step, x, y, radius, ships, production, owner, nature, dist_tgt_src, step_diff,
ships_sent, fleet_speed, dist_min, dist_prev, t1, t2, collision, t1_eff, t2_eff,
p_t1_x, p_t1_y, p_t2_x, p_t2_y, angle_t1, angle_t2, d_s_t1, d_s_t2, d_f_t1, d_f_t2,
angle_radius_t1, angle_radius_t2, angle_min, angle_max, angle, planet_disp, x_prev, y_prev, final_angle`

`ships_min` in reach context = minimum ships across all simulation steps for that planet
(same `pl.min("ships")` aggregate, kept so the column exists — not used for filtering in
`_02_get_reach`, only needed for schema compatibility).

### 3. Updated method: `_05_get_GNN(df_s, reach_df, safe_attacks)`

- Remove `pa` parameter
- Replace `pa` with `reach_df` in reaches edge construction
- Attack node logic unchanged

### 4. Updated `agent()`

```python
pa_df    = StrategyPipeline._02_get_all_opportunities(df_s, planet_disp, 0).collect()
safe_df  = StrategyPipeline._03_filter_collision(pa_df.lazy()).collect()

reach_raw = StrategyPipeline._02_get_reach(df_s, planet_disp).collect()
reach_df  = StrategyPipeline._03_filter_collision(reach_raw.lazy()).collect()

data = StrategyPipeline._05_get_GNN(df_s, reach_df, safe_df)
```

## Out of scope

- Updating `109-Train_GNN_Polars.py` (separate task)
- Changing `OrbitGNN` model architecture
- Changing `SHIPS_LIST` values
