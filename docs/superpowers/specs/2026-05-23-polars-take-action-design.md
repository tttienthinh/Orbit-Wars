# Polars Port: IntervalProcessorPolars + take_action_polars

**Date:** 2026-05-23  
**Status:** Approved

---

## Goal

Port `IntervalProcessor` and `take_action` from `44-Dataframe_comet.py` to Polars equivalents (`IntervalProcessorPolars`, `take_action_polars`) with identical output semantics and measurably faster wall-clock time. The result is validated in `45-Polars_comet.ipynb` against the test cases from `43-Dataframe_comet.ipynb`, then shipped as `46-Polars_comet.py` — a fully working Kaggle agent.

---

## Scope

**In scope:**
- `IntervalProcessorPolars` class (mirrors all 5 static methods of `IntervalProcessor`)
- `take_action_polars(df, player_id, nb_steps_sim, return_df)` function
- Notebook `45-Polars_comet.ipynb` with comparison + timing cells for each test
- Submission file `46-Polars_comet.py` with the Polars agent enabled

**Out of scope:**
- Changes to `_simulate`, `interpreter`, `_eta`, `_aim_angle`
- Changes to `IntervalProcessor` or `take_action` (kept as-is for reference)
- RL / Phase 2 work

---

## Architecture

### `IntervalProcessorPolars`

All five static methods preserved with the same signatures and list-based semantics. Changes per method:

| Method | Change |
|---|---|
| `merge_intervals(intervals)` | **No change** — pure Python list logic |
| `subtract_intervals(target_min, target_max, blocked)` | **No change** — pure Python list logic |
| `create_cumulative_obstacles(possible_attacks, min_step=0)` | Accepts and returns `pl.DataFrame`. Replaces `groupby` + `iterrows()` with a single `.to_dicts()` pass to build `attack_map`. Same sequential cumulative loop (inherently non-vectorizable). |
| `compute_free_angles(row)` | Called via `map_elements` on a Polars struct column. Logic unchanged; signature adapted to receive a dict from Polars. |
| `interval_to_final_angle(series)` | Accepts `pl.Series`, returns `pl.Series`. Uses `map_elements` instead of `.map`. |

### `take_action_polars`

**Signature:** `take_action_polars(df: pd.DataFrame, player_id: int, nb_steps_sim=NB_STEPS_SIM, return_df=False) -> list`

Accepts the same pandas DataFrame from `_simulate` (no change to caller). Converts to Polars at the top of the function. Returns the same `[[id_src, angle, ships_sent], ...]` list format.

**Pipeline:**

```
pd.DataFrame
    └─ pl.from_pandas(df)                    # convert once at entry
    └─ mine_across_sim                        # group_by + agg
    └─ expanded_mine                          # with_columns + explode
    └─ df_src_tgt                             # cross join + filter
    └─ possible_attacks                       # with_columns (vectorized)
          ├─ dist/speed/collision columns
          └─ crossing_sun (vectorized)        ← key optimization
    └─ IntervalProcessorPolars
          .create_cumulative_obstacles(...)
    └─ attacks_with_angle                     # join + map_elements(compute_free_angles)
    └─ planet_id_top_5_id_src                 # sort + group_by + head(5)
    └─ comet branch                           # same logic, Polars filter
    └─ attacks                                # join + score + group_by + first
    └─ [[id_src, final_angle, ships_sent]]    # to_numpy().tolist()
```

---

## Key Optimization: Vectorized `crossing_sun`

The pandas version calls `point_to_segment_distance` row-by-row via `apply(axis=1)`, which is the main bottleneck. The Polars version computes it as column expressions:

```
Given: p = sun = (CENTER, CENTER), v = (x_src, y_src), w = (x_tgt, y_tgt)

dx_vw = col("x") - col("x_src")
dy_vw = col("y") - col("y_src")
l2    = dx_vw² + dy_vw²

dot   = (CENTER - col("x_src")) * dx_vw + (CENTER - col("y_src")) * dy_vw
t     = (dot / l2).clip(0, 1)

proj_x = col("x_src") + t * dx_vw
proj_y = col("y_src") + t * dy_vw

dist_sun = sqrt((CENTER - proj_x)² + (CENTER - proj_y)²)

crossing_sun = dist_sun < SUN_RADIUS + PLANET_MARGIN

# Edge case: l2 == 0 (src == tgt) → dist from sun to src directly
crossing_sun = when(l2 == 0)
               .then(sqrt((CENTER - x_src)² + (CENTER - y_src)²) < SUN_RADIUS + PLANET_MARGIN)
               .otherwise(crossing_sun)
```

No Python loop. No `apply`. Runs as a single Polars expression over the entire column.

---

## Data Types & Gotchas

- **Polars `group_by` is unordered by default.** Use `.sort(...)` explicitly before any `first()` aggregation that depends on row order — specifically `mine_across_sim` (needs first row per group to match pandas `"first"` aggregation semantics).
- **`ships_sent` expansion:** Polars `explode` requires a list-typed column. Create with `pl.int_ranges(1, n+1)` expression instead of a Python lambda.
- **`crossing_sun` with `l2 == 0`:** Guard with a `when/then/otherwise` to avoid division by zero.
- **`compute_free_angles` via `map_elements`:** The input struct must include `angle_min`, `angle_max`, `obstacle_list`. Return type is `pl.List(pl.List(pl.Float64))`.
- **`angle_list.str.len() > 0` filter:** In Polars, use `.filter(pl.col("angle_list").list.len() > 0)`.
- **`total_time_cost` window function:** Pandas uses `groupby.transform("sum")`. Polars equivalent is `over("id_src")` window expression.
- **Final output:** Convert Polars result to Python list via `.to_numpy().tolist()` or `.rows()`.

---

## Notebook `45-Polars_comet.ipynb`

Structure:

```
Cell 0   — imports + copy of interpreter/_simulate from 44
Cell 1   — copy of IntervalProcessor + take_action from 44 (reference)
Cell 2   — IntervalProcessorPolars + take_action_polars (new)
Cell 3   — shared test helpers (make_obs, assert_moves_equal)

# Tests 1–5, 7–8 (fast):
Cell N   — ## Test K — <name>
           obs = make_obs_K(...)
           df  = _simulate(obs, ...)
           moves_pd = take_action(df, ...)
           moves_pl = take_action_polars(df, ...)
           assert_moves_equal(moves_pd, moves_pl)
           %%timeit comparison

# Test 6 (slow, full game):
Cell M   — ## Test 6 — Our Agent vs Random  [run last]
           Full game loop, assert identical moves at each step
           Timing comparison
```

`assert_moves_equal`: sorts both move lists by `id_src`, checks `ships_sent` exactly equal, `final_angle` within `1e-6` tolerance.

---

## Submission `46-Polars_comet.py`

Content: `44-Dataframe_comet.py` with:
1. `import polars as pl` added
2. `IntervalProcessorPolars` + `take_action_polars` added after `IntervalProcessor`/`take_action`
3. Agent body changed to:
   ```python
   df = _simulate(obs, step, num_agents, n_steps=NB_STEPS_SIM)
   moves = take_action_polars(df, player_id=player_id, nb_steps_sim=NB_STEPS_SIM)
   ```
4. `return moves` (not `return []`)

---

## Testing Strategy

1. **Unit equality:** For each test 1–8, assert `take_action_polars` returns the same moves as `take_action` (exact on `id_src`/`ships_sent`, float tolerance on angle).
2. **Empty-case:** If `possible_attacks` is empty, both functions return `[]`.
3. **Timing:** `%%timeit` cells show wall-clock improvement. Target: faster than pandas on tests 1–5.
4. **Test 6 last:** Full game loop is slow; run after all others pass.
