# Performance Analysis: 44 vs 41

## Summary of Differences

| Area | 41-Dataframe_5nearest_aiming_ok | 44-Dataframe_comet |
|---|---|---|
| Source planet selection | Pre-filter `owner == player_id` before groupby | Group ALL planets, check `row_count == is_mine` |
| Source planet guard | `row_count >= nb_steps_sim + 1 AND ships_min > 0` | No ships_min guard, no minimum row_count |
| Comet sources | Never included | Included when comet is owned throughout sim |
| Comet attack block | Commented out | Active — separate move generation for far comets |
| Score formula | `(total_time_cost - time_cost) * production` | `(total_time_cost - time_cost - step_diff) * production` |
| `planet_id_top_5_id_src` sort | One sort before groupby | Extra sort after first groupby |

---

## Why 44 is Slower

### 1. Full-df groupby instead of pre-filtered groupby (minor)

In 41, `mine_across_sim` starts with `.query("owner == @player_id")`, cutting the dataframe down to only rows belonging to the player before the `.groupby().agg()`. In 44 that filter is commented out — the `.assign(is_mine=...)` column is computed across every row and the full groupby runs on the entire `df`.

Cost: proportional to `(num_planets + num_comets) × 21` rows instead of `(owned_planets) × 21`.

### 2. Comet sources explode `expanded_mine` and the cross merge (dominant)

`expanded_mine` generates one row per `ships_sent` value per source planet, where `ships_sent` ranges from `1` to `ships_min + production * NB_STEPS_SIM`.

In 41, comets are never sources (the `owner == player_id` pre-filter combined with `ships_min > 0` guard keeps them out). In 44, any comet owned by the player for every row of the simulation is included as a source. By the time comets expire they can have accumulated dozens of ships — e.g. a comet with `ships_min = 60` and `production = 1` generates `range(1, 81)` = **80 ships_sent rows**.

The next step is a cross merge:

```
|df_src_tgt| = |expanded_mine| × |df|
```

`|df| = (num_planets + num_comets) × 21`. Every extra ships_sent row in `expanded_mine` multiplies the whole dataframe. Adding even one comet source with 60 ships adds `60 × |df|` rows — easily 10–100× more rows than a typical static planet adds.

### 3. `create_cumulative_obstacles` — quadratic in (sources × ship counts)

`create_cumulative_obstacles` has this inner structure:

```python
for id_src, ship in unique_combinations:          # O(sources × ship_range)
    for step in range(min_step, max_step + 1):    # O(max_step) — grows with game step
        ...
```

`unique_combinations` is the Cartesian product of source planets × their full ships_sent range. Comet sources with large ship counts directly multiply the outer loop. As the game progresses, `max_step` (an absolute step number) also grows, so this function gets slower each turn regardless of agent version — but 44 makes it worse by adding comet entries to `unique_combinations`.

### 4. Missing `ships_min > 0` guard in `expanded_mine`

41 filters `ships_min > 0` before exploding. 44 does not. A planet with `ships_min = 0` generates `range(1, production * 20 + 1)` — up to 20 extra ships_sent rows — and participates in the full pipeline. Minor but cumulative.

### 5. Comet attack block (minor)

The new comet attack block (lines 641–655) adds an extra `.query()`, `.sort_values()`, `.groupby()`, and `.first()` on `attacks_with_angle`. Small cost relative to #2 and #3.

---

## Root Cause in One Line

`expanded_mine` now includes comet sources with large ship counts, causing the cross merge (`df_src_tgt`) and the `create_cumulative_obstacles` loop to grow proportionally — likely 5–20× more rows depending on how many ships the active comets hold.

---

## Suggested Fix

Filter comets out of `mine_across_sim` before the explode, handle them separately as 44 already tries to do in the comet attack block:

```python
mine_across_sim = (
    df
    .query("owner == @player_id and nature_src != 'comet'")  # keep comets separate
    .groupby("id")
    .agg(...)
    .query("row_count >= @nb_steps_sim + 1 and ships_min > 0")
    ...
)
```

Then compute comet moves independently (as in the existing comet block) using only the first row of each comet's sim data, without entering the full cross-merge pipeline.
