# 126-precompute Redesign

**Date:** 2026-06-17  
**File:** `126-precompute.py`  
**Output:** `126-precompute/{episode_id}/`

## Motivation

Replace the monolithic `df_s` table with a cleaner normalized schema that separates:
- static/spatial planet info (deterministic, physics-only)
- dynamic planet state (ships/owner, depends on game history)
- reachability with actual ship counts (not just power-of-2 buckets)

---

## Output Schema (5 parquet files per episode)

### `planete.parquet` — spatial info per sim step
| Column | Type | Notes |
|---|---|---|
| `id` | Int64 | planet id |
| `step` | Int64 | simulation step |
| `x` | Float64 | position at this step |
| `y` | Float64 | position at this step |
| `production` | Int64 | constant per planet |
| `nature` | Utf8 | `fix`, `moving`, `comet` |

- One row per `(id, step)`, deduped across obs steps (positions are deterministic).
- Source: `df_s_full.select(["id", "step", "x", "y", "production", "nature"])` from each sim run, then `unique(subset=["id", "step"])`.

---

### `planete_step.parquet` — dynamic state per obs step × future step
| Column | Type | Notes |
|---|---|---|
| `id` | Int64 | planet id |
| `step` | Int64 | obs game_step (when observation was recorded) |
| `future_step` | Int64 | simulated future step, `future_step ∈ [step, step+20]` |
| `ships` | Int64 | ships at `future_step` per fresh sim from `step` |
| `owner` | Int64 | owner at `future_step` per fresh sim from `step` |

- 21 rows per `(id, obs_step)` (future_step = step, step+1, … step+20).
- Each obs game_step restarts the simulation independently (`deepcopy(obs)` in `_01_get_obs_dataframe`).
- Simulation uses no actions (`[[], []]`), so ships/owner reflect passive evolution.
- Source: tag each `df_s_full` slice with `game_step`, keep all future_step ≥ game_step.

---

### `actions_to_copy.parquet` — winner's moves with resolved targets
| Column | Type | Notes |
|---|---|---|
| `step` | Int64 | game step (renamed from `game_step`) |
| `id_src` | Int64 | source planet |
| `angle` | Float64 | launch angle |
| `ships_sent` | Int64 | ships sent |
| `id_tgt` | Int64 | resolved target planet (renamed from `id`, may be null) |

- Same logic as current `actions.parquet`, column renames only.

---

### `reachable_base_2.parquet` — reachability with power-of-2 fleet sizes
| Column | Type | Notes |
|---|---|---|
| `id_src` | Int64 | source planet |
| `step_src` | Int64 | source obs step |
| `id_tgt` | Int64 | target planet (renamed from `id`) |
| `step_tgt` | Int64 | arrival step (renamed from `step`) |
| `ships_sent` | Int64 | fleet size ∈ {1,2,4,8,16,32,64,128,256,512,1024} |

- Same computation as current `reach.parquet`, collision-filtered. Column renames only.

---

### `reachable_max_ships.parquet` — reachability sending all ships
| Column | Type | Notes |
|---|---|---|
| `id_src` | Int64 | source planet |
| `step_src` | Int64 | source obs step |
| `angle` | Float64 | optimal launch angle |
| `ships_sent` | Int64 | actual ships at source at `step_src` |
| `id_tgt` | Int64 | target planet |
| `step_tgt` | Int64 | earliest arrival step |

- One row per `(id_src, step_src, id_tgt)` — earliest `step_tgt` kept.
- `ships_sent` = `ships` at `(id_src, step_src)` in df_s (first sim step = obs step).
- Same coarse filter + swept-pair collision filter as `reachable_base_2`, but no explode over ship buckets.
- New helper `_get_reach_max_ships(df_s, planet_disp)` — copy of `_02_get_reach` replacing `.with_columns(pl.lit(ships_list)).explode()` with `.with_columns(pl.col("ships_at_src").alias("ships_sent"))` where `ships_at_src = pl.first("ships")` in the source aggregation.

---

## Implementation Changes

### `_process_episode` loop

```
# Replace:
df_s_parts.append(df_s_full.select(DF_S_COLS))

# With:
planete_parts.append(df_s_full.select(["id", "step", "x", "y", "production", "nature"]))

planete_step_parts.append(
    df_s_full.select(["id", "step", "ships", "owner"])
    .filter(pl.col("step") >= game_step)
    .with_columns(pl.lit(game_step).cast(pl.Int64).alias("obs"))
    .rename({"obs": "step", "step": "future_step"})
    .select(["id", "step", "future_step", "ships", "owner"])
)

# Replace reach collection:
reach_b2_parts.append(filtered.select(REACH_B2_COLS))   # with id→id_tgt, step→step_tgt

raw_max = _get_reach_max_ships(df_s_full, planet_disp).collect()
filtered_max = SP._03_filter_collision(raw_max.lazy()).collect()
reach_max_parts.append(
    filtered_max
    .rename({"id": "id_tgt", "step": "step_tgt"})
    .sort("step_tgt")
    .group_by(["id_src", "step_src", "id_tgt"], maintain_order=True).first()
    .select(REACH_MAX_COLS)
)
```

### Assembly

```
planete_out     = pl.concat(planete_parts).unique(subset=["id", "step"])
planete_step_out = pl.concat(planete_step_parts)  (unique by ["id","step","future_step"])
reachable_b2_out = pl.concat(reach_b2_parts).unique(keep="first")
reachable_max_out = pl.concat(reach_max_parts)
    .sort("step_tgt").group_by(["id_src","step_src","id_tgt"]).first()
act_out         = (renamed columns)
```

### New helper `_get_reach_max_ships`

Copy of `SP._02_get_reach` with:
1. `all_base_lf` adds `pl.first("ships").alias("ships_at_src")` (not `pl.min`)
2. Replace `.with_columns(pl.lit(ships_list)).explode()` → `.with_columns(pl.col("ships_at_src").alias("ships_sent"))`
3. No explode step — one row per `(id_src, step_src)` in the cross join.

---

## File layout

```
126-precompute/
  {episode_id}/
    planete.parquet
    planete_step.parquet
    actions_to_copy.parquet
    reachable_base_2.parquet
    reachable_max_ships.parquet
```

Skip-if-exists check updated to match all 5 filenames.
