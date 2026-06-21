# 134-Polars_ships_src: All-In Ships Design

**Date:** 2026-06-21  
**Base:** `133-Polars_speed.py`  
**Hypothesis:** Removing the ships_sent expansion (O(ships_min) rows/pair → O(1)) saves enough compute to be a net win, even though it locks in the all-in strategy.

---

## What Changes

### 1. `mine_base_lf` aggregation — add `ships_src`

```python
pl.first("ships").alias("ships_src"),   # current ships at source (step 0)
```

`ships_src` is the ships count at the source planet at the current game step (step 0 of the sim window). It can exceed `ships_min` when ships decrease later in the window.

### 2. Ships_sent expansion → single assignment

Replace the 5-line `int_ranges + explode` block in `_02_get_all_opportunities`:

```python
# BEFORE (O(ships_min) rows per source-target pair)
expanded_lf = (
    coarse_lf
    .with_columns(pl.int_ranges(1, pl.col("ships_min") + ...).alias("ships_sent"))
    .explode("ships_sent")
)

# AFTER (1 row per source-target pair)
expanded_lf = coarse_lf.with_columns(pl.col("ships_src").alias("ships_sent"))
```

### 3. `_03_filter_collision` — unchanged

The blocking self-join keys on `["id_src", "ships_sent"]`. With `ships_sent` fixed per `id_src`, it degenerates to per-source blocking. Logic is still correct.

### 4. `_04_score_and_decide` — replace Conqueror/Supplier/Conqueror2 with 72-style unified scoring

Drop all three branches. Replace with the approach from `72-Polars_scoring.py`:

```python
attacks = (
    attacks_with_angle
    .with_columns(
        pl.when(pl.col("owner") == -1)
        .then(pl.col("ships"))
        .otherwise(pl.col("ships") + pl.col("production"))
        .alias("ships_needed")
    )
    .filter(pl.col("ships_needed") < pl.col("ships_sent"))   # can conquer all-in?
    .sort(["step", "ships_sent"])
    .group_by(["id_src", "id"], maintain_order=True).first()
    .join(top5_ids, on=["id_src", "id"], how="left")
    .with_columns(pl.col("is_top5").fill_null(False))
    .with_columns((pl.col("ships_needed") / pl.col("production_src")).alias("time_cost"))
    .with_columns(pl.col("time_cost").sum().over("id_src").alias("total_time_cost"))
    .with_columns(
        (
            (pl.col("total_time_cost") - pl.col("time_cost") - pl.col("step_diff"))
            * pl.col("production")
            - pl.when(~pl.col("is_top5")).then(pl.lit(100.0)).otherwise(pl.lit(0.0))
            - pl.when(pl.col("owner") == player_id).then(pl.lit(100.0)).otherwise(pl.lit(0.0))
        ).alias("score")
    )
    .filter(pl.col("score") > 0)
    .sort("score", descending=True)
    .group_by("id_src", maintain_order=True).first()
)
```

**Why drop the upper-bound range filter** (`ships_sent <= ships_needed + production_src + 1`): with fixed `ships_sent = ships_src`, this would block valid all-in attacks whenever we have many more ships than needed. We always send all ships, so only the lower bound matters.

**Why drop the final `ships_sent <= ships_min` guard**: `ships_src >= ships_min` by definition, so this guard would incorrectly reject valid moves.

### 5. Comet evasion — simplify

Remove the `ships_sent <= ships_min` sub-filter in comet evasion. Always flee with `ships_src` ships (all-in evasion).

---

## What Does NOT Change

- `interpreter`, `PhysicsEngine`, `GameCache`, `GameCache.build_df_s` — all untouched
- `_02_get_all_opportunities` Phase A (coarse filter) and Phase B (swept-pair collision, angle cone) — untouched
- `_03_filter_collision` — untouched
- `agent()` entry point — untouched

---

## Trade-offs

| | 133 (current) | 134 (all-in) |
|---|---|---|
| Rows per src-target pair | O(ships_min) | 1 |
| Ships sent | Tuned to "just enough" | Always all-in |
| Supplier path | Yes (partial reinforcement) | Removed |
| Conqueror2 (2-planet attack) | Yes | Removed |
| Scoring | Production / (time_cost + step_diff) | 72-style unified score |

**Risk:** All-in commits all ships every turn — leaves source planet exposed. Supplier/Conqueror2 coordination is lost.  
**Upside:** Dramatically smaller Polars frame, faster per-turn, simpler logic to reason about.
