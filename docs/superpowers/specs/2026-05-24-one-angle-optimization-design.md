# Design: One-Angle Attack Filter (60-Dataframe_one_angle.py)

**Date:** 2026-05-24  
**Base file:** `59-Dataframe_sun_collision.py`  
**Output file:** `60-Dataframe_one_angle.py`

## Problem

The current angle-computation pipeline in `take_action` has three expensive steps:

1. `IntervalProcessor.create_cumulative_obstacles` — Python loop building cumulative blocked interval lists per `(id_src, ships_sent)` track
2. `.apply(compute_free_angles, axis=1)` — row-by-row interval subtraction
3. `interval_to_final_angle(angle_list)` — widest-free-gap midpoint computation

These replace a direct angle with an "optimal" angle through free space, at significant complexity and runtime cost.

## Simplification

A shot is valid if the direct angle from source to target is not blocked by any obstacle planet the fleet would encounter at an earlier step. Use the direct `atan2` angle and check blockage via a vectorized pandas cross-merge.

## Algorithm

### Input

`possible_attacks` — rows filtered to collision=True, crossing_sun=False. Columns include:  
`id_src`, `ships_sent`, `step`, `id` (target), `angle`, `angle_min`, `angle_max`

- `angle` = `atan2(y_tgt - y_src, x_tgt - x_src)` at the simulated step position (already accounts for orbital motion since `_simulate` advances planet positions)
- `[angle_min, angle_max]` = cone half-width via law-of-cosines; may wrap around 2π if `angle_min > angle_max`

### Step 1 — Build obstacle pairs

Self-merge `possible_attacks` on `["id_src", "ships_sent"]`, using suffixes to distinguish target vs obstacle sides. Filter:
- `step_obs < step` — obstacle is encountered before the target
- `id_obs != id` — obstacle is a different planet than the target

### Step 2 — Vectorized block check

For each (target, obstacle) pair, check if the target's `angle` falls inside the obstacle's cone:

```
if angle_min_obs > angle_max_obs:  # cone wraps around 2π
    is_blocked = (angle >= angle_min_obs) OR (angle <= angle_max_obs)
else:
    is_blocked = (angle >= angle_min_obs) AND (angle <= angle_max_obs)
```

Implemented as a single `np.where` expression on arrays — no Python loops.

### Step 3 — Anti-join

Collect all `(id_src, ships_sent, step, id)` tuples where `is_blocked` is True (`drop_duplicates`). Left-join back to `possible_attacks` with `indicator=True`; keep only `left_only` rows (never appeared as blocked).

Rows with no obstacles at all (earliest step in their group) are never added to `pairs`, never appear in `blocked`, and survive naturally.

### Step 4 — Assign final angle

```python
attacks_with_angle["final_angle"] = attacks_with_angle["angle"]
```

No interval list, no gap search, no midpoint computation.

## Changes vs 59-Dataframe_sun_collision.py

| Component | Change |
|---|---|
| `IntervalProcessor` class (5 methods) | Removed entirely |
| `create_cumulative_obstacles` call | Removed |
| `.apply(compute_free_angles, axis=1)` | Removed |
| `interval_to_final_angle` call in `attacks` block | Removed |
| Cross-merge block check | Added (~15 lines) |
| `final_angle = angle` on `attacks_with_angle` | Added |
| Comets handling | Unchanged (already uses `angle` directly) |
| Scoring, groupby-first, agent loop | Unchanged |

## Trade-offs

**Gains:**
- Eliminates all Python-level loops in the angle pipeline
- No cumulative state to maintain across steps
- Code is ~80 lines shorter and straightforward to reason about

**Losses:**
- Always shoots at the direct center-to-center angle; loses the ability to find a slightly offset angle that avoids an obstacle while still hitting the target
- A valid shot that exists at a non-direct angle will now be dropped rather than routed around the obstacle

At the current game scale (NB_STEPS_SIM=10, ~20 planets), the cross-merge produces a small intermediate frame and runs fast. If scale grows significantly, a sort-based scan would be more efficient.
