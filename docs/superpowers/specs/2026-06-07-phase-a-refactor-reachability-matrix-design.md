# Phase A Refactor: Reachability Matrix & Opponent Threat Model

**Date:** 2026-06-07
**File:** `102-Simulate10Next.py`
**Scope:** Refactor `_02_get_all_opportunities` to support a second call path that computes a reachability matrix (any-planet → any-planet, fixed ship counts) for use in scoring and opponent threat assessment.

---

## Problem

`_02_get_all_opportunities` is monolithic: it builds the source base (mine-only), runs Phase A (cross-join + sun-crossing filter), expands ships_sent via a per-row formula, and runs Phase B + swept-pair + angle geometry — all in one function. Adding a second call path (all planets, fixed ship counts) requires either duplicating ~80 lines or threading conditionals through the existing body.

---

## Design

### New static methods on `StrategyPipeline`

#### `_sun_crossing_filter(coarse: pd.DataFrame) -> pd.DataFrame`

Extracts the existing vectorised sun-crossing block (lines ~373–400) into a standalone helper. Takes a `coarse` DataFrame that already has `x_src`, `y_src`, `x`, `y` columns. Returns `coarse` with rows that cross the sun removed. Called by both pre functions.

#### `_02_pre_mine(df_s: pd.DataFrame, player_id: int) -> pd.DataFrame`

Phase A for mine-only sources:
1. Build `mine_base`: group `df_s` by `id`, aggregate `step_src / x_src / y_src / radius_src / ships_min / production_src / nature_src / owner_src`, filter to planets fully owned by `player_id` across all sim steps.
2. Cross-join `mine_base × df_s`, filter `step > step_src` and `id != id_src`, join `planet_disp`, compute `dist_tgt_src` and `step_diff`.
3. Call `_sun_crossing_filter`.
4. Assign `ships_sent` column: `list(range(1, ships_min + production_src * NB_STEPS_SIM + 1))` per row (formula-based list).

Returns `coarse` with a `ships_sent` list-per-row column.

#### `_02_pre_all(df_s: pd.DataFrame, ships_list: list) -> pd.DataFrame`

Phase A for all-planet sources:
1. Build `all_base`: same aggregation as `mine_base` but **no ownership filter** — every planet is a valid source. `ships_min` is still aggregated (kept for column consistency; not used for capping).
2. Cross-join `all_base × df_s`, same distance and step filters.
3. Call `_sun_crossing_filter`.
4. Assign `ships_sent = [ships_list] * len(coarse)` (fixed list, same for every row).

Returns `coarse` with the same shape as `_02_pre_mine` output.

#### `_02_get_all_opportunities(coarse: pd.DataFrame, df_s: pd.DataFrame, planet_disp: pd.DataFrame) -> pd.DataFrame`

No longer builds the source base or performs Phase A. Starts at `.explode("ships_sent")`:
1. Explode `ships_sent` → cast to `int64`.
2. Phase B fleet-speed filter (unchanged).
3. Vectorised swept-pair collision (unchanged).
4. Angle geometry (unchanged).

Returns `pa` DataFrame. **No `player_id` parameter, no `ships_list` parameter** — both concerns are handled upstream by the pre functions.

---

### Updated call site in `agent()`

```python
coarse_mine = StrategyPipeline._02_pre_mine(df_s, 0)
coarse_all  = StrategyPipeline._02_pre_all(df_s, [4, 16, 64, 256])
pa          = StrategyPipeline._02_get_all_opportunities(coarse_mine, df_s, planet_disp)
pa_reach    = StrategyPipeline._02_get_all_opportunities(coarse_all,  df_s, planet_disp)
safe        = StrategyPipeline._03_filter_collision(pa)
reach       = StrategyPipeline._03_filter_collision(pa_reach)
moves       = StrategyPipeline._04_score_and_decide(safe, reach, 0)
```

### `_03_filter_collision` — unchanged signature

Takes a `pa` DataFrame; works identically for both call paths.

### `_04_score_and_decide(attacks_with_angle, reach_matrix, player_id)` — gains second arg

`reach_matrix` is the `_03`-filtered output of `pa_reach`. Existing scoring logic operates on `attacks_with_angle` as before; `reach_matrix` is available for threat assessment and scoring enhancements.

---

## Data Flow

```
df_s  ──► _02_pre_mine(player_id=0) ──► coarse_mine (ships_sent=formula list)
                                              │
                                              ▼
                              _02_get_all_opportunities ──► pa ──► _03 ──► safe
                                              ▲
df_s  ──► _02_pre_all(ships_list=[4,16,64,256]) ──► coarse_all (ships_sent=fixed list)
                                              │
                                              ▼
                              _02_get_all_opportunities ──► pa_reach ──► _03 ──► reach
                                                                                   │
safe ─────────────────────────────────────────────────────────────────────────────►│
                                                                                   ▼
                                                              _04_score_and_decide ──► moves
```

---

## Invariants

- `coarse_mine` and `coarse_all` have identical column schemas (including `ships_sent` as a list column). `_02_get_all_opportunities` makes no assumptions about which pre function produced its input.
- `_sun_crossing_filter` is pure: no side effects, same filter logic for both paths.
- Planet/fleet IDs are already remapped by `remap_player_ids` before this pipeline runs; no ID-mapping logic belongs here.
- `_03_filter_collision` signature is unchanged; it is safe to call on either `pa` or `pa_reach`.
