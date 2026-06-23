# Design: Restricted Minimax Inner Loop

**Date:** 2026-06-22  
**File:** `135-Polars_mini_max.py` → `136-Polars_mini_max_fast.py`  
**Status:** Approved

---

## Problem

`_04_minimax_search` performs O(N × M) leaf simulations per turn, where:
- N = step-0 candidates (≤ 6: "do nothing" + up to 5 moves)
- M = all safe step-5 moves (can be 20+)

Each leaf simulation calls `_simulate` (a full `copy.deepcopy` + game loop). With N=6, M=20 that is 120 simulations. In practice this exceeds the 1-second turn budget.

---

## Solution

Replace the symmetric O(N × M) search with an asymmetric two-pass strategy:

**Pass 1 — Baseline (full):**  
Simulate "do nothing at step 0" for 5 steps, enumerate all M step-5 candidates, evaluate each leaf.  
Record `best_c5_base` (the single best step-5 move from baseline) and `best_score_base`.

**Pass 2 — Per move (restricted):**  
For each non-None step-0 candidate `c0`:
1. Simulate `c0` for 5 steps.
2. Build `merged5` as today (changed-planets shortcut preserved).
3. Construct a restricted step-5 candidate set of size ≤ K:
   - `None` (do nothing at step 5)
   - All attacks from `src_id` (planet that sent the fleet at step 0)
   - All attacks from `tgt_id` (planet targeted by the fleet at step 0)
   - `best_c5_base` if its source planet is not already covered above
4. Evaluate only these K candidates.

**New complexity:** O(M) + O(N × K), where K ≪ M in practice.

---

## Data-structure change

`step0_candidates` currently holds `list[move | None]` where `move = [id_src, angle, ships_sent]`.

New: `list[tuple[move | None, tgt_id | None]]`

```python
step0_candidates = [(None, None)]
for row in top5_df.select(["id_src", "id", "final_angle", "ships_sent"]).iter_rows():
    src_id, tgt_id, angle, ships = row
    step0_candidates.append(([src_id, angle, ships], tgt_id))
```

`move` passed to `_simulate` / `interpreter` keeps its 3-element shape — `tgt_id` is carried separately in the tuple and never passed into the physics engine.

---

## Baseline pass (replaces the c0=None inner loop)

```python
obs_base5 = _simulate(obs, None, NB_STEPS_5, current_step, num_agents, player_id)
safe5_base = ... # same build as today

best_c5_base: list | None = None
best_score_base: tuple | None = None
base_step5_candidates = [None]
if not safe5_base.is_empty():
    for row in safe5_base.select(["id_src", "final_angle", "ships_sent"]).iter_rows():
        base_step5_candidates.append(list(row))

for c5 in base_step5_candidates:
    obs_leaf = _simulate(obs_base5, c5, NB_STEPS_5, current_step + NB_STEPS_5, ...)
    score = evaluate(obs_leaf, player_id)
    if best_score_base is None or score > best_score_base:
        best_score_base, best_c5_base = score, c5

best_score = best_score_base
best_c0    = None
```

---

## Non-None c0 loop

```python
for c0_move, c0_tgt_id in step0_candidates[1:]:
    obs_c5 = _simulate(obs, c0_move, NB_STEPS_5, current_step, num_agents, player_id)

    # --- changed_ids / merged5 build (unchanged from today) ---
    ...

    src_id = c0_move[0]

    # Restricted step-5 candidates
    restricted: list = [None]
    covered_srcs = {src_id}
    if c0_tgt_id is not None:
        covered_srcs.add(c0_tgt_id)

    if not merged5.is_empty():
        filtered = merged5.filter(pl.col("id_src").is_in(list(covered_srcs)))
        for row in filtered.select(["id_src", "final_angle", "ships_sent"]).iter_rows():
            restricted.append(list(row))

    # best_c5_base as a tie-breaker from a planet not already covered
    if best_c5_base is not None and best_c5_base[0] not in covered_srcs:
        restricted.append(best_c5_base)

    # Evaluate restricted set
    best_score_5: tuple | None = None
    for c5 in restricted:
        obs_leaf = _simulate(obs_c5, c5, NB_STEPS_5, current_step + NB_STEPS_5, ...)
        score = evaluate(obs_leaf, player_id)
        if best_score_5 is None or score > best_score_5:
            best_score_5 = score

    if best_score is None or best_score_5 > best_score:
        best_score, best_c0 = best_score_5, c0_move
```

---

## What stays the same

- `_simulate`, `build_df_s_n`, `_02_get_all_opportunities`, `_03_filter_collision` — untouched
- `changed_ids` shortcut (skips rebuilding `safe5_c` when nothing changed) — preserved
- Comet evasion block — untouched
- Output format (`moves_out`) — untouched

---

## What this does not cover

The `build_df_s_n + _02 + _03` pipeline is still called per non-None c0 when `changed_ids` is non-empty. A follow-up optimization could restrict that computation to only `{src_id, tgt_id}` instead of all changed planets.

---

## Success criteria

- Turn time stays under 1 second in typical game states
- Win rate vs. baseline (`134`/`135`) is equal or better in local backtests
