# Design: Board Cache — Incremental Ship-State Recomputation

**Date:** 2026-06-22
**File:** `138-BoardCache.py`
**Predecessor:** `137-Polars_minimax_fast2.py`
**Status:** Approved

---

## Problem

`137`'s `_04_minimax_search` calls `_simulate()` up to N + N×K times per turn (N ≤ 6 step-0
candidates, K ≤ 5 restricted step-5 candidates). Each `_simulate()` does:

1. `copy.deepcopy(obs)` — copies full game state
2. A 5-step physics loop touching every planet and fleet

With N=6, K=5 that is 36 deepcopy+physics calls per turn, easily blowing the 1-second budget.

---

## Solution

Replace deepcopy + physics loops with a `Board` class that:

- Maintains **five Polars DataFrames** as the authoritative game state
- Recomputes ship states **incrementally**: only the ≤ 10 planet-steps affected by a
  simulated fleet action, not the full 150-row table
- Evaluates the c5 inner loop using **pure-Python dict arithmetic** — zero Polars allocation
  in the hot path

Reset strategy: **immutable base** — `df_planete_ships` is never mutated during minimax.
Per-candidate copies are built by filtering + recomputing only dirty rows.

---

## Data Model

### Five tables stored on `Board`

```
df_planete_nature : pl.DataFrame  shape (P, 3)
    columns: id, production, nature
    # Static per planet. Updated only when comets expire.

df_planete_pos    : pl.DataFrame  shape (P × W, 4)   W = 11 steps
    columns: id, step, x, y
    # Sliding window of planet positions. One new far-end row added per turn.

df_fleet          : pl.DataFrame  shape (F, 5)
    columns: id, owner, ships, id_tgt, step_tgt
    # Real in-flight fleets from obs. id_tgt/step_tgt = None if arrival > window.

df_planete_ships  : pl.DataFrame  shape (P × W, 5)
    columns: id, step, ships, owner, recompute
    # Base ship state from df_fleet only. Never mutated during minimax.
    # recompute is always False in the stored base.

df_fleet_sim      : pl.DataFrame  shape (0–2, 6)
    columns: id_src, step_src, ships_sent, owner, id_tgt, step_tgt
    # Hypothetical fleet actions for current minimax search.
    # Cleared to empty at the start of each turn's minimax.
    # Holds at most 2 rows at any time (c0 + c5).
```

### Derived (not stored, computed per c0 candidate)

```
ships_horizon : dict[pid -> (ships, owner, production)]
    # Extracted from df_planete_ships at step = current_step + 10.
    # Used by _evaluate_dict() inside the c5 inner loop.
```

---

## Per-Turn Update (Steps 1–3)

Called once at the start of each agent turn, before minimax.

### Step 1 — Slide `df_planete_pos`

```
advance(obs, step):
    drop rows where step < current_step          # evict just-passed step
    for pid in df_planete_nature.id:
        append (pid, current_step+10, x, y)      # _planet_pos_analytical()
    drop rows for expired comet pids
    drop expired pids from df_planete_nature
```

Cost: P inserts + P deletes on ~165 rows. Negligible.

### Step 2 — Sync `df_fleet`

```
    drop rows where id not in obs.fleets          # fleet landed or expired
    for fleet in obs.fleets not yet in df_fleet:
        arrival = _compute_fleet_arrival(fleet)   # swept-pair, same as GameCache
        append (fleet.id, fleet.owner, fleet.ships, arrival.pid, arrival.step)
        # arrival = None → id_tgt=None, step_tgt=None
```

Cost: O(F × W × P) swept-pair checks — same as current `GameCache`.

### Step 3 — Recompute `df_planete_ships` (base)

Full recompute for this turn, using Python dicts, then materialise as Polars DataFrame:

```python
ship_state  = {pid: planet.ships for planet in obs.planets}
owner_state = {pid: planet.owner for planet in obs.planets}
production  = {pid: row.production for row in df_planete_nature}

rows = []
for k in range(11):
    game_step = current_step + k
    for pid in ship_state:
        rows.append((pid, game_step, ship_state[pid], owner_state[pid], False))
    if k == 10:
        break
    # production
    for pid, owner in owner_state.items():
        if owner != -1:
            ship_state[pid] += production[pid]
    # arrivals from df_fleet at game_step
    for fleet_row in df_fleet.filter(step_tgt == game_step):
        resolve_combat(fleet_row, ship_state, owner_state)   # same logic as 1_29_3.py

df_planete_ships = pl.DataFrame(rows, schema=["id","step","ships","owner","recompute"])
```

This runs once per turn. It replaces `build_df_s()` from `GameCache` and
`build_df_s_n()` from the standalone helper.

---

## Minimax Search (Steps 4–5)

### Helper: `_recompute_from_sim(df_ships_base, sim_row) → pl.DataFrame`

Used **once per c0 candidate** to build the post-c0 ship state for `_02_get_all_opportunities`.

```
dirty pids and min dirty step:
    src: step >= sim_row.step_src + 1
    tgt: step >= sim_row.step_tgt   (if step_tgt is not None)
    min_dirty = min(step_src+1, step_tgt)

keep_rows = df_ships_base.filter(
    ~pl.col("id").is_in([src, tgt]) | (pl.col("step") < min_dirty)
)

re-run Step 3 loop for dirty pids only, from min_dirty to current_step+10,
seeding ship/owner from keep_rows at step = min_dirty - 1

return pl.concat([keep_rows, new_rows]).sort("step")
```

Cost: at most 2 planets × 5 steps = **10 rows recomputed** (vs. deepcopy + full 5-step physics).

### Helper: `extract_horizon_dict(df_ships, horizon_step) → dict[pid, tuple]`

```python
rows = df_ships.filter(pl.col("step") == horizon_step)
return {row["id"]: (row["ships"], row["owner"]) for row in rows.iter_rows(named=True)}
```

### Helper: `_apply_sim_fleet(ships_horizon, sim_row) → dict`

Used inside the **c5 inner loop** — no Polars involved. If `sim_row is None` (do-nothing),
returns `ships_horizon` unchanged.

```python
if sim_row is None:
    return ships_horizon
d = dict(ships_horizon)          # shallow copy, ~15 entries
# ships leave source (if source is tracked at horizon)
if sim_row.id_src in d:
    ships, owner = d[sim_row.id_src]
    d[sim_row.id_src] = (ships - sim_row.ships_sent, owner)
# combat at target — only if fleet arrives within the window
if sim_row.step_tgt is not None and sim_row.id_tgt in d:
    resolve_combat_dict(d, sim_row)
return d
```

### Helper: `_evaluate_dict(ships_horizon, player_id) → tuple`

Pure-Python version of `evaluate()`, operating on the horizon dict.

```python
my_prod  = sum(v.production for v in d.values() if v.owner == player_id)
my_ships = sum(v.ships      for v in d.values() if v.owner == player_id)
opp_prod = max(...)
opp_ships = max(...)
return (my_prod - opp_prod, my_ships - opp_ships)
```

### Full minimax loop

```
# ── Baseline ─────────────────────────────────────────────────────────────
df_ships_base    = board.df_planete_ships          # from Step 3
pa_lf_base       = _02_get_all_opportunities(df_ships_base, df_planete_pos, ...)
safe_base        = _03_filter_collision(pa_lf_base).collect()

c5_candidates_base = [None] + list(safe_base rows as [id_src, angle, ships])

ships_horizon_base = extract_horizon_dict(df_ships_base, current_step + 10)
top3_scored = []
for c5 in c5_candidates_base:
    d = _apply_sim_fleet(ships_horizon_base, c5) if c5 else ships_horizon_base
    score = _evaluate_dict(d, player_id)
    top3_scored.append((score, c5))
top3_scored.sort(reverse=True)
top3_c5_base = [c5 for _, c5 in top3_scored[:3]]
best_score = top3_scored[0][0]
best_c0 = None

# ── Per c0 candidate ─────────────────────────────────────────────────────
for c0_move, c0_tgt_id in step0_candidates[1:]:

    df_ships_c0 = _recompute_from_sim(df_ships_base, c0_move)
                  # Polars: filter + recompute 10 rows + concat

    # changed-ids shortcut (preserved from 137)
    changed_ids = {pid for pid, owner, ships in iter_c0_planets()
                   if (owner, ships) != base_planets5[pid]}
    if changed_ids:
        pa_lf_c0 = _02_get_all_opportunities(df_ships_c0, ...)
        safe_c0  = _03_filter_collision(pa_lf_c0).collect()
    else:
        safe_c0  = safe_base

    # restricted c5 set (same logic as 137)
    covered_srcs = {c0_move[0]}
    if c0_tgt_id: covered_srcs.add(c0_tgt_id)
    restricted = [None]
    restricted += attacks_from_or_to(covered_srcs, safe_c0)
    for c5 in top3_c5_base:
        if c5 is not None and c5[0] not in covered_srcs:
            restricted.append(c5); break

    # evaluate restricted set — pure Python, zero Polars
    ships_horizon_c0 = extract_horizon_dict(df_ships_c0, current_step + 10)
    best_score_c0 = None
    for c5 in restricted:
        d = _apply_sim_fleet(ships_horizon_c0, c5) if c5 else ships_horizon_c0
        score = _evaluate_dict(d, player_id)
        if best_score_c0 is None or score > best_score_c0:
            best_score_c0 = score

    if best_score_c0 > best_score:
        best_score = best_score_c0
        best_c0 = c0_move
```

---

## Complexity Comparison

| Operation | 137 | 138 |
|---|---|---|
| `copy.deepcopy` calls per turn | N + N×K | 0 |
| Full 5-step physics loops | N + N×K | 0 |
| `_02`+`_03` Polars calls | 1 + N (changed only) | 1 + N (changed only) |
| Dirty-row recomputes (Polars) | 0 | N × ≤10 rows |
| Dict evaluate calls (c5 loop) | 0 | 1 + N×K (P≈15 planets each) |

N ≤ 6, K ≤ 5, P ≈ 15. The 36 deepcopy+physics calls become 30 dict copies of 15 entries.

---

## What Stays the Same

- `_02_get_all_opportunities` — untouched (still receives a `df_s`-shaped DataFrame)
- `_03_filter_collision` — untouched
- `_planet_pos_analytical` / `_compute_fleet_arrival` — copied verbatim from `GameCache`
- Combat resolution formula — same as `1_29_3.py` interpreter, extracted into a shared helper
- Comet evasion block — untouched
- Output format (`moves_out`) — untouched
- `angular_velocity`, `initial_planets` handling — unchanged

---

## What Changes

- `GameCache` is replaced by `Board`
- `_simulate()` and `build_df_s_n()` are removed — their work is absorbed into
  `_recompute_from_sim()` + `_apply_sim_fleet()`
- `evaluate()` gains a `_evaluate_dict()` twin for the hot path
- `agent()` calls `board.advance()` then `board.build_base_ships()` instead of `CACHE.advance()` + `CACHE.build_df_s()`

---

## File Structure

`138-BoardCache.py` is a self-contained file mirroring the structure of `137`:

```
GameConfig          — unchanged
PhysicsEngine       — unchanged
Board               — replaces GameCache
  __init__          — initialise all five tables
  advance           — step 1+2: slide pos window, sync fleets
  build_base_ships  — step 3: recompute df_planete_ships
  _recompute_from_sim  — per-c0 dirty-row recompute
  extract_horizon_dict — step+10 slice → Python dict
  _apply_sim_fleet     — per-c5 dict update (handles sim_row=None)
  _evaluate_dict       — pure-Python evaluate
StrategyPipeline    — _02, _03 unchanged; _04 rewritten
agent               — entry point, uses Board
__main__            — smoke test
```

---

## Success Criteria

- Turn time < 1 second in typical game states (P≈15, N≈6, K≈5)
- Win rate vs. `137` equal or better in local backtests
- Smoke test (`__main__`) passes: `agent()` returns a valid move list in < 5 s
