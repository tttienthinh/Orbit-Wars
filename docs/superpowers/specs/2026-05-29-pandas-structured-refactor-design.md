# Design: Functional Core / Imperative Shell Refactor of 80-70-Pandas_structured.py

**Date:** 2026-05-29  
**Source file:** `80-70-Pandas_structured.py`  
**Target file:** `81-80-Pandas_fc_ish.py` (or successor)

---

## Objective

Refactor the monolithic `take_action` + `_simulate` pipeline into a strict "Functional Core, Imperative Shell" architecture. The code is reorganized for testability and clarity — no math or logic changes.

---

## Constraints

1. **Single file** — entire output is one `.py` script.
2. **Untouched interpreter** — `Planet`, `Fleet` namedtuples and `interpreter()` function are copied verbatim.
3. **No logic loss** — all Pandas merges, vectorized numpy ops, and physics calculations preserved exactly.

---

## Architecture

### 1. `GameConfig` (constants class)

All module-level constants become class attributes:

```python
class GameConfig:
    CENTER = 50.0
    SUN_RADIUS = 10.0
    ROTATION_RADIUS_LIMIT = 50.0
    MAX_SPEED = 6.0
    NB_STEPS_SIM = 10
    PLANET_MARGIN = 0.1
    PLANET_MOVEMENT_SLACK = 3.0
```

### 2. `PhysicsEngine` (functional core — geometry)

Static methods moved verbatim from module scope:

- `distance(p1, p2) -> float`
- `point_to_segment_distance(p, v, w) -> float`
- `swept_pair_hit(A, B, P0, P1, r) -> bool`
- `_fleet_speed(ships) -> float`

### 3. `StrategyPipeline` (functional core — data pipeline)

Four `@staticmethod`s forming a linear pipeline:

#### `_01_get_obs_dataframe(obs, step: int, num_agents: int) -> tuple[pd.DataFrame, pd.DataFrame]`

- Inlines `_simulate`: runs the interpreter for `NB_STEPS_SIM` steps, collecting planet state rows.
- Computes `planet_disp` DataFrame (`[id, step, planet_disp]`) from the shift-merge pattern in the original `take_action` lines 318–336.
- Returns `(df_s, planet_disp)`.

#### `_02_get_all_opportunities(df_s, planet_disp, player_id) -> pd.DataFrame`

Covers original `take_action` lines 338–538:

1. `mine_base` — groupby aggregation to find fully-owned source planets.
2. Coarse cross-join with `df_s` target rows, distance bound filter, sun-crossing filter.
3. `ships_sent` expansion via `explode`.
4. Phase B: fleet-speed-specific distance filter + `prev_pos` join.
5. Swept-pair collision vectorization → `pa` DataFrame with `collision`, `t1`, `t2` columns.
6. Angle geometry computations (`angle_t1`, `angle_t2`, `angle_radius_*`, `angle_min`, `angle_max`, `angle`).

Returns `pa: pd.DataFrame`.

#### `_03_filter_collision(pa) -> pd.DataFrame`

Covers original `take_action` lines 541–581:

- Blocking self-join on `[id_src, ships_sent]`.
- Wrapping-cone check (`angle_min_obs > angle_max_obs`).
- Anti-join to drop blocked trajectories.

Returns `attacks_with_angle: pd.DataFrame`.

#### `_04_score_and_decide(attacks_with_angle, player_id) -> list`

Covers original `take_action` lines 583–658:

- Comet evasion: if a comet source is far from center (>45 units offset), emit its move immediately.
- `planet_id_top_5`: top-5 reachable targets per source.
- `attacks_joined`: filter `owner != player_id`, compute `ships_needed`, apply count bounds.
- Scoring: `time_cost`, `total_time_cost`, `score = (total_time_cost - time_cost - step_diff) * production`.
- Final groupby per `id_src`, pick best score, filter `ships_sent <= ships_min`.

Returns `list[[id_src, final_angle, ships_sent]]`.

---

### 4. Imperative Shell

```python
step = 0
num_agents = None
player_id = None

def agent(obs):
    global step, num_agents, player_id

    if num_agents is None:
        initial = obs.initial_planets if hasattr(obs, "initial_planets") else obs["initial_planets"]
        owners = {p[1] for p in initial if p[1] != -1}
        num_agents = 4 if len(owners) > 2 else 2
    if player_id is None:
        player_id = obs.get("player", 0) if isinstance(obs, dict) else obs.player

    df_s, planet_disp = StrategyPipeline._01_get_obs_dataframe(obs, step, num_agents)
    pa = StrategyPipeline._02_get_all_opportunities(df_s, planet_disp, player_id)
    safe_attacks = StrategyPipeline._03_filter_collision(pa)
    moves = StrategyPipeline._04_score_and_decide(safe_attacks, player_id)

    step += 1
    return moves
```

---

## File Layout (top to bottom)

```
imports
GameConfig
PhysicsEngine
# ── Kaggle interpreter (untouched) ──
namedtuples (Planet, Fleet)
interpreter()
# ── Pipeline ──
StrategyPipeline
    _01_get_obs_dataframe
    _02_get_all_opportunities
    _03_filter_collision
    _04_score_and_decide
# ── Entry point ──
step / num_agents / player_id globals
agent()
```

---

## Out of Scope

- No behavioral changes, new heuristics, or parameter tuning.
- No splitting into multiple files.
- No RL or simulation changes.
