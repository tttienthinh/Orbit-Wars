# 132-Polars_speed — Cached Simulation Design

**Date:** 2026-06-21  
**Source:** `92-90-Polars.py`  
**Target:** `132-Polars_speed.py`

## Goal

Eliminate redundant interpreter calls in `_01_get_obs_dataframe` by introducing a `GameCache`
singleton that carries over planet positions between consecutive agent calls and caches
in-transit fleet arrival destinations. Outputs must match `92-90-Polars.py` exactly on every
game step.

## Background

`_01_get_obs_dataframe` calls `interpreter()` 10 times per agent call (NB_STEPS_SIM=10) in a
no-action simulation to collect planet positions and ship counts over the next 10 steps.

Key insight: at step T the simulation covers steps T..T+10. At step T+1 it covers T+1..T+11.
Steps T+1..T+10 are computed identically in both calls because:
- Orbiting planet positions depend only on `initial_angle + ω × (game_step − 1)` — deterministic
- Fixed planet positions never change
- Comet positions come from the `paths` array — deterministic

Only step T+11 (one new step) needs new computation. This reduces position computation from 10
interpreter calls to 1 analytical formula evaluation per call.

## Architecture

### `GameCache` (global singleton)

One instance created on the first `agent()` call, lives for the entire game.

```
CACHE: GameCache | None = None

def agent(obs):
    global CACHE
    if CACHE is None:
        CACHE = GameCache(obs, step=0, ...)
    CACHE.advance(obs, current_step)
    df_s, planet_disp = CACHE.build_df_s(obs, current_step)
    ...
```

### State fields

| Field | Type | Description |
|-------|------|-------------|
| `step` | `int` | Current game step (replaces global) |
| `num_agents` | `int` | 2 or 4 (replaces global) |
| `player_id` | `int` | Agent's player index (replaces global) |
| `angular_velocity` | `float` | From `obs.angular_velocity` |
| `planet_meta` | `dict[int, dict]` | Per planet: nature, r, θ₀, radius, production |
| `comet_paths` | `dict[int, list]` | planet_id → path coordinate list |
| `_pos_window` | `dict[(int,int), (float,float)]` | (planet_id, game_step) → (x, y) |
| `_fleet_arrival` | `dict[int, tuple\|None]` | fleet_id → (planet_id, arrival_step) or None |

### Rolling position window

**Initialization** (step T, first call):
- Compute `(planet_id, game_step) → (x, y)` for every planet × every step in T..T+NB_STEPS_SIM.

**Per-call update** (step T+1 onwards):
- The window already holds steps T+1..T+NB_STEPS_SIM.
- Compute one new entry: step T+NB_STEPS_SIM+1.
- Evict entries at step T.

**Position formula per planet type:**

| Type | k=0 (current step) | k≥1 (future steps) |
|------|--------------------|--------------------|
| orbiting | `obs.planets[id].(x,y)` | `θ = θ₀ + ω × (T+k−1)`; `x = 50 + r·cos(θ)`, `y = 50 + r·sin(θ)` |
| fixed | `obs.planets[id].(x,y)` | same constant `(x₀, y₀)` — cached once, never evicted |
| comet | `obs.planets[id].(x,y)` | `comet_paths[id][obs_path_index + k]` if in range, else expired |

> **Critical — orbiting formula:** Uses `T+k−1`, not `T+k`. The interpreter called with
> `step=T` applies `θ₀ + ω × T`, which sets the planet position AFTER advancing step T→T+1.
> This position is recorded at i=1 (game_step T+1 = T+k where k=1), giving
> `θ = θ₀ + ω × T = θ₀ + ω × (T+1−1)`. ✓
>
> **Critical — comet indexing:** Comets use a `path_index` that advances by 1 each step.
> At i=0, obs already has the current `path_index` (call it `idx`); after interpreter step k,
> the comet is at `path[idx + k]`. So position at game_step T+k = `path[idx + k]` where
> `idx = group["path_index"]` from `obs.comets` at the start of this agent call.
> The rolling cache stores `_pos_window[(comet_id, T+k)] = path[idx + k]`, which is
> consistent across calls because at step T+1 the new `idx` = old `idx + 1` and
> `path[(idx+1) + (k-1)] = path[idx + k]`. ✓

### Fleet arrival cache

**Purpose:** For each in-transit fleet, compute once which planet it hits and at which step,
instead of re-running swept_pair_hit inside the interpreter every call.

**Cache key:** `fleet_id` (fleet[0]).

**On each `advance()` call:**
1. Evict entries whose fleet_id is absent from `obs.fleets` (fleet landed or destroyed).
2. For each new fleet_id in `obs.fleets`: compute arrival by stepping the fleet forward
   one step at a time (up to NB_STEPS_SIM), running `swept_pair_hit(f_old, f_new, p_old,
   p_new, radius)` against all non-expired planets using positions from `_pos_window`.
   Store `(planet_id, arrival_step)` or `None` (fleet exits bounds or hits sun without
   hitting a planet in the simulation window).

**Reuse:** Existing fleet_ids use their cached entry unchanged.

### Analytical ships/owner timeline

Replaces the 10 interpreter calls with 11 lightweight iterations.

**Algorithm:**

```
state[0] = {planet_id: (ships, owner)} from obs.planets

for k in 0 .. NB_STEPS_SIM:
    state[k+1] = copy of state[k]
    # 1. Production (same order as interpreter: before combat)
    for each planet p where state[k][p].owner != -1:
        state[k+1][p].ships += p.production
    # 2. Fleet arrivals at game_step = current_step + k + 1
    arrivals = [f for f in obs.fleets if fleet_arrival[f.id] == (pid, current_step+k+1)]
    for each planet p with arriving fleets:
        resolve combat using same logic as interpreter:
          player_ships = sum ships by owner
          sort descending; survivor = top − second (or 0 if tie)
          if planet owner == survivor: ships += survivor_ships
          else: ships -= survivor_ships; if ships < 0: flip owner
```

**Correctness invariants:**
- Production added before combat at each step.
- Same combat math as interpreter (top − second, tie → 0 survivors).
- Ownership changes propagate: after a conquest at step k, production at step k+1 uses the
  new owner.

### `build_df_s` output

Returns `(df_s, planet_disp)` — Polars DataFrames with **exactly the same column schema** as
`_01_get_obs_dataframe`:

`df_s` columns: `step, id, x, y, radius, ships, production, owner, nature`  
`planet_disp` columns: `id, step, planet_disp`

Downstream methods `_02_`, `_03_`, `_04_` are **zero-diff** — no changes.

## Data Flow

```
agent(obs)
  │
  ├─ CACHE.initialize(obs)          [first call: build planet_meta, comet_paths, fill pos_window]
  │
  ├─ CACHE.advance(obs, step)
  │    ├─ extend pos_window → add (planet_id, step+NB_STEPS_SIM+1)
  │    ├─ evict pos_window  → drop entries at step-1
  │    ├─ evict fleet_arrival: remove ids not in obs.fleets
  │    └─ compute fleet_arrival for new fleet ids
  │
  ├─ df_s, planet_disp = CACHE.build_df_s(obs, step)
  │    ├─ build analytical ships/owner timeline (11 iterations)
  │    ├─ for k in 0..10: positions ← pos_window[(planet_id, step+k)]
  │    └─ assemble df_s and planet_disp as Polars DataFrames
  │
  ├─ pa_lf   = _02_get_all_opportunities(df_s, planet_disp, player_id)   [unchanged]
  ├─ safe_lf = _03_filter_collision(pa_lf)                                [unchanged]
  └─ moves   = _04_score_and_decide(safe_lf, player_id)                  [unchanged]
```

## Correctness Verification

Exact match is required on all outputs. Verification is done in three layers:

1. **DataFrame-level**: Run both agents on the same obs sequence from a saved Kaggle episode.
   Assert `df_s` DataFrames are equal row-by-row (positions, ships, owner, nature columns).

2. **Moves-level**: Shadow-run both agents for a full 500-step game on a local replay.
   Assert `moves` lists are identical at every step.

3. **Kaggle env**: Submit `132-Polars_speed.py` and compare episode replays against
   `92-90-Polars.py` replays — same decisions at every step confirms correct output.

**Key correctness checks:**
- Position formula: `θ₀ + ω × (step − 1)` not `× step`
- Production before combat (not after)
- Expired comets evicted from both pos_window and fleet_arrival
- Fixed planets never evicted from pos_window
- Multi-fleet combat at same planet same step resolved identically to interpreter

## Edge Cases

| Case | Handling |
|------|---------|
| Comet expires mid-window | Evict from pos_window; evict fleet_arrival entries targeting it |
| Fleet hits sun or boundary | `fleet_arrival[id] = None`; no ownership change |
| Two fleets same planet same step | Combat resolved once (aggregate, same as interpreter) |
| Planet conquered mid-sim | Owner update propagates to production in later steps |
| step=0 (first call) | pos_window filled from scratch; fleet_arrival empty → compute all |

## Performance Expectation

- `_01_get_obs_dataframe`: from 10 interpreter calls to **0 interpreter calls** (after initialization)
- Rolling window: **1 analytical position computation** per call (one new step)
- Fleet arrival: **0 swept_pair checks** for existing fleets; only new fleets computed
- Expected speedup on `_01_get_obs_dataframe`: ~80–90%
- Downstream pipeline (`_02_`–`_04_`): unchanged
