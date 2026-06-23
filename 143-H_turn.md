# 143-H_turn.py — How It Works

`143-H_turn.py` is a tensor-driven competitive agent built on the `orbit_lite` package. Instead of simulating the game step-by-step in Python or Polars, it projects the entire battle 18 steps forward as batched PyTorch tensors and picks moves that maximise a competitive "flow diff" score.

---

## High-Level Pipeline (one turn)

```
obs (raw Kaggle dict/namespace)
  │
  ▼  single_obs_to_tensor
obs_tensors (dict of PyTorch tensors)
  │
  ▼  ensure_planet_movement
PlanetMovement  ← rolling cache (re-used across turns)
  │
  ├─▶ build_distance_cache   → cross_dist[k] per step
  ├─▶ garrison_status(H)     → [P, H+1] ship projections
  └─▶ plan_lite_waves        → LaunchEntries
        │
        ├─▶ disambiguate_duplicate_launches
        ├─▶ infer_planned_launches_from_entries
        └─▶ apply_private_planned_launches  (updates movement for next turn)
  │
  ▼  entries_to_sparse_payload → sparse_action_row
  │
  ▼  sparse_action_row_to_moves
moves  (list of [planet_id, angle, ships])
```

---

## Key Components

### `ProducerLiteConfig` — Behaviour knobs

| Knob | Default (2P) | 4P FFA | Role |
|---|---|---|---|
| `horizon` | 18 | 13 | Steps to look ahead |
| `max_sources_per_lane` | 12 | 6 | Cap on source planets |
| `max_offensive_targets` | 12 | 12 | Enemy/neutral shortlist size |
| `max_defensive_targets` | 4 | 2 | Friendly reinforce shortlist |
| `max_waves_per_turn` | 6 | 6 | Max fleet launches per step |
| `roi_threshold` | 1.5 | 1.5 | Minimum score to fire |
| `min_ships_to_launch` | 4.0 | 4.0 | Don't launch tiny fleets |
| `max_regroup_time` | 7.0 | 6.0 | Max regroup travel time |

2P vs 4P is auto-detected at step 0 via `largest_initial_player_count`.

---

### `PlanetMovement` — Rolling cache

Built by `ensure_planet_movement` from `orbit_lite.movement`. Tracks:
- Orbital positions of every planet for all `horizon` future steps
- Fleet arrivals (for both players) pre-computed via `infer_planned_launches_from_entries` and `apply_private_planned_launches`
- A `garrison_status` tensor `[P, H+1]` — projected ship counts for each planet at each future step, accounting for production and known fleet arrivals

The cache is persisted in `ProducerLiteMemory.movement` and **updated each turn** with the launches we just decided to fire, so the next turn's garrison projection already knows our fleets are in flight.

---

### `plan_lite_waves` — Attack planner

#### Step 1: Build shortlists
- **Sources**: up to `max_sources_per_lane` owned planets with ≥ `min_ships_to_launch` ships → tensor `source_idx [S]`
- **Targets**: up to `max_offensive_targets` enemy/neutral + `max_defensive_targets` friendly planets → tensor `target_idx [T]`

#### Step 2: Fleet size = `safe_drain`
One fleet size per source: the maximum ships the source can safely launch without leaving itself defenceless over the horizon window. Result: `drain [S]`.

#### Step 3: Reachability precheck
`reachable_mask` filters (source, target) pairs by whether the fleet can physically arrive within `horizon` steps, given the fleet speed formula. Produces `active [S, T]`.

#### Step 4: Intercept aiming
`intercept_angle` computes the angle and ETA for each (source, target, fleet-size) triple, accounting for orbital drift of moving planets. Returns `angle [S, T]`, `eta [S, T]`, `viable [S, T]`.

#### Step 5: Capture-floor gate
`capture_floor` asks: how many ships does the target have at arrival step k (counting production accrued during transit)? The fleet must meet or exceed this floor. Ensures we only send winning attacks.

#### Step 6: Score candidates
`score_candidates` computes a competitive flow diff for each (source → target) pair — essentially: how much better off is our total production/ship balance if this fleet lands versus not landing? Result: `score [C]` where C = S × T.

#### Step 7: Greedy select
`_greedy_select` fires waves in score order, deducting each launch from the source's budget and skipping targets already covered. Stops after `max_waves_per_turn` or when `score < roi_threshold`.

#### Step 8: Regroup
`cheap_enemy_pressure` estimates incoming enemy threat per planet (distance-decayed enemy garrison). `_plan_regroup` moves surplus ships from safe backlines toward threatened frontline planets, using leftover capacity after attack waves.

---

### `agent(obs)` — Entry point

```python
def agent(obs):
    obs_tensors = single_obs_to_tensor(obs, player_id=player_id)
    sparse_row  = _RUNTIME.tensor_action(obs_tensors)   # full pipeline above
    return sparse_action_row_to_moves(sparse_row, obs, player_id=player_id)
```

`_RUNTIME` is a module-level singleton that keeps `ProducerLiteMemory` alive across turns.

---

## What Makes This Different From Earlier Agents

| Earlier agents (60–142) | 143-H_turn |
|---|---|
| Polars / Pandas DataFrames | PyTorch tensors throughout |
| Step-by-step Python simulation | Analytical H-step garrison projection |
| Ships_sent = all-in or 1..N range | `safe_drain`: source-aware garrison budget |
| 1–2 ply lookahead | 18-step forward projection |
| Net-production delta scoring | Competitive flow diff (production + ships) |
| No fleet tracking across turns | Rolling `PlanetMovement` cache updated each turn |
| No regroup | Distance-weighted regroup to frontline |

The main speed advantage: the entire planning pass is O(S × T) batched tensor ops — no Python loops over candidates.
