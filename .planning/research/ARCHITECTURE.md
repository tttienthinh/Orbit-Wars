# Architecture Research — Orbit Wars Competitive Agent

**Analysis Date:** 2026-05-06

## Recommended Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    agent(obs) entry point                    │
│                        main.py                              │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                     State Layer                              │
│  parse_obs()  →  predict_positions(T)  →  GameState         │
│                                                             │
│  - Parse Planet/Fleet namedtuples                           │
│  - Predict orbiting planet positions 0..HORIZON turns ahead  │
│  - Classify: my_planets, enemy_planets, neutral_planets     │
│  - Identify incoming threats per owned planet               │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                   Candidate Generator                        │
│  for each (source_planet, target_planet) pair:              │
│    - compute intercept turn T, intercept angle              │
│    - check sun collision on that angle                      │
│    - compute ships_needed (garrison + margin + prod*T)       │
│    - compute value score                                    │
│  → list[Candidate(source, target, angle, ships, score)]     │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                     Planner                                  │
│  - Sort candidates by score (descending)                    │
│  - Greedily assign: deduct ships from source budget         │
│  - Enforce: source keeps MIN_GARRISON after all sends       │
│  - Skip conflicting sends (same source, over-budget)         │
│  → list of approved moves                                   │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                    Defense Pass                              │
│  - For each owned planet with incoming threat:              │
│    - compute deficit at arrival turn                        │
│    - insert reinforcement move from nearest safe source     │
│    - override attack move if defense is higher priority     │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
                    return moves list
```

## Component Boundaries

| Component | Inputs | Outputs | File |
|-----------|--------|---------|------|
| State layer | raw obs dict | `GameState` dataclass | `agent/state.py` |
| Physics | planet x/y/angular_velocity, turn T | predicted (x, y) at turn T | `agent/physics.py` |
| Intercept solver | source pos, target trajectory, fleet speed | (angle, arrival_turn) or None | `agent/physics.py` |
| Sun check | source pos, launch angle | bool (safe / blocked) | `agent/physics.py` |
| Scorer | Candidate | float score | `agent/scorer.py` |
| Planner | list[Candidate] | list[move] | `agent/planner.py` |
| Defense | GameState + planned moves | additional/override moves | `agent/defense.py` |
| Entry point | obs | list[move] | `main.py` |

## Physics Module — Key Functions

```python
# agent/physics.py

def fleet_speed(ships: int) -> float: ...

def planet_position(planet, turn: int, angular_velocity: float) -> tuple[float, float]:
    """Returns (x, y) of orbiting planet at future turn. Static planets return current pos."""

def intercept(
    src_x: float, src_y: float, src_radius: float,
    tgt_x: float, tgt_y: float, tgt_is_orbiting: bool,
    angular_velocity: float, ships: int,
    max_turns: int = 120
) -> tuple[float, int] | None:
    """
    Returns (angle_radians, arrival_turn) or None if unreachable.
    CRITICAL: src position is planet center; fleet spawns at surface.
    Adjust: effective_src = center + unit_vector(angle) * src_radius
    This requires iterative solve since angle affects spawn position.
    """

def path_crosses_sun(src_x, src_y, angle, board_size=100, sun_cx=50, sun_cy=50,
                      sun_r=10, safety=1.5) -> bool:
    """Parametric ray vs circle intersection test."""
```

## RL Wrapper Architecture

```
┌─────────────────────────────────────────────────────────────┐
│              OrbitWarsEnv(pettingzoo.AECEnv)                 │
│                                                             │
│  reset() → {agent_id: observation}                          │
│  step(action) → reward, done, info                          │
│                                                             │
│  Observation: flat numpy array (826 floats, see STACK.md)   │
│  Action: MultiDiscrete([41] * MAX_MY_PLANETS)               │
│          per-planet: index 0=hold, 1..40=target planet      │
│                                                             │
│  Reward: Δ(my_ships - enemy_ships) per turn                 │
│          + 100 on win, -100 on loss                         │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│              supersuit wrappers                              │
│  ss.pad_observations_v0()   ← handles variable planet count │
│  ss.normalize_obs_v0()      ← normalize to [-1, 1]          │
│  ss.pettingzoo_env_to_vec_env_v1()  ← SB3 compatibility     │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│              StableBaselines3 PPO                            │
│  policy: MlpPolicy (default; try LSTM if needed)            │
│  n_steps: 2048, batch_size: 256, n_epochs: 10               │
│  ent_coef: 0.01 (encourage exploration)                     │
│  Self-play: train against snapshot pool of past checkpoints  │
└─────────────────────────────────────────────────────────────┘
```

## Build Order (phase dependencies)

```
1. Physics module (state.py, physics.py)
   — everything downstream depends on correct intercept/sun math

2. Scorer + Planner (scorer.py, planner.py)
   — needs physics; produces moves

3. Defense pass (defense.py)
   — needs scorer/planner and threat detection

4. main.py integration + local testing
   — wire all modules; submit to Kaggle; benchmark in lab

5. RL env wrapper (train/orbit_wars_env.py)
   — needs stable heuristic agent as warm-start policy

6. PPO training loop (train/train.py)
   — needs env wrapper; train on CPU first, GPU if needed
```

## File Structure (proposed)

```
main.py                     ← Kaggle submission entry point (imports from agent/)
agent/
  __init__.py
  state.py                  ← GameState dataclass, parse_obs()
  physics.py                ← fleet_speed, planet_position, intercept, path_crosses_sun
  scorer.py                 ← Candidate, score_candidate()
  planner.py                ← select_moves()
  defense.py                ← detect_threats(), insert_defense_moves()
train/
  orbit_wars_env.py         ← PettingZoo AECEnv wrapper
  train.py                  ← SB3 PPO self-play training loop
  eval.py                   ← evaluate checkpoint vs agent zoo
```

**Submission constraint:** `main.py` must be self-contained OR submitted as `submission.tar.gz`
with the full `agent/` package. The multi-file tar approach is already documented in `agents.md`.

---
*Architecture research: 2026-05-06 (inline, agents rate-limited)*
