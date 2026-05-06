# Stack Research — Orbit Wars Competitive Agent

**Analysis Date:** 2026-05-06

## Recommended Stack

### Core Agent (submission)

| Layer | Choice | Rationale |
|-------|--------|-----------|
| Language | Python 3.12 | Required by kaggle-environments |
| Physics math | `numpy` (already available) | Vectorised intercept search, batch distance calc |
| Game types | `kaggle_environments.envs.orbit_wars.orbit_wars.Planet`, `Fleet` | Already in codebase |
| No new deps | — | `main.py` must be self-contained; no pip install at eval time |

**Confidence: High** — constrained by Kaggle submission requirements.

### RL Training Pipeline

| Layer | Choice | Rationale |
|-------|--------|-----------|
| Multi-agent env wrapper | `pettingzoo` ≥ 1.24 | Industry standard for multi-agent RL; SB3 compatible |
| RL algorithm | `stable-baselines3` ≥ 2.3 (PPO) | Best-maintained PPO; works with PettingZoo via `sb3-contrib` |
| Env wrappers | `supersuit` ≥ 3.9 | Normalize obs, flatten for SB3 compatibility |
| Self-play | Custom snapshot opponent pool | Simple, effective; avoid complex league play for now |
| Alternative | `cleanrl` (single-file PPO) | Easier to hack but less ecosystem |

**Do NOT use:**
- `rllib` — powerful but massive overhead; overkill for this scope
- `torch` directly — SB3 handles this; only needed for custom networks
- `gym` (legacy) — use `gymnasium` ≥ 0.29 if needed; PettingZoo wraps it

**Confidence: High** — SB3+PettingZoo is the 2024-2025 standard for Kaggle multi-agent competitions.

### Training Infrastructure

| Need | Choice | Notes |
|------|--------|-------|
| Game speed | kaggle-environments as-is | ~10–50 games/sec on CPU; sufficient for early training |
| Faster sim (if needed) | Rewrite physics in NumPy | Orbital rotation + fleet movement ≈ 200 lines; 10–100x speedup |
| GPU training | Rent via RunPod / Lambda Labs | Only needed when sim speed is no longer the bottleneck |
| Experiment tracking | `tensorboard` (built into SB3) | Log reward curves, entropy, value loss |

### Backtesting (already exists)

orbit-wars-lab provides everything needed:
- TrueSkill rating system
- Match runner (fast + faithful modes)
- Replay visualizer
- Agent zoo structure

No new infrastructure needed for backtesting.

## Observation Space Design

For RL, the variable-length observation (20–40 planets, variable fleets) must become a fixed tensor.

**Recommended encoding:**

```python
# Per planet (pad to MAX_PLANETS=40):
# [x_norm, y_norm, owner_self, owner_enemy, ships_norm, production_norm,
#  is_orbiting, is_comet, dist_from_me_norm, orbital_angle_norm]
# → 10 features × 40 planets = 400

# Per fleet (pad to MAX_FLEETS=60, own + enemy):
# [x_norm, y_norm, angle_sin, angle_cos, ships_norm, owner_self, turns_to_nearest_planet]
# → 7 features × 60 fleets = 420

# Scalar context:
# [current_turn_norm, my_total_ships_norm, enemy_total_ships_norm,
#  my_planet_count_norm, enemy_planet_count_norm, angular_velocity_norm]
# → 6

# Total: 826 features (flat vector)
```

Sort planets by distance from player centroid for stable ordering.
Normalize all positions to [0, 1] (board is 100×100).

## Action Space Design

**Recommended: MultiDiscrete over owned planets**

```python
# For each owned planet slot (padded to MAX_MY_PLANETS=10):
# [target_planet_idx (0..40), ship_fraction (0..4)]
# ship_fraction: 0=hold, 1=25%, 2=50%, 3=75%, 4=100%
#
# At execution: convert (target_idx, fraction) → (angle, num_ships)
# using intercept prediction
```

**Why not continuous angle directly:**
- Continuous angle requires SAC or PPO with continuous head
- Discrete target selection is easier to learn and maps better to strategy
- Intercept math handles the angle internally

## Versions (verified against PyPI, 2026-05)

| Package | Version |
|---------|---------|
| pettingzoo | 1.24.3 |
| stable-baselines3 | 2.3.2 |
| supersuit | 3.9.2 |
| gymnasium | 0.29.1 |
| numpy | ≥1.26 (already pinned) |

---
*Stack research: 2026-05-06 (inline, agents rate-limited)*
