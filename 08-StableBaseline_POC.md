# 08 — MaskablePPO Proof of Concept

## Goal

Build a minimal but complete RL training pipeline for Orbit Wars using
Stable-Baselines3 + sb3-contrib, so we have a drop-in replacement path
for the rule-based agent (`07-claude_code.py`).

---

## Architecture

### Observation space — `Box(320,)`

40 planet slots sorted by **distance from the sun** (stable ordering even
as ownership changes). Each slot has 8 features:

| idx | feature | range |
|-----|---------|-------|
| 0 | `x / 100` | [0, 1] |
| 1 | `y / 100` | [0, 1] |
| 2 | `is_mine` | {0, 1} |
| 3 | `is_enemy` | {0, 1} |
| 4 | `is_neutral` | {0, 1} |
| 5 | `log1p(ships) / log1p(1000)` | [0, 1] |
| 6 | `production / 10` | [0, 1] |
| 7 | `is_valid` (0 = padding slot) | {0, 1} |

Total: `40 × 8 = 320` floats, all in `[0, 1]`.

### Action space — `MultiDiscrete([40] × 10)`

One discrete head per owned-planet slot (up to `MAX_OWNED = 10`).
Each head outputs a target planet index `[0, 40)`.

At decode time:
- Send **50 %** of ships from `owned[slot]` toward `planets[action[slot]]`
- Angle = `atan2(tgt.y - src.y, tgt.x - src.x)`
- Skip if source has < 2 ships or action points at itself

### Action mask — `bool[400]`

`mask[slot * 40 + j]` is `True` iff:
- Slot has an owned planet with >= 2 ships
- `j` is not the source planet's own sorted index

Safety: if nothing is sendable, `mask[0]` is forced `True` to prevent
NaN loss in MaskablePPO.

### Reward

| event | reward |
|-------|--------|
| each step | `+float(my_planet_count)` |
| terminal win | `+10` |
| terminal loss | `-10` |

Dense per-step signal drives planet accumulation; sparse terminal bonus
encourages winning.

### Opponent (self-play seed)

`random_opponent`: every owned planet sends 50 % of ships to a uniformly
random target each step.

---

## Files

| file | purpose |
|------|---------|
| `08-StableBaseline_POC.py` | full implementation (env + training + agent loader) |
| `orbit_wars_ppo.zip` | trained MaskablePPO weights (500 k steps) |

---

## Training run

**Config:**

```
algorithm  : MaskablePPO (sb3-contrib 2.8.0)
policy     : MlpPolicy
n_steps    : 2048
batch_size : 64
n_epochs   : 4
lr         : 3e-4
gamma      : 0.99
gae_lambda : 0.95
clip_range : 0.2
ent_coef   : 0.01
opponent   : random_opponent
episode_steps : 300
total_timesteps : 500 000
```

**Reward curve (mean over last 50 episodes):**

```
step    10 000  |  mean_ep_reward =  2515   episodes =    34
step    50 000  |  mean_ep_reward =  2282   episodes =   172
step   100 000  |  mean_ep_reward =  1427   episodes =   344
step   150 000  |  mean_ep_reward =  2120   episodes =   517
step   200 000  |  mean_ep_reward =  2184   episodes =   692
step   250 000  |  mean_ep_reward =  2427   episodes =   865
step   300 000  |  mean_ep_reward =  2150   episodes =  1038
step   400 000  |  mean_ep_reward =  2297   episodes =  1380
step   500 000  |  mean_ep_reward =  2364   episodes =  1725
```

Mean ~2200 / 300 steps ≈ **7–8 planets held per step** (out of ~20 total).
Reward oscillates rather than climbing — consistent with training against
a chaotic random opponent that provides a noisy gradient signal.

---

## Evaluation (5 games vs random opponent)

```
Game 1: WIN   (r0= 1.0  r1=-1.0)
Game 2: LOSS  (r0=-1.0  r1= 1.0)
Game 3: WIN   (r0= 1.0  r1=-1.0)
Game 4: LOSS  (r0=-1.0  r1= 1.0)
Game 5: LOSS  (r0=-1.0  r1= 1.0)

Result: 2W 3L 0D  (40 %)
```

**Caveat:** 5 games has a 95 % CI of ±22 %, so 40 % is not conclusive.
A proper benchmark needs 20–50 games via `simulate.py`.

---

## Diagnosed issues

### 1. Agent may reinforce instead of attack

`decode_action` skips moves where `tgt.id == src.id` (same planet), but the
agent can still legally send ships to **other owned planets**, which
wastes attack capacity. The network may have learned this as a safe
local optimum under the dense planet-count reward.

### 2. Dense reward doesn't teach winning

`+planets_owned` rewards holding, not taking. A random agent that spams
attacks everywhere can beat a cautious holder even with fewer ships.

### 3. Random opponent gives noisy gradient

Random actions create high-variance episode outcomes, making it hard
for PPO to identify which decisions caused wins vs losses.

---

## Recommended next steps

| priority | change | expected impact |
|----------|--------|----------------|
| High | Add **differential reward**: `+0.1 * (my_count - opp_count)` per step | incentivises taking planets from opponent |
| High | **Mask own-planet targets** (extend mask to exclude `p.owner == player`) | eliminates reinforce-instead-of-attack local optimum |
| High | Replace random opponent with **`07-claude_code.py`** rule agent | cleaner gradient signal; harder opponent forces better play |
| Medium | **Curriculum self-play**: save checkpoint every 100 k steps, use previous checkpoint as opponent | prevents policy collapse |
| Medium | Add **orbital intercept** to `decode_action` (copy from `07`) | agent learns to lead moving targets |
| Medium | Include **fleet features** in observation (incoming threats per planet) | agent can defend reactively |
| Low | Run `simulate.py --opponent yuriygreben --games 20` after each curriculum stage | tracks progress vs the real benchmark |

---

## How to load the trained agent

```python
from sb3_contrib import MaskablePPO
import importlib.util, math, numpy as np
from kaggle_environments.envs.orbit_wars.orbit_wars import Planet

spec = importlib.util.spec_from_file_location("poc", "08-StableBaseline_POC.py")
mod  = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

agent = mod.make_kaggle_agent("orbit_wars_ppo")
# agent(obs) is now a drop-in for simulate.py or Kaggle submission
```
