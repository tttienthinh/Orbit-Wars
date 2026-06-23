# Orbit Wars v8 — Max 1250 Score Notebook

## Purpose

A Kaggle competition notebook implementing a **Fusion Agent** for the Orbit Wars game.
It combines two previous strategies:
- **Producer V2** — reinforcement-risk heuristics
- **Producer Hybrid v4** — 4-player FFA bonuses and aggression tuning

Built on top of the `orbit_lite` library (physics, geometry, planning utilities).

---

## Notebook Structure

| Cell | Type | Purpose |
|------|------|---------|
| 0–2 | Markdown | Intro notes and context |
| 3 | Code | Reinstall `kaggle-environments ≥ 1.28.0` |
| 4 | Code | `%%writefile main.py` — full agent source |
| 5 | Code | Package `main.py` + `orbit_lite/` into `submission.tar.gz` |
| 6–7 | Empty | Placeholders |

---

## Agent Architecture (`main.py`)

### Configuration (`ProducerLiteConfig`)

A frozen dataclass holding all strategy parameters:

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `horizon` | 18 | Steps to simulate ahead |
| `max_waves_per_turn` | 6 | Max attack waves per turn |
| `roi_threshold` | 1.5 | Min return-on-investment to launch |
| `min_ships_to_launch` | 4.0 | Minimum fleet size |
| `reinforce_size_beta` | 2.2 | Weight for enemy reinforcement risk |
| `ffa_leader_attack_bonus` | 0.035 | Bonus score for attacking the FFA leader |
| `ffa_target_prod_bonus` | 0.08 | Bonus score for targeting high-production planets |

A separate **4-player config** (`CONFIG_4P`) reduces horizon to 13, tightens sources/targets, and raises the ROI threshold to 1.55.

---

### Key Functions

#### `cheap_enemy_pressure(obs, cache, horizon, player_id) → Tensor`

Estimates how many enemy ships can reach each planet within `horizon` steps.
Used to penalize attacks that expose friendly planets to counter-strikes (reinforcement risk).

Algorithm:
1. Compute fleet speed for each enemy planet's ship count.
2. Derive reachable distance = `speed × horizon`.
3. For each enemy → target pair, apply a linear decay weight: `1 - dist/reach`.
4. Sum contributions over all enemy planets.

---

#### `plan_lite_waves(...)` — Core Planner

Plans all attacks and regrouping for one turn. Steps:

1. **Source selection** — Filter own planets with `ships ≥ min_ships_to_launch`. Pick top `max_sources_per_lane` by ship count.
2. **Target selection** — `build_target_shortlist()` ranks offensive + defensive targets using distance, production value, and garrison state.
3. **Safe drain** — For each source, compute how many ships can safely leave without over-exposing it.
4. **Reinforcement risk (V2 feature)** — Scale the capture requirement upward based on expected enemy counter-fleets, controlled by `reinforce_size_beta`.
5. **Capture floor** — Minimum ships needed to take each target at each arrival time step.
6. **Reachability check** — Verify the source can actually reach the target within the horizon.
7. **Intercept aiming** — `intercept_angle()` calculates the optimal launch angle accounting for planet orbital motion (iterative, 5 rounds).
8. **Candidate matrix** — Build `S × T` grid of (source, target) attack candidates.
9. **Scoring** — `score_candidates()` evaluates each attack by projected production gain.
10. **4P FFA bonuses (Hybrid v4 feature)** — When `player_count ≥ 4`, add bonus score for attacking the current FFA leader and for targeting high-production enemy planets.
11. **Greedy selection** — `_greedy_select()` picks up to `max_waves_per_turn` best non-conflicting attacks.
12. **Regroup** — Transfer leftover ships between own planets under enemy pressure.

---

#### `run_turn(obs_tensors, config, player_count, memory) → dict`

Executes one game turn:
1. Parse raw observation tensors into a typed `obs` struct.
2. Update/reuse cached `PlanetMovement` (orbital sim model).
3. Build distance cache up to `horizon` steps.
4. Call `plan_lite_waves()` to get launch entries.
5. Deduplicate and apply launches to the movement model.
6. Return a sparse action payload (planet_id → angle + ships).

---

#### `agent(obs) → list`

The Kaggle entry point called every turn:
```
obs → single_obs_to_tensor → _RUNTIME.tensor_action → sparse_action_row_to_moves → [[planet_id, angle, ships], ...]
```

`_RUNTIME` is a module-level singleton that holds `ProducerLiteMemory` across turns (caches movement model and player count).

---

## Physics / Strategy Notes

- **Orbital intercept**: Launch angle is computed iteratively using `intercept_angle()` from `orbit_lite`, which accounts for planetary orbit during fleet travel time.
- **Reinforcement risk**: Before attacking, the agent checks if enemy can reinforce the target before arrival. If yes, the required ship count is raised by `beta × enemy_mass`.
- **4P mode**: Automatically activates when `player_count ≥ 4`. Shrinks horizon, adds FFA-leader aggression, and raises the ROI bar to avoid overcommitting in multi-player chaos.

---

## Packaging (Cell 5)

```python
build = Path(tempfile.mkdtemp())
shutil.copy2(WORK / "main.py", build / "main.py")
shutil.copytree(SRC, build / "orbit_lite")  # copy orbit_lite library
archive = WORK / "submission.tar.gz"
# tar main.py + orbit_lite/ → submission.tar.gz
```

Bundles `main.py` and the `orbit_lite` dependency from the Kaggle dataset into a single archive for submission.

---

## Dependency Map

```
agent(obs)
  └── ProducerLiteRuntime.tensor_action()
        └── run_turn()
              ├── parse_obs()               [orbit_lite.obs]
              ├── PlanetMovement            [orbit_lite.movement]
              ├── build_distance_cache()    [orbit_lite.distance_cache]
              └── plan_lite_waves()
                    ├── cheap_enemy_pressure()
                    ├── build_target_shortlist()  [orbit_lite.planner_core]
                    ├── safe_drain()              [orbit_lite.planner_core]
                    ├── capture_floor()           [orbit_lite.planner_core]
                    ├── intercept_angle()         [orbit_lite.intercept_aim]
                    ├── score_candidates()        [orbit_lite.planner_core]
                    ├── _greedy_select()          [orbit_lite.planner_core]
                    └── _plan_regroup()           [orbit_lite.planner_core]
```
