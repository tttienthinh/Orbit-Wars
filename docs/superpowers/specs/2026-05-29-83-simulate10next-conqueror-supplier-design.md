# Design: 83-Simulate10Next_Conqueror_Supplier.ipynb

**Date:** 2026-05-29  
**Status:** Approved

## Purpose

A test notebook for the `StrategyPipeline` defined in `82-Simulate10Next_Conqueror_Supplier.py`. Each test exposes all intermediate pipeline DataFrames so they can be inspected and experimented with interactively.

## Code Loading

```python
%run 82-Simulate10Next_Conqueror_Supplier.py
```

Loads: `StrategyPipeline`, `interpreter`, `GameConfig`, `PhysicsEngine`, `agent`.

The `Obs` class, `simulate_with_action`, and `make_animation` are NOT in the `.py` file — they are defined inline in the notebook (copied from `43-Dataframe_comet.ipynb`). Also import `kaggle_environments as ke` for Test 05.

## Notebook Structure

### Setup (cells 0–2)

| Cell | Type | Content |
|------|------|---------|
| 0 | md | Title |
| 1 | code | `%run 82-Simulate10Next_Conqueror_Supplier.py` |
| 2 | code | imports (`copy`, `math`, `matplotlib`, etc.), `Obs` class, `simulate_with_action`, `make_animation`, `import kaggle_environments as ke` |

### Per-test pattern (5 cells each, Tests 01–04)

```
[md]   ## Test N — <description>
[code] obs = Obs(planets=[...], angular_velocity=...)
       df_s, planet_disp = StrategyPipeline._01_get_obs_dataframe(obs, 0, 2)
       df_s                          # renders DataFrame inline
[code] pa = StrategyPipeline._02_get_all_opportunities(df_s, planet_disp, player_id=0)
       pa
[code] safe = StrategyPipeline._03_filter_collision(pa)
       safe
[code] action = StrategyPipeline._04_score_and_decide(safe, player_id=0)
       print("Action:", action)
       snaps = simulate_with_action(copy.deepcopy(obs), action, 20)
       make_animation(snaps, title='Test N — ...')
```

### Test 05 — vs random agent (kaggle_environments)

Mirrors Test 6 from `43-Dataframe_comet.ipynb`:
- `ke.make("orbit_wars")`, `env.reset(2)`
- Loop up to `N_STEPS=100`: player 0 uses `agent(obs)` from `82-...py`, player 1 uses a local `random_agent_fn`
- Print final ship counts and winner
- `make_animation` on captured snapshots

## Planet Configurations

Planet format: `[id, owner, x, y, radius, ships, production]`  
`owner`: 0 = us, 1 = enemy, -1 = neutral  
`radius=1+math.log(3)` ≈ 2.099

### Test 01 — 1 Supplier, 1 Conqueror, 1 Enemy

```python
planets=[
    [0, 0, 20.0, 20.0, 1+math.log(3), 50, 3],  # Supplier
    [1, 0, 30.0, 20.0, 1+math.log(3), 50, 3],  # Conqueror (closer to enemy)
    [2, 1, 70.0, 20.0, 1.0,           1,  1],  # Enemy
]
angular_velocity=0.0
```

### Test 02 — 1 Supplier, 2 Conquerors, one more in need

```python
planets=[
    [0, 0, 20.0, 20.0, 1+math.log(3), 50,  3],
    [1, 0, 30.0, 20.0, 1+math.log(3), 50,  3],
    [2, 1, 70.0, 20.0, 1+math.log(3), 1,   1],
    [3, 0, 20.0, 30.0, 1+math.log(3), 50,  3],
    [4, 1, 20.0, 70.0, 1+math.log(3), 100, 1],
]
angular_velocity=0.0
```

### Test 03 — 4 Suppliers, 2 Conquerors, one more in need

```python
planets=[
    [0, 0, 20.0, 20.0, 1+math.log(3), 50,  3],
    [1, 0, 30.0, 20.0, 1+math.log(3), 50,  3],
    [2, 1, 70.0, 20.0, 1+math.log(3), 1,   1],
    [3, 0, 20.0, 30.0, 1+math.log(3), 50,  3],
    [4, 1, 20.0, 70.0, 1+math.log(3), 100, 1],
    [5, 0, 10.0, 10.0, 1+math.log(3), 50,  3],
    [6, 0, 10.0, 25.0, 1+math.log(3), 50,  3],
    [7, 0, 25.0, 10.0, 1+math.log(3), 50,  3],
]
angular_velocity=0.0
```

### Test 04 — 4 Suppliers, 1 Conqueror orbiting

```python
planets=[
    [0, 0, 20.0, 20.0, 1+math.log(3), 50, 3],
    [1, 0, 10.0, 10.0, 1+math.log(3), 50, 3],
    [2, 0, 10.0, 25.0, 1+math.log(3), 50, 3],
    [3, 0, 25.0, 10.0, 1+math.log(3), 50, 3],
    [4, 0, 25.0, 25.0, 1.0,           50, 1],  # orbiting (dist≈35 < 50−r)
    [5, 1, 30.0, 30.0, 1.0,           50, 1],  # orbiting enemy (dist≈28 < 50−r)
]
angular_velocity=0.05
```

Planets 4 and 5 are within orbit radius (`dist_from_center + radius < 50`), so `angular_velocity=0.05` makes orbiting visible.

### Test 05 — vs random agent

Uses `kaggle_environments`:
```python
def random_agent_fn(obs):
    player = obs.player
    my_planets = [p for p in obs.planets if p[1] == player]
    if not my_planets:
        return []
    planet = random.choice(my_planets)
    ships = planet[5] // 2
    if ships < 1:
        return []
    return [[planet[0], random.uniform(0, 2 * math.pi), ships]]
```

## Key Constraints

- `%run` is required because the filename starts with a digit (not importable as a module)
- `Obs` must `deepcopy` planets on init so test mutations don't bleed across cells
- `simulate_with_action` applies `action` on step 0 only, then idle steps
- `make_animation` y-axis is inverted (`ylim(100, 0)`) matching the game coordinate system
