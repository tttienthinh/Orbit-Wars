# Board Class — Design Spec

## Goal

A `Board` class in `game.py` that wraps game state in pandas DataFrames and provides a faithful
one-step forward simulator. Intended for look-ahead planning in agent scripts.

## Source of truth

`orbit-wars-lab/.venv/Lib/site-packages/kaggle_environments/envs/orbit_wars/orbit_wars.py`
— `interpreter()` at line 313 is the authoritative next-state function.

Physics helpers to reuse (copy into `game.py`):
`16-Getting_started_moving_precision_src_tgt_cst.py`
— `fleet_speed`, `next_position`, `get_distance`, `position_to_angle`, `angle_to_position`

---

## Data Structures

### Planet (list in obs, namedtuple for agent use)
```
index:  0=id  1=owner  2=x  3=y  4=radius  5=ships  6=production
namedtuple: Planet = namedtuple("Planet", ["id","owner","x","y","radius","ships","production"])
owner: -1=neutral, 0-3=players
```

### Fleet (list in obs)
```
index:  0=id  1=owner  2=x  3=y  4=angle  5=from_planet_id  6=ships
namedtuple: Fleet = namedtuple("Fleet", ["id","owner","x","y","angle","from_planet_id","ships"])
angle: radians, standard math (0=right, π/2=up)
```

### Observation dict keys
```python
obs = {
    "player":              int,       # 0-3
    "planets":             list,      # list of [id,owner,x,y,radius,ships,production]
    "fleets":              list,      # list of [id,owner,x,y,angle,from_planet_id,ships]
    "angular_velocity":    float,     # per-game constant in [0.025, 0.05]
    "initial_planets":     list,      # planet positions at t=0, never mutated
    "next_fleet_id":       int,
    "comets":              list,      # list of group dicts (see below)
    "comet_planet_ids":    list[int],
    "step":                int,       # 0-indexed turn counter
    "remainingOverageTime":float,
}
```

### Comet group dict
```python
group = {
    "planet_ids":  [int, int, int, int],   # 4 symmetric planet IDs
    "paths":       [[[x,y], ...], ...],    # 4 pre-computed position paths
    "path_index":  int,                    # current position in path (-1 before first advance)
}
```

### Action format
```python
action = [from_planet_id, angle_radians, ships_count]
# agent returns: list of actions
```

---

## DataFrame Schemas

| name | columns |
|------|---------|
| `df_planets` | id, owner, x, y, radius, ships, production |
| `df_fleets`  | id, owner, x, y, angle, from_planet_id, ships |
| `df_actions` | planet_id, angle, ships_nb |

---

## Internal State (non-DataFrame fields)

```python
angular_velocity   float
_step              int    # current obs.step; incremented BEFORE planet rotation in next()
_player            int    # player ID this Board simulates for
_next_fleet_id     int
_initial_planets   dict[int → list]   # {planet_id: [id,owner,x,y,radius,ships,production]}
_comet_planet_ids  set[int]
_comets            list[dict]
```

---

## Key Constants

```python
CENTER_X = CENTER_Y = 50.0
BOARD_SIZE = 100.0
SUN_RADIUS = 10.0
ROTATION_RADIUS_LIMIT = 50.0   # orbital_radius + radius < 50 → planet rotates
MAX_SPEED = 6.0
PLANET_MARGIN = 0.1            # fleet spawns at radius + 0.1 from planet center
```

---

## Key Formulas

### Fleet speed
```python
def fleet_speed(ships, max_speed=6.0):
    if ships <= 1: return 1.0
    ratio = math.log(ships) / math.log(1000.0)
    ratio = max(0.0, min(1.0, ratio))
    return 1.0 + (max_speed - 1.0) * (ratio ** 1.5)
```

### Planet orbital rotation (absolute formula, not iterative)
```python
dx, dy = initial_x - 50, initial_y - 50
orbital_r = math.sqrt(dx**2 + dy**2)
initial_angle = math.atan2(dy, dx)
current_angle = initial_angle + angular_velocity * step   # step = self._step after increment
planet_x = 50 + orbital_r * math.cos(current_angle)
planet_y = 50 + orbital_r * math.sin(current_angle)
```

### Fleet spawn position
```python
start_x = planet_x + math.cos(angle) * (planet_radius + 0.1)
start_y = planet_y + math.sin(angle) * (planet_radius + 0.1)
```

### Continuous collision detection (segment-to-point)
```python
from kaggle_environments.envs.orbit_wars.orbit_wars import point_to_segment_distance
# hit if: point_to_segment_distance(planet_pos, old_fleet_pos, new_fleet_pos) < planet_radius
```

---

## Game Step Order (from interpreter(), line 455+)

```
0. Fleet Launch      — validate & spawn fleets from df_actions
1. Production        — owned planets gain planet[6] ships
2. Fleet Movement    — move by fleet_speed(); continuous collision detection
3. Planet Movement   — rotate orbiting planets; sweep stationary fleets; advance comets
4. Combat Resolution — resolve all fleet arrivals at planets simultaneously
```

### Combat resolution detail
```python
player_ships = {owner: sum_of_arriving_ships}
sorted_desc = sorted(player_ships.items(), key=lambda x: x[1], reverse=True)
top_owner, top_ships = sorted_desc[0]
if len(sorted_desc) > 1:
    survivor = top_ships - sorted_desc[1][1]
    if sorted_desc[0][1] == sorted_desc[1][1]: survivor = 0
    winner = top_owner if survivor > 0 else -1
else:
    winner, survivor = top_owner, top_ships

if survivor > 0:
    if planet_owner == winner:
        garrison += survivor
    else:
        garrison -= survivor
        if garrison < 0:
            planet_owner = winner
            garrison = abs(garrison)
```

---

## Implementation Approach for `next()`

**Do NOT reimplement the physics.** Use the real kaggle interpreter via state injection:

```python
# In next():
obs0 = self._env.state[0].observation
obs0.planets        = self.df_planets.values.tolist()
obs0.fleets         = self.df_fleets.values.tolist()
obs0.angular_velocity = self.angular_velocity
obs0.initial_planets = list(self._initial_planets.values())
obs0.next_fleet_id  = self._next_fleet_id
obs0.comet_planet_ids = list(self._comet_planet_ids)
obs0.comets         = self._comets
obs0.step           = self._step

actions = self.df_actions[["planet_id","angle","ships_nb"]].values.tolist()
self._env.step([actions, []])   # player 0 actions; empty for player 1

# Read back
new_obs = self._env.state[0].observation
self.obs_2_df(new_obs)          # re-parses into DataFrames + updates internal state
```

`Board.__init__` creates and holds a private `self._env = make("orbit_wars")`.
The env is used purely as a physics engine; its internal step counter stays in sync via `obs0.step`.

---

## API Summary

```python
board = Board()                          # creates internal kaggle env
board.obs_2_df(obs)                      # load state from any obs (dict or SimpleNamespace)
board.df_2_obs()         -> dict         # export current state as obs dict
board.add_action([pid, angle, ships])    # queue one move for next step
board.next()                             # advance one step (consumes + clears df_actions)
```

---

## Verification Sketch

```python
from kaggle_environments import make
from game import Board

env = make("orbit_wars", debug=True)
env.reset()
obs = env.state[0].observation

board = Board()
board.obs_2_df(obs)

# Round-trip
obs2 = board.df_2_obs()
assert len(obs2["planets"]) == len(obs.planets)

# Step both real env and Board with no actions; compare planet positions
env.step([[], []])
board.next()
real_planets = {p[0]: (p[2], p[3]) for p in env.state[0].observation.planets}
board_planets = {row.id: (row.x, row.y) for row in board.df_planets.itertuples()}
for pid, pos in real_planets.items():
    bpos = board_planets[pid]
    assert abs(pos[0]-bpos[0]) < 1e-9 and abs(pos[1]-bpos[1]) < 1e-9, f"Planet {pid} mismatch"
```
