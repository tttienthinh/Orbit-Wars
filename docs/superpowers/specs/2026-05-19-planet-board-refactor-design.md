# Planet + Board Refactor Design

**Date:** 2026-05-19  
**File:** `24-Rules_Target_big_prod.py`  
**Constraint:** No behaviour change — identical move output, identical game logic.

---

## Goal

Reorganise the `Planet` class so that:
1. Everything computed once is stored as an attribute — no redundant computation.
2. There is a clear distinction between module-level functions (pure math) and instance methods (state-bearing logic).
3. A new `Board` class owns the full game state and orchestrates the planet setup pipeline and move generation.
4. `Planet` becomes a focused data + physics object, easy to extend later.

---

## Module Level

Only one module-level function: `_fleet_speed(nb_ships)` — unchanged.

---

## Planet class

### `__init__` — two explicit blocks

```python
# Block 1: raw data from obs (immutable after construction)
self.id, self.owner, self.x, self.y, self.radius, self.ships, self.production
self.angular_velocity
self.is_moving   # computed once from x/y/radius
self.comet_planet_ids
self.is_comet    # computed once from id

# Block 2: computed state — declared here, populated later by Board
self.nearby_planets = []
self.nb_nearby_mine = None
self.nb_nearby_enemy = None
self.nb_nearby_neutral = None
self.nature = None
self.nexts = []   # list of dicts {id, owner, x, y, radius, ships, production}
```

### Setup methods (called by Board in sequence)

| Method | Reads | Writes |
|--------|-------|--------|
| `compute_neighbors(planets_no_comets)` | planet list | `nearby_planets` (sorted by ETA) |
| `compute_counts()` | `nearby_planets` | `nb_nearby_mine`, `nb_nearby_enemy`, `nb_nearby_neutral` |
| `assign_nature()` | `nb_nearby_*` counts | `nature` |

### Physics methods (unchanged, stay on Planet)

- `predict_position(t)`
- `_find_intercept(tgt, speed, t_max=100)`
- `aiming(tgt, nb_ships)` → returns `(angle, eta)` or `None`

### Removed from Planet

- `create_one_action` → moves to `Board._create_one_action(planet, target)`
- `create_actions` → moves to `Board._create_actions(planet)`

---

## Board class

```python
class Board:
    def __init__(self, obs):
        # 1. Extract raw obs fields
        self.player = ...
        self.angular_velocity = ...
        self.comet_planet_ids = ...
        self.fleets = obs.fleets

        # 2. Build planets
        self.planets = [Planet(*p, ...) for p in obs.planets]
        self.planets_no_comets = [p for p in self.planets if not p.is_comet]
        self.planets_dico = {p.id: p for p in self.planets}
        self.my_planets = [p for p in self.planets if p.owner == self.player]
        self.targets = [p for p in self.planets_no_comets if p.owner != self.player]

        # 3. Forward simulation — populate planet.nexts (10 env.step calls total)
        self._run_simulation(obs)

        # 4. Planet setup pipeline
        for p in self.my_planets:
            p.compute_neighbors(self.planets_no_comets)
            p.compute_counts()
            p.assign_nature()

    def _run_simulation(self, obs):
        # env setup + copy obs state + 10x env.step + distribute nexts to planets_dico

    def get_moves(self) -> list:
        # replaces the agent() loop

    def _create_actions(self, planet) -> list:
        # replaces Planet.create_actions — planet passed explicitly

    def _create_one_action(self, planet, target, ships_needed=None) -> list:
        # replaces Planet.create_one_action — calls planet.aiming() for physics
```

---

## agent() entry point

```python
def agent(obs):
    return Board(obs).get_moves()
```

---

## Responsibility map

| Concern | Owner |
|---------|-------|
| Pure speed math | `_fleet_speed` (module) |
| Raw planet data | `Planet.__init__` block 1 |
| Cached computed state | `Planet.__init__` block 2 (declared), Board (populated) |
| Physics (position, intercept, aiming) | `Planet` methods |
| Planet setup pipeline | `Board.__init__` |
| Forward simulation | `Board._run_simulation` |
| Move generation | `Board.get_moves`, `Board._create_actions`, `Board._create_one_action` |
| Entry point | `agent(obs)` → `Board(obs).get_moves()` |

---

## What does NOT change

- All game logic, thresholds, and conditions are identical.
- `Planet.aiming()` is still called from `Board._create_one_action` — physics unchanged.
- The 10-step sim loop runs exactly as before, just encapsulated in `Board._run_simulation`.
- `num_agents` inference from `max_planet_id` is unchanged, moves to `Board._run_simulation`.
