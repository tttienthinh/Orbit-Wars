# OrbitWarsSimulator — Design Spec

Date: 2026-05-20  
File: `27-Board_new_env.py`

---

## Problem

`Board._run_simulation` uses `ke.make` to run a 20-step forward sim from the current game state.
The approach tries to inject the live game state by overwriting `env.state[i].observation.*`
fields (planets, fleets, initial_planets, etc.) after `env.reset()`. This does not work:
`env.step()` calls `__loop_through_interpreter` which runs `structify(action_state)` internally,
rebuilding the state from the framework's own copy rather than the user-modified attributes.
Every `env.step()` call therefore runs the interpreter against the randomly-generated game
created at `env.reset()` time — the injected state is silently discarded.

Removing ke.make from the simulation path also eliminates framework overhead per agent turn.

---

## Solution

A standalone `OrbitWarsSimulator` class that ports the relevant physics from the orbit_wars
interpreter directly. No kaggle_environments dependency in the hot path.

---

## Architecture

```
OrbitWarsSimulator(obs)
    ├── __init__      deep-copy all mutable state from obs
    ├── step()        run one physics step; return planet snapshot list
    └── run(n)        call step() n times; return list of snapshots

Board._run_simulation(obs)
    └── calls OrbitWarsSimulator(obs).run(NB_FORECAST_STEPS)
        and feeds each snapshot into planets_dico[id].nexts
```

The class lives in `27-Board_new_env.py`, defined above `Board`.

---

## Data Structures

All state is stored as plain Python lists (same format as the interpreter):

| Attribute | Type | Description |
|---|---|---|
| `planets` | `list[list]` | `[id, owner, x, y, radius, ships, production]` — mutable |
| `initial_planets` | `list[list]` | Same structure, frozen at current game positions |
| `initial_by_id` | `dict[int, list]` | Planet id → initial planet list (for rotation formula) |
| `fleets` | `list[list]` | `[id, owner, x, y, angle, from_planet_id, ships]` — mutable |
| `comets` | `list[dict]` | Each: `{planet_ids, paths, path_index}` — deep-copied |
| `comet_pid_set` | `set[int]` | Fast comet membership test |
| `angular_velocity` | `float` | Shared rotation rate |
| `sim_step` | `int` | Starts at 0; incremented at top of each `step()` call |

**Key invariant:** `initial_planets` always holds the positions at sim construction time
(= current game state). The orbital formula `initial_angle + ω * sim_step` therefore
produces the correct position at `game_step + sim_step` for any `sim_step ≥ 1`.

---

## Physics — `step()` execution order

Mirrors the interpreter exactly. Each call increments `sim_step` first.

### 1. Comet expiry (pre-step cleanup)
Remove comets whose `path_index` already equals or exceeds their path length.
Remove the corresponding planet entries from `self.planets` and `self.comet_pid_set`.

### 2. Production
```python
for planet in self.planets:
    if planet[1] != -1:   # owned
        planet[5] += planet[6]
```

### 3. Fleet movement + continuous collision
For each fleet:
- `speed = 1.0 + (MAX_SPEED - 1.0) * (log(ships) / log(1000))^1.5`, capped at MAX_SPEED
- Advance position by `cos(angle)*speed`, `sin(angle)*speed`
- Use `_pt_seg_dist(planet_pos, old_pos, new_pos) < planet_radius` to detect hits
- If hit: add fleet to that planet's combat list, mark for removal
- If out of bounds or crosses sun: mark for removal

### 4. Planet rotation + sweep
For each non-comet planet:
```python
if math.hypot(initial_p[2] - CENTER, initial_p[3] - CENTER) + planet[4] < 50:
    θ = atan2(initial_p[3] - CENTER, initial_p[2] - CENTER) + ω * sim_step
    planet[2] = CENTER + r * cos(θ)
    planet[3] = CENTER + r * sin(θ)
```
After updating position: sweep any fleet whose current position falls within the planet's
swept arc (`_pt_seg_dist(fleet_pos, old_pos, new_pos) < planet_radius`).

### 5. Comet movement
For each active comet group:
```python
group["path_index"] += 1
idx = group["path_index"]
planet[2] = group["paths"][i][idx][0]
planet[3] = group["paths"][i][idx][1]
```
If `idx >= len(path)`: mark comet planet for expiry, remove immediately.
Sweep fleets caught by comet movement (skip sweep on first placement, old_pos is off-board).

### 6. Combat resolution
Per planet with arriving fleets:
- Sum ships per player from all arriving fleets
- Sort descending; winner = top player; survivors = top − second
- If tie: 0 survivors, planet stays neutral
- Apply to planet: if same owner → add survivors; if different → subtract, flip if negative

### 7. Snapshot
Return `[{id, owner, x, y, radius, ships, production}, ...]` for all current planets.

---

## Helper

```python
@staticmethod
def _pt_seg_dist(p, v, w) -> float:
```
Minimum distance from point `p` to segment `v→w`. Direct port from the interpreter's
`point_to_segment_distance`. Used for both fleet-hits-planet and fleet-caught-by-sweep.

---

## Integration with Board

```python
def _run_simulation(self, obs):
    for snap in OrbitWarsSimulator(obs).run(NB_FORECAST_STEPS):
        for planet_state in snap:
            pid = planet_state["id"]
            if pid in self.planets_dico:
                self.planets_dico[pid].nexts.append(planet_state)
```

`Planet.nexts` shape is unchanged: list of dicts with keys `id, owner, x, y, radius, ships, production`.

---

## Limitations

- **New comet spawns are not predicted.** The episode seed (needed to generate spawn paths)
  is hidden from agents in `env.info`. Spawns occur at steps 50/150/250/350/450. Within ~10
  steps of a spawn, a new comet may appear at an unknown position. This affects `nexts`
  accuracy only in those windows.
- **No new fleet actions.** The sim assumes all players send empty action lists, matching
  the current ke.make usage (`[[]] * num_agents`).

---

## Files Changed

| File | Change |
|---|---|
| `27-Board_new_env.py` | Add `OrbitWarsSimulator` class above `Board`; replace `Board._run_simulation` with 4-line wrapper |

No new files. No changes to `Planet`, `Board`, or decision logic.
