# Design: `33-Kaggle_env_nearest_10steps.py`

## Goal

A self-contained Kaggle agent that improves on the `16-...py` nearest-planet sniper by
forward-simulating 10 steps before deciding. Only fires from planets that remain consistently
mine for the full simulation window; aims at the target that is enemy/neutral at step 10 with
the smallest ETA; sends exactly `target.ships_at_step10 + 1` ships.

---

## File Structure

Sections in order within the single `.py` file:

1. **Imports** — `math`, `copy`, `pandas`, `collections.namedtuple`
2. **Constants** — `CENTER=50.0`, `SUN_RADIUS=10.0`, `MAX_SPEED=6.0`, `NB_STEPS_SIM=10`
3. **`interpreter(obs, actions, step, num_agents)`** — copied verbatim from notebook
   `32-board_from_kaggle.ipynb` cell 0
4. **`_fleet_speed(ships)`** — speed formula
5. **`_simulate(obs, global_step, num_agents, n_steps) → pd.DataFrame`**
6. **`_eta(src_x, src_y, src_r, tgt, angular_velocity, ships=1) → int`**
7. **`_aim_angle(src_x, src_y, src_r, tgt, angular_velocity, ships) → float`**
8. **`global_board = {"step": 0, "num_agents": None}`**
9. **`nearest_planet_sniper(obs, global_board=global_board) → list`**
10. **`agent = nearest_planet_sniper`** — Kaggle entry point

---

## Data Flow

```
obs (current state)
  └─► deep copy ──► interpreter × NB_STEPS_SIM ──► df_planets (DataFrame)
                                                         │
obs.planets (current) ──► consistently_mine_ids ────────┤
                      ──► target_ids_at_final ───────────┤
                                                         │
                                               for each source planet:
                                                 find nearest target by _eta()
                                                 ships_needed = df[target, final_step].ships + 1
                                                 if source.ships >= ships_needed:
                                                     angle = _aim_angle()
                                                     append [id, angle, ships_needed]
```

---

## `df_planets` Schema

Built by `_simulate()`. One row per (planet, simulated step).

| column       | type    | notes                                              |
|--------------|---------|----------------------------------------------------|
| `step`       | int     | matches real env step: `global_step + i + 1`       |
| `id`         | int     | planet id                                          |
| `x`          | float   | position after this step                           |
| `y`          | float   | position after this step                           |
| `owner`      | int     | -1 neutral, 0/1/2/3 player                         |
| `ships`      | float   | garrison after this step                           |
| `production` | float   | production rate (constant)                         |
| `radius`     | float   | planet radius (constant)                           |
| `nature`     | str     | `"comet"`, `"moving"`, or `"fix"`                  |

Step labels: if current env step is `s`, snapshots are labeled `s+1` … `s+NB_STEPS_SIM`.
Snapshot is taken **after** each interpreter call so `step == s+NB_STEPS_SIM` is the fully
simulated final state.

**Orbital angle note:** `θ(step) = θ₀ + ω × (step - 1)` — step 1 = no rotation yet.
The interpreter receives the real env step so this is handled automatically.

---

## `_simulate(obs, global_step, num_agents, n_steps=NB_STEPS_SIM)`

```
deep copy obs
no_actions = [[] for _ in range(num_agents)]
for i in 0..n_steps-1:
    interpreter(sim, no_actions, global_step + i, num_agents)
    snapshot each planet → append row with step = global_step + i + 1
return pd.DataFrame(rows)
```

Nature is derived from geometry each snapshot:
- `pid in sim.comet_planet_ids` → `"comet"`
- `hypot(x-50, y-50) + radius < 50` → `"moving"`
- else → `"fix"`

---

## `_eta(src_x, src_y, src_r, tgt, angular_velocity, ships=1) → int`

`tgt` is a raw planet list from `obs.planets` (current positions).

- Static planet (`hypot(tx-50,ty-50) + tr >= 50`): straight-line distance ÷ speed, ceil.
- Moving planet: iterate `t = 1..100`, compute intercept position, return first `t` where
  `dist(src, intercept) - t*speed < src_r + tgt_r`. Return `9999` if no intercept found.

---

## `_aim_angle(src_x, src_y, src_r, tgt, angular_velocity, ships) → float`

Same intercept loop as `_eta`, returns `atan2` toward the intercept point.
Falls back to direct angle if no intercept found (shouldn't happen for reachable targets).

---

## `nearest_planet_sniper()` Logic

```
s = global_board["step"]

# step 0: detect num_agents from initial_planets owner ids
if s == 0:
    owners = {p[1] for p in obs.initial_planets if p[1] != -1}
    global_board["num_agents"] = 4 if len(owners) > 2 else 2

final_step = s + NB_STEPS_SIM
df = _simulate(obs, s, num_agents)

consistently_mine_ids =
    df.query("owner == @player")
      .groupby("id").filter(lambda g: len(g) == NB_STEPS_SIM)["id"]
      → set

target_ids_at_final =
    df.query("step == @final_step and owner != @player")["id"]
    → set

current = {p[0]: p for p in obs.planets}

for each source planet (owner == player, id in consistently_mine_ids):
    best_target = argmin _eta over target_ids_at_final ∩ current
    ships_needed = df.query("id == @tid and step == @final_step")["ships"].iloc[0] + 1
    if source.ships >= ships_needed:
        angle = _aim_angle(...)
        moves.append([source.id, angle, ships_needed])

global_board["step"] += 1
return moves
```

---

## Key Constraints

- No minimum ship floor — fire whenever `source.ships >= ships_needed`
- One action per source planet (nearest target only)
- No cross-planet coordination (each planet decides independently)
- Comets are never source planets (they won't be consistently mine); they appear in df_planets
  but are filtered out of `target_ids_at_final` only if they happen to be owned by the player

---

## Out of Scope

- Multi-target coordination / overkill avoidance
- Comet interception logic
- Supplier / Explorer / Conqueror role assignment (that's `25-Board_refactor.py`)
