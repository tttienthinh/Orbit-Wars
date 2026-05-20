# 29-Score.py — Agent Logic

Credit: based on [djenkivanov's proto agent](https://www.kaggle.com/code/djenkivanov/orbit-wars-agent-ow-proto-passed-1-000).

---

## Architecture overview

```
agent(obs)
├── Initialisation guard (skip first 2 steps, detect moving planets on step 2)
├── Refresh local observation (namedtuples)
├── Update trajectory trackers (fleets sent, reinforcements in flight)
├── Detect threats  ──────────────────── get_planets_under_attack()
├── Plan defence    ──────────────────── get_reinforcement_plans()
│     └── For each endangered planet → find nearest friendly → send reinforcement
└── Plan offence
      └── For each friendly planet (richest first)
            ├── Score every target       get_custom_score()
            ├── Single attack (if enough ships)
            └── Cooperative attack       (multiple planets pool ships)
```

---

## Global state

| Variable | Purpose |
|---|---|
| `fleet_trajectories` | Attacks already dispatched this game (tracked across turns) |
| `reinforcement_trajectories` | Defensive sends in flight |
| `moving_planets` | IDs of orbiting planets (detected at step 2) |
| `steps` | Turn counter; used for the init guard |

State persists across turns because the agent is a module-level object. Updated every turn via `update_fleet_trajectories` / `update_reinforcement_trajectories`.

---

## Key scoring function — `get_custom_score(m, t)`

Scores how attractive target `t` is for planet `m` to attack.

```
score = (FORMULA_DIST - dist)           # +close targets
      + (FORMULA_PROD_MULT * t.prod)    # +high production
      + (FORMULA_ENEMY_BONUS_MULT * t.prod if t is enemy)  # +extra value for flipping enemy
      - (FORMULA_TOTAL_SHIPS_PERCENT * ships_needed)        # −cost to take
      - (2 * eta)                        # −slow arrivals
```

`eta` and `ships_needed` account for production that accrues while the fleet is in transit.

---

## Moving planet support

Orbiting planets require intercept aiming rather than straight-line angles.

- **`fill_moving_planets(obs)`** — compares current vs initial positions at step 2 to build `moving_planets`.
- **`get_planet_trajectories(p, vel)`** — projects a planet's circular orbit for 60 ticks.
- **`find_angle_to_moving_planet(p, t, ships, vel)`** — iterates over the target's future positions to find the earliest tick where a fleet of given speed can intercept; returns `(angle, tick)` or `(None, None)` if unreachable or sun-blocked.

---

## Threat detection — `get_planets_under_attack`

For every enemy fleet currently on the board, simulates its straight-line path tick by tick (up to 60 ticks) and records any friendly planet it will hit, along with the arrival tick.

Moving friendly planets use pre-computed trajectories; static ones use their current position.

---

## Defensive logic — `get_reinforcement_plans`

For each threatened planet, simulates the timeline of incoming attacks **in arrival order**, subtracting enemy ships and adding production + already-planned reinforcements. If the planet goes negative at any attack, it is flagged as needing reinforcement.

The planner then searches nearby friendly planets for one that:
- is not already exhausted (sent an attack this turn)
- has enough ships after accounting for its own threats and already-planned sends
- can arrive before the threat does

---

## Offensive logic

### Ship requirement calculation
- **Static target:** `calculate_req_ships` — iterates up to 3 times, computing how much production the defender accrues during transit, until the required ship count converges.
- **Moving target:** `calculate_req_ships_moving` — same loop but uses per-attacker intercept ticks.

### Single attack
If the attacking planet alone has enough ships, fire a single fleet. Moving targets use `find_angle_to_moving_planet`; static targets use `calculate_angle`. Sun collision is always checked with `sun_collision`.

### Cooperative (coop) attack
If one planet is short, up to `COOP_PLANET_CAP = 8` nearby safe planets pool ships. Each contributing planet fires its own fleet. Coop is only triggered when the target has ≥ `MIN_SHIPS_TARGET_COOP_ATTACK = 20` ships (worth the coordination overhead).

### "Release all havoc" mode
When the agent controls ≥ 75 % of planets, it stops skipping already-targeted planets (the `en_route >= needed_now` gate is relaxed), flooding the remaining targets.

---

## Constants

| Constant | Value | Effect |
|---|---|---|
| `MIN_SHIPS_MINE_ATTACK` | 5 | Minimum fleet size sent from any planet |
| `MIN_SHIPS_TARGET_COOP_ATTACK` | 20 | Minimum target ships to trigger coop |
| `COOP_PLANET_CAP` | 8 | Max contributors in a coop attack |
| `FORMULA_DIST` | 100 | Distance baseline in scoring |
| `FORMULA_PROD_MULT` | 15 | Weight on target production |
| `FORMULA_ENEMY_BONUS_MULT` | 10 | Extra weight for flipping enemy planets |
| `FORMULA_TOTAL_SHIPS_PERCENT` | 0.7 | Penalty per ship needed to capture |

---

## Known limitations / ideas

- `get_planet_trajectories` uses `current_angle + vel * tick` — correct for single-step lookahead but misses the step-0 rotation offset found in `23-next10.ipynb`.
- Reinforcement only sends one rescue fleet per endangered planet per turn.
- Score ignores friendly fleets already en route to the same target (only `en_route` count is checked, not their ETA vs production).
- Global mutable state (`fleet_trajectories`, etc.) means the agent is **not re-entrant** — running two instances in the same process corrupts both.
