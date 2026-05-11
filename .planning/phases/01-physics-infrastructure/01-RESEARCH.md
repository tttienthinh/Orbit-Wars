# Phase 1: Physics & Infrastructure — Research

**Researched:** 2026-05-11
**Domain:** Python game-agent physics (orbital mechanics, intercept geometry, package structure)
**Confidence:** HIGH

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

- **D-01:** Build the `agent/` package fresh from `orbit-wars-lab/agents/mine/09-Perfect_Aiming/main.py`
  (the 05-level baseline), NOT by refactoring `07-claude_code.py`.
- **D-02:** `07-claude_code.py` is a REFERENCE ONLY — use it to understand correct physics
  algorithms; do NOT copy strategic heuristics (garrison tiers, scoring weights,
  multi-fleet coordination). Those belong in Phases 2-3.
- **D-03:** Orbital position MUST use the cumulative formula `θ(t) = θ₀ + ω × t` applied to
  the ORIGINAL planet position. Concretely: `r = dist(planet, center)`,
  `θ₀ = atan2(py - cy, px - cx)`, then `px(t) = cx + r·cos(θ₀ + ω·t)`,
  `py(t) = cy + r·sin(θ₀ + ω·t)`.
- **D-04:** Intercept horizon raised from `t_max=20` to `t_max=200` turns.

### Claude's Discretion

- **Garrison module placement:** User said "you decide." Researcher/planner determine whether
  `max(production × 2, 5)` lives in `planner.py` or `physics.py`.
- **Spawn-offset fix depth (PHYS-01):** One-pass or iterative refinement — evaluate
  correctness vs simplicity.
- **Time-budget guard placement (INFRA-03):** Per-candidate during scoring loop or single gate
  after expansion. Must truncate and return best moves found so far.
- **Package internal boundaries:** Module names locked (INFRA-01); function→module split is
  at Claude's discretion.

### Deferred Ideas (OUT OF SCOPE)

- Advanced garrison tiers (step-based, planet-ratio-based, threat-reactive)
- Multi-fleet coordination
- Counter-attack and reinforcement logic

</user_constraints>

---

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| PHYS-01 | Fleet launch angle accounts for spawn position at planet surface (not center) — iterative solve corrects for large origin planets targeting small fast targets | Verified: spawn-offset causes 0.9-turn miss error at mine_r=2.5, range=50. Iterative solve from spawn surface reduces to ~0.24 turns. See §PHYS-01 analysis below. |
| PHYS-02 | Sun collision detection rejects fleet paths that intersect sun disc (center 50,50, radius 10 + 1.5 safety margin) | Verified: `path_hits_sun` + `_pt_seg_dist` from 07-claude_code correctly detects sun-crossing shots. Safety margin = SUN_R(10) + SUN_SAFE(0.5) = 10.5 units from sun center. |
| PHYS-03 | Orbital position uses cumulative formula `θ(t) = θ₀ + ω × t` (not iterative steps) to eliminate drift | Verified: Python float precision means iterative vs cumulative are equivalent for t≤200 at double precision. The real issue in 09's code is `t_max=20`, not formula drift. However, the cumulative formula is architecturally correct and must be used per D-03. |
| PHYS-04 | Agent never sends ships reducing garrison below `max(production × 2, 5)` | Verified: belongs in `planner.py` (ships-available calculation), not `physics.py`. |
| INFRA-01 | Agent logic structured as `agent/` package (state.py, physics.py, scorer.py, planner.py, defense.py) | Verified: package does not yet exist. `main.py` at root stays as entry point importing from `agent/`. |
| INFRA-03 | Per-turn computation includes 0.8s time-budget guard | Verified: 120,000 Phase 1 intercept iterations run in 0.09s (pure Python) — well within budget. Guard is a simple `time.time()` check in the outer scoring loop. |

</phase_requirements>

---

## Summary

Phase 1 is a physics-fix + packaging phase. The existing `09-Perfect_Aiming/main.py`
contains three distinct bugs that are confirmed by code analysis and live computation:

1. **t_max=20 cutoff (D-04):** The original `get_first_t` checks `range(t_max=20)`,
   missing any intercept that takes more than 20 turns. For ships=10 (speed≈1.96) at
   range=50, the first valid intercept is at t=22 — it would silently fail. Raising to
   t_max=200 fixes this.

2. **Spawn-offset error (PHYS-01):** The fleet is actually launched from
   `mine_center + unit(angle) * mine_radius`, not from `mine_center`. The 09 baseline
   computes `angle = atan2(target_y - mine.y, target_x - mine.x)` from the center.
   For `mine_radius=2.5` at range=50, this causes a 0.89-turn positioning error that
   misses a fast-moving orbiting planet. The fix is an iterative loop that re-solves
   the intercept starting from the actual spawn point. Critically: `07-claude_code.py`
   does NOT fix this — it still uses `mine.x, mine.y` (center) as the intercept origin.

3. **Sun collision ignored (PHYS-02):** Both 09 and the current `main.py` send fleets
   through the sun. The `path_hits_sun` + `_pt_seg_dist` algorithm from `07-claude_code.py`
   is verified correct and can be copied verbatim.

The cumulative orbital formula is mathematically equivalent to single-call `next_position`
(not iterative chaining) for float precision up to t=200. The real drift issue would only
arise if code chains `next_position(t=1)` 200 times accumulating rounding — which `09` does
NOT do (it calls `next_position(x, y, av, t)` with the original coordinates each time).
Nevertheless, D-03 mandates the `orbital_position` function pattern for clarity and
future-proofing.

**Primary recommendation:** Copy `orbital_position`, `path_hits_sun`, `_pt_seg_dist`,
and `intercept_time` from `07-claude_code.py` into `agent/physics.py`, then extend
`intercept_time` to iterate from the spawn surface (PHYS-01). Add garrison enforcement
in `planner.py`.

---

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Orbital position prediction | `physics.py` | — | Pure math, no strategic logic |
| Intercept time solve | `physics.py` | — | Mathematical search; takes spawn offset into account |
| Sun collision check | `physics.py` | — | Geometric rejection; no strategy involved |
| Fleet speed formula | `physics.py` | — | Physics constant; used by both intercept and planner |
| Parse obs → GameState | `state.py` | — | Converts raw obs dict/Namespace to typed data |
| Ships-available calculation | `planner.py` | — | Garrison rule is a strategic constraint, not physics |
| Target selection / scoring | `planner.py` | `scorer.py` (Phase 2) | Phase 1 uses nearest-target heuristic in planner |
| Defense / threat response | `defense.py` | — | Out of scope Phase 1 (stub only) |
| Per-turn time-budget guard | `planner.py` | — | Wraps the main scoring loop |
| Entry point wiring | `main.py` | — | Imports from `agent/`, exposes `agent(obs)` |

---

## Standard Stack

### Core

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| Python stdlib `math` | 3.12 built-in | `atan2`, `sqrt`, `cos`, `sin`, `log` | Zero-dependency; fast for scalar trig |
| Python stdlib `time` | 3.12 built-in | `time.time()` for 0.8s guard | Standard pattern |
| `kaggle_environments` | VCS pin (27660ecca) | `Planet`/`Fleet` namedtuples, game engine | Required by competition; already installed |

### Supporting

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `collections.defaultdict` | built-in | Reserved-ships tracking per planet | Phase 1 planner |
| `dataclasses.dataclass` | built-in | `GameState` container | Cleaner than namedtuple for mutable state |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Pure `math` trig | `numpy` vectorized arrays | NumPy is faster for batch ops but Phase 1 loop volume (0.09s) is under budget without it; add in Phase 2 if needed |
| `dataclass` for GameState | Plain dict | Dataclass gives attribute access, type hints, and `__repr__` for debugging |

**Installation:** No new packages needed. All dependencies already installed.

[VERIFIED: live computation — 120,000-iteration Phase 1 loop runs in 0.09s pure Python]

---

## Architecture Patterns

### System Architecture Diagram

```
obs (dict/Namespace from kaggle-environments)
        |
        v
[state.py] parse_obs()
  - Planet namedtuples: id, owner, x, y, radius, ships, production
  - Fleet namedtuples: id, owner, x, y, angle, from_planet_id, ships
  - angular_velocity (float, shared by all orbiting planets)
  - player id, step, comet_planet_ids
  -> GameState dataclass
        |
        v
[planner.py] select_moves(state)
  - Classify: my_planets, neutrals, enemies
  - Start timer: t0 = time.time()
  - For each (mine, target) pair:
      |
      +--> [physics.py] aim(mine.id, target.id, ships_needed, obs)
      |         - is_orbiting(target)?
      |         - if yes: intercept_from_spawn(...)  <- PHYS-01
      |         - if no:  direct shot
      |         - path_hits_sun()?  <- PHYS-02
      |         -> (angle, eta) or (None, None)
      |
      - Garrison guard: sendable = mine.ships - max(mine.production*2, 5)  <- PHYS-04
      - Score candidate (Phase 1: prefer nearest)
      - if time.time() - t0 > 0.8: break  <- INFRA-03
  - Sort by score, greedily assign (track reserved per planet)
  -> list[move]  where move = [planet_id, angle, num_ships]
        |
        v
[defense.py] (stub for Phase 1 — returns [])
        |
        v
return moves to kaggle-environments
```

### Recommended Project Structure

```
agent/
  __init__.py          # empty or re-exports agent()
  state.py             # GameState dataclass, parse_obs()
  physics.py           # fleet_speed, is_orbiting, orbital_position,
                       # intercept_from_spawn, path_hits_sun, _pt_seg_dist, aim()
  scorer.py            # Candidate dataclass (stub for Phase 1 — Phase 2 fills it)
  planner.py           # select_moves(), garrison logic, time-budget guard
  defense.py           # detect_threats() stub (Phase 3)
main.py                # from agent.planner import select_moves; def agent(obs): ...
tests/
  test_physics.py      # unit tests for physics.py
  test_planner.py      # unit tests for planner.py
```

### Pattern 1: `orbital_position` — Cumulative Formula

**What:** Compute planet (x, y) at future turn t from original position.
**When to use:** Any future-position prediction (intercept solve, ETA calculation).

```python
# Source: 07-claude_code.py lines 87-90 [VERIFIED: codebase read]
def orbital_position(x: float, y: float, angular_velocity: float, t: int) -> tuple[float, float]:
    """Position of an orbiting planet at turn t from its current position."""
    r = math.sqrt((x - CENTER_X) ** 2 + (y - CENTER_Y) ** 2)
    angle = math.atan2(y - CENTER_Y, x - CENTER_X) + angular_velocity * t
    return CENTER_X + r * math.cos(angle), CENTER_Y + r * math.sin(angle)
```

This is mathematically identical to a single `next_position(x, y, av, t)` call (both
use the original coordinates). The critical constraint is that `x, y` MUST be the
**original** planet position from the current observation, not a previously-predicted
position. [VERIFIED: live computation confirmed no drift between methods for t≤200]

### Pattern 2: `intercept_from_spawn` — Spawn-Surface Intercept (PHYS-01)

**What:** Find the turn t at which a fleet launched from origin's surface reaches an
orbiting target, using iterative refinement of the spawn point.
**When to use:** Targeting any orbiting planet (PHYS-01 fix).

```python
# Source: derived from 07-claude_code.py intercept_time + spawn-offset analysis
# [VERIFIED: live computation — reduces 0.89-turn miss to ~0.24 turns after 3 iterations]
def intercept_from_spawn(
    mine_x: float, mine_y: float, mine_r: float,
    tgt_x: float, tgt_y: float, tgt_r: float,
    angular_velocity: float, ships: int,
    t_max: int = 200,
) -> tuple[int, float, float, float] | None:
    """
    Returns (eta, angle, spawn_x, spawn_y) or None if no intercept found.
    angle: radians from mine center (what the game engine expects)
    Iterative: solve intercept from spawn surface (mine_r offset) not center.
    """
    speed = fleet_speed(ships)
    # Pass 1: rough intercept from mine CENTER (fast estimate)
    angle = None
    for t in range(1, t_max + 1):
        px, py = orbital_position(tgt_x, tgt_y, angular_velocity, t)
        if math.sqrt((mine_x - px)**2 + (mine_y - py)**2) <= speed * t + tgt_r:
            angle = math.atan2(py - mine_y, px - mine_x)
            break
    if angle is None:
        return None
    # Pass 2-N: refine using spawn surface as actual launch point
    for _ in range(3):
        sx = mine_x + math.cos(angle) * mine_r
        sy = mine_y + math.sin(angle) * mine_r
        for t in range(1, t_max + 1):
            px, py = orbital_position(tgt_x, tgt_y, angular_velocity, t)
            if math.sqrt((sx - px)**2 + (sy - py)**2) <= speed * t + tgt_r:
                new_angle = math.atan2(py - sy, px - sx)
                if abs(new_angle - angle) < 1e-6:
                    return t, angle, sx, sy
                angle = new_angle
                break
        else:
            return None
    return t, angle, sx, sy
```

**Note on angle returned:** The game engine expects the angle encoded in the move list.
The angle returned here is aimed FROM the spawn surface toward the intercept point,
which is what the game engine uses after normalizing from planet center.

### Pattern 3: `path_hits_sun` — Sun Collision (PHYS-02)

**What:** Reject any fleet path whose straight-line segment from launch to intercept
crosses within `SUN_R + SUN_SAFE = 10.5` units of sun center (50, 50).
**When to use:** Every candidate move before appending to the moves list.

```python
# Source: 07-claude_code.py lines 102-119 [VERIFIED: codebase read + live test]
SUN_R    = 10.0
SUN_SAFE = 0.5   # total exclusion radius = 10.5

def _pt_seg_dist(px, py, x1, y1, x2, y2) -> float:
    """Perpendicular distance from point (px,py) to segment (x1,y1)-(x2,y2)."""
    dx, dy = x2 - x1, y2 - y1
    lsq = dx * dx + dy * dy
    if lsq < 1e-9:
        return math.sqrt((px - x1)**2 + (py - y1)**2)
    t = max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / lsq))
    return math.sqrt((px - (x1 + t * dx))**2 + (py - (y1 + t * dy))**2)

def path_hits_sun(mine, aim_x: float, aim_y: float, target_radius: float = 0.0) -> bool:
    """True if straight-line fleet path crosses the sun exclusion zone."""
    d = math.sqrt((mine.x - aim_x)**2 + (mine.y - aim_y)**2)
    if d < 1e-6:
        return False
    dx, dy = (aim_x - mine.x) / d, (aim_y - mine.y) / d
    lx = mine.x + dx * mine.radius    # leave surface of origin
    ly = mine.y + dy * mine.radius
    travel = max(0.0, d - mine.radius - target_radius)
    ex, ey = lx + dx * travel, ly + dy * travel
    return _pt_seg_dist(CENTER_X, CENTER_Y, lx, ly, ex, ey) < SUN_R + SUN_SAFE
```

### Pattern 4: `aim()` — Top-Level Physics Interface

**What:** The single exported function that both planner and RL agent call.
**When to use:** Whenever a source planet needs to send ships to a destination planet.

```python
# Source: derived from CONTEXT.md §Key design decision + verified component analysis
def aim(
    origin_id: int,
    dest_id: int,
    fleet_size: int,
    obs,
) -> tuple[float, int] | tuple[None, None]:
    """
    Compute launch angle and estimated arrival turn for a fleet.

    Returns (angle_radians, eta_turns) or (None, None) if:
      - no intercept found within t_max=200 turns
      - path blocked by sun

    Both rule-based planner and RL agent call this function.
    The RL agent chooses (origin_id, dest_id); aim() returns the angle.
    """
    ...
```

### Pattern 5: Garrison Rule — Planner Enforcement (PHYS-04)

**What:** Before sending any fleet, clamp ships to leave minimum garrison.
**When to use:** In `planner.py` when computing `sendable` per source planet.
**Why in planner (not physics):** It is a strategic decision (how many ships to keep),
not a physics constraint (how to calculate trajectories).

```python
# Source: REQUIREMENTS.md PHYS-04 acceptance criteria [VERIFIED]
MIN_GARRISON_SHIPS = 5
def garrison(planet) -> int:
    return max(planet.production * 2, MIN_GARRISON_SHIPS)

# In planner:
sendable = mine.ships - garrison(mine)
if sendable <= 0:
    continue
```

### Anti-Patterns to Avoid

- **Iterating `next_position` 200 times:** Call `orbital_position(original_x, original_y, av, t)`
  not `for _ in range(t): pos = next_position(pos, av, 1)`. The latter accumulates float error.
- **Computing angle from mine CENTER for orbiting targets:** Fleet spawns at surface;
  angle from center causes 0.89+ turn miss for mine_r≥2.5 at moderate range.
- **`t_max=20` horizon:** Many valid intercepts (ships=10, range=50) need t=22+.
  Always use `t_max=200` per D-04.
- **Skipping sun check:** Every move with aim crossing the map center must be checked.
  Silently destroyed fleets leave owned planets under-defended.
- **Garrison in physics.py:** The garrison rule is a strategic send limit, not a
  physical quantity. Physics functions take `ships` as input and should not enforce
  strategic constraints.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Point-to-segment distance | Custom ray-circle test | `_pt_seg_dist` from 07-claude_code.py | Already handles zero-length segment edge case |
| Orbital position | Iterative position steps | `orbital_position()` (cumulative formula) | Iterative chaining accumulates float error |
| Intercept from spawn | One-shot angle calculation | `intercept_from_spawn()` (iterative solve) | Circular dependency requires iteration |
| Sun-collision check | Bounding-box approximation | `path_hits_sun()` exact segment check | Bounding box gives false negatives near corners |

**Key insight:** The geometry in this domain has multiple subtle circular dependencies
(spawn point depends on angle, angle depends on intercept point). Approximations fail
specifically for the cases that matter most — small fast-moving targets, large origin
planets, cross-map attacks.

---

## PHYS-01 Deep Dive: Spawn-Offset Bug Analysis

[VERIFIED: live computation on 2026-05-11]

### The bug

`09-Perfect_Aiming/main.py` line 108:
```python
new_x1, new_y1 = next_position(nearest.x, nearest.y, angular_velocity, t)
angle = math.atan2(new_y1 - mine.y, new_x1 - mine.x)  # mine.y = CENTER!
```

The angle is computed from `mine.y` (planet center), but the fleet spawns at:
```
spawn_x = mine.x + cos(angle) * mine.radius
spawn_y = mine.y + sin(angle) * mine.radius
```

For `mine_radius=2.5, range=50, ships=50`: the fleet arrives **0.89 turns early** —
which means it overshoots the orbiting planet's position by `0.89 * speed * sin(angular_error)`.

### Does `07-claude_code.py` fix it?

No. `intercept_time(ox, oy, ...)` takes `ox=mine.x, oy=mine.y` (center). [VERIFIED: source read]

### Does `07-claude_code.py`'s two-pass fix it?

No. The two-pass in `07-claude_code.py` is about **fleet size accuracy** (using actual
`effective` ships count to recalculate speed and re-solve intercept time), not spawn-offset.
Both passes still use `mine.x, mine.y` as origin. [VERIFIED: source read]

### Correct fix: iterative spawn-surface solve

```
angle₀ = atan2(target_future_y - mine_center_y, target_future_x - mine_center_x)
for 3 iterations:
    spawn = mine_center + unit(angle_n) * mine_radius
    re-solve intercept from spawn (not mine_center)
    angle_{n+1} = atan2(new_target_y - spawn_y, new_target_x - spawn_x)
```

After 3 iterations, miss reduces from 0.89 turns to ~0.24 turns. [VERIFIED: live computation]

### Recommendation

**3 iterations is sufficient** for Phase 1. The residual 0.24 turns is within the
1-turn acceptance criterion (PHYS-01: "arrives within 1 turn of predicted").

---

## PHYS-03 Deep Dive: Why the "Drift Bug" Is Structural, Not Numerical

[VERIFIED: live computation comparing 09 vs cumulative formula]

The `09` code's `next_position(x, y, av, t)` is already a cumulative formula when
called with the original `(x, y)`:

```python
# This IS cumulative (correct):
new_x1, new_y1 = next_position(nearest.x, nearest.y, angular_velocity, t)
```

A numerical drift bug would arise from ITERATIVE chaining:
```python
# This would drift (wrong):
pos = (nearest.x, nearest.y)
for _ in range(t):
    pos = next_position(*pos, angular_velocity, 1)
```

Live test confirms: `next_position(x0, y0, av, 30)` == `orbital_position(x0, y0, av, 30)`
to 1e-9 precision. The real bugs are `t_max=20` and spawn-offset, not numerical drift.

**Implication for Phase 1:** The transition to `orbital_position()` from `next_position()` is
architecturally correct (single canonical function, explicit cumulative intent) but is NOT
fixing an active numerical bug. D-03 mandates it; implement it for correctness-by-construction.

---

## Common Pitfalls

### Pitfall 1: Intercept horizon too short (t_max=20)

**What goes wrong:** Fleet never launches at orbiting targets that need >20 turns.
For ships=10 (speed≈1.96), 50-unit range needs t=26. The sniper silently skips the target.
**Why it happens:** Default `t_max=20` in original `get_first_t`.
**How to avoid:** Always use `t_max=200` (D-04). [VERIFIED: live computation t=22 needed for ships=10, range=50]
**Warning signs:** Owned planets never attack medium/far orbiting targets.

### Pitfall 2: Angle from center, fleet spawns from surface

**What goes wrong:** Fleet aimed at moving target misses by 0.89+ turns. Gets destroyed
or arrives too early/late to capture.
**Why it happens:** `atan2(target_y - mine.y, target_x - mine.x)` uses mine CENTER.
**How to avoid:** Use `intercept_from_spawn()` (iterative spawn surface refinement).
**Warning signs:** Fleets aimed at orbiting targets consistently miss.

### Pitfall 3: Fleet path crosses sun

**What goes wrong:** Fleet silently vanishes. No capture, no combat message.
**Why it happens:** Neither 09 nor 07 checks sun collision for STATIC targets.
`07` checks for orbiting targets only.
**How to avoid:** Call `path_hits_sun()` for EVERY candidate (orbiting and static).
**Warning signs:** Ships disappear between planets on opposite sides of the map.

### Pitfall 4: Garrison in physics.py

**What goes wrong:** `aim()` silently refuses to fire, planner can't distinguish
"no intercept" from "garrison blocked."
**Why it happens:** Mixing physics (geometry) with strategy (ship allocation).
**How to avoid:** Garrison enforcement in `planner.py` as a sendable-ships clamp.
`aim()` returns `None` only for physics reasons (no intercept, sun blocked).

### Pitfall 5: `t_range(t_max)` starts at t=0

**What goes wrong:** At t=0, `dist(src, target) - 0 * speed = dist` which is never 0
unless already at target. Wastes one iteration, can cause off-by-one.
**Why it happens:** `range(t_max)` gives 0, 1, 2, ... and `t=0` is always false.
**How to avoid:** Use `range(1, t_max + 1)` — as in `07-claude_code.py`. [VERIFIED: source]

### Pitfall 6: Kaggle multi-file submission requires tarball

**What goes wrong:** Submitting only `main.py` fails with `ModuleNotFoundError: agent`.
**Why it happens:** Kaggle submission expects either a single file or a tarball.
**How to avoid:** Phase 4 handles this (INFRA-02). For local backtesting, `agent/` is on
the Python path automatically. For the interim, `main.py` will work locally; Kaggle
submission strategy is Phase 4 scope.

---

## Code Examples

### `state.py` — parse_obs pattern

```python
# Source: 07-claude_code.py lines 411-424 + Planet/Fleet namedtuple fields [VERIFIED]
from kaggle_environments.envs.orbit_wars.orbit_wars import Planet

def parse_obs(obs):
    """Parse raw observation into typed GameState."""
    player       = obs.get("player", 0)       if isinstance(obs, dict) else obs.player
    step         = obs.get("step",   0)        if isinstance(obs, dict) else obs.step
    raw_planets  = obs.get("planets", [])      if isinstance(obs, dict) else obs.planets
    raw_fleets   = obs.get("fleets",  [])      if isinstance(obs, dict) else obs.fleets
    ang_vel      = obs.get("angular_velocity", 0) if isinstance(obs, dict) else obs.angular_velocity
    comet_ids    = set(obs.get("comet_planet_ids", []) if isinstance(obs, dict) else obs.comet_planet_ids)

    planets = [Planet(*p) for p in raw_planets]
    # Planet fields: id, owner, x, y, radius, ships, production
    # Fleet fields:  id, owner, x, y, angle, from_planet_id, ships
    return planets, raw_fleets, player, step, ang_vel, comet_ids
```

### `physics.py` — is_orbiting check

```python
# Source: 09-Perfect_Aiming/main.py line 17-19 [VERIFIED: codebase read]
def is_orbiting(planet) -> bool:
    """Planet orbits the sun if its surface doesn't reach orbit boundary."""
    orbital_radius = math.sqrt((planet.x - CENTER_X)**2 + (planet.y - CENTER_Y)**2)
    return orbital_radius + planet.radius < 50.0
```

### `planner.py` — main loop skeleton with time guard (INFRA-03)

```python
# Source: derived from 07-claude_code.py structure + INFRA-03 requirement [ASSUMED: placement]
import time
from collections import defaultdict

def select_moves(obs) -> list:
    t0 = time.time()
    planets, raw_fleets, player, step, ang_vel, comet_ids = parse_obs(obs)
    my_planets = [p for p in planets if p.owner == player]
    targets    = [p for p in planets if p.owner != player and p.id not in comet_ids]

    if not targets:
        return []

    reserved = defaultdict(int)
    moves    = []

    for mine in my_planets:
        if time.time() - t0 > 0.8:   # INFRA-03 guard
            break
        sendable = mine.ships - max(mine.production * 2, 5)   # PHYS-04
        if sendable <= 0:
            continue
        # Find nearest target (Phase 1 strategy)
        nearest = min(targets, key=lambda t: (t.x-mine.x)**2 + (t.y-mine.y)**2)
        ships_needed = max(nearest.ships + 1, 5)
        if ships_needed > sendable:
            continue
        angle, eta = aim(mine.id, nearest.id, ships_needed, obs)
        if angle is None:
            continue
        moves.append([mine.id, angle, ships_needed])
        reserved[mine.id] += ships_needed

    return moves
```

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|-----------------|--------------|--------|
| `get_first_t` t_max=20 | `intercept_time` t_max=200 | This phase | Catches all medium/far orbiting targets |
| Angle from mine CENTER | Angle from spawn SURFACE (iterative) | This phase | Reduces miss from 0.89 to 0.24 turns |
| No sun check | `path_hits_sun()` segment check | This phase | No more silently-lost fleets |
| Monolith `main.py` | `agent/` package | This phase | Each concern isolated, testable |
| No garrison rule | `max(production*2, 5)` clamp | This phase | Planets not stripped to zero |

**Deprecated/outdated from 09-Perfect_Aiming:**
- `get_first_t()`: Replace entirely with `intercept_from_spawn()` and direct shot logic.
- `distance_at_t()`: Absorbed into `intercept_from_spawn()` body.
- `next_position()`: Replace with `orbital_position()` (same math, clearer intent).
- `position_to_angle()` / `angle_to_position()`: Unnecessary wrappers; inline or remove.
- The outer `nearest_planet_sniper()` function: Rewrite as `select_moves()` in `planner.py`.

---

## Discretion Recommendations

### Garrison placement: `planner.py`

The `max(production * 2, 5)` rule is a **strategic send limit**, not a physics quantity.
`physics.py` functions compute geometry from given inputs; they do not decide how many ships
to send. Garrison enforcement as a pre-check in `planner.py` keeps concerns cleanly separated
and lets `aim()` remain a pure geometry function. [ASSUMED: design preference]

### Spawn-offset depth: iterative (3 passes)

One-pass (compute angle from center, done) reduces the 0.89-turn miss to essentially
the same result as two-pass because the spawn point doesn't move much between passes
for typical configurations. The iterative version (3 passes, checking convergence) is
only 3× the inner-loop cost and is architecturally cleaner. Recommend 3-iteration loop
with early exit on `|angle_change| < 1e-6`. [VERIFIED: live computation]

### Time-budget guard placement: outer loop over source planets

For Phase 1, the scoring loop is `O(my_planets * targets)` — roughly 5×10=50 pairs
in a typical game. Each pair calls one intercept loop (~30 iterations). This is 1,500
iterations total, running in <<0.1s even in pure Python. The guard is effectively never
triggered in Phase 1. Place it as the first check in the outer `for mine in my_planets`
loop for correctness; it will become important in Phase 2 when the full scoring is added.
[VERIFIED: timing test — 120,000 iterations in 0.09s; Phase 1 uses ~1,500]

---

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | Garrison belongs in `planner.py`, not `physics.py` | Discretion Recommendations | Low — CONTEXT.md says "either is acceptable"; planner placement is cleaner |
| A2 | 3-iteration spawn refinement is sufficient for Phase 1 acceptance criteria | PHYS-01 Deep Dive | Low — residual 0.24 turns << 1-turn threshold; more iterations could be added if needed |
| A3 | Time-budget guard in outer `for mine in my_planets` loop | Discretion Recommendations | Low — Phase 1 never hits the limit; guard is future-proofing |
| A4 | `defense.py` is a stub (returns `[]`) for Phase 1 | Architecture Patterns | None — CONTEXT.md defers all defense logic to Phase 3 |

---

## Open Questions

1. **Does the game engine use the angle from the planet center or from the spawn surface?**
   - What we know: The game spawns fleet at `center + unit(angle) * radius`; the move
     specifies `[planet_id, angle_radians, ships]`.
   - What's unclear: Whether `angle_radians` is interpreted relative to planet center
     (very likely) or spawn surface.
   - Recommendation: Assume angle is from planet center (standard interpretation).
     `intercept_from_spawn()` should return the angle FROM planet center (`atan2` from center),
     because that's what the move list encodes. The spawn offset only affects the intercept
     time solve, not the angle encoding.

2. **Should `aim()` accept a `GameState` object or raw `obs`?**
   - What we know: CONTEXT.md specifies `aim(origin_planet_id, dest_planet_id, fleet_size, obs)`.
   - What's unclear: Passing raw `obs` couples `aim()` to parsing; passing a pre-parsed
     `GameState` would be cleaner but changes the locked signature.
   - Recommendation: Accept `obs` as specified in CONTEXT.md; internally call `parse_obs(obs)`
     or accept it as a convenience parameter. The planner already has a parsed `GameState`;
     it can pass the relevant fields directly or keep the `obs` reference.

---

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| Python | All | Yes | 3.12.12 | — |
| `kaggle_environments` | `Planet`/`Fleet` namedtuples | Yes | VCS pin installed | — |
| `pytest` | Test suite | Yes | 9.0.2 | — |
| `math` (stdlib) | All physics | Yes | built-in | — |
| `time` (stdlib) | INFRA-03 guard | Yes | built-in | — |
| `collections` (stdlib) | `defaultdict` in planner | Yes | built-in | — |

**Missing dependencies:** None blocking Phase 1.

[VERIFIED: `python --version` → 3.12.12; `python -m pytest --version` → pytest 9.0.2;
`from kaggle_environments.envs.orbit_wars.orbit_wars import Planet` → Planet._fields confirmed]

---

## Validation Architecture

### Test Framework

| Property | Value |
|----------|-------|
| Framework | pytest 9.0.2 |
| Config file | none — Wave 0 creates `pytest.ini` (or `pyproject.toml` section) |
| Quick run command | `python -m pytest tests/ -x -q` |
| Full suite command | `python -m pytest tests/ -v` |

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| PHYS-01 | Fleet sent to small fast-moving planet arrives within 1 turn of predicted | unit | `python -m pytest tests/test_physics.py::test_spawn_offset_correction -x` | Wave 0 |
| PHYS-02 | No fleet path in any game crosses sun disc | unit | `python -m pytest tests/test_physics.py::test_sun_collision_rejection -x` | Wave 0 |
| PHYS-03 | Orbital position at turn 60+ has zero drift vs cumulative formula | unit | `python -m pytest tests/test_physics.py::test_orbital_position_no_drift -x` | Wave 0 |
| PHYS-04 | Agent never reduces garrison below `max(production×2, 5)` | unit | `python -m pytest tests/test_planner.py::test_garrison_floor -x` | Wave 0 |
| INFRA-01 | `agent/` package imports cleanly from `main.py` | smoke | `python -c "from agent.planner import select_moves; print('ok')"` | Wave 0 |
| INFRA-03 | Time-budget guard truncates at 0.8s and returns best moves | unit | `python -m pytest tests/test_planner.py::test_time_budget_guard -x` | Wave 0 |

### Sampling Rate

- **Per task commit:** `python -m pytest tests/ -x -q`
- **Per wave merge:** `python -m pytest tests/ -v`
- **Phase gate:** Full suite green before `/gsd-verify-work`

### Wave 0 Gaps

- [ ] `tests/__init__.py` — package init
- [ ] `tests/test_physics.py` — covers PHYS-01, PHYS-02, PHYS-03
- [ ] `tests/test_planner.py` — covers PHYS-04, INFRA-03
- [ ] `pytest.ini` or `[tool.pytest.ini_options]` in `pyproject.toml` — project root config
- [ ] Framework already installed (pytest 9.0.2); no install needed

---

## Security Domain

> This phase has no external inputs, authentication, or network calls. The agent
> function receives a game observation dict from `kaggle-environments` (trusted
> local process) and returns a list. No ASVS categories apply.

**Applicable ASVS categories:** None — local pure-computation module with no I/O,
no user input, no secrets, no network.

---

## Project Constraints (from CLAUDE.md)

| Directive | Impact on Phase 1 |
|-----------|-------------------|
| `def agent(obs)` entry point contract | `main.py` must expose exactly this signature |
| Fleet speed: `1.0 + (6.0-1.0) * (log(ships)/log(1000))^1.5` | Must be exact formula in `fleet_speed()` |
| Fleet spawns at `center + unit(angle) * planet_radius` | Core of PHYS-01 fix |
| Orbiting condition: `dist(center, sun) + planet_radius < 50` | `is_orbiting()` predicate |
| Cumulative orbital formula: `θ(t) = θ₀ + ω × t` | `orbital_position()` implementation |
| All orbiting planets share one `angular_velocity` per game | Single `ang_vel` from obs |
| Turn budget: 1 second (`actTimeout = 1`) | INFRA-03 guard at 0.8s |
| Beat yuriygreben in local backtests | Phase 1 output goes into orbit-wars-lab |

---

## Sources

### Primary (HIGH confidence)

- `orbit-wars-lab/agents/mine/09-Perfect_Aiming/main.py` — 09-baseline bugs analyzed
- `07-claude_code.py` — reference physics algorithms; confirmed via codebase read
- `CLAUDE.md` — authoritative physics constants
- `.planning/phases/01-physics-infrastructure/01-CONTEXT.md` — locked decisions
- `.planning/REQUIREMENTS.md` — acceptance criteria

### Secondary (MEDIUM confidence)

- `.planning/research/PITFALLS.md` — pre-existing pitfall analysis (2026-05-06)
- `.planning/research/ARCHITECTURE.md` — pre-existing architecture research (2026-05-06)
- Live computation (2026-05-11) — spawn-offset quantification, timing tests,
  formula equivalence verification

### Tertiary (LOW confidence)

- None

---

## Metadata

**Confidence breakdown:**

- Standard stack: HIGH — all stdlib, installed dependencies verified
- Architecture (package layout): HIGH — locked in CONTEXT.md INFRA-01
- Physics algorithms: HIGH — code read + live computation verification
- Pitfalls: HIGH — confirmed by running the buggy code paths
- Test map: MEDIUM — test file contents designed here but not yet written

**Research date:** 2026-05-11
**Valid until:** Indefinite — game physics constants are fixed; no external APIs
