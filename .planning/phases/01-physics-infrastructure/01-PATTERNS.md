# Phase 1: Physics & Infrastructure - Pattern Map

**Mapped:** 2026-05-11
**Files analyzed:** 10
**Analogs found:** 10 / 10

---

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|---|---|---|---|---|
| `agent/__init__.py` | package-init | — | `orbit-wars-lab/agents/external/kashiwaba-rl/src/__init__.py` | exact |
| `agent/state.py` | model | transform | `orbit-wars-lab/agents/external/kashiwaba-rl/src/game_types.py` | exact |
| `agent/physics.py` | utility | transform | `07-claude_code.py` lines 67-119 | exact |
| `agent/scorer.py` | utility | transform | `07-claude_code.py` lines 242-267 (stub only Phase 1) | role-match |
| `agent/planner.py` | service | request-response | `07-claude_code.py` lines 411-520 + `09-Perfect_Aiming/main.py` lines 67-114 | exact |
| `agent/defense.py` | service | event-driven | `07-claude_code.py` lines 193-235 (stub only Phase 1) | role-match |
| `main.py` (root) | entry-point | request-response | `main.py` (current root), `09-Perfect_Aiming/main.py` | exact |
| `tests/__init__.py` | test-init | — | `orbit-wars-lab/tests/__init__.py` | exact |
| `tests/test_physics.py` | test | transform | `orbit-wars-lab/tests/unit/test_discovery.py` | role-match |
| `tests/test_planner.py` | test | request-response | `orbit-wars-lab/tests/unit/test_discovery.py` | role-match |

---

## Pattern Assignments

### `agent/__init__.py` (package-init)

**Analog:** `orbit-wars-lab/agents/external/kashiwaba-rl/src/__init__.py` (line 1)

**Pattern:** Either empty or re-exports the top-level `agent()` entry point. The kashiwaba-rl package re-exports its public API. For Phase 1, an empty file is acceptable; an explicit re-export is better for callers.

**Imports / re-export pattern** (analog lines 1-3):
```python
from .planner import select_moves

__all__ = ["select_moves"]
```

---

### `agent/state.py` (model, transform)

**Analog:** `orbit-wars-lab/agents/external/kashiwaba-rl/src/game_types.py` (lines 1-72) — exact match: dataclass GameState, parse_observation pattern, dict/Namespace dual-access.

**Imports pattern** (analog lines 1-5):
```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Any
```

**Core dataclass pattern** (analog lines 7-31):
```python
@dataclass(slots=True)
class PlanetState:
    id: int
    owner: int
    x: float
    y: float
    radius: float
    ships: int
    production: int


@dataclass(slots=True)
class FleetState:
    id: int
    owner: int
    x: float
    y: float
    angle: float
    from_planet_id: int
    ships: int


@dataclass(slots=True)
class GameState:
    step: int
    player: int
    planets: list[PlanetState]
    fleets: list[FleetState]
```

**parse_observation pattern** (analog lines 37-72) — dual dict/Namespace access, row-indexed construction:
```python
def parse_observation(observation: Any) -> GameState:
    def obs_get(key: str, default: Any) -> Any:
        if isinstance(observation, dict):
            return observation.get(key, default)
        return getattr(observation, key, default)

    planets = [
        PlanetState(
            id=int(row[0]), owner=int(row[1]),
            x=float(row[2]), y=float(row[3]),
            radius=float(row[4]), ships=int(row[5]),
            production=int(row[6]),
        )
        for row in obs_get("planets", [])
    ]
    fleets = [
        FleetState(
            id=int(row[0]), owner=int(row[1]),
            x=float(row[2]), y=float(row[3]),
            angle=float(row[4]), from_planet_id=int(row[5]),
            ships=int(row[6]),
        )
        for row in obs_get("fleets", [])
    ]
    return GameState(
        step=int(obs_get("step", 0)),
        player=int(obs_get("player", 0)),
        planets=planets,
        fleets=fleets,
    )
```

**Adaptation notes for `agent/state.py`:**
- Add `angular_velocity` and `comet_planet_ids` fields to `GameState` (not present in kashiwaba-rl analog — Orbit Wars specific).
- Keep the `obs_get()` inner-function pattern — it handles both dict and Namespace obs formats from `kaggle-environments`.
- The `kaggle_environments` `Planet` namedtuple import used in `09-Perfect_Aiming/main.py` line 2 (`from kaggle_environments.envs.orbit_wars.orbit_wars import Planet`) is an alternative; the dataclass approach from kashiwaba-rl is preferred for Phase 1 because it gives typed attributes.

---

### `agent/physics.py` (utility, transform)

**Analog:** `07-claude_code.py` lines 67-119 — exact match for all four physics functions required.

**Imports pattern** (`07-claude_code.py` lines 22-27):
```python
import math
from collections import defaultdict
from kaggle_environments.envs.orbit_wars.orbit_wars import Planet
```
For `physics.py` only `math` is needed (no `defaultdict`, no `Planet`).

**Constants pattern** (`07-claude_code.py` lines 24-28):
```python
CENTER_X  = 50.0
CENTER_Y  = 50.0
SUN_R     = 10.0
SUN_SAFE  = 0.5
MAX_SPEED = 6.0
```

**`fleet_speed` pattern** (`07-claude_code.py` lines 79-84):
```python
def fleet_speed(ships):
    if ships <= 1:
        return 1.0
    ratio = math.log(max(ships, 1)) / math.log(1000.0)
    ratio = max(0.0, min(1.0, ratio))
    return 1.0 + (MAX_SPEED - 1.0) * (ratio ** 1.5)
```

**`is_orbiting` pattern** (`07-claude_code.py` lines 75-76):
```python
def is_orbiting(planet) -> bool:
    return math.sqrt((planet.x - CENTER_X)**2 + (planet.y - CENTER_Y)**2) + planet.radius < 50.0
```

**`orbital_position` pattern** (`07-claude_code.py` lines 87-90 — cumulative formula, D-03):
```python
def orbital_position(x: float, y: float, angular_velocity: float, t: int) -> tuple[float, float]:
    r     = math.sqrt((x - CENTER_X) ** 2 + (y - CENTER_Y) ** 2)
    angle = math.atan2(y - CENTER_Y, x - CENTER_X) + angular_velocity * t
    return CENTER_X + r * math.cos(angle), CENTER_Y + r * math.sin(angle)
```
CRITICAL: `x, y` must be the ORIGINAL planet position from the current observation, never a previously-predicted position.

**`_pt_seg_dist` helper** (`07-claude_code.py` lines 102-108):
```python
def _pt_seg_dist(px, py, x1, y1, x2, y2) -> float:
    dx, dy = x2 - x1, y2 - y1
    lsq    = dx * dx + dy * dy
    if lsq < 1e-9:
        return math.sqrt((px - x1)**2 + (py - y1)**2)
    t = max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / lsq))
    return math.sqrt((px - (x1 + t * dx))**2 + (py - (y1 + t * dy))**2)
```

**`path_hits_sun` pattern** (`07-claude_code.py` lines 111-119):
```python
def path_hits_sun(mine, aim_x: float, aim_y: float, target_radius: float = 0.0) -> bool:
    d = math.sqrt((mine.x - aim_x)**2 + (mine.y - aim_y)**2)
    if d < 1e-6:
        return False
    dx, dy  = (aim_x - mine.x) / d, (aim_y - mine.y) / d
    lx, ly  = mine.x + dx * mine.radius, mine.y + dy * mine.radius
    travel  = max(0.0, d - mine.radius - target_radius)
    ex, ey  = lx + dx * travel, ly + dy * travel
    return _pt_seg_dist(CENTER_X, CENTER_Y, lx, ly, ex, ey) < SUN_R + SUN_SAFE
```
Apply to EVERY candidate move — orbiting and static targets alike (PHYS-02).

**`intercept_from_spawn` pattern** — NEW for Phase 1, NOT in any analog (PHYS-01 fix):

The `07-claude_code.py` `intercept_time` (`lines 93-99`) is the base, but it uses mine center as origin. The Phase 1 fix adds iterative spawn-surface refinement:

```python
# Base from 07-claude_code.py lines 93-99:
def intercept_time(ox, oy, tx, ty, angular_velocity, ships, target_radius=0.0):
    speed = fleet_speed(ships)
    for t in range(1, INTERCEPT_LIMIT + 1):  # INTERCEPT_LIMIT = 200, not 20 (D-04)
        px, py = orbital_position(tx, ty, angular_velocity, t)
        if math.sqrt((ox - px)**2 + (oy - py)**2) <= speed * t + target_radius:
            return t, px, py
    return None, None, None

# Extension for PHYS-01 — iterative spawn-surface solve (new, no codebase analog):
def intercept_from_spawn(
    mine_x, mine_y, mine_r,
    tgt_x, tgt_y, tgt_r,
    angular_velocity, ships,
    t_max=200,
):
    """Returns (eta, angle_from_center, spawn_x, spawn_y) or None."""
    speed = fleet_speed(ships)
    # Pass 1: rough intercept from mine CENTER
    angle = None
    for t in range(1, t_max + 1):
        px, py = orbital_position(tgt_x, tgt_y, angular_velocity, t)
        if math.sqrt((mine_x - px)**2 + (mine_y - py)**2) <= speed * t + tgt_r:
            angle = math.atan2(py - mine_y, px - mine_x)
            break
    if angle is None:
        return None
    # Passes 2-4: refine from actual spawn surface
    for _ in range(3):
        sx = mine_x + math.cos(angle) * mine_r
        sy = mine_y + math.sin(angle) * mine_r
        for t in range(1, t_max + 1):
            px, py = orbital_position(tgt_x, tgt_y, angular_velocity, t)
            if math.sqrt((sx - px)**2 + (sy - py)**2) <= speed * t + tgt_r:
                new_angle = math.atan2(py - mine_y, px - mine_x)  # angle from CENTER
                if abs(new_angle - angle) < 1e-6:
                    return t, angle, sx, sy
                angle = new_angle
                break
        else:
            return None
    return t, angle, sx, sy
```
Note: The returned `angle` is `atan2` from the mine CENTER (not spawn surface), because the game engine encodes angles relative to planet center.

**`aim()` top-level interface pattern** — derived from `07-claude_code.py` `compute_shot` (lines 170-186):
```python
# compute_shot analog from 07-claude_code.py lines 170-186:
def compute_shot(mine, target, angular_velocity, actual_ships=None):
    speed_ships = actual_ships or max(target.ships + 1, 5)
    if is_orbiting(target):
        t, px, py = intercept_time(mine.x, mine.y, target.x, target.y,
                                   angular_velocity, speed_ships,
                                   target_radius=target.radius)
        if t is None:
            return None, None, None
        if path_hits_sun(mine, px, py):
            return None, None, None
        return target.ships, t, math.atan2(py - mine.y, px - mine.x)
    else:
        if path_hits_sun(mine, target.x, target.y, target_radius=target.radius):
            return None, None, None
        travel = max(1, int(math.sqrt((mine.x-target.x)**2+(mine.y-target.y)**2) /
                             fleet_speed(speed_ships)))
        return target.ships, travel, math.atan2(target.y - mine.y, target.x - mine.x)

# Phase 1 aim() replaces compute_shot — uses intercept_from_spawn for orbiting targets:
def aim(origin, dest, fleet_size, angular_velocity):
    """Returns (angle_radians, eta_turns) or (None, None)."""
    ...
```

---

### `agent/scorer.py` (utility, transform)

**Analog:** `07-claude_code.py` lines 242-267 (`score_attack`) — role-match only; Phase 1 is a stub.

**Phase 1 stub pattern:**
```python
# scorer.py — Phase 1 stub; full scoring implemented in Phase 2
# Nearest-distance heuristic lives in planner.py for Phase 1.
# This file is a placeholder to satisfy INFRA-01 package structure.
```

No concrete code to copy for Phase 1 — the file should exist but contain only the `Candidate` dataclass placeholder and a docstring noting Phase 2 expansion.

---

### `agent/planner.py` (service, request-response)

**Analog:** `07-claude_code.py` lines 411-520 (agent function) + `09-Perfect_Aiming/main.py` lines 67-114 (nearest_planet_sniper) — exact match for structure; Phase 1 uses nearest-target only (no scoring weights, no multi-fleet coordination).

**Imports pattern** (derived from both analogs):
```python
import time
import math
from collections import defaultdict
from agent.state import parse_observation, GameState
from agent.physics import aim, is_orbiting
```

**obs parsing pattern** (`07-claude_code.py` lines 412-421):
```python
player    = obs.get("player", 0) if isinstance(obs, dict) else obs.player
step      = obs.get("step",   0) if isinstance(obs, dict) else obs.step
raw_p     = obs.get("planets", []) if isinstance(obs, dict) else obs.planets
raw_f     = obs.get("fleets",  []) if isinstance(obs, dict) else obs.fleets
ang_vel   = obs.get("angular_velocity", 0) if isinstance(obs, dict) else obs.angular_velocity
comet_ids = set(obs.get("comet_planet_ids", []) if isinstance(obs, dict) else obs.comet_planet_ids)
```
Phase 1 delegates this to `parse_observation()` from `state.py`.

**Planet classification pattern** (`07-claude_code.py` lines 422-424):
```python
planets    = [Planet(*p) for p in raw_p]
my_planets = [p for p in planets if p.owner == player]
neutrals   = [p for p in planets if p.owner == -1 and p.id not in comet_ids]
enemies    = [p for p in planets if p.owner not in (-1, player) and p.id not in comet_ids]
```

**Garrison rule pattern (PHYS-04)** (`07-claude_code.py` uses `garrison_for()` — Phase 1 uses the simpler locked formula):
```python
# Phase 1 garrison — locked formula from REQUIREMENTS.md PHYS-04:
MIN_GARRISON = 5
def garrison(planet) -> int:
    return max(planet.production * 2, MIN_GARRISON)

# In loop:
sendable = mine.ships - garrison(mine)
if sendable <= 0:
    continue
```

**Time-budget guard pattern (INFRA-03)** (outer loop, from RESEARCH.md Pattern 5):
```python
t0 = time.time()
for mine in my_planets:
    if time.time() - t0 > 0.8:   # INFRA-03: truncate and return best so far
        break
    ...
```

**Nearest-target loop pattern** (`09-Perfect_Aiming/main.py` lines 82-111):
```python
# Phase 1 strategy: nearest unowned planet (copied from 09-Perfect_Aiming/main.py)
targets = [p for p in planets if p.owner != player and p.id not in comet_ids]

reserved = defaultdict(int)
moves    = []

for mine in my_planets:
    if time.time() - t0 > 0.8:
        break
    sendable = mine.ships - garrison(mine) - reserved[mine.id]
    if sendable <= 0:
        continue
    nearest = min(
        (t for t in targets),
        key=lambda t: (t.x - mine.x)**2 + (t.y - mine.y)**2,
        default=None,
    )
    if nearest is None:
        continue
    ships_needed = max(nearest.ships + 1, 5)
    if ships_needed > sendable:
        continue
    angle, eta = aim(mine, nearest, ships_needed, ang_vel)
    if angle is None:
        continue
    moves.append([mine.id, angle, ships_needed])
    reserved[mine.id] += ships_needed
```

**Entry-point wrapper** (`09-Perfect_Aiming/main.py` line 67, `main.py` root lines 22-24):
```python
def select_moves(obs) -> list:
    ...
    return moves

def agent(obs) -> list:
    return select_moves(obs)
```

---

### `agent/defense.py` (service, event-driven)

**Analog:** `07-claude_code.py` lines 193-235 (`plan_defense`) — role-match; Phase 1 is a stub only.

**Phase 1 stub pattern:**
```python
# defense.py — Phase 1 stub; full implementation in Phase 3
def detect_threats(state, reserved):
    """Detect incoming enemy fleets and generate defensive moves.
    Phase 1: returns empty list (no-op stub).
    Phase 3: multi-source defense pooling.
    """
    return []
```

---

### `main.py` (root) (entry-point, request-response)

**Analog:** Current `main.py` root (lines 1-60) and `09-Perfect_Aiming/main.py` (lines 1-4, 22-24).

**Pattern:** Keep root `main.py` minimal — import from `agent/` and expose the `def agent(obs)` contract. The current `main.py` is a self-contained monolith; after Phase 1 it becomes a thin wrapper.

**Imports pattern** (current `main.py` lines 1-2, adapted):
```python
"""Orbit Wars agent — Kaggle submission entry point."""
from agent.planner import select_moves


def agent(obs):
    return select_moves(obs)
```

**Note:** The docstring block (current `main.py` lines 1-16) provides competition context; keep a short version. The `kaggle_environments` import moves into `agent/state.py`.

---

### `tests/__init__.py` (test-init)

**Analog:** `orbit-wars-lab/tests/__init__.py` — empty file.

**Pattern:** Empty file. No re-exports needed.

---

### `tests/test_physics.py` (test, transform)

**Analog:** `orbit-wars-lab/tests/unit/test_discovery.py` — role-match (unit test structure, pytest conventions).

**File header pattern** (`test_discovery.py` lines 1-12):
```python
"""Tests for agent.physics — PHYS-01, PHYS-02, PHYS-03."""
from __future__ import annotations

import math
import pytest

from agent.physics import (
    orbital_position,
    path_hits_sun,
    intercept_from_spawn,
    fleet_speed,
    is_orbiting,
)
```

**Test function naming pattern** (`test_discovery.py` lines 24, 29, 46, 83):
```python
def test_orbital_position_no_drift():         # PHYS-03
def test_spawn_offset_correction():            # PHYS-01
def test_sun_collision_rejection():            # PHYS-02
def test_sun_collision_static_target():        # PHYS-02 static variant
```

**Assertion style** (from `test_discovery.py` and `test_schemas.py`):
- Use plain `assert` with descriptive messages, no `unittest.TestCase`
- Numeric comparisons use `abs(a - b) < tolerance` or `pytest.approx`
- No fixtures needed for pure-math unit tests

**PHYS-03 test pattern** (value test, no external state):
```python
def test_orbital_position_no_drift():
    """orbital_position with original coords must match cumulative formula exactly."""
    x0, y0, av = 60.0, 50.0, 0.05
    for t in [1, 10, 30, 60, 100, 200]:
        px, py = orbital_position(x0, y0, av, t)
        r = math.sqrt((x0 - 50.0)**2 + (y0 - 50.0)**2)
        angle = math.atan2(y0 - 50.0, x0 - 50.0) + av * t
        expected_x = 50.0 + r * math.cos(angle)
        expected_y = 50.0 + r * math.sin(angle)
        assert abs(px - expected_x) < 1e-9
        assert abs(py - expected_y) < 1e-9
```

**PHYS-01 test pattern** (spawn-offset quantification):
```python
def test_spawn_offset_correction():
    """intercept_from_spawn reduces miss vs center-based intercept."""
    # mine_r=2.5, tgt at (80,50) orbiting, ships=50 — known 0.89-turn miss without fix
    result = intercept_from_spawn(
        mine_x=20.0, mine_y=50.0, mine_r=2.5,
        tgt_x=80.0, tgt_y=50.0, tgt_r=1.5,
        angular_velocity=0.05, ships=50,
    )
    assert result is not None
    eta, angle, sx, sy = result
    assert eta <= 200
    # spawn point must be on surface of mine planet
    assert abs(math.sqrt((sx - 20.0)**2 + (sy - 50.0)**2) - 2.5) < 0.01
```

---

### `tests/test_planner.py` (test, request-response)

**Analog:** `orbit-wars-lab/tests/unit/test_discovery.py` — role-match.

**File header pattern**:
```python
"""Tests for agent.planner — PHYS-04, INFRA-03."""
from __future__ import annotations

import time
import pytest

from agent.planner import select_moves, garrison
```

**Test fixture pattern** — synthetic obs dict (no pytest fixture needed, inline dict):
```python
def _make_obs(my_planets, other_planets, step=0, ang_vel=0.0):
    """Build minimal obs dict for planner tests."""
    planets = []
    for p in my_planets + other_planets:
        planets.append([p["id"], p["owner"], p["x"], p["y"],
                        p["radius"], p["ships"], p["production"]])
    return {
        "player": 0,
        "step": step,
        "planets": planets,
        "fleets": [],
        "angular_velocity": ang_vel,
        "comet_planet_ids": [],
    }
```

**PHYS-04 garrison floor test**:
```python
def test_garrison_floor():
    """Agent never sends ships reducing garrison below max(production*2, 5)."""
    obs = _make_obs(
        my_planets=[{"id": 0, "owner": 0, "x": 20.0, "y": 50.0,
                     "radius": 2.0, "ships": 10, "production": 2}],
        other_planets=[{"id": 1, "owner": -1, "x": 80.0, "y": 50.0,
                        "radius": 1.5, "ships": 3, "production": 1}],
    )
    moves = select_moves(obs)
    # With ships=10, garrison=max(2*2,5)=5, sendable=5, needed=4 — should fire
    for move in moves:
        from_id, angle, num_ships = move
        if from_id == 0:
            assert num_ships <= 5  # sendable = ships - garrison = 10 - 5

def test_garrison_blocks_when_insufficient():
    """No move sent when ships <= garrison."""
    obs = _make_obs(
        my_planets=[{"id": 0, "owner": 0, "x": 20.0, "y": 50.0,
                     "radius": 2.0, "ships": 5, "production": 3}],
        other_planets=[{"id": 1, "owner": -1, "x": 80.0, "y": 50.0,
                        "radius": 1.5, "ships": 3, "production": 1}],
    )
    moves = select_moves(obs)
    # garrison = max(3*2, 5) = 6 > ships=5, sendable <= 0 — no moves
    assert moves == []
```

**INFRA-03 time-budget guard test**:
```python
def test_time_budget_guard(monkeypatch):
    """Time-budget guard truncates evaluation and returns best moves found so far."""
    call_count = [0]
    original_time = time.time

    def slow_time():
        call_count[0] += 1
        # Simulate budget exceeded after first planet check
        return original_time() + (1.0 if call_count[0] > 2 else 0.0)

    monkeypatch.setattr(time, "time", slow_time)
    obs = _make_obs(
        my_planets=[
            {"id": 0, "owner": 0, "x": 20.0, "y": 50.0, "radius": 2.0, "ships": 20, "production": 1},
            {"id": 2, "owner": 0, "x": 30.0, "y": 50.0, "radius": 2.0, "ships": 20, "production": 1},
        ],
        other_planets=[{"id": 1, "owner": -1, "x": 80.0, "y": 50.0,
                        "radius": 1.5, "ships": 3, "production": 1}],
    )
    moves = select_moves(obs)
    # Should not raise; returns list (possibly partial)
    assert isinstance(moves, list)
```

---

## Shared Patterns

### Obs Dual-Access (dict / Namespace)

**Source:** `07-claude_code.py` lines 412-419, `09-Perfect_Aiming/main.py` lines 69-73, `orbit-wars-lab/agents/external/kashiwaba-rl/src/game_types.py` lines 38-41

**Apply to:** `agent/state.py` `parse_observation()` function — centralizes this pattern so other modules never need to handle raw obs directly.

```python
# Canonical dual-access pattern — use obs_get() inner function:
def obs_get(key: str, default):
    if isinstance(observation, dict):
        return observation.get(key, default)
    return getattr(observation, key, default)
```

### Physics Constants Block

**Source:** `07-claude_code.py` lines 24-28

**Apply to:** `agent/physics.py` top of file (only file that uses these constants):
```python
CENTER_X  = 50.0
CENTER_Y  = 50.0
SUN_R     = 10.0
SUN_SAFE  = 0.5
MAX_SPEED = 6.0
INTERCEPT_LIMIT = 200  # D-04: raised from 20
```

### `range(1, t_max + 1)` Intercept Loop

**Source:** `07-claude_code.py` line 95 — avoids t=0 false-positive

**Apply to:** `agent/physics.py` all intercept loops:
```python
for t in range(1, INTERCEPT_LIMIT + 1):  # NOT range(t_max) which includes t=0
```

### Kaggle Move Format

**Source:** `09-Perfect_Aiming/main.py` lines 103, 111; `main.py` root line 58; `07-claude_code.py` lines 507, 508

**Apply to:** `agent/planner.py` move construction — always `[planet_id, angle_radians, num_ships]`:
```python
moves.append([mine.id, angle, ships_needed])
```

### Test File Header Convention

**Source:** `orbit-wars-lab/tests/unit/test_discovery.py` lines 1-11, `orbit-wars-lab/tests/unit/test_schemas.py` lines 1-13

**Apply to:** All `tests/test_*.py` files:
```python
"""Tests for <module> — <REQ-IDs covered>."""
from __future__ import annotations

import pytest

from agent.<module> import <symbols>
```

---

## No Analog Found

All files have analogs. No entries in this section.

---

## Deprecated Patterns (Do NOT Copy)

These patterns exist in the analogs but must NOT be used in Phase 1:

| Pattern | Source | Why Deprecated |
|---|---|---|
| `get_first_t` with `t_max=20` | `09-Perfect_Aiming/main.py` lines 58-62 | Misses intercepts requiring t>20; replace with `intercept_from_spawn` using `t_max=200` |
| `next_position` iteratively chained | `09-Perfect_Aiming/main.py` lines 46-50 | Replace with `orbital_position(original_x, original_y, av, t)` per D-03 |
| `distance_at_t` | `09-Perfect_Aiming/main.py` lines 53-55 | Absorbed into `intercept_from_spawn` body |
| `nearest_planet_sniper` as top-level function | `09-Perfect_Aiming/main.py` lines 67-114 | Rewrite as `select_moves()` in `planner.py` |
| `position_to_angle` / `angle_to_position` wrappers | `09-Perfect_Aiming/main.py` lines 34-43 | Inline `atan2`/`cos`/`sin` calls are clearer |
| `garrison_for(step, ships, planet_ratio, net_threat)` | `07-claude_code.py` lines 41-59 | Phase 2-3 complexity; Phase 1 uses `max(production*2, 5)` only |
| Multi-fleet coordination (`coordinated_attacks`) | `07-claude_code.py` lines 274-349 | Phase 2-3 scope |
| Frontier reinforcement (`plan_reinforcement`) | `07-claude_code.py` lines 356-404 | Phase 2-3 scope |
| `build_ledgers` / `_fleet_target` | `07-claude_code.py` lines 126-163 | Phase 2-3 scope |

---

## Metadata

**Analog search scope:** `C:\Users\trant\Documents\Programmation\Orbit Wars\` (repo root), `orbit-wars-lab/agents/`, `orbit-wars-lab/agents/external/kashiwaba-rl/src/`

**Files scanned:** `07-claude_code.py`, `09-Perfect_Aiming/main.py`, `main.py` (root), `orbit-wars-lab/agents/external/kashiwaba-rl/src/game_types.py`, `orbit-wars-lab/agents/external/kashiwaba-rl/src/__init__.py`, `orbit-wars-lab/tests/unit/test_discovery.py`, `orbit-wars-lab/tests/unit/test_schemas.py`, `orbit-wars-lab/tests/conftest.py`

**Pattern extraction date:** 2026-05-11
