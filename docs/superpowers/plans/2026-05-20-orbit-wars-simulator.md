# OrbitWarsSimulator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace `Board._run_simulation`'s broken `ke.make` forward sim with a standalone `OrbitWarsSimulator` class that ports physics directly from the orbit_wars interpreter.

**Architecture:** Single class `OrbitWarsSimulator` added above `Board` in `27-Board_new_env.py`. Constructor deep-copies all mutable game state from `obs`. `step()` runs one physics tick in interpreter order and returns a planet snapshot list. `Board._run_simulation` becomes a 4-line wrapper.

**Tech Stack:** Python stdlib only (`math`). Physics ported from `C:\Users\trant\miniforge3\Lib\site-packages\kaggle_environments\envs\orbit_wars\orbit_wars.py`. Tests use `pytest`.

---

## File Map

| File | Action | Purpose |
|---|---|---|
| `27-Board_new_env.py` | Modify | Add `OrbitWarsSimulator` above `Board`; replace `_run_simulation` |
| `test_simulator.py` | Create | Unit tests for every physics phase |

---

## Task 1: Test infrastructure + `_pt_seg_dist` + `__init__`

**Files:**
- Create: `test_simulator.py`
- Modify: `27-Board_new_env.py` (add class skeleton + `_pt_seg_dist` + `__init__`)

- [ ] **Step 1.1: Create `test_simulator.py` with helper and first two tests**

```python
# test_simulator.py
import math
from types import SimpleNamespace
from Board_new_env_27 import OrbitWarsSimulator  # import adjusted in step 1.3


def make_obs(planets, fleets=None, angular_velocity=0.05,
             comet_planet_ids=None, comets=None):
    obs = SimpleNamespace()
    obs.angular_velocity = angular_velocity
    obs.planets = [list(p) for p in planets]
    obs.fleets = [list(f) for f in (fleets or [])]
    obs.comet_planet_ids = list(comet_planet_ids or [])
    obs.comets = list(comets or [])
    return obs


# --- _pt_seg_dist ---

def test_pt_seg_dist_point_on_segment():
    assert OrbitWarsSimulator._pt_seg_dist((5, 0), (0, 0), (10, 0)) == 0.0


def test_pt_seg_dist_perpendicular():
    dist = OrbitWarsSimulator._pt_seg_dist((5, 3), (0, 0), (10, 0))
    assert abs(dist - 3.0) < 1e-9


def test_pt_seg_dist_past_end():
    dist = OrbitWarsSimulator._pt_seg_dist((15, 0), (0, 0), (10, 0))
    assert abs(dist - 5.0) < 1e-9


def test_pt_seg_dist_degenerate_segment():
    # v == w → distance to point v
    dist = OrbitWarsSimulator._pt_seg_dist((3, 4), (0, 0), (0, 0))
    assert abs(dist - 5.0) < 1e-9


# --- __init__ ---

def test_init_deep_copies_planets():
    obs = make_obs([[0, -1, 60.0, 50.0, 2.0, 10, 3]])
    sim = OrbitWarsSimulator(obs)
    obs.planets[0][5] = 999          # mutate original
    assert sim.planets[0][5] == 10   # sim unaffected


def test_init_deep_copies_fleets():
    fleet = [0, 0, 55.0, 50.0, 0.0, -1, 5]
    obs = make_obs([[0, 0, 60.0, 50.0, 2.0, 10, 1]], fleets=[fleet])
    sim = OrbitWarsSimulator(obs)
    obs.fleets[0][6] = 999
    assert sim.fleets[0][6] == 5


def test_init_builds_initial_by_id():
    obs = make_obs([[7, -1, 70.0, 50.0, 2.0, 5, 1]])
    sim = OrbitWarsSimulator(obs)
    assert 7 in sim.initial_by_id
    assert sim.initial_by_id[7][2] == 70.0


def test_init_comet_pid_set():
    obs = make_obs([[0, -1, 60.0, 50.0, 1.0, 5, 1]], comet_planet_ids=[0])
    sim = OrbitWarsSimulator(obs)
    assert 0 in sim.comet_pid_set


def test_init_sim_step_zero():
    obs = make_obs([[0, -1, 60.0, 50.0, 2.0, 5, 1]])
    sim = OrbitWarsSimulator(obs)
    assert sim.sim_step == 0
```

- [ ] **Step 1.2: Run tests — expect ImportError (class not yet defined)**

```
pytest test_simulator.py -v
```
Expected: `ImportError` or `ModuleNotFoundError` (file exists, class missing).

- [ ] **Step 1.3: Add the class skeleton + `_pt_seg_dist` + `__init__` to `27-Board_new_env.py`**

Insert the following block **above** the `class Planet:` line in `27-Board_new_env.py`:

```python
class OrbitWarsSimulator:
    MAX_SPEED = 6.0
    BOARD_SIZE = 100.0
    CENTER = 50.0
    SUN_RADIUS = 10.0
    ROTATION_RADIUS_LIMIT = 50.0

    def __init__(self, obs):
        def _get(attr, default):
            return obs.get(attr, default) if isinstance(obs, dict) else getattr(obs, attr, default)

        self.angular_velocity = _get("angular_velocity", 0.0)
        self.comet_pid_set = set(_get("comet_planet_ids", []))

        raw_planets = _get("planets", [])
        self.planets = [list(p) for p in raw_planets]
        self.initial_planets = [list(p) for p in raw_planets]
        self.initial_by_id = {p[0]: p for p in self.initial_planets}

        self.fleets = [list(f) for f in _get("fleets", [])]

        self.comets = [
            {
                "planet_ids": list(g["planet_ids"]),
                "paths": g["paths"],       # read-only path data, no deep copy needed
                "path_index": g["path_index"],
            }
            for g in _get("comets", [])
        ]

        self.sim_step = 0

    @staticmethod
    def _pt_seg_dist(p, v, w):
        """Minimum distance from point p to segment v→w."""
        l2 = (v[0] - w[0]) ** 2 + (v[1] - w[1]) ** 2
        if l2 == 0.0:
            return math.hypot(p[0] - v[0], p[1] - v[1])
        t = max(0.0, min(1.0, (
            (p[0] - v[0]) * (w[0] - v[0]) + (p[1] - v[1]) * (w[1] - v[1])
        ) / l2))
        proj_x = v[0] + t * (w[0] - v[0])
        proj_y = v[1] + t * (w[1] - v[1])
        return math.hypot(p[0] - proj_x, p[1] - proj_y)

    def step(self):
        raise NotImplementedError

    def run(self, n):
        raise NotImplementedError
```

- [ ] **Step 1.4: Fix the import in `test_simulator.py`**

The file is named `27-Board_new_env.py` — Python can't import it with a leading digit. Use `importlib`:

```python
# Replace the import at the top of test_simulator.py with:
import importlib.util, sys
spec = importlib.util.spec_from_file_location(
    "board", "27-Board_new_env.py"
)
mod = importlib.util.load_from_spec(spec)
spec.loader.exec_module(mod)
OrbitWarsSimulator = mod.OrbitWarsSimulator
```

- [ ] **Step 1.5: Run tests — all should pass**

```
pytest test_simulator.py -v -k "pt_seg or init"
```
Expected: 8 PASSED.

- [ ] **Step 1.6: Commit**

```
git add 27-Board_new_env.py test_simulator.py
git commit -m "feat: add OrbitWarsSimulator skeleton with _pt_seg_dist and __init__"
```

---

## Task 2: Production + fleet movement (`step()` phases 2–3)

**Files:**
- Modify: `27-Board_new_env.py` (`step()` body)
- Modify: `test_simulator.py`

Physics source: `orbit_wars.py` lines 491–533.

- [ ] **Step 2.1: Add production and fleet movement tests**

Append to `test_simulator.py`:

```python
# --- Production ---

def test_production_adds_ships_to_owned_planet():
    planet = [0, 0, 60.0, 50.0, 2.0, 10, 3]   # owner=0, ships=10, prod=3
    obs = make_obs([planet])
    sim = OrbitWarsSimulator(obs)
    snap = sim.step()
    p = next(s for s in snap if s['id'] == 0)
    assert p['ships'] == 13


def test_production_skips_neutral_planet():
    planet = [0, -1, 60.0, 50.0, 2.0, 10, 3]  # owner=-1
    obs = make_obs([planet])
    sim = OrbitWarsSimulator(obs)
    snap = sim.step()
    p = next(s for s in snap if s['id'] == 0)
    assert p['ships'] == 10


# --- Fleet movement ---

def test_fleet_removed_when_out_of_bounds():
    planet = [0, -1, 60.0, 50.0, 2.0, 5, 1]
    # Fleet aimed at angle π/2 (upward) from near the top edge
    fleet = [0, 0, 50.0, 98.0, math.pi / 2, -1, 1]
    obs = make_obs([planet], fleets=[fleet])
    sim = OrbitWarsSimulator(obs)
    sim.step()
    assert len(sim.fleets) == 0


def test_fleet_removed_when_crossing_sun():
    planet = [0, -1, 80.0, 50.0, 2.0, 5, 1]
    # Fleet aimed directly at sun center (50, 50) from (50, 70)
    fleet = [0, 0, 50.0, 70.0, -math.pi / 2, -1, 100]  # 100 ships → fast
    obs = make_obs([planet], fleets=[fleet])
    sim = OrbitWarsSimulator(obs)
    sim.step()
    assert len(sim.fleets) == 0


def test_fleet_hits_static_planet_after_8_steps():
    # Planet at (60, 50) radius=3. Fleet at (50, 50) angle=0, 1 ship (speed=1.0).
    # Distance to surface: (60-3) - 50 = 7 → hits when fleet x ≥ 57.
    # After 8 steps: fleet x=58, seg (57→58), pt_seg_dist((60,50),(57,50),(58,50))=2<3 ✓
    planet = [0, 1, 60.0, 50.0, 3.0, 20, 2]   # owner=1, ships=20, prod=2
    fleet  = [0, 0, 50.0, 50.0, 0.0, -1, 1]   # 1 ship, speed=1.0
    obs = make_obs([planet], fleets=[fleet])
    sim = OrbitWarsSimulator(obs)
    for _ in range(8):
        snap = sim.step()
    assert len(sim.fleets) == 0
    p = next(s for s in snap if s['id'] == 0)
    # Production: +2/step × 8 steps = +16 → 36 ships on planet at moment of combat.
    # Fleet 1 ship (owner 0) vs planet owner 1: survivor_ships=1, tries to take planet.
    # planet[5] -= 1 → 35, planet keeps owner 1.
    assert p['owner'] == 1
    assert p['ships'] == 35
```

- [ ] **Step 2.2: Run tests — expect failures (step() raises NotImplementedError)**

```
pytest test_simulator.py -v -k "production or fleet"
```
Expected: FAILED with `NotImplementedError`.

- [ ] **Step 2.3: Implement `step()` phases 1–3 (comet pre-expiry + production + fleet movement)**

Replace `def step(self): raise NotImplementedError` in `27-Board_new_env.py` with:

```python
def step(self):
    self.sim_step += 1

    # Phase 1: comet pre-expiry (remove any comets already past their path end)
    pre_expired = {
        pid
        for g in self.comets
        for i, pid in enumerate(g["planet_ids"])
        if g["path_index"] >= len(g["paths"][i])
    }
    self._expire_comets(pre_expired)

    # Phase 2: production
    for planet in self.planets:
        if planet[1] != -1:
            planet[5] += planet[6]

    # Phase 3: fleet movement + continuous collision
    fleets_to_remove = set()
    combat_lists = {p[0]: [] for p in self.planets}

    for fleet in self.fleets:
        angle = fleet[4]
        ships = fleet[6]
        speed = 1.0 + (self.MAX_SPEED - 1.0) * (math.log(ships) / math.log(1000)) ** 1.5
        speed = min(speed, self.MAX_SPEED)
        old_pos = (fleet[2], fleet[3])
        fleet[2] += math.cos(angle) * speed
        fleet[3] += math.sin(angle) * speed
        new_pos = (fleet[2], fleet[3])

        hit = False
        for planet in self.planets:
            if self._pt_seg_dist((planet[2], planet[3]), old_pos, new_pos) < planet[4]:
                combat_lists[planet[0]].append(fleet)
                fleets_to_remove.add(id(fleet))
                hit = True
                break
        if hit:
            continue

        if not (0 <= fleet[2] <= self.BOARD_SIZE and 0 <= fleet[3] <= self.BOARD_SIZE):
            fleets_to_remove.add(id(fleet))
            continue

        if self._pt_seg_dist(
            (self.CENTER, self.CENTER), old_pos, new_pos
        ) < self.SUN_RADIUS:
            fleets_to_remove.add(id(fleet))

    # Phases 4–6 stubbed until next tasks
    self.fleets = [f for f in self.fleets if id(f) not in fleets_to_remove]

    return [
        {"id": p[0], "owner": p[1], "x": p[2], "y": p[3],
         "radius": p[4], "ships": p[5], "production": p[6]}
        for p in self.planets
    ]

def _sweep(self, planet, old_pos, new_pos, fleets_to_remove, combat_lists):
    if old_pos == new_pos:
        return
    for fleet in self.fleets:
        if id(fleet) not in fleets_to_remove:
            if self._pt_seg_dist(
                (fleet[2], fleet[3]), old_pos, new_pos
            ) < planet[4]:
                combat_lists[planet[0]].append(fleet)
                fleets_to_remove.add(id(fleet))

def _expire_comets(self, expired_pids):
    if not expired_pids:
        return
    self.planets = [p for p in self.planets if p[0] not in expired_pids]
    self.initial_by_id = {
        pid: p for pid, p in self.initial_by_id.items()
        if pid not in expired_pids
    }
    self.comet_pid_set -= expired_pids
    for g in self.comets:
        g["planet_ids"] = [pid for pid in g["planet_ids"] if pid not in expired_pids]
    self.comets = [g for g in self.comets if g["planet_ids"]]

def run(self, n):
    raise NotImplementedError
```

- [ ] **Step 2.4: Run tests**

```
pytest test_simulator.py -v -k "production or fleet"
```
Expected: all PASSED. (`test_fleet_hits_static_planet_after_8_steps` may show wrong combat result until Task 5 — that's fine, check the fleet is gone.)

- [ ] **Step 2.5: Commit**

```
git add 27-Board_new_env.py test_simulator.py
git commit -m "feat: OrbitWarsSimulator production and fleet movement phases"
```

---

## Task 3: Planet rotation + `_sweep` (`step()` phase 4)

**Files:**
- Modify: `27-Board_new_env.py` (`step()` — insert phase 4 before the stub snapshot)
- Modify: `test_simulator.py`

Physics source: `orbit_wars.py` lines 535–572.

- [ ] **Step 3.1: Add rotation and sweep tests**

Append to `test_simulator.py`:

```python
# --- Planet rotation ---

def test_orbiting_planet_rotates_by_omega():
    omega = 0.05
    # Planet at (60, 50): orbital radius=10, radius=1 → 10+1=11 < 50 → orbiting
    planet = [0, -1, 60.0, 50.0, 1.0, 5, 1]
    obs = make_obs([planet], angular_velocity=omega)
    sim = OrbitWarsSimulator(obs)
    snap = sim.step()  # sim_step=1
    p = next(s for s in snap if s['id'] == 0)
    initial_angle = math.atan2(50.0 - 50.0, 60.0 - 50.0)  # atan2(0, 10) = 0
    expected_x = 50.0 + 10.0 * math.cos(initial_angle + omega)
    expected_y = 50.0 + 10.0 * math.sin(initial_angle + omega)
    assert abs(p['x'] - expected_x) < 1e-9
    assert abs(p['y'] - expected_y) < 1e-9


def test_static_planet_does_not_rotate():
    # Planet at (90, 50): orbital radius=40, radius=15 → 40+15=55 ≥ 50 → static
    planet = [0, -1, 90.0, 50.0, 15.0, 5, 1]
    obs = make_obs([planet], angular_velocity=0.05)
    sim = OrbitWarsSimulator(obs)
    snap = sim.step()
    p = next(s for s in snap if s['id'] == 0)
    assert p['x'] == 90.0
    assert p['y'] == 50.0


def test_orbiting_planet_sweeps_fleet():
    omega = 0.1
    # Planet at (60, 50), radius=3, orbiting.
    # Place a fleet right where the planet will be after 1 rotation step.
    r = 10.0
    angle_after = math.atan2(0, r) + omega  # initial angle 0 → rotates to omega
    fleet_x = 50.0 + r * math.cos(angle_after)
    fleet_y = 50.0 + r * math.sin(angle_after)
    planet = [0, 0, 60.0, 50.0, 3.0, 5, 1]
    fleet  = [0, 1, fleet_x, fleet_y, 0.0, -1, 1]  # stationary-ish fleet at sweep target
    obs = make_obs([planet], fleets=[fleet], angular_velocity=omega)
    sim = OrbitWarsSimulator(obs)
    sim.step()
    assert len(sim.fleets) == 0  # fleet was swept
```

- [ ] **Step 3.2: Run tests — rotation tests fail (phase 4 not yet inserted)**

```
pytest test_simulator.py -v -k "rotat or static or sweep"
```
Expected: FAILED.

- [ ] **Step 3.3: Insert phase 4 into `step()` before `self.fleets = [...]`**

In `step()`, find the line `# Phases 4–6 stubbed until next tasks` and replace it with:

```python
    # Phase 4: planet rotation + sweep
    comet_pid_set = self.comet_pid_set
    for planet in self.planets:
        if planet[0] in comet_pid_set:
            continue
        init_p = self.initial_by_id.get(planet[0])
        if not init_p:
            continue
        dx = init_p[2] - self.CENTER
        dy = init_p[3] - self.CENTER
        r = math.hypot(dx, dy)
        old_pos = (planet[2], planet[3])
        if r + planet[4] < self.ROTATION_RADIUS_LIMIT:
            theta = math.atan2(dy, dx) + self.angular_velocity * self.sim_step
            planet[2] = self.CENTER + r * math.cos(theta)
            planet[3] = self.CENTER + r * math.sin(theta)
        self._sweep(planet, old_pos, (planet[2], planet[3]), fleets_to_remove, combat_lists)
```

- [ ] **Step 3.4: Run tests**

```
pytest test_simulator.py -v -k "rotat or static or sweep"
```
Expected: all PASSED.

- [ ] **Step 3.5: Commit**

```
git add 27-Board_new_env.py test_simulator.py
git commit -m "feat: OrbitWarsSimulator planet rotation and fleet sweep phases"
```

---

## Task 4: Comet movement + `_expire_comets` (`step()` phase 5)

**Files:**
- Modify: `27-Board_new_env.py` (`step()` — insert phase 5)
- Modify: `test_simulator.py`

Physics source: `orbit_wars.py` lines 574–608.

- [ ] **Step 4.1: Add comet tests**

Append to `test_simulator.py`:

```python
# --- Comet movement ---

def test_comet_advances_along_path():
    path = [[10.0, 20.0], [11.0, 21.0], [12.0, 22.0]]
    planet = [100, -1, 10.0, 20.0, 1.0, 5, 1]   # starts at path[0]
    group = {"planet_ids": [100], "paths": [path], "path_index": 0}
    obs = make_obs([planet], comet_planet_ids=[100], comets=[group])
    sim = OrbitWarsSimulator(obs)
    snap = sim.step()   # path_index → 1
    p = next(s for s in snap if s['id'] == 100)
    assert abs(p['x'] - 11.0) < 1e-9
    assert abs(p['y'] - 21.0) < 1e-9


def test_comet_expires_when_path_exhausted():
    path = [[10.0, 20.0]]   # only one position → expires after one advance
    planet = [100, -1, 10.0, 20.0, 1.0, 5, 1]
    group = {"planet_ids": [100], "paths": [path], "path_index": 0}
    obs = make_obs([planet], comet_planet_ids=[100], comets=[group])
    sim = OrbitWarsSimulator(obs)
    snap = sim.step()   # path_index → 1 ≥ len(path)=1 → expire
    assert all(s['id'] != 100 for s in snap)
    assert 100 not in sim.comet_pid_set
    assert len(sim.comets) == 0


def test_comet_pre_expiry_cleans_already_expired():
    # path_index already at len(path) before step starts
    path = [[10.0, 20.0]]
    planet = [100, -1, 10.0, 20.0, 1.0, 5, 1]
    group = {"planet_ids": [100], "paths": [path], "path_index": 1}  # already expired
    obs = make_obs([planet], comet_planet_ids=[100], comets=[group])
    sim = OrbitWarsSimulator(obs)
    snap = sim.step()
    assert all(s['id'] != 100 for s in snap)
```

- [ ] **Step 4.2: Run tests — comet tests fail**

```
pytest test_simulator.py -v -k "comet"
```
Expected: FAILED (phase 5 not yet inserted).

- [ ] **Step 4.3: Insert phase 5 into `step()` after phase 4 and before `self.fleets = [...]`**

```python
    # Phase 5: comet movement + expiry
    newly_expired = set()
    for group in self.comets:
        group["path_index"] += 1
        idx = group["path_index"]
        for i, pid in enumerate(group["planet_ids"]):
            planet = next((p for p in self.planets if p[0] == pid), None)
            if planet is None:
                continue
            p_path = group["paths"][i]
            if idx >= len(p_path):
                newly_expired.add(pid)
            else:
                old_pos = (planet[2], planet[3])
                planet[2] = p_path[idx][0]
                planet[3] = p_path[idx][1]
                if old_pos[0] >= 0:   # skip sweep on first off-board placement
                    self._sweep(planet, old_pos, (planet[2], planet[3]),
                                fleets_to_remove, combat_lists)
    self._expire_comets(newly_expired)
```

- [ ] **Step 4.4: Run tests**

```
pytest test_simulator.py -v -k "comet"
```
Expected: all PASSED.

- [ ] **Step 4.5: Commit**

```
git add 27-Board_new_env.py test_simulator.py
git commit -m "feat: OrbitWarsSimulator comet movement and expiry phases"
```

---

## Task 5: Combat resolution + snapshot + `run(n)`

**Files:**
- Modify: `27-Board_new_env.py` (`step()` — insert phase 6; implement `run()`)
- Modify: `test_simulator.py`

Physics source: `orbit_wars.py` lines 612–651.

- [ ] **Step 5.1: Add combat and run tests**

Append to `test_simulator.py`:

```python
# --- Combat ---

def test_combat_fleet_reduces_defending_planet():
    # Attacker fleet (1 ship, owner 0) close to planet (20 ships, owner 1, radius 3)
    # Same setup as test_fleet_hits_static_planet_after_8_steps but checking combat result.
    planet = [0, 1, 60.0, 50.0, 3.0, 20, 2]
    fleet  = [0, 0, 50.0, 50.0, 0.0, -1, 1]
    obs = make_obs([planet], fleets=[fleet])
    sim = OrbitWarsSimulator(obs)
    for _ in range(8):
        snap = sim.step()
    p = next(s for s in snap if s['id'] == 0)
    assert p['owner'] == 1
    assert p['ships'] == 35   # 20 + 8*2 - 1


def test_combat_attacker_captures_neutral():
    # Two fleets arrive simultaneously at neutral planet (0 ships).
    # Fleet 0: 10 ships. Fleet 1: 5 ships. Winner: fleet 0 with 5 survivors.
    # Neutral planet has 0 ships → planet flips to owner 0 with 5 ships.
    planet = [0, -1, 60.0, 50.0, 2.0, 0, 1]
    # Both fleets close enough to hit in step 1
    fleet0 = [0, 0, 57.0, 50.0, 0.0,       -1, 10]   # speed≈1.962 → arrives in 1 step
    fleet1 = [1, 1, 63.0, 50.0, math.pi,   -1,  5]   # speed≈1.442 → arrives in 1 step
    obs = make_obs([planet], fleets=[fleet0, fleet1])
    sim = OrbitWarsSimulator(obs)
    snap = sim.step()
    p = next(s for s in snap if s['id'] == 0)
    assert p['owner'] == 0
    assert p['ships'] == 5


def test_combat_tie_leaves_planet_neutral():
    planet = [0, -1, 60.0, 50.0, 2.0, 0, 1]
    fleet0 = [0, 0, 57.0, 50.0, 0.0,     -1, 10]
    fleet1 = [1, 1, 63.0, 50.0, math.pi, -1, 10]
    obs = make_obs([planet], fleets=[fleet0, fleet1])
    sim = OrbitWarsSimulator(obs)
    snap = sim.step()
    p = next(s for s in snap if s['id'] == 0)
    assert p['owner'] == -1
    assert p['ships'] == 0


# --- run(n) ---

def test_run_returns_n_snapshots():
    planet = [0, 0, 60.0, 50.0, 3.0, 10, 2]
    obs = make_obs([planet])
    sim = OrbitWarsSimulator(obs)
    snaps = sim.run(5)
    assert len(snaps) == 5


def test_run_production_accumulates():
    planet = [0, 0, 60.0, 50.0, 3.0, 10, 2]  # prod=2
    obs = make_obs([planet])
    sim = OrbitWarsSimulator(obs)
    snaps = sim.run(5)
    ships = [next(p['ships'] for p in s if p['id'] == 0) for s in snaps]
    assert ships == [12, 14, 16, 18, 20]
```

- [ ] **Step 5.2: Run tests — combat and run tests fail**

```
pytest test_simulator.py -v -k "combat or run"
```
Expected: FAILED.

- [ ] **Step 5.3: Insert phase 6 into `step()` (after `self.fleets = [...]` and before the return)**

Find `return [` in `step()` and insert before it:

```python
    # Phase 6: combat resolution
    for pid, arriving in combat_lists.items():
        if not arriving:
            continue
        planet = next((p for p in self.planets if p[0] == pid), None)
        if not planet:
            continue
        per_player = {}
        for fleet in arriving:
            per_player[fleet[1]] = per_player.get(fleet[1], 0) + fleet[6]
        ranked = sorted(per_player.items(), key=lambda kv: kv[1], reverse=True)
        top_owner, top_ships = ranked[0]
        if len(ranked) > 1:
            second = ranked[1][1]
            survivors = top_ships - second
            winner = top_owner if survivors > 0 else -1
        else:
            survivors = top_ships
            winner = top_owner
        if survivors > 0:
            if planet[1] == winner:
                planet[5] += survivors
            else:
                planet[5] -= survivors
                if planet[5] < 0:
                    planet[1] = winner
                    planet[5] = abs(planet[5])
```

- [ ] **Step 5.4: Implement `run()`**

Replace `def run(self, n): raise NotImplementedError` with:

```python
def run(self, n):
    return [self.step() for _ in range(n)]
```

- [ ] **Step 5.5: Run all tests**

```
pytest test_simulator.py -v
```
Expected: all PASSED.

- [ ] **Step 5.6: Commit**

```
git add 27-Board_new_env.py test_simulator.py
git commit -m "feat: OrbitWarsSimulator combat resolution, snapshot, and run()"
```

---

## Task 6: Board integration + remove ke.make

**Files:**
- Modify: `27-Board_new_env.py` (`Board._run_simulation`)

- [ ] **Step 6.1: Replace `Board._run_simulation` with the 4-line wrapper**

Find and replace the entire `_run_simulation` method in `Board`:

```python
def _run_simulation(self, obs):
    for snap in OrbitWarsSimulator(obs).run(NB_FORECAST_STEPS):
        for planet_state in snap:
            pid = planet_state["id"]
            if pid in self.planets_dico:
                self.planets_dico[pid].nexts.append(planet_state)
```

- [ ] **Step 6.2: Remove the ke.make import**

Find and delete this line at the top of `27-Board_new_env.py`:

```python
import kaggle_environments as ke
```

- [ ] **Step 6.3: Smoke test — run the agent on a real episode**

```python
# In a Python shell or notebook cell:
import importlib.util, sys
spec = importlib.util.spec_from_file_location("board", "27-Board_new_env.py")
mod = importlib.util.load_from_spec(spec)
spec.loader.exec_module(mod)

import kaggle_environments as ke
env = ke.make("orbit_wars", debug=True)
env.reset(2)
result = env.run([mod.agent, "random"])
print("Episode done, steps:", len(result))
print("Rewards:", [s.reward for s in result[-1]])
```

Expected: no exceptions, rewards printed (1 and -1 or similar).

- [ ] **Step 6.4: Verify `nexts` is populated correctly**

Add this quick check to `test_simulator.py` and run it:

```python
def test_board_simulation_populates_nexts():
    import importlib.util
    spec = importlib.util.spec_from_file_location("board", "27-Board_new_env.py")
    mod = importlib.util.load_from_spec(spec)
    spec.loader.exec_module(mod)

    import kaggle_environments as ke
    env = ke.make("orbit_wars", debug=True)
    env.reset(2)
    obs = env.state[0].observation

    board = mod.Board(obs)
    for planet in board.my_planets:
        assert len(planet.nexts) == mod.NB_FORECAST_STEPS, (
            f"Planet {planet.id} has {len(planet.nexts)} nexts, expected {mod.NB_FORECAST_STEPS}"
        )
```

```
pytest test_simulator.py::test_board_simulation_populates_nexts -v
```
Expected: PASSED.

- [ ] **Step 6.5: Run full test suite**

```
pytest test_simulator.py -v
```
Expected: all PASSED.

- [ ] **Step 6.6: Commit**

```
git add 27-Board_new_env.py test_simulator.py
git commit -m "feat: wire OrbitWarsSimulator into Board, remove ke.make dependency"
```

---

## Self-Review Checklist

- [x] `_pt_seg_dist` — tested and implemented (Task 1)
- [x] `__init__` deep copy — tested (Task 1)
- [x] Production — tested (Task 2)
- [x] Fleet movement + continuous collision + out-of-bounds + sun check — tested (Task 2)
- [x] Planet rotation (orbiting vs static) — tested (Task 3)
- [x] `_sweep` — tested (Task 3)
- [x] Comet movement along path — tested (Task 4)
- [x] Comet expiry (post-move + pre-step) — tested (Task 4)
- [x] `_expire_comets` — tested (Task 4)
- [x] Combat resolution (defender, capture, tie) — tested (Task 5)
- [x] `run(n)` — tested (Task 5)
- [x] `Board._run_simulation` integration — tested (Task 6)
- [x] ke.make import removed — Task 6
- [x] Comet pre-expiry matches interpreter line 388 behavior — Task 4 `test_comet_pre_expiry_cleans_already_expired`
