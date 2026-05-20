# Nearest-10-Steps Agent Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create `33-Kaggle_env_nearest_10steps.py`, a self-contained Kaggle agent that forward-simulates 10 steps, fires only from consistently-owned planets, and sends exactly enough ships to take the nearest (by ETA) non-owned target at step 10.

**Architecture:** Single `.py` file — interpreter pasted verbatim from notebook 32, three physics helpers (`_fleet_speed`, `_eta`, `_aim_angle`), a `_simulate()` that builds a pandas DataFrame of all 10 simulation snapshots, and the main `nearest_planet_sniper()` orchestrator. Global state is a single dict `global_board` holding only `step` and `num_agents`.

**Tech Stack:** Python 3, pandas, math, copy, kaggle_environments (for integration test only)

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `33-Kaggle_env_nearest_10steps.py` | Create | Full agent — interpreter + helpers + agent fn |
| `test_33_nearest_10steps.py` | Create | Unit + integration tests |

---

### Task 1: Scaffold file and paste interpreter

**Files:**
- Create: `33-Kaggle_env_nearest_10steps.py`

- [ ] **Step 1: Create the file with imports, constants, and interpreter**

Copy the content below verbatim. The interpreter block is copied from cell 0 of
`32-board_from_kaggle.ipynb` — do NOT modify it.

```python
import math
import copy
import pandas as pd

# ── Constants ─────────────────────────────────────────────────────────────────
CENTER = 50.0
SUN_RADIUS = 10.0
ROTATION_RADIUS_LIMIT = 50.0
MAX_SPEED = 6.0
NB_STEPS_SIM = 10

# ── Interpreter (verbatim from 32-board_from_kaggle.ipynb cell 0) ─────────────
from collections import namedtuple

Planet = namedtuple(
    "Planet", ["id", "owner", "x", "y", "radius", "ships", "production"]
)
Fleet = namedtuple(
    "Fleet", ["id", "owner", "x", "y", "angle", "from_planet_id", "ships"]
)

BOARD_SIZE = 100.0
COMET_RADIUS = 1.0
COMET_PRODUCTION = 1
PLANET_CLEARANCE = 7
MIN_PLANET_GROUPS = 5
MAX_PLANET_GROUPS = 10
MIN_STATIC_GROUPS = 3
COMET_SPAWN_STEPS = [50, 150, 250, 350, 450]
CENTER_X = 50.0
CENTER_Y = 50.0
MAX_NB_STEP = 500


def distance(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def point_to_segment_distance(p, v, w):
    """Minimum distance from point p to line segment v-w."""
    l2 = (v[0] - w[0]) ** 2 + (v[1] - w[1]) ** 2
    if l2 == 0.0:
        return distance(p, v)
    t = max(
        0, min(1, ((p[0] - v[0]) * (w[0] - v[0]) + (p[1] - v[1]) * (w[1] - v[1])) / l2)
    )
    projection = (v[0] + t * (w[0] - v[0]), v[1] + t * (w[1] - v[1]))
    return distance(p, projection)


def interpreter(obs, actions, step, num_agents=2):
    obs0 = obs

    expired_comet_pids = []
    for group in obs0.comets:
        idx = group["path_index"]
        for i, pid in enumerate(group["planet_ids"]):
            if idx >= len(group["paths"][i]):
                expired_comet_pids.append(pid)
    if expired_comet_pids:
        expired_set = set(expired_comet_pids)
        obs0.planets = [p for p in obs0.planets if p[0] not in expired_set]
        obs0.initial_planets = [
            p for p in obs0.initial_planets if p[0] not in expired_set
        ]
        obs0.comet_planet_ids = [
            pid for pid in obs0.comet_planet_ids if pid not in expired_set
        ]
        for group in obs0.comets:
            group["planet_ids"] = [
                pid for pid in group["planet_ids"] if pid not in expired_set
            ]
        obs0.comets = [g for g in obs0.comets if g["planet_ids"]]

    def process_moves(player_id, action):
        if not action or not isinstance(action, list):
            return
        for move in action:
            if len(move) != 3:
                continue
            from_id, angle, ships = move
            ships = int(ships)
            from_planet = next((p for p in obs0.planets if p[0] == from_id), None)
            if from_planet and from_planet[1] == player_id:
                if from_planet[5] >= ships and ships > 0:
                    from_planet[5] -= ships
                    start_x = from_planet[2] + math.cos(angle) * (from_planet[4] + 0.1)
                    start_y = from_planet[3] + math.sin(angle) * (from_planet[4] + 0.1)
                    obs0.fleets.append([
                        obs0.next_fleet_id, player_id,
                        start_x, start_y, angle, from_id, ships,
                    ])
                    obs0.next_fleet_id += 1

    for i in range(num_agents):
        process_moves(i, actions[i])

    for planet in obs0.planets:
        if planet[1] != -1:
            planet[5] += planet[6]

    max_speed = MAX_SPEED
    fleets_to_remove = []
    combat_lists = {p[0]: [] for p in obs0.planets}

    for fleet in obs0.fleets:
        angle = fleet[4]
        ships = fleet[6]
        speed = 1.0 + (max_speed - 1.0) * (math.log(ships) / math.log(1000)) ** 1.5
        speed = min(speed, max_speed)
        old_pos = (fleet[2], fleet[3])
        fleet[2] += math.cos(angle) * speed
        fleet[3] += math.sin(angle) * speed
        new_pos = (fleet[2], fleet[3])

        hit_planet = False
        for planet in obs0.planets:
            planet_pos = (planet[2], planet[3])
            if point_to_segment_distance(planet_pos, old_pos, new_pos) < planet[4]:
                combat_lists[planet[0]].append(fleet)
                fleets_to_remove.append(fleet)
                hit_planet = True
                break
        if hit_planet:
            continue
        if not (0 <= fleet[2] <= BOARD_SIZE and 0 <= fleet[3] <= BOARD_SIZE):
            fleets_to_remove.append(fleet)
            continue
        if point_to_segment_distance((CENTER, CENTER), old_pos, new_pos) < SUN_RADIUS:
            fleets_to_remove.append(fleet)
            continue

    angular_velocity = obs0.angular_velocity
    comet_pid_set = set(obs0.comet_planet_ids)
    initial_by_id = {p[0]: p for p in obs0.initial_planets}

    def sweep_fleets(planet, old_pos, new_pos):
        if old_pos == new_pos:
            return
        for fleet in obs0.fleets:
            if fleet not in fleets_to_remove:
                if point_to_segment_distance((fleet[2], fleet[3]), old_pos, new_pos) < planet[4]:
                    combat_lists[planet[0]].append(fleet)
                    fleets_to_remove.append(fleet)

    for planet in obs0.planets:
        if planet[0] in comet_pid_set:
            continue
        initial_p = initial_by_id.get(planet[0])
        if not initial_p:
            continue
        dx = initial_p[2] - CENTER
        dy = initial_p[3] - CENTER
        r = math.sqrt(dx**2 + dy**2)
        old_pos = (planet[2], planet[3])
        if r + planet[4] < ROTATION_RADIUS_LIMIT:
            initial_angle = math.atan2(dy, dx)
            current_angle = initial_angle + angular_velocity * step
            planet[2] = CENTER + r * math.cos(current_angle)
            planet[3] = CENTER + r * math.sin(current_angle)
        sweep_fleets(planet, old_pos, (planet[2], planet[3]))

    expired_comet_pids = []
    for group in obs0.comets:
        group["path_index"] += 1
        idx = group["path_index"]
        for i, pid in enumerate(group["planet_ids"]):
            planet = next((p for p in obs0.planets if p[0] == pid), None)
            if planet is None:
                continue
            p_path = group["paths"][i]
            if idx >= len(p_path):
                expired_comet_pids.append(pid)
            else:
                old_pos = (planet[2], planet[3])
                planet[2] = p_path[idx][0]
                planet[3] = p_path[idx][1]
                if old_pos[0] >= 0:
                    sweep_fleets(planet, old_pos, (planet[2], planet[3]))

    if expired_comet_pids:
        expired_set = set(expired_comet_pids)
        obs0.planets = [p for p in obs0.planets if p[0] not in expired_set]
        obs0.initial_planets = [
            p for p in obs0.initial_planets if p[0] not in expired_set
        ]
        obs0.comet_planet_ids = [
            pid for pid in obs0.comet_planet_ids if pid not in expired_set
        ]
        for group in obs0.comets:
            group["planet_ids"] = [
                pid for pid in group["planet_ids"] if pid not in expired_set
            ]
        obs0.comets = [g for g in obs0.comets if g["planet_ids"]]

    obs0.fleets = [f for f in obs0.fleets if f not in fleets_to_remove]

    for pid, planet_fleets in combat_lists.items():
        planet = next((p for p in obs0.planets if p[0] == pid), None)
        if not planet or not planet_fleets:
            continue
        player_ships = {}
        for fleet in planet_fleets:
            owner = fleet[1]
            player_ships[owner] = player_ships.get(owner, 0) + fleet[6]
        if not player_ships:
            continue
        sorted_players = sorted(player_ships.items(), key=lambda item: item[1], reverse=True)
        top_player, top_ships = sorted_players[0]
        if len(sorted_players) > 1:
            second_ships = sorted_players[1][1]
            survivor_ships = top_ships - second_ships
            if sorted_players[0][1] == sorted_players[1][1]:
                survivor_ships = 0
            survivor_owner = top_player if survivor_ships > 0 else -1
        else:
            survivor_owner = top_player
            survivor_ships = top_ships
        if survivor_ships > 0:
            if planet[1] == survivor_owner:
                planet[5] += survivor_ships
            else:
                planet[5] -= survivor_ships
                if planet[5] < 0:
                    planet[1] = survivor_owner
                    planet[5] = abs(planet[5])

    obs1 = {
        "planets": obs0.planets,
        "initial_planets": obs0.initial_planets,
        "fleets": obs0.fleets,
        "next_fleet_id": obs0.next_fleet_id,
        "comets": obs0.comets,
        "comet_planet_ids": obs0.comet_planet_ids,
    }

    terminated = False
    if step >= MAX_NB_STEP - 2:
        terminated = True
    alive_players = set()
    for p in obs0.planets:
        if p[1] != -1:
            alive_players.add(p[1])
    for f in obs0.fleets:
        alive_players.add(f[1])
    if len(alive_players) <= 1:
        terminated = True

    return obs1
```

- [ ] **Step 2: Verify syntax**

```
python -c "import ast; ast.parse(open('33-Kaggle_env_nearest_10steps.py').read()); print('OK')"
```

Expected output: `OK`

- [ ] **Step 3: Commit**

```bash
git add 33-Kaggle_env_nearest_10steps.py
git commit -m "feat: scaffold 33-agent with interpreter"
```

---

### Task 2: `_fleet_speed` and `_simulate`

**Files:**
- Modify: `33-Kaggle_env_nearest_10steps.py`
- Create: `test_33_nearest_10steps.py`

- [ ] **Step 1: Create test file with `_fleet_speed` tests**

```python
# test_33_nearest_10steps.py
import importlib.util, pathlib, math

def load_agent():
    p = pathlib.Path(__file__).parent / "33-Kaggle_env_nearest_10steps.py"
    spec = importlib.util.spec_from_file_location("agent33", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m

m = load_agent()


class MockObs:
    def __init__(self, planets, angular_velocity=0.0):
        self.planets          = [list(p) for p in planets]
        self.initial_planets  = [list(p) for p in planets]
        self.fleets           = []
        self.next_fleet_id    = 100
        self.comets           = []
        self.comet_planet_ids = []
        self.angular_velocity = angular_velocity
        self.player           = 0


# ── _fleet_speed ──────────────────────────────────────────────────────────────
def test_fleet_speed_one_ship():
    assert m._fleet_speed(1) == 1.0

def test_fleet_speed_max():
    assert math.isclose(m._fleet_speed(1000), 6.0, rel_tol=1e-6)

def test_fleet_speed_midrange():
    v = m._fleet_speed(100)
    assert 1.0 < v < 6.0


# ── _simulate ─────────────────────────────────────────────────────────────────
_PLANETS_2P = [
    # [id, owner, x,  y,  radius, ships, production]
    [0,  0,   3,  50,  5,    50,    5],   # static (dist=47, 47+5=52 >= 50)
    [1,  1,  97,  50,  5,    10,    3],   # static
]

def test_simulate_shape():
    obs = MockObs(_PLANETS_2P)
    df = m._simulate(obs, global_step=0, num_agents=2)
    assert set(df.columns) == {"step", "id", "x", "y", "radius", "ships",
                                "production", "owner", "nature"}
    assert len(df) == 10 * 2  # 10 steps × 2 planets

def test_simulate_step_labels():
    obs = MockObs(_PLANETS_2P)
    df = m._simulate(obs, global_step=5, num_agents=2)
    assert df["step"].min() == 6
    assert df["step"].max() == 15

def test_simulate_nature_static():
    obs = MockObs(_PLANETS_2P)
    df = m._simulate(obs, global_step=0, num_agents=2)
    assert (df["nature"] == "fix").all()

def test_simulate_nature_moving():
    moving_planets = [
        [0, 0, 20, 50, 5, 50, 5],   # dist=30, 30+5=35 < 50 → moving
        [1, 1, 80, 50, 5, 10, 3],   # dist=30, moving
    ]
    obs = MockObs(moving_planets, angular_velocity=0.01)
    df = m._simulate(obs, global_step=0, num_agents=2)
    assert (df["nature"] == "moving").all()

def test_simulate_production_grows():
    obs = MockObs(_PLANETS_2P)
    df = m._simulate(obs, global_step=0, num_agents=2)
    p0_ships = df.query("id == 0").sort_values("step")["ships"].values
    assert p0_ships[-1] > p0_ships[0]
```

- [ ] **Step 2: Run tests — expect failures**

```
python -m pytest test_33_nearest_10steps.py -v -k "fleet_speed or simulate"
```

Expected: all tests FAIL with `AttributeError: module 'agent33' has no attribute '_fleet_speed'`

- [ ] **Step 3: Add `_fleet_speed` and `_simulate` to agent file**

Append after the `interpreter` function:

```python
# ── Physics helpers ───────────────────────────────────────────────────────────

def _fleet_speed(ships):
    if ships <= 1:
        return 1.0
    ratio = math.log(ships) / math.log(1000.0)
    return 1.0 + (MAX_SPEED - 1.0) * max(0.0, min(1.0, ratio)) ** 1.5


def _simulate(obs, global_step, num_agents, n_steps=NB_STEPS_SIM):
    sim = copy.deepcopy(obs)
    no_actions = [[] for _ in range(num_agents)]
    rows = []
    for i in range(n_steps):
        interpreter(sim, no_actions, global_step + i, num_agents)
        for p in sim.planets:
            pid, owner, x, y, radius, ships, production = (
                p[0], p[1], p[2], p[3], p[4], p[5], p[6]
            )
            r = math.hypot(x - CENTER, y - CENTER)
            if pid in sim.comet_planet_ids:
                nature = "comet"
            elif r + radius < ROTATION_RADIUS_LIMIT:
                nature = "moving"
            else:
                nature = "fix"
            rows.append({
                "step": global_step + i + 1,
                "id": pid,
                "x": x,
                "y": y,
                "radius": radius,
                "ships": ships,
                "production": production,
                "owner": owner,
                "nature": nature,
            })
    return pd.DataFrame(rows)
```

- [ ] **Step 4: Run tests — expect pass**

```
python -m pytest test_33_nearest_10steps.py -v -k "fleet_speed or simulate"
```

Expected: all 8 tests PASS

- [ ] **Step 5: Commit**

```bash
git add 33-Kaggle_env_nearest_10steps.py test_33_nearest_10steps.py
git commit -m "feat: add _fleet_speed and _simulate to 33-agent"
```

---

### Task 3: `_eta` and `_aim_angle`

**Files:**
- Modify: `33-Kaggle_env_nearest_10steps.py`
- Modify: `test_33_nearest_10steps.py`

- [ ] **Step 1: Add `_eta` and `_aim_angle` tests to test file**

Append to `test_33_nearest_10steps.py`:

```python
# ── _eta ──────────────────────────────────────────────────────────────────────
# Static scenario: src=(3,50) radius=5, tgt at (97,50) radius=5, no rotation
# travel_dist = 94 - 5 - 5 = 84, speed=1.0 (1 ship), ETA=84
_SRC = (3.0, 50.0, 5.0)   # x, y, radius
_TGT_STATIC = [1, 1, 97.0, 50.0, 5.0, 10, 3]  # planet list

def test_eta_static_planet():
    eta = m._eta(*_SRC, _TGT_STATIC, angular_velocity=0.0)
    assert eta == 84

def test_eta_zero_distance():
    # source and target touching — ETA should be 1 (min)
    src = (50.0, 10.0, 3.0)
    tgt = [2, 1, 50.0, 17.0, 3.0, 5, 1]  # dist between centers = 7, radii sum = 6 → gap=1
    eta = m._eta(*src, tgt, angular_velocity=0.0)
    assert eta >= 1

def test_eta_unreachable_moving():
    # angular_velocity so fast the planet runs away — should return 9999
    src = (3.0, 50.0, 5.0)
    tgt = [1, 1, 30.0, 50.0, 5.0, 10, 3]  # moving planet (dist=20, 20+5=25 < 50)
    eta = m._eta(*src, tgt, angular_velocity=999.0)
    assert eta == 9999


# ── _aim_angle ────────────────────────────────────────────────────────────────
def test_aim_angle_static_horizontal():
    # src left of tgt, same y → angle should be 0.0 (pointing right)
    angle = m._aim_angle(*_SRC, _TGT_STATIC, angular_velocity=0.0, ships=1)
    assert math.isclose(angle, 0.0, abs_tol=1e-9)

def test_aim_angle_static_vertical():
    src = (50.0, 3.0, 5.0)
    tgt = [2, 1, 50.0, 97.0, 5.0, 10, 3]
    angle = m._aim_angle(*src, tgt, angular_velocity=0.0, ships=1)
    assert math.isclose(angle, math.pi / 2, abs_tol=1e-9)
```

- [ ] **Step 2: Run tests — expect failures**

```
python -m pytest test_33_nearest_10steps.py -v -k "eta or aim"
```

Expected: all 5 tests FAIL with `AttributeError: module 'agent33' has no attribute '_eta'`

- [ ] **Step 3: Add `_eta` and `_aim_angle` to agent file**

Append after `_simulate`:

```python
def _eta(src_x, src_y, src_r, tgt, angular_velocity, ships=1):
    """ETA in steps for a fleet of `ships` to reach tgt from (src_x, src_y)."""
    tx, ty, tr = tgt[2], tgt[3], tgt[4]
    speed = _fleet_speed(ships)
    if math.hypot(tx - CENTER, ty - CENTER) + tr >= ROTATION_RADIUS_LIMIT:
        dist = max(0.0, math.hypot(tx - src_x, ty - src_y) - src_r - tr)
        return max(1, math.ceil(dist / speed))
    base_angle = math.atan2(ty - CENTER, tx - CENTER)
    r = math.hypot(tx - CENTER, ty - CENTER)
    for t in range(1, 101):
        fx = CENTER + r * math.cos(base_angle + angular_velocity * t)
        fy = CENTER + r * math.sin(base_angle + angular_velocity * t)
        if math.hypot(fx - src_x, fy - src_y) - t * speed < src_r + tr:
            return t
    return 9999


def _aim_angle(src_x, src_y, src_r, tgt, angular_velocity, ships):
    """Angle to fire at tgt, accounting for orbital intercept."""
    tx, ty, tr = tgt[2], tgt[3], tgt[4]
    speed = _fleet_speed(ships)
    if math.hypot(tx - CENTER, ty - CENTER) + tr >= ROTATION_RADIUS_LIMIT:
        return math.atan2(ty - src_y, tx - src_x)
    base_angle = math.atan2(ty - CENTER, tx - CENTER)
    r = math.hypot(tx - CENTER, ty - CENTER)
    for t in range(1, 101):
        fx = CENTER + r * math.cos(base_angle + angular_velocity * t)
        fy = CENTER + r * math.sin(base_angle + angular_velocity * t)
        if math.hypot(fx - src_x, fy - src_y) - t * speed < src_r + tr:
            return math.atan2(fy - src_y, fx - src_x)
    return math.atan2(ty - src_y, tx - src_x)
```

- [ ] **Step 4: Run tests — expect pass**

```
python -m pytest test_33_nearest_10steps.py -v -k "eta or aim"
```

Expected: all 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add 33-Kaggle_env_nearest_10steps.py test_33_nearest_10steps.py
git commit -m "feat: add _eta and _aim_angle to 33-agent"
```

---

### Task 4: `nearest_planet_sniper` and integration test

**Files:**
- Modify: `33-Kaggle_env_nearest_10steps.py`
- Modify: `test_33_nearest_10steps.py`

- [ ] **Step 1: Add integration test to test file**

Append to `test_33_nearest_10steps.py`:

```python
# ── nearest_planet_sniper integration ─────────────────────────────────────────
import kaggle_environments as ke

def _fresh_board():
    """Return a fresh global_board dict (avoids shared state between tests)."""
    return {"step": 0, "num_agents": None}

def test_agent_returns_list():
    env = ke.make("orbit_wars", debug=False)
    env.reset(2)
    obs = env.state[0].observation
    gb = _fresh_board()
    result = m.nearest_planet_sniper(obs, gb)
    assert isinstance(result, list)

def test_agent_move_format():
    """Every move must be [planet_id, angle, ships] with ships >= 1."""
    env = ke.make("orbit_wars", debug=False)
    env.reset(2)
    obs = env.state[0].observation
    gb = _fresh_board()
    moves = m.nearest_planet_sniper(obs, gb)
    for move in moves:
        assert len(move) == 3
        pid, angle, ships = move
        assert isinstance(pid, int)
        assert -math.pi <= angle <= math.pi
        assert ships >= 1

def test_agent_increments_step():
    env = ke.make("orbit_wars", debug=False)
    env.reset(2)
    obs = env.state[0].observation
    gb = _fresh_board()
    m.nearest_planet_sniper(obs, gb)
    assert gb["step"] == 1

def test_agent_detects_num_agents_2p():
    env = ke.make("orbit_wars", debug=False)
    env.reset(2)
    obs = env.state[0].observation
    gb = _fresh_board()
    m.nearest_planet_sniper(obs, gb)
    assert gb["num_agents"] == 2

def test_agent_no_crash_10_steps():
    """Run 10 full turns without exception."""
    env = ke.make("orbit_wars", debug=False)
    env.reset(2)
    gb = _fresh_board()
    for _ in range(10):
        obs = env.state[0].observation
        moves = m.nearest_planet_sniper(obs, gb)
        assert isinstance(moves, list)
        env.step([moves, []])
```

- [ ] **Step 2: Run tests — expect failures**

```
python -m pytest test_33_nearest_10steps.py -v -k "agent"
```

Expected: all 5 tests FAIL with `AttributeError: module 'agent33' has no attribute 'nearest_planet_sniper'`

- [ ] **Step 3: Add `nearest_planet_sniper` to agent file**

Append to `33-Kaggle_env_nearest_10steps.py`:

```python
# ── Agent ─────────────────────────────────────────────────────────────────────

global_board = {"step": 0, "num_agents": None}


def nearest_planet_sniper(obs, global_board=global_board):
    player = obs.player if hasattr(obs, "player") else obs["player"]
    s = global_board["step"]

    if s == 0:
        initial = (
            obs.initial_planets if hasattr(obs, "initial_planets")
            else obs["initial_planets"]
        )
        owners = {p[1] for p in initial if p[1] != -1}
        global_board["num_agents"] = 4 if len(owners) > 2 else 2

    num_agents = global_board["num_agents"]
    final_step = s + NB_STEPS_SIM

    df = _simulate(obs, s, num_agents)

    consistently_mine_ids = set(
        df.query("owner == @player")
          .groupby("id")
          .filter(lambda g: len(g) == NB_STEPS_SIM)["id"]
    )

    target_ids_at_final = set(
        df.query("step == @final_step and owner != @player")["id"]
    )

    planets_raw = obs.planets if hasattr(obs, "planets") else obs["planets"]
    current = {p[0]: p for p in planets_raw}
    angular_velocity = (
        obs.angular_velocity if hasattr(obs, "angular_velocity")
        else obs.get("angular_velocity", 0.0)
    )

    moves = []
    for p in planets_raw:
        pid, owner, x, y, radius, ships = p[0], p[1], p[2], p[3], p[4], p[5]
        if owner != player or pid not in consistently_mine_ids:
            continue

        best_tgt, best_eta = None, 9999
        for tid in target_ids_at_final:
            if tid not in current:
                continue
            eta = _eta(x, y, radius, current[tid], angular_velocity)
            if eta < best_eta:
                best_eta, best_tgt = eta, current[tid]

        if best_tgt is None:
            continue

        tid = best_tgt[0]
        ships_needed = int(
            df.query("id == @tid and step == @final_step")["ships"].iloc[0]
        ) + 1
        min_ships = df.query(f"id == {pid}")["ships"].min()

        if min_ships >= ships_needed:
            angle = _aim_angle(x, y, radius, best_tgt, angular_velocity, ships_needed)
            moves.append([pid, angle, int(ships_needed)])

    global_board["step"] += 1
    return moves


agent = nearest_planet_sniper
```

- [ ] **Step 4: Run all tests — expect pass**

```
python -m pytest test_33_nearest_10steps.py -v
```

Expected: all 18 tests PASS

- [ ] **Step 5: Commit**

```bash
git add 33-Kaggle_env_nearest_10steps.py test_33_nearest_10steps.py
git commit -m "feat: complete 33-nearest-10steps agent with tests"
```
