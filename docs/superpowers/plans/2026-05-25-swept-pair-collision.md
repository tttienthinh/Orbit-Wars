# Swept-Pair Collision Update Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Update `66-One_angle_polars_updated.py` to match kaggle-environments 1.29.3 — replacing static-planet collision detection with continuous swept-pair detection in both the interpreter and `take_action`.

**Architecture:** Two independent change surfaces: (1) the Python interpreter loop — add `swept_pair_hit`, precompute `planet_paths` before fleet movement, apply positions after; (2) the Polars `take_action` pipeline — add prev-pos join, replace `dist_min/dist_max` collision with the quadratic swept-pair test, derive angle cone from contact-time planet positions t1/t2.

**Tech Stack:** Python 3.11, Polars (lazy), math stdlib, kaggle-environments 1.29.3

---

## File Map

| File | Action |
|---|---|
| `66-One_angle_polars_updated.py` | All changes |
| `tests/test_swept_pair.py` | Create — unit + integration tests |

---

### Task 1: Add `swept_pair_hit` + unit tests

**Files:**
- Modify: `66-One_angle_polars_updated.py` — insert after line 51 (end of `point_to_segment_distance`)
- Create: `tests/test_swept_pair.py`

- [ ] **Step 1: Create `tests/test_swept_pair.py`**

```python
import math, importlib.util, os

def load_module():
    path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "66-One_angle_polars_updated.py"))
    spec = importlib.util.spec_from_file_location("agent66", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def test_swept_pair_static_inside():
    mod = load_module()
    assert mod.swept_pair_hit((0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0), 1.0)

def test_swept_pair_static_miss():
    mod = load_module()
    assert not mod.swept_pair_hit((0.0, 0.0), (0.0, 0.0), (5.0, 0.0), (5.0, 0.0), 1.0)

def test_swept_pair_fleet_through_static_planet():
    mod = load_module()
    assert mod.swept_pair_hit((0.0, 0.0), (10.0, 0.0), (5.0, 0.0), (5.0, 0.0), 0.5)

def test_swept_pair_fleet_misses_static_planet():
    mod = load_module()
    assert not mod.swept_pair_hit((0.0, 0.0), (10.0, 0.0), (5.0, 2.0), (5.0, 2.0), 0.5)

def test_swept_pair_tunneling_detected():
    """Fleet and planet cross paths — old point_to_segment_distance misses, swept detects."""
    mod = load_module()
    A, B   = (0.0, 0.5), (2.0, 0.5)   # fleet moves right at y=0.5
    P0, P1 = (1.0, 1.5), (1.0, -0.5)  # planet moves down through y=0.5
    r = 0.6
    assert mod.swept_pair_hit(A, B, P0, P1, r)
    # Confirm old static check would miss
    assert mod.point_to_segment_distance(P0, A, B) > r
    assert mod.point_to_segment_distance(P1, A, B) > r
```

- [ ] **Step 2: Run — expect AttributeError (function absent)**

```
cd "C:\Users\trant\Documents\Programmation\Orbit Wars"
pytest tests/test_swept_pair.py -v
```
Expected: `AttributeError: module 'agent66' has no attribute 'swept_pair_hit'`

- [ ] **Step 3: Insert `swept_pair_hit` in `66-One_angle_polars_updated.py` after line 51**

```python
def swept_pair_hit(A, B, P0, P1, r):
    """True iff a fleet moving A->B and a planet moving P0->P1 come within r
    of each other for some t in [0, 1]."""
    d0x, d0y = A[0] - P0[0], A[1] - P0[1]
    dvx = (B[0] - A[0]) - (P1[0] - P0[0])
    dvy = (B[1] - A[1]) - (P1[1] - P0[1])
    a = dvx * dvx + dvy * dvy
    b = 2.0 * (d0x * dvx + d0y * dvy)
    c = d0x * d0x + d0y * d0y - r * r
    if a < 1e-12:
        return c <= 0.0
    disc = b * b - 4.0 * a * c
    if disc < 0.0:
        return False
    sq = math.sqrt(disc)
    t1 = (-b - sq) / (2.0 * a)
    t2 = (-b + sq) / (2.0 * a)
    return t2 >= 0.0 and t1 <= 1.0
```

- [ ] **Step 4: Run — all 5 pass**

```
pytest tests/test_swept_pair.py -v
```
Expected: 5 PASSED

- [ ] **Step 5: Commit**

```bash
git add "66-One_angle_polars_updated.py" tests/test_swept_pair.py
git commit -m "feat: add swept_pair_hit with unit tests"
```

---

### Task 2: Refactor interpreter

Replace lines 105–197 of `66-One_angle_polars_updated.py` (fleet movement + planet rotation + sweep_fleets) with the precompute-then-apply pattern.

**Files:**
- Modify: `66-One_angle_polars_updated.py` lines 105–197

- [ ] **Step 1: Add interpreter integration test to `tests/test_swept_pair.py`**

Append:

```python
def make_obs(planets, fleets, initial_planets, angular_velocity=0.0):
    class Obs:
        comets = []
        comet_planet_ids = []
        next_fleet_id = 100
    obs = Obs()
    obs.planets = [list(p) for p in planets]
    obs.fleets = [list(f) for f in fleets]
    obs.initial_planets = [list(p) for p in initial_planets]
    obs.angular_velocity = angular_velocity
    return obs

def test_interpreter_fleet_hits_static_planet():
    """Basic sanity: fleet aimed at static planet is caught."""
    mod = load_module()
    # Planet id=0, owner=1, at (56,50), radius=2 — outside orbit radius (dist=6 < 48 ✓)
    planet = [0, 1, 56.0, 50.0, 2.0, 10, 1]
    # Fleet aimed east at speed ~6 (100 ships), starting just west of planet
    fleet = [0, 0, 49.0, 50.0, 0.0, -1, 100]
    obs = make_obs([planet], [fleet], [planet], angular_velocity=0.0)
    result = mod.interpreter(obs, [[], []], 1, 2)
    assert len(result["fleets"]) == 0, "Fleet must be removed after hitting planet"

def test_interpreter_rotating_planet_sweeps_fleet():
    """Rotating planet sweeps through a stationary fleet — must be caught."""
    mod = load_module()
    # Planet orbits at radius 10 from sun (50,50), starts at (50,40)
    # angular_velocity = pi/2 → moves to (60,50) after 1 tick
    # Fleet sits at (55, 45) — on the chord from (50,40) to (60,50)
    planet = [0, 1, 50.0, 40.0, 2.0, 10, 1]
    # Fleet with 1 ship → speed=1, barely moves; place it on the planet's chord
    fleet = [0, 0, 55.0, 45.0, 0.0, -1, 1]
    obs = make_obs([planet], [fleet], [planet], angular_velocity=math.pi / 2)
    result = mod.interpreter(obs, [[], []], 1, 2)
    assert len(result["fleets"]) == 0, "Fleet on planet sweep path must be caught"
```

- [ ] **Step 2: Run — second test fails (old interpreter misses sweep)**

```
pytest tests/test_swept_pair.py::test_interpreter_rotating_planet_sweeps_fleet -v
```
Expected: FAIL (old code uses static planet position, misses the sweep)

- [ ] **Step 3: Replace lines 105–197 in `66-One_angle_polars_updated.py`**

Delete from `max_speed = MAX_SPEED` (line 105) through the end of the comet movement block (line 197, ending `obs0.comets = [g for g in obs0.comets if g["planet_ids"]]`).

Insert this replacement:

```python
    max_speed = MAX_SPEED
    fleets_to_remove = []
    combat_lists = {p[0]: [] for p in obs0.planets}

    angular_velocity = obs0.angular_velocity
    comet_pid_set = set(obs0.comet_planet_ids)
    initial_by_id = {p[0]: p for p in obs0.initial_planets}

    planet_paths = {}
    for planet in obs0.planets:
        if planet[0] in comet_pid_set:
            continue
        p_old = (planet[2], planet[3])
        p_new = p_old
        initial_p = initial_by_id.get(planet[0])
        if initial_p is not None:
            dx_p = initial_p[2] - CENTER
            dy_p = initial_p[3] - CENTER
            r_p = math.sqrt(dx_p ** 2 + dy_p ** 2)
            if r_p + planet[4] < ROTATION_RADIUS_LIMIT:
                initial_angle = math.atan2(dy_p, dx_p)
                current_angle = initial_angle + angular_velocity * step
                p_new = (
                    CENTER + r_p * math.cos(current_angle),
                    CENTER + r_p * math.sin(current_angle),
                )
        planet_paths[planet[0]] = (p_old, p_new)

    for fleet in obs0.fleets:
        angle = fleet[4]
        ships = fleet[6]
        speed = 1.0 + (max_speed - 1.0) * (math.log(ships) / math.log(1000)) ** 1.5
        speed = min(speed, max_speed)
        f_old = (fleet[2], fleet[3])
        fleet[2] += math.cos(angle) * speed
        fleet[3] += math.sin(angle) * speed
        f_new = (fleet[2], fleet[3])

        hit_planet = False
        for planet in obs0.planets:
            path = planet_paths.get(planet[0])
            if path is None:
                continue
            p_old, p_new = path
            if swept_pair_hit(f_old, f_new, p_old, p_new, planet[4]):
                combat_lists[planet[0]].append(fleet)
                fleets_to_remove.append(fleet)
                hit_planet = True
                break
        if hit_planet:
            continue
        if not (0 <= fleet[2] <= BOARD_SIZE and 0 <= fleet[3] <= BOARD_SIZE):
            fleets_to_remove.append(fleet)
            continue
        if point_to_segment_distance((CENTER, CENTER), f_old, f_new) < SUN_RADIUS:
            fleets_to_remove.append(fleet)
            continue

    for planet in obs0.planets:
        path = planet_paths.get(planet[0])
        if path is not None:
            planet[2], planet[3] = path[1]

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
                c_old = (planet[2], planet[3])
                planet[2] = p_path[idx][0]
                planet[3] = p_path[idx][1]
                if c_old[0] >= 0:
                    c_new = (planet[2], planet[3])
                    for fleet in obs0.fleets:
                        if fleet not in fleets_to_remove:
                            if point_to_segment_distance((fleet[2], fleet[3]), c_old, c_new) < planet[4]:
                                combat_lists[planet[0]].append(fleet)
                                fleets_to_remove.append(fleet)

    if expired_comet_pids:
        expired_set = set(expired_comet_pids)
        obs0.planets = [p for p in obs0.planets if p[0] not in expired_set]
        obs0.initial_planets = [p for p in obs0.initial_planets if p[0] not in expired_set]
        obs0.comet_planet_ids = [pid for pid in obs0.comet_planet_ids if pid not in expired_set]
        for group in obs0.comets:
            group["planet_ids"] = [pid for pid in group["planet_ids"] if pid not in expired_set]
        obs0.comets = [g for g in obs0.comets if g["planet_ids"]]
```

- [ ] **Step 4: Run all interpreter tests**

```
pytest tests/test_swept_pair.py -v
```
Expected: all 7 tests PASSED

- [ ] **Step 5: Commit**

```bash
git add "66-One_angle_polars_updated.py" tests/test_swept_pair.py
git commit -m "feat: refactor interpreter with swept_pair_hit and planet_paths precomputation"
```

---

### Task 3: Add `prev_pos_lf` join in `take_action`

**Files:**
- Modify: `66-One_angle_polars_updated.py` — Section 1 + Section 3 join chain

- [ ] **Step 1: Add smoke test to `tests/test_swept_pair.py`**

Append:

```python
def test_prev_pos_join_present():
    """After the prev-pos join, pa_lf intermediate should have x_prev/y_prev."""
    import polars as pl, pandas as pd
    mod = load_module()
    # Minimal df: two steps of one planet
    rows = [
        {"step": 0, "id": 1, "x": 50.0, "y": 40.0, "radius": 2.0,
         "ships": 10, "production": 1, "owner": 0, "nature": "moving"},
        {"step": 1, "id": 1, "x": 51.0, "y": 41.0, "radius": 2.0,
         "ships": 11, "production": 1, "owner": 0, "nature": "moving"},
    ]
    df = pd.DataFrame(rows)
    df_lf = pl.from_pandas(df).sort("step").lazy()
    prev_pos_lf = (
        df_lf.select(["id", "step", "x", "y"])
        .rename({"x": "x_prev", "y": "y_prev"})
        .with_columns((pl.col("step") + 1).alias("step"))
    )
    joined = df_lf.join(prev_pos_lf, on=["id", "step"], how="left").collect()
    row_step1 = joined.filter(pl.col("step") == 1)
    assert row_step1["x_prev"][0] == 50.0
    assert row_step1["y_prev"][0] == 40.0
```

- [ ] **Step 2: Run — should pass immediately (pure Polars, no file dependency)**

```
pytest tests/test_swept_pair.py::test_prev_pos_join_present -v
```
Expected: PASS

- [ ] **Step 3: Add `prev_pos_lf` in Section 1 of `take_action`**

In `66-One_angle_polars_updated.py`, after line 298 (`df_lf = pl.from_pandas(df).sort("step").lazy()`), insert:

```python
    prev_pos_lf = (
        df_lf.select(["id", "step", "x", "y"])
        .rename({"x": "x_prev", "y": "y_prev"})
        .with_columns((pl.col("step") + 1).alias("step"))
    )
```

- [ ] **Step 4: Add the join in the `pa_lf` chain (Section 3)**

In the `pa_lf` chain, after `.join(df_lf, how="cross")`, insert:

```python
        .join(prev_pos_lf, on=["id", "step"], how="left")
```

- [ ] **Step 5: Commit**

```bash
git add "66-One_angle_polars_updated.py" tests/test_swept_pair.py
git commit -m "feat: add prev_pos_lf join for swept-pair contact computation"
```

---

### Task 4: Replace Section 3 — swept-pair collision + t1/t2 angle cone

This is the main Polars rewrite. Replace the expression block (lines 334–386) entirely.

**Files:**
- Modify: `66-One_angle_polars_updated.py` lines 334–386

- [ ] **Step 1: Add regression test to `tests/test_swept_pair.py`**

Append:

```python
def test_take_action_produces_valid_moves():
    """take_action must return a list of [planet_id, angle, ships] triples."""
    import copy, math
    mod = load_module()

    class MockObs:
        angular_velocity = 0.01
        comets = []
        comet_planet_ids = []
        next_fleet_id = 10

    obs = MockObs()
    # Player 0 owns planet 0; planet 1 is neutral
    obs.planets = [[0, 0, 30.0, 50.0, 3.0, 50, 2], [1, -1, 70.0, 50.0, 3.0, 5, 1]]
    obs.initial_planets = copy.deepcopy(obs.planets)
    obs.fleets = []

    df = mod._simulate(obs, 0, 2, n_steps=mod.NB_STEPS_SIM)
    moves = mod.take_action(df, player_id=0)

    assert isinstance(moves, list)
    for move in moves:
        pid, angle, ships = move
        assert isinstance(pid, (int, float))
        assert -math.pi <= angle <= math.pi or 0 <= angle <= 2 * math.pi
        assert ships > 0
```

- [ ] **Step 2: Run — should pass with current code (establishes baseline)**

```
pytest tests/test_swept_pair.py::test_take_action_produces_valid_moves -v
```
Expected: PASS (verifies baseline before rewrite)

- [ ] **Step 3: Replace Section 3 expression block and `pa_lf` chain**

Delete lines 334–386 (from `dx = pl.col("x") - pl.col("x_src")` through `.sort("step")`).

Replace with:

```python
    # ── Section 3: cross-join + swept-pair collision + sun (all lazy) ────────
    dx = pl.col("x") - pl.col("x_src")
    dy = pl.col("y") - pl.col("y_src")
    l2 = dx.pow(2) + dy.pow(2)
    dist_tgt_src = l2.sqrt()
    safe_dist = pl.when(dist_tgt_src < 1e-9).then(pl.lit(1.0)).otherwise(dist_tgt_src)
    step_diff = pl.col("step") - pl.col("step_src")
    fleet_speed = 1.0 + (MAX_SPEED - 1.0) * (
        pl.col("ships_sent").cast(pl.Float64).log(base=math.e) / math.log(1000.0)
    ).pow(1.5)
    dist_min = step_diff * fleet_speed + PLANET_MARGIN + pl.col("radius_src")
    dist_prev = dist_min - fleet_speed   # fleet distance at t=0 (start of tick)

    dot = (CENTER - pl.col("x_src")) * dx + (CENTER - pl.col("y_src")) * dy
    t_sun = (dot / pl.when(l2 == 0).then(pl.lit(1.0)).otherwise(l2)).clip(0.0, 1.0)
    proj_dist_sun = (
        (CENTER - pl.col("x_src") - t_sun * dx).pow(2) +
        (CENTER - pl.col("y_src") - t_sun * dy).pow(2)
    ).sqrt()
    crossing_sun = pl.when(l2 == 0).then(
        ((CENTER - pl.col("x_src")).pow(2) + (CENTER - pl.col("y_src")).pow(2)).sqrt()
    ).otherwise(proj_dist_sun) < (SUN_RADIUS + PLANET_MARGIN)

    unit_x = dx / safe_dist
    unit_y = dy / safe_dist
    fleet_x0  = pl.col("x_src") + unit_x * dist_prev
    fleet_y0  = pl.col("y_src") + unit_y * dist_prev
    planet_vx = pl.col("x") - pl.col("x_prev")
    planet_vy = pl.col("y") - pl.col("y_prev")
    dvx_sp = unit_x * fleet_speed - planet_vx
    dvy_sp = unit_y * fleet_speed - planet_vy
    d0x_sp = fleet_x0 - pl.col("x_prev")
    d0y_sp = fleet_y0 - pl.col("y_prev")
    a_sp   = dvx_sp.pow(2) + dvy_sp.pow(2)
    b_sp   = 2.0 * (d0x_sp * dvx_sp + d0y_sp * dvy_sp)
    c_sp   = d0x_sp.pow(2) + d0y_sp.pow(2) - pl.col("radius").pow(2)
    disc_sp = b_sp.pow(2) - 4.0 * a_sp * c_sp
    sq_sp  = disc_sp.clip(lower_bound=0.0).sqrt()
    t1_expr = pl.when(a_sp < 1e-12).then(pl.lit(0.0)).otherwise((-b_sp - sq_sp) / (2.0 * a_sp))
    t2_expr = pl.when(a_sp < 1e-12).then(pl.lit(1.0)).otherwise((-b_sp + sq_sp) / (2.0 * a_sp))
    collision = pl.when(a_sp < 1e-12).then(c_sp <= 0.0).otherwise(
        (disc_sp >= 0.0) & (t2_expr >= 0.0) & (t1_expr <= 1.0)
    )

    pa_lf = (
        mine_lf
        .join(df_lf, how="cross")
        .join(prev_pos_lf, on=["id", "step"], how="left")
        .filter((pl.col("step") > pl.col("step_src")) & (pl.col("id") != pl.col("id_src")))
        .with_columns([
            dist_tgt_src.alias("dist_tgt_src"),
            step_diff.alias("step_diff"),
            fleet_speed.alias("fleet_speed"),
            dist_min.alias("dist_min"),
            dist_prev.alias("dist_prev"),
            t1_expr.alias("t1"),
            t2_expr.alias("t2"),
            collision.alias("collision"),
        ])
        .filter(pl.col("collision"))
        .filter(~crossing_sun)
        .with_columns([
            pl.col("t1").clip(0.0, 1.0).alias("t1_eff"),
            pl.col("t2").clip(0.0, 1.0).alias("t2_eff"),
        ])
        .with_columns([
            (pl.col("x_prev") + pl.col("t1_eff") * (pl.col("x") - pl.col("x_prev"))).alias("p_t1_x"),
            (pl.col("y_prev") + pl.col("t1_eff") * (pl.col("y") - pl.col("y_prev"))).alias("p_t1_y"),
            (pl.col("x_prev") + pl.col("t2_eff") * (pl.col("x") - pl.col("x_prev"))).alias("p_t2_x"),
            (pl.col("y_prev") + pl.col("t2_eff") * (pl.col("y") - pl.col("y_prev"))).alias("p_t2_y"),
        ])
        .with_columns([
            pl.arctan2(pl.col("p_t1_y") - pl.col("y_src"), pl.col("p_t1_x") - pl.col("x_src")).alias("angle_t1"),
            pl.arctan2(pl.col("p_t2_y") - pl.col("y_src"), pl.col("p_t2_x") - pl.col("x_src")).alias("angle_t2"),
            ((pl.col("p_t1_x") - pl.col("x_src")).pow(2) + (pl.col("p_t1_y") - pl.col("y_src")).pow(2)).sqrt().alias("d_s_t1"),
            ((pl.col("p_t2_x") - pl.col("x_src")).pow(2) + (pl.col("p_t2_y") - pl.col("y_src")).pow(2)).sqrt().alias("d_s_t2"),
        ])
        .with_columns([
            (pl.col("dist_prev") + pl.col("t1_eff") * pl.col("fleet_speed")).alias("d_f_t1"),
            (pl.col("dist_prev") + pl.col("t2_eff") * pl.col("fleet_speed")).alias("d_f_t2"),
        ])
        .with_columns([
            ((pl.col("d_s_t1").pow(2) + pl.col("d_f_t1").pow(2) - pl.col("radius").pow(2))
             / (2.0 * pl.col("d_s_t1") * pl.col("d_f_t1"))).clip(-1.0, 1.0).arccos().alias("angle_radius_t1"),
            ((pl.col("d_s_t2").pow(2) + pl.col("d_f_t2").pow(2) - pl.col("radius").pow(2))
             / (2.0 * pl.col("d_s_t2") * pl.col("d_f_t2"))).clip(-1.0, 1.0).arccos().alias("angle_radius_t2"),
        ])
        .with_columns([
            pl.min_horizontal(
                pl.col("angle_t1") - pl.col("angle_radius_t1"),
                pl.col("angle_t2") - pl.col("angle_radius_t2"),
            ).mod(2 * math.pi).alias("angle_min"),
            pl.max_horizontal(
                pl.col("angle_t1") + pl.col("angle_radius_t1"),
                pl.col("angle_t2") + pl.col("angle_radius_t2"),
            ).mod(2 * math.pi).alias("angle_max"),
            ((pl.col("angle_t1") + pl.col("angle_t2")) / 2.0).alias("angle"),
        ])
        .sort("step")
    )
```

**Note:** `dist_fleet_src_min` and `dist_fleet_src_max` are removed — they are no longer referenced downstream. `angle` now carries the midpoint of the contact arc; Section 4 `in_cone` uses `pl.col("angle")` unchanged.

- [ ] **Step 4: Run all tests**

```
pytest tests/test_swept_pair.py -v
```
Expected: all 9 tests PASSED

- [ ] **Step 5: Smoke-run the agent for one step**

```python
import kaggle_environments as ke
env = ke.make("orbit_wars", num_agents=2)
obs = env.reset()
import importlib.util, os
spec = importlib.util.spec_from_file_location("a", "66-One_angle_polars_updated.py")
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
moves = mod.agent(obs[0])
print("moves:", moves)
```
Expected: runs without error, prints a list (possibly empty on step 0).

- [ ] **Step 6: Commit**

```bash
git add "66-One_angle_polars_updated.py" tests/test_swept_pair.py
git commit -m "feat: swept-pair collision + t1/t2 angle cone in take_action"
```
