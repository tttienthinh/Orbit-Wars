# Polars Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create `87-Polars_Simulate10Next_Conqueror_Supplier_fixed.py` — an exact port of `84-Simulate10Next_Conqueror_Supplier_fixed.py` using Polars Lazy, with a single `.collect()` in `_04`.

**Architecture:** `GameConfig`, `PhysicsEngine`, and `interpreter()` are copied verbatim. `StrategyPipeline._01` returns `(pl.DataFrame, pl.DataFrame)`. `_02` and `_03` build and extend a `pl.LazyFrame` chain. `_04` calls `.collect()` once, then scores with eager Polars.

**Tech Stack:** Python 3.10+, Polars (Lazy API), math (stdlib)

---

## File Structure

| File | Action | Purpose |
|------|--------|---------|
| `87-Polars_Simulate10Next_Conqueror_Supplier_fixed.py` | Create | Full Polars port of `84` |

---

### Task 1: Scaffold the file — unchanged parts

**Files:**
- Create: `87-Polars_Simulate10Next_Conqueror_Supplier_fixed.py`

- [ ] **Step 1.1: Create the file with imports and unchanged classes**

```python
import math
import copy
import polars as pl


# ── Configuration ─────────────────────────────────────────────────────────────
class GameConfig:
    CENTER = 50.0
    SUN_RADIUS = 10.0
    ROTATION_RADIUS_LIMIT = 50.0
    MAX_SPEED = 6.0
    NB_STEPS_SIM = 10
    PLANET_MARGIN = 0.1
    PLANET_MOVEMENT_SLACK = 3.0


# ── Physics helpers ───────────────────────────────────────────────────────────
class PhysicsEngine:
    @staticmethod
    def distance(p1, p2):
        return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

    @staticmethod
    def point_to_segment_distance(p, v, w):
        l2 = (v[0] - w[0]) ** 2 + (v[1] - w[1]) ** 2
        if l2 == 0.0:
            return PhysicsEngine.distance(p, v)
        t = max(
            0, min(1, ((p[0] - v[0]) * (w[0] - v[0]) + (p[1] - v[1]) * (w[1] - v[1])) / l2)
        )
        projection = (v[0] + t * (w[0] - v[0]), v[1] + t * (w[1] - v[1]))
        return PhysicsEngine.distance(p, projection)

    @staticmethod
    def swept_pair_hit(A, B, P0, P1, r):
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

    @staticmethod
    def fleet_speed(ships):
        if ships <= 1:
            return 1.0
        ratio = math.log(ships) / math.log(1000.0)
        return 1.0 + (GameConfig.MAX_SPEED - 1.0) * max(0.0, min(1.0, ratio)) ** 1.5


CENTER = GameConfig.CENTER
SUN_RADIUS = GameConfig.SUN_RADIUS
ROTATION_RADIUS_LIMIT = GameConfig.ROTATION_RADIUS_LIMIT

BOARD_SIZE = 100.0
MAX_NB_STEP = 500
```

Copy `interpreter()` verbatim from `84-Simulate10Next_Conqueror_Supplier_fixed.py` lines 72–269. Then add the `StrategyPipeline` class shell:

```python
# ── Strategy Pipeline ─────────────────────────────────────────────────────────
class StrategyPipeline:

    @staticmethod
    def _01_get_obs_dataframe(obs, step: int, num_agents: int) -> tuple:
        raise NotImplementedError

    @staticmethod
    def _02_get_all_opportunities(
        df_s: pl.DataFrame,
        planet_disp: pl.DataFrame,
        player_id: int,
    ) -> pl.LazyFrame:
        raise NotImplementedError

    @staticmethod
    def _03_filter_collision(pa_lf: pl.LazyFrame) -> pl.LazyFrame:
        raise NotImplementedError

    @staticmethod
    def _04_score_and_decide(safe_lf: pl.LazyFrame, player_id: int) -> list:
        raise NotImplementedError


# ── Entry point ───────────────────────────────────────────────────────────────
step = 0
num_agents = None
player_id = None


def agent(obs):
    global step, num_agents, player_id

    if num_agents is None:
        initial = (
            obs.initial_planets if hasattr(obs, "initial_planets")
            else obs["initial_planets"]
        )
        owners = {p[1] for p in initial if p[1] != -1}
        num_agents = 4 if len(owners) > 2 else 2
    if player_id is None:
        player_id = obs.get("player", 0) if isinstance(obs, dict) else obs.player

    df_s, planet_disp = StrategyPipeline._01_get_obs_dataframe(obs, step, num_agents)
    pa_lf = StrategyPipeline._02_get_all_opportunities(df_s, planet_disp, player_id)
    safe_lf = StrategyPipeline._03_filter_collision(pa_lf)
    moves = StrategyPipeline._04_score_and_decide(safe_lf, player_id)

    step += 1
    return moves
```

- [ ] **Step 1.2: Verify the file is syntactically valid**

```bash
python -c "import ast; ast.parse(open('87-Polars_Simulate10Next_Conqueror_Supplier_fixed.py').read()); print('OK')"
```

Expected: `OK`

- [ ] **Step 1.3: Commit scaffold**

```bash
git add 87-Polars_Simulate10Next_Conqueror_Supplier_fixed.py
git commit -m "feat: scaffold 87-Polars pipeline — unchanged parts + stubs"
```

---

### Task 2: Implement `_01_get_obs_dataframe`

**Files:**
- Modify: `87-Polars_Simulate10Next_Conqueror_Supplier_fixed.py`

Replace the `_01_get_obs_dataframe` stub with:

- [ ] **Step 2.1: Implement `_01`**

```python
@staticmethod
def _01_get_obs_dataframe(obs, step: int, num_agents: int) -> tuple:
    sim = copy.deepcopy(obs)
    no_actions = [[] for _ in range(num_agents)]
    rows = []
    for i in range(GameConfig.NB_STEPS_SIM + 1):
        for p in sim.planets:
            pid, owner, x, y, radius, ships, production = (
                p[0], p[1], p[2], p[3], p[4], p[5], p[6]
            )
            r = math.hypot(x - GameConfig.CENTER, y - GameConfig.CENTER)
            if pid in sim.comet_planet_ids:
                nature = "comet"
            elif r + radius < GameConfig.ROTATION_RADIUS_LIMIT:
                nature = "moving"
            else:
                nature = "fix"
            rows.append({
                "step": step + i,
                "id": pid,
                "x": x,
                "y": y,
                "radius": radius,
                "ships": ships,
                "production": production,
                "owner": owner,
                "nature": nature,
            })
        interpreter(sim, no_actions, step + i, num_agents)

    df_s = pl.DataFrame(rows).sort("step")

    prev_pos = (
        df_s.lazy()
        .select(["id", "step", "x", "y"])
        .rename({"x": "x_prev", "y": "y_prev"})
        .with_columns((pl.col("step") + 1).alias("step"))
    )
    planet_disp = (
        df_s.lazy()
        .select(["id", "step", "x", "y"])
        .join(prev_pos, on=["id", "step"], how="left")
        .with_columns(
            ((pl.col("x") - pl.col("x_prev").fill_null(pl.col("x"))).pow(2) +
             (pl.col("y") - pl.col("y_prev").fill_null(pl.col("y"))).pow(2)
            ).sqrt().alias("planet_disp")
        )
        .select(["id", "step", "planet_disp"])
        .collect()
    )
    return df_s, planet_disp
```

- [ ] **Step 2.2: Smoke test `_01`**

Run this in a Python shell or notebook cell:

```python
import math, copy
# %run 87-Polars_Simulate10Next_Conqueror_Supplier_fixed.py  # in notebook
# or: exec(open('87-Polars_Simulate10Next_Conqueror_Supplier_fixed.py').read())

class Obs:
    def __init__(self, planets, angular_velocity=0.0):
        self.planets = [list(p) for p in planets]
        self.initial_planets = [list(p) for p in planets]
        self.fleets = []
        self.next_fleet_id = 100
        self.comets = []
        self.comet_planet_ids = []
        self.angular_velocity = angular_velocity

obs = Obs([
    [0, 0, 5.0, 5.0, 1 + math.log(3), 30, 3],
    [1, 1, 80.0, 20.0, 1 + math.log(3), 50, 2],
], angular_velocity=0.05)

df_s, planet_disp = StrategyPipeline._01_get_obs_dataframe(obs, step=0, num_agents=2)
assert df_s.schema["step"] == pl.Int64
assert "planet_disp" in planet_disp.columns
assert len(df_s) == (GameConfig.NB_STEPS_SIM + 1) * 2  # 11 steps × 2 planets
print("_01 OK:", df_s.shape, planet_disp.shape)
```

Expected: `_01 OK: (22, 9) (22, 3)` (or similar non-zero shapes)

- [ ] **Step 2.3: Commit**

```bash
git add 87-Polars_Simulate10Next_Conqueror_Supplier_fixed.py
git commit -m "feat: implement _01_get_obs_dataframe with Polars"
```

---

### Task 3: Implement `_02_get_all_opportunities`

**Files:**
- Modify: `87-Polars_Simulate10Next_Conqueror_Supplier_fixed.py`

Replace the `_02_get_all_opportunities` stub with:

- [ ] **Step 3.1: Implement `_02`**

```python
@staticmethod
def _02_get_all_opportunities(
    df_s: pl.DataFrame,
    planet_disp: pl.DataFrame,
    player_id: int,
) -> pl.LazyFrame:
    df_s_lf = df_s.lazy()
    planet_disp_lf = planet_disp.lazy()

    mine_base_lf = (
        df_s_lf
        .with_columns(
            pl.when(pl.col("owner") == player_id).then(1).otherwise(0).alias("is_mine")
        )
        .group_by("id", maintain_order=True)
        .agg([
            pl.first("step").alias("step_src"),
            pl.first("x").alias("x_src"),
            pl.first("y").alias("y_src"),
            pl.first("radius").alias("radius_src"),
            pl.min("ships").alias("ships_min"),
            pl.first("production").alias("production_src"),
            pl.first("nature").alias("nature_src"),
            pl.first("owner").alias("owner_src"),
            pl.len().alias("row_count"),
            pl.sum("is_mine").alias("is_mine"),
        ])
        .filter(
            (pl.col("row_count") == pl.col("is_mine")) & (pl.col("owner_src") == player_id)
        )
        .rename({"id": "id_src"})
    )

    # Phase A: planet-level cross-join with sun-crossing filter
    dx = pl.col("x") - pl.col("x_src")
    dy = pl.col("y") - pl.col("y_src")
    l2 = dx.pow(2) + dy.pow(2)
    dist_tgt_src = l2.sqrt()
    step_diff = (pl.col("step") - pl.col("step_src")).cast(pl.Float64)

    dot = (GameConfig.CENTER - pl.col("x_src")) * dx + (GameConfig.CENTER - pl.col("y_src")) * dy
    t_sun = (dot / pl.when(l2 == 0).then(pl.lit(1.0)).otherwise(l2)).clip(0.0, 1.0)
    proj_dist_sun = (
        (GameConfig.CENTER - pl.col("x_src") - t_sun * dx).pow(2) +
        (GameConfig.CENTER - pl.col("y_src") - t_sun * dy).pow(2)
    ).sqrt()
    crossing_sun = pl.when(l2 == 0).then(
        ((GameConfig.CENTER - pl.col("x_src")).pow(2) +
         (GameConfig.CENTER - pl.col("y_src")).pow(2)).sqrt()
    ).otherwise(proj_dist_sun) < (GameConfig.SUN_RADIUS + GameConfig.PLANET_MARGIN)

    coarse_lf = (
        mine_base_lf
        .join(df_s_lf, how="cross")
        .filter(
            (pl.col("step") > pl.col("step_src")) & (pl.col("id") != pl.col("id_src"))
        )
        .join(planet_disp_lf, on=["id", "step"], how="left")
        .with_columns([
            dist_tgt_src.alias("dist_tgt_src"),
            step_diff.alias("step_diff"),
        ])
        .filter(
            (pl.col("dist_tgt_src") <
             (pl.col("step_diff") + 1) * GameConfig.MAX_SPEED
             + pl.col("radius_src") + GameConfig.PLANET_MARGIN + pl.col("radius")
             + pl.col("planet_disp").fill_null(0.0))
            & ~crossing_sun
        )
    )

    # Ships_sent expansion
    nb_steps_sim = GameConfig.NB_STEPS_SIM
    expanded_lf = (
        coarse_lf
        .with_columns(
            pl.int_ranges(
                1,
                pl.col("ships_min") + pl.col("production_src") * nb_steps_sim + 1,
                dtype=pl.Int64,
            ).alias("ships_sent")
        )
        .explode("ships_sent")
    )

    # Phase B: fleet-speed filter
    fleet_speed_expr = 1.0 + (GameConfig.MAX_SPEED - 1.0) * (
        pl.col("ships_sent").cast(pl.Float64).log(base=math.e) / math.log(1000.0)
    ).clip(lower_bound=0.0).pow(1.5)
    dist_min_expr = pl.col("step_diff") * fleet_speed_expr + GameConfig.PLANET_MARGIN + pl.col("radius_src")
    dist_prev_expr = dist_min_expr - fleet_speed_expr

    prev_pos_lf = (
        df_s_lf.select(["id", "step", "x", "y"])
        .rename({"x": "x_prev", "y": "y_prev"})
        .with_columns((pl.col("step") + 1).alias("step"))
    )

    # Swept-pair collision (quadratic discriminant)
    unit_x = (pl.col("x") - pl.col("x_src")) / pl.when(
        pl.col("dist_tgt_src") < 1e-9
    ).then(pl.lit(1.0)).otherwise(pl.col("dist_tgt_src"))
    unit_y = (pl.col("y") - pl.col("y_src")) / pl.when(
        pl.col("dist_tgt_src") < 1e-9
    ).then(pl.lit(1.0)).otherwise(pl.col("dist_tgt_src"))
    fleet_x0 = pl.col("x_src") + unit_x * pl.col("dist_prev")
    fleet_y0 = pl.col("y_src") + unit_y * pl.col("dist_prev")
    planet_vx = pl.col("x") - pl.col("x_prev").fill_null(pl.col("x"))
    planet_vy = pl.col("y") - pl.col("y_prev").fill_null(pl.col("y"))
    dvx_sp = unit_x * pl.col("fleet_speed") - planet_vx
    dvy_sp = unit_y * pl.col("fleet_speed") - planet_vy
    d0x_sp = fleet_x0 - pl.col("x_prev").fill_null(pl.col("x"))
    d0y_sp = fleet_y0 - pl.col("y_prev").fill_null(pl.col("y"))
    a_sp = dvx_sp.pow(2) + dvy_sp.pow(2)
    b_sp = 2.0 * (d0x_sp * dvx_sp + d0y_sp * dvy_sp)
    c_sp = d0x_sp.pow(2) + d0y_sp.pow(2) - pl.col("radius").pow(2)
    disc_sp = b_sp.pow(2) - 4.0 * a_sp * c_sp
    sq_sp = disc_sp.clip(lower_bound=0.0).sqrt()
    t1_expr = pl.when(a_sp < 1e-12).then(pl.lit(0.0)).otherwise((-b_sp - sq_sp) / (2.0 * a_sp))
    t2_expr = pl.when(a_sp < 1e-12).then(pl.lit(1.0)).otherwise((-b_sp + sq_sp) / (2.0 * a_sp))
    collision = pl.when(a_sp < 1e-12).then(c_sp <= 0.0).otherwise(
        (disc_sp >= 0.0) & (t2_expr >= 0.0) & (t1_expr <= 1.0)
    )

    # Angle geometry
    x_prev_f = pl.col("x_prev").fill_null(pl.col("x"))
    y_prev_f = pl.col("y_prev").fill_null(pl.col("y"))

    pa_lf = (
        expanded_lf
        .with_columns([
            fleet_speed_expr.alias("fleet_speed"),
            dist_min_expr.alias("dist_min"),
            dist_prev_expr.alias("dist_prev"),
        ])
        .filter(
            pl.col("dist_tgt_src") < pl.col("dist_min") + pl.col("fleet_speed")
            + pl.col("radius") + GameConfig.PLANET_MOVEMENT_SLACK
        )
        .join(prev_pos_lf, on=["id", "step"], how="left")
        .with_columns([
            t1_expr.alias("t1"),
            t2_expr.alias("t2"),
            collision.alias("collision"),
        ])
        .filter(pl.col("collision"))
        .with_columns([
            pl.col("t1").clip(0.0, 1.0).alias("t1_eff"),
            pl.col("t2").clip(0.0, 1.0).alias("t2_eff"),
        ])
        .with_columns([
            (x_prev_f + pl.col("t1_eff") * (pl.col("x") - x_prev_f)).alias("p_t1_x"),
            (y_prev_f + pl.col("t1_eff") * (pl.col("y") - y_prev_f)).alias("p_t1_y"),
            (x_prev_f + pl.col("t2_eff") * (pl.col("x") - x_prev_f)).alias("p_t2_x"),
            (y_prev_f + pl.col("t2_eff") * (pl.col("y") - y_prev_f)).alias("p_t2_y"),
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
            pl.arctan2(
                pl.col("angle_t1").sin() + pl.col("angle_t2").sin(),
                pl.col("angle_t1").cos() + pl.col("angle_t2").cos(),
            ).alias("angle"),
        ])
        .sort("step")
    )

    return pa_lf
```

- [ ] **Step 3.2: Smoke test `_02`**

```python
pa_lf = StrategyPipeline._02_get_all_opportunities(df_s, planet_disp, player_id=0)
assert isinstance(pa_lf, pl.LazyFrame), "Must return LazyFrame"
pa = pa_lf.collect()
assert "angle" in pa.columns
assert "angle_min" in pa.columns
assert "ships_sent" in pa.columns
assert len(pa) > 0, "Expected opportunities for a reachable enemy planet"
print("_02 OK: rows =", len(pa))
```

Expected: `_02 OK: rows = <N>` where N > 0 for the test obs.

- [ ] **Step 3.3: Commit**

```bash
git add 87-Polars_Simulate10Next_Conqueror_Supplier_fixed.py
git commit -m "feat: implement _02_get_all_opportunities with Polars Lazy"
```

---

### Task 4: Implement `_03_filter_collision`

**Files:**
- Modify: `87-Polars_Simulate10Next_Conqueror_Supplier_fixed.py`

Replace the `_03_filter_collision` stub with:

- [ ] **Step 4.1: Implement `_03`**

```python
@staticmethod
def _03_filter_collision(pa_lf: pl.LazyFrame) -> pl.LazyFrame:
    angle_norm = pl.col("angle") % (2 * math.pi)
    wraps = pl.col("angle_min_obs") > pl.col("angle_max_obs")
    in_cone = pl.when(wraps).then(
        (angle_norm >= pl.col("angle_min_obs")) | (angle_norm <= pl.col("angle_max_obs"))
    ).otherwise(
        (angle_norm >= pl.col("angle_min_obs")) & (angle_norm <= pl.col("angle_max_obs"))
    )

    blocked_lf = (
        pa_lf.select(["id_src", "ships_sent", "step", "id", "angle", "angle_min", "angle_max"])
        .join(
            pa_lf.select(["id_src", "ships_sent", "step", "id", "angle_min", "angle_max"])
                 .rename({"step": "step_obs", "id": "id_obs",
                          "angle_min": "angle_min_obs", "angle_max": "angle_max_obs"}),
            on=["id_src", "ships_sent"],
            how="inner",
        )
        .filter(
            (pl.col("step_obs") < pl.col("step")) & (pl.col("id_obs") != pl.col("id"))
        )
        .filter(in_cone)
        .select(["id_src", "ships_sent", "step", "id"])
        .unique()
    )

    return (
        pa_lf
        .join(blocked_lf, on=["id_src", "ships_sent", "step", "id"], how="anti")
        .with_columns(pl.col("angle").alias("final_angle"))
    )
```

- [ ] **Step 4.2: Smoke test `_03`**

```python
safe_lf = StrategyPipeline._03_filter_collision(pa_lf)
assert isinstance(safe_lf, pl.LazyFrame), "Must return LazyFrame"
safe = safe_lf.collect()
assert "final_angle" in safe.columns
assert len(safe) <= len(pa), "Filtering can only reduce or keep rows"
print("_03 OK: rows before =", len(pa), "after =", len(safe))
```

Expected: rows after ≤ rows before, `final_angle` present.

- [ ] **Step 4.3: Commit**

```bash
git add 87-Polars_Simulate10Next_Conqueror_Supplier_fixed.py
git commit -m "feat: implement _03_filter_collision with Polars Lazy anti-join"
```

---

### Task 5: Implement `_04_score_and_decide`

**Files:**
- Modify: `87-Polars_Simulate10Next_Conqueror_Supplier_fixed.py`

Replace the `_04_score_and_decide` stub with:

- [ ] **Step 5.1: Implement `_04`**

```python
@staticmethod
def _04_score_and_decide(safe_lf: pl.LazyFrame, player_id: int) -> list:
    attacks_with_angle = safe_lf.collect()
    if attacks_with_angle.is_empty():
        return []

    moves = []

    # Comet evasion
    awa_comets = attacks_with_angle.filter(pl.col("nature_src") == "comet")
    if not awa_comets.is_empty():
        x_off = (awa_comets["x_src"] - GameConfig.CENTER).abs().max() or 0
        y_off = (awa_comets["y_src"] - GameConfig.CENTER).abs().max() or 0
        if max(x_off, y_off) > 45:
            moves += [list(r) for r in (
                awa_comets
                .filter(pl.col("ships_sent") <= pl.col("ships_min"))
                .sort(["ships_sent", "step"], descending=[True, False])
                .group_by("id_src", maintain_order=True)
                .first()
                .select(["id_src", "final_angle", "ships_sent"])
                .rows()
            )]
            id_to_avoid = awa_comets["id_src"].unique().to_list()
            attacks_with_angle = attacks_with_angle.filter(~pl.col("id_src").is_in(id_to_avoid))

    if attacks_with_angle.is_empty():
        return moves

    # Top-5 targets per source planet
    top5_ids = (
        attacks_with_angle
        .sort(["step", "ships_sent"])
        .group_by(["id_src", "id"], maintain_order=True)
        .first()
        .sort(["step", "ships_sent"])
        .group_by("id_src", maintain_order=True)
        .head(5)
        .select(["id_src", "id"])
        .with_columns(pl.lit(True).alias("is_top5"))
    )

    mine_src_ids = attacks_with_angle["id_src"].unique().to_list()

    # Classify Supplier / Conqueror
    src_nature = (
        top5_ids
        .with_columns(pl.col("id").is_in(mine_src_ids).alias("target_is_mine"))
        .group_by("id_src")
        .agg([
            pl.col("target_is_mine").sum().alias("mine_count"),
            pl.len().alias("total_count"),
        ])
        .with_columns(
            pl.when(pl.col("mine_count") == pl.col("total_count"))
            .then(pl.lit("Supplier"))
            .otherwise(pl.lit("Conqueror"))
            .alias("status")
        )
    )
    conqueror_ids = src_nature.filter(pl.col("status") == "Conqueror")["id_src"].to_list()
    supplier_ids = src_nature.filter(pl.col("status") == "Supplier")["id_src"].to_list()

    # ── Conqueror: attack enemy/neutral planets ──────────────────────────────
    attacks_conqueror = pl.DataFrame()
    conqueror_needs = None
    if conqueror_ids:
        _c = (
            attacks_with_angle
            .filter(pl.col("id_src").is_in(conqueror_ids))
            .join(top5_ids.select(["id_src", "id", "is_top5"]), on=["id_src", "id"], how="left")
            .with_columns(pl.col("is_top5").fill_null(False))
            .filter(pl.col("is_top5"))
            .filter(pl.col("owner") != player_id)
            .with_columns(
                pl.when(pl.col("owner") == -1)
                .then(pl.col("ships"))
                .otherwise(pl.col("ships") + pl.col("production"))
                .alias("ships_needed")
            )
            .filter(
                (pl.col("ships_needed") + 1 <= pl.col("ships_sent")) &
                (pl.col("ships_sent") <= pl.col("ships_needed") + pl.col("production_src") + 1)
            )
            .sort(["step", "ships_sent"])
            .group_by(["id_src", "id"], maintain_order=True)
            .first()
            .with_columns(
                (pl.col("ships_needed") / pl.col("production_src")).alias("time_cost")
            )
        )
        if not _c.is_empty():
            conqueror_needs = (
                _c
                .group_by("id_src", maintain_order=True)
                .agg([
                    pl.col("ships_min").min().alias("ship_min"),
                    pl.col("ships_sent").sum().alias("all_need"),
                    pl.col("ships_sent").min().alias("lowest_need"),
                    pl.len().alias("nb_need"),
                ])
            )
            total_tc = _c.group_by("id_src").agg(
                pl.col("time_cost").sum().alias("total_time_cost")
            )
            attacks_conqueror = (
                _c
                .join(total_tc, on="id_src", how="left")
                .with_columns(
                    ((pl.col("total_time_cost") - pl.col("time_cost") - pl.col("step_diff"))
                     * pl.col("production")).alias("score")
                )
                .sort("score", descending=True)
                .group_by("id_src", maintain_order=True)
                .first()
                .filter(pl.col("ships_sent") <= pl.col("ships_min"))
            )

    # ── Supplier: reinforce own planets ─────────────────────────────────────
    attacks_supplier = pl.DataFrame()
    if supplier_ids and conqueror_needs is not None:
        _s = (
            attacks_with_angle
            .filter(pl.col("id_src").is_in(supplier_ids))
            .join(top5_ids.select(["id_src", "id", "is_top5"]), on=["id_src", "id"], how="left")
            .with_columns(pl.col("is_top5").fill_null(False))
            .filter(pl.col("is_top5"))
            .with_columns(pl.col("id").is_in(supplier_ids).alias("target_is_supplier"))
            .filter(~pl.col("target_is_supplier"))
            .join(
                conqueror_needs.rename({"id_src": "id"}),
                on="id",
                how="right",
            )
            .filter((pl.col("lowest_need") - pl.col("ships_min")) * 1.5 < pl.col("ships_sent"))
            .filter(
                (pl.col("ships_min") * 0.75 < pl.col("ships_sent")) &
                (pl.col("ships_sent") < pl.col("ships_min"))
            )
            .sort(["all_need", "ships_sent"], descending=[True, False])
            .group_by("id_src", maintain_order=True)
            .first()
        )
        attacks_supplier = _s

    # ── Combine and emit ─────────────────────────────────────────────────────
    parts = [df for df in [attacks_conqueror, attacks_supplier] if not df.is_empty()]
    if not parts:
        return moves

    attacks = pl.concat(parts, how="diagonal")
    print("Currently using testing _04_score_and_decide")
    for row in attacks.iter_rows(named=True):
        print(f"From {row['id_src']}, To {row['id']} at step {row['step']} "
              f"with {row['ships_sent']} ships (target has min {row['ships_min']})")

    moves += [list(r) for r in attacks.select(["id_src", "final_angle", "ships_sent"]).rows()]
    return moves
```

- [ ] **Step 5.2: Smoke test `_04`**

```python
moves = StrategyPipeline._04_score_and_decide(safe_lf, player_id=0)
assert isinstance(moves, list)
for m in moves:
    assert len(m) == 3, f"Move must be [id_src, angle, ships]: {m}"
    assert isinstance(m[2], (int, float)) and m[2] > 0
print("_04 OK: moves =", moves)
```

Expected: at least one move attacking the enemy planet.

- [ ] **Step 5.3: Commit**

```bash
git add 87-Polars_Simulate10Next_Conqueror_Supplier_fixed.py
git commit -m "feat: implement _04_score_and_decide — single collect, Polars eager scoring"
```

---

### Task 6: End-to-end smoke test via `agent()`

**Files:**
- Read: `87-Polars_Simulate10Next_Conqueror_Supplier_fixed.py`

- [ ] **Step 6.1: Run full agent loop**

```python
# Reset global state (required between runs)
import importlib, sys
if '87-Polars_Simulate10Next_Conqueror_Supplier_fixed' in sys.modules:
    del sys.modules['87-Polars_Simulate10Next_Conqueror_Supplier_fixed']

exec(open('87-Polars_Simulate10Next_Conqueror_Supplier_fixed.py').read())

class Obs:
    def __init__(self, planets, angular_velocity=0.0):
        self.planets = [list(p) for p in planets]
        self.initial_planets = [list(p) for p in planets]
        self.fleets = []
        self.next_fleet_id = 100
        self.comets = []
        self.comet_planet_ids = []
        self.angular_velocity = angular_velocity
        self.player = 0

obs = Obs([
    [0, 0, 5.0, 5.0, 1 + math.log(3), 30, 3],
    [1, 0, 5.0, 15.0, 1 + math.log(3), 30, 3],
    [2, 1, 80.0, 20.0, 1 + math.log(3), 50, 2],
], angular_velocity=0.05)

# Reset module-level globals
step = 0; num_agents = None; player_id = None

moves = agent(obs)
print("agent() OK:", moves)
assert isinstance(moves, list)
```

Expected: list of moves (possibly empty if ships insufficient, but no crash).

- [ ] **Step 6.2: Verify single collect guarantee**

Add a temporary patch to count collects:

```python
original_collect = pl.LazyFrame.collect
collect_count = [0]

def counting_collect(self, *args, **kwargs):
    collect_count[0] += 1
    return original_collect(self, *args, **kwargs)

pl.LazyFrame.collect = counting_collect

step = 0; num_agents = None; player_id = None
obs2 = Obs([
    [0, 0, 5.0, 5.0, 1 + math.log(3), 30, 3],
    [1, 1, 80.0, 20.0, 1 + math.log(3), 50, 2],
], angular_velocity=0.05)
agent(obs2)

# _01 uses one collect for planet_disp; _04 uses one collect for attacks_with_angle
assert collect_count[0] <= 2, f"Expected ≤2 collects per agent call, got {collect_count[0]}"
print(f"Collect count: {collect_count[0]} (expected ≤2)")

pl.LazyFrame.collect = original_collect  # restore
```

Expected: `Collect count: 2 (expected ≤2)`

- [ ] **Step 6.3: Commit**

```bash
git add 87-Polars_Simulate10Next_Conqueror_Supplier_fixed.py
git commit -m "feat: complete 87-Polars pipeline — verified end-to-end and collect count"
```

---

### Task 7: Deploy to lab

**Files:**
- Read: `87-Polars_Simulate10Next_Conqueror_Supplier_fixed.py`

- [ ] **Step 7.1: Deploy via `/lab`**

```
/lab 87-Polars_Simulate10Next_Conqueror_Supplier_fixed.py
```

Expected: `Created orbit-wars-lab/agents/mine/87-Polars_Simulate10Next_Conqueror_Supplier_fixed/main.py`

- [ ] **Step 7.2: Final commit**

```bash
git add docs/superpowers/plans/2026-06-01-polars-pipeline.md
git commit -m "docs: add Polars pipeline implementation plan"
```

---

## Self-Review

**Spec coverage:**
- `_01` returns `(pl.DataFrame, pl.DataFrame)` ✓ Task 2
- `_02` returns `pl.LazyFrame` ✓ Task 3
- `_03` returns `pl.LazyFrame` ✓ Task 4
- `_04` single collect, Conqueror/Supplier scoring ✓ Task 5
- `agent()` unchanged logic ✓ Task 1
- `GameConfig/PhysicsEngine/interpreter` unchanged ✓ Task 1
- `maintain_order=True` matches pandas `sort=False` ✓ used throughout Task 3/5
- `how="diagonal"` for concat ✓ Task 5

**Placeholder scan:** No TBDs, all code blocks complete. ✓

**Type consistency:**
- `pa_lf: pl.LazyFrame` — created in `_02`, passed to `_03`, passed to `_04` ✓
- `safe_lf: pl.LazyFrame` — created in `_03`, passed to `_04` ✓
- `attacks_with_angle: pl.DataFrame` — result of `.collect()` in `_04` ✓
- `conqueror_needs: pl.DataFrame | None` — used in supplier filter ✓
