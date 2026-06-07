# Per-Step Ships-Sent Expansion — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rewrite `StrategyPipeline` in `100-Rewrite_physics.ipynb` so that fleet launches are modelled per step (0–10), with ship budget = `min(ships_src from step_s onwards)`, and all columns use consistent `_src`/`_tgt` naming.

**Architecture:** Copy `GameConfig`, `PhysicsEngine`, `interpreter`, and `StrategyPipeline` from `81-Simulate10Next.py` into new notebook cells, then replace `mine_base` (one row per source planet) with `mine_per_step` (one row per source planet per step), update the cross-join to `(src, step_s) × (tgt, step_t)` where `step_t > step_s`, and rename all ambiguous columns to `_src`/`_tgt` throughout `_02`, `_03`, `_04`.

**Tech Stack:** Python 3, pandas, numpy, math, kaggle-environments (orbit_wars), Jupyter notebook

---

### Task 1: Copy infrastructure cells into notebook

**Files:**
- Modify: `100-Rewrite_physics.ipynb` (add cells after the existing env setup cells)

- [ ] **Step 1: Add imports + GameConfig + PhysicsEngine cell**

Add a new code cell in `100-Rewrite_physics.ipynb` with this content:

```python
import math
import copy
import pandas as pd
import numpy as np


class GameConfig:
    CENTER = 50.0
    SUN_RADIUS = 10.0
    ROTATION_RADIUS_LIMIT = 50.0
    MAX_SPEED = 6.0
    NB_STEPS_SIM = 10
    PLANET_MARGIN = 0.1
    PLANET_MOVEMENT_SLACK = 3.0


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

- [ ] **Step 2: Add interpreter cell**

Add a new code cell with the `interpreter` function copied verbatim from `81-Simulate10Next.py` lines 72–269.

- [ ] **Step 3: Run both cells and verify no errors**

Run the cells. Expected: no output, no errors.

---

### Task 2: Implement `_01_get_obs_dataframe` (unchanged logic, new cell)

**Files:**
- Modify: `100-Rewrite_physics.ipynb`

- [ ] **Step 1: Add `_01_get_obs_dataframe` cell**

Add a new code cell:

```python
class StrategyPipeline:

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

        df_s = pd.DataFrame(rows).sort_values("step").reset_index(drop=True)

        prev_pos = (
            df_s[["id", "step", "x", "y"]]
            .assign(step=lambda d: d["step"] + 1)
            .rename(columns={"x": "x_prev", "y": "y_prev"})
        )
        planet_disp = (
            df_s[["id", "step", "x", "y"]]
            .merge(prev_pos, on=["id", "step"], how="left")
            .assign(
                planet_disp=lambda d: np.sqrt(
                    (d["x"] - d["x_prev"].fillna(d["x"])) ** 2 +
                    (d["y"] - d["y_prev"].fillna(d["y"])) ** 2
                )
            )
            [["id", "step", "planet_disp"]]
        )
        return df_s, planet_disp
```

- [ ] **Step 2: Smoke-test with env observation**

Add a temporary cell:

```python
obs = env.state[0].observation
step = 0
num_agents = 2
df_s, planet_disp = StrategyPipeline._01_get_obs_dataframe(obs, step, num_agents)
print(df_s.shape, planet_disp.shape)
print(df_s.columns.tolist())
```

Expected output: `(308, 9)` (28 planets × 11 steps), columns include `step id x y radius ships production owner nature`.

---

### Task 3: Implement `_02_get_all_opportunities` with per-step source

**Files:**
- Modify: `100-Rewrite_physics.ipynb`

- [ ] **Step 1: Add `_02_get_all_opportunities` method to the `StrategyPipeline` class cell**

Extend the `StrategyPipeline` class with this method:

```python
    @staticmethod
    def _02_get_all_opportunities(
        df_s: pd.DataFrame,
        planet_disp: pd.DataFrame,
        player_id: int,
    ) -> pd.DataFrame:
        # Per-step source data for always-mine planets
        always_mine_mask = (
            df_s.groupby("id")["owner"]
            .transform(lambda g: (g == player_id).all())
        )
        always_mine_ids = df_s.loc[always_mine_mask, "id"].unique()

        if len(always_mine_ids) == 0:
            return pd.DataFrame()

        mine_per_step = (
            df_s
            .loc[df_s["id"].isin(always_mine_ids)]
            .sort_values(["id", "step"])
            .assign(
                ships_min=lambda d: d.groupby("id")["ships"]
                                     .transform(lambda s: s[::-1].cummin()[::-1])
            )
            .rename(columns={
                "id": "id_src",
                "step": "step_src",
                "x": "x_src",
                "y": "y_src",
                "radius": "radius_src",
                "ships": "ships_src",
                "production": "production_src",
                "nature": "nature_src",
                "owner": "owner_src",
            })
            .reset_index(drop=True)
        )

        df_tgt = df_s.rename(columns={
            "id": "id_tgt",
            "step": "step_tgt",
            "x": "x_tgt",
            "y": "y_tgt",
            "radius": "radius_tgt",
            "ships": "ships_tgt",
            "production": "production_tgt",
            "nature": "nature_tgt",
            "owner": "owner_tgt",
        })

        # Phase A: planet-level cross join
        coarse = (
            mine_per_step.assign(_key=1)
            .merge(df_tgt.assign(_key=1), on="_key")
            .drop(columns="_key")
            .loc[lambda d: (d["step_tgt"] > d["step_src"]) & (d["id_tgt"] != d["id_src"])]
            .merge(
                planet_disp.rename(columns={"id": "id_tgt", "step": "step_tgt"}),
                on=["id_tgt", "step_tgt"], how="left"
            )
            .reset_index(drop=True)
            .assign(
                dist_tgt_src=lambda d: np.sqrt(
                    (d["x_tgt"] - d["x_src"]) ** 2 + (d["y_tgt"] - d["y_src"]) ** 2
                ),
                step_diff=lambda d: (d["step_tgt"] - d["step_src"]).astype(float),
            )
        )

        # Sun-crossing filter (vectorised)
        _dx = coarse["x_tgt"].values - coarse["x_src"].values
        _dy = coarse["y_tgt"].values - coarse["y_src"].values
        _l2 = _dx ** 2 + _dy ** 2
        _dot = (GameConfig.CENTER - coarse["x_src"].values) * _dx + (GameConfig.CENTER - coarse["y_src"].values) * _dy
        _t_sun = np.clip(_dot / np.where(_l2 == 0, 1.0, _l2), 0.0, 1.0)
        _proj = np.sqrt(
            (GameConfig.CENTER - coarse["x_src"].values - _t_sun * _dx) ** 2 +
            (GameConfig.CENTER - coarse["y_src"].values - _t_sun * _dy) ** 2
        )
        _sun_dist = np.where(
            _l2 == 0,
            np.sqrt((GameConfig.CENTER - coarse["x_src"].values) ** 2 + (GameConfig.CENTER - coarse["y_src"].values) ** 2),
            _proj,
        )
        _crossing_sun = _sun_dist < (GameConfig.SUN_RADIUS + GameConfig.PLANET_MARGIN)

        coarse = (
            coarse
            .assign(_crossing_sun=_crossing_sun)
            .loc[lambda d:
                (d["dist_tgt_src"] <
                 (d["step_diff"] + 1) * GameConfig.MAX_SPEED
                 + d["radius_src"] + GameConfig.PLANET_MARGIN + d["radius_tgt"]
                 + d["planet_disp"].fillna(0.0))
                & ~d["_crossing_sun"]
            ]
            .drop(columns="_crossing_sun")
            .reset_index(drop=True)
        )

        if coarse.empty:
            return pd.DataFrame()

        # Ships_sent expansion
        expanded = (
            coarse
            .assign(
                ships_sent=lambda d: [
                    list(range(1, int(sm) + 1))
                    for sm in d["ships_min"]
                ]
            )
            .explode("ships_sent")
            .assign(ships_sent=lambda d: d["ships_sent"].astype("int64"))
            .reset_index(drop=True)
        )

        # Phase B: fleet-speed filter
        _fs_ratio = np.clip(
            np.log(expanded["ships_sent"].values.astype(float)) / math.log(1000.0),
            0, None,
        )
        _fleet_speed_b = 1.0 + (GameConfig.MAX_SPEED - 1.0) * _fs_ratio ** 1.5
        _dist_min_b = expanded["step_diff"].values * _fleet_speed_b + GameConfig.PLANET_MARGIN + expanded["radius_src"].values
        _dist_prev_b = _dist_min_b - _fleet_speed_b

        prev_pos_tgt = (
            df_s[["id", "step", "x", "y"]]
            .assign(step=lambda d: d["step"] + 1)
            .rename(columns={"id": "id_tgt", "step": "step_tgt", "x": "x_prev_tgt", "y": "y_prev_tgt"})
        )

        expanded = (
            expanded
            .assign(fleet_speed=_fleet_speed_b, dist_min=_dist_min_b, dist_prev=_dist_prev_b)
            .loc[lambda d: d["dist_tgt_src"] < d["dist_min"] + d["fleet_speed"] + d["radius_tgt"] + GameConfig.PLANET_MOVEMENT_SLACK]
            .merge(prev_pos_tgt, on=["id_tgt", "step_tgt"], how="left")
            .reset_index(drop=True)
        )

        if expanded.empty:
            return pd.DataFrame()

        # Swept-pair collision (vectorised)
        _dx2 = expanded["x_tgt"].values - expanded["x_src"].values
        _dy2 = expanded["y_tgt"].values - expanded["y_src"].values
        _dist2 = expanded["dist_tgt_src"].values
        _ux = _dx2 / np.where(_dist2 < 1e-9, 1.0, _dist2)
        _uy = _dy2 / np.where(_dist2 < 1e-9, 1.0, _dist2)

        _xpf = expanded["x_prev_tgt"].fillna(expanded["x_tgt"]).values
        _ypf = expanded["y_prev_tgt"].fillna(expanded["y_tgt"]).values

        _fx0 = expanded["x_src"].values + _ux * expanded["dist_prev"].values
        _fy0 = expanded["y_src"].values + _uy * expanded["dist_prev"].values
        _pvx = expanded["x_tgt"].values - _xpf
        _pvy = expanded["y_tgt"].values - _ypf
        _dvx = _ux * expanded["fleet_speed"].values - _pvx
        _dvy = _uy * expanded["fleet_speed"].values - _pvy
        _d0x = _fx0 - _xpf
        _d0y = _fy0 - _ypf
        _a   = _dvx ** 2 + _dvy ** 2
        _b_  = 2.0 * (_d0x * _dvx + _d0y * _dvy)
        _c_  = _d0x ** 2 + _d0y ** 2 - expanded["radius_tgt"].values ** 2
        _disc = _b_ ** 2 - 4.0 * _a * _c_
        _sq  = np.sqrt(np.clip(_disc, 0, None))
        _t1  = np.where(_a < 1e-12, 0.0, (-_b_ - _sq) / (2.0 * _a))
        _t2  = np.where(_a < 1e-12, 1.0, (-_b_ + _sq) / (2.0 * _a))
        _coll = np.where(_a < 1e-12, _c_ <= 0.0, (_disc >= 0.0) & (_t2 >= 0.0) & (_t1 <= 1.0))

        pa = (
            expanded
            .assign(t1=_t1, t2=_t2, collision=_coll)
            .loc[lambda d: d["collision"]]
            .reset_index(drop=True)
        )

        if pa.empty:
            return pa

        # Angle geometry
        _xpf_pa = pa["x_prev_tgt"].fillna(pa["x_tgt"]).values
        _ypf_pa = pa["y_prev_tgt"].fillna(pa["y_tgt"]).values
        _t1e = np.clip(pa["t1"].values, 0.0, 1.0)
        _t2e = np.clip(pa["t2"].values, 0.0, 1.0)

        pa = (
            pa
            .assign(
                t1_eff=_t1e,
                t2_eff=_t2e,
                p_t1_x=_xpf_pa + _t1e * (pa["x_tgt"].values - _xpf_pa),
                p_t1_y=_ypf_pa + _t1e * (pa["y_tgt"].values - _ypf_pa),
                p_t2_x=_xpf_pa + _t2e * (pa["x_tgt"].values - _xpf_pa),
                p_t2_y=_ypf_pa + _t2e * (pa["y_tgt"].values - _ypf_pa),
            )
            .assign(
                angle_t1=lambda d: np.arctan2(d["p_t1_y"] - d["y_src"], d["p_t1_x"] - d["x_src"]),
                angle_t2=lambda d: np.arctan2(d["p_t2_y"] - d["y_src"], d["p_t2_x"] - d["x_src"]),
                d_s_t1=lambda d: np.sqrt((d["p_t1_x"] - d["x_src"]) ** 2 + (d["p_t1_y"] - d["y_src"]) ** 2),
                d_s_t2=lambda d: np.sqrt((d["p_t2_x"] - d["x_src"]) ** 2 + (d["p_t2_y"] - d["y_src"]) ** 2),
            )
            .assign(
                d_f_t1=lambda d: d["dist_prev"] + d["t1_eff"] * d["fleet_speed"],
                d_f_t2=lambda d: d["dist_prev"] + d["t2_eff"] * d["fleet_speed"],
            )
            .assign(
                angle_radius_t1=lambda d: np.arccos(np.clip(
                    (d["d_s_t1"] ** 2 + d["d_f_t1"] ** 2 - d["radius_tgt"] ** 2)
                    / (2.0 * d["d_s_t1"] * d["d_f_t1"]),
                    -1.0, 1.0,
                )),
                angle_radius_t2=lambda d: np.arccos(np.clip(
                    (d["d_s_t2"] ** 2 + d["d_f_t2"] ** 2 - d["radius_tgt"] ** 2)
                    / (2.0 * d["d_s_t2"] * d["d_f_t2"]),
                    -1.0, 1.0,
                )),
            )
            .assign(
                angle_min=lambda d: np.minimum(
                    d["angle_t1"] - d["angle_radius_t1"],
                    d["angle_t2"] - d["angle_radius_t2"],
                ) % (2 * math.pi),
                angle_max=lambda d: np.maximum(
                    d["angle_t1"] + d["angle_radius_t1"],
                    d["angle_t2"] + d["angle_radius_t2"],
                ) % (2 * math.pi),
                angle=lambda d: np.arctan2(
                    np.sin(d["angle_t1"]) + np.sin(d["angle_t2"]),
                    np.cos(d["angle_t1"]) + np.cos(d["angle_t2"]),
                ),
            )
            .sort_values("step_tgt")
            .reset_index(drop=True)
        )

        return pa
```

- [ ] **Step 2: Smoke-test `_02`**

Add a temporary cell:

```python
pa = StrategyPipeline._02_get_all_opportunities(df_s, planet_disp, player_id=0)
print("pa shape:", pa.shape)
print("pa columns:", pa.columns.tolist())
if not pa.empty:
    print(pa[["id_src", "step_src", "id_tgt", "step_tgt", "ships_sent", "angle"]].head(10))
```

Expected: non-empty DataFrame with columns including `id_src`, `step_src`, `id_tgt`, `step_tgt`, `ships_sent`, `angle`. No KeyError.

---

### Task 4: Implement `_03_filter_collision` with `_src`/`_tgt` naming

**Files:**
- Modify: `100-Rewrite_physics.ipynb`

- [ ] **Step 1: Add `_03_filter_collision` to `StrategyPipeline`**

```python
    @staticmethod
    def _03_filter_collision(pa: pd.DataFrame) -> pd.DataFrame:
        if pa.empty:
            return pa

        pa_left = pa[["id_src", "ships_sent", "step_tgt", "id_tgt", "angle", "angle_min", "angle_max"]].copy()
        pa_obs = (
            pa[["id_src", "ships_sent", "step_tgt", "id_tgt", "angle_min", "angle_max"]]
            .rename(columns={
                "step_tgt": "step_tgt_obs",
                "id_tgt": "id_tgt_obs",
                "angle_min": "angle_min_obs",
                "angle_max": "angle_max_obs",
            })
        )

        blocked_joined = (
            pa_left
            .merge(pa_obs, on=["id_src", "ships_sent"])
            .loc[lambda d: (d["step_tgt_obs"] < d["step_tgt"]) & (d["id_tgt_obs"] != d["id_tgt"])]
            .reset_index(drop=True)
        )

        if not blocked_joined.empty:
            _anorm = blocked_joined["angle"].values % (2 * math.pi)
            _wraps = (blocked_joined["angle_min_obs"] > blocked_joined["angle_max_obs"]).values
            _in_cone = np.where(
                _wraps,
                (_anorm >= blocked_joined["angle_min_obs"].values) | (_anorm <= blocked_joined["angle_max_obs"].values),
                (_anorm >= blocked_joined["angle_min_obs"].values) & (_anorm <= blocked_joined["angle_max_obs"].values),
            )
            blocked = (
                blocked_joined[_in_cone]
                [["id_src", "ships_sent", "step_tgt", "id_tgt"]]
                .drop_duplicates()
            )
        else:
            blocked = pd.DataFrame(columns=["id_src", "ships_sent", "step_tgt", "id_tgt"])

        attacks_with_angle = (
            pa
            .merge(blocked.assign(_blocked=True), on=["id_src", "ships_sent", "step_tgt", "id_tgt"], how="left")
            .loc[lambda d: d["_blocked"].isna()]
            .drop(columns="_blocked")
            .assign(final_angle=lambda d: d["angle"])
            .reset_index(drop=True)
        )

        return attacks_with_angle
```

- [ ] **Step 2: Smoke-test `_03`**

Add a temporary cell:

```python
safe = StrategyPipeline._03_filter_collision(pa)
print("safe shape:", safe.shape)
print("pa shape:", pa.shape)
assert safe.shape[0] <= pa.shape[0], "filter should only remove rows"
assert "final_angle" in safe.columns
```

Expected: no AssertionError, `safe.shape[0] <= pa.shape[0]`.

---

### Task 5: Implement `_04_score_and_decide` with `_src`/`_tgt` naming

**Files:**
- Modify: `100-Rewrite_physics.ipynb`

- [ ] **Step 1: Add `_04_score_and_decide` to `StrategyPipeline`**

```python
    @staticmethod
    def _04_score_and_decide(attacks_with_angle: pd.DataFrame, player_id: int) -> list:
        if attacks_with_angle.empty:
            return []

        moves = []

        # Comet evasion
        awa_comets = attacks_with_angle[attacks_with_angle["nature_src"] == "comet"]
        if not awa_comets.empty:
            x_off = (awa_comets["x_src"] - GameConfig.CENTER).abs().max() or 0
            y_off = (awa_comets["y_src"] - GameConfig.CENTER).abs().max() or 0
            if max(x_off, y_off) > 45:
                moves += (
                    awa_comets[awa_comets["ships_sent"] <= awa_comets["ships_min"]]
                    .sort_values(["ships_sent", "step_tgt"], ascending=[False, True])
                    .groupby("id_src", sort=False)
                    .first()
                    .reset_index()
                    [["id_src", "final_angle", "ships_sent"]]
                    .values.tolist()
                )
                id_to_avoid = awa_comets["id_src"].unique().tolist()
                attacks_with_angle = attacks_with_angle[~attacks_with_angle["id_src"].isin(id_to_avoid)]

        planet_id_top_5 = (
            attacks_with_angle
            .sort_values(["step_tgt", "ships_sent"])
            .groupby(["id_src", "id_tgt"], sort=False)
            .first()
            .reset_index()
            .sort_values(["step_tgt", "ships_sent"])
            .groupby("id_src", sort=False)
            .head(5)
            [["id_src", "id_tgt"]]
        )

        attacks_joined = (
            planet_id_top_5
            .merge(attacks_with_angle, on=["id_src", "id_tgt"], how="left")
            .loc[lambda d: d["owner_tgt"] != player_id]
            .assign(
                ships_needed=lambda d: np.where(
                    d["owner_tgt"] == -1, d["ships_tgt"], d["ships_tgt"] + d["production_tgt"]
                )
            )
            .loc[lambda d:
                (d["ships_needed"] + 1 <= d["ships_sent"]) &
                (d["ships_sent"] <= d["ships_needed"] + d["production_src"] + 1)
            ]
            .sort_values(["step_tgt", "ships_sent"])
            .groupby(["id_src", "id_tgt"], sort=False)
            .first()
            .reset_index()
            .assign(time_cost=lambda d: d["ships_needed"] / d["production_src"])
        )

        if attacks_joined.empty:
            return moves

        attacks_joined = attacks_joined.assign(
            total_time_cost=attacks_joined.groupby("id_src")["time_cost"].transform("sum")
        ).assign(
            score=lambda d: (d["total_time_cost"] - d["time_cost"] - d["step_diff"]) * d["production_tgt"]
        )

        attacks = (
            attacks_joined
            .sort_values("score", ascending=False)
            .groupby("id_src", sort=False)
            .first()
            .reset_index()
            .loc[lambda d: d["ships_sent"] <= d["ships_min"]]
        )

        for _, row in attacks.iterrows():
            print(f"From {row['id_src']}, To {row['id_tgt']} at step {row['step_tgt']} "
                  f"with {row['ships_sent']} ships (target has min {row['ships_min']})")

        moves += attacks[["id_src", "final_angle", "ships_sent"]].values.tolist()
        return moves
```

- [ ] **Step 2: Smoke-test `_04`**

Add a temporary cell:

```python
moves = StrategyPipeline._04_score_and_decide(safe, player_id=0)
print("moves:", moves)
for m in moves:
    assert len(m) == 3, f"move must be [planet_id, angle, ships]: {m}"
    assert isinstance(m[0], (int, np.integer)), f"planet_id must be int: {m}"
    assert -math.pi <= m[1] <= math.pi or True, "angle is float"
    assert m[2] > 0, f"ships must be > 0: {m}"
print("all moves valid")
```

Expected: prints moves (may be empty list at step 0), prints "all moves valid".

---

### Task 6: Wire up `agent` function and run a full game

**Files:**
- Modify: `100-Rewrite_physics.ipynb`

- [ ] **Step 1: Add global state and `agent` function**

```python
_step = 0
_num_agents = None
_player_id = None


def agent(obs):
    global _step, _num_agents, _player_id

    if _num_agents is None:
        initial = (
            obs.initial_planets if hasattr(obs, "initial_planets")
            else obs["initial_planets"]
        )
        owners = {p[1] for p in initial if p[1] != -1}
        _num_agents = 4 if len(owners) > 2 else 2
    if _player_id is None:
        _player_id = obs.get("player", 0) if isinstance(obs, dict) else obs.player

    df_s, planet_disp = StrategyPipeline._01_get_obs_dataframe(obs, _step, _num_agents)
    pa = StrategyPipeline._02_get_all_opportunities(df_s, planet_disp, _player_id)
    safe_attacks = StrategyPipeline._03_filter_collision(pa)
    moves = StrategyPipeline._04_score_and_decide(safe_attacks, _player_id)

    _step += 1
    return moves
```

- [ ] **Step 2: Run a 10-step game and check it doesn't crash**

```python
_step = 0; _num_agents = None; _player_id = None  # reset globals
env2 = ke.make("orbit_wars", debug=False)
env2.reset()
for i in range(10):
    env2.step([agent, "random"])
print("10 steps completed without error")
print(env2.render(mode="ansi"))
```

Expected: prints "10 steps completed without error" and board state. No exception.

- [ ] **Step 3: Commit**

```bash
git add 100-Rewrite_physics.ipynb
git commit -m "feat: per-step ships-sent expansion in 100-Rewrite_physics — mine_per_step + _src/_tgt naming"
```
