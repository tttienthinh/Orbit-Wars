# Phase A Refactor: Reachability Matrix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor `_02_get_all_opportunities` into `_02_pre_mine`/`_02_pre_all` + shared core, then wire up a reachability matrix (any→any, fixed ships [4,16,64,256]) computed in parallel and passed to `_04_score_and_decide`.

**Architecture:** Extract Phase A (source building + cross-join + step/dist filter + sun-crossing + ships_sent assignment) into two pre functions that return a `coarse` DataFrame with a `ships_sent` list column. The shared `_sun_crossing_filter` helper is called by both. `_02_get_all_opportunities(coarse, df_s, planet_disp)` now starts from the planet_disp merge + distance filter + `.explode("ships_sent")`. `_03_filter_collision` is unchanged. `_04_score_and_decide` gains a `reach_matrix` second arg (accepted but not yet used in scoring).

**Tech Stack:** Python 3, pandas, numpy, pytest

---

### Task 1: Test infrastructure

**Files:**
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`

- [ ] **Step 1: Create test directory and conftest**

Create `tests/__init__.py` (empty) and `tests/conftest.py`:

```python
import importlib.util, os, types
import pandas as pd
import numpy as np
import pytest

_FILE = os.path.join(os.path.dirname(__file__), "..", "102-Simulate10Next.py")

def _load():
    spec = importlib.util.spec_from_file_location("m102", os.path.abspath(_FILE))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_mod = _load()
StrategyPipeline = _mod.StrategyPipeline
GameConfig = _mod.GameConfig


@pytest.fixture
def simple_obs():
    """Two fixed planets far from sun: mine (id=0) at (1,10), enemy (id=1) at (99,10).
    Direct path stays 40 units from sun — no crossing.
    r ≈ 63 for both, so neither orbits (r+radius > 50)."""
    planets = [
        [0, 0,  1.0, 10.0, 3.0, 10, 1],
        [1, 1, 99.0, 10.0, 3.0,  5, 1],
    ]
    obs = types.SimpleNamespace(
        planets=[list(p) for p in planets],
        initial_planets=[list(p) for p in planets],
        fleets=[],
        next_fleet_id=0,
        comets=[],
        comet_planet_ids=[],
        angular_velocity=0.0,
        player=0,
    )
    return obs
```

- [ ] **Step 2: Verify conftest loads without errors**

```
pytest tests/ --collect-only
```

Expected: `no tests ran` with no import errors.

- [ ] **Step 3: Commit**

```bash
git add tests/__init__.py tests/conftest.py
git commit -m "test: add test infrastructure and obs fixture for pipeline refactor"
```

---

### Task 2: Extract `_sun_crossing_filter`

**Files:**
- Modify: `102-Simulate10Next.py`
- Create: `tests/test_sun_crossing_filter.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_sun_crossing_filter.py`:

```python
import pandas as pd
from tests.conftest import StrategyPipeline


def test_removes_path_crossing_sun():
    # (20,50) → (80,50) passes through sun center (50,50), dist=0
    coarse = pd.DataFrame({'x_src': [20.0], 'y_src': [50.0], 'x': [80.0], 'y': [50.0]})
    result = StrategyPipeline._sun_crossing_filter(coarse)
    assert len(result) == 0


def test_keeps_path_not_crossing_sun():
    # (1,10) → (99,10): closest point to (50,50) is (50,10), dist=40 > 10.1
    coarse = pd.DataFrame({'x_src': [1.0], 'y_src': [10.0], 'x': [99.0], 'y': [10.0]})
    result = StrategyPipeline._sun_crossing_filter(coarse)
    assert len(result) == 1


def test_filters_only_crossing_rows():
    coarse = pd.DataFrame({
        'x_src': [20.0,  1.0],
        'y_src': [50.0, 10.0],
        'x':     [80.0, 99.0],
        'y':     [50.0, 10.0],
    })
    result = StrategyPipeline._sun_crossing_filter(coarse)
    assert len(result) == 1
    assert float(result.iloc[0]['x_src']) == 1.0
```

- [ ] **Step 2: Run to confirm FAIL**

```
pytest tests/test_sun_crossing_filter.py -v
```

Expected: `AttributeError: type object 'StrategyPipeline' has no attribute '_sun_crossing_filter'`

- [ ] **Step 3: Add `_sun_crossing_filter` to `StrategyPipeline` in `102-Simulate10Next.py`**

Add this static method directly above `_02_get_all_opportunities` (around line 325):

```python
    @staticmethod
    def _sun_crossing_filter(coarse: pd.DataFrame) -> pd.DataFrame:
        _dx = coarse["x"].values - coarse["x_src"].values
        _dy = coarse["y"].values - coarse["y_src"].values
        _l2 = _dx ** 2 + _dy ** 2
        _dot = (
            (GameConfig.CENTER - coarse["x_src"].values) * _dx
            + (GameConfig.CENTER - coarse["y_src"].values) * _dy
        )
        _t_sun = np.clip(_dot / np.where(_l2 == 0, 1.0, _l2), 0.0, 1.0)
        _proj = np.sqrt(
            (GameConfig.CENTER - coarse["x_src"].values - _t_sun * _dx) ** 2
            + (GameConfig.CENTER - coarse["y_src"].values - _t_sun * _dy) ** 2
        )
        _sun_dist = np.where(
            _l2 == 0,
            np.sqrt(
                (GameConfig.CENTER - coarse["x_src"].values) ** 2
                + (GameConfig.CENTER - coarse["y_src"].values) ** 2
            ),
            _proj,
        )
        _crossing = _sun_dist < (GameConfig.SUN_RADIUS + GameConfig.PLANET_MARGIN)
        return coarse[~_crossing].reset_index(drop=True)
```

- [ ] **Step 4: Replace inline sun-crossing block inside `_02_get_all_opportunities`**

In `_02_get_all_opportunities`, find the block that starts with `# Sun-crossing filter (vectorised)` (around line 373) and ends with the combined `.loc` filter (around line 401). Replace it with:

```python
        coarse = StrategyPipeline._sun_crossing_filter(coarse)

        coarse = (
            coarse
            .loc[lambda d:
                (d["dist_tgt_src"] <
                 (d["step_diff"] + 1) * GameConfig.MAX_SPEED
                 + d["radius_src"] + GameConfig.PLANET_MARGIN + d["radius"]
                 + d["planet_disp"].fillna(0.0))
            ]
            .reset_index(drop=True)
        )
```

The old combined block to remove looks like this (from `_dx = coarse["x"]...` down to `.drop(columns="_crossing_sun")`):

```python
        # Sun-crossing filter (vectorised)
        _dx = coarse["x"].values - coarse["x_src"].values
        _dy = coarse["y"].values - coarse["y_src"].values
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
                 + d["radius_src"] + GameConfig.PLANET_MARGIN + d["radius"]
                 + d["planet_disp"].fillna(0.0))
                & ~d["_crossing_sun"]
            ]
            .drop(columns="_crossing_sun")
            .reset_index(drop=True)
        )
```

- [ ] **Step 5: Run all tests**

```
pytest tests/ -v
```

Expected: all 3 tests PASS.

- [ ] **Step 6: Commit**

```bash
git add 102-Simulate10Next.py tests/test_sun_crossing_filter.py
git commit -m "refactor: extract _sun_crossing_filter as shared static helper"
```

---

### Task 3: Create `_02_pre_mine`

**Files:**
- Modify: `102-Simulate10Next.py`
- Create: `tests/test_02_pre_mine.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_02_pre_mine.py`:

```python
import pytest
from tests.conftest import StrategyPipeline


def test_returns_only_mine_planet_as_source(simple_obs):
    df_s, _ = StrategyPipeline._01_get_obs_dataframe(simple_obs, step=0, num_agents=2)
    coarse = StrategyPipeline._02_pre_mine(df_s, player_id=0)
    assert not coarse.empty
    assert set(coarse["id_src"].unique()) == {0}


def test_ships_sent_is_list_starting_at_1(simple_obs):
    df_s, _ = StrategyPipeline._01_get_obs_dataframe(simple_obs, step=0, num_agents=2)
    coarse = StrategyPipeline._02_pre_mine(df_s, player_id=0)
    assert not coarse.empty
    first_list = coarse.iloc[0]["ships_sent"]
    assert isinstance(first_list, list)
    assert first_list[0] == 1


def test_enemy_player_id_returns_empty(simple_obs):
    df_s, _ = StrategyPipeline._01_get_obs_dataframe(simple_obs, step=0, num_agents=2)
    # player 1 owns planet 1; planet 1 is adjacent to planet 0 which is owned by player 0 the whole sim
    # planet 1 IS always owned by player 1 so it should appear
    coarse = StrategyPipeline._02_pre_mine(df_s, player_id=1)
    assert not coarse.empty
    assert set(coarse["id_src"].unique()) == {1}


def test_no_self_attacks(simple_obs):
    df_s, _ = StrategyPipeline._01_get_obs_dataframe(simple_obs, step=0, num_agents=2)
    coarse = StrategyPipeline._02_pre_mine(df_s, player_id=0)
    assert not coarse.empty
    assert (coarse["id_src"] != coarse["id"]).all()
```

- [ ] **Step 2: Run to confirm FAIL**

```
pytest tests/test_02_pre_mine.py -v
```

Expected: `AttributeError: type object 'StrategyPipeline' has no attribute '_02_pre_mine'`

- [ ] **Step 3: Add `_02_pre_mine` to `StrategyPipeline` in `102-Simulate10Next.py`**

Add this method directly above `_02_get_all_opportunities`:

```python
    @staticmethod
    def _02_pre_mine(df_s: pd.DataFrame, player_id: int) -> pd.DataFrame:
        nb_steps_sim = GameConfig.NB_STEPS_SIM
        mine_base = (
            df_s
            .assign(is_mine=(df_s["owner"] == player_id).astype(int))
            .groupby("id", sort=False)
            .agg(
                step_src=("step", "first"),
                x_src=("x", "first"),
                y_src=("y", "first"),
                radius_src=("radius", "first"),
                ships_min=("ships", "min"),
                production_src=("production", "first"),
                nature_src=("nature", "first"),
                owner_src=("owner", "first"),
                row_count=("id", "count"),
                is_mine=("is_mine", "sum"),
            )
            .reset_index()
            .loc[lambda d: (d["row_count"] == d["is_mine"]) & (d["owner_src"] == player_id)]
            .rename(columns={"id": "id_src"})
            .reset_index(drop=True)
        )

        if mine_base.empty:
            return pd.DataFrame()

        coarse = (
            mine_base.assign(_key=1)
            .merge(df_s.assign(_key=1), on="_key")
            .drop(columns="_key")
            .loc[lambda d: (d["step"] > d["step_src"]) & (d["id"] != d["id_src"])]
            .reset_index(drop=True)
            .assign(
                dist_tgt_src=lambda d: np.sqrt(
                    (d["x"] - d["x_src"]) ** 2 + (d["y"] - d["y_src"]) ** 2
                ),
                step_diff=lambda d: (d["step"] - d["step_src"]).astype(float),
            )
        )

        coarse = StrategyPipeline._sun_crossing_filter(coarse)

        if coarse.empty:
            return pd.DataFrame()

        coarse = coarse.assign(
            ships_sent=lambda d: [
                list(range(1, int(sm) + int(ps) * nb_steps_sim + 1))
                for sm, ps in zip(d["ships_min"], d["production_src"])
            ]
        )

        return coarse
```

- [ ] **Step 4: Run tests**

```
pytest tests/test_02_pre_mine.py -v
```

Expected: all 4 tests PASS.

- [ ] **Step 5: Run full suite to confirm no regressions**

```
pytest tests/ -v
```

Expected: all tests PASS.

- [ ] **Step 6: Commit**

```bash
git add 102-Simulate10Next.py tests/test_02_pre_mine.py
git commit -m "feat: add _02_pre_mine — Phase A for mine-only sources with formula ships_sent"
```

---

### Task 4: Create `_02_pre_all`

**Files:**
- Modify: `102-Simulate10Next.py`
- Create: `tests/test_02_pre_all.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_02_pre_all.py`:

```python
import pytest
from tests.conftest import StrategyPipeline


def test_includes_all_planets_as_sources(simple_obs):
    df_s, _ = StrategyPipeline._01_get_obs_dataframe(simple_obs, step=0, num_agents=2)
    coarse = StrategyPipeline._02_pre_all(df_s, ships_list=[4, 16, 64, 256])
    assert not coarse.empty
    sources = set(coarse["id_src"].unique())
    assert 0 in sources
    assert 1 in sources


def test_ships_sent_is_the_fixed_list(simple_obs):
    df_s, _ = StrategyPipeline._01_get_obs_dataframe(simple_obs, step=0, num_agents=2)
    coarse = StrategyPipeline._02_pre_all(df_s, ships_list=[4, 16, 64, 256])
    assert not coarse.empty
    for row_list in coarse["ships_sent"]:
        assert row_list == [4, 16, 64, 256]


def test_no_self_attacks(simple_obs):
    df_s, _ = StrategyPipeline._01_get_obs_dataframe(simple_obs, step=0, num_agents=2)
    coarse = StrategyPipeline._02_pre_all(df_s, ships_list=[4, 16, 64, 256])
    assert not coarse.empty
    assert (coarse["id_src"] != coarse["id"]).all()
```

- [ ] **Step 2: Run to confirm FAIL**

```
pytest tests/test_02_pre_all.py -v
```

Expected: `AttributeError: type object 'StrategyPipeline' has no attribute '_02_pre_all'`

- [ ] **Step 3: Add `_02_pre_all` to `StrategyPipeline` in `102-Simulate10Next.py`**

Add directly below `_02_pre_mine`:

```python
    @staticmethod
    def _02_pre_all(df_s: pd.DataFrame, ships_list: list) -> pd.DataFrame:
        all_base = (
            df_s
            .groupby("id", sort=False)
            .agg(
                step_src=("step", "first"),
                x_src=("x", "first"),
                y_src=("y", "first"),
                radius_src=("radius", "first"),
                ships_min=("ships", "min"),
                production_src=("production", "first"),
                nature_src=("nature", "first"),
                owner_src=("owner", "first"),
            )
            .reset_index()
            .rename(columns={"id": "id_src"})
            .reset_index(drop=True)
        )

        if all_base.empty:
            return pd.DataFrame()

        coarse = (
            all_base.assign(_key=1)
            .merge(df_s.assign(_key=1), on="_key")
            .drop(columns="_key")
            .loc[lambda d: (d["step"] > d["step_src"]) & (d["id"] != d["id_src"])]
            .reset_index(drop=True)
            .assign(
                dist_tgt_src=lambda d: np.sqrt(
                    (d["x"] - d["x_src"]) ** 2 + (d["y"] - d["y_src"]) ** 2
                ),
                step_diff=lambda d: (d["step"] - d["step_src"]).astype(float),
            )
        )

        coarse = StrategyPipeline._sun_crossing_filter(coarse)

        if coarse.empty:
            return pd.DataFrame()

        coarse = coarse.assign(ships_sent=[ships_list] * len(coarse))

        return coarse
```

- [ ] **Step 4: Run tests**

```
pytest tests/test_02_pre_all.py -v
```

Expected: all 3 tests PASS.

- [ ] **Step 5: Run full suite**

```
pytest tests/ -v
```

Expected: all tests PASS.

- [ ] **Step 6: Commit**

```bash
git add 102-Simulate10Next.py tests/test_02_pre_all.py
git commit -m "feat: add _02_pre_all — Phase A for all-planet sources with fixed ships_sent"
```

---

### Task 5: Refactor `_02_get_all_opportunities` to new signature + update `agent()`

**Files:**
- Modify: `102-Simulate10Next.py`
- Create: `tests/test_02_get_all_opportunities.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_02_get_all_opportunities.py`:

```python
import pytest
import pandas as pd
from tests.conftest import StrategyPipeline


def test_new_api_returns_dataframe_for_mine_path(simple_obs):
    df_s, planet_disp = StrategyPipeline._01_get_obs_dataframe(simple_obs, step=0, num_agents=2)
    coarse = StrategyPipeline._02_pre_mine(df_s, player_id=0)
    result = StrategyPipeline._02_get_all_opportunities(coarse, df_s, planet_disp)
    assert isinstance(result, pd.DataFrame)
    if not result.empty:
        assert "angle" in result.columns
        assert "ships_sent" in result.columns
        assert "id_src" in result.columns


def test_reach_path_ships_only_from_fixed_list(simple_obs):
    df_s, planet_disp = StrategyPipeline._01_get_obs_dataframe(simple_obs, step=0, num_agents=2)
    coarse = StrategyPipeline._02_pre_all(df_s, ships_list=[4, 16, 64, 256])
    result = StrategyPipeline._02_get_all_opportunities(coarse, df_s, planet_disp)
    assert isinstance(result, pd.DataFrame)
    if not result.empty:
        assert set(result["ships_sent"].unique()).issubset({4, 16, 64, 256})


def test_empty_coarse_returns_empty_dataframe(simple_obs):
    df_s, planet_disp = StrategyPipeline._01_get_obs_dataframe(simple_obs, step=0, num_agents=2)
    result = StrategyPipeline._02_get_all_opportunities(pd.DataFrame(), df_s, planet_disp)
    assert isinstance(result, pd.DataFrame)
    assert result.empty
```

These tests will currently PASS (old signature still works via positional arg if we pass coarse as df_s). We'll verify they still pass after the refactor.

- [ ] **Step 2: Run to record baseline**

```
pytest tests/test_02_get_all_opportunities.py -v
```

Note: `test_new_api_returns_dataframe_for_mine_path` and `test_reach_path_ships_only_from_fixed_list` will likely FAIL because the current `_02_get_all_opportunities` takes `(df_s, planet_disp, player_id)`, not `(coarse, df_s, planet_disp)`. That is expected — these are the tests we need to make pass.

- [ ] **Step 3: Replace the body of `_02_get_all_opportunities` in `102-Simulate10Next.py`**

Change the method signature from:
```python
    @staticmethod
    def _02_get_all_opportunities(
        df_s: pd.DataFrame,
        planet_disp: pd.DataFrame,
        player_id: int,
    ) -> pd.DataFrame:
```

To:
```python
    @staticmethod
    def _02_get_all_opportunities(
        coarse: pd.DataFrame,
        df_s: pd.DataFrame,
        planet_disp: pd.DataFrame,
    ) -> pd.DataFrame:
```

Replace the entire method body with:

```python
        if coarse.empty:
            return pd.DataFrame()

        coarse = (
            coarse
            .merge(planet_disp, on=["id", "step"], how="left")
            .loc[lambda d:
                d["dist_tgt_src"] <
                (d["step_diff"] + 1) * GameConfig.MAX_SPEED
                + d["radius_src"] + GameConfig.PLANET_MARGIN + d["radius"]
                + d["planet_disp"].fillna(0.0)
            ]
            .reset_index(drop=True)
        )

        if coarse.empty:
            return pd.DataFrame()

        # Ships_sent expansion
        expanded = (
            coarse
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

        prev_pos = (
            df_s[["id", "step", "x", "y"]]
            .assign(step=lambda d: d["step"] + 1)
            .rename(columns={"x": "x_prev", "y": "y_prev"})
        )

        expanded = (
            expanded
            .assign(fleet_speed=_fleet_speed_b, dist_min=_dist_min_b, dist_prev=_dist_prev_b)
            .loc[lambda d: d["dist_tgt_src"] < d["dist_min"] + d["fleet_speed"] + d["radius"] + GameConfig.PLANET_MOVEMENT_SLACK]
            .merge(prev_pos, on=["id", "step"], how="left")
            .reset_index(drop=True)
        )

        if expanded.empty:
            return pd.DataFrame()

        # Swept-pair collision (vectorised)
        _dx2 = expanded["x"].values - expanded["x_src"].values
        _dy2 = expanded["y"].values - expanded["y_src"].values
        _dist2 = expanded["dist_tgt_src"].values
        _ux = _dx2 / np.where(_dist2 < 1e-9, 1.0, _dist2)
        _uy = _dy2 / np.where(_dist2 < 1e-9, 1.0, _dist2)

        _xpf = expanded["x_prev"].fillna(expanded["x"]).values
        _ypf = expanded["y_prev"].fillna(expanded["y"]).values

        _fx0 = expanded["x_src"].values + _ux * expanded["dist_prev"].values
        _fy0 = expanded["y_src"].values + _uy * expanded["dist_prev"].values
        _pvx = expanded["x"].values - _xpf
        _pvy = expanded["y"].values - _ypf
        _dvx = _ux * expanded["fleet_speed"].values - _pvx
        _dvy = _uy * expanded["fleet_speed"].values - _pvy
        _d0x = _fx0 - _xpf
        _d0y = _fy0 - _ypf
        _a   = _dvx ** 2 + _dvy ** 2
        _b_  = 2.0 * (_d0x * _dvx + _d0y * _dvy)
        _c_  = _d0x ** 2 + _d0y ** 2 - expanded["radius"].values ** 2
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
        _xpf_pa = pa["x_prev"].fillna(pa["x"]).values
        _ypf_pa = pa["y_prev"].fillna(pa["y"]).values
        _t1e = np.clip(pa["t1"].values, 0.0, 1.0)
        _t2e = np.clip(pa["t2"].values, 0.0, 1.0)

        pa = (
            pa
            .assign(
                t1_eff=_t1e,
                t2_eff=_t2e,
                p_t1_x=_xpf_pa + _t1e * (pa["x"].values - _xpf_pa),
                p_t1_y=_ypf_pa + _t1e * (pa["y"].values - _ypf_pa),
                p_t2_x=_xpf_pa + _t2e * (pa["x"].values - _xpf_pa),
                p_t2_y=_ypf_pa + _t2e * (pa["y"].values - _ypf_pa),
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
                    (d["d_s_t1"] ** 2 + d["d_f_t1"] ** 2 - d["radius"] ** 2)
                    / (2.0 * d["d_s_t1"] * d["d_f_t1"]),
                    -1.0, 1.0,
                )),
                angle_radius_t2=lambda d: np.arccos(np.clip(
                    (d["d_s_t2"] ** 2 + d["d_f_t2"] ** 2 - d["radius"] ** 2)
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
            .sort_values("step")
            .reset_index(drop=True)
        )

        return pa
```

- [ ] **Step 4: Update `agent()` call site in `102-Simulate10Next.py`**

Find the `agent()` function at the bottom of the file (around line 716). Replace:

```python
    df_s, planet_disp = StrategyPipeline._01_get_obs_dataframe(obs, step, num_agents)
    pa = StrategyPipeline._02_get_all_opportunities(df_s, planet_disp, 0)
    safe_attacks = StrategyPipeline._03_filter_collision(pa)
    moves = StrategyPipeline._04_score_and_decide(safe_attacks, 0)
```

With:

```python
    df_s, planet_disp = StrategyPipeline._01_get_obs_dataframe(obs, step, num_agents)
    coarse_mine = StrategyPipeline._02_pre_mine(df_s, 0)
    coarse_all  = StrategyPipeline._02_pre_all(df_s, [4, 16, 64, 256])
    pa          = StrategyPipeline._02_get_all_opportunities(coarse_mine, df_s, planet_disp)
    pa_reach    = StrategyPipeline._02_get_all_opportunities(coarse_all,  df_s, planet_disp)
    safe_attacks = StrategyPipeline._03_filter_collision(pa)
    reach        = StrategyPipeline._03_filter_collision(pa_reach)
    moves        = StrategyPipeline._04_score_and_decide(safe_attacks, reach, 0)
```

- [ ] **Step 5: Run all tests**

```
pytest tests/ -v
```

Expected: all tests PASS.

- [ ] **Step 6: Commit**

```bash
git add 102-Simulate10Next.py tests/test_02_get_all_opportunities.py
git commit -m "refactor: _02_get_all_opportunities now takes pre-built coarse + add parallel reach matrix pipeline"
```

---

### Task 6: Update `_04_score_and_decide` signature

**Files:**
- Modify: `102-Simulate10Next.py`
- Create: `tests/test_04_score_and_decide.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_04_score_and_decide.py`:

```python
import pandas as pd
from tests.conftest import StrategyPipeline


def test_accepts_reach_matrix_as_second_arg():
    moves = StrategyPipeline._04_score_and_decide(
        pd.DataFrame(), pd.DataFrame(), player_id=0
    )
    assert moves == []


def test_existing_scoring_unaffected(simple_obs):
    from tests.conftest import StrategyPipeline
    df_s, planet_disp = StrategyPipeline._01_get_obs_dataframe(simple_obs, step=0, num_agents=2)
    coarse_mine = StrategyPipeline._02_pre_mine(df_s, 0)
    coarse_all  = StrategyPipeline._02_pre_all(df_s, [4, 16, 64, 256])
    pa          = StrategyPipeline._02_get_all_opportunities(coarse_mine, df_s, planet_disp)
    pa_reach    = StrategyPipeline._02_get_all_opportunities(coarse_all,  df_s, planet_disp)
    safe        = StrategyPipeline._03_filter_collision(pa)
    reach       = StrategyPipeline._03_filter_collision(pa_reach)
    moves       = StrategyPipeline._04_score_and_decide(safe, reach, player_id=0)
    assert isinstance(moves, list)
```

- [ ] **Step 2: Run to confirm FAIL**

```
pytest tests/test_04_score_and_decide.py -v
```

Expected: `TypeError: _04_score_and_decide() takes 2 positional arguments but 3 were given`

- [ ] **Step 3: Update `_04_score_and_decide` signature in `102-Simulate10Next.py`**

Change the method signature from:

```python
    @staticmethod
    def _04_score_and_decide(attacks_with_angle: pd.DataFrame, player_id: int) -> list:
```

To:

```python
    @staticmethod
    def _04_score_and_decide(
        attacks_with_angle: pd.DataFrame,
        reach_matrix: pd.DataFrame,
        player_id: int,
    ) -> list:
```

The body of `_04_score_and_decide` is otherwise **unchanged**.

- [ ] **Step 4: Run all tests**

```
pytest tests/ -v
```

Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
git add 102-Simulate10Next.py tests/test_04_score_and_decide.py
git commit -m "feat: _04_score_and_decide accepts reach_matrix — pipeline wired end-to-end"
```
