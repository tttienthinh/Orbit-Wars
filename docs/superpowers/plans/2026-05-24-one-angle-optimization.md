# One-Angle Attack Filter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create `60-Dataframe_one_angle.py` which replaces the `IntervalProcessor` cumulative-interval pipeline with a vectorized pandas cross-merge that keeps only attacks whose direct angle to the target is not blocked by any earlier-step obstacle.

**Architecture:** Extract the blocking logic into a module-level helper `_filter_blocked_attacks(possible_attacks)` that self-merges `possible_attacks` on `(id_src, ships_sent)`, identifies (target, obstacle) pairs where `step_obs < step_tgt` and the target's angle falls inside the obstacle's cone, then anti-joins to drop blocked targets. `take_action` calls this helper instead of the three-step `IntervalProcessor` pipeline. `IntervalProcessor` is removed entirely.

**Tech Stack:** Python 3, pandas, numpy, pytest, importlib (for loading dash-named modules in tests)

---

### Task 1: Write failing tests for `_filter_blocked_attacks`

**Files:**
- Create: `tests/test_60_one_angle.py`

- [ ] **Step 1: Create the test file**

```python
# tests/test_60_one_angle.py
import sys
import os
import importlib.util
import math
import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def load_module():
    path = os.path.join(ROOT, "60-Dataframe_one_angle.py")
    spec = importlib.util.spec_from_file_location("mod60", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["mod60"] = mod
    spec.loader.exec_module(mod)
    return mod

mod = load_module()
_filter_blocked_attacks = mod._filter_blocked_attacks


def _make_row(id_src, ships_sent, step, id_, angle, angle_min, angle_max):
    return {
        "id_src": id_src, "ships_sent": ships_sent, "step": step, "id": id_,
        "angle": angle, "angle_min": angle_min, "angle_max": angle_max,
        # Extra columns present in real possible_attacks — not used by the filter
        "x_src": 20.0, "y_src": 50.0, "x": 60.0, "y": 50.0,
        "radius_src": 3.0, "ships_min": 10, "production_src": 1,
        "nature_src": "fix", "owner_src": 0,
        "radius": 3.0, "ships": 5, "production": 1, "owner": 1,
        "dist_tgt_src": 40.0, "step_diff": step,
        "fleet_speed": 1.0, "dist_fleet_src_min": 38.0, "dist_fleet_src_max": 39.0,
    }


def test_no_obstacles_all_pass():
    """Single target with no earlier-step obstacles survives unchanged."""
    pa = pd.DataFrame([_make_row(0, 5, 1, 1, 0.0, -0.2, 0.2)])
    result = _filter_blocked_attacks(pa)
    assert len(result) == 1
    assert result["final_angle"].iloc[0] == pytest.approx(0.0)


def test_earlier_obstacle_blocks_shot():
    """Target at step 2 is blocked when its angle falls inside earlier planet's cone."""
    pa = pd.DataFrame([
        _make_row(0, 5, 1, 1, 0.0, -0.5, 0.5),   # obstacle at step 1
        _make_row(0, 5, 2, 2, 0.0, -0.3, 0.3),   # target at step 2, angle=0.0 ∈ [-0.5, 0.5]
    ])
    result = _filter_blocked_attacks(pa)
    assert len(result) == 1
    assert result["id"].iloc[0] == 1   # only the obstacle-step planet survives


def test_obstacle_different_angle_does_not_block():
    """Target at step 2 survives when its angle is outside the earlier planet's cone."""
    pa = pd.DataFrame([
        _make_row(0, 5, 1, 1, 0.0,  -0.5,  0.5),  # obstacle cone [-0.5, 0.5]
        _make_row(0, 5, 2, 2, 2.0,   1.7,  2.3),  # target angle=2.0 ∉ [-0.5, 0.5]
    ])
    result = _filter_blocked_attacks(pa)
    assert len(result) == 2


def test_wrapped_cone_blocks():
    """Wrapped obstacle cone (angle_min > angle_max) correctly blocks a target."""
    # cone wraps: covers [5.9, 2π] ∪ [0, 0.4]
    pa = pd.DataFrame([
        _make_row(0, 5, 1, 1, 6.1, 5.9, 0.4),   # obstacle with wrapped cone
        _make_row(0, 5, 2, 2, 6.0, 5.8, 6.2),   # target angle=6.0 >= 5.9 → blocked
    ])
    result = _filter_blocked_attacks(pa)
    assert len(result) == 1
    assert result["id"].iloc[0] == 1


def test_negative_angle_normalized():
    """Negative atan2 angle is normalised to [0, 2π] before cone comparison."""
    # angle=-0.1 ≡ 6.18 in [0, 2π]; obstacle cone [5.9, 6.3] should block it
    pa = pd.DataFrame([
        _make_row(0, 5, 1, 1, 6.1, 5.9, 6.3),   # obstacle cone [5.9, 6.3]
        _make_row(0, 5, 2, 2, -0.1, -0.3, 0.1), # target angle=-0.1 ≡ 6.18 ∈ [5.9, 6.3]
    ])
    result = _filter_blocked_attacks(pa)
    assert len(result) == 1
    assert result["id"].iloc[0] == 1


def test_different_ships_sent_are_independent():
    """Obstacle in ships=5 group does not block ships=10 group."""
    pa = pd.DataFrame([
        # ships=5: obstacle at step 1 blocks target at step 2
        _make_row(0,  5, 1, 1, 0.0, -0.5, 0.5),
        _make_row(0,  5, 2, 2, 0.0, -0.3, 0.3),
        # ships=10: only target at step 2, no earlier obstacle
        _make_row(0, 10, 2, 2, 0.0, -0.3, 0.3),
    ])
    result = _filter_blocked_attacks(pa)
    assert len(result) == 2
    assert set(result["ships_sent"].tolist()) == {5, 10}
    assert result[result["ships_sent"] == 5]["id"].iloc[0] == 1
    assert result[result["ships_sent"] == 10]["id"].iloc[0] == 2
```

- [ ] **Step 2: Run tests — expect ImportError (module does not exist yet)**

```
pytest tests/test_60_one_angle.py -v
```

Expected: `ModuleNotFoundError` or file-not-found on `60-Dataframe_one_angle.py`. All tests should error at collection time, not pass.

---

### Task 2: Create `60-Dataframe_one_angle.py`

**Files:**
- Create: `60-Dataframe_one_angle.py`

- [ ] **Step 1: Copy `59-Dataframe_sun_collision.py` as the base**

```
cp "59-Dataframe_sun_collision.py" "60-Dataframe_one_angle.py"
```

- [ ] **Step 2: Remove the `IntervalProcessor` class**

Delete the entire `class IntervalProcessor:` block (all five methods: `merge_intervals`, `create_cumulative_obstacles`, `subtract_intervals`, `compute_free_angles`, `interval_to_final_angle`). In the file from Task 1 copy, this class starts just after `import numpy as np` and ends just before `def take_action(...)`.

- [ ] **Step 3: Add `_filter_blocked_attacks` before `take_action`**

Insert this function in place of the removed `IntervalProcessor` class:

```python
def _filter_blocked_attacks(possible_attacks):
    """Drop attacks whose direct angle to target is blocked by an earlier obstacle.

    For each (id_src, ships_sent) trajectory, an obstacle at step_obs blocks a
    target at step_tgt when step_obs < step_tgt and the direct angle to the target
    falls inside the obstacle's collision cone [angle_min_obs, angle_max_obs].
    The cone may wrap around 2π (angle_min > angle_max); handled via np.where.
    The target's raw atan2 angle (in [-π, π]) is normalised to [0, 2π] so that
    negative angles compare correctly against cone bounds in [0, 2π].
    """
    pairs = (
        possible_attacks[["id_src", "ships_sent", "step", "id", "angle", "angle_min", "angle_max"]]
        .merge(
            possible_attacks[["id_src", "ships_sent", "step", "id", "angle_min", "angle_max"]]
            .rename(columns={
                "step": "step_obs",
                "id": "id_obs",
                "angle_min": "angle_min_obs",
                "angle_max": "angle_max_obs",
            }),
            on=["id_src", "ships_sent"],
        )
        .query("step_obs < step and id_obs != id")
    )

    if pairs.empty:
        return possible_attacks.assign(final_angle=lambda d: d["angle"])

    angle_norm = np.mod(pairs["angle"].values, 2 * np.pi)
    pairs = pairs.assign(
        is_blocked=np.where(
            pairs["angle_min_obs"].values > pairs["angle_max_obs"].values,
            (angle_norm >= pairs["angle_min_obs"].values) | (angle_norm <= pairs["angle_max_obs"].values),
            (angle_norm >= pairs["angle_min_obs"].values) & (angle_norm <= pairs["angle_max_obs"].values),
        )
    )

    blocked = (
        pairs.query("is_blocked")[["id_src", "ships_sent", "step", "id"]]
        .drop_duplicates()
    )

    return (
        possible_attacks
        .merge(blocked, on=["id_src", "ships_sent", "step", "id"], how="left", indicator=True)
        .query("_merge == 'left_only'")
        .drop(columns="_merge")
        .assign(final_angle=lambda d: d["angle"])
    )
```

- [ ] **Step 4: Replace the interval pipeline in `take_action`**

Find this block in `take_action` (just after the `if possible_attacks.empty:` guard):

```python
    df_obstacles = IntervalProcessor.create_cumulative_obstacles(possible_attacks)

    attacks_with_angle = (
        possible_attacks
        .merge(
            df_obstacles,
            how="left",
            on=["id_src", "step", "ships_sent"],
        )
        .assign(
            angle_list=lambda d: d.apply(IntervalProcessor.compute_free_angles, axis=1)
        )
        .query("angle_list.str.len() > 0")  # Only keep rows where there is at least some free angle
    )
```

Replace it with:

```python
    attacks_with_angle = _filter_blocked_attacks(possible_attacks)
```

- [ ] **Step 5: Remove `interval_to_final_angle` call from the `attacks` block**

Find this assign at the end of the `attacks` chain:

```python
        .assign(
            final_angle = lambda d: IntervalProcessor.interval_to_final_angle(d["angle_list"])
        )
```

Delete it. `final_angle` is already present on `attacks_with_angle` from `_filter_blocked_attacks`.

---

### Task 3: Run tests and verify

- [ ] **Step 1: Run the unit tests**

```
pytest tests/test_60_one_angle.py -v
```

Expected output (all 6 pass):
```
tests/test_60_one_angle.py::test_no_obstacles_all_pass PASSED
tests/test_60_one_angle.py::test_earlier_obstacle_blocks_shot PASSED
tests/test_60_one_angle.py::test_obstacle_different_angle_does_not_block PASSED
tests/test_60_one_angle.py::test_wrapped_cone_blocks PASSED
tests/test_60_one_angle.py::test_negative_angle_normalized PASSED
tests/test_60_one_angle.py::test_different_ships_sent_are_independent PASSED
```

If any test fails, check:
- Normalisation line: `angle_norm = np.mod(pairs["angle"].values, 2 * np.pi)`
- Wrap condition: `pairs["angle_min_obs"].values > pairs["angle_max_obs"].values`
- Anti-join: `indicator=True` + `.query("_merge == 'left_only'")`

- [ ] **Step 2: Run the full test suite to check for regressions**

```
pytest tests/ -v --ignore=tests/test_notebook19_sanity.py --ignore=tests/test_notebook20_sanity.py
```

Expected: all previously-passing tests still pass.

---

### Task 4: Commit

- [ ] **Step 1: Stage and commit**

```bash
git add 60-Dataframe_one_angle.py tests/test_60_one_angle.py
git commit -m "feat: add 60-Dataframe_one_angle with cross-merge block check replacing IntervalProcessor"
```
