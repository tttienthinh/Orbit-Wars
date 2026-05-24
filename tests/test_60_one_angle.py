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
        "x_src": 20.0, "y_src": 50.0, "x": 60.0, "y": 60.0,
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
