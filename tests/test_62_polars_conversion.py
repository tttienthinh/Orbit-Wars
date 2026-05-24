# tests/test_62_polars_conversion.py
import sys
import os
import importlib.util
import math
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_62():
    path = os.path.join(ROOT, "62-One_angle_polars.py")
    spec = importlib.util.spec_from_file_location("mod62", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["mod62"] = mod
    spec.loader.exec_module(mod)
    return mod


def _make_df(rows_per_planet=11):
    """Two-planet fixture: planet 1 (mine) at (20,80), planet 2 (enemy) at (20,50).
    Path is vertical x=20; sun at (50,50) is 30 units away — no crossing.
    At step_diff=10, ships_sent≈12 reaches planet 2 (distance 30, radius 5).
    """
    rows = []
    for step in range(rows_per_planet):
        rows.append({
            "step": step, "id": 1, "x": 20.0, "y": 80.0, "radius": 5.0,
            "ships": 50, "production": 3, "owner": 0, "nature": "fix",
        })
        rows.append({
            "step": step, "id": 2, "x": 20.0, "y": 50.0, "radius": 5.0,
            "ships": 10, "production": 1, "owner": 1, "nature": "fix",
        })
    return pd.DataFrame(rows)


def _make_sun_blocked_df(rows_per_planet=11):
    """Planets aligned through sun: (40,50) → (60,50). Path passes through sun at (50,50).
    All attacks should be filtered out → moves = [].
    """
    rows = []
    for step in range(rows_per_planet):
        rows.append({
            "step": step, "id": 1, "x": 40.0, "y": 50.0, "radius": 3.0,
            "ships": 50, "production": 3, "owner": 0, "nature": "fix",
        })
        rows.append({
            "step": step, "id": 2, "x": 60.0, "y": 50.0, "radius": 3.0,
            "ships": 5, "production": 1, "owner": 1, "nature": "fix",
        })
    return pd.DataFrame(rows)


def _normalize(moves):
    """Normalize move list to Python native types for stable comparison."""
    return [[int(m[0]), float(m[1]), int(m[2])] for m in moves]


mod62 = load_62()


def test_take_action_produces_moves():
    """Mine planet attacks enemy planet — expect at least one move."""
    df = _make_df()
    moves = mod62.take_action(df, player_id=0)
    assert len(moves) >= 1
    for m in moves:
        assert len(m) == 3, "each move is [id_src, angle, ships_sent]"
        id_src, angle, ships_sent = m
        assert int(id_src) == 1
        assert isinstance(float(angle), float)
        assert int(ships_sent) > 0


def test_take_action_angle_is_downward():
    """Planet 1 at (20,80) attacking planet 2 at (20,50): angle should be ≈ -π/2."""
    df = _make_df()
    moves = mod62.take_action(df, player_id=0)
    assert len(moves) >= 1
    _, angle, _ = moves[0]
    assert math.isclose(float(angle), -math.pi / 2, abs_tol=0.1)


def test_take_action_no_mine_returns_empty():
    """player_id with no owned planets returns empty move list."""
    df = _make_df()
    moves = mod62.take_action(df, player_id=99)
    assert moves == []


def test_take_action_sun_blocked_path_returns_empty():
    """Path directly through sun is rejected by crossing_sun filter."""
    df = _make_sun_blocked_df()
    moves = mod62.take_action(df, player_id=0)
    assert moves == []


def test_take_action_return_df_flag():
    """return_df=True returns (moves, DataFrame) tuple with a polars DataFrame."""
    import polars as pl
    df = _make_df()
    result = mod62.take_action(df, player_id=0, return_df=True)
    assert isinstance(result, tuple) and len(result) == 2
    moves, attacks_df = result
    assert isinstance(moves, list)
    assert isinstance(attacks_df, pl.DataFrame)
