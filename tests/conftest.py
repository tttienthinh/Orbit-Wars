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
