import importlib.util
import os
import types
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
    """Two fixed planets: mine (id=0) at (20,10), enemy (id=1) at (45,10).
    dist=25, well within NB_STEPS_SIM=10 reach. Path stays 40 units from sun."""
    planets = [
        [0, 0, 20.0, 10.0, 3.0, 10, 1],
        [1, 1, 45.0, 10.0, 3.0,  5, 1],
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
