# test_simulator.py
import math
import importlib.util
from types import SimpleNamespace

spec = importlib.util.spec_from_file_location("board", "27-Board_new_env.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
OrbitWarsSimulator = mod.OrbitWarsSimulator


def make_obs(planets, fleets=None, angular_velocity=0.05,
             comet_planet_ids=None, comets=None):
    obs = SimpleNamespace()
    obs.angular_velocity = angular_velocity
    obs.planets = [list(p) for p in planets]
    obs.fleets = [list(f) for f in (fleets or [])]
    obs.comet_planet_ids = list(comet_planet_ids or [])
    obs.comets = list(comets or [])
    return obs


# --- _pt_seg_dist ---

def test_pt_seg_dist_point_on_segment():
    assert OrbitWarsSimulator._pt_seg_dist((5, 0), (0, 0), (10, 0)) == 0.0


def test_pt_seg_dist_perpendicular():
    dist = OrbitWarsSimulator._pt_seg_dist((5, 3), (0, 0), (10, 0))
    assert abs(dist - 3.0) < 1e-9


def test_pt_seg_dist_past_end():
    dist = OrbitWarsSimulator._pt_seg_dist((15, 0), (0, 0), (10, 0))
    assert abs(dist - 5.0) < 1e-9


def test_pt_seg_dist_degenerate_segment():
    dist = OrbitWarsSimulator._pt_seg_dist((3, 4), (0, 0), (0, 0))
    assert abs(dist - 5.0) < 1e-9


# --- __init__ ---

def test_init_deep_copies_planets():
    obs = make_obs([[0, -1, 60.0, 50.0, 2.0, 10, 3]])
    sim = OrbitWarsSimulator(obs)
    obs.planets[0][5] = 999
    assert sim.planets[0][5] == 10


def test_init_deep_copies_fleets():
    fleet = [0, 0, 55.0, 50.0, 0.0, -1, 5]
    obs = make_obs([[0, 0, 60.0, 50.0, 2.0, 10, 1]], fleets=[fleet])
    sim = OrbitWarsSimulator(obs)
    obs.fleets[0][6] = 999
    assert sim.fleets[0][6] == 5


def test_init_builds_initial_by_id():
    obs = make_obs([[7, -1, 70.0, 50.0, 2.0, 5, 1]])
    sim = OrbitWarsSimulator(obs)
    assert 7 in sim.initial_by_id
    assert sim.initial_by_id[7][2] == 70.0


def test_init_comet_pid_set():
    obs = make_obs([[0, -1, 60.0, 50.0, 1.0, 5, 1]], comet_planet_ids=[0])
    sim = OrbitWarsSimulator(obs)
    assert 0 in sim.comet_pid_set


def test_init_sim_step_zero():
    obs = make_obs([[0, -1, 60.0, 50.0, 2.0, 5, 1]])
    sim = OrbitWarsSimulator(obs)
    assert sim.sim_step == 0


def test_init_deep_copies_comet_paths():
    path0 = [[10.0, 20.0], [11.0, 21.0]]
    path1 = [[30.0, 40.0], [31.0, 41.0]]
    group = {"planet_ids": [100, 101], "paths": [path0, path1], "path_index": 0}
    planet0 = [100, -1, 10.0, 20.0, 1.0, 5, 1]
    planet1 = [101, -1, 30.0, 40.0, 1.0, 5, 1]
    obs = make_obs([planet0, planet1], comet_planet_ids=[100, 101], comets=[group])
    sim = OrbitWarsSimulator(obs)
    obs.comets[0]["paths"][0].append([99.0, 99.0])   # mutate original
    assert len(sim.comets[0]["paths"][0]) == 2        # sim unaffected
