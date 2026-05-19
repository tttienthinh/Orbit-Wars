# test_simulator.py
import math
import importlib.util
from types import SimpleNamespace
import pytest

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


# --- Production ---

def test_production_adds_ships_to_owned_planet():
    planet = [0, 0, 60.0, 50.0, 2.0, 10, 3]   # owner=0, ships=10, prod=3
    obs = make_obs([planet])
    sim = OrbitWarsSimulator(obs)
    snap = sim.step()
    p = next(s for s in snap if s['id'] == 0)
    assert p['ships'] == 13


def test_production_skips_neutral_planet():
    planet = [0, -1, 60.0, 50.0, 2.0, 10, 3]  # owner=-1
    obs = make_obs([planet])
    sim = OrbitWarsSimulator(obs)
    snap = sim.step()
    p = next(s for s in snap if s['id'] == 0)
    assert p['ships'] == 10


# --- Fleet movement ---

def test_fleet_removed_when_out_of_bounds():
    planet = [0, -1, 60.0, 50.0, 2.0, 5, 1]
    # Fleet aimed at angle π/2 (+y direction) from near the bottom edge.
    # 1 ship → speed=1.0; y=99.5+1=100.5 > 100 → out of bounds after 1 step.
    fleet = [0, 0, 50.0, 99.5, math.pi / 2, -1, 1]
    obs = make_obs([planet], fleets=[fleet])
    sim = OrbitWarsSimulator(obs)
    sim.step()
    assert len(sim.fleets) == 0


def test_fleet_removed_when_crossing_sun():
    planet = [0, -1, 80.0, 50.0, 2.0, 5, 1]
    # Fleet aimed directly at sun center (50, 50) from (50, 63).
    # 100 ships → speed≈3.72; segment (50,63)→(50,59.28); dist to (50,50)=9.28<10 ✓
    fleet = [0, 0, 50.0, 63.0, -math.pi / 2, -1, 100]  # 100 ships → fast
    obs = make_obs([planet], fleets=[fleet])
    sim = OrbitWarsSimulator(obs)
    sim.step()
    assert len(sim.fleets) == 0


@pytest.mark.xfail(reason="combat resolution stubbed until Task 5", strict=True)
def test_fleet_hits_static_planet_after_8_steps():
    # Planet at (60, 50) radius=3. Fleet at (50, 50) angle=0, 1 ship (speed=1.0).
    # Distance to surface: (60-3) - 50 = 7 → hits when fleet x ≥ 57.
    # After 8 steps: fleet x=58, seg (57→58), pt_seg_dist((60,50),(57,50),(58,50))=2<3 ✓
    planet = [0, 1, 60.0, 50.0, 3.0, 20, 2]   # owner=1, ships=20, prod=2
    fleet  = [0, 0, 50.0, 50.0, 0.0, -1, 1]   # 1 ship, speed=1.0
    obs = make_obs([planet], fleets=[fleet])
    sim = OrbitWarsSimulator(obs)
    for _ in range(8):
        snap = sim.step()
    assert len(sim.fleets) == 0
    p = next(s for s in snap if s['id'] == 0)
    # Production: +2/step × 8 steps = +16 → 36 ships on planet at moment of combat.
    # Fleet 1 ship (owner 0) vs planet owner 1: survivor_ships=1, tries to take planet.
    # planet[5] -= 1 → 35, planet keeps owner 1.
    assert p['owner'] == 1
    assert p['ships'] == 35


# --- Planet rotation ---

def test_orbiting_planet_rotates_by_omega():
    omega = 0.05
    # Planet at (60, 50): orbital radius=10, radius=1 → 10+1=11 < 50 → orbiting
    planet = [0, -1, 60.0, 50.0, 1.0, 5, 1]
    obs = make_obs([planet], angular_velocity=omega)
    sim = OrbitWarsSimulator(obs)
    snap = sim.step()  # sim_step=1
    p = next(s for s in snap if s['id'] == 0)
    initial_angle = math.atan2(50.0 - 50.0, 60.0 - 50.0)  # atan2(0, 10) = 0
    expected_x = 50.0 + 10.0 * math.cos(initial_angle + omega)
    expected_y = 50.0 + 10.0 * math.sin(initial_angle + omega)
    assert abs(p['x'] - expected_x) < 1e-9
    assert abs(p['y'] - expected_y) < 1e-9


def test_static_planet_does_not_rotate():
    # Planet at (90, 50): orbital radius=40, radius=15 → 40+15=55 ≥ 50 → static
    planet = [0, -1, 90.0, 50.0, 15.0, 5, 1]
    obs = make_obs([planet], angular_velocity=0.05)
    sim = OrbitWarsSimulator(obs)
    snap = sim.step()
    p = next(s for s in snap if s['id'] == 0)
    assert p['x'] == 90.0
    assert p['y'] == 50.0


def test_orbiting_planet_sweeps_fleet():
    omega = 0.1
    # Planet at (60, 50), radius=3, orbiting.
    # Place a fleet right where the planet will be after 1 rotation step.
    r = 10.0
    angle_after = math.atan2(0, r) + omega  # initial angle 0 → rotates to omega
    fleet_x = 50.0 + r * math.cos(angle_after)
    fleet_y = 50.0 + r * math.sin(angle_after)
    planet = [0, 0, 60.0, 50.0, 3.0, 5, 1]
    fleet  = [0, 1, fleet_x, fleet_y, 0.0, -1, 1]  # stationary-ish fleet at sweep target
    obs = make_obs([planet], fleets=[fleet], angular_velocity=omega)
    sim = OrbitWarsSimulator(obs)
    sim.step()
    assert len(sim.fleets) == 0  # fleet was swept
