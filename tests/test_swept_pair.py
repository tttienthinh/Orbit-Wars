import math, importlib.util, os

def load_module():
    path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "66-One_angle_polars_updated.py"))
    spec = importlib.util.spec_from_file_location("agent66", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def test_swept_pair_static_inside():
    mod = load_module()
    assert mod.swept_pair_hit((0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0), 1.0)

def test_swept_pair_static_miss():
    mod = load_module()
    assert not mod.swept_pair_hit((0.0, 0.0), (0.0, 0.0), (5.0, 0.0), (5.0, 0.0), 1.0)

def test_swept_pair_fleet_through_static_planet():
    mod = load_module()
    assert mod.swept_pair_hit((0.0, 0.0), (10.0, 0.0), (5.0, 0.0), (5.0, 0.0), 0.5)

def test_swept_pair_fleet_misses_static_planet():
    mod = load_module()
    assert not mod.swept_pair_hit((0.0, 0.0), (10.0, 0.0), (5.0, 2.0), (5.0, 2.0), 0.5)

def test_swept_pair_tunneling_detected():
    """Fleet and planet cross paths — old point_to_segment_distance misses, swept detects."""
    mod = load_module()
    A, B   = (0.0, 0.5), (2.0, 0.5)   # fleet moves right at y=0.5
    P0, P1 = (1.0, 1.5), (1.0, -0.5)  # planet moves down through y=0.5
    r = 0.6
    assert mod.swept_pair_hit(A, B, P0, P1, r)
    # Confirm old static check would miss
    assert mod.point_to_segment_distance(P0, A, B) > r
    assert mod.point_to_segment_distance(P1, A, B) > r


def make_obs(planets, fleets, initial_planets, angular_velocity=0.0):
    class Obs:
        pass
    obs = Obs()
    obs.comets = []
    obs.comet_planet_ids = []
    obs.next_fleet_id = 100
    obs.planets = [list(p) for p in planets]
    obs.fleets = [list(f) for f in fleets]
    obs.initial_planets = [list(p) for p in initial_planets]
    obs.angular_velocity = angular_velocity
    return obs

def test_interpreter_fleet_hits_static_planet():
    """Basic sanity: fleet aimed at static planet is caught."""
    mod = load_module()
    # Planet id=0, owner=1, at (56,50), radius=2 — outside orbit radius (dist=6 < 48 ✓)
    planet = [0, 1, 56.0, 50.0, 2.0, 10, 1]
    # Fleet aimed east at speed ~6 (100 ships), starting just west of planet
    fleet = [0, 0, 53.5, 50.0, 0.0, -1, 100]
    obs = make_obs([planet], [fleet], [planet], angular_velocity=0.0)
    result = mod.interpreter(obs, [[], []], 1, 2)
    assert len(result["fleets"]) == 0, "Fleet must be removed after hitting planet"

def test_interpreter_rotating_planet_sweeps_fleet():
    """Rotating planet sweeps through a stationary fleet — must be caught."""
    mod = load_module()
    # Planet orbits at radius 10 from sun (50,50), starts at (50,40)
    # angular_velocity = pi/2 → moves to (60,50) after 1 tick
    # Fleet sits at (55, 45) — on the chord from (50,40) to (60,50)
    planet = [0, 1, 50.0, 40.0, 2.0, 10, 1]
    # Fleet with 1 ship → speed=1, barely moves; place it on the planet's chord
    fleet = [0, 0, 55.0, 45.0, 0.0, -1, 1]
    obs = make_obs([planet], [fleet], [planet], angular_velocity=math.pi / 2)
    result = mod.interpreter(obs, [[], []], 1, 2)
    assert len(result["fleets"]) == 0, "Fleet on planet sweep path must be caught"


def test_prev_pos_join_present():
    """After the prev-pos join, pa_lf intermediate should have x_prev/y_prev."""
    import polars as pl, pandas as pd
    mod = load_module()
    # Minimal df: two steps of one planet
    rows = [
        {"step": 0, "id": 1, "x": 50.0, "y": 40.0, "radius": 2.0,
         "ships": 10, "production": 1, "owner": 0, "nature": "moving"},
        {"step": 1, "id": 1, "x": 51.0, "y": 41.0, "radius": 2.0,
         "ships": 11, "production": 1, "owner": 0, "nature": "moving"},
    ]
    df = pd.DataFrame(rows)
    df_lf = pl.from_pandas(df).sort("step").lazy()
    prev_pos_lf = (
        df_lf.select(["id", "step", "x", "y"])
        .rename({"x": "x_prev", "y": "y_prev"})
        .with_columns((pl.col("step") + 1).alias("step"))
    )
    joined = df_lf.join(prev_pos_lf, on=["id", "step"], how="left").collect()
    row_step1 = joined.filter(pl.col("step") == 1)
    assert row_step1["x_prev"][0] == 50.0
    assert row_step1["y_prev"][0] == 40.0
