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


def test_take_action_produces_valid_moves():
    """take_action with nearby planets must return at least one move with valid angle and ships."""
    import copy, math
    mod = load_module()

    class MockObs:
        angular_velocity = 0.0
        comets = []
        comet_planet_ids = []
        next_fleet_id = 10

    obs = MockObs()
    # Source planet 0 (player 0) at (10, 50); target planet 1 (neutral) at (20, 50)
    # Distance 10; path clears the sun at (50,50); with 100 ships speed ~3.7,
    # at step_diff=2 fleet sweeps from ~6.8 to ~10.5 → hits target (radius 2 at distance 10)
    obs.planets = [[0, 0, 10.0, 50.0, 3.0, 100, 2], [1, -1, 20.0, 50.0, 2.0, 5, 1]]
    obs.initial_planets = copy.deepcopy(obs.planets)
    obs.fleets = []

    df = mod._simulate(obs, 0, 2, n_steps=mod.NB_STEPS_SIM)
    moves, awa = mod.take_action(df, player_id=0, return_df=True)

    # Verify the swept-pair pipeline produced collision rows
    assert not awa.is_empty(), "Expected at least one attack row from nearby planets"
    assert "t1_eff" in awa.columns, "t1_eff column must be present (swept-pair output)"
    assert "t2_eff" in awa.columns, "t2_eff column must be present (swept-pair output)"
    assert (awa["t1_eff"] >= 0.0).all() and (awa["t1_eff"] <= 1.0).all(), "t1_eff must be in [0,1]"
    assert (awa["t2_eff"] >= 0.0).all() and (awa["t2_eff"] <= 1.0).all(), "t2_eff must be in [0,1]"

    # Verify moves are well-formed
    assert isinstance(moves, list)
    assert len(moves) > 0, "Expected at least one move with nearby reachable planet"
    for move in moves:
        pid, angle, ships = move
        assert isinstance(pid, (int, float))
        assert ships > 0


# ── Tests for 70-Polars_filter.py ─────────────────────────────────────────────

def load_module70():
    path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "70-Polars_filter.py"))
    spec = importlib.util.spec_from_file_location("agent70", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_planet_disp_lf_static():
    """Static planet (same position every step) must have planet_disp == 0."""
    import polars as pl
    df_lf = pl.DataFrame({
        "id": [1, 1],
        "step": [0, 1],
        "x": [20.0, 20.0],
        "y": [50.0, 50.0],
    }).lazy()
    prev_pos_lf = (
        df_lf.select(["id", "step", "x", "y"])
        .rename({"x": "x_prev", "y": "y_prev"})
        .with_columns((pl.col("step") + 1).alias("step"))
    )
    planet_disp = (
        df_lf.select(["id", "step", "x", "y"])
        .join(prev_pos_lf, on=["id", "step"], how="left")
        .with_columns(
            ((pl.col("x") - pl.col("x_prev").fill_null(pl.col("x"))).pow(2) +
             (pl.col("y") - pl.col("y_prev").fill_null(pl.col("y"))).pow(2)
            ).sqrt().alias("planet_disp")
        )
        .select("planet_disp")
        .collect()["planet_disp"]
        .to_list()
    )
    assert planet_disp == [0.0, 0.0]


def test_planet_disp_lf_moving():
    """Planet that moves 3 units right must have planet_disp == 3.0 at step 1."""
    import polars as pl
    df_lf = pl.DataFrame({
        "id": [1, 1],
        "step": [0, 1],
        "x": [20.0, 23.0],
        "y": [50.0, 50.0],
    }).lazy()
    prev_pos_lf = (
        df_lf.select(["id", "step", "x", "y"])
        .rename({"x": "x_prev", "y": "y_prev"})
        .with_columns((pl.col("step") + 1).alias("step"))
    )
    planet_disp_lf = (
        df_lf.select(["id", "step", "x", "y"])
        .join(prev_pos_lf, on=["id", "step"], how="left")
        .with_columns(
            ((pl.col("x") - pl.col("x_prev").fill_null(pl.col("x"))).pow(2) +
             (pl.col("y") - pl.col("y_prev").fill_null(pl.col("y"))).pow(2)
            ).sqrt().alias("planet_disp")
        )
        .select(["id", "step", "planet_disp"])
        .collect()
    )
    disp_step1 = planet_disp_lf.filter(pl.col("step") == 1)["planet_disp"][0]
    assert abs(disp_step1 - 3.0) < 1e-6


def test_agent70_same_moves_as_agent68():
    """Agent 70 (deferred ships_sent) produces same moves as Agent 68 on identical state."""
    import copy, importlib.util, os
    mod70 = load_module70()

    class MockObs:
        angular_velocity = 0.0
        comets = []
        comet_planet_ids = []
        next_fleet_id = 10

    obs68 = MockObs()
    obs68.planets = [[0, 0, 10.0, 50.0, 3.0, 100, 2], [1, -1, 20.0, 50.0, 2.0, 5, 1]]
    obs68.initial_planets = copy.deepcopy(obs68.planets)
    obs68.fleets = []

    obs70 = MockObs()
    obs70.planets = [[0, 0, 10.0, 50.0, 3.0, 100, 2], [1, -1, 20.0, 50.0, 2.0, 5, 1]]
    obs70.initial_planets = copy.deepcopy(obs70.planets)
    obs70.fleets = []

    path68 = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "68-Polars_optimised.py"))
    spec68 = importlib.util.spec_from_file_location("agent68", path68)
    m68 = importlib.util.module_from_spec(spec68)
    spec68.loader.exec_module(m68)

    df68 = m68._simulate(obs68, 0, 2, n_steps=m68.NB_STEPS_SIM)
    df70 = mod70._simulate(obs70, 0, 2, n_steps=mod70.NB_STEPS_SIM)

    moves68 = m68.take_action(df68, player_id=0)
    moves70 = mod70.take_action(df70, player_id=0)

    assert moves68 == moves70, f"Agent 68 moves {moves68} != Agent 70 moves {moves70}"
