# test_33_nearest_10steps.py
import importlib.util, pathlib, math

def load_agent():
    p = pathlib.Path(__file__).parent / "33-Kaggle_env_nearest_10steps.py"
    spec = importlib.util.spec_from_file_location("agent33", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m

m = load_agent()


class MockObs:
    def __init__(self, planets, angular_velocity=0.0):
        self.planets          = [list(p) for p in planets]
        self.initial_planets  = [list(p) for p in planets]
        self.fleets           = []
        self.next_fleet_id    = 100
        self.comets           = []
        self.comet_planet_ids = []
        self.angular_velocity = angular_velocity
        self.player           = 0


# ── _fleet_speed ──────────────────────────────────────────────────────────────
def test_fleet_speed_one_ship():
    assert m._fleet_speed(1) == 1.0

def test_fleet_speed_max():
    assert math.isclose(m._fleet_speed(1000), 6.0, rel_tol=1e-6)

def test_fleet_speed_midrange():
    v = m._fleet_speed(100)
    assert 1.0 < v < 6.0


# ── _simulate ─────────────────────────────────────────────────────────────────
_PLANETS_2P = [
    # [id, owner, x,  y,  radius, ships, production]
    [0,  0,   3,  50,  5,    50,    5],   # static (dist=47, 47+5=52 >= 50)
    [1,  1,  97,  50,  5,    10,    3],   # static
]

def test_simulate_shape():
    obs = MockObs(_PLANETS_2P)
    df = m._simulate(obs, global_step=0, num_agents=2)
    assert set(df.columns) == {"step", "id", "x", "y", "radius", "ships",
                                "production", "owner", "nature"}
    assert len(df) == 10 * 2  # 10 steps × 2 planets

def test_simulate_step_labels():
    obs = MockObs(_PLANETS_2P)
    df = m._simulate(obs, global_step=5, num_agents=2)
    assert df["step"].min() == 6
    assert df["step"].max() == 15

def test_simulate_nature_static():
    obs = MockObs(_PLANETS_2P)
    df = m._simulate(obs, global_step=0, num_agents=2)
    assert (df["nature"] == "fix").all()

def test_simulate_nature_moving():
    moving_planets = [
        [0, 0, 20, 50, 5, 50, 5],   # dist=30, 30+5=35 < 50 → moving
        [1, 1, 80, 50, 5, 10, 3],   # dist=30, moving
    ]
    obs = MockObs(moving_planets, angular_velocity=0.01)
    df = m._simulate(obs, global_step=0, num_agents=2)
    assert (df["nature"] == "moving").all()

def test_simulate_production_grows():
    obs = MockObs(_PLANETS_2P)
    df = m._simulate(obs, global_step=0, num_agents=2)
    p0_ships = df.query("id == 0").sort_values("step")["ships"].values
    assert p0_ships[-1] > p0_ships[0]


# ── _eta ──────────────────────────────────────────────────────────────────────
# Static scenario: src=(3,50) radius=5, tgt at (97,50) radius=5, no rotation
# travel_dist = 94 - 5 - 5 = 84, speed=1.0 (1 ship), ETA=84
_SRC = (3.0, 50.0, 5.0)   # x, y, radius
_TGT_STATIC = [1, 1, 97.0, 50.0, 5.0, 10, 3]  # planet list

def test_eta_static_planet():
    eta = m._eta(*_SRC, _TGT_STATIC, angular_velocity=0.0)
    assert eta == 84

def test_eta_zero_distance():
    # source and target touching — ETA should be 1 (min)
    src = (50.0, 10.0, 3.0)
    tgt = [2, 1, 50.0, 17.0, 3.0, 5, 1]  # dist between centers = 7, radii sum = 6 → gap=1
    eta = m._eta(*src, tgt, angular_velocity=0.0)
    assert eta >= 1

def test_eta_fast_spinning_moving_planet():
    # angular_velocity=999.0 creates a fast-spinning planet, which is still reachable
    # (the planet is not escaping, just rotating rapidly)
    src = (3.0, 50.0, 5.0)
    tgt = [1, 1, 30.0, 50.0, 5.0, 10, 3]  # moving planet (dist=20, 20+5=25 < 50)
    eta = m._eta(*src, tgt, angular_velocity=999.0)
    assert eta == 23


# ── _aim_angle ────────────────────────────────────────────────────────────────
def test_aim_angle_static_horizontal():
    # src left of tgt, same y → angle should be 0.0 (pointing right)
    angle = m._aim_angle(*_SRC, _TGT_STATIC, angular_velocity=0.0, ships=1)
    assert math.isclose(angle, 0.0, abs_tol=1e-9)

def test_aim_angle_static_vertical():
    src = (50.0, 3.0, 5.0)
    tgt = [2, 1, 50.0, 97.0, 5.0, 10, 3]
    angle = m._aim_angle(*src, tgt, angular_velocity=0.0, ships=1)
    assert math.isclose(angle, math.pi / 2, abs_tol=1e-9)


# ── nearest_planet_sniper integration ─────────────────────────────────────────

def _fresh_board():
    """Return a fresh global_board dict (avoids shared state between tests)."""
    return {"step": 0, "num_agents": None}

def test_agent_returns_list():
    import kaggle_environments as ke
    env = ke.make("orbit_wars", debug=False)
    env.reset(2)
    obs = env.state[0].observation
    gb = _fresh_board()
    result = m.nearest_planet_sniper(obs, gb)
    assert isinstance(result, list)

def test_agent_move_format():
    """Every move must be [planet_id, angle, ships] with ships >= 1."""
    import kaggle_environments as ke
    env = ke.make("orbit_wars", debug=False)
    env.reset(2)
    obs = env.state[0].observation
    gb = _fresh_board()
    moves = m.nearest_planet_sniper(obs, gb)
    for move in moves:
        assert len(move) == 3
        pid, angle, ships = move
        assert isinstance(pid, int)
        assert -math.pi <= angle <= math.pi
        assert ships >= 1

def test_agent_increments_step():
    import kaggle_environments as ke
    env = ke.make("orbit_wars", debug=False)
    env.reset(2)
    obs = env.state[0].observation
    gb = _fresh_board()
    m.nearest_planet_sniper(obs, gb)
    assert gb["step"] == 1

def test_agent_detects_num_agents_2p():
    import kaggle_environments as ke
    env = ke.make("orbit_wars", debug=False)
    env.reset(2)
    obs = env.state[0].observation
    gb = _fresh_board()
    m.nearest_planet_sniper(obs, gb)
    assert gb["num_agents"] == 2

def test_agent_no_crash_10_steps():
    """Run 10 full turns without exception."""
    import kaggle_environments as ke
    env = ke.make("orbit_wars", debug=False)
    env.reset(2)
    gb = _fresh_board()
    for _ in range(10):
        obs = env.state[0].observation
        moves = m.nearest_planet_sniper(obs, gb)
        assert isinstance(moves, list)
        env.step([moves, []])
