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
