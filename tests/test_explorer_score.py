# tests/test_explorer_score.py
import os
import sys
import importlib.util

import kaggle_environments as ke

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_module(filename, module_name):
    path = os.path.join(ROOT, filename)
    spec = importlib.util.spec_from_file_location(module_name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


class MockTarget:
    def __init__(self, production, ships, owner=-1):
        self.production = production
        self.ships = ships
        self.owner = owner


def test_explorer_score_formula_picks_best():
    """Score = production * (T - ships/planet_prod). Best target has max score."""
    planet_production = 4
    targets = [
        MockTarget(production=1, ships=10),  # T-cost=2.5, score=1*2.0=2.0
        MockTarget(production=5, ships=2),   # T-cost=0.5, score=5*4.0=20.0 <- wins
        MockTarget(production=3, ships=6),   # T-cost=1.5, score=3*3.0=9.0
    ]
    T = sum(t.ships for t in targets) / planet_production  # (10+2+6)/4 = 4.5
    best = max(targets, key=lambda t: t.production * (T - t.ships / planet_production))
    assert best is targets[1], f"Expected targets[1] (prod=5,ships=2) but got prod={best.production}"


def test_explorer_score_formula_prefers_high_prod_low_cost():
    """Given equal ships, prefer higher production."""
    planet_production = 2
    targets = [
        MockTarget(production=3, ships=4),
        MockTarget(production=6, ships=4),  # wins: same cost, higher production
    ]
    T = sum(t.ships for t in targets) / planet_production  # 8/2 = 4.0
    best = max(targets, key=lambda t: t.production * (T - t.ships / planet_production))
    assert best is targets[1]


def test_agent_26_produces_valid_moves():
    """Agent returns a list of [planet_id, angle, ships] triples (or empty list)."""
    mod = load_module("26-Board_Explorer.py", "agent26")
    env = ke.make("orbit_wars", debug=True)
    env.reset(2)
    obs = env.state[0].observation
    moves = mod.agent(obs)
    assert isinstance(moves, list)
    for move in moves:
        assert len(move) == 3, f"Move must be [id, angle, ships], got {move}"
        assert isinstance(move[2], (int, float)) and move[2] > 0, f"Ships must be positive: {move}"


def test_explorer_score_empty_neutral_returns_no_action():
    """_explorer_score_action returns [] when no neutral neighbours exist."""
    mod = load_module("26-Board_Explorer.py", "agent26b")

    class FakePlanet:
        NB_NEARBY_OUT_OF = 5
        owner = 0
        production = 3
        nearby_planets = [
            MockTarget(production=2, ships=5, owner=1),  # enemy, not neutral
        ]

    class FakeBoard:
        _explorer_score_action = mod.Board._explorer_score_action
        def _create_one_action(self, planet, target, ships_needed=None):
            return [[planet.owner, 0.0, 1]]

    result = FakeBoard()._explorer_score_action(FakePlanet())
    assert result == [], f"Expected [] when no neutral planets, got {result}"


def test_explorer_score_zero_production_returns_no_action():
    """_explorer_score_action returns [] when planet.production == 0."""
    mod = load_module("26-Board_Explorer.py", "agent26c")

    class FakePlanet:
        NB_NEARBY_OUT_OF = 5
        owner = 0
        production = 0  # can't compute score
        nearby_planets = [MockTarget(production=2, ships=5, owner=-1)]

    class FakeBoard:
        _explorer_score_action = mod.Board._explorer_score_action
        def _create_one_action(self, planet, target, ships_needed=None):
            return [[0, 0.0, 1]]

    result = FakeBoard()._explorer_score_action(FakePlanet())
    assert result == [], f"Expected [] when production==0, got {result}"
