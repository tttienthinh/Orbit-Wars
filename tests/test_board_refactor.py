# tests/test_board_refactor.py
import sys
import os
import importlib.util

import kaggle_environments as ke

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_agent(filename, module_name):
    path = os.path.join(ROOT, filename)
    spec = importlib.util.spec_from_file_location(module_name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod.agent


def get_test_obs():
    env = ke.make("orbit_wars", debug=True)
    env.reset(2)
    return env.state[0].observation


def test_board_refactor_equivalence():
    agent_24 = load_agent("24-Rules_Target_big_prod.py", "agent24")
    agent_25 = load_agent("25-Board_refactor.py", "agent25")
    obs = get_test_obs()
    moves_24 = agent_24(obs)
    moves_25 = agent_25(obs)
    assert moves_24 == moves_25, f"\n24 moves: {moves_24}\n25 moves: {moves_25}"
