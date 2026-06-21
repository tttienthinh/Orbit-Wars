"""Direct output comparison: replay captured obs to both agents, verify identical moves.

Both agents must produce exactly the same moves when fed the same observation
sequence with the same internal step counter. This avoids the game-level
symmetry trap (identical agents still diverge when the board evolves).
"""
import sys
import importlib.util
from pathlib import Path

ROOT = Path(__file__).parent
AGENT_92_PY = ROOT / "orbit-wars-lab/agents/mine/92-90-Polars/main.py"
AGENT_132_PY = ROOT / "orbit-wars-lab/agents/mine/132-Polars_speed/main.py"


def load_agent_fresh(py_path: Path, unique_name: str):
    """Load agent module into a clean namespace so globals start at zero."""
    spec = importlib.util.spec_from_file_location(unique_name, py_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[unique_name] = mod
    spec.loader.exec_module(mod)
    return mod.agent


def check(n_steps: int = 20) -> bool:
    from kaggle_environments import make

    # Step 1: run a real game and capture every obs player 0 receives.
    a92_game = load_agent_fresh(AGENT_92_PY, "_a92_game")
    captured_obs: list = []

    def recording_proxy(obs):
        captured_obs.append(obs)
        return a92_game(obs)

    env = make("orbit_wars", debug=False)
    env.run([recording_proxy, lambda obs: []])  # 92 plays p0; p1 idles

    print(f"Captured {len(captured_obs)} observations. Comparing outputs…")

    # Step 2: replay the same obs to fresh instances of both agents.
    # Both start with step-counter=0 and advance identically.
    a92_rep = load_agent_fresh(AGENT_92_PY, "_a92_rep")
    a132_rep = load_agent_fresh(AGENT_132_PY, "_a132_rep")

    mismatches = []
    for step_idx, obs in enumerate(captured_obs[:n_steps]):
        m92 = a92_rep(obs)
        m132 = a132_rep(obs)
        if m92 != m132:
            mismatches.append((step_idx, m92, m132))

    n_tested = min(len(captured_obs), n_steps)
    if mismatches:
        print(f"FAIL: {len(mismatches)} mismatches in {n_tested} steps")
        for i, m92, m132 in mismatches[:5]:
            print(f"  step {i}:")
            print(f"    92:  {m92}")
            print(f"    132: {m132}")
        return False

    print(f"PASS: identical moves over {n_tested} steps")
    return True


if __name__ == "__main__":
    sys.exit(0 if check() else 1)
