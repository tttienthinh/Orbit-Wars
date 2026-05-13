"""
18-Agent_ML.py — Test MLP imitation-learning agent vs random.

Usage:
    python 18-Agent_ML.py [--games N]
"""

import argparse
import importlib.util
import os
import sys
import warnings

warnings.filterwarnings("ignore")

import kaggle_environments as ke
from kaggle_environments.envs.orbit_wars.orbit_wars import random_agent


def _load_play_module():
    spec = importlib.util.spec_from_file_location(
        "_play_ml", os.path.join(os.path.dirname(os.path.abspath(__file__)), "18-Play_ML.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run_game(my_agent_fn, opp_agent_fn, my_player=0, num_steps=500):
    agents = [my_agent_fn, opp_agent_fn] if my_player == 0 else [opp_agent_fn, my_agent_fn]
    env = ke.make("orbit_wars", debug=False, configuration={"episodeSteps": num_steps})
    try:
        env.run(agents)
    except Exception as e:
        return None, None, str(e)

    final = env.steps[-1]
    if final and len(final) >= 2:
        my_r  = final[my_player].reward or 0
        opp_r = final[1 - my_player].reward or 0
    else:
        my_r, opp_r = 0, 0
    return my_r, opp_r, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=4, help="Number of games (default 4)")
    args = parser.parse_args()

    print("Loading MLP agent...")
    play_mod = _load_play_module()
    mlp_agent = play_mod.MLPAgent()
    print(f"  input_dim : {mlp_agent.input_dim}")

    _mlp_instance = mlp_agent

    def agent_fn(obs):
        obs_dict = obs if isinstance(obs, dict) else dict(obs)
        raw = _mlp_instance.predict_actions(obs_dict)
        return [[a[0], a[1], a[2]] for a in raw]

    wins = losses = draws = crashes = 0
    print(f"\nRunning {args.games} games vs random agent...\n")

    for g in range(args.games):
        my_player = g % 2
        my_r, opp_r, err = run_game(agent_fn, random_agent, my_player=my_player)

        if err is not None:
            crashes += 1
            print(f"  Game {g+1:2d} [p{my_player}]: CRASH — {err}")
            continue

        if my_r > opp_r:
            outcome, wins = "WIN ", wins + 1
        elif my_r < opp_r:
            outcome, losses = "LOSS", losses + 1
        else:
            outcome, draws = "DRAW", draws + 1

        print(f"  Game {g+1:2d} [p{my_player}]: {outcome}  score={my_r:.1f} vs {opp_r:.1f}")

    total = wins + losses + draws + crashes
    wr = wins / total * 100 if total else 0
    print(f"\nResult: {wins}W {losses}L {draws}D {crashes}crash  winrate={wr:.0f}%")


if __name__ == "__main__":
    main()
