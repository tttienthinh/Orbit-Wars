"""
simulate.py — Run 07-claude_code against all external agents and log results.

Usage:
    python simulate.py [--games N] [--opponent NAME] [--debug]

Runs N 2-player games per opponent (default 10), swapping sides each game.
Prints diagnostics on losses and a final win-rate table.
"""

import argparse
import importlib.util
import os
import sys
import traceback
import warnings
warnings.filterwarnings("ignore")

import kaggle_environments as ke

LAB_DIR = os.path.join(os.path.dirname(__file__), "orbit-wars-lab")
EXTERNAL_DIR = os.path.join(LAB_DIR, "agents", "external")
MY_AGENT_PATH = os.path.join(os.path.dirname(__file__), "07-claude_code.py")


def load_agent(path):
    spec = importlib.util.spec_from_file_location("agent_mod", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.agent


def analyse_steps(steps, my_player, debug=False):
    """Return (my_reward, opp_reward, log_lines)."""
    log = []
    prev_my_count = None

    for i, step in enumerate(steps):
        if not step:
            continue
        obs = step[0].observation
        if not obs:
            continue
        planets = obs.get("planets", [])

        my_count  = sum(1 for p in planets if p[1] == my_player)
        opp_count = sum(1 for p in planets if p[1] == (1 - my_player))
        my_ships  = sum(p[5] for p in planets if p[1] == my_player)
        opp_ships = sum(p[5] for p in planets if p[1] == (1 - my_player))

        fleets   = obs.get("fleets", [])
        inc_enemy = sum(int(f[6]) for f in fleets
                        if f[1] != my_player
                        and any(p[1] == my_player and _fleet_aims_at(f, p) for p in planets))

        lost_planet = (prev_my_count is not None and my_count < prev_my_count)

        if debug or lost_planet or i % 50 == 0 or i < 5:
            marker = " *** LOST PLANET ***" if lost_planet else ""
            log.append(
                f"  t{i:3d}: mine={my_count}p/{my_ships:.0f}s "
                f"opp={opp_count}p/{opp_ships:.0f}s "
                f"enemy_inc={inc_enemy:.0f}s{marker}"
            )

        prev_my_count = my_count

    final = steps[-1]
    if final and len(final) >= 2:
        my_r  = final[my_player].reward or 0
        opp_r = final[1 - my_player].reward or 0
    else:
        my_r, opp_r = 0, 0

    return my_r, opp_r, log


def _fleet_aims_at(fleet, planet):
    """Quick check: does this fleet's angle roughly point at this planet?"""
    import math
    dx = planet[2] - fleet[2]  # planet.x - fleet.x
    dy = planet[3] - fleet[3]
    d = math.sqrt(dx*dx + dy*dy)
    if d < 1e-6:
        return True
    dot = (dx/d) * math.cos(fleet[4]) + (dy/d) * math.sin(fleet[4])
    cross_sq = d*d - (dx*math.cos(fleet[4]) + dy*math.sin(fleet[4]))**2
    return dot > 0 and cross_sq < (planet[4] + 1)**2  # planet[4] = radius


def run_game(my_agent, opp_agent, my_player=0, num_steps=500, debug=False):
    if my_player == 0:
        agents = [my_agent, opp_agent]
    else:
        agents = [opp_agent, my_agent]

    env = ke.make("orbit_wars", debug=False, configuration={"episodeSteps": num_steps})
    try:
        env.run(agents)
    except Exception as e:
        return None, None, [f"CRASH: {e}\n{traceback.format_exc()}"]

    return analyse_steps(env.steps, my_player, debug=debug)


def simulate_vs(my_agent, opp_name, opp_path, n_games=10, debug=False):
    opp_agent = load_agent(opp_path)
    wins = losses = draws = crashes = 0
    print(f"\n{'='*60}")
    print(f"  vs {opp_name}  ({n_games} games)")
    print(f"{'='*60}")

    for g in range(n_games):
        my_player = g % 2
        my_r, opp_r, log = run_game(my_agent, opp_agent, my_player=my_player, debug=debug)

        if my_r is None:
            crashes += 1
            print(f"  Game {g+1:2d} [p{my_player}]: CRASH")
            for l in log[:5]:
                print(l)
            continue

        if my_r > opp_r:
            outcome, wins = "WIN ", wins + 1
        elif my_r < opp_r:
            outcome, losses = "LOSS", losses + 1
        else:
            outcome, draws = "DRAW", draws + 1

        print(f"  Game {g+1:2d} [p{my_player}]: {outcome}  score={my_r:.1f} vs {opp_r:.1f}")
        if outcome in ("LOSS", "WIN") or debug:
            for line in log:
                print(line)

    total = wins + losses + draws + crashes
    wr = wins / total * 100 if total else 0
    print(f"\n  Result: {wins}W {losses}L {draws}D {crashes}crash  winrate={wr:.0f}%")
    return wins, losses, draws, crashes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=10)
    parser.add_argument("--opponent", type=str, default=None)
    parser.add_argument("--debug", action="store_true", help="Print every turn")
    args = parser.parse_args()

    my_agent = load_agent(MY_AGENT_PATH)

    opponents = []
    for name in sorted(os.listdir(EXTERNAL_DIR)):
        path = os.path.join(EXTERNAL_DIR, name, "main.py")
        if os.path.isfile(path):
            opponents.append((name, path))

    if args.opponent:
        opponents = [(n, p) for n, p in opponents if args.opponent in n]
        if not opponents:
            print(f"No opponent matching '{args.opponent}'")
            sys.exit(1)

    summary = []
    for name, path in opponents:
        try:
            w, l, d, c = simulate_vs(my_agent, name, path,
                                     n_games=args.games, debug=args.debug)
            summary.append((name, w, l, d, c))
        except Exception:
            print(f"\nFailed to load {name}:")
            traceback.print_exc()

    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    total_w = total_l = total_d = 0
    for name, w, l, d, c in summary:
        t = w + l + d + c
        wr = w / t * 100 if t else 0
        flag = " <-- WEAK" if wr < 50 else ""
        print(f"  {name:<35s}  {w}W {l}L {d}D  {wr:.0f}%{flag}")
        total_w += w; total_l += l; total_d += d
    total = total_w + total_l + total_d
    if total:
        print(f"\n  Overall: {total_w}W {total_l}L {total_d}D  {total_w/total*100:.0f}% winrate")


if __name__ == "__main__":
    main()
