"""
76-RL_tournament.py
Run all unresolved tournament folders under 76-RL_tournament/.
Each folder must contain exactly 4 agent config JSON files (001–004).
Writes results.json and prints a leaderboard summary per folder.
"""
import importlib.util
import json
import math
import random
import sys
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

from kaggle_environments import make as _kenv_make

TOURNAMENT_DIR = Path(__file__).parent / "76-RL_tournament"

SCORING_CONSTANTS = [
    "PROD_MULT", "TIME_PROD_MULT", "ENEMY_MULT", "COMPOUND_MULT",
    "MINE_NEAR_TGT_MULT", "ENEMY_NEAR_TGT_MULT", "PROD_SRC_MULT",
    "ORBIT_BONUS", "PROXIMITY_MULT", "DIST_MULT", "SHIPS_MULT",
    "ETA_MULT", "OVEREXTEND_MULT", "PROXIMITY_DIST",
]


def _load_main_module():
    main_path = TOURNAMENT_DIR / "main.py"
    if not main_path.exists():
        raise FileNotFoundError(f"Missing {main_path} — cannot run tournament.")
    spec = importlib.util.spec_from_file_location("_orbit_main", str(main_path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_mod = _load_main_module()


def make_agent(cfg: dict):
    """Return a callable that context-switches module constants before each call."""
    def _agent(obs):
        # inject scoring constants into the shared module
        for key in SCORING_CONSTANTS:
            if key in cfg:
                setattr(_mod, key, float(cfg[key]))
        return _mod.nearest_planet_sniper(obs)
    return _agent


def load_configs(folder: Path) -> dict:
    """
    Returns {agent_id: cfg_dict} for all *.json files in folder,
    sorted by filename. Raises ValueError if count != 4.
    """
    json_files = sorted(folder.glob("*.json"))
    if len(json_files) != 4:
        raise ValueError(
            f"{folder.name}: expected 4 config files, found {len(json_files)}"
        )
    configs = {}
    for path in json_files:
        agent_id = path.stem  # e.g. "001"
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
        configs[agent_id] = {k: v for k, v in raw.items() if not k.startswith("_")}
        # preserve metadata for results.json
        configs[agent_id]["__meta__"] = {k: v for k, v in raw.items() if k.startswith("_")}
    return configs


def _init_accum(player_ids: list) -> dict:
    """Create a blank accumulator dict for each player."""
    return {
        pid: {
            "planets_owned_sum": 0,
            "planets_owned_steps": 0,
            "peak_planets": 0,
            "ships_sent_total": 0,
            "neutral_captures": 0,
            "enemy_flips": 0,
            "planets_lost": 0,
            "first_action_step": None,
        }
        for pid in player_ids
    }


def _planets_snapshot(states: list) -> dict:
    """Return {planet_id: owner} from the first non-None agent state."""
    for s in states:
        if s is None:
            continue
        obs = s.observation
        planets = (
            obs.planets if hasattr(obs, "planets") else obs.get("planets", [])
        )
        return {p[0]: p[1] for p in planets}
    return {}


def _fleet_ids_snapshot(states: list) -> set:
    """Return the set of all current fleet IDs from the first valid state."""
    for s in states:
        if s is None:
            continue
        obs = s.observation
        fleets = (
            obs.fleets if hasattr(obs, "fleets") else obs.get("fleets", [])
        )
        return {f[0] for f in fleets}
    return set()


def _update_step_stats(accum, states, prev_planets, prev_fleet_ids, player_ids, step):
    """Update per-player accumulators for one completed step."""
    for i, pid in enumerate(player_ids):
        s = states[i]
        if s is None:
            continue
        obs = s.observation
        planets = (
            obs.planets if hasattr(obs, "planets") else obs.get("planets", [])
        )
        fleets = (
            obs.fleets if hasattr(obs, "fleets") else obs.get("fleets", [])
        )
        a = accum[pid]

        # planets owned this step
        owned = sum(1 for p in planets if p[1] == i)  # owner index == player index
        a["planets_owned_sum"] += owned
        a["planets_owned_steps"] += 1
        a["peak_planets"] = max(a["peak_planets"], owned)

        # ownership transitions vs prev step
        current_map = {p[0]: p[1] for p in planets}
        for planet_id, prev_owner in prev_planets.items():
            curr_owner = current_map.get(planet_id, prev_owner)
            if curr_owner == i and prev_owner == -1:
                a["neutral_captures"] += 1
            elif curr_owner == i and prev_owner != i and prev_owner != -1:
                a["enemy_flips"] += 1
            elif prev_owner == i and curr_owner != i:
                a["planets_lost"] += 1

        # ships sent: count ships in new fleet IDs
        current_fleet_ids = {f[0] for f in fleets}
        new_ids = current_fleet_ids - prev_fleet_ids
        for f in fleets:
            if f[0] in new_ids and f[1] == i:  # owner == player index
                a["ships_sent_total"] += f[6]  # Fleet.ships at index 6
                if a["first_action_step"] is None:
                    a["first_action_step"] = step


def _finalize_stats(accum, states, player_ids) -> dict:
    """Compute final_score and per-player average stats at game end."""
    result = {}
    for i, pid in enumerate(player_ids):
        a = accum[pid]
        s = states[i]
        final_score = 0
        if s is not None:
            obs = s.observation
            planets = (
                obs.planets if hasattr(obs, "planets") else obs.get("planets", [])
            )
            fleets = (
                obs.fleets if hasattr(obs, "fleets") else obs.get("fleets", [])
            )
            final_score = (
                sum(p[5] for p in planets if p[1] == i)  # ships on owned planets
                + sum(f[6] for f in fleets if f[1] == i)  # ships in owned fleets
            )
        steps = max(a["planets_owned_steps"], 1)
        result[pid] = {
            "planets_owned_avg": round(a["planets_owned_sum"] / steps, 2),
            "peak_planets": a["peak_planets"],
            "ships_sent_total": a["ships_sent_total"],
            "neutral_captures": a["neutral_captures"],
            "enemy_flips": a["enemy_flips"],
            "planets_lost": a["planets_lost"],
            "first_action_step": a["first_action_step"],
            "final_score": final_score,
        }
    return result


def _extract_obs(state):
    """Return the observation object from a single agent's state entry."""
    if state is None:
        return None
    return state.observation


def run_game(match_id: str, agents: list, player_ids: list, seed: int) -> dict:
    """
    Run one game and return a match result dict.
    agents: list of callables (same length as player_ids)
    player_ids: e.g. ["001", "002"] or ["001","002","003","004"]
    """
    N = len(agents)

    try:
        # reset per-player state in the shared module before each game
        _mod._states.clear()
        env = _kenv_make("orbit_wars", configuration={"seed": seed}, debug=False)
        env.reset(num_agents=N)
        states = env.step([[] for _ in range(N)])

        accum = _init_accum(player_ids)
        step = 0

        while True:
            statuses = [
                (s.status if s is not None else "DONE") for s in states
            ]
            if not any(st == "ACTIVE" for st in statuses):
                break

            prev_planets = _planets_snapshot(states)
            prev_fleet_ids = _fleet_ids_snapshot(states)

            actions = []
            for i, agent in enumerate(agents):
                obs = _extract_obs(states[i])
                try:
                    action = agent(obs) if obs is not None else []
                except Exception:
                    action = []
                actions.append(action)

            states = env.step(actions)
            step += 1
            _update_step_stats(accum, states, prev_planets, prev_fleet_ids, player_ids, step)

        # determine winner by final score
        per_player = _finalize_stats(accum, states, player_ids)
        scores = {pid: per_player[pid]["final_score"] for pid in player_ids}
        max_score = max(scores.values())
        winners = [pid for pid, sc in scores.items() if sc == max_score]
        winner = winners[0] if len(winners) == 1 else None  # None = draw

        return {
            "match_id": match_id,
            "format": f"{N}p",
            "players": player_ids,
            "seed": seed,
            "status": "ok",
            "winner": winner,
            "stats": {
                "win_turn": step,
                "per_player": per_player,
            },
        }

    except Exception as exc:
        return {
            "match_id": match_id,
            "format": f"{N}p",
            "players": player_ids,
            "seed": seed,
            "status": "crashed",
            "error": str(exc),
            "winner": None,
            "stats": {
                "win_turn": 0,
                "per_player": {
                    pid: {
                        "planets_owned_avg": 0.0,
                        "peak_planets": 0,
                        "ships_sent_total": 0,
                        "neutral_captures": 0,
                        "enemy_flips": 0,
                        "planets_lost": 0,
                        "first_action_step": None,
                        "final_score": 0,
                    }
                    for pid in player_ids
                },
            },
        }


def _print_game_result(result: dict):
    pid = result["players"]
    label = " v ".join(pid)
    mid = result["match_id"]
    seed = result["seed"]
    winner = result["winner"] or "draw"
    turn = result["stats"]["win_turn"]
    scores_str = "  ".join(
        f"{p}={result['stats']['per_player'][p]['final_score']}"
        for p in pid
    )
    status = "" if result["status"] == "ok" else "  [CRASHED]"
    print(f"[{label} {mid.split('_')[-1]}] seed={seed}  {winner} wins  turn={turn}  scores: {scores_str}{status}")


def build_schedule(agent_ids: list, root_seed: int) -> list:
    """
    Returns a list of 24 match dicts:
      {"match_id": str, "player_ids": list, "seed": int}
    18 × 2p (C(4,2)=6 pairs × 3 games) then 6 × 4p.
    Seeds are derived deterministically from root_seed.
    """
    rng = random.Random(root_seed)
    game_seeds = [rng.randint(0, 2**31) for _ in range(24)]

    schedule = []
    seed_idx = 0

    # 2-player: all C(4,2) pairs × 3 games
    for a, b in combinations(agent_ids, 2):
        for _ in range(3):
            match_id = f"2p_{a}v{b}_g{seed_idx}"
            schedule.append({
                "match_id": match_id,
                "player_ids": [a, b],
                "seed": game_seeds[seed_idx],
            })
            seed_idx += 1

    # 4-player: 6 fixed orderings (Agent1 always at position 0)
    a1, a2, a3, a4 = agent_ids
    orderings = [
        [a1, a2, a3, a4],
        [a1, a2, a4, a3],
        [a1, a3, a2, a4],
        [a1, a3, a4, a2],
        [a1, a4, a2, a3],
        [a1, a4, a3, a2],
    ]
    for i, order in enumerate(orderings):
        match_id = f"4p_{i}"
        schedule.append({
            "match_id": match_id,
            "player_ids": order,
            "seed": game_seeds[seed_idx],
        })
        seed_idx += 1

    assert seed_idx == 24
    return schedule


def compute_summary(matches: list, agent_ids: list) -> dict:
    summary_2p = {pid: {"wins": 0, "losses": 0, "draws": 0} for pid in agent_ids}
    summary_4p = {pid: {"wins": 0, "top2": 0, "losses": 0} for pid in agent_ids}

    for m in matches:
        if m["format"] == "2p":
            a, b = m["players"]
            if m["winner"] is None:  # draw
                summary_2p[a]["draws"] += 1
                summary_2p[b]["draws"] += 1
            else:
                winner, loser = (a, b) if m["winner"] == a else (b, a)
                summary_2p[winner]["wins"] += 1
                summary_2p[loser]["losses"] += 1

        elif m["format"] == "4p":
            players = m["players"]
            # rank by final_score descending
            scores = {
                pid: m["stats"]["per_player"][pid]["final_score"]
                for pid in players
            }
            ranked = sorted(players, key=lambda p: scores[p], reverse=True)
            for rank, pid in enumerate(ranked):
                if rank == 0:
                    summary_4p[pid]["wins"] += 1
                    summary_4p[pid]["top2"] += 1
                elif rank == 1:
                    summary_4p[pid]["top2"] += 1
                else:
                    summary_4p[pid]["losses"] += 1

    return {"2p": summary_2p, "4p": summary_4p}


def print_leaderboard(folder_name: str, root_seed: int, summary: dict, configs: dict):
    W = 42
    bar = "═" * W
    print(f"\n{bar}")
    print(f"  LEADERBOARD — {folder_name}  (seed={root_seed})")
    print(bar)

    # 2-player ranking by win%
    print("  2-PLAYER")
    print(f"  {'Rank':<5} {'Agent':<18} {'W':>3} {'L':>3} {'D':>3} {'Win%':>5}")
    s2 = summary["2p"]
    ranked_2p = sorted(
        s2.keys(),
        key=lambda p: (s2[p]["wins"] / max(s2[p]["wins"] + s2[p]["losses"] + s2[p]["draws"], 1)),
        reverse=True,
    )
    for rank, pid in enumerate(ranked_2p, 1):
        name = configs[pid].get("__meta__", {}).get("_name", pid)
        label = f"{pid}-{name}"[:16]
        w, l, d = s2[pid]["wins"], s2[pid]["losses"], s2[pid]["draws"]
        total = w + l + d
        pct = f"{round(100 * w / total)}%" if total > 0 else "—"
        print(f"  {rank:<5} {label:<18} {w:>3} {l:>3} {d:>3} {pct:>5}")

    # 4-player ranking by wins then top2
    print()
    print("  4-PLAYER")
    print(f"  {'Rank':<5} {'Agent':<18} {'Wins':>5} {'Top2':>5} {'Losses':>7}")
    s4 = summary["4p"]
    ranked_4p = sorted(
        s4.keys(),
        key=lambda p: (s4[p]["wins"], s4[p]["top2"]),
        reverse=True,
    )
    for rank, pid in enumerate(ranked_4p, 1):
        name = configs[pid].get("__meta__", {}).get("_name", pid)
        label = f"{pid}-{name}"[:16]
        w, t2, l = s4[pid]["wins"], s4[pid]["top2"], s4[pid]["losses"]
        print(f"  {rank:<5} {label:<18} {w:>5} {t2:>5} {l:>7}")

    print(bar)


def run_tournament(folder: Path):
    print(f"\n{'='*50}")
    print(f"  Tournament: {folder.name}")
    print(f"{'='*50}")

    configs = load_configs(folder)
    agent_ids = list(configs.keys())  # already sorted by glob

    root_seed = random.randint(0, 100)
    print(f"  root_seed = {root_seed}")

    # build agents: inject __meta__ into the cfg passed to make_agent
    agents_by_id = {}
    for pid, cfg in configs.items():
        clean_cfg = {k: v for k, v in cfg.items() if k != "__meta__"}
        agents_by_id[pid] = make_agent(clean_cfg)

    schedule = build_schedule(agent_ids, root_seed)

    matches = []
    for entry in schedule:
        match_agents = [agents_by_id[pid] for pid in entry["player_ids"]]
        result = run_game(entry["match_id"], match_agents, entry["player_ids"], entry["seed"])
        _print_game_result(result)
        matches.append(result)

    summary = compute_summary(matches, agent_ids)

    # build results.json
    agents_section = {}
    for pid, cfg in configs.items():
        meta = cfg.get("__meta__", {})
        entry = dict(meta)
        entry.update({k: v for k, v in cfg.items() if k != "__meta__"})
        agents_section[pid] = entry

    results = {
        "root_seed": root_seed,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "agents": agents_section,
        "matches": matches,
        "summary": summary,
    }

    out_path = folder / "results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved → {out_path}")

    print_leaderboard(folder.name, root_seed, summary, configs)


def main():
    if not TOURNAMENT_DIR.exists():
        raise FileNotFoundError(f"Tournament directory not found: {TOURNAMENT_DIR}")

    folders = sorted(
        d for d in TOURNAMENT_DIR.iterdir()
        if d.is_dir() and not (d / "results.json").exists()
    )

    if not folders:
        print("No unresolved tournament folders found.")
        return

    print(f"Found {len(folders)} unresolved folder(s): {[f.name for f in folders]}")

    for folder in folders:
        try:
            run_tournament(folder)
        except ValueError as e:
            print(f"  SKIP {folder.name}: {e}")


if __name__ == "__main__":
    main()
