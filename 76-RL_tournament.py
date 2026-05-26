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
from pathlib import Path

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
