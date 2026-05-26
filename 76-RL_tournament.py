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
        with open(path) as f:
            raw = json.load(f)
        configs[agent_id] = {k: v for k, v in raw.items() if not k.startswith("_")}
        # preserve metadata for results.json
        configs[agent_id]["__meta__"] = {k: v for k, v in raw.items() if k.startswith("_")}
    return configs
