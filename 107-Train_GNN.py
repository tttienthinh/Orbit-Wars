"""Training loop for OrbitGNN — behavioral cloning from _04_get_selected.

Usage:
    python 107-Train_GNN.py

Requires:
    pip install mlflow kaggle-environments torch torch-geometric
"""
import copy
import json
import random
import types
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import Batch

import importlib.util as _ilu

# ── Load pipeline from 106 ────────────────────────────────────────────────────
_spec = _ilu.spec_from_file_location("m106", "106-Simulate20Next_GNN.py")
m = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(m)
SP = m.StrategyPipeline
OrbitGNN = m.OrbitGNN

AGENTS_DIR = Path("orbit-wars-lab/agents")
LOGS_DIR = Path("62-logs")
BATCH_SIZE = 32
LR = 1e-3
SAVE_EVERY = 5
WEIGHTS_PATH = Path("gnn_weights.pt")


# ── Data helpers ──────────────────────────────────────────────────────────────

def _process_step(obs_dict: dict) -> "tuple | None":
    """Return (HeteroData, y_tensor) for one observation, or None if no attack nodes."""
    obs = types.SimpleNamespace(**copy.deepcopy(obs_dict))
    player_id = getattr(obs, "player", 0)
    env_step = getattr(obs, "step", 0)

    initial = getattr(obs, "initial_planets", [])
    owners = {p[1] for p in initial if p[1] != -1}
    num_agents = 4 if len(owners) > 2 else 2

    df_s, planet_disp = SP._01_get_obs_dataframe(obs, env_step, num_agents)
    df_s = SP._00_remap_owner(df_s, obs, player_id)
    coarse = SP._02_pre_mine(df_s, 0)
    pa = SP._02_get_all_opportunities(coarse, df_s, planet_disp)
    safe_attacks = SP._03_filter_collision(pa)

    attack_df = safe_attacks.query("ships_sent <= ships_min").reset_index(drop=True)
    if attack_df.empty:
        return None

    data = SP._05_get_GNN(df_s, pa, safe_attacks)
    if "attack" not in data.node_types or data["attack"].x.numel() == 0:
        return None

    selected = SP._04_get_selected(safe_attacks, player_id=0)
    key_cols = ["id_src", "id", "step", "ships_sent"]
    selected_keys = set(map(tuple, selected[key_cols].values.tolist()))
    y = torch.tensor(
        [
            1.0 if tuple(r) in selected_keys else 0.0
            for r in attack_df[key_cols].values.tolist()
        ],
        dtype=torch.float32,
    )
    return data, y


def collect_samples(replay: dict) -> list:
    """Extract (HeteroData, y) pairs from every step of a replay dict."""
    samples = []
    for step_states in replay.get("steps", []):
        if not step_states:
            continue
        state0 = step_states[0]
        if not isinstance(state0, dict):
            continue
        if state0.get("status", "ACTIVE") not in ("ACTIVE", None, ""):
            continue
        obs_dict = state0.get("observation")
        if not obs_dict:
            continue
        try:
            result = _process_step(obs_dict)
        except Exception:
            continue
        if result is not None:
            samples.append(result)
    return samples


def sample_opponent() -> Path:
    """Pick a random agent from orbit-wars-lab/agents/ (mine + external, no baselines)."""
    candidates = [
        p.parent
        for p in AGENTS_DIR.rglob("main.py")
        if "baselines" not in str(p)
    ]
    if not candidates:
        raise RuntimeError(f"No agents found under {AGENTS_DIR}")
    return random.choice(candidates)


def run_game(agent_path: Path, opponent_path: Path) -> dict:
    """Run one 1v1 game (agent_path=player0, opponent_path=player1). Return replay dict."""
    from kaggle_environments import make
    env = make("orbit_wars", debug=False)
    env.run([str(agent_path / "main.py"), str(opponent_path / "main.py")])
    return env.toJSON()
