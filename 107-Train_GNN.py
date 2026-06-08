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

    if safe_attacks.empty or "ships_sent" not in safe_attacks.columns:
        return None
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
        except Exception as e:
            print(f"  step skipped: {e}")
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
    """Run one 1v1 game in a subprocess to avoid PyTorch/PyG double-init segfault.

    Writes the replay to a temp file to avoid stdout pollution from agent prints.
    agent_path may be a .py file or a directory containing main.py.
    """
    import subprocess, tempfile, os
    a0 = str(agent_path) if str(agent_path).endswith(".py") else str(agent_path / "main.py")
    a1 = str(opponent_path / "main.py")
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
    tmp.close()
    tmp_path = tmp.name
    code = (
        "import json; from kaggle_environments import make; "
        f"env = make('orbit_wars', debug=False); "
        f"env.run([{a0!r}, {a1!r}]); "
        f"open({tmp_path!r}, 'w').write(json.dumps(env.toJSON()))"
    )
    result = subprocess.run(
        ["python", "-c", code],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        timeout=600,
    )
    try:
        if result.returncode != 0:
            raise RuntimeError(f"Game subprocess exited {result.returncode}")
        with open(tmp_path) as fh:
            return json.load(fh)
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


# ── Training utilities ────────────────────────────────────────────────────────

def compute_pos_weight(batch_ys: list) -> torch.Tensor:
    """BCEWithLogitsLoss pos_weight: ratio of negatives to positives per batch."""
    y_cat = torch.cat(batch_ys)
    n_pos = y_cat.sum().item()
    n_neg = len(y_cat) - n_pos
    if n_pos == 0:
        return torch.tensor(1.0)
    return torch.tensor(n_neg / n_pos)


def train_step(
    model: "OrbitGNN",
    optimizer: "torch.optim.Optimizer",
    batch_graphs: list,
    batch_ys: list,
) -> "tuple[float, float]":
    """One gradient update. Returns (loss, accuracy)."""
    model.train()
    data_batch = Batch.from_data_list(batch_graphs)
    y = torch.cat(batch_ys)
    pos_weight = compute_pos_weight(batch_ys)

    logits = model(data_batch)
    loss = F.binary_cross_entropy_with_logits(logits, y, pos_weight=pos_weight)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    with torch.no_grad():
        preds = (logits.sigmoid() > 0.5)
        acc = preds.eq(y.bool()).float().mean().item()

    return loss.item(), acc


# ── Main training loop ────────────────────────────────────────────────────────

def main() -> None:
    import mlflow

    model = OrbitGNN(hidden_dim=16)
    if WEIGHTS_PATH.exists():
        model.load_state_dict(torch.load(str(WEIGHTS_PATH), map_location="cpu", weights_only=True))
        print(f"Loaded weights from {WEIGHTS_PATH}")
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    agent_106_path = Path("106-Simulate20Next_GNN.py")

    # Bootstrap from existing logs
    bootstrap_samples: list = []
    for log_file in sorted(LOGS_DIR.glob("*.json")):
        try:
            replay = json.loads(log_file.read_text(encoding="utf-8"))
            bootstrap_samples.extend(collect_samples(replay))
        except Exception as e:
            print(f"  skip {log_file.name}: {e}")
    print(f"Bootstrap: {len(bootstrap_samples)} samples from {LOGS_DIR}")

    buffer_graphs: list = [s[0] for s in bootstrap_samples]
    buffer_ys: list = [s[1] for s in bootstrap_samples]
    game_idx = 0
    update_idx = 0

    with mlflow.start_run():
        mlflow.log_params({
            "hidden_dim": 16,
            "lr": LR,
            "batch_size": BATCH_SIZE,
            "save_every": SAVE_EVERY,
        })

        while True:
            # ── Run one game ──────────────────────────────────────────────────
            opponent_path = sample_opponent()
            try:
                replay = run_game(agent_106_path, opponent_path)
            except Exception as e:
                print(f"Game {game_idx} failed ({opponent_path.name}): {e}")
                game_idx += 1
                continue

            # Win rate (player 0 = our GNN agent)
            rewards = replay.get("rewards", [])
            winner_is_gnn = (len(rewards) > 0 and rewards[0] == 1)
            mlflow.log_metrics(
                {"win": int(winner_is_gnn)},
                step=game_idx,
            )
            print(
                f"Game {game_idx} vs {opponent_path.name}: "
                f"{'WIN' if winner_is_gnn else 'LOSS'}"
            )

            # ── Collect samples from the just-played game ─────────────────────
            new_samples = collect_samples(replay)
            buffer_graphs.extend(s[0] for s in new_samples)
            buffer_ys.extend(s[1] for s in new_samples)

            # ── Batch updates while buffer has enough samples ─────────────────
            while len(buffer_graphs) >= BATCH_SIZE:
                batch_g = buffer_graphs[:BATCH_SIZE]
                batch_y = buffer_ys[:BATCH_SIZE]
                buffer_graphs = buffer_graphs[BATCH_SIZE:]
                buffer_ys = buffer_ys[BATCH_SIZE:]

                loss, acc = train_step(model, optimizer, batch_g, batch_y)
                mlflow.log_metrics(
                    {"loss": loss, "accuracy": acc},
                    step=update_idx,
                )
                print(f"  update {update_idx}: loss={loss:.4f} acc={acc:.3f}")
                update_idx += 1

            # ── Save weights every SAVE_EVERY games ───────────────────────────
            game_idx += 1
            if game_idx % SAVE_EVERY == 0:
                torch.save(model.state_dict(), str(WEIGHTS_PATH))
                print(f"Saved weights -> {WEIGHTS_PATH}  (game {game_idx})")


if __name__ == "__main__":
    main()
