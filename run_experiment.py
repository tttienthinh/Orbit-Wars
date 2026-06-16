"""Parameterized GNN experiment runner for autoresearch.

Each run loads from a checkpoint, trains for N epochs with the specified
optimizer/scheduler, saves models + metrics to an experiment subdirectory.

Usage:
    python run_experiment.py \\
        --experiment-name 001-CosinusAnnealing \\
        --resume-from 117-NewGNN/model_epoch82.pt \\
        --scheduler cosine \\
        --lr 1e-3 \\
        --optimizer adam \\
        --n-epochs 10
"""
import argparse
import math
import random
import shutil
from pathlib import Path

import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch_geometric.data import HeteroData
from torch_geometric.nn import GATConv, HeteroConv, SAGEConv
from torch_geometric.transforms import ToUndirected

PRECOMPUTE_DIR     = Path("114-precompute")
BASE_OUT_DIR       = Path("119-NewGNN_autoresearch")
NB_STEPS_SIM       = 20
EPISODES_PER_EPOCH = 8
TRAIN_RATIO        = 0.8


def build_graph(df_s_t, reach_t):
    if reach_t.is_empty():
        raise ValueError("reach_t must be non-empty")
    game_step = int(reach_t["step_src"][0])

    planet_rows = df_s_t.sort("step").unique(subset=["id"], keep="first").sort("id")
    planet_ids  = planet_rows["id"].to_list()
    planet_idx  = {pid: i for i, pid in enumerate(planet_ids)}

    nature_arr  = planet_rows["nature"].to_numpy()
    planet_feat = np.column_stack([
        (nature_arr == "fix").astype(np.float32),
        (nature_arr == "moving").astype(np.float32),
        (nature_arr == "comet").astype(np.float32),
        planet_rows["production"].to_numpy().astype(np.float32) / 5.0,
    ])

    ps_idx_df = df_s_t.with_row_index("ps_idx").select(["id", "step", "ps_idx"])
    owner_arr = df_s_t["owner"].to_numpy()
    step_arr  = df_s_t["step"].to_numpy().astype(np.float32)
    ps_feat   = np.column_stack([
        (step_arr - game_step) / NB_STEPS_SIM,
        df_s_t["x"].to_numpy().astype(np.float32) / 100.0,
        df_s_t["y"].to_numpy().astype(np.float32) / 100.0,
        np.log(np.clip(df_s_t["ships"].to_numpy().astype(np.float32), 1, None)) / np.log(1024),
        (owner_arr == -1).astype(np.float32),
        (owner_arr == 0).astype(np.float32),
        (owner_arr == 1).astype(np.float32),
        (owner_arr == 2).astype(np.float32),
        (owner_arr == 3).astype(np.float32),
    ])

    pid_to_pidx = np.zeros(max(planet_ids) + 1, dtype=np.int64)
    for pid, idx in planet_idx.items():
        pid_to_pidx[pid] = idx
    snap_src = pid_to_pidx[df_s_t["id"].to_numpy()]
    snap_dst = np.arange(len(df_s_t), dtype=np.int64)

    reach_joined = (
        reach_t
        .join(ps_idx_df.rename({"id": "id_src", "step": "step_src", "ps_idx": "src_ps_idx"}),
              on=["id_src", "step_src"], how="inner")
        .join(ps_idx_df.rename({"ps_idx": "dst_ps_idx"}), on=["id", "step"], how="inner")
    )

    data = HeteroData()
    data["planet"].x      = torch.tensor(planet_feat, dtype=torch.float32)
    data["planet_step"].x = torch.tensor(ps_feat,     dtype=torch.float32)
    data["planet", "has_snapshot", "planet_step"].edge_index = torch.tensor(
        np.stack([snap_src, snap_dst]), dtype=torch.long
    )
    if not reach_joined.is_empty():
        ships = reach_joined["ships_sent"].to_numpy().astype(np.float32)
        data["planet_step", "reaches", "planet_step"].edge_index = torch.tensor(
            np.stack([reach_joined["src_ps_idx"].to_numpy().astype(np.int64),
                      reach_joined["dst_ps_idx"].to_numpy().astype(np.int64)]),
            dtype=torch.long,
        )
        data["planet_step", "reaches", "planet_step"].edge_attr = torch.tensor(
            np.log2(np.clip(ships, 1, None)) / 10.0, dtype=torch.float32
        ).unsqueeze(1)

    return ToUndirected()(data), planet_idx


_REACHES_KEY = ("planet_step", "reaches", "planet_step")


class OrbitGNN(nn.Module):
    def __init__(self, hidden_dim=64, num_layers=3):
        super().__init__()
        H = hidden_dim
        self.hidden_dim = H
        self.planet_proj      = nn.Linear(4, H, bias=False)
        self.planet_step_proj = nn.Linear(9, H, bias=False)
        self.convs = nn.ModuleList([
            HeteroConv({
                ("planet",      "has_snapshot",     "planet_step"): SAGEConv((H, H), H),
                ("planet_step", "rev_has_snapshot", "planet"):      SAGEConv((H, H), H),
                ("planet_step", "reaches",          "planet_step"): GATConv(
                    (H, H), H, edge_dim=1, heads=1, add_self_loops=False
                ),
                ("planet_step", "rev_reaches",      "planet_step"): SAGEConv((H, H), H),
            }, aggr="sum")
            for _ in range(num_layers)
        ])
        self.pair_mlp = nn.Sequential(
            nn.Linear(2 * H, H), nn.ReLU(),
            nn.Linear(H, H // 2), nn.ReLU(),
            nn.Linear(H // 2, 1),
        )

    def encode(self, data):
        x_dict = {
            "planet":      self.planet_proj(data["planet"].x),
            "planet_step": self.planet_step_proj(data["planet_step"].x),
        }
        edge_index_dict = data.edge_index_dict
        edge_attr_map = {}
        if _REACHES_KEY in edge_index_dict:
            try:
                attr = data[_REACHES_KEY].edge_attr
                if attr is not None:
                    edge_attr_map[_REACHES_KEY] = attr
            except AttributeError:
                pass
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict, edge_attr_dict=edge_attr_map)
            x_dict = {k: F.relu(v) for k, v in x_dict.items()}
        return x_dict["planet"]

    def score_pairs(self, h_planet, src_idx, tgt_idx):
        return self.pair_mlp(
            torch.cat([h_planet[src_idx], h_planet[tgt_idx]], dim=-1)
        ).squeeze(-1)


def build_attack_pairs(df_s_at_t, actions_set, planet_idx, game_step):
    rows = df_s_at_t.select(["id", "owner"]).to_dicts()
    src_planets = [r["id"] for r in rows if r["owner"] == 0 and r["id"] in planet_idx]
    all_planets = [r["id"] for r in rows if r["id"] in planet_idx]
    src_list, tgt_list, label_list = [], [], []
    for id_src in src_planets:
        for id_tgt in all_planets:
            if id_src == id_tgt:
                continue
            src_list.append(planet_idx[id_src])
            tgt_list.append(planet_idx[id_tgt])
            label_list.append(1.0 if (game_step, id_src, id_tgt) in actions_set else 0.0)
    if not src_list:
        empty = torch.zeros(0, dtype=torch.long)
        return empty, empty, torch.zeros(0, dtype=torch.float32)
    return (
        torch.tensor(src_list,   dtype=torch.long),
        torch.tensor(tgt_list,   dtype=torch.long),
        torch.tensor(label_list, dtype=torch.float32),
    )


def train_episode(ep_dir, model):
    df_s    = pl.read_parquet(ep_dir / "df_s.parquet")
    reach   = pl.read_parquet(ep_dir / "reach.parquet")
    actions = pl.read_parquet(ep_dir / "actions.parquet")
    reach = (
        reach
        .group_by(["id_src", "step_src", "id", "ships_sent"])
        .agg(pl.all().sort_by("step").first())
    )
    actions_set = set(zip(
        actions["game_step"].to_list(),
        actions["id_src"].to_list(),
        actions["id"].to_list(),
    ))
    all_scores, all_labels = [], []
    total_loss, n_steps = 0.0, 0
    model.train()
    for t in sorted(reach["step_src"].unique().to_list()):
        reach_t       = reach.filter(pl.col("step_src") == t)
        arrival_steps = reach_t["step"].unique().to_list()
        df_s_t        = df_s.filter(pl.col("step").is_in([t] + arrival_steps))
        df_s_at_t     = df_s.filter(pl.col("step") == t)
        if df_s_t.is_empty() or df_s_at_t.is_empty():
            continue
        data, planet_idx = build_graph(df_s_t, reach_t)
        src_idx, tgt_idx, labels = build_attack_pairs(df_s_at_t, actions_set, planet_idx, t)
        if len(src_idx) == 0:
            continue
        h_planet = model.encode(data)
        logits   = model.score_pairs(h_planet, src_idx, tgt_idx)
        n_pos      = labels.sum().item()
        n_neg      = len(labels) - n_pos
        pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32)
        loss       = F.binary_cross_entropy_with_logits(logits, labels, pos_weight=pos_weight)
        loss.backward()
        total_loss += loss.item()
        n_steps    += 1
        with torch.no_grad():
            all_scores.extend(torch.sigmoid(logits).tolist())
            all_labels.extend(labels.tolist())
    return total_loss / max(n_steps, 1), all_scores, all_labels


def _log(msg, log_fh=None):
    print(msg, flush=True)
    if log_fh is not None:
        log_fh.write(msg + "\n")
        log_fh.flush()


def make_optimizer(model, args):
    if args.optimizer == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == "sgd":
        return torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    else:
        return torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)


def make_scheduler(optimizer, args, n_ep):
    if args.scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.n_epochs, eta_min=1e-6)
    elif args.scheduler == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=max(args.n_epochs // 3, 1), gamma=0.5)
    elif args.scheduler == "onecycle":
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=args.lr * 10,
            epochs=args.n_epochs, steps_per_epoch=n_ep, pct_start=0.3)
    elif args.scheduler == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=2, factor=0.5, min_lr=1e-6)
    return None


def main():
    parser = argparse.ArgumentParser(description="GNN experiment runner")
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--resume-from",     default="")
    parser.add_argument("--scheduler",       default="none",
                        choices=["none", "cosine", "step", "onecycle", "plateau"])
    parser.add_argument("--lr",              type=float, default=1e-3)
    parser.add_argument("--optimizer",       default="adam",
                        choices=["adam", "adamw", "sgd"])
    parser.add_argument("--n-epochs",        type=int, default=10)
    parser.add_argument("--hidden-dim",      type=int, default=64)
    parser.add_argument("--num-layers",      type=int, default=3)
    parser.add_argument("--weight-decay",    type=float, default=0.0)
    args = parser.parse_args()

    OUT_DIR = BASE_OUT_DIR / args.experiment_name
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    log_path = OUT_DIR / "metrics.log"

    # Seed baseline history into this experiment's log (copy 117-NewGNN baseline)
    baseline_log = Path("117-NewGNN/metrics.log")
    if baseline_log.exists() and not log_path.exists():
        shutil.copy(baseline_log, log_path)

    log_fh = open(log_path, "a", encoding="utf-8")

    # Episode setup (same deterministic split as base training)
    ep_dirs = sorted([d for d in PRECOMPUTE_DIR.iterdir() if d.is_dir()])
    ep_ids  = [int(d.name) for d in ep_dirs]
    random.seed(42)
    random.shuffle(ep_ids)
    split      = int(TRAIN_RATIO * len(ep_ids))
    train_ids  = set(ep_ids[:split])
    train_dirs = [PRECOMPUTE_DIR / str(eid) for eid in ep_ids if eid in train_ids]
    n_ep = min(EPISODES_PER_EPOCH, len(train_dirs))

    _log(f"\n--- Experiment: {args.experiment_name} ---", log_fh)
    _log(f"scheduler={args.scheduler}  lr={args.lr}  optimizer={args.optimizer}"
         f"  n_epochs={args.n_epochs}  weight_decay={args.weight_decay}", log_fh)
    _log(f"Episodes: {len(ep_ids)} total, {len(train_dirs)} train, {n_ep}/epoch", log_fh)

    model     = OrbitGNN(hidden_dim=args.hidden_dim, num_layers=args.num_layers)
    optimizer = make_optimizer(model, args)

    # Resume from explicit checkpoint
    if args.resume_from:
        ckpt_path = Path(args.resume_from)
        if ckpt_path.exists():
            model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
            _log(f"Loaded weights from {ckpt_path}", log_fh)
        else:
            _log(f"WARNING: {ckpt_path} not found; starting from random weights", log_fh)

    scheduler = make_scheduler(optimizer, args, n_ep)

    best_auc = 0.0
    for epoch in range(1, args.n_epochs + 1):
        all_scores, all_labels = [], []
        epoch_loss = 0.0
        epoch_dirs = random.sample(train_dirs, n_ep)

        for ep_dir in epoch_dirs:
            try:
                optimizer.zero_grad()
                loss, scores, labels = train_episode(ep_dir, model)
                optimizer.step()
                if args.scheduler == "onecycle" and scheduler is not None:
                    scheduler.step()
                epoch_loss += loss
                all_scores.extend(scores)
                all_labels.extend(labels)
            except Exception as e:
                _log(f"  episode {ep_dir.name} failed: {e}", log_fh)

        avg_loss = epoch_loss / max(n_ep, 1)
        n_pairs  = len(all_labels)
        n_pos    = int(sum(all_labels))
        pos_rate = n_pos / max(n_pairs, 1)

        if len(set(all_labels)) > 1:
            auc = roc_auc_score(all_labels, all_scores)
        else:
            auc = float("nan")

        current_lr = optimizer.param_groups[0]["lr"]
        line = (f"Epoch {epoch:3d}  loss={avg_loss:.4f}  train_auc={auc:.4f}"
                f"  pairs={n_pairs}  pos={n_pos}  pos_rate={pos_rate:.4f}"
                f"  lr={current_lr:.2e}")
        _log(line, log_fh)

        ckpt = OUT_DIR / f"model_epoch{epoch}.pt"
        torch.save(model.state_dict(), ckpt)
        _log(f"  -> saved {ckpt}", log_fh)

        if not math.isnan(auc):
            best_auc = max(best_auc, auc)

        # Epoch-level schedulers (not OneCycleLR which steps per episode)
        if scheduler is not None and args.scheduler != "onecycle":
            if args.scheduler == "plateau":
                scheduler.step(avg_loss)
            else:
                scheduler.step()

    _log(f"\nBest AUC: {best_auc:.4f}", log_fh)
    log_fh.close()
    print(f"METRIC train_auc={best_auc:.6f}")


if __name__ == "__main__":
    main()
