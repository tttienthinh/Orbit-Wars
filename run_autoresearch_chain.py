"""Autoresearch chain runner.

Imports torch/torch_geometric ONCE, then runs experiments in sequence.
Each experiment saves to 119-NewGNN_autoresearch/{name}/ and appends to experiments.jsonl.

Usage:
    python run_autoresearch_chain.py
"""
import json
import math
import random
import shutil
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch_geometric.data import HeteroData
from torch_geometric.nn import GATConv, HeteroConv, SAGEConv
from torch_geometric.transforms import ToUndirected

print("All imports loaded.", flush=True)

# ── Constants ──────────────────────────────────────────────────────────────────
PRECOMPUTE_DIR     = Path("114-precompute")
BASE_OUT_DIR       = Path("119-NewGNN_autoresearch")
JSONL_PATH         = BASE_OUT_DIR / "experiments.jsonl"
NB_STEPS_SIM       = 20
EPISODES_PER_EPOCH = 8
TRAIN_RATIO        = 0.8


# ── Experiment config ──────────────────────────────────────────────────────────
@dataclass
class Exp:
    name: str
    resume_from: str = ""
    scheduler: str = "none"
    lr: float = 1e-3
    optimizer: str = "adam"
    n_epochs: int = 10
    hidden_dim: int = 64
    num_layers: int = 3
    weight_decay: float = 0.0
    clip_grad_norm: float = 0.0  # 0 = disabled


# ── GNN model (identical to 119-NewGNN_autoresearch.py) ───────────────────────
_REACHES_KEY = ("planet_step", "reaches", "planet_step")


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


def _next_run_number():
    if not JSONL_PATH.exists():
        return 1
    run = 0
    for line in JSONL_PATH.read_text().splitlines():
        try:
            obj = json.loads(line)
            if obj.get("type") != "config":
                run = max(run, obj.get("run", 0))
        except json.JSONDecodeError:
            pass
    return run + 1


def _best_auc_so_far():
    if not JSONL_PATH.exists():
        return 0.0
    best = 0.0
    for line in JSONL_PATH.read_text().splitlines():
        try:
            obj = json.loads(line)
            if obj.get("type") != "config" and obj.get("status") == "keep":
                best = max(best, obj.get("metric", 0.0))
        except json.JSONDecodeError:
            pass
    return best


def make_optimizer(model, exp):
    if exp.optimizer == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=exp.lr, weight_decay=exp.weight_decay)
    elif exp.optimizer == "sgd":
        return torch.optim.SGD(model.parameters(), lr=exp.lr, momentum=0.9, weight_decay=exp.weight_decay)
    else:
        return torch.optim.Adam(model.parameters(), lr=exp.lr, weight_decay=exp.weight_decay)


def make_scheduler(optimizer, exp, n_ep):
    if exp.scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=exp.n_epochs, eta_min=1e-6)
    elif exp.scheduler == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(exp.n_epochs // 3, 1), gamma=0.5)
    elif exp.scheduler == "onecycle":
        return torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=exp.lr * 10,
            epochs=exp.n_epochs, steps_per_epoch=n_ep, pct_start=0.3)
    elif exp.scheduler == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.5, min_lr=1e-6)
    return None


def run_experiment(exp: Exp, train_dirs, n_ep):
    """Run one experiment. Returns best_auc."""
    OUT_DIR = BASE_OUT_DIR / exp.name
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    log_path = OUT_DIR / "metrics.log"
    baseline_log = Path("117-NewGNN/metrics.log")
    if baseline_log.exists() and not log_path.exists():
        shutil.copy(baseline_log, log_path)

    log_fh = open(log_path, "a", encoding="utf-8")

    _log(f"\n{'='*60}", log_fh)
    _log(f"Experiment: {exp.name}", log_fh)
    _log(f"scheduler={exp.scheduler}  lr={exp.lr}  optimizer={exp.optimizer}"
         f"  n_epochs={exp.n_epochs}  weight_decay={exp.weight_decay}", log_fh)

    model     = OrbitGNN(hidden_dim=exp.hidden_dim, num_layers=exp.num_layers)
    optimizer = make_optimizer(model, exp)

    if exp.resume_from:
        ckpt = Path(exp.resume_from)
        if ckpt.exists():
            model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
            _log(f"Loaded weights from {ckpt}", log_fh)
        else:
            _log(f"WARNING: {ckpt} not found — starting from scratch", log_fh)

    scheduler = make_scheduler(optimizer, exp, n_ep)
    best_auc  = 0.0

    for epoch in range(1, exp.n_epochs + 1):
        all_scores, all_labels = [], []
        epoch_loss = 0.0
        epoch_dirs = random.sample(train_dirs, n_ep)

        for ep_dir in epoch_dirs:
            try:
                optimizer.zero_grad()
                loss, scores, labels = train_episode(ep_dir, model)
                if exp.clip_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), exp.clip_grad_norm)
                optimizer.step()
                if exp.scheduler == "onecycle" and scheduler is not None:
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

        if scheduler is not None and exp.scheduler != "onecycle":
            if exp.scheduler == "plateau":
                scheduler.step(avg_loss)
            else:
                scheduler.step()

    _log(f"Best AUC for {exp.name}: {best_auc:.4f}", log_fh)
    log_fh.close()
    return best_auc, OUT_DIR / f"model_epoch{exp.n_epochs}.pt"


def log_experiment(run_num, exp, best_auc, prior_best):
    status = "keep" if best_auc > prior_best else "discard"
    entry = {
        "run":     run_num,
        "metric":  round(best_auc, 6),
        "status":  status,
        "description": exp.name,
        "timestamp": int(time.time()),
        "config": {
            "scheduler":      exp.scheduler,
            "lr":             exp.lr,
            "optimizer":      exp.optimizer,
            "n_epochs":       exp.n_epochs,
            "weight_decay":   exp.weight_decay,
            "clip_grad_norm": exp.clip_grad_norm,
            "resume_from":    exp.resume_from,
        }
    }
    with open(JSONL_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")
    print(f"[JSONL] run={run_num} best_auc={best_auc:.4f} status={status}", flush=True)
    return status


def make_readme(exp_dir, exp, best_auc, prev_exp_name):
    readme = exp_dir / "README.md"
    if readme.exists():
        return
    desc_map = {
        "cosine":   "CosineAnnealingLR (lr→1e-6 over N epochs)",
        "step":     "StepLR (gamma=0.5 every N/3 epochs)",
        "onecycle": "OneCycleLR (1-cycle policy, max_lr=10×base)",
        "plateau":  "ReduceLROnPlateau (patience=2, factor=0.5)",
        "none":     "no scheduler (constant LR)",
    }
    sched_desc = desc_map.get(exp.scheduler, exp.scheduler)
    resume_desc = f"Continues from `{prev_exp_name}`" if prev_exp_name else "Starts from `117-NewGNN/model_epoch82.pt` (baseline)"
    clip_str = f", clip_grad_norm={exp.clip_grad_norm}" if exp.clip_grad_norm > 0 else ""
    lines = [
        f"# Experiment {exp.name}",
        "",
        f"{resume_desc}, using **{exp.optimizer.upper()}** optimizer with {sched_desc}.",
        f"Trained for {exp.n_epochs} epochs with lr={exp.lr}, weight_decay={exp.weight_decay}{clip_str}.",
        f"**Best AUC achieved: {best_auc:.4f}**",
        "",
        f"**Config:** {exp.optimizer} lr={exp.lr}, scheduler={exp.scheduler}, "
        f"n_epochs={exp.n_epochs}, weight_decay={exp.weight_decay}{clip_str}",
    ]
    readme.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    BASE_OUT_DIR.mkdir(exist_ok=True)

    # Ensure config header exists
    if not JSONL_PATH.exists():
        with open(JSONL_PATH, "w") as f:
            f.write(json.dumps({
                "type": "config",
                "name": "gnn-training-autoresearch",
                "metricName": "best_auc",
                "metricUnit": "ROC AUC",
                "bestDirection": "higher"
            }) + "\n")

    # Episode split (deterministic)
    ep_dirs = sorted([d for d in PRECOMPUTE_DIR.iterdir() if d.is_dir()])
    ep_ids  = [int(d.name) for d in ep_dirs]
    random.seed(42)
    random.shuffle(ep_ids)
    split      = int(TRAIN_RATIO * len(ep_ids))
    train_ids  = set(ep_ids[:split])
    train_dirs = [PRECOMPUTE_DIR / str(eid) for eid in ep_ids if eid in train_ids]
    n_ep = min(EPISODES_PER_EPOCH, len(train_dirs))
    print(f"Episodes: {len(ep_ids)} total, {len(train_dirs)} train, {n_ep}/epoch", flush=True)

    # ── Planned experiment chain ──────────────────────────────────────────────
    planned = [
        Exp("001-CosinusAnnealing",
            resume_from="117-NewGNN/model_epoch82.pt",
            scheduler="cosine", lr=1e-3, optimizer="adam", n_epochs=10),
        Exp("002-AdamW-WeightDecay",
            resume_from="119-NewGNN_autoresearch/001-CosinusAnnealing/model_epoch10.pt",
            scheduler="cosine", lr=1e-3, optimizer="adamw", weight_decay=1e-4, n_epochs=10),
        Exp("003-LowLR-FineTune",
            resume_from="119-NewGNN_autoresearch/002-AdamW-WeightDecay/model_epoch10.pt",
            scheduler="cosine", lr=1e-4, optimizer="adam", n_epochs=10),
        Exp("004-StepLR-AdamW",
            resume_from="119-NewGNN_autoresearch/003-LowLR-FineTune/model_epoch10.pt",
            scheduler="step", lr=5e-4, optimizer="adamw", weight_decay=1e-4, n_epochs=10),
        Exp("005-OneCycle",
            resume_from="119-NewGNN_autoresearch/004-StepLR-AdamW/model_epoch10.pt",
            scheduler="onecycle", lr=1e-4, optimizer="adamw", weight_decay=1e-4, n_epochs=10),
        Exp("006-Plateau-SGD",
            resume_from="119-NewGNN_autoresearch/005-OneCycle/model_epoch10.pt",
            scheduler="plateau", lr=1e-3, optimizer="sgd", weight_decay=1e-4, n_epochs=10),
        Exp("007-LongFineTune",
            resume_from="119-NewGNN_autoresearch/006-Plateau-SGD/model_epoch10.pt",
            scheduler="cosine", lr=5e-5, optimizer="adamw", weight_decay=1e-5, n_epochs=20),
        # Wave 2: resume from the ACTUAL best checkpoint (003/epoch7, AUC=0.9611)
        # SGD (006) destroyed the chain; restart from the peak rather than endpoint
        Exp("008-BestCkpt-VeryLowLR",
            resume_from="119-NewGNN_autoresearch/003-LowLR-FineTune/model_epoch7.pt",
            scheduler="cosine", lr=5e-5, optimizer="adam", n_epochs=10),
        Exp("009-BestCkpt-AdamW",
            resume_from="119-NewGNN_autoresearch/003-LowLR-FineTune/model_epoch7.pt",
            scheduler="cosine", lr=5e-5, optimizer="adamw", weight_decay=1e-4, n_epochs=10),
        Exp("010-BestCkpt-LongRun",
            resume_from="119-NewGNN_autoresearch/003-LowLR-FineTune/model_epoch7.pt",
            scheduler="cosine", lr=1e-4, optimizer="adam", n_epochs=20),
        # Wave 3: resume from 009/ep3 (AUC=0.9664, new best) and refine further
        Exp("011-NewPeak-VeryLowLR",
            resume_from="119-NewGNN_autoresearch/009-BestCkpt-AdamW/model_epoch3.pt",
            scheduler="cosine", lr=2e-5, optimizer="adam", n_epochs=10),
        Exp("012-NewPeak-AdamW-Lower",
            resume_from="119-NewGNN_autoresearch/009-BestCkpt-AdamW/model_epoch3.pt",
            scheduler="cosine", lr=2e-5, optimizer="adamw", weight_decay=1e-4, n_epochs=10),
        Exp("013-NewPeak-GradClip",
            resume_from="119-NewGNN_autoresearch/009-BestCkpt-AdamW/model_epoch3.pt",
            scheduler="cosine", lr=5e-5, optimizer="adam", n_epochs=10, clip_grad_norm=1.0),
        Exp("014-NewPeak-LongRun",
            resume_from="119-NewGNN_autoresearch/009-BestCkpt-AdamW/model_epoch3.pt",
            scheduler="cosine", lr=5e-5, optimizer="adam", n_epochs=20),
        Exp("015-NewPeak-AdamW-GradClip",
            resume_from="119-NewGNN_autoresearch/009-BestCkpt-AdamW/model_epoch3.pt",
            scheduler="cosine", lr=3e-5, optimizer="adamw", weight_decay=1e-4,
            n_epochs=10, clip_grad_norm=1.0),
    ]

    prior_best = _best_auc_so_far()
    run_num    = _next_run_number()
    prev_name  = None

    for exp in planned:
        # Skip if already done (model_epoch{N}.pt exists)
        out_dir = BASE_OUT_DIR / exp.name
        last_ckpt = out_dir / f"model_epoch{exp.n_epochs}.pt"
        if last_ckpt.exists():
            print(f"Skipping {exp.name} (already done)", flush=True)
            # Still need to count it for run_num and prev_name tracking
            try:
                # Estimate best_auc from log
                log_text = (out_dir / "metrics.log").read_text()
                aucs = [float(line.split("train_auc=")[1].split()[0])
                        for line in log_text.splitlines() if "train_auc=" in line]
                best_auc = max(aucs) if aucs else 0.0
            except Exception:
                best_auc = 0.0
            prior_best = max(prior_best, best_auc)
            prev_name  = exp.name
            run_num   += 1
            continue

        print(f"\n>>> Starting {exp.name} <<<", flush=True)
        best_auc, _ = run_experiment(exp, train_dirs, n_ep)
        status = log_experiment(run_num, exp, best_auc, prior_best)
        make_readme(out_dir, exp, best_auc, prev_name)

        if status == "keep":
            prior_best = best_auc
        prev_name = exp.name
        run_num  += 1

    print("\nAll planned experiments complete. Chain done.", flush=True)


if __name__ == "__main__":
    main()
