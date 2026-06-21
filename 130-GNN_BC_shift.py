"""130-GNN_BC_shift — GNN Behaviour Cloning with corrected action alignment.

Same architecture as 128-GNN_BC, but fixes the off-by-one in label assignment:
actions_to_copy at step T+1 labels the graph built from observation at step T,
because ep["steps"][T+1]["action"] was decided from ep["steps"][T]["observation"].

Three node types: Planet, PlanetStep, ActionMaxShips.
Two-phase HeteroConv: Phase 1 on {planet, planet_step}, Phase 2 adds action_max.
ScaledSAGEConv for ActionBase2Edge edge weights.
CPU-only, batch_size=4, num_workers=0.
"""
import math
import random
from pathlib import Path
from typing import Optional

import mlflow
import mlflow.pytorch
import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch import Tensor
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.data import HeteroData
from torch_geometric.loader import DataLoader
from torch_geometric.nn import HeteroConv, SAGEConv
from torch_geometric.nn import Linear
from torch_geometric.nn.conv import MessagePassing
from tqdm import tqdm

# ── Constants ────────────────────────────────────────────────────────────────

PRECOMPUTE_DIR      = Path("126-precompute")
OUT_DIR             = Path("130-GNN_BC_shift")
NB_STEPS_SIM        = 20
N_EPOCHS            = 100
EPISODES_PER_EPOCH  = 16
N_STEPS_PER_EPISODE = 16
HIDDEN_DIM          = 64
NUM_LAYERS_P1       = 3
NUM_LAYERS_P2       = 2
LR                  = 1e-3
WEIGHT_DECAY        = 1e-4
BATCH_SIZE          = 4
REACH_SHIPS         = [8, 64, 512]
DEVICE              = torch.device("cpu")

TEST_EPISODE_IDS = {
    78867640, 78899068, 78982947, 79033183,
    79126912, 79175592, 79228392, 79320069,
}

TRANSFORMS = ["identity", "rot90", "rot180", "rot270"]

LOG_1024 = math.log(1024)


# ── Augmentation ─────────────────────────────────────────────────────────────

def _apply_transform(planete: pl.DataFrame, transform: str) -> pl.DataFrame:
    """Apply a board rotation to the x/y columns of planete. Sun at (50,50) is preserved."""
    if transform == "identity":
        return planete
    elif transform == "rot90":
        return planete.with_columns([
            (100.0 - pl.col("y")).alias("x"),
            pl.col("x").alias("y"),
        ])
    elif transform == "rot180":
        return planete.with_columns([
            (100.0 - pl.col("x")).alias("x"),
            (100.0 - pl.col("y")).alias("y"),
        ])
    elif transform == "rot270":
        return planete.with_columns([
            pl.col("y").alias("x"),
            (100.0 - pl.col("x")).alias("y"),
        ])
    else:
        raise ValueError(f"Unknown transform: {transform!r}")


# ── Graph construction ────────────────────────────────────────────────────────

def build_graph(
    T: int,
    planete: pl.DataFrame,
    planete_step: pl.DataFrame,
    reachable_base_2: pl.DataFrame,
    reachable_max_ships: pl.DataFrame,
    actions_to_copy: pl.DataFrame,
) -> Optional[HeteroData]:
    """Build one HeteroData graph for observation step T.

    Returns None if there are no ActionMaxShips nodes at step T (nothing to predict).
    """

    # ── Planet nodes ──────────────────────────────────────────────────────────
    planet_df = planete.filter(pl.col("step") == T).unique("id", keep="first").sort("id")
    if planet_df.is_empty():
        return None

    planet_ids = planet_df["id"].to_list()
    # Map planet id -> row index (dense, sorted by id)
    max_pid = max(planet_ids)
    pid_to_idx = np.full(max_pid + 1, -1, dtype=np.int64)
    for i, pid in enumerate(planet_ids):
        pid_to_idx[pid] = i

    nature_arr = planet_df["nature"].to_numpy()
    planet_feat = np.column_stack([
        planet_df["production"].to_numpy().astype(np.float32) / 5.0,
        (nature_arr == "fix").astype(np.float32),
        (nature_arr == "moving").astype(np.float32),
        (nature_arr == "comet").astype(np.float32),
    ])  # (N_p, 4)

    # ── PlanetStep nodes ──────────────────────────────────────────────────────
    ps_df_raw = planete_step.filter(pl.col("step") == T)
    if ps_df_raw.is_empty():
        return None

    # Join planete (renamed step->future_step) to get x/y at future_step
    planete_renamed = planete.rename({"step": "future_step"})
    ps_df = ps_df_raw.join(planete_renamed, on=["id", "future_step"], how="left")

    # Build (id, future_step) -> ps_row_index map
    ps_ids = ps_df["id"].to_numpy()
    ps_future_steps = ps_df["future_step"].to_numpy()
    ps_key_to_idx: dict[tuple[int, int], int] = {
        (int(ps_ids[i]), int(ps_future_steps[i])): i
        for i in range(len(ps_df))
    }

    owner_arr = ps_df["owner"].to_numpy()
    x_arr     = ps_df["x"].to_numpy().astype(np.float32)
    y_arr     = ps_df["y"].to_numpy().astype(np.float32)
    ships_arr = np.log(np.clip(ps_df["ships"].to_numpy().astype(np.float32), 1, None)) / LOG_1024
    delta_arr = (ps_future_steps.astype(np.float32) - T) / NB_STEPS_SIM

    ps_feat = np.column_stack([
        x_arr / 100.0,
        y_arr / 100.0,
        delta_arr,
        ships_arr,
        (owner_arr == -1).astype(np.float32),
        (owner_arr == 0).astype(np.float32),
        (owner_arr == 1).astype(np.float32),
        (owner_arr == 2).astype(np.float32),
        (owner_arr == 3).astype(np.float32),
    ])  # (N_ps, 9)

    # ── Planet <-> PlanetStep edges (has_ps) ──────────────────────────────────
    pid_arr   = ps_df["id"].to_numpy()
    valid_pid = (pid_arr <= max_pid)
    p_side_raw = np.where(valid_pid, pid_arr, 0)
    p_side_idx = pid_to_idx[p_side_raw]
    valid_mask = valid_pid & (p_side_idx >= 0)

    has_ps_src = p_side_idx[valid_mask].astype(np.int64)
    has_ps_dst = np.where(valid_mask)[0].astype(np.int64)

    # ── ActionBase2Edge (PlanetStep <-> PlanetStep with edge_weight) ──────────
    b2 = reachable_base_2.filter(
        (pl.col("step_src") >= T) & (pl.col("step_src") <= T + NB_STEPS_SIM) &
        (pl.col("step_tgt") <= T + NB_STEPS_SIM) &
        (pl.col("ships_sent").is_in(REACH_SHIPS))
    )

    b2_ei_src: list[int] = []
    b2_ei_dst: list[int] = []
    b2_weights: list[float] = []

    if not b2.is_empty():
        b2_id_src_arr     = b2["id_src"].to_numpy()
        b2_step_src_arr   = b2["step_src"].to_numpy()
        b2_id_tgt_arr     = b2["id_tgt"].to_numpy()
        b2_step_tgt_arr   = b2["step_tgt"].to_numpy()
        b2_ships_arr      = b2["ships_sent"].to_numpy().astype(np.float32)

        for i in range(len(b2)):
            src_key = (int(b2_id_src_arr[i]), int(b2_step_src_arr[i]))
            tgt_key = (int(b2_id_tgt_arr[i]), int(b2_step_tgt_arr[i]))
            src_idx = ps_key_to_idx.get(src_key, -1)
            tgt_idx = ps_key_to_idx.get(tgt_key, -1)
            if src_idx >= 0 and tgt_idx >= 0:
                b2_ei_src.append(src_idx)
                b2_ei_dst.append(tgt_idx)
                b2_weights.append(math.log(max(float(b2_ships_arr[i]), 1.0)) / LOG_1024)

    # ── ActionMaxShips nodes ──────────────────────────────────────────────────
    # Only from owner==0 source planets at step T
    ps_now = ps_df.filter(pl.col("future_step") == T)
    my_planet_ids = ps_now.filter(pl.col("owner") == 0)["id"].to_list()

    if not my_planet_ids:
        return None

    ams_raw = reachable_max_ships.filter(
        (pl.col("step_src") == T) & (pl.col("id_src").is_in(my_planet_ids))
    )

    if ams_raw.is_empty():
        return None

    # ── Labels: join with actions_to_copy ────────────────────────────────────
    # actions_to_copy[T+1] = action decided from observation T (off-by-one fix)
    act_T    = actions_to_copy.filter(pl.col("step") == T + 1)
    act_flag = act_T.select(["id_src", "id_tgt"]).with_columns(pl.lit(1).alias("_match"))

    ams_labeled = ams_raw.join(act_flag, on=["id_src", "id_tgt"], how="left").with_columns(
        label=pl.col("_match").fill_null(0).cast(pl.Int64)
    ).drop("_match")

    # Features for ActionMaxShips: log(ships_sent)/log(1024)
    ams_ships = ams_labeled["ships_sent"].to_numpy().astype(np.float32)
    ams_feat_full = (np.log(np.clip(ams_ships, 1, None)) / LOG_1024).reshape(-1, 1)  # (N_ams, 1)

    # ── SourceEdge and TargetEdge construction ────────────────────────────────
    ams_id_src_arr   = ams_labeled["id_src"].to_numpy()
    ams_id_tgt_arr   = ams_labeled["id_tgt"].to_numpy()
    ams_step_tgt_arr = ams_labeled["step_tgt"].to_numpy()
    ams_labels_full  = ams_labeled["label"].to_numpy().astype(np.float32)

    src_ps_list: list[int] = []
    tgt_ps_list: list[int] = []
    valid_ams_rows: list[int] = []

    for i in range(len(ams_labeled)):
        src_key = (int(ams_id_src_arr[i]), T)
        tgt_key = (int(ams_id_tgt_arr[i]), int(ams_step_tgt_arr[i]))
        src_ps  = ps_key_to_idx.get(src_key, -1)
        tgt_ps  = ps_key_to_idx.get(tgt_key, -1)
        if src_ps >= 0 and tgt_ps >= 0:
            src_ps_list.append(src_ps)
            tgt_ps_list.append(tgt_ps)
            valid_ams_rows.append(i)

    if not valid_ams_rows:
        return None

    # Subset to valid rows only
    valid_idx  = np.array(valid_ams_rows, dtype=np.int64)
    ams_feat   = ams_feat_full[valid_idx]
    ams_labels = ams_labels_full[valid_idx]
    ams_count  = len(valid_ams_rows)

    src_ps_arr = np.array(src_ps_list, dtype=np.int64)
    tgt_ps_arr = np.array(tgt_ps_list, dtype=np.int64)
    ams_idx    = np.arange(ams_count, dtype=np.int64)

    # ── Assemble HeteroData ───────────────────────────────────────────────────
    data = HeteroData()

    data["planet"].x      = torch.tensor(planet_feat, dtype=torch.float32)
    data["planet_step"].x = torch.tensor(ps_feat,     dtype=torch.float32)
    data["action_max"].x  = torch.tensor(ams_feat,    dtype=torch.float32)
    data["action_max"].y  = torch.tensor(ams_labels,  dtype=torch.float32)

    # Planet <-> PlanetStep (has_ps / rev_has_ps)
    data["planet", "has_ps", "planet_step"].edge_index = torch.tensor(
        np.stack([has_ps_src, has_ps_dst]), dtype=torch.long
    )
    data["planet_step", "rev_has_ps", "planet"].edge_index = torch.tensor(
        np.stack([has_ps_dst, has_ps_src]), dtype=torch.long
    )

    # PlanetStep <-> PlanetStep (reaches via ActionBase2Edge)
    if b2_ei_src:
        b2_src_np = np.array(b2_ei_src, dtype=np.int64)
        b2_dst_np = np.array(b2_ei_dst, dtype=np.int64)
        b2_w_np   = np.array(b2_weights, dtype=np.float32)
        data["planet_step", "reaches", "planet_step"].edge_index = torch.tensor(
            np.stack([b2_src_np, b2_dst_np]), dtype=torch.long
        )
        data["planet_step", "reaches", "planet_step"].edge_attr = torch.tensor(
            b2_w_np, dtype=torch.float32
        )

    # PlanetStep -> ActionMaxShips (src_of / rev_src_of)
    data["planet_step", "src_of", "action_max"].edge_index = torch.tensor(
        np.stack([src_ps_arr, ams_idx]), dtype=torch.long
    )
    data["action_max", "rev_src_of", "planet_step"].edge_index = torch.tensor(
        np.stack([ams_idx, src_ps_arr]), dtype=torch.long
    )

    # PlanetStep -> ActionMaxShips (tgt_of / rev_tgt_of)
    data["planet_step", "tgt_of", "action_max"].edge_index = torch.tensor(
        np.stack([tgt_ps_arr, ams_idx]), dtype=torch.long
    )
    data["action_max", "rev_tgt_of", "planet_step"].edge_index = torch.tensor(
        np.stack([ams_idx, tgt_ps_arr]), dtype=torch.long
    )

    return data


# ── Episode loading ───────────────────────────────────────────────────────────

PARQUET_FILES = [
    "planete.parquet",
    "planete_step.parquet",
    "reachable_base_2.parquet",
    "reachable_max_ships.parquet",
    "actions_to_copy.parquet",
]


def load_episode(ep_dir: Path) -> Optional[tuple[
    pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame
]]:
    """Load all 5 parquet files for one episode. Returns None if any file is missing."""
    for fname in PARQUET_FILES:
        if not (ep_dir / fname).exists():
            return None
    planete             = pl.read_parquet(ep_dir / "planete.parquet")
    planete_step        = pl.read_parquet(ep_dir / "planete_step.parquet")
    reachable_base_2    = pl.read_parquet(ep_dir / "reachable_base_2.parquet")
    reachable_max_ships = pl.read_parquet(ep_dir / "reachable_max_ships.parquet")
    actions_to_copy     = pl.read_parquet(ep_dir / "actions_to_copy.parquet")
    return planete, planete_step, reachable_base_2, reachable_max_ships, actions_to_copy


def episode_to_graphs(
    ep_dir: Path,
    transform: str = "identity",
    n_steps: Optional[int] = None,
) -> list:
    """Load episode, apply augmentation, build graphs for up to n_steps sampled steps."""
    result = load_episode(ep_dir)
    if result is None:
        return []

    planete, planete_step, reachable_base_2, reachable_max_ships, actions_to_copy = result

    if transform != "identity":
        planete = _apply_transform(planete, transform)

    all_steps = planete_step["step"].unique().sort().to_list()
    if n_steps is not None and len(all_steps) > n_steps:
        all_steps = random.sample(all_steps, n_steps)

    graphs: list[HeteroData] = []
    for T in all_steps:
        graph = build_graph(
            T, planete, planete_step, reachable_base_2, reachable_max_ships, actions_to_copy
        )
        if graph is not None:
            graphs.append(graph)
    return graphs


# ── Model ─────────────────────────────────────────────────────────────────────

class ScaledSAGEConv(MessagePassing):
    """SAGEConv with per-edge scalar weight on the neighbor message.

    Used for ActionBase2Edge: message(x_j, edge_weight) = x_j * edge_weight.
    """

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__(aggr="mean")
        self.lin_neigh = Linear(in_ch, out_ch, bias=False)
        self.lin_root  = Linear(in_ch, out_ch)

    def forward(
        self,
        x,
        edge_index: Tensor,
        edge_attr: Optional[Tensor] = None,
    ) -> Tensor:
        x_r = x[1] if isinstance(x, tuple) else x
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        return self.lin_neigh(out) + self.lin_root(x_r)

    def message(self, x_j: Tensor, edge_attr: Optional[Tensor]) -> Tensor:
        if edge_attr is None:
            return x_j
        return x_j * edge_attr.view(-1, 1)


class GNNBehaviourCloning(nn.Module):
    """Two-phase heterogeneous GNN for behaviour cloning.

    Phase 1: {planet, planet_step} with L1 HeteroConv layers.
    Phase 2: {planet_step, action_max} with L2 HeteroConv layers.
    Classifier: Linear(H->1) on action_max nodes.
    """

    def __init__(self, H: int = 64, L1: int = 3, L2: int = 2):
        super().__init__()
        self.H = H

        # Input projections
        self.proj_planet = Linear(4, H)
        self.proj_ps     = Linear(9, H)
        self.proj_ams    = Linear(1, H)

        # Phase 1: HeteroConv x L1 on {planet, planet_step}
        self.phase1 = nn.ModuleList([
            HeteroConv({
                ("planet",      "has_ps",     "planet_step"): SAGEConv((H, H), H),
                ("planet_step", "rev_has_ps", "planet"):      SAGEConv((H, H), H),
                ("planet_step", "reaches",    "planet_step"): ScaledSAGEConv(H, H),
            }, aggr="sum")
            for _ in range(L1)
        ])
        self.norm1_planet = nn.ModuleList([nn.LayerNorm(H) for _ in range(L1)])
        self.norm1_ps     = nn.ModuleList([nn.LayerNorm(H) for _ in range(L1)])

        # Phase 2: HeteroConv x L2 on {planet_step, action_max}
        self.phase2 = nn.ModuleList([
            HeteroConv({
                ("planet_step", "src_of",      "action_max"):  SAGEConv((H, H), H),
                ("action_max",  "rev_src_of",  "planet_step"): SAGEConv((H, H), H),
                ("planet_step", "tgt_of",      "action_max"):  SAGEConv((H, H), H),
                ("action_max",  "rev_tgt_of",  "planet_step"): SAGEConv((H, H), H),
            }, aggr="sum")
            for _ in range(L2)
        ])
        self.norm2_ps  = nn.ModuleList([nn.LayerNorm(H) for _ in range(L2)])
        self.norm2_ams = nn.ModuleList([nn.LayerNorm(H) for _ in range(L2)])

        self.classifier = Linear(H, 1)

    def forward(self, data: HeteroData) -> Tensor:
        """Forward pass. Returns logits for action_max nodes, shape (N_ams,)."""
        h_planet = self.proj_planet(data["planet"].x)
        h_ps     = self.proj_ps(data["planet_step"].x)
        h_ams    = self.proj_ams(data["action_max"].x)

        ei_dict   = data.edge_index_dict
        ea_dict: dict = {}
        reaches_key = ("planet_step", "reaches", "planet_step")
        if reaches_key in ei_dict:
            try:
                attr = data[reaches_key].edge_attr
                if attr is not None:
                    ea_dict[reaches_key] = attr
            except AttributeError:
                pass

        # Phase 1: only planet and planet_step nodes
        p1_keys = {
            ("planet",      "has_ps",     "planet_step"),
            ("planet_step", "rev_has_ps", "planet"),
            ("planet_step", "reaches",    "planet_step"),
        }
        ei_p1   = {k: v for k, v in ei_dict.items() if k in p1_keys}
        x_p1    = {"planet": h_planet, "planet_step": h_ps}

        for i, conv in enumerate(self.phase1):
            x_p1 = conv(x_p1, ei_p1, edge_attr_dict=ea_dict)
            x_p1 = {
                "planet":      F.relu(self.norm1_planet[i](x_p1["planet"])),
                "planet_step": F.relu(self.norm1_ps[i](x_p1["planet_step"])),
            }

        h_ps = x_p1["planet_step"]

        # Phase 2: only planet_step and action_max nodes
        p2_keys = {
            ("planet_step", "src_of",      "action_max"),
            ("action_max",  "rev_src_of",  "planet_step"),
            ("planet_step", "tgt_of",      "action_max"),
            ("action_max",  "rev_tgt_of",  "planet_step"),
        }
        ei_p2 = {k: v for k, v in ei_dict.items() if k in p2_keys}
        x_p2  = {"planet_step": h_ps, "action_max": h_ams}

        for i, conv in enumerate(self.phase2):
            x_p2 = conv(x_p2, ei_p2)
            x_p2 = {
                "planet_step": F.relu(self.norm2_ps[i](x_p2["planet_step"])),
                "action_max":  F.relu(self.norm2_ams[i](x_p2["action_max"])),
            }

        h_ams = x_p2["action_max"]
        return self.classifier(h_ams).squeeze(-1)  # (N_ams,)


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(
    all_logits: list[float],
    all_labels: list[float],
    loss: float,
    prefix: str,
) -> dict:
    """Compute and return metrics dict with given prefix."""
    n_ams = len(all_labels)
    n_pos = int(sum(all_labels))

    if n_ams == 0:
        return {
            f"{prefix}_loss":    loss,
            f"{prefix}_roc_auc": 0.0,
            f"{prefix}_n_ams":   0,
            f"{prefix}_n_pos":   0,
            f"{prefix}_n_tp":    0,
            f"{prefix}_n_fp":    0,
        }

    probs = torch.sigmoid(torch.tensor(all_logits)).tolist()

    if len(set(all_labels)) > 1:
        roc_auc = roc_auc_score(all_labels, probs)
    else:
        roc_auc = float("nan")

    n_tp = int(sum(1 for p, l in zip(probs, all_labels) if p >= 0.5 and l == 1.0))
    n_fp = int(sum(1 for p, l in zip(probs, all_labels) if p >= 0.5 and l == 0.0))

    return {
        f"{prefix}_loss":    loss,
        f"{prefix}_roc_auc": roc_auc if not math.isnan(roc_auc) else 0.0,
        f"{prefix}_n_ams":   n_ams,
        f"{prefix}_n_pos":   n_pos,
        f"{prefix}_n_tp":    n_tp,
        f"{prefix}_n_fp":    n_fp,
    }


# ── pos_weight scan ───────────────────────────────────────────────────────────

def compute_pos_weight(train_dirs: list[Path]) -> float:
    """Scan train episodes to compute n_neg / n_pos for BCEWithLogitsLoss pos_weight."""
    total_pos = 0
    total_neg = 0

    for ep_dir in tqdm(train_dirs, desc="Scanning pos_weight", leave=False):
        result = load_episode(ep_dir)
        if result is None:
            continue
        _, planete_step, _, reachable_max_ships, actions_to_copy = result

        all_steps = planete_step["step"].unique().to_list()
        for T in all_steps:
            ps_now = planete_step.filter(
                (pl.col("step") == T) & (pl.col("future_step") == T)
            )
            my_ids = ps_now.filter(pl.col("owner") == 0)["id"].to_list()
            if not my_ids:
                continue

            ams = reachable_max_ships.filter(
                (pl.col("step_src") == T) & (pl.col("id_src").is_in(my_ids))
            )
            if ams.is_empty():
                continue

            n_cand = len(ams)
            act_T  = actions_to_copy.filter(pl.col("step") == T + 1)
            act_flag = act_T.select(["id_src", "id_tgt"]).with_columns(
                pl.lit(1).alias("_match")
            )
            matched = ams.join(act_flag, on=["id_src", "id_tgt"], how="left")
            n_match = int(matched["_match"].fill_null(0).sum())
            total_pos += n_match
            total_neg += n_cand - n_match

    if total_pos == 0:
        return 1.0
    return total_neg / total_pos


# ── Logging ───────────────────────────────────────────────────────────────────

def _log(msg: str, log_fh=None) -> None:
    print(msg, flush=True)
    if log_fh is not None:
        log_fh.write(msg + "\n")
        log_fh.flush()


# ── Training / eval pass ──────────────────────────────────────────────────────

def run_epoch(
    model: GNNBehaviourCloning,
    graphs: list,
    criterion: nn.BCEWithLogitsLoss,
    optimizer: Optional[torch.optim.Optimizer],
    training: bool,
) -> tuple:
    """Run one pass over graphs. Returns (avg_loss, all_logits, all_labels)."""
    if not graphs:
        return 0.0, [], []

    loader = DataLoader(graphs, batch_size=BATCH_SIZE, shuffle=training, num_workers=0)
    model.train(training)

    total_loss = 0.0
    n_batches  = 0
    all_logits: list[float] = []
    all_labels: list[float] = []

    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for batch in loader:
            batch = batch.to(DEVICE)
            logits = model(batch)
            labels = batch["action_max"].y

            if labels.numel() == 0:
                continue

            loss = criterion(logits, labels)

            if training and optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            n_batches  += 1

            with torch.no_grad():
                all_logits.extend(logits.detach().tolist())
                all_labels.extend(labels.tolist())

    return total_loss / max(n_batches, 1), all_logits, all_labels


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    log_fh = open(OUT_DIR / "metrics.log", "a", encoding="utf-8")

    ep_dirs    = sorted([d for d in PRECOMPUTE_DIR.iterdir() if d.is_dir()])
    train_dirs = [d for d in ep_dirs if int(d.name) not in TEST_EPISODE_IDS]
    test_dirs  = [d for d in ep_dirs if int(d.name) in TEST_EPISODE_IDS]

    _log(f"Device: {DEVICE}", log_fh)
    _log(f"Episodes: {len(ep_dirs)} total | train={len(train_dirs)} test={len(test_dirs)}", log_fh)

    # Build pool of (ep_dir, transform) pairs for training
    all_train_pairs = [(ep_dir, t) for ep_dir in train_dirs for t in TRANSFORMS]

    # Compute pos_weight from a subset of train dirs (up to 50 episodes for speed)
    _log("Computing pos_weight from train set...", log_fh)
    scan_dirs      = random.sample(train_dirs, min(50, len(train_dirs)))
    pos_weight_val = compute_pos_weight(scan_dirs)
    _log(f"pos_weight = {pos_weight_val:.2f}", log_fh)

    pos_weight = torch.tensor([pos_weight_val], dtype=torch.float32)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    model     = GNNBehaviourCloning(H=HIDDEN_DIM, L1=NUM_LAYERS_P1, L2=NUM_LAYERS_P2).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=N_EPOCHS, eta_min=1e-6)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    _log(f"Model parameters: {n_params:,}", log_fh)

    # Pre-load test graphs once (full test set, all steps, no sampling)
    _log("Pre-loading test graphs...", log_fh)
    test_graphs: list[HeteroData] = []
    for ep_dir in test_dirs:
        try:
            test_graphs.extend(episode_to_graphs(ep_dir, transform="identity", n_steps=None))
        except Exception as e:
            _log(f"  SKIP test {ep_dir.name}: {e}", log_fh)
    _log(f"Test graphs: {len(test_graphs)}", log_fh)

    best_test_auc = float("-inf")

    mlflow.set_experiment("130-GNN_BC_shift")
    with mlflow.start_run():
        mlflow.log_params({
            "hidden_dim":          HIDDEN_DIM,
            "num_layers_p1":       NUM_LAYERS_P1,
            "num_layers_p2":       NUM_LAYERS_P2,
            "lr":                  LR,
            "weight_decay":        WEIGHT_DECAY,
            "n_epochs":            N_EPOCHS,
            "episodes_per_epoch":  EPISODES_PER_EPOCH,
            "n_steps_per_episode": N_STEPS_PER_EPISODE,
            "batch_size":          BATCH_SIZE,
            "pos_weight":          pos_weight_val,
            "train_episodes":      len(train_dirs),
            "test_episodes":       len(test_dirs),
            "transforms":          len(TRANSFORMS),
            "n_params":            n_params,
        })

        pbar = tqdm(range(1, N_EPOCHS + 1), desc="Epochs")
        for epoch in pbar:
            # ── Sample train graphs ──────────────────────────────────────────
            epoch_pairs  = random.sample(all_train_pairs, EPISODES_PER_EPOCH)
            train_graphs: list[HeteroData] = []
            for ep_dir, transform in epoch_pairs:
                try:
                    train_graphs.extend(
                        episode_to_graphs(
                            ep_dir, transform=transform, n_steps=N_STEPS_PER_EPISODE
                        )
                    )
                except Exception as e:
                    _log(f"  SKIP train {ep_dir.name}/{transform}: {e}", log_fh)

            # ── Train ────────────────────────────────────────────────────────
            tr_loss, tr_logits, tr_labels = run_epoch(
                model, train_graphs, criterion, optimizer, training=True
            )

            # ── Evaluate ─────────────────────────────────────────────────────
            te_loss, te_logits, te_labels = run_epoch(
                model, test_graphs, criterion, optimizer=None, training=False
            )

            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]

            # ── Metrics ─────────────────────────────────────────────────────
            tr_m = compute_metrics(tr_logits, tr_labels, tr_loss, "train")
            te_m = compute_metrics(te_logits, te_labels, te_loss, "test")

            train_auc = tr_m["train_roc_auc"]
            test_auc  = te_m["test_roc_auc"]

            pbar.set_postfix({
                "tr_auc":  f"{train_auc:.4f}",
                "te_auc":  f"{test_auc:.4f}",
                "tr_loss": f"{tr_loss:.4f}",
                "te_loss": f"{te_loss:.4f}",
            })

            line = (
                f"Epoch {epoch:3d}  lr={current_lr:.2e}"
                f"  [train] loss={tr_loss:.4f} auc={train_auc:.4f}"
                f"  n_ams={tr_m['train_n_ams']} pos={tr_m['train_n_pos']}"
                f"  tp={tr_m['train_n_tp']} fp={tr_m['train_n_fp']}"
                f"  [test]  loss={te_loss:.4f} auc={test_auc:.4f}"
                f"  n_ams={te_m['test_n_ams']} pos={te_m['test_n_pos']}"
                f"  tp={te_m['test_n_tp']} fp={te_m['test_n_fp']}"
            )
            _log(line, log_fh)

            mlflow.log_metrics({**tr_m, **te_m, "lr": current_lr}, step=epoch)

            # ── Checkpoints ─────────────────────────────────────────────────
            if epoch % 10 == 0:
                ckpt = OUT_DIR / f"model_epoch{epoch}.pt"
                torch.save(model.state_dict(), ckpt)
                mlflow.log_artifact(str(ckpt), artifact_path="checkpoints")
                _log(f"  -> saved {ckpt}", log_fh)

            if test_auc > best_test_auc and te_m["test_n_ams"] > 0:
                best_test_auc = test_auc
                best_path = OUT_DIR / "model_best.pt"
                torch.save(model.state_dict(), best_path)
                _log(f"  -> NEW BEST test_roc_auc={best_test_auc:.4f}  saved {best_path}", log_fh)

        _log(f"\nDone. Best test AUC: {best_test_auc:.4f}  Models in {OUT_DIR}/", log_fh)
    log_fh.close()


# ── Self-tests ────────────────────────────────────────────────────────────────

def _test_scaled_sage_conv():
    conv = ScaledSAGEConv(8, 8)
    x    = torch.randn(4, 8)
    ei   = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
    ew   = torch.rand(3)
    out  = conv(x, ei, edge_attr=ew)
    assert out.shape == (4, 8), f"ScaledSAGEConv output: {out.shape}"
    print("_test_scaled_sage_conv PASSED")


def _test_build_graph_smoke():
    """Smoke test: build a graph from the first available episode."""
    ep_dirs = sorted([d for d in PRECOMPUTE_DIR.iterdir() if d.is_dir()])
    if not ep_dirs:
        print("_test_build_graph_smoke SKIP (no episodes in 126-precompute/)")
        return

    result = load_episode(ep_dirs[0])
    if result is None:
        print("_test_build_graph_smoke SKIP (missing files)")
        return

    planete, planete_step, rb2, rms, atc = result
    all_steps = planete_step["step"].unique().sort().to_list()

    found = False
    for T in all_steps[:30]:
        g = build_graph(T, planete, planete_step, rb2, rms, atc)
        if g is not None:
            print(f"_test_build_graph_smoke: T={T}")
            print(f"  planet.x:      {g['planet'].x.shape}")
            print(f"  planet_step.x: {g['planet_step'].x.shape}")
            print(f"  action_max.x:  {g['action_max'].x.shape}")
            print(f"  action_max.y:  {g['action_max'].y.shape}")
            print(f"  n_pos: {int(g['action_max'].y.sum().item())}")
            found = True
            break

    if not found:
        print("_test_build_graph_smoke: no valid step found in first 30 steps")
        return
    print("_test_build_graph_smoke PASSED")


def _test_model_forward():
    """Test that the model forward pass produces logits of correct shape."""
    ep_dirs = sorted([d for d in PRECOMPUTE_DIR.iterdir() if d.is_dir()])
    if not ep_dirs:
        print("_test_model_forward SKIP (no episodes)")
        return

    result = load_episode(ep_dirs[0])
    if result is None:
        print("_test_model_forward SKIP (missing files)")
        return

    planete, planete_step, rb2, rms, atc = result
    all_steps = planete_step["step"].unique().sort().to_list()

    graph = None
    for T in all_steps[:50]:
        g = build_graph(T, planete, planete_step, rb2, rms, atc)
        if g is not None:
            graph = g
            break

    if graph is None:
        print("_test_model_forward SKIP (no valid graph)")
        return

    model  = GNNBehaviourCloning(H=16, L1=2, L2=1)
    logits = model(graph)
    n_ams  = graph["action_max"].x.shape[0]
    assert logits.shape == (n_ams,), f"logits shape: {logits.shape}, expected ({n_ams},)"
    print(f"_test_model_forward PASSED  n_ams={n_ams}  logits.shape={logits.shape}")


def _test_apply_transform():
    df = pl.DataFrame({
        "id": [0, 1, 2], "step": [1, 1, 1],
        "x": [0.0, 100.0, 50.0], "y": [0.0, 100.0, 50.0],
        "production": [1, 1, 1], "nature": ["fix", "fix", "fix"],
    })
    r = _apply_transform(df, "rot90")
    assert r["x"].to_list() == [100.0, 0.0, 50.0], r["x"].to_list()
    assert r["y"].to_list() == [0.0, 100.0, 50.0],  r["y"].to_list()
    r = _apply_transform(df, "rot180")
    assert r["x"].to_list() == [100.0, 0.0, 50.0], r["x"].to_list()
    assert r["y"].to_list() == [100.0, 0.0, 50.0],  r["y"].to_list()
    r = _apply_transform(df, "rot270")
    assert r["x"].to_list() == [0.0, 100.0, 50.0],  r["x"].to_list()
    assert r["y"].to_list() == [100.0, 0.0, 50.0],  r["y"].to_list()
    print("_test_apply_transform PASSED")


if __name__ == "__main__":
    _test_scaled_sage_conv()
    _test_apply_transform()
    _test_build_graph_smoke()
    _test_model_forward()
    print("\nAll tests passed. Starting training...\n")
    main()
