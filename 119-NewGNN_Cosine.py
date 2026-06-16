"""119-NewGNN_Cosine — Two-tower GNN with AdamW+CosineAnnealingLR, augmentation, train/test split."""
import math
import random
from pathlib import Path

import mlflow
import mlflow.pytorch
import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch_geometric.data import HeteroData
from torch_geometric.nn import GATConv, HeteroConv, SAGEConv
from torch_geometric.transforms import ToUndirected

PRECOMPUTE_DIR = Path("114-precompute")
OUT_DIR        = Path("119-NewGNN_Cosine")
NB_STEPS_SIM   = 20
N_EPOCHS       = 30
HIDDEN_DIM     = 64
NUM_LAYERS     = 3
LR             = 1e-3
WEIGHT_DECAY   = 1e-4

TEST_EPISODE_IDS = {
    78867640, 78899068, 78982947, 79033183,
    79126912, 79175592, 79228392, 79320069,
}

TRANSFORMS = ["identity", "rot90", "rot180", "rot270"]


def build_graph(
    df_s_t: pl.DataFrame,
    reach_t: pl.DataFrame,
) -> tuple[HeteroData, dict[int, int]]:
    """Build one HeteroData graph for game step t.

    Args:
        df_s_t: planet state rows — all planets at step t plus arrival-step rows from reach_t.
        reach_t: reach edges with step_src == t (already groupby-deduped).

    Returns:
        (data, planet_idx) where planet_idx maps planet_id -> Planet node row index.
    """
    if reach_t.is_empty():
        raise ValueError("reach_t must be non-empty (filtered to a specific step_src)")
    game_step = int(reach_t["step_src"][0])

    # ── Planet nodes ──────────────────────────────────────────────────────────
    # One node per unique planet; use earliest step occurrence for static features.
    planet_rows = df_s_t.sort("step").unique(subset=["id"], keep="first").sort("id")
    planet_ids  = planet_rows["id"].to_list()
    planet_idx: dict[int, int] = {pid: i for i, pid in enumerate(planet_ids)}

    nature_arr  = planet_rows["nature"].to_numpy()
    planet_feat = np.column_stack([
        (nature_arr == "fix").astype(np.float32),
        (nature_arr == "moving").astype(np.float32),
        (nature_arr == "comet").astype(np.float32),
        planet_rows["production"].to_numpy().astype(np.float32) / 5.0,
    ])  # shape (N_planets, 4)

    # ── PlanetStep nodes ──────────────────────────────────────────────────────
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
    ])  # shape (N_ps, 9)

    # ── has_snapshot edges ────────────────────────────────────────────────────
    pid_to_pidx = np.zeros(max(planet_ids) + 1, dtype=np.int64)
    for pid, idx in planet_idx.items():
        pid_to_pidx[pid] = idx
    snap_src = pid_to_pidx[df_s_t["id"].to_numpy()]          # planet idx per ps row
    snap_dst = np.arange(len(df_s_t), dtype=np.int64)        # ps_idx per ps row

    # ── reaches edges ─────────────────────────────────────────────────────────
    reach_joined = (
        reach_t
        .join(
            ps_idx_df.rename({"id": "id_src", "step": "step_src", "ps_idx": "src_ps_idx"}),
            on=["id_src", "step_src"], how="inner",
        )
        .join(
            ps_idx_df.rename({"ps_idx": "dst_ps_idx"}),
            on=["id", "step"], how="inner",
        )
    )

    # ── Assemble HeteroData ───────────────────────────────────────────────────
    data = HeteroData()
    data["planet"].x      = torch.tensor(planet_feat, dtype=torch.float32)
    data["planet_step"].x = torch.tensor(ps_feat,     dtype=torch.float32)
    data["planet", "has_snapshot", "planet_step"].edge_index = torch.tensor(
        np.stack([snap_src, snap_dst]), dtype=torch.long
    )
    if not reach_joined.is_empty():
        ships = reach_joined["ships_sent"].to_numpy().astype(np.float32)
        data["planet_step", "reaches", "planet_step"].edge_index = torch.tensor(
            np.stack([
                reach_joined["src_ps_idx"].to_numpy().astype(np.int64),
                reach_joined["dst_ps_idx"].to_numpy().astype(np.int64),
            ]),
            dtype=torch.long,
        )
        data["planet_step", "reaches", "planet_step"].edge_attr = torch.tensor(
            np.log2(np.clip(ships, 1, None)) / 10.0, dtype=torch.float32
        ).unsqueeze(1)

    return ToUndirected()(data), planet_idx


_REACHES_KEY = ("planet_step", "reaches", "planet_step")


class OrbitGNN(nn.Module):
    def __init__(self, hidden_dim: int = 64, num_layers: int = 3):
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
            nn.Linear(2 * H, H),
            nn.ReLU(),
            nn.Linear(H, H // 2),
            nn.ReLU(),
            nn.Linear(H // 2, 1),
        )

    def encode(self, data: HeteroData) -> torch.Tensor:
        """Run message passing. Returns Planet node embeddings, shape (N_planets, H)."""
        x_dict = {
            "planet":      self.planet_proj(data["planet"].x),
            "planet_step": self.planet_step_proj(data["planet_step"].x),
        }
        edge_index_dict = data.edge_index_dict
        edge_attr_map: dict = {}
        if _REACHES_KEY in edge_index_dict:
            try:
                attr = data[_REACHES_KEY].edge_attr
            except AttributeError:
                attr = None
            if attr is not None:
                edge_attr_map[_REACHES_KEY] = attr

        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict, edge_attr_dict=edge_attr_map)
            x_dict = {k: F.relu(v) for k, v in x_dict.items()}

        return x_dict["planet"]  # (N_planets, H)

    def score_pairs(
        self,
        h_planet: torch.Tensor,
        src_idx: torch.Tensor,
        tgt_idx: torch.Tensor,
    ) -> torch.Tensor:
        """Two-tower MLP scorer. Returns logits of shape (N_pairs,)."""
        return self.pair_mlp(
            torch.cat([h_planet[src_idx], h_planet[tgt_idx]], dim=-1)
        ).squeeze(-1)


def build_attack_pairs(
    df_s_at_t: pl.DataFrame,
    actions_set: set[tuple[int, int, int]],
    planet_idx: dict[int, int],
    game_step: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Enumerate all (owner-0 src, other tgt) pairs and assign binary labels.

    Args:
        df_s_at_t: df_s rows where step == game_step.
        actions_set: set of (game_step, id_src, id_tgt) from actions.parquet.
        planet_idx: planet_id -> Planet node row index.
        game_step: current game step.

    Returns:
        (src_indices, tgt_indices, labels) tensors of shape (N_pairs,).
        Empty tensors if no owner-0 planet exists.
    """
    rows = df_s_at_t.select(["id", "owner"]).to_dicts()
    src_planets = [r["id"] for r in rows if r["owner"] == 0 and r["id"] in planet_idx]
    all_planets = [r["id"] for r in rows if r["id"] in planet_idx]

    src_list:   list[int]   = []
    tgt_list:   list[int]   = []
    label_list: list[float] = []

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


def _apply_transform(df_s: pl.DataFrame, transform: str) -> pl.DataFrame:
    """Apply a board rotation to the x/y columns of df_s. Sun at (50,50) is preserved."""
    if transform == "identity":
        return df_s
    elif transform == "rot90":
        return df_s.with_columns([
            (100.0 - pl.col("y")).alias("x"),
            pl.col("x").alias("y"),
        ])
    elif transform == "rot180":
        return df_s.with_columns([
            (100.0 - pl.col("x")).alias("x"),
            (100.0 - pl.col("y")).alias("y"),
        ])
    elif transform == "rot270":
        return df_s.with_columns([
            pl.col("y").alias("x"),
            (100.0 - pl.col("x")).alias("y"),
        ])
    else:
        raise ValueError(f"Unknown transform: {transform!r}")


def train_episode(
    ep_dir: Path,
    model: OrbitGNN,
    transform: str = "identity",
) -> tuple[float, list[float], list[float]]:
    """Load one episode and accumulate gradients across all game steps.

    Caller must call optimizer.zero_grad() before and optimizer.step() after.

    Returns:
        (avg_loss, all_sigmoid_scores, all_labels) across all steps in this episode.
    """
    df_s    = pl.read_parquet(ep_dir / "df_s.parquet")
    reach   = pl.read_parquet(ep_dir / "reach.parquet")
    actions = pl.read_parquet(ep_dir / "actions.parquet")
    df_s = _apply_transform(df_s, transform)

    # Keep earliest arrival step per (id_src, step_src, id, ships_sent)
    reach = (
        reach
        .group_by(["id_src", "step_src", "id", "ships_sent"])
        .agg(pl.all().sort_by("step").first())
    )

    actions_set: set[tuple[int, int, int]] = set(
        zip(
            actions["game_step"].to_list(),
            actions["id_src"].to_list(),
            actions["id"].to_list(),
        )
    )

    all_scores: list[float] = []
    all_labels: list[float] = []
    total_loss = 0.0
    n_steps    = 0

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

        loss.backward()  # accumulate gradients; caller steps optimizer once per episode

        total_loss += loss.item()
        n_steps    += 1

        with torch.no_grad():
            all_scores.extend(torch.sigmoid(logits).tolist())
            all_labels.extend(labels.tolist())

    return total_loss / max(n_steps, 1), all_scores, all_labels


def evaluate_episode(
    ep_dir: Path,
    model: OrbitGNN,
) -> tuple[float, list[float], list[float]]:
    """Evaluate one episode without gradients. Returns (avg_loss, scores, labels)."""
    df_s    = pl.read_parquet(ep_dir / "df_s.parquet")
    reach   = pl.read_parquet(ep_dir / "reach.parquet")
    actions = pl.read_parquet(ep_dir / "actions.parquet")

    reach = (
        reach
        .group_by(["id_src", "step_src", "id", "ships_sent"])
        .agg(pl.all().sort_by("step").first())
    )

    actions_set: set[tuple[int, int, int]] = set(
        zip(
            actions["game_step"].to_list(),
            actions["id_src"].to_list(),
            actions["id"].to_list(),
        )
    )

    all_scores: list[float] = []
    all_labels: list[float] = []
    total_loss = 0.0
    n_steps    = 0

    model.eval()
    with torch.no_grad():
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

            total_loss += loss.item()
            n_steps    += 1
            all_scores.extend(torch.sigmoid(logits).tolist())
            all_labels.extend(labels.tolist())

    return total_loss / max(n_steps, 1), all_scores, all_labels


def _test_orbit_gnn():
    df_s_t = pl.DataFrame({
        "id":         [0,     1,     0,     1    ],
        "step":       [10,    10,    12,    12   ],
        "x":          [30.0,  70.0,  30.1,  70.1 ],
        "y":          [30.0,  70.0,  30.1,  70.1 ],
        "ships":      [10,    5,     11,    5    ],
        "owner":      [0,     1,     0,     1    ],
        "production": [2,     1,     2,     1    ],
        "nature":     ["fix", "fix", "fix", "fix"],
    })
    reach_t = pl.DataFrame({
        "id_src": [0], "step_src": [10], "id": [1], "step": [12], "ships_sent": [4],
    })
    data, planet_idx = build_graph(df_s_t, reach_t)
    model    = OrbitGNN(hidden_dim=8, num_layers=2)
    h_planet = model.encode(data)
    assert h_planet.shape == (2, 8), f"h_planet: {h_planet.shape}"
    scores = model.score_pairs(h_planet, torch.tensor([0]), torch.tensor([1]))
    assert scores.shape == (1,), f"scores: {scores.shape}"
    print("_test_orbit_gnn PASSED")


def _test_train_episode():
    ep_dirs = sorted([d for d in PRECOMPUTE_DIR.iterdir() if d.is_dir()])
    if not ep_dirs:
        print("_test_train_episode SKIP (no episodes in 114-precompute/)")
        return
    model     = OrbitGNN(hidden_dim=8, num_layers=1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=WEIGHT_DECAY)
    optimizer.zero_grad()
    loss, scores, labels = train_episode(ep_dirs[0], model)
    optimizer.step()
    assert isinstance(loss, float)
    assert len(scores) == len(labels)
    n_pos = int(sum(labels))
    print(f"_test_train_episode PASSED  loss={loss:.4f}  pairs={len(scores)}  positives={n_pos}")


def _test_build_attack_pairs():
    df_s_at_t = pl.DataFrame({
        "id":    [0, 1, 2],
        "step":  [10, 10, 10],
        "owner": [0, 1, -1],
    })
    planet_idx  = {0: 0, 1: 1, 2: 2}
    actions_set = {(10, 0, 1)}  # only src=0 → tgt=1 is a real action
    src_idx, tgt_idx, labels = build_attack_pairs(df_s_at_t, actions_set, planet_idx, game_step=10)
    # owner-0 is planet 0; targets are planets 1 and 2 → 2 pairs
    assert len(src_idx) == 2,           f"expected 2 pairs, got {len(src_idx)}"
    assert labels.sum().item() == 1.0,  f"expected 1 positive, got {labels.sum()}"
    print("_test_build_attack_pairs PASSED")


def _test_apply_transform():
    df = pl.DataFrame({"x": [0.0, 100.0, 50.0], "y": [0.0, 100.0, 50.0],
                       "id": [0, 1, 2], "step": [1, 1, 1]})
    # rot90: new_x = 100-y, new_y = x
    r = _apply_transform(df, "rot90")
    assert r["x"].to_list() == [100.0, 0.0, 50.0], r["x"].to_list()
    assert r["y"].to_list() == [0.0, 100.0, 50.0], r["y"].to_list()
    # rot180: new_x = 100-x, new_y = 100-y
    r = _apply_transform(df, "rot180")
    assert r["x"].to_list() == [100.0, 0.0, 50.0], r["x"].to_list()
    assert r["y"].to_list() == [100.0, 0.0, 50.0], r["y"].to_list()
    # rot270: new_x = y, new_y = 100-x
    r = _apply_transform(df, "rot270")
    assert r["x"].to_list() == [0.0, 100.0, 50.0], r["x"].to_list()
    assert r["y"].to_list() == [100.0, 0.0, 50.0], r["y"].to_list()
    print("_test_apply_transform PASSED")


def _compute_metrics(scores: list[float], labels: list[float]) -> dict:
    """Compute ROC-AUC, accuracy, TP, TN, pos count, neg count at threshold 0.5."""
    n_pos = int(sum(labels))
    n_neg = len(labels) - n_pos

    if len(set(labels)) > 1:
        auc = roc_auc_score(labels, scores)
    else:
        auc = float("nan")

    tp = int(sum(1 for s, l in zip(scores, labels) if s >= 0.5 and l == 1.0))
    tn = int(sum(1 for s, l in zip(scores, labels) if s <  0.5 and l == 0.0))
    acc = (tp + tn) / max(len(labels), 1)

    return {"auc": auc, "acc": acc, "n_pos": n_pos, "n_neg": n_neg, "tp": tp, "tn": tn}


def _log(msg: str, log_fh=None) -> None:
    """Print with flush and optionally mirror to an explicit log file."""
    print(msg, flush=True)
    if log_fh is not None:
        log_fh.write(msg + "\n")
        log_fh.flush()


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    log_fh = open(OUT_DIR / "metrics.log", "a", encoding="utf-8")

    ep_dirs    = sorted([d for d in PRECOMPUTE_DIR.iterdir() if d.is_dir()])
    train_dirs = [d for d in ep_dirs if int(d.name) not in TEST_EPISODE_IDS]
    test_dirs  = [d for d in ep_dirs if int(d.name) in     TEST_EPISODE_IDS]

    n_pairs = len(train_dirs) * len(TRANSFORMS)
    _log(f"Episodes: {len(ep_dirs)} total | train={len(train_dirs)} test={len(test_dirs)}", log_fh)
    _log(f"Pairs/epoch: {len(train_dirs)} × {len(TRANSFORMS)} = {n_pairs} (all, shuffled)", log_fh)

    all_pairs = [(ep_dir, t) for ep_dir in train_dirs for t in TRANSFORMS]

    model     = OrbitGNN(hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=N_EPOCHS, eta_min=1e-6)

    best_test_auc = float("-inf")

    mlflow.set_experiment("119-NewGNN_Cosine")
    with mlflow.start_run():
        mlflow.log_params({
            "hidden_dim":      HIDDEN_DIM,
            "num_layers":      NUM_LAYERS,
            "lr":              LR,
            "weight_decay":    WEIGHT_DECAY,
            "n_epochs":        N_EPOCHS,
            "train_episodes":  len(train_dirs),
            "test_episodes":   len(test_dirs),
            "transforms":      len(TRANSFORMS),
            "pairs_per_epoch": n_pairs,
        })

        for epoch in range(1, N_EPOCHS + 1):
            # ── Training ────────────────────────────────────────────────────
            epoch_pairs = all_pairs.copy()
            random.shuffle(epoch_pairs)

            tr_scores: list[float] = []
            tr_labels: list[float] = []
            tr_loss_sum = 0.0
            tr_steps    = 0

            model.train()
            for ep_dir, transform in epoch_pairs:
                try:
                    optimizer.zero_grad()
                    loss, scores, labels = train_episode(ep_dir, model, transform)
                    optimizer.step()
                    tr_loss_sum += loss
                    tr_steps    += 1
                    tr_scores.extend(scores)
                    tr_labels.extend(labels)
                except Exception as e:
                    _log(f"  SKIP train {ep_dir.name}/{transform}: {e}", log_fh)

            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]

            # ── Test evaluation ─────────────────────────────────────────────
            te_scores: list[float] = []
            te_labels: list[float] = []
            te_loss_sum = 0.0
            te_steps    = 0

            for ep_dir in test_dirs:
                try:
                    loss, scores, labels = evaluate_episode(ep_dir, model)
                    te_loss_sum += loss
                    te_steps    += 1
                    te_scores.extend(scores)
                    te_labels.extend(labels)
                except Exception as e:
                    _log(f"  SKIP test  {ep_dir.name}: {e}", log_fh)

            # ── Metrics ─────────────────────────────────────────────────────
            avg_tr = tr_loss_sum / max(tr_steps, 1)
            avg_te = te_loss_sum / max(te_steps, 1)
            tr_m   = _compute_metrics(tr_scores, tr_labels)
            te_m   = _compute_metrics(te_scores, te_labels)

            def _fmt(m: dict, loss: float) -> str:
                auc = f"{m['auc']:.4f}" if not math.isnan(m['auc']) else "  nan"
                return (f"loss={loss:.4f}  auc={auc}  acc={m['acc']:.4f}"
                        f"  pos={m['n_pos']}  neg={m['n_neg']}  tp={m['tp']}  tn={m['tn']}")

            _log(f"Epoch {epoch:3d}  lr={current_lr:.2e}", log_fh)
            _log(f"  [train] {_fmt(tr_m, avg_tr)}", log_fh)
            _log(f"  [test]  {_fmt(te_m, avg_te)}", log_fh)

            mlflow.log_metrics({
                "train_loss": avg_tr,
                "train_auc":  tr_m["auc"]  if not math.isnan(tr_m["auc"])  else 0.0,
                "train_acc":  tr_m["acc"],
                "train_tp":   tr_m["tp"],
                "train_tn":   tr_m["tn"],
                "test_loss":  avg_te,
                "test_auc":   te_m["auc"]  if not math.isnan(te_m["auc"])  else 0.0,
                "test_acc":   te_m["acc"],
                "test_tp":    te_m["tp"],
                "test_tn":    te_m["tn"],
                "lr":         current_lr,
            }, step=epoch)

            # ── Checkpoints ─────────────────────────────────────────────────
            ckpt = OUT_DIR / f"model_epoch{epoch}.pt"
            torch.save(model.state_dict(), ckpt)
            mlflow.log_artifact(str(ckpt), artifact_path="checkpoints")
            _log(f"  -> saved {ckpt}", log_fh)

            if not math.isnan(te_m["auc"]) and te_m["auc"] > best_test_auc:
                best_test_auc = te_m["auc"]
                best_path = OUT_DIR / "best_model.pt"
                torch.save(model.state_dict(), best_path)
                _log(f"  -> NEW BEST test_auc={best_test_auc:.4f}  saved {best_path}", log_fh)

    _log(f"\nDone. Best test AUC: {best_test_auc:.4f}  Models in {OUT_DIR}/", log_fh)
    log_fh.close()


def _test_build_graph():
    df_s_t = pl.DataFrame({
        "id":         [0,     1,     0,     1    ],
        "step":       [10,    10,    12,    12   ],
        "x":          [30.0,  70.0,  30.1,  70.1 ],
        "y":          [30.0,  70.0,  30.1,  70.1 ],
        "ships":      [10,    5,     11,    5    ],
        "owner":      [0,     1,     0,     1    ],
        "production": [2,     1,     2,     1    ],
        "nature":     ["fix", "fix", "fix", "fix"],
    })
    reach_t = pl.DataFrame({
        "id_src":     [0],
        "step_src":   [10],
        "id":         [1],
        "step":       [12],
        "ships_sent": [4],
    })
    data, planet_idx = build_graph(df_s_t, reach_t)
    assert data["planet"].x.shape == (2, 4),       f"planet.x: {data['planet'].x.shape}"
    assert data["planet_step"].x.shape == (4, 9),  f"planet_step.x: {data['planet_step'].x.shape}"
    assert planet_idx == {0: 0, 1: 1}
    snap = data["planet", "has_snapshot", "planet_step"].edge_index
    assert snap.shape[1] == 4, f"has_snapshot edges: {snap.shape}"
    reaches_ei = data["planet_step", "reaches", "planet_step"].edge_index
    assert reaches_ei.shape == (2, 2), f"reaches edge_index: {reaches_ei.shape}"
    reaches_ea = data["planet_step", "reaches", "planet_step"].edge_attr
    assert reaches_ea.shape == (2, 1), f"reaches edge_attr: {reaches_ea.shape}"
    print("_test_build_graph PASSED")


if __name__ == "__main__":
    _test_build_graph()
    _test_orbit_gnn()
    _test_build_attack_pairs()
    _test_train_episode()
    print("\nAll tests passed. Starting training...\n")
    main()
