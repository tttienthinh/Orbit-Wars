"""117-NewGNN — Two-tower GNN: predict winner attack (id_src, id_tgt) per game step."""
import math
import random
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

PRECOMPUTE_DIR = Path("114-precompute")
OUT_DIR        = Path("117-NewGNN")
NB_STEPS_SIM   = 20
N_EPOCHS       = 10
HIDDEN_DIM     = 64
NUM_LAYERS     = 3
LR             = 1e-3
TRAIN_RATIO    = 0.8


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
