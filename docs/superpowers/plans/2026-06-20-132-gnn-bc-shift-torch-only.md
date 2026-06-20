# 132-GNN_BC_shift_torch_only Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `132-GNN_BC_shift_torch_only.py` — a pure-PyTorch Kaggle submission agent that runs the trained `GNNBehaviourCloning` model from `130-GNN_BC_shift.py` with no torch_geometric dependency.

**Architecture:** Single self-contained Python file following the `112-GNN_torch_only.py` pattern. Re-implements PyG's HeteroConv/SAGEConv/ScaledSAGEConv in vanilla PyTorch with matching parameter names for direct weight loading. At each game step, builds a heterogeneous graph from live obs via the same pipeline as `126-precompute.py`, runs the model, and selects attacks where logit > log(576) (calibrated to match training positive rate).

**Tech Stack:** Python 3, PyTorch, Polars, NumPy, math, copy, types, pathlib

## Global Constraints

- No torch_geometric import anywhere in the output file
- Output file: `132-GNN_BC_shift_torch_only.py` (root of project)
- Weights: `model_epoch100.pt` — copy from `130-GNN_BC_shift/model_epoch100.pt` to same dir as script
- Model hyperparameters: H=64, L1=3, L2=2 (must not be changed)
- REACH_SHIPS = [8, 64, 512] (matches 130-GNN_BC_shift.py training filter)
- NB_STEPS_SIM = 20 (must match training)
- LOGIT_THRESHOLD = math.log(576.0) ≈ 6.358  (calibrated from pos_weight=576)
- All physics constants verbatim from 112-GNN_torch_only.py: CENTER=50.0, SUN_RADIUS=10.0, ROTATION_RADIUS_LIMIT=50.0, MAX_SPEED=6.0, PLANET_MARGIN=0.1, PLANET_MOVEMENT_SLACK=3.0, BOARD_SIZE=100.0

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `132-GNN_BC_shift_torch_only.py` | Create | Complete agent: model + pipeline + agent() |
| `model_epoch100.pt` | Copy | Weights to bundle alongside script |

---

## Task 1: Pure PyTorch Model + Weight Loading

**Files:**
- Create: `132-GNN_BC_shift_torch_only.py` (first section — model only, no pipeline yet)

**Interfaces:**
- Produces:
  - `GNNBehaviourCloningPure(H=64, L1=3, L2=2)` — `forward(data) -> Tensor (N_ams,)`
  - `_load_weights(model, path) -> (missing_keys, unexpected_keys)`
  - `data` is a `types.SimpleNamespace` with attributes: `planet_x`, `ps_x`, `ams_x`, `ei_has_ps`, `ei_rev_has_ps`, `ei_reaches`, `reaches_attr`, `ei_src_of`, `ei_rev_src_of`, `ei_tgt_of`, `ei_rev_tgt_of`

- [ ] **Step 1: Write the smoke test** (append to end of file as `__main__` block)

At the end of `132-GNN_BC_shift_torch_only.py`, add:

```python
if __name__ == "__main__":
    import types, math
    _test_model = GNNBehaviourCloningPure(H=64, L1=3, L2=2)
    _weights_path = Path("130-GNN_BC_shift") / "model_epoch100.pt"
    missing, unexpected = _load_weights(_test_model, _weights_path)
    assert not missing, f"Missing keys: {missing}"
    assert not unexpected, f"Unexpected keys: {unexpected}"
    print("Weight loading: OK")

    # Minimal synthetic graph: 3 planets, 6 ps nodes, 2 action_max nodes
    N_p, N_ps, N_ams = 3, 6, 2
    data = types.SimpleNamespace(
        planet_x     = torch.randn(N_p, 4),
        ps_x         = torch.randn(N_ps, 9),
        ams_x        = torch.randn(N_ams, 1),
        ei_has_ps    = torch.tensor([[0,0,1,1,2,2],[0,1,2,3,4,5]], dtype=torch.long),
        ei_rev_has_ps= torch.tensor([[0,1,2,3,4,5],[0,0,1,1,2,2]], dtype=torch.long),
        ei_reaches   = torch.tensor([[0,2],[1,3]], dtype=torch.long),
        reaches_attr = torch.tensor([0.3, 0.7]),
        ei_src_of    = torch.tensor([[0,1],[0,1]], dtype=torch.long),
        ei_rev_src_of= torch.tensor([[0,1],[0,1]], dtype=torch.long),
        ei_tgt_of    = torch.tensor([[2,3],[0,1]], dtype=torch.long),
        ei_rev_tgt_of= torch.tensor([[0,1],[2,3]], dtype=torch.long),
    )
    _test_model.eval()
    with torch.no_grad():
        logits = _test_model(data)
    assert logits.shape == (N_ams,), f"Expected ({N_ams},), got {logits.shape}"
    print(f"Forward pass: OK  logits={logits.tolist()}")
    print("All checks passed.")
```

- [ ] **Step 2: Write `132-GNN_BC_shift_torch_only.py` — model section**

Create the file with all imports and the model:

```python
"""132-GNN_BC_shift_torch_only — pure-PyTorch inference agent for GNNBehaviourCloning.

Architecture matches 130-GNN_BC_shift.py (H=64, L1=3, L2=2).
Weights loaded from model_epoch100.pt via key remapping from PyG HeteroConv format.
No torch_geometric dependency.
"""
import math
import copy
import types
from pathlib import Path

import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Constants ─────────────────────────────────────────────────────────────────
NB_STEPS_SIM     = 20
REACH_SHIPS      = [8, 64, 512]
HIDDEN_DIM       = 64
NUM_LAYERS_P1    = 3
NUM_LAYERS_P2    = 2
LOG_1024         = math.log(1024.0)
POS_WEIGHT       = 576.0
LOGIT_THRESHOLD  = math.log(POS_WEIGHT)  # ≈ 6.358; calibrated to training positive rate


# ── Aggregation helpers ───────────────────────────────────────────────────────

def _mean_agg(h_src: torch.Tensor, edge_index: torch.Tensor, num_dst: int) -> torch.Tensor:
    src_i, dst_i = edge_index
    D = h_src.shape[1]
    agg = h_src.new_zeros(num_dst, D)
    cnt = h_src.new_zeros(num_dst, 1)
    agg.scatter_add_(0, dst_i.unsqueeze(1).expand(-1, D), h_src[src_i])
    cnt.scatter_add_(0, dst_i.unsqueeze(1), torch.ones(len(dst_i), 1, device=h_src.device))
    return agg / cnt.clamp(min=1)


def _scaled_mean_agg(
    h: torch.Tensor,
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
    num_nodes: int,
) -> torch.Tensor:
    src_i, dst_i = edge_index
    D = h.shape[1]
    msgs = h[src_i] * edge_attr.view(-1, 1)
    agg = h.new_zeros(num_nodes, D)
    cnt = h.new_zeros(num_nodes, 1)
    agg.scatter_add_(0, dst_i.unsqueeze(1).expand(-1, D), msgs)
    cnt.scatter_add_(0, dst_i.unsqueeze(1), torch.ones(len(dst_i), 1, device=h.device))
    return agg / cnt.clamp(min=1)


# ── Pure-PyTorch conv layers (parameter names match PyG for direct weight load) ─

class _SAGEConvPure(nn.Module):
    """Matches PyG SAGEConv((H,H), H): lin_l=dst root, lin_r=neighbor mean-agg."""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.lin_l = nn.Linear(in_ch, out_ch, bias=True)
        self.lin_r = nn.Linear(in_ch, out_ch, bias=False)

    def forward(
        self,
        h_src: torch.Tensor,
        h_dst: torch.Tensor,
        edge_index: torch.Tensor,
        num_dst: int,
    ) -> torch.Tensor:
        return self.lin_l(h_dst) + self.lin_r(_mean_agg(h_src, edge_index, num_dst))


class _ScaledSAGEConvPure(nn.Module):
    """Matches PyG ScaledSAGEConv: lin_neigh=weighted-mean-agg, lin_root=self."""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.lin_neigh = nn.Linear(in_ch, out_ch, bias=False)
        self.lin_root  = nn.Linear(in_ch, out_ch, bias=True)

    def forward(
        self,
        h: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        num_nodes: int,
    ) -> torch.Tensor:
        agg = _scaled_mean_agg(h, edge_index, edge_attr, num_nodes)
        return self.lin_neigh(agg) + self.lin_root(h)


class _Phase1ConvPure(nn.Module):
    """One HeteroConv layer on {planet, planet_step}. Attributes named to match PyG HeteroConv."""
    def __init__(self, H: int):
        super().__init__()
        self.has_ps     = _SAGEConvPure(H, H)
        self.rev_has_ps = _SAGEConvPure(H, H)
        self.reaches    = _ScaledSAGEConvPure(H, H)

    def forward(
        self,
        h_planet: torch.Tensor,
        h_ps: torch.Tensor,
        ei_has_ps: torch.Tensor,
        ei_rev_has_ps: torch.Tensor,
        ei_reaches: torch.Tensor,
        reaches_attr: torch.Tensor,
    ):
        n_planet = h_planet.shape[0]
        n_ps     = h_ps.shape[0]
        # planet_step: sum contributions from has_ps + reaches (HeteroConv aggr="sum")
        new_ps = self.has_ps(h_planet, h_ps, ei_has_ps, n_ps)
        if ei_reaches.shape[1] > 0:
            new_ps = new_ps + self.reaches(h_ps, ei_reaches, reaches_attr, n_ps)
        # planet: sole contribution from rev_has_ps
        new_planet = self.rev_has_ps(h_ps, h_planet, ei_rev_has_ps, n_planet)
        return new_planet, new_ps


class _Phase2ConvPure(nn.Module):
    """One HeteroConv layer on {planet_step, action_max}. Attributes named to match PyG."""
    def __init__(self, H: int):
        super().__init__()
        self.src_of     = _SAGEConvPure(H, H)
        self.rev_src_of = _SAGEConvPure(H, H)
        self.tgt_of     = _SAGEConvPure(H, H)
        self.rev_tgt_of = _SAGEConvPure(H, H)

    def forward(
        self,
        h_ps: torch.Tensor,
        h_ams: torch.Tensor,
        ei_src_of: torch.Tensor,
        ei_rev_src_of: torch.Tensor,
        ei_tgt_of: torch.Tensor,
        ei_rev_tgt_of: torch.Tensor,
    ):
        n_ps  = h_ps.shape[0]
        n_ams = h_ams.shape[0]
        # action_max: sum from src_of + tgt_of
        new_ams = (
            self.src_of(h_ps, h_ams, ei_src_of, n_ams) +
            self.tgt_of(h_ps, h_ams, ei_tgt_of, n_ams)
        )
        # planet_step: sum from rev_src_of + rev_tgt_of
        new_ps = (
            self.rev_src_of(h_ams, h_ps, ei_rev_src_of, n_ps) +
            self.rev_tgt_of(h_ams, h_ps, ei_rev_tgt_of, n_ps)
        )
        return new_ps, new_ams


class GNNBehaviourCloningPure(nn.Module):
    """Pure-PyTorch equivalent of GNNBehaviourCloning from 130-GNN_BC_shift.py.

    Input `data` must be a SimpleNamespace with attributes:
        planet_x, ps_x, ams_x          — feature tensors
        ei_has_ps, ei_rev_has_ps        — planet<->planet_step edge indices
        ei_reaches, reaches_attr        — planet_step self-reach edges + weights
        ei_src_of, ei_rev_src_of        — planet_step->action_max (source)
        ei_tgt_of, ei_rev_tgt_of        — planet_step->action_max (target)
    """
    def __init__(self, H: int = 64, L1: int = 3, L2: int = 2):
        super().__init__()
        self.proj_planet  = nn.Linear(4, H)
        self.proj_ps      = nn.Linear(9, H)
        self.proj_ams     = nn.Linear(1, H)
        self.phase1       = nn.ModuleList([_Phase1ConvPure(H) for _ in range(L1)])
        self.norm1_planet = nn.ModuleList([nn.LayerNorm(H) for _ in range(L1)])
        self.norm1_ps     = nn.ModuleList([nn.LayerNorm(H) for _ in range(L1)])
        self.phase2       = nn.ModuleList([_Phase2ConvPure(H) for _ in range(L2)])
        self.norm2_ps     = nn.ModuleList([nn.LayerNorm(H) for _ in range(L2)])
        self.norm2_ams    = nn.ModuleList([nn.LayerNorm(H) for _ in range(L2)])
        self.classifier   = nn.Linear(H, 1)

    def forward(self, data) -> torch.Tensor:
        h_planet = self.proj_planet(data.planet_x)
        h_ps     = self.proj_ps(data.ps_x)
        h_ams    = self.proj_ams(data.ams_x)

        for i, conv in enumerate(self.phase1):
            h_planet, h_ps = conv(
                h_planet, h_ps,
                data.ei_has_ps, data.ei_rev_has_ps,
                data.ei_reaches, data.reaches_attr,
            )
            h_planet = F.relu(self.norm1_planet[i](h_planet))
            h_ps     = F.relu(self.norm1_ps[i](h_ps))

        for i, conv in enumerate(self.phase2):
            h_ps, h_ams = conv(
                h_ps, h_ams,
                data.ei_src_of, data.ei_rev_src_of,
                data.ei_tgt_of, data.ei_rev_tgt_of,
            )
            h_ps  = F.relu(self.norm2_ps[i](h_ps))
            h_ams = F.relu(self.norm2_ams[i](h_ams))

        return self.classifier(h_ams).squeeze(-1)


# ── Weight loading ────────────────────────────────────────────────────────────

_KEY_MAP = {
    "convs.<planet___has_ps___planet_step>":         "has_ps",
    "convs.<planet_step___rev_has_ps___planet>":     "rev_has_ps",
    "convs.<planet_step___reaches___planet_step>":   "reaches",
    "convs.<planet_step___src_of___action_max>":     "src_of",
    "convs.<action_max___rev_src_of___planet_step>": "rev_src_of",
    "convs.<planet_step___tgt_of___action_max>":     "tgt_of",
    "convs.<action_max___rev_tgt_of___planet_step>": "rev_tgt_of",
}


def _load_weights(model: nn.Module, path) -> tuple:
    sd = torch.load(str(path), map_location="cpu", weights_only=True)
    new_sd = {}
    for k, v in sd.items():
        new_k = k
        for pyg_name, our_name in _KEY_MAP.items():
            new_k = new_k.replace(pyg_name, our_name)
        new_sd[new_k] = v
    missing, unexpected = model.load_state_dict(new_sd, strict=True)
    return list(missing), list(unexpected)
```

- [ ] **Step 3: Run the smoke test**

```
cd "C:\Users\trant\Documents\Programmation\Orbit Wars"
python 132-GNN_BC_shift_torch_only.py
```

Expected output (no assertion errors):
```
Weight loading: OK
Forward pass: OK  logits=[..., ...]
All checks passed.
```

If you see `Missing keys: [...]`, the `_KEY_MAP` remapping has a typo — compare against the actual keys by printing `list(sd.keys())[:10]` before the remapping.

- [ ] **Step 4: Commit**

```bash
git add 132-GNN_BC_shift_torch_only.py
git commit -m "feat: 132 — pure-PyTorch GNNBehaviourCloning model + weight loading"
```

---

## Task 2: Physics Pipeline + Reachability Functions

**Files:**
- Modify: `132-GNN_BC_shift_torch_only.py` (append physics + pipeline section)

**Interfaces:**
- Consumes: existing imports at top of file
- Produces:
  - `GameConfig` class with constants
  - `_interpreter(obs, actions, step, num_agents)` — modifies obs in-place
  - `_01_get_obs_dataframe(obs, step, num_agents) -> (pl.DataFrame, pl.DataFrame)` — (df_s, planet_disp)
  - `_00_remap_owner(df_s, obs, player_id) -> pl.DataFrame`
  - `_02_get_reach(df_s, planet_disp, ships_list) -> pl.LazyFrame`
  - `_get_reach_max_ships(df_s, planet_disp) -> pl.LazyFrame`
  - `_03_filter_collision(pa_lf) -> pl.LazyFrame`

- [ ] **Step 1: Append physics + pipeline to `132-GNN_BC_shift_torch_only.py`**

After the weight-loading section, append:

```python
# ── Game constants ────────────────────────────────────────────────────────────

CENTER                 = 50.0
SUN_RADIUS             = 10.0
ROTATION_RADIUS_LIMIT  = 50.0
MAX_SPEED              = 6.0
PLANET_MARGIN          = 0.1
PLANET_MOVEMENT_SLACK  = 3.0
BOARD_SIZE             = 100.0
MAX_NB_STEP            = 500


def _fleet_speed(ships: float) -> float:
    if ships <= 1:
        return 1.0
    ratio = math.log(ships) / math.log(1000.0)
    return 1.0 + (MAX_SPEED - 1.0) * max(0.0, min(1.0, ratio)) ** 1.5


# ── Interpreter (verbatim from 112-GNN_torch_only.py) ────────────────────────

def _interpreter(obs, actions, step, num_agents=2):
    obs0 = obs

    expired_comet_pids = []
    for group in obs0.comets:
        idx = group["path_index"]
        for i, pid in enumerate(group["planet_ids"]):
            if idx >= len(group["paths"][i]):
                expired_comet_pids.append(pid)
    if expired_comet_pids:
        expired_set = set(expired_comet_pids)
        obs0.planets = [p for p in obs0.planets if p[0] not in expired_set]
        obs0.initial_planets = [p for p in obs0.initial_planets if p[0] not in expired_set]
        obs0.comet_planet_ids = [pid for pid in obs0.comet_planet_ids if pid not in expired_set]
        for group in obs0.comets:
            group["planet_ids"] = [pid for pid in group["planet_ids"] if pid not in expired_set]
        obs0.comets = [g for g in obs0.comets if g["planet_ids"]]

    def process_moves(player_id, action):
        if not action or not isinstance(action, list):
            return
        for move in action:
            if len(move) != 3:
                continue
            from_id, angle, ships = move
            ships = int(ships)
            from_planet = next((p for p in obs0.planets if p[0] == from_id), None)
            if from_planet and from_planet[1] == player_id:
                if from_planet[5] >= ships and ships > 0:
                    from_planet[5] -= ships
                    start_x = from_planet[2] + math.cos(angle) * (from_planet[4] + 0.1)
                    start_y = from_planet[3] + math.sin(angle) * (from_planet[4] + 0.1)
                    obs0.fleets.append([
                        obs0.next_fleet_id, player_id,
                        start_x, start_y, angle, from_id, ships,
                    ])
                    obs0.next_fleet_id += 1

    for i in range(num_agents):
        process_moves(i, actions[i])

    for planet in obs0.planets:
        if planet[1] != -1:
            planet[5] += planet[6]

    fleets_to_remove = []
    combat_lists = {p[0]: [] for p in obs0.planets}
    angular_velocity = obs0.angular_velocity
    comet_pid_set = set(obs0.comet_planet_ids)
    initial_by_id = {p[0]: p for p in obs0.initial_planets}

    planet_paths = {}
    for planet in obs0.planets:
        if planet[0] in comet_pid_set:
            continue
        p_old = (planet[2], planet[3])
        p_new = p_old
        initial_p = initial_by_id.get(planet[0])
        if initial_p is not None:
            dx_p = initial_p[2] - CENTER
            dy_p = initial_p[3] - CENTER
            r_p  = math.sqrt(dx_p**2 + dy_p**2)
            if r_p + planet[4] < ROTATION_RADIUS_LIMIT:
                initial_angle  = math.atan2(dy_p, dx_p)
                current_angle  = initial_angle + angular_velocity * step
                p_new = (
                    CENTER + r_p * math.cos(current_angle),
                    CENTER + r_p * math.sin(current_angle),
                )
        planet_paths[planet[0]] = (p_old, p_new)

    for fleet in obs0.fleets:
        angle = fleet[4]
        ships = fleet[6]
        speed = _fleet_speed(ships)
        f_old = (fleet[2], fleet[3])
        fleet[2] += math.cos(angle) * speed
        fleet[3] += math.sin(angle) * speed
        f_new = (fleet[2], fleet[3])

        hit_planet = False
        for planet in obs0.planets:
            path = planet_paths.get(planet[0])
            if path is None:
                continue
            p_old, p_new = path
            d0x, d0y = f_old[0] - p_old[0], f_old[1] - p_old[1]
            dvx = (f_new[0] - f_old[0]) - (p_new[0] - p_old[0])
            dvy = (f_new[1] - f_old[1]) - (p_new[1] - p_old[1])
            a = dvx*dvx + dvy*dvy
            b = 2.0*(d0x*dvx + d0y*dvy)
            c = d0x*d0x + d0y*d0y - planet[4]**2
            if a < 1e-12:
                hit = (c <= 0.0)
            else:
                disc = b*b - 4.0*a*c
                if disc < 0.0:
                    hit = False
                else:
                    sq = math.sqrt(disc)
                    t1 = (-b - sq)/(2.0*a)
                    t2 = (-b + sq)/(2.0*a)
                    hit = t2 >= 0.0 and t1 <= 1.0
            if hit:
                combat_lists[planet[0]].append(fleet)
                fleets_to_remove.append(fleet)
                hit_planet = True
                break
        if hit_planet:
            continue
        if not (0 <= fleet[2] <= BOARD_SIZE and 0 <= fleet[3] <= BOARD_SIZE):
            fleets_to_remove.append(fleet)
            continue
        l2 = (CENTER - f_old[0])**2 + (CENTER - f_old[1])**2
        dx_s = f_new[0] - f_old[0]; dy_s = f_new[1] - f_old[1]
        l2_seg = dx_s**2 + dy_s**2
        if l2_seg > 0:
            t = max(0, min(1, ((CENTER - f_old[0])*dx_s + (CENTER - f_old[1])*dy_s)/l2_seg))
            proj = (f_old[0] + t*dx_s, f_old[1] + t*dy_s)
            seg_dist = math.sqrt((CENTER - proj[0])**2 + (CENTER - proj[1])**2)
        else:
            seg_dist = math.sqrt(l2)
        if seg_dist < SUN_RADIUS:
            fleets_to_remove.append(fleet)
            continue

    for planet in obs0.planets:
        path = planet_paths.get(planet[0])
        if path is not None:
            planet[2], planet[3] = path[1]

    expired_comet_pids = []
    for group in obs0.comets:
        group["path_index"] += 1
        idx = group["path_index"]
        for i, pid in enumerate(group["planet_ids"]):
            planet = next((p for p in obs0.planets if p[0] == pid), None)
            if planet is None:
                continue
            p_path = group["paths"][i]
            if idx >= len(p_path):
                expired_comet_pids.append(pid)
            else:
                c_old = (planet[2], planet[3])
                planet[2] = p_path[idx][0]
                planet[3] = p_path[idx][1]
                if c_old[0] >= 0:
                    c_new = (planet[2], planet[3])
                    for fleet in obs0.fleets:
                        if fleet not in fleets_to_remove:
                            dx_f = c_new[0]-c_old[0]; dy_f = c_new[1]-c_old[1]
                            l2_f = dx_f**2 + dy_f**2
                            if l2_f > 0:
                                tf = max(0, min(1, ((fleet[2]-c_old[0])*dx_f + (fleet[3]-c_old[1])*dy_f)/l2_f))
                                proj = (c_old[0]+tf*dx_f, c_old[1]+tf*dy_f)
                                d = math.sqrt((fleet[2]-proj[0])**2 + (fleet[3]-proj[1])**2)
                            else:
                                d = math.sqrt((fleet[2]-c_old[0])**2 + (fleet[3]-c_old[1])**2)
                            if d < planet[4]:
                                combat_lists[planet[0]].append(fleet)
                                fleets_to_remove.append(fleet)

    if expired_comet_pids:
        expired_set = set(expired_comet_pids)
        obs0.planets = [p for p in obs0.planets if p[0] not in expired_set]
        obs0.initial_planets = [p for p in obs0.initial_planets if p[0] not in expired_set]
        obs0.comet_planet_ids = [pid for pid in obs0.comet_planet_ids if pid not in expired_set]
        for group in obs0.comets:
            group["planet_ids"] = [pid for pid in group["planet_ids"] if pid not in expired_set]
        obs0.comets = [g for g in obs0.comets if g["planet_ids"]]

    obs0.fleets = [f for f in obs0.fleets if f not in fleets_to_remove]

    for pid_c, planet_fleets in combat_lists.items():
        planet = next((p for p in obs0.planets if p[0] == pid_c), None)
        if not planet or not planet_fleets:
            continue
        player_ships: dict = {}
        for fleet in planet_fleets:
            owner = fleet[1]
            player_ships[owner] = player_ships.get(owner, 0) + fleet[6]
        if not player_ships:
            continue
        sorted_players = sorted(player_ships.items(), key=lambda x: x[1], reverse=True)
        top_player, top_ships = sorted_players[0]
        if len(sorted_players) > 1:
            second_ships = sorted_players[1][1]
            survivor_ships = top_ships - second_ships
            survivor_owner = top_player if survivor_ships > 0 else -1
            if sorted_players[0][1] == sorted_players[1][1]:
                survivor_ships = 0
        else:
            survivor_owner = top_player
            survivor_ships = top_ships
        if survivor_ships > 0:
            if planet[1] == survivor_owner:
                planet[5] += survivor_ships
            else:
                planet[5] -= survivor_ships
                if planet[5] < 0:
                    planet[1] = survivor_owner
                    planet[5] = abs(planet[5])


# ── Observation → DataFrame ───────────────────────────────────────────────────

def _01_get_obs_dataframe(obs, step: int, num_agents: int):
    """Returns (df_s, planet_disp). df_s covers steps [step, step+NB_STEPS_SIM]."""
    sim = copy.deepcopy(obs)
    no_actions = [[] for _ in range(num_agents)]
    rows = []
    for i in range(NB_STEPS_SIM + 1):
        for p in sim.planets:
            pid, owner, x, y, radius, ships, production = (
                p[0], p[1], p[2], p[3], p[4], p[5], p[6]
            )
            r = math.hypot(x - CENTER, y - CENTER)
            if pid in sim.comet_planet_ids:
                nature = "comet"
            elif r + radius < ROTATION_RADIUS_LIMIT:
                nature = "moving"
            else:
                nature = "fix"
            rows.append({
                "step": step + i, "id": pid, "x": x, "y": y,
                "radius": radius, "ships": ships,
                "production": production, "owner": owner, "nature": nature,
            })
        _interpreter(sim, no_actions, step + i, num_agents)

    df_s = pl.DataFrame(rows).sort("step")

    prev_pos = (
        df_s.lazy().select(["id", "step", "x", "y"])
        .rename({"x": "x_prev", "y": "y_prev"})
        .with_columns((pl.col("step") + 1).alias("step"))
    )
    planet_disp = (
        df_s.lazy().select(["id", "step", "x", "y"])
        .join(prev_pos, on=["id", "step"], how="left")
        .with_columns(
            ((pl.col("x") - pl.col("x_prev").fill_null(pl.col("x"))).pow(2) +
             (pl.col("y") - pl.col("y_prev").fill_null(pl.col("y"))).pow(2)
             ).sqrt().alias("planet_disp")
        )
        .select(["id", "step", "planet_disp"])
        .collect()
    )
    return df_s, planet_disp


def _00_remap_owner(df_s: pl.DataFrame, obs, player_id: int) -> pl.DataFrame:
    ships_by_player: dict = {}
    for p in obs.planets:
        if p[1] != -1:
            ships_by_player[p[1]] = ships_by_player.get(p[1], 0) + p[5]
    for f in obs.fleets:
        ships_by_player[f[1]] = ships_by_player.get(f[1], 0) + f[6]
    opponents = sorted(
        [(pid, s) for pid, s in ships_by_player.items() if pid != player_id],
        key=lambda x: x[1], reverse=True,
    )
    id_map = {player_id: 0}
    for new_id, (old_id, _) in enumerate(opponents, start=1):
        id_map[old_id] = new_id
    return df_s.with_columns(
        pl.col("owner").map_elements(lambda x: id_map.get(x, x), return_dtype=pl.Int64)
    )


# ── Reachability: discrete ship sizes (from 113-Polars_GNN_Corrected.py) ─────

def _02_get_reach(
    df_s: pl.DataFrame,
    planet_disp: pl.DataFrame,
    ships_list: list,
) -> pl.LazyFrame:
    df_s_lf = df_s.lazy()
    planet_disp_lf = planet_disp.lazy()

    all_base_lf = (
        df_s_lf
        .group_by("id", maintain_order=True)
        .agg([
            pl.first("step").alias("step_src"),
            pl.first("x").alias("x_src"),
            pl.first("y").alias("y_src"),
            pl.first("radius").alias("radius_src"),
            pl.min("ships").alias("ships_min"),
            pl.first("production").alias("production_src"),
            pl.first("nature").alias("nature_src"),
            pl.first("owner").alias("owner_src"),
        ])
        .rename({"id": "id_src"})
    )

    dx = pl.col("x") - pl.col("x_src")
    dy = pl.col("y") - pl.col("y_src")
    l2 = dx.pow(2) + dy.pow(2)
    dist_tgt_src = l2.sqrt()
    step_diff = (pl.col("step") - pl.col("step_src")).cast(pl.Float64)

    dot = (CENTER - pl.col("x_src")) * dx + (CENTER - pl.col("y_src")) * dy
    t_sun = (dot / pl.when(l2 == 0).then(pl.lit(1.0)).otherwise(l2)).clip(0.0, 1.0)
    proj_dist_sun = (
        (CENTER - pl.col("x_src") - t_sun * dx).pow(2) +
        (CENTER - pl.col("y_src") - t_sun * dy).pow(2)
    ).sqrt()
    crossing_sun = pl.when(l2 == 0).then(
        ((CENTER - pl.col("x_src")).pow(2) + (CENTER - pl.col("y_src")).pow(2)).sqrt()
    ).otherwise(proj_dist_sun) < (SUN_RADIUS + PLANET_MARGIN)

    coarse_lf = (
        all_base_lf.join(df_s_lf, how="cross")
        .filter((pl.col("step") > pl.col("step_src")) & (pl.col("id") != pl.col("id_src")))
        .join(planet_disp_lf, on=["id", "step"], how="left")
        .with_columns([dist_tgt_src.alias("dist_tgt_src"), step_diff.alias("step_diff")])
        .filter(
            (pl.col("dist_tgt_src") < (pl.col("step_diff") + 1) * MAX_SPEED
             + pl.col("radius_src") + PLANET_MARGIN + pl.col("radius")
             + pl.col("planet_disp").fill_null(0.0))
            & ~crossing_sun
        )
        .with_columns(pl.lit(ships_list).alias("ships_sent"))
        .explode("ships_sent")
    )

    fs_expr  = 1.0 + (MAX_SPEED - 1.0) * (
        pl.col("ships_sent").cast(pl.Float64).log(base=math.e) / math.log(1000.0)
    ).clip(lower_bound=0.0).pow(1.5)
    dm_expr  = pl.col("step_diff") * fs_expr + PLANET_MARGIN + pl.col("radius_src")
    dp_expr  = dm_expr - fs_expr

    prev_pos_lf = (
        df_s_lf.select(["id", "step", "x", "y"])
        .rename({"x": "x_prev", "y": "y_prev"})
        .with_columns((pl.col("step") + 1).alias("step"))
    )

    ux = (pl.col("x") - pl.col("x_src")) / pl.when(
        pl.col("dist_tgt_src") < 1e-9
    ).then(pl.lit(1.0)).otherwise(pl.col("dist_tgt_src"))
    uy = (pl.col("y") - pl.col("y_src")) / pl.when(
        pl.col("dist_tgt_src") < 1e-9
    ).then(pl.lit(1.0)).otherwise(pl.col("dist_tgt_src"))
    fx0  = pl.col("x_src") + ux * pl.col("dist_prev")
    fy0  = pl.col("y_src") + uy * pl.col("dist_prev")
    pvx  = pl.col("x") - pl.col("x_prev").fill_null(pl.col("x"))
    pvy  = pl.col("y") - pl.col("y_prev").fill_null(pl.col("y"))
    dvx  = ux * pl.col("fleet_speed") - pvx
    dvy  = uy * pl.col("fleet_speed") - pvy
    d0x  = fx0 - pl.col("x_prev").fill_null(pl.col("x"))
    d0y  = fy0 - pl.col("y_prev").fill_null(pl.col("y"))
    a_sp = dvx.pow(2) + dvy.pow(2)
    b_sp = 2.0 * (d0x * dvx + d0y * dvy)
    c_sp = d0x.pow(2) + d0y.pow(2) - pl.col("radius").pow(2)
    disc = b_sp.pow(2) - 4.0 * a_sp * c_sp
    sq   = disc.clip(lower_bound=0.0).sqrt()
    t1e  = pl.when(a_sp < 1e-12).then(pl.lit(0.0)).otherwise((-b_sp - sq) / (2.0 * a_sp))
    t2e  = pl.when(a_sp < 1e-12).then(pl.lit(1.0)).otherwise((-b_sp + sq) / (2.0 * a_sp))
    coll = pl.when(a_sp < 1e-12).then(c_sp <= 0.0).otherwise(
        (disc >= 0.0) & (t2e >= 0.0) & (t1e <= 1.0)
    )

    xpf = pl.col("x_prev").fill_null(pl.col("x"))
    ypf = pl.col("y_prev").fill_null(pl.col("y"))

    return (
        coarse_lf
        .with_columns([fs_expr.alias("fleet_speed"), dm_expr.alias("dist_min"), dp_expr.alias("dist_prev")])
        .filter(pl.col("dist_tgt_src") < pl.col("dist_min") + pl.col("fleet_speed") + pl.col("radius") + PLANET_MOVEMENT_SLACK)
        .join(prev_pos_lf, on=["id", "step"], how="left")
        .with_columns([t1e.alias("t1"), t2e.alias("t2"), coll.alias("collision")])
        .filter(pl.col("collision"))
        .with_columns([pl.col("t1").clip(0.0, 1.0).alias("t1_eff"), pl.col("t2").clip(0.0, 1.0).alias("t2_eff")])
        .with_columns([
            (xpf + pl.col("t1_eff") * (pl.col("x") - xpf)).alias("p_t1_x"),
            (ypf + pl.col("t1_eff") * (pl.col("y") - ypf)).alias("p_t1_y"),
            (xpf + pl.col("t2_eff") * (pl.col("x") - xpf)).alias("p_t2_x"),
            (ypf + pl.col("t2_eff") * (pl.col("y") - ypf)).alias("p_t2_y"),
        ])
        .with_columns([
            pl.arctan2(pl.col("p_t1_y") - pl.col("y_src"), pl.col("p_t1_x") - pl.col("x_src")).alias("angle_t1"),
            pl.arctan2(pl.col("p_t2_y") - pl.col("y_src"), pl.col("p_t2_x") - pl.col("x_src")).alias("angle_t2"),
            ((pl.col("p_t1_x")-pl.col("x_src")).pow(2)+(pl.col("p_t1_y")-pl.col("y_src")).pow(2)).sqrt().alias("d_s_t1"),
            ((pl.col("p_t2_x")-pl.col("x_src")).pow(2)+(pl.col("p_t2_y")-pl.col("y_src")).pow(2)).sqrt().alias("d_s_t2"),
        ])
        .with_columns([
            (pl.col("dist_prev") + pl.col("t1_eff") * pl.col("fleet_speed")).alias("d_f_t1"),
            (pl.col("dist_prev") + pl.col("t2_eff") * pl.col("fleet_speed")).alias("d_f_t2"),
        ])
        .with_columns([
            ((pl.col("d_s_t1").pow(2)+pl.col("d_f_t1").pow(2)-pl.col("radius").pow(2))
             /(2.0*pl.col("d_s_t1")*pl.col("d_f_t1"))).clip(-1.0,1.0).arccos().alias("angle_radius_t1"),
            ((pl.col("d_s_t2").pow(2)+pl.col("d_f_t2").pow(2)-pl.col("radius").pow(2))
             /(2.0*pl.col("d_s_t2")*pl.col("d_f_t2"))).clip(-1.0,1.0).arccos().alias("angle_radius_t2"),
        ])
        .with_columns([
            pl.min_horizontal(
                pl.col("angle_t1") - pl.col("angle_radius_t1"),
                pl.col("angle_t2") - pl.col("angle_radius_t2"),
            ).mod(2 * math.pi).alias("angle_min"),
            pl.max_horizontal(
                pl.col("angle_t1") + pl.col("angle_radius_t1"),
                pl.col("angle_t2") + pl.col("angle_radius_t2"),
            ).mod(2 * math.pi).alias("angle_max"),
            pl.arctan2(
                pl.col("angle_t1").sin() + pl.col("angle_t2").sin(),
                pl.col("angle_t1").cos() + pl.col("angle_t2").cos(),
            ).alias("angle"),
        ])
        .sort("step")
    )


# ── Reachability: max ships at source (from 126-precompute.py) ────────────────

def _get_reach_max_ships(df_s: pl.DataFrame, planet_disp: pl.DataFrame) -> pl.LazyFrame:
    """Like _02_get_reach but ships_sent = actual ships at source; no explode."""
    df_s_lf = df_s.lazy()
    planet_disp_lf = planet_disp.lazy()

    all_base_lf = (
        df_s_lf
        .group_by("id", maintain_order=True)
        .agg([
            pl.first("step").alias("step_src"),
            pl.first("x").alias("x_src"),
            pl.first("y").alias("y_src"),
            pl.first("radius").alias("radius_src"),
            pl.first("ships").alias("ships_at_src"),
            pl.first("production").alias("production_src"),
            pl.first("nature").alias("nature_src"),
            pl.first("owner").alias("owner_src"),
        ])
        .rename({"id": "id_src"})
    )

    dx = pl.col("x") - pl.col("x_src")
    dy = pl.col("y") - pl.col("y_src")
    l2 = dx.pow(2) + dy.pow(2)
    dist_tgt_src = l2.sqrt()
    step_diff = (pl.col("step") - pl.col("step_src")).cast(pl.Float64)

    dot = (CENTER - pl.col("x_src")) * dx + (CENTER - pl.col("y_src")) * dy
    t_sun = (dot / pl.when(l2 == 0).then(pl.lit(1.0)).otherwise(l2)).clip(0.0, 1.0)
    proj_dist_sun = (
        (CENTER - pl.col("x_src") - t_sun * dx).pow(2) +
        (CENTER - pl.col("y_src") - t_sun * dy).pow(2)
    ).sqrt()
    crossing_sun = pl.when(l2 == 0).then(
        ((CENTER - pl.col("x_src")).pow(2) + (CENTER - pl.col("y_src")).pow(2)).sqrt()
    ).otherwise(proj_dist_sun) < (SUN_RADIUS + PLANET_MARGIN)

    coarse_lf = (
        all_base_lf.join(df_s_lf, how="cross")
        .filter((pl.col("step") > pl.col("step_src")) & (pl.col("id") != pl.col("id_src")))
        .join(planet_disp_lf, on=["id", "step"], how="left")
        .with_columns([dist_tgt_src.alias("dist_tgt_src"), step_diff.alias("step_diff")])
        .filter(
            (pl.col("dist_tgt_src") < (pl.col("step_diff") + 1) * MAX_SPEED
             + pl.col("radius_src") + PLANET_MARGIN + pl.col("radius")
             + pl.col("planet_disp").fill_null(0.0))
            & ~crossing_sun
        )
        .with_columns(pl.col("ships_at_src").alias("ships_sent"))
    )

    fs_expr = 1.0 + (MAX_SPEED - 1.0) * (
        pl.col("ships_sent").cast(pl.Float64).log(base=math.e) / math.log(1000.0)
    ).clip(lower_bound=0.0).pow(1.5)
    dm_expr = pl.col("step_diff") * fs_expr + PLANET_MARGIN + pl.col("radius_src")
    dp_expr = dm_expr - fs_expr

    prev_pos_lf = (
        df_s_lf.select(["id", "step", "x", "y"])
        .rename({"x": "x_prev", "y": "y_prev"})
        .with_columns((pl.col("step") + 1).alias("step"))
    )

    ux = (pl.col("x") - pl.col("x_src")) / pl.when(
        pl.col("dist_tgt_src") < 1e-9
    ).then(pl.lit(1.0)).otherwise(pl.col("dist_tgt_src"))
    uy = (pl.col("y") - pl.col("y_src")) / pl.when(
        pl.col("dist_tgt_src") < 1e-9
    ).then(pl.lit(1.0)).otherwise(pl.col("dist_tgt_src"))
    fx0  = pl.col("x_src") + ux * pl.col("dist_prev")
    fy0  = pl.col("y_src") + uy * pl.col("dist_prev")
    pvx  = pl.col("x") - pl.col("x_prev").fill_null(pl.col("x"))
    pvy  = pl.col("y") - pl.col("y_prev").fill_null(pl.col("y"))
    dvx  = ux * pl.col("fleet_speed") - pvx
    dvy  = uy * pl.col("fleet_speed") - pvy
    d0x  = fx0 - pl.col("x_prev").fill_null(pl.col("x"))
    d0y  = fy0 - pl.col("y_prev").fill_null(pl.col("y"))
    a_sp = dvx.pow(2) + dvy.pow(2)
    b_sp = 2.0 * (d0x * dvx + d0y * dvy)
    c_sp = d0x.pow(2) + d0y.pow(2) - pl.col("radius").pow(2)
    disc = b_sp.pow(2) - 4.0 * a_sp * c_sp
    sq   = disc.clip(lower_bound=0.0).sqrt()
    t1e  = pl.when(a_sp < 1e-12).then(pl.lit(0.0)).otherwise((-b_sp - sq) / (2.0 * a_sp))
    t2e  = pl.when(a_sp < 1e-12).then(pl.lit(1.0)).otherwise((-b_sp + sq) / (2.0 * a_sp))
    coll = pl.when(a_sp < 1e-12).then(c_sp <= 0.0).otherwise(
        (disc >= 0.0) & (t2e >= 0.0) & (t1e <= 1.0)
    )

    xpf = pl.col("x_prev").fill_null(pl.col("x"))
    ypf = pl.col("y_prev").fill_null(pl.col("y"))

    return (
        coarse_lf
        .with_columns([fs_expr.alias("fleet_speed"), dm_expr.alias("dist_min"), dp_expr.alias("dist_prev")])
        .filter(pl.col("dist_tgt_src") < pl.col("dist_min") + pl.col("fleet_speed") + pl.col("radius") + PLANET_MOVEMENT_SLACK)
        .join(prev_pos_lf, on=["id", "step"], how="left")
        .with_columns([t1e.alias("t1"), t2e.alias("t2"), coll.alias("collision")])
        .filter(pl.col("collision"))
        .with_columns([pl.col("t1").clip(0.0, 1.0).alias("t1_eff"), pl.col("t2").clip(0.0, 1.0).alias("t2_eff")])
        .with_columns([
            (xpf + pl.col("t1_eff") * (pl.col("x") - xpf)).alias("p_t1_x"),
            (ypf + pl.col("t1_eff") * (pl.col("y") - ypf)).alias("p_t1_y"),
            (xpf + pl.col("t2_eff") * (pl.col("x") - xpf)).alias("p_t2_x"),
            (ypf + pl.col("t2_eff") * (pl.col("y") - ypf)).alias("p_t2_y"),
        ])
        .with_columns([
            pl.arctan2(pl.col("p_t1_y") - pl.col("y_src"), pl.col("p_t1_x") - pl.col("x_src")).alias("angle_t1"),
            pl.arctan2(pl.col("p_t2_y") - pl.col("y_src"), pl.col("p_t2_x") - pl.col("x_src")).alias("angle_t2"),
            ((pl.col("p_t1_x")-pl.col("x_src")).pow(2)+(pl.col("p_t1_y")-pl.col("y_src")).pow(2)).sqrt().alias("d_s_t1"),
            ((pl.col("p_t2_x")-pl.col("x_src")).pow(2)+(pl.col("p_t2_y")-pl.col("y_src")).pow(2)).sqrt().alias("d_s_t2"),
        ])
        .with_columns([
            (pl.col("dist_prev") + pl.col("t1_eff") * pl.col("fleet_speed")).alias("d_f_t1"),
            (pl.col("dist_prev") + pl.col("t2_eff") * pl.col("fleet_speed")).alias("d_f_t2"),
        ])
        .with_columns([
            ((pl.col("d_s_t1").pow(2)+pl.col("d_f_t1").pow(2)-pl.col("radius").pow(2))
             /(2.0*pl.col("d_s_t1")*pl.col("d_f_t1"))).clip(-1.0,1.0).arccos().alias("angle_radius_t1"),
            ((pl.col("d_s_t2").pow(2)+pl.col("d_f_t2").pow(2)-pl.col("radius").pow(2))
             /(2.0*pl.col("d_s_t2")*pl.col("d_f_t2"))).clip(-1.0,1.0).arccos().alias("angle_radius_t2"),
        ])
        .with_columns([
            pl.min_horizontal(
                pl.col("angle_t1") - pl.col("angle_radius_t1"),
                pl.col("angle_t2") - pl.col("angle_radius_t2"),
            ).mod(2 * math.pi).alias("angle_min"),
            pl.max_horizontal(
                pl.col("angle_t1") + pl.col("angle_radius_t1"),
                pl.col("angle_t2") + pl.col("angle_radius_t2"),
            ).mod(2 * math.pi).alias("angle_max"),
            pl.arctan2(
                pl.col("angle_t1").sin() + pl.col("angle_t2").sin(),
                pl.col("angle_t1").cos() + pl.col("angle_t2").cos(),
            ).alias("angle"),
        ])
        .sort("step")
    )


# ── Collision filter (from 112-GNN_torch_only.py / 113-Polars_GNN_Corrected.py) ─

def _03_filter_collision(pa_lf: pl.LazyFrame) -> pl.LazyFrame:
    angle_norm = pl.col("angle") % (2 * math.pi)
    wraps    = pl.col("angle_min_obs") > pl.col("angle_max_obs")
    in_cone  = pl.when(wraps).then(
        (angle_norm >= pl.col("angle_min_obs")) | (angle_norm <= pl.col("angle_max_obs"))
    ).otherwise(
        (angle_norm >= pl.col("angle_min_obs")) & (angle_norm <= pl.col("angle_max_obs"))
    )

    blocked_lf = (
        pa_lf.select(["id_src", "ships_sent", "step", "id", "angle", "angle_min", "angle_max"])
        .join(
            pa_lf.select(["id_src", "ships_sent", "step", "id", "angle_min", "angle_max"])
                 .rename({"step": "step_obs", "id": "id_obs",
                          "angle_min": "angle_min_obs", "angle_max": "angle_max_obs"}),
            on=["id_src", "ships_sent"],
            how="inner",
        )
        .filter((pl.col("step_obs") < pl.col("step")) & (pl.col("id_obs") != pl.col("id")))
        .filter(in_cone)
        .select(["id_src", "ships_sent", "step", "id"])
        .unique()
    )

    return (
        pa_lf
        .join(blocked_lf, on=["id_src", "ships_sent", "step", "id"], how="anti")
        .with_columns(pl.col("angle").alias("final_angle"))
    )
```

- [ ] **Step 2: Update `__main__` to add a pipeline smoke test**

Replace the existing `__main__` block with:

```python
if __name__ == "__main__":
    # ── Model smoke test ──────────────────────────────────────────────────────
    _test_model = GNNBehaviourCloningPure(H=64, L1=3, L2=2)
    _weights_path = Path("130-GNN_BC_shift") / "model_epoch100.pt"
    missing, unexpected = _load_weights(_test_model, _weights_path)
    assert not missing,    f"Missing keys: {missing}"
    assert not unexpected, f"Unexpected keys: {unexpected}"
    print("Weight loading: OK")

    N_p, N_ps, N_ams = 3, 6, 2
    _d = types.SimpleNamespace(
        planet_x     = torch.randn(N_p, 4),
        ps_x         = torch.randn(N_ps, 9),
        ams_x        = torch.randn(N_ams, 1),
        ei_has_ps    = torch.tensor([[0,0,1,1,2,2],[0,1,2,3,4,5]], dtype=torch.long),
        ei_rev_has_ps= torch.tensor([[0,1,2,3,4,5],[0,0,1,1,2,2]], dtype=torch.long),
        ei_reaches   = torch.tensor([[0,2],[1,3]], dtype=torch.long),
        reaches_attr = torch.tensor([0.3, 0.7]),
        ei_src_of    = torch.tensor([[0,1],[0,1]], dtype=torch.long),
        ei_rev_src_of= torch.tensor([[0,1],[0,1]], dtype=torch.long),
        ei_tgt_of    = torch.tensor([[2,3],[0,1]], dtype=torch.long),
        ei_rev_tgt_of= torch.tensor([[0,1],[2,3]], dtype=torch.long),
    )
    _test_model.eval()
    with torch.no_grad():
        logits = _test_model(_d)
    assert logits.shape == (N_ams,), f"Expected ({N_ams},), got {logits.shape}"
    print(f"Forward pass: OK  logits={logits.tolist()}")

    # ── Pipeline smoke test (needs 126-precompute/ with at least one episode) ─
    ep_dirs = sorted([d for d in Path("126-precompute").iterdir() if d.is_dir()])
    if ep_dirs:
        import json
        ep_id = int(ep_dirs[0].name)
        # Find episode JSON
        replay_root = Path("110-replays")
        ep_json_path = None
        for sub in sorted(replay_root.iterdir()):
            if not sub.is_dir():
                continue
            cand = sub / f"episode_{ep_id}.json"
            if cand.exists():
                ep_json_path = cand
                break
        if ep_json_path:
            ep_json = json.loads(ep_json_path.read_text(encoding="utf-8"))
            step0 = ep_json["steps"][0]
            obs_dict = step0[0]["observation"]
            obs = types.SimpleNamespace(**obs_dict)
            initial = getattr(obs, "initial_planets", [])
            owners = {p[1] for p in initial if p[1] != -1}
            num_agents = 4 if len(owners) > 2 else 2
            pid = obs.player

            df_s, planet_disp = _01_get_obs_dataframe(obs, 0, num_agents)
            df_s = _00_remap_owner(df_s, obs, pid)
            print(f"Pipeline: df_s shape={df_s.shape}  planet_disp shape={planet_disp.shape}")

            rb2_raw = _02_get_reach(df_s, planet_disp, REACH_SHIPS).collect()
            rb2 = _03_filter_collision(rb2_raw.lazy()).collect()
            print(f"reachable_base_2: {len(rb2)} rows")

            rms_raw = _get_reach_max_ships(df_s, planet_disp).collect()
            rms = _03_filter_collision(rms_raw.lazy()).collect()
            print(f"reachable_max_ships (pre-select): {len(rms)} rows")
            print("Pipeline smoke test: OK")
        else:
            print("Pipeline smoke test SKIP (no episode JSON found)")
    else:
        print("Pipeline smoke test SKIP (no 126-precompute/ dirs)")

    print("All checks passed.")
```

- [ ] **Step 3: Run updated smoke test**

```
cd "C:\Users\trant\Documents\Programmation\Orbit Wars"
python 132-GNN_BC_shift_torch_only.py
```

Expected output (last lines):
```
Weight loading: OK
Forward pass: OK  logits=[..., ...]
Pipeline: df_s shape=(...)  ...
reachable_base_2: N rows
reachable_max_ships (pre-select): M rows
Pipeline smoke test: OK
All checks passed.
```

- [ ] **Step 4: Commit**

```bash
git add 132-GNN_BC_shift_torch_only.py
git commit -m "feat: 132 — add physics pipeline + reachability functions"
```

---

## Task 3: Graph Builder + Agent Function

**Files:**
- Modify: `132-GNN_BC_shift_torch_only.py` (append graph builder + agent sections; replace `__main__`)

**Interfaces:**
- Consumes:
  - `_01_get_obs_dataframe`, `_00_remap_owner`, `_02_get_reach`, `_get_reach_max_ships`, `_03_filter_collision` from Task 2
  - `GNNBehaviourCloningPure`, `_load_weights` from Task 1
- Produces:
  - `build_graph_inference(T, planete, planete_step, reachable_base_2, reachable_max_ships) -> (SimpleNamespace | None, pl.DataFrame | None)`
    - Returns `(graph, action_table)` where `action_table` has columns `["id_src", "angle", "ships_sent"]`, indexed same as `action_max` nodes
  - `agent(obs) -> list[list]` — Kaggle entry point

- [ ] **Step 1: Append graph builder to `132-GNN_BC_shift_torch_only.py`**

After `_03_filter_collision`, append:

```python
# ── Graph construction for inference ─────────────────────────────────────────

def build_graph_inference(
    T: int,
    planete: pl.DataFrame,
    planete_step: pl.DataFrame,
    reachable_base_2: pl.DataFrame,
    reachable_max_ships: pl.DataFrame,
):
    """Build a SimpleNamespace graph for step T.

    Returns (graph, action_table) or (None, None) if no valid action_max nodes.
    action_table has columns [id_src, angle, ships_sent] indexed by action_max node.
    """
    # ── Planet nodes ──────────────────────────────────────────────────────────
    planet_df = planete.filter(pl.col("step") == T).unique("id", keep="first").sort("id")
    if planet_df.is_empty():
        return None, None

    planet_ids = planet_df["id"].to_list()
    max_pid    = max(planet_ids)
    pid_to_idx = np.full(max_pid + 1, -1, dtype=np.int64)
    for i, pid in enumerate(planet_ids):
        pid_to_idx[pid] = i

    nature_arr  = planet_df["nature"].to_numpy()
    planet_feat = np.column_stack([
        planet_df["production"].to_numpy().astype(np.float32) / 5.0,
        (nature_arr == "fix").astype(np.float32),
        (nature_arr == "moving").astype(np.float32),
        (nature_arr == "comet").astype(np.float32),
    ])  # (N_p, 4)

    # ── PlanetStep nodes ──────────────────────────────────────────────────────
    ps_df_raw = planete_step.filter(pl.col("step") == T)
    if ps_df_raw.is_empty():
        return None, None

    planete_renamed = planete.rename({"step": "future_step"})
    ps_df = ps_df_raw.join(planete_renamed, on=["id", "future_step"], how="left")

    ps_ids         = ps_df["id"].to_numpy()
    ps_future_steps = ps_df["future_step"].to_numpy()
    ps_key_to_idx: dict = {
        (int(ps_ids[i]), int(ps_future_steps[i])): i
        for i in range(len(ps_df))
    }

    owner_arr  = ps_df["owner"].to_numpy()
    x_arr      = ps_df["x"].to_numpy().astype(np.float32)
    y_arr      = ps_df["y"].to_numpy().astype(np.float32)
    ships_arr  = np.log(np.clip(ps_df["ships"].to_numpy().astype(np.float32), 1, None)) / LOG_1024
    delta_arr  = (ps_future_steps.astype(np.float32) - T) / NB_STEPS_SIM

    ps_feat = np.column_stack([
        x_arr / 100.0, y_arr / 100.0, delta_arr, ships_arr,
        (owner_arr == -1).astype(np.float32),
        (owner_arr ==  0).astype(np.float32),
        (owner_arr ==  1).astype(np.float32),
        (owner_arr ==  2).astype(np.float32),
        (owner_arr ==  3).astype(np.float32),
    ])  # (N_ps, 9)

    # ── Planet <-> PlanetStep edges ───────────────────────────────────────────
    pid_arr    = ps_df["id"].to_numpy()
    valid_pid  = pid_arr <= max_pid
    p_side_raw = np.where(valid_pid, pid_arr, 0)
    p_side_idx = pid_to_idx[p_side_raw]
    valid_mask = valid_pid & (p_side_idx >= 0)

    has_ps_src = p_side_idx[valid_mask].astype(np.int64)
    has_ps_dst = np.where(valid_mask)[0].astype(np.int64)

    ei_has_ps     = torch.tensor(np.stack([has_ps_src, has_ps_dst]), dtype=torch.long)
    ei_rev_has_ps = torch.tensor(np.stack([has_ps_dst, has_ps_src]), dtype=torch.long)

    # ── ActionBase2 edges (PlanetStep <-> PlanetStep, weighted) ──────────────
    b2 = reachable_base_2.filter(
        (pl.col("step_src") >= T) & (pl.col("step_src") <= T + NB_STEPS_SIM) &
        (pl.col("step_tgt") <= T + NB_STEPS_SIM) &
        (pl.col("ships_sent").is_in(REACH_SHIPS))
    )
    b2_src_list: list = []
    b2_dst_list: list = []
    b2_w_list:   list = []
    if not b2.is_empty():
        b2_id_src   = b2["id_src"].to_numpy()
        b2_step_src = b2["step_src"].to_numpy()
        b2_id_tgt   = b2["id_tgt"].to_numpy()
        b2_step_tgt = b2["step_tgt"].to_numpy()
        b2_ships    = b2["ships_sent"].to_numpy().astype(np.float32)
        for i in range(len(b2)):
            sk = (int(b2_id_src[i]),   int(b2_step_src[i]))
            tk = (int(b2_id_tgt[i]),   int(b2_step_tgt[i]))
            si = ps_key_to_idx.get(sk, -1)
            ti = ps_key_to_idx.get(tk, -1)
            if si >= 0 and ti >= 0:
                b2_src_list.append(si); b2_dst_list.append(ti)
                b2_w_list.append(math.log(max(float(b2_ships[i]), 1.0)) / LOG_1024)

    if b2_src_list:
        ei_reaches   = torch.tensor(np.stack([np.array(b2_src_list), np.array(b2_dst_list)]), dtype=torch.long)
        reaches_attr = torch.tensor(b2_w_list, dtype=torch.float32)
    else:
        ei_reaches   = torch.zeros(2, 0, dtype=torch.long)
        reaches_attr = torch.zeros(0)

    # ── ActionMaxShips nodes ──────────────────────────────────────────────────
    ps_now       = ps_df.filter(pl.col("future_step") == T)
    my_planet_ids = ps_now.filter(pl.col("owner") == 0)["id"].to_list()
    if not my_planet_ids:
        return None, None

    ams_raw = reachable_max_ships.filter(
        (pl.col("step_src") == T) & (pl.col("id_src").is_in(my_planet_ids))
    )
    if ams_raw.is_empty():
        return None, None

    ams_ships    = ams_raw["ships_sent"].to_numpy().astype(np.float32)
    ams_feat_all = (np.log(np.clip(ams_ships, 1, None)) / LOG_1024).reshape(-1, 1)

    ams_id_src   = ams_raw["id_src"].to_numpy()
    ams_id_tgt   = ams_raw["id_tgt"].to_numpy()
    ams_step_tgt = ams_raw["step_tgt"].to_numpy()

    src_ps_list: list = []
    tgt_ps_list: list = []
    valid_rows:  list = []
    for i in range(len(ams_raw)):
        sk = (int(ams_id_src[i]), T)
        tk = (int(ams_id_tgt[i]), int(ams_step_tgt[i]))
        sp = ps_key_to_idx.get(sk, -1)
        tp = ps_key_to_idx.get(tk, -1)
        if sp >= 0 and tp >= 0:
            src_ps_list.append(sp); tgt_ps_list.append(tp)
            valid_rows.append(i)

    if not valid_rows:
        return None, None

    vidx     = np.array(valid_rows, dtype=np.int64)
    ams_feat = ams_feat_all[vidx]
    n_ams    = len(valid_rows)

    src_ps = np.array(src_ps_list, dtype=np.int64)
    tgt_ps = np.array(tgt_ps_list, dtype=np.int64)
    aidx   = np.arange(n_ams, dtype=np.int64)

    graph = types.SimpleNamespace(
        planet_x      = torch.tensor(planet_feat, dtype=torch.float32),
        ps_x          = torch.tensor(ps_feat,     dtype=torch.float32),
        ams_x         = torch.tensor(ams_feat,    dtype=torch.float32),
        ei_has_ps     = ei_has_ps,
        ei_rev_has_ps = ei_rev_has_ps,
        ei_reaches    = ei_reaches,
        reaches_attr  = reaches_attr,
        ei_src_of     = torch.tensor(np.stack([src_ps, aidx]), dtype=torch.long),
        ei_rev_src_of = torch.tensor(np.stack([aidx, src_ps]), dtype=torch.long),
        ei_tgt_of     = torch.tensor(np.stack([tgt_ps, aidx]), dtype=torch.long),
        ei_rev_tgt_of = torch.tensor(np.stack([aidx, tgt_ps]), dtype=torch.long),
    )

    action_table = ams_raw[vidx.tolist()].select(["id_src", "angle", "ships_sent"])
    return graph, action_table
```

- [ ] **Step 2: Append model instantiation + agent() to `132-GNN_BC_shift_torch_only.py`**

```python
# ── Model instantiation ───────────────────────────────────────────────────────

_here         = Path(__file__).parent if "__file__" in dir() else Path(".")
_weights_path = _here / "model_epoch100.pt"

_model = GNNBehaviourCloningPure(H=HIDDEN_DIM, L1=NUM_LAYERS_P1, L2=NUM_LAYERS_P2)
if _weights_path.exists():
    _missing, _unexpected = _load_weights(_model, _weights_path)
    if _missing or _unexpected:
        print(f"[132] weight load warnings — missing={_missing} unexpected={_unexpected}")
_model.eval()


# ── Agent ─────────────────────────────────────────────────────────────────────

_agent_states: dict = {}


def agent(obs):
    pid = obs.player if hasattr(obs, "player") else obs.get("player", 0)
    if pid not in _agent_states:
        _agent_states[pid] = {"step": 0, "num_agents": None}
    s = _agent_states[pid]

    if s["num_agents"] is None:
        initial = getattr(obs, "initial_planets", None) or obs.get("initial_planets", [])
        owners  = {p[1] for p in initial if p[1] != -1}
        s["num_agents"] = 4 if len(owners) > 2 else 2

    T          = s["step"]
    num_agents = s["num_agents"]

    # ── Build observation dataframe ──────────────────────────────────────────
    df_s, planet_disp = _01_get_obs_dataframe(obs, T, num_agents)
    df_s = _00_remap_owner(df_s, obs, pid)

    # ── Reachable with discrete ship sizes (for graph edges) ─────────────────
    rb2_raw = _02_get_reach(df_s, planet_disp, REACH_SHIPS).collect()
    if rb2_raw.is_empty():
        s["step"] += 1
        return []
    rb2 = (
        _03_filter_collision(rb2_raw.lazy()).collect()
        .select(["id_src", "step_src", "id", "step", "ships_sent"])
        .rename({"id": "id_tgt", "step": "step_tgt"})
    )

    # ── Reachable with max ships at source (for action_max nodes) ────────────
    rms_raw = _get_reach_max_ships(df_s, planet_disp).collect()
    if rms_raw.is_empty():
        s["step"] += 1
        return []
    rms_filtered = _03_filter_collision(rms_raw.lazy()).collect()
    rms = (
        rms_filtered
        .select(["id_src", "step_src", "angle", "ships_sent", "id", "step"])
        .rename({"id": "id_tgt", "step": "step_tgt"})
        .sort("step_tgt")
        .group_by(["id_src", "step_src", "id_tgt"], maintain_order=True)
        .first()
        .select(["id_src", "step_src", "angle", "ships_sent", "id_tgt", "step_tgt"])
    )

    # ── Build planete / planete_step tables ───────────────────────────────────
    planete = df_s.select(["id", "step", "x", "y", "production", "nature"])
    planete_step = (
        df_s.select(["id", "step", "ships", "owner"])
        .with_columns(pl.lit(T).cast(pl.Int64).alias("obs_step"))
        .rename({"obs_step": "step", "step": "future_step"})
        .select(["id", "step", "future_step", "ships", "owner"])
    )

    # ── Build graph + run model ───────────────────────────────────────────────
    graph, action_table = build_graph_inference(T, planete, planete_step, rb2, rms)

    s["step"] += 1

    if graph is None:
        return []

    with torch.no_grad():
        logits = _model(graph)

    logits_np = logits.numpy()
    selected  = logits_np > LOGIT_THRESHOLD

    if not selected.any():
        return []

    # Deduplicate: per id_src, keep highest-logit action
    best: dict = {}  # id_src -> (logit_val, row_idx)
    for i, (is_sel, lv) in enumerate(zip(selected, logits_np)):
        if not is_sel:
            continue
        id_src = int(action_table["id_src"][i])
        if id_src not in best or lv > best[id_src][0]:
            best[id_src] = (lv, i)

    moves = []
    for _, (_, row_idx) in best.items():
        row = action_table.row(row_idx, named=True)
        moves.append([int(row["id_src"]), float(row["angle"]), int(row["ships_sent"])])

    return moves
```

- [ ] **Step 3: Update `__main__` with end-to-end agent test**

Replace the `__main__` block at the bottom of the file with:

```python
if __name__ == "__main__":
    import json

    # ── 1. Weight loading + forward pass ─────────────────────────────────────
    _tm = GNNBehaviourCloningPure(H=64, L1=3, L2=2)
    _wp = Path("130-GNN_BC_shift") / "model_epoch100.pt"
    miss, unexp = _load_weights(_tm, _wp)
    assert not miss,  f"Missing keys: {miss}"
    assert not unexp, f"Unexpected keys: {unexp}"
    print("Weight loading: OK")

    N_p, N_ps, N_ams = 3, 6, 2
    _d = types.SimpleNamespace(
        planet_x     = torch.randn(N_p, 4),
        ps_x         = torch.randn(N_ps, 9),
        ams_x        = torch.randn(N_ams, 1),
        ei_has_ps    = torch.tensor([[0,0,1,1,2,2],[0,1,2,3,4,5]], dtype=torch.long),
        ei_rev_has_ps= torch.tensor([[0,1,2,3,4,5],[0,0,1,1,2,2]], dtype=torch.long),
        ei_reaches   = torch.tensor([[0,2],[1,3]], dtype=torch.long),
        reaches_attr = torch.tensor([0.3, 0.7]),
        ei_src_of    = torch.tensor([[0,1],[0,1]], dtype=torch.long),
        ei_rev_src_of= torch.tensor([[0,1],[0,1]], dtype=torch.long),
        ei_tgt_of    = torch.tensor([[2,3],[0,1]], dtype=torch.long),
        ei_rev_tgt_of= torch.tensor([[0,1],[2,3]], dtype=torch.long),
    )
    _tm.eval()
    with torch.no_grad():
        _logits = _tm(_d)
    assert _logits.shape == (N_ams,), f"Bad shape: {_logits.shape}"
    print(f"Forward pass: OK  logits={_logits.tolist()}")

    # ── 2. End-to-end agent() test ────────────────────────────────────────────
    replay_root = Path("110-replays")
    ep_dirs = sorted([d for d in Path("126-precompute").iterdir() if d.is_dir()])
    if ep_dirs:
        ep_id = int(ep_dirs[0].name)
        ep_json_path = None
        for sub in sorted(replay_root.iterdir()):
            if not sub.is_dir():
                continue
            cand = sub / f"episode_{ep_id}.json"
            if cand.exists():
                ep_json_path = cand
                break

        if ep_json_path:
            ep_json = json.loads(ep_json_path.read_text(encoding="utf-8"))
            # Test first 3 steps
            for step_idx in range(min(3, len(ep_json["steps"]))):
                step_state = ep_json["steps"][step_idx]
                if not step_state:
                    continue
                obs_dict = step_state[0]["observation"]
                obs = types.SimpleNamespace(**obs_dict)
                # Reset agent state for each test call
                _agent_states.clear()
                # Manually set step counter to step_idx
                pid = obs.player
                _agent_states[pid] = {"step": step_idx, "num_agents": None}
                moves = agent(obs)
                print(f"  step {step_idx}: agent() returned {len(moves)} moves: {moves}")
            print("End-to-end agent test: OK")
        else:
            print("End-to-end test SKIP (no episode JSON found)")
    else:
        print("End-to-end test SKIP (no 126-precompute/ dirs)")

    print("All checks passed.")
```

- [ ] **Step 4: Run the full test**

```
cd "C:\Users\trant\Documents\Programmation\Orbit Wars"
python 132-GNN_BC_shift_torch_only.py
```

Expected output:
```
Weight loading: OK
Forward pass: OK  logits=[..., ...]
  step 0: agent() returned N moves: [...]
  step 1: agent() returned N moves: [...]
  step 2: agent() returned N moves: [...]
End-to-end agent test: OK
All checks passed.
```

No exceptions and no assertion errors. It's fine if `N moves = 0` for some steps (threshold is strict by design).

- [ ] **Step 5: Commit**

```bash
git add 132-GNN_BC_shift_torch_only.py
git commit -m "feat: 132 — add build_graph_inference + agent() entry point"
```

---

## Task 4: Bundle Weights + Verify Submission Package

**Files:**
- Create: `model_epoch100.pt` (copy of `130-GNN_BC_shift/model_epoch100.pt` alongside the script)

**Interfaces:**
- Consumes: `130-GNN_BC_shift/model_epoch100.pt`
- Produces: `model_epoch100.pt` in project root (same dir as `132-GNN_BC_shift_torch_only.py`)

- [ ] **Step 1: Copy weights file**

```bash
cp "130-GNN_BC_shift/model_epoch100.pt" "model_epoch100.pt"
```

Verify:
```bash
ls -lh model_epoch100.pt
```
Expected: file exists, size ≈ 570 KB (142K params × 4 bytes).

- [ ] **Step 2: Confirm the agent loads from the right path**

The model loading code uses:
```python
_here = Path(__file__).parent if "__file__" in dir() else Path(".")
_weights_path = _here / "model_epoch100.pt"
```

When submitted as a zip (script + weights in the same folder), `__file__` points to the script and `_here` is that directory. The weights will be found. ✓

- [ ] **Step 3: Run `python 132-GNN_BC_shift_torch_only.py` one final time to confirm everything works with the co-located weights file**

```
cd "C:\Users\trant\Documents\Programmation\Orbit Wars"
python 132-GNN_BC_shift_torch_only.py
```

Expected: `All checks passed.`

- [ ] **Step 4: Final commit**

```bash
git add 132-GNN_BC_shift_torch_only.py model_epoch100.pt
git commit -m "feat: 132 — complete GNN_BC_shift pure-PyTorch submission agent"
```

- [ ] **Step 5: Use /lab and /sub skills to submit**

Deploy to lab for local testing:
```
/lab 132-GNN_BC_shift_torch_only.py
```

Then submit to Kaggle:
```
/sub 132-GNN_BC_shift_torch_only.py
```

---

## Self-Review

**Spec coverage:**
- ✅ Pure PyTorch model (no torch_geometric): Task 1
- ✅ _SAGEConvPure, _ScaledSAGEConvPure, _Phase1ConvPure, _Phase2ConvPure: Task 1
- ✅ Weight remapping via _KEY_MAP: Task 1
- ✅ Physics pipeline (_01, _00_remap, _02_get_reach, _get_reach_max_ships, _03_filter_collision): Task 2
- ✅ build_graph_inference with action_table return: Task 3
- ✅ Threshold logit > log(576), deduplicate per id_src: Task 3
- ✅ model_epoch100.pt bundled in same directory: Task 4
- ✅ File named 132-GNN_BC_shift_torch_only.py: Task 1 header

**Placeholder scan:** No TBDs, no "implement later", all code blocks are complete.

**Type consistency:**
- `_SAGEConvPure.forward(h_src, h_dst, edge_index, num_dst)` → used same signature in `_Phase1ConvPure` and `_Phase2ConvPure` ✅
- `_ScaledSAGEConvPure.forward(h, edge_index, edge_attr, num_nodes)` → used same in `_Phase1ConvPure` ✅
- `GNNBehaviourCloningPure.forward(data)` with `data.planet_x/ps_x/ams_x/ei_*` → produced by `build_graph_inference` same attribute names ✅
- `_load_weights(model, path) -> (list, list)` → used in Task 1 test and Task 3 model loading ✅
- `build_graph_inference` returns `(SimpleNamespace | None, pl.DataFrame | None)` → consumed by `agent()` ✅
