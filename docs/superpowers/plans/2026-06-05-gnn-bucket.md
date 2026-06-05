# GNN Bucket Action Selector Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `97-library.py` (action-node graph construction + heuristic labelling) and `97-GNN_bucket.ipynb` (tripartite SAGEConv GNN that selects attacks and ships from full multi-planet game states).

**Architecture:** Action opportunities `(src, dst, eta)` become graph nodes with 2-dim ship-range features. Three SAGEConv passes propagate planet context into action nodes, action context into destination planets, and enriched planet context back into actions. Two output heads — binary select + sigmoid ships — are trained by imitating the `90-Simulate10Next_Conqueror2_Supplier_prod_per_step.py` heuristic.

**Tech Stack:** Python 3.10+, PyTorch, PyTorch Geometric (`SAGEConv`), pandas, numpy, scikit-learn, matplotlib

---

## File Map

| File | Action | Responsibility |
|------|--------|---------------|
| `97-library.py` | Create | `_fleet_hits_at_step`, `enumerate_action_nodes`, `label_action_nodes`, `build_hetero_data_97`, `generate_sample_97`, `place_planets_randomly` |
| `97-GNN_bucket.ipynb` | Create | Dataset cells, `GNNActionSelector` model, training loop, evaluation, inference demos |

`97-library.py` opens by exec-ing `96-library.py` and `90-Simulate10Next_Conqueror2_Supplier_prod_per_step.py` so all their symbols are available. It adds only the new functions for the action-node graph.

---

### Task 1: Create `97-library.py` — base + `_fleet_hits_at_step`

**Files:**
- Create: `97-library.py`

- [ ] **Step 1: Write the file header with execs and `_fleet_hits_at_step` helper**

```python
import math, copy
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData

# Pull in all symbols from 96 and 90 (interpreter, PhysicsEngine, Obs, GameConfig,
# get_obs_dataframe, get_opportunities, StrategyPipeline, etc.)
exec(open('96-library.py').read())
exec(open('90-Simulate10Next_Conqueror2_Supplier_prod_per_step.py').read())

_LOG1024 = math.log(1024.0)
_MAX_SHIPS_SEARCH = 1024


def _fleet_hits_at_step(ships: int, src_x: float, src_y: float, r_src: float,
                         df_s: pd.DataFrame, dst_id: int, target_step: int,
                         base_step: int = 0) -> bool:
    """True iff a fleet of `ships` ships launched from (src_x, src_y) aimed at
    dst's position at base_step+target_step collides with dst at that exact step.

    Uses the same swept-pair intercept logic as get_opportunities:
    - fleet travels in a straight line toward dst's position at target_step
    - swept_pair_hit checks fleet segment vs planet segment between steps
    """
    speed = PhysicsEngine.fleet_speed(ships)
    dst_rows = df_s[df_s['id'] == dst_id]

    dst_at = dst_rows[dst_rows['step'] == base_step + target_step]
    if dst_at.empty:
        return False

    tgt_x = float(dst_at['x'].values[0])
    tgt_y = float(dst_at['y'].values[0])
    r_dst = float(dst_at['radius'].values[0])

    dx, dy = tgt_x - src_x, tgt_y - src_y
    dist_to_tgt = math.hypot(dx, dy)
    if dist_to_tgt < 1e-9:
        return False

    ux, uy = dx / dist_to_tgt, dy / dist_to_tgt
    start_dist = r_src + GameConfig.PLANET_MARGIN

    f_old = (src_x + ux * (start_dist + (target_step - 1) * speed),
             src_y + uy * (start_dist + (target_step - 1) * speed))
    f_new = (src_x + ux * (start_dist + target_step * speed),
             src_y + uy * (start_dist + target_step * speed))

    # Planet p_old: position at target_step-1 (for step 1, use step 0)
    prev_abs = base_step + target_step - 1
    dst_prev = dst_rows[dst_rows['step'] == prev_abs]
    if not dst_prev.empty:
        p_old = (float(dst_prev['x'].values[0]), float(dst_prev['y'].values[0]))
    else:
        p_old = (tgt_x, tgt_y)

    return PhysicsEngine.swept_pair_hit(f_old, f_new, p_old, (tgt_x, tgt_y), r_dst)
```

- [ ] **Step 2: Verify it parses**

```bash
python -c "
import os; os.chdir(r'C:\Users\trant\Documents\Programmation\Orbit Wars')
exec(open('97-library.py').read())
print('97-library.py loaded OK')
"
```

Expected: `97-library.py loaded OK`

- [ ] **Step 3: Commit**

```bash
git add 97-library.py
git commit -m "feat: 97-library.py base — imports, _fleet_hits_at_step helper"
```

---

### Task 2: Add `enumerate_action_nodes` to `97-library.py`

**Files:**
- Modify: `97-library.py`

- [ ] **Step 1: Append `_find_eta_range` and `enumerate_action_nodes`**

```python
def _find_eta_range(src_x: float, src_y: float, r_src: float,
                    df_s: pd.DataFrame, dst_id: int, eta: int,
                    base_step: int = 0):
    """Return (min_ships, max_ships) for ETA bucket eta, or (None, None) if empty.

    min_ships: fewest ships that arrive at step eta (fast enough).
    max_ships: most ships that arrive at step eta but NOT step eta-1 (not too fast).
    Binary search over [1, 1024].  More ships = faster = smaller ETA.
    """
    # min_ships: smallest s where fleet hits at step eta
    if not _fleet_hits_at_step(_MAX_SHIPS_SEARCH, src_x, src_y, r_src, df_s, dst_id, eta, base_step):
        return None, None  # even max ships can't reach in eta steps
    if _fleet_hits_at_step(1, src_x, src_y, r_src, df_s, dst_id, eta, base_step):
        min_ships = 1
    else:
        lo, hi = 1, _MAX_SHIPS_SEARCH
        while lo < hi - 1:
            mid = (lo + hi) // 2
            if _fleet_hits_at_step(mid, src_x, src_y, r_src, df_s, dst_id, eta, base_step):
                hi = mid
            else:
                lo = mid
        min_ships = hi

    # max_ships: largest s that does NOT arrive at step eta-1
    if eta == 1:
        max_ships = _MAX_SHIPS_SEARCH
    else:
        if _fleet_hits_at_step(1, src_x, src_y, r_src, df_s, dst_id, eta - 1, base_step):
            return None, None  # slowest fleet already arrives at eta-1; bucket empty
        if not _fleet_hits_at_step(_MAX_SHIPS_SEARCH, src_x, src_y, r_src, df_s, dst_id, eta - 1, base_step):
            max_ships = _MAX_SHIPS_SEARCH
        else:
            lo2, hi2 = 1, _MAX_SHIPS_SEARCH
            while lo2 < hi2 - 1:
                mid = (lo2 + hi2) // 2
                if _fleet_hits_at_step(mid, src_x, src_y, r_src, df_s, dst_id, eta - 1, base_step):
                    hi2 = mid
                else:
                    lo2 = mid
            max_ships = lo2  # last that does NOT hit at eta-1

    if min_ships > max_ships:
        return None, None
    return min_ships, max_ships


def enumerate_action_nodes(obs, df_s: pd.DataFrame, player_id: int = 0,
                            base_step: int = 0) -> list:
    """Return list of (src_id, dst_id, eta, min_ships, max_ships) for all
    feasible attack opportunities owned by player_id.

    One entry per valid ETA bucket (1..9) for each (src, dst) pair.
    Paths crossing the sun are skipped.
    """
    step0_rows = df_s[df_s['step'] == base_step].set_index('id')

    action_nodes = []
    owned_ids = [p[0] for p in obs.planets if p[1] == player_id]

    for src_id in owned_ids:
        if src_id not in step0_rows.index:
            continue
        src_row = step0_rows.loc[src_id]
        src_x, src_y, r_src = float(src_row['x']), float(src_row['y']), float(src_row['radius'])

        for dst_id, dst_row in step0_rows.iterrows():
            if dst_id == src_id:
                continue
            dst_x0, dst_y0 = float(dst_row['x']), float(dst_row['y'])
            if not _path_clears_sun(src_x, src_y, dst_x0, dst_y0):
                continue

            for eta in range(1, 10):  # 1..9
                min_s, max_s = _find_eta_range(src_x, src_y, r_src, df_s, dst_id, eta, base_step)
                if min_s is not None:
                    action_nodes.append((src_id, dst_id, eta, min_s, max_s))

    return action_nodes
```

- [ ] **Step 2: Smoke test**

```bash
python -c "
import os, math; os.chdir(r'C:\Users\trant\Documents\Programmation\Orbit Wars')
exec(open('97-library.py').read())
R = lambda p: 1 + math.log(p)
obs = Obs(planets=[
    [0, 0, 30., 30., R(2), 50, 2],
    [1, -1, 45., 30., R(3), 30, 3],
], angular_velocity=0.03)
df_s, pd_ = get_obs_dataframe(obs, 0)
nodes = enumerate_action_nodes(obs, df_s, player_id=0)
assert len(nodes) > 0, 'Expected at least one action node'
assert all(len(n) == 5 for n in nodes)
assert all(n[2] in range(1,10) for n in nodes)
assert all(n[3] <= n[4] for n in nodes)
print(f'enumerate_action_nodes OK: {len(nodes)} nodes, ETAs={sorted(set(n[2] for n in nodes))}')
"
```

- [ ] **Step 3: Commit**

```bash
git add 97-library.py
git commit -m "feat: add _find_eta_range and enumerate_action_nodes to 97-library.py"
```

---

### Task 3: Add `label_action_nodes` to `97-library.py`

**Files:**
- Modify: `97-library.py`

- [ ] **Step 1: Append `label_action_nodes`**

```python
def label_action_nodes(action_nodes: list, heuristic_moves: list,
                        obs, df_s: pd.DataFrame, base_step: int = 0):
    """Label each action node from heuristic_moves.

    heuristic_moves: [[src_id, angle, ships_sent], ...] from StrategyPipeline._04_score_and_decide

    Returns:
        labels:       np.ndarray shape (N,) float32 — 1.0 if heuristic selected this action
        ships_targets: np.ndarray shape (N,) float32 — log(ships)/log(1024) normalised
    """
    n = len(action_nodes)
    labels = np.zeros(n, dtype=np.float32)
    ships_targets = np.array(
        [math.log(max(an[3], 1)) / _LOG1024 for an in action_nodes],
        dtype=np.float32,
    )  # default: log(min_ships) / log(1024)

    if not heuristic_moves:
        return labels, ships_targets

    step0 = df_s[df_s['step'] == base_step].set_index('id')

    for move in heuristic_moves:
        src_id, move_angle, ships_sent = int(move[0]), float(move[1]), int(move[2])

        # Find dst: planet with smallest angular difference from move_angle
        best_dst_id, best_adiff = None, float('inf')
        src_row = step0.loc[src_id] if src_id in step0.index else None
        if src_row is None:
            continue
        sx, sy = float(src_row['x']), float(src_row['y'])

        for pid, row in step0.iterrows():
            if pid == src_id:
                continue
            a = math.atan2(float(row['y']) - sy, float(row['x']) - sx)
            diff = abs((a - move_angle + math.pi) % (2 * math.pi) - math.pi)
            if diff < best_adiff:
                best_adiff, best_dst_id = diff, pid

        if best_dst_id is None:
            continue

        # Find matching action node: same src, same dst, eta where min<=ships_sent<=max
        s_norm = math.log(max(ships_sent, 1)) / _LOG1024
        for k, (asrc, adst, aeta, amin, amax) in enumerate(action_nodes):
            if asrc == src_id and adst == best_dst_id and amin <= ships_sent <= amax:
                labels[k] = 1.0
                ships_targets[k] = s_norm
                break  # first matching eta bucket wins

    return labels, ships_targets
```

- [ ] **Step 2: Smoke test**

```bash
python -c "
import os, math; os.chdir(r'C:\Users\trant\Documents\Programmation\Orbit Wars')
exec(open('97-library.py').read())
R = lambda p: 1 + math.log(p)
obs = Obs(planets=[
    [0, 0, 30., 30., R(2), 50, 2],
    [1, -1, 45., 30., R(3), 30, 3],
], angular_velocity=0.03)
df_s, pd_ = get_obs_dataframe(obs, 0)
nodes = enumerate_action_nodes(obs, df_s)
# Fake heuristic move from planet 0 toward planet 1
import math
angle_01 = math.atan2(30-30, 45-30)
labels, ships_t = label_action_nodes(nodes, [[0, angle_01, 15]], obs, df_s)
assert labels.shape == (len(nodes),)
assert ships_t.shape == (len(nodes),)
print(f'label_action_nodes OK: {int(labels.sum())} positive labels out of {len(nodes)}')
"
```

- [ ] **Step 3: Commit**

```bash
git add 97-library.py
git commit -m "feat: add label_action_nodes to 97-library.py"
```

---

### Task 4: Add `build_hetero_data_97` + `generate_sample_97` to `97-library.py`

**Files:**
- Modify: `97-library.py`

- [ ] **Step 1: Append `build_hetero_data_97`**

```python
def build_hetero_data_97(obs, step: int = 0, player_id: int = 0):
    """Build HeteroData with action nodes labelled by heuristic.

    Returns (HeteroData, action_index) where action_index[k] = (src_id, dst_id, eta).
    """
    df_s, planet_disp = get_obs_dataframe(obs, step)

    # Heuristic labels via 90-Simulate StrategyPipeline
    source_ids = {p[0] for p in obs.planets if p[1] == player_id}
    pa   = get_opportunities(df_s, planet_disp, source_ids)
    safe = StrategyPipeline._03_filter_collision(pa)
    heuristic_moves = StrategyPipeline._04_score_and_decide(safe, player_id)

    # Action nodes
    action_nodes = enumerate_action_nodes(obs, df_s, player_id, base_step=step)
    action_index = [(an[0], an[1], an[2]) for an in action_nodes]

    labels, ships_targets = label_action_nodes(action_nodes, heuristic_moves, obs, df_s, base_step=step)

    # ── Planet node features (22-dim, same as 96-library) ────────────────────
    planets    = obs.planets
    planet_ids = [p[0] for p in planets]
    pid_to_idx = {pid: i for i, pid in enumerate(planet_ids)}

    ships_pivot = (
        df_s[df_s['step'].isin(range(step, step + 11))]
        .pivot_table(index='id', columns='step', values='ships', aggfunc='first')
        .reindex(index=planet_ids, columns=range(step, step + 11), fill_value=0)
    )
    nature_at0 = df_s[df_s['step'] == step].set_index('id')['nature']

    planet_feats = []
    for p in planets:
        pid, owner, x, y, radius, _, production = p
        nat = nature_at0.get(pid, 'fix')
        owner_oh = [0.0] * 5
        if owner == -1:
            owner_oh[0] = 1.0
        elif 0 <= owner <= 3:
            owner_oh[owner + 1] = 1.0
        ships_ts = ships_pivot.loc[pid].values.tolist()
        ships_feats = [math.log(max(min(float(s), 1024.0), 1.0)) / _LOG1024 for s in ships_ts]
        planet_feats.append([
            x / 100.0, y / 100.0,
            1.0 if nat == 'fix' else 0.0,
            1.0 if nat == 'moving' else 0.0,
            1.0 if nat == 'comet' else 0.0,
            production / 5.0,
        ] + owner_oh + ships_feats)

    # ── Master node features (6-dim) ─────────────────────────────────────────
    total_ships = sum(max(p[5], 0) for p in planets) or 1.0
    proportions = [sum(max(p[5], 0) for p in planets if p[1] == pid_) / total_ships
                   for pid_ in range(4)]
    master_feat = [step / 500.0, (obs.angular_velocity - 0.025) / (0.05 - 0.025)] + proportions

    # ── Action node features (2-dim) ─────────────────────────────────────────
    action_feats = [
        [math.log(max(an[3], 1)) / _LOG1024, math.log(max(an[4], 1)) / _LOG1024]
        for an in action_nodes
    ]

    # ── Assemble HeteroData ───────────────────────────────────────────────────
    data = HeteroData()
    data['master'].x = torch.tensor([master_feat], dtype=torch.float)
    data['planet'].x = torch.tensor(planet_feats, dtype=torch.float)

    n_planets = len(planet_ids)
    data['planet', 'to_master', 'master'].edge_index = torch.tensor(
        [list(range(n_planets)), [0] * n_planets], dtype=torch.long)
    data['master', 'to_planet', 'planet'].edge_index = torch.tensor(
        [[0] * n_planets, list(range(n_planets))], dtype=torch.long)

    if action_nodes:
        data['action'].x = torch.tensor(action_feats, dtype=torch.float)
        data['action'].y = torch.tensor(labels, dtype=torch.float)
        data['action'].ships_target = torch.tensor(ships_targets, dtype=torch.float)

        spawns_src = [pid_to_idx[an[0]] for an in action_nodes]
        attacks_dst = [pid_to_idx[an[1]] for an in action_nodes]
        n_actions = len(action_nodes)
        action_idx_range = list(range(n_actions))

        data['planet', 'spawns', 'action'].edge_index = torch.tensor(
            [spawns_src, action_idx_range], dtype=torch.long)
        data['action', 'attacks', 'planet'].edge_index = torch.tensor(
            [action_idx_range, attacks_dst], dtype=torch.long)
    else:
        data['action'].x = torch.zeros((0, 2), dtype=torch.float)
        data['action'].y = torch.zeros(0, dtype=torch.float)
        data['action'].ships_target = torch.zeros(0, dtype=torch.float)
        data['planet', 'spawns', 'action'].edge_index = torch.zeros((2, 0), dtype=torch.long)
        data['action', 'attacks', 'planet'].edge_index = torch.zeros((2, 0), dtype=torch.long)

    return data, action_index
```

- [ ] **Step 2: Append `place_planets_randomly` + `generate_sample_97`**

```python
def _place_planets_randomly(rng, n_owned, n_enemy, n_neutral, player_id=0):
    """Place planets on the board, avoiding sun and each other."""
    planets = []
    pid = 0

    def try_add(owner, attempts=500):
        nonlocal pid
        for _ in range(attempts):
            x = float(rng.uniform(10, 90))
            y = float(rng.uniform(10, 90))
            if math.hypot(x - 50, y - 50) < 15:
                continue
            prod = int(rng.integers(1, 6))
            radius = 1.0 + math.log(prod)
            if any(math.hypot(x - p[2], y - p[3]) < radius + p[4] + 1.0 for p in planets):
                continue
            ships = int(rng.integers(10, 101))
            planets.append([pid, owner, x, y, radius, ships, prod])
            pid += 1
            return True
        return False

    for _ in range(n_owned):
        try_add(player_id)
    for _ in range(n_enemy):
        try_add(1)
    for _ in range(n_neutral):
        try_add(-1)
    return planets


def generate_sample_97(seed: int, player_id: int = 0):
    """Returns (HeteroData, snapshots, action_index).

    snapshots: 11 dicts {step, planets, fleets} for make_animation.
    action_index: list[(src_id, dst_id, eta)] mapping action node k → game action.
    """
    rng = np.random.default_rng(seed)
    angular_velocity = float(rng.uniform(0.025, 0.05))

    n_total  = int(rng.integers(5, 21))
    n_owned  = int(rng.integers(2, min(5, n_total - 2) + 1))
    n_enemy  = int(rng.integers(1, min(3, n_total - n_owned - 1) + 1))
    n_neutral = n_total - n_owned - n_enemy

    planets = _place_planets_randomly(rng, n_owned, n_enemy, n_neutral, player_id)
    obs = Obs(planets=planets, angular_velocity=angular_velocity)

    # Snapshots for animation
    sim = copy.deepcopy(obs)
    snapshots = []
    for i in range(11):
        snapshots.append({
            'step':    i,
            'planets': [p[:] for p in sim.planets],
            'fleets':  [f[:] for f in sim.fleets],
        })
        interpreter(sim, [[], []], i)

    data, action_index = build_hetero_data_97(obs, step=0, player_id=player_id)
    return data, snapshots, action_index
```

- [ ] **Step 3: Full pipeline smoke test**

```bash
python -c "
import os; os.chdir(r'C:\Users\trant\Documents\Programmation\Orbit Wars')
exec(open('97-library.py').read())
data, snaps, ai = generate_sample_97(42)
assert data['master'].x.shape == (1, 6)
assert data['planet'].x.shape[1] == 22
assert data['action'].x.shape[1] == 2
assert data['action'].y.shape[0] == data['action'].x.shape[0]
assert data['action'].ships_target.shape[0] == data['action'].x.shape[0]
assert data['planet','spawns','action'].edge_index.shape[0] == 2
assert data['action','attacks','planet'].edge_index.shape[0] == 2
assert len(snaps) == 11
n_act = data['action'].x.shape[0]
n_pos = int(data['action'].y.sum().item())
print(f'generate_sample_97 OK: {data[\"planet\"].x.shape[0]} planets, {n_act} actions, {n_pos} positive')
"
```

- [ ] **Step 4: Commit**

```bash
git add 97-library.py
git commit -m "feat: add build_hetero_data_97, place_planets_randomly, generate_sample_97"
```

---

### Task 5: Create `97-GNN_bucket.ipynb` — setup + dataset cells (0–4)

**Files:**
- Create: `97-GNN_bucket.ipynb`

- [ ] **Step 1: Create notebook and add Cell 0 — markdown title**

Create via `mcp__jupyter__notebook_create` or Write tool with raw JSON.

```markdown
# 97 — GNN Bucket Action Selector

Heterogeneous GNN (tripartite: planet ↔ action ↔ planet) that selects attacks and
ship counts, trained by imitating the 90-Simulate heuristic.
Output: [[src_id, dst_id, eta, ships_to_send], ...]
```

- [ ] **Step 2: Cell 1 — imports**

```python
%run 96-library.py
%run 90-Simulate10Next_Conqueror2_Supplier_prod_per_step.py
%run 97-library.py
import torch
import torch.nn as nn
from torch_geometric.nn import SAGEConv
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML, display
import math, copy
```

- [ ] **Step 3: Cell 2 — animation helpers (identical to 94/96)**

```python
_COLORS = {0: 'steelblue', 1: 'tomato', -1: '#888888'}

def make_animation(snapshots, title='', interval=150):
    fig, ax = plt.subplots(figsize=(6, 6))
    fig.patch.set_facecolor('#111122')
    def draw(frame):
        snap = snapshots[frame]
        ax.cla()
        ax.set_xlim(0, 100); ax.set_ylim(100, 0)
        ax.set_aspect('equal'); ax.set_facecolor('#111122')
        ax.tick_params(colors='#aaaaaa')
        for sp in ax.spines.values(): sp.set_edgecolor('#444444')
        ax.set_title(f"{title}  (step {snap['step']})", color='white', fontsize=11)
        ax.add_patch(plt.Circle((50, 50), 10, color='gold', zorder=2, alpha=0.9))
        for p in snap['planets']:
            pid, owner, x, y, radius, ships, production = p
            c = _COLORS.get(owner, '#888888')
            ax.add_patch(plt.Circle((x, y), radius, color=c, alpha=0.85, zorder=3))
            ax.text(x, y,   str(ships),         ha='center', va='center', color='white', fontsize=7, fontweight='bold', zorder=4)
            ax.text(x, y+2, str(pid),            ha='center', va='center', color='red',   fontsize=7, fontweight='bold', zorder=4)
            ax.text(x, y-2, '+'+str(production), ha='center', va='center', color='white', fontsize=5, fontweight='bold', zorder=4)
        return []
    ani = animation.FuncAnimation(fig, draw, frames=len(snapshots), interval=interval)
    plt.close()
    return HTML(ani.to_jshtml())
```

- [ ] **Step 4: Cell 3 — inspect one sample**

```python
data0, snaps0, ai0 = generate_sample_97(42)
n_act = data0['action'].x.shape[0]
n_pos = int(data0['action'].y.sum().item())
print(f"Planets: {data0['planet'].x.shape[0]}")
print(f"Action nodes: {n_act}  |  Positive (heuristic selected): {n_pos}")
print(f"Spawns edges: {data0['planet','spawns','action'].edge_index.shape[1]}")
print(f"Attacks edges: {data0['action','attacks','planet'].edge_index.shape[1]}")
make_animation(snaps0, title='Sample seed=42', interval=200)
```

- [ ] **Step 5: Cell 4 — generate datasets**

```python
print("Generating train dataset (1000 samples)...")
train_dataset = [generate_sample_97(i) for i in range(1000)]
print("Generating test dataset (20 samples)...")
test_dataset  = [generate_sample_97(10000 + i) for i in range(20)]

train_pos = sum(int(d.y.sum()) for d, _, _ in train_dataset)
train_tot = sum(d['action'].x.shape[0] for d, _, _ in train_dataset)
print(f"Train: {train_tot} action nodes total, {train_pos} positive ({100*train_pos/max(train_tot,1):.1f}%)")
```

- [ ] **Step 6: Commit**

```bash
git add 97-GNN_bucket.ipynb
git commit -m "feat: 97-GNN_bucket.ipynb cells 0-4 — setup, animation, inspect, datasets"
```

---

### Task 6: Add `GNNActionSelector` model + training loop — cells 5–6

**Files:**
- Modify: `97-GNN_bucket.ipynb`

- [ ] **Step 1: Add Cell 5 — GNNActionSelector**

```python
import os

class GNNActionSelector(nn.Module):
    """Tripartite GNN: 3 SAGEConv passes, 2 output heads per action node.

    Layer 1: planet → action  (source context into action)
             planet → master, master → planet
    Layer 2: action → planet  (action context into destination)
    Layer 3: planet → action  (enriched destination context back into action)

    Output heads per action node:
      select_head: Linear(H→1) raw logit  (BCEWithLogitsLoss)
      ships_head:  Linear(H→1)+Sigmoid    (MSELoss vs normalised ships target)
    """
    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        H = hidden_dim
        self.master_lin = nn.Linear(6,  H)
        self.planet_lin = nn.Linear(22, H)
        self.action_lin = nn.Linear(2,  H)

        # Layer 1: planet→action, planet→master, master→planet
        self.sage1_pa = SAGEConv(H, H)
        self.sage1_pm = SAGEConv(H, H)
        self.sage1_mp = SAGEConv(H, H)

        # Layer 2: action→planet
        self.sage2_ap = SAGEConv(H, H)

        # Layer 3: planet→action
        self.sage3_pa = SAGEConv(H, H)

        self.select_head = nn.Linear(H, 1)
        self.ships_head  = nn.Sequential(nn.Linear(H, 1), nn.Sigmoid())

    def forward(self, data):
        h_m = torch.relu(self.master_lin(data['master'].x))
        h_p = torch.relu(self.planet_lin(data['planet'].x))
        h_a = torch.relu(self.action_lin(data['action'].x))

        ei_spawns  = data['planet', 'spawns',    'action'].edge_index
        ei_attacks = data['action', 'attacks',   'planet'].edge_index
        ei_pm      = data['planet', 'to_master', 'master'].edge_index
        ei_mp      = data['master', 'to_planet', 'planet'].edge_index

        # Layer 1
        h_a = torch.relu(self.sage1_pa((h_p, h_a), ei_spawns))
        h_m = torch.relu(self.sage1_pm((h_p, h_m), ei_pm))
        h_p = torch.relu(self.sage1_mp((h_m, h_p), ei_mp))

        # Layer 2
        h_p = torch.relu(self.sage2_ap((h_a, h_p), ei_attacks))

        # Layer 3
        h_a = torch.relu(self.sage3_pa((h_p, h_a), ei_spawns))

        return self.select_head(h_a).squeeze(-1), self.ships_head(h_a).squeeze(-1)


MODEL_PATH = "97-model.pt"
model = GNNActionSelector(hidden_dim=64)
n_params = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {n_params:,}")

# Shape check
data0, _, _ = train_dataset[0]
sl, sh = model(data0)
n_act = data0['action'].x.shape[0]
assert sl.shape == (n_act,), f"select logit shape {sl.shape}"
assert sh.shape == (n_act,), f"ships head shape {sh.shape}"
print(f"Forward pass OK — {n_act} action nodes, select logit shape {tuple(sl.shape)}")

if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
    model.eval()
    _model_loaded = True
    print(f"Loaded model from {MODEL_PATH}")
else:
    _model_loaded = False
    print("No saved model — will train from scratch.")
```

- [ ] **Step 2: Add Cell 6 — training loop**

```python
if _model_loaded:
    print("Model already loaded — skipping training.")
else:
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    bce = nn.BCEWithLogitsLoss()
    mse = nn.MSELoss()

    model.train()
    for epoch in range(50):
        total_loss = total_sel = total_shp = 0.0
        for data, _, _ in train_dataset:
            if data['action'].x.shape[0] == 0:
                continue
            optimizer.zero_grad()
            sel_logit, ships_pred = model(data)
            loss_sel = bce(sel_logit, data['action'].y)
            loss_shp = mse(ships_pred, data['action'].ships_target)
            loss = loss_sel + loss_shp
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_sel  += loss_sel.item()
            total_shp  += loss_shp.item()
        if epoch % 10 == 0:
            n = len(train_dataset)
            print(f"Epoch {epoch:3d} | total {total_loss/n:.4f} | bce {total_sel/n:.4f} | mse {total_shp/n:.4f}")

    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Training complete. Saved to {MODEL_PATH}.")
```

- [ ] **Step 3: Commit**

```bash
git add 97-GNN_bucket.ipynb
git commit -m "feat: add GNNActionSelector model and training loop"
```

---

### Task 7: Add evaluation + inference demo cells (7–27)

**Files:**
- Modify: `97-GNN_bucket.ipynb`

- [ ] **Step 1: Add Cell 7 — evaluation metrics**

```python
model.eval()
all_preds, all_labels = [], []
ships_abs_errors = []

with torch.no_grad():
    for data, _, _ in test_dataset:
        if data['action'].x.shape[0] == 0:
            continue
        sel_logit, ships_pred = model(data)
        preds  = (torch.sigmoid(sel_logit) > 0.5).int().tolist()
        labels = data['action'].y.int().tolist()
        all_preds.extend(preds)
        all_labels.extend(labels)
        # Ships MAE only for positive (selected) ground-truth nodes
        pos_mask = data['action'].y.bool()
        if pos_mask.any():
            ships_abs_errors.extend(
                (ships_pred[pos_mask] - data['action'].ships_target[pos_mask]).abs().tolist()
            )

print(f"Select Accuracy: {accuracy_score(all_labels, all_preds):.2f}\n")
print(classification_report(all_labels, all_preds, target_names=['skip','attack']))
if ships_abs_errors:
    mae = sum(ships_abs_errors) / len(ships_abs_errors)
    # Decode to actual ships: round(exp(val * log(1024)))
    mae_ships = sum(
        abs(round(math.exp(e * math.log(1024))) - 0)  # rough magnitude check
        for e in ships_abs_errors
    )
    print(f"Ships head MAE (normalised): {mae:.4f}")
```

- [ ] **Step 2: Add Cells 8–27 — inference demo per test sample**

For i = 0..19, each cell is identical except for `i`:

```python
# Cell 8: i=0, Cell 9: i=1, ... Cell 27: i=19
i = 0  # change for each cell
data_i, snaps_i, ai_i = test_dataset[i]

model.eval()
with torch.no_grad():
    sel_logit_i, ships_pred_i = model(data_i)

select_prob_i = torch.sigmoid(sel_logit_i)
candidates = sorted(enumerate(select_prob_i.tolist()), key=lambda x: -x[1])

# Greedy budget-aware decoder
ships_remaining = {}
for p in snaps_i[0]['planets']:
    if p[1] == 0:  # player_id=0
        ships_remaining[p[0]] = p[5]

output_i = []
for k, prob in candidates:
    if prob < 0.5:
        break
    src_id, dst_id, eta = ai_i[k]
    s_norm = float(ships_pred_i[k])
    ships = round(math.exp(s_norm * math.log(1024)))
    ships = max(1, min(ships, ships_remaining.get(src_id, 0)))
    if ships_remaining.get(src_id, 0) >= 1:
        output_i.append([src_id, dst_id, eta, ships])
        ships_remaining[src_id] -= ships

n_pos_gt = int(data_i['action'].y.sum().item())
print(f"Test {i}: {data_i['planet'].x.shape[0]} planets | {data_i['action'].x.shape[0]} actions | {n_pos_gt} heuristic | {len(output_i)} selected")
print("Output:", output_i)
make_animation(snaps_i, title=f"Test {i} | selected={len(output_i)}", interval=200)
```

Create 20 such cells (i=0 through i=19).

- [ ] **Step 3: Final commit**

```bash
git add 97-GNN_bucket.ipynb
git commit -m "feat: add evaluation metrics and 20 inference demo cells to 97-GNN_bucket.ipynb"
```

---

## Self-Review

**Spec coverage:**

| Spec requirement | Task |
|-----------------|------|
| Master (6), planet (22), action (2) node features | Task 4 `build_hetero_data_97` |
| Action nodes = (src,dst,eta) with min/max ships via binary search | Tasks 2+4 |
| `spawns` planet→action, `attacks` action→planet edges | Task 4 |
| `to_master`, `to_planet` planet↔master edges | Task 4 |
| 3-layer SAGEConv: pa, ap, pa | Task 6 |
| select head (BCE) + ships head (MSE+sigmoid) | Task 6 |
| Imitation labels from 90-Simulate `_04_score_and_decide` | Task 3+4 |
| Match heuristic by closest-angle dst + ships bracket | Task 3 |
| 1000 train / 20 test, 5–20 planets per sample | Task 4+5 |
| Save/load `97-model.pt` | Task 6 |
| Greedy budget-aware inference decoder | Task 7 |
| Output format `[[src,dst,eta,ships], ...]` | Task 7 |
| 20 inference demo cells with animation | Task 7 |

**Placeholder scan:** No TBDs or vague steps found. All code blocks complete.

**Type consistency:**
- `enumerate_action_nodes` returns `list[tuple[int,int,int,int,int]]` (src,dst,eta,min,max) ✓
- `label_action_nodes` takes same `action_nodes` list, returns `(ndarray, ndarray)` ✓
- `build_hetero_data_97` returns `(HeteroData, list[tuple[int,int,int]])` — action_index is `(src,dst,eta)` triples ✓
- `generate_sample_97` returns `(HeteroData, list[dict], list[tuple])` — used as `(data, snaps, ai)` throughout ✓
- Cell 5 `model(data)` returns `(select_logit, ships_pred)` — both used in cells 6+7 ✓
- `data['action'].y` and `data['action'].ships_target` set in Task 4, consumed in Tasks 6+7 ✓
