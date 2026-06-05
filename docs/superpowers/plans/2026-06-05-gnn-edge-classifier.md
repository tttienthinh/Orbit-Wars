# GNN Edge Classifier Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create `96-library.py` (physics + graph construction helpers rewritten from `93-Supplier_to_moving.py`) and `96-GNN_test.ipynb` (2-layer HeteroConv+NNConv GNN trained to classify attack edges in 2-planet scenarios).

**Architecture:** `96-library.py` keeps `GameConfig`, `PhysicsEngine`, `interpreter`, and `Obs` unchanged from `93`, replaces `_02_get_all_opportunities` with `get_opportunities(source_planet_ids)` that expands only `SHIPS_OPTIONS=[1,2,4,8,16]`, and adds `build_edge_features`, `build_hetero_data`, and `generate_sample`. The notebook `%run`s the library, generates 100 train + 10 test `HeteroData` graphs, trains the model with `BCEWithLogitsLoss`, and shows 10 labelled animations.

**Tech Stack:** Python 3.10+, PyTorch, PyTorch Geometric (`SAGEConv`, `NNConv`), scikit-learn, matplotlib, numpy, pandas

---

## File Map

| File | Action | Responsibility |
|------|--------|---------------|
| `96-library.py` | Create | Physics core, simulation helpers, graph construction |
| `96-GNN_test.ipynb` | Create | Dataset generation, model, training, evaluation, animations |

---

### Task 1: Create `96-library.py` — physics core, Obs, and imports

**Files:**
- Create: `96-library.py`

- [ ] **Step 1: Write the file with physics core copied verbatim from `93-Supplier_to_moving.py` plus `Obs` from `94-93-Load_Test_Enhance.ipynb`**

```python
import math
import copy
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData

# ── Configuration ─────────────────────────────────────────────────────────────
class GameConfig:
    CENTER = 50.0
    SUN_RADIUS = 10.0
    ROTATION_RADIUS_LIMIT = 50.0
    MAX_SPEED = 6.0
    NB_STEPS_SIM = 10
    PLANET_MARGIN = 0.1
    PLANET_MOVEMENT_SLACK = 3.0


CENTER = GameConfig.CENTER
SUN_RADIUS = GameConfig.SUN_RADIUS
ROTATION_RADIUS_LIMIT = GameConfig.ROTATION_RADIUS_LIMIT
BOARD_SIZE = 100.0
MAX_NB_STEP = 500

SHIPS_OPTIONS = [1, 2, 4, 8, 16]
_SHIP_TO_IDX  = {s: i for i, s in enumerate(SHIPS_OPTIONS)}


# ── Physics helpers ───────────────────────────────────────────────────────────
class PhysicsEngine:
    @staticmethod
    def distance(p1, p2):
        return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

    @staticmethod
    def point_to_segment_distance(p, v, w):
        l2 = (v[0]-w[0])**2 + (v[1]-w[1])**2
        if l2 == 0.0:
            return PhysicsEngine.distance(p, v)
        t = max(0, min(1, ((p[0]-v[0])*(w[0]-v[0])+(p[1]-v[1])*(w[1]-v[1]))/l2))
        proj = (v[0]+t*(w[0]-v[0]), v[1]+t*(w[1]-v[1]))
        return PhysicsEngine.distance(p, proj)

    @staticmethod
    def swept_pair_hit(A, B, P0, P1, r):
        d0x, d0y = A[0]-P0[0], A[1]-P0[1]
        dvx = (B[0]-A[0])-(P1[0]-P0[0])
        dvy = (B[1]-A[1])-(P1[1]-P0[1])
        a = dvx*dvx + dvy*dvy
        b = 2.0*(d0x*dvx + d0y*dvy)
        c = d0x*d0x + d0y*d0y - r*r
        if a < 1e-12:
            return c <= 0.0
        disc = b*b - 4.0*a*c
        if disc < 0.0:
            return False
        sq = math.sqrt(disc)
        t1 = (-b-sq)/(2.0*a)
        t2 = (-b+sq)/(2.0*a)
        return t2 >= 0.0 and t1 <= 1.0

    @staticmethod
    def fleet_speed(ships):
        if ships <= 1:
            return 1.0
        ratio = math.log(ships) / math.log(1000.0)
        return 1.0 + (GameConfig.MAX_SPEED - 1.0) * max(0.0, min(1.0, ratio))**1.5


# ── Obs ───────────────────────────────────────────────────────────────────────
class Obs:
    def __init__(self, planets, initial_planets=None, fleets=None,
                 next_fleet_id=100, comets=None, comet_planet_ids=None,
                 angular_velocity=0.0):
        self.planets          = [list(p) for p in planets]
        self.initial_planets  = [list(p) for p in (initial_planets if initial_planets is not None else planets)]
        self.fleets           = [list(f) for f in (fleets or [])]
        self.next_fleet_id    = next_fleet_id
        self.comets           = comets or []
        self.comet_planet_ids = comet_planet_ids or []
        self.angular_velocity = angular_velocity
```

- [ ] **Step 2: Append the `interpreter` function (copied verbatim from `93-Supplier_to_moving.py` lines 72–269)**

Paste the full `interpreter(obs, actions, step, num_agents=2)` function. No changes needed.

- [ ] **Step 3: Verify the file parses**

```bash
python -c "import importlib.util; spec=importlib.util.spec_from_file_location('lib','96-library.py'); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m); print('OK')"
```

Expected output: `OK`

- [ ] **Step 4: Commit**

```bash
git add 96-library.py
git commit -m "feat: 96-library.py physics core (GameConfig, PhysicsEngine, interpreter, Obs)"
```

---

### Task 2: Add `get_obs_dataframe` to `96-library.py`

**Files:**
- Modify: `96-library.py`

- [ ] **Step 1: Append `get_obs_dataframe` (adapted from `StrategyPipeline._01_get_obs_dataframe`)**

```python
def get_obs_dataframe(obs, step: int, num_agents: int = 2):
    sim = copy.deepcopy(obs)
    no_actions = [[] for _ in range(num_agents)]
    rows = []
    for i in range(GameConfig.NB_STEPS_SIM + 1):
        for p in sim.planets:
            pid, owner, x, y, radius, ships, production = p[0], p[1], p[2], p[3], p[4], p[5], p[6]
            r = math.hypot(x - GameConfig.CENTER, y - GameConfig.CENTER)
            if pid in sim.comet_planet_ids:
                nature = "comet"
            elif r + radius < GameConfig.ROTATION_RADIUS_LIMIT:
                nature = "moving"
            else:
                nature = "fix"
            rows.append({"step": step+i, "id": pid, "x": x, "y": y,
                         "radius": radius, "ships": ships,
                         "production": production, "owner": owner, "nature": nature})
        interpreter(sim, no_actions, step+i, num_agents)

    df_s = pd.DataFrame(rows).sort_values("step").reset_index(drop=True)

    prev_pos = (
        df_s[["id","step","x","y"]]
        .assign(step=lambda d: d["step"]+1)
        .rename(columns={"x":"x_prev","y":"y_prev"})
    )
    planet_disp = (
        df_s[["id","step","x","y"]]
        .merge(prev_pos, on=["id","step"], how="left")
        .assign(planet_disp=lambda d: np.sqrt(
            (d["x"]-d["x_prev"].fillna(d["x"]))**2 +
            (d["y"]-d["y_prev"].fillna(d["y"]))**2))
        [["id","step","planet_disp"]]
    )
    return df_s, planet_disp
```

- [ ] **Step 2: Verify with a quick smoke test in a Python REPL**

```python
exec(open('96-library.py').read())
import math
R = lambda p: 1 + math.log(p)
obs = Obs(planets=[[0,0,30.,30.,R(2),50,2],[1,-1,40.,40.,R(3),30,3]], angular_velocity=0.03)
df_s, pd_ = get_obs_dataframe(obs, step=0)
assert df_s.shape == (22, 9), f"Expected (22,9), got {df_s.shape}"
assert set(df_s['step'].unique()) == set(range(11))
print("get_obs_dataframe OK")
```

- [ ] **Step 3: Commit**

```bash
git add 96-library.py
git commit -m "feat: add get_obs_dataframe to 96-library.py"
```

---

### Task 3: Add `get_opportunities` to `96-library.py`

**Files:**
- Modify: `96-library.py`

- [ ] **Step 1: Append `get_opportunities(df_s, planet_disp, source_planet_ids)`**

Key changes from `_02_get_all_opportunities`:
- `mine_base` uses `source_planet_ids` instead of player_id multi-step filter
- ships expansion uses `SHIPS_OPTIONS` cross-join instead of `range()`

```python
def get_opportunities(df_s: pd.DataFrame, planet_disp: pd.DataFrame,
                      source_planet_ids: set) -> pd.DataFrame:
    step0 = int(df_s['step'].min())
    src_rows = df_s[(df_s['step'] == step0) & (df_s['id'].isin(source_planet_ids))]
    if src_rows.empty:
        return pd.DataFrame()

    mine_base = (
        src_rows
        .rename(columns={'id':'id_src','x':'x_src','y':'y_src','radius':'radius_src',
                         'ships':'ships_min','production':'production_src',
                         'nature':'nature_src','owner':'owner_src','step':'step_src'})
        [['id_src','x_src','y_src','radius_src','ships_min',
          'production_src','nature_src','owner_src','step_src']]
        .reset_index(drop=True)
    )

    # Phase A: cross join src × all (dst, step) pairs
    coarse = (
        mine_base.assign(_key=1)
        .merge(df_s.assign(_key=1), on='_key').drop(columns='_key')
        .loc[lambda d: (d['step'] > d['step_src']) & (d['id'] != d['id_src'])]
        .merge(planet_disp, on=['id','step'], how='left')
        .reset_index(drop=True)
        .assign(
            dist_tgt_src=lambda d: np.sqrt((d['x']-d['x_src'])**2+(d['y']-d['y_src'])**2),
            step_diff=lambda d: (d['step']-d['step_src']).astype(float),
        )
    )

    # Sun-crossing filter
    _dx = coarse['x'].values - coarse['x_src'].values
    _dy = coarse['y'].values - coarse['y_src'].values
    _l2 = _dx**2 + _dy**2
    _dot = (GameConfig.CENTER-coarse['x_src'].values)*_dx + (GameConfig.CENTER-coarse['y_src'].values)*_dy
    _t   = np.clip(_dot / np.where(_l2==0, 1.0, _l2), 0.0, 1.0)
    _proj = np.sqrt(
        (GameConfig.CENTER-coarse['x_src'].values - _t*_dx)**2 +
        (GameConfig.CENTER-coarse['y_src'].values - _t*_dy)**2)
    _sun_dist = np.where(_l2==0,
        np.sqrt((GameConfig.CENTER-coarse['x_src'].values)**2+(GameConfig.CENTER-coarse['y_src'].values)**2),
        _proj)
    _crossing = _sun_dist < (GameConfig.SUN_RADIUS + GameConfig.PLANET_MARGIN)

    coarse = (
        coarse.assign(_cross=_crossing)
        .loc[lambda d:
            (d['dist_tgt_src'] < (d['step_diff']+1)*GameConfig.MAX_SPEED
             + d['radius_src'] + GameConfig.PLANET_MARGIN + d['radius']
             + d['planet_disp'].fillna(0.0))
            & ~d['_cross']]
        .drop(columns='_cross').reset_index(drop=True)
    )
    if coarse.empty:
        return pd.DataFrame()

    # Ships expansion: only SHIPS_OPTIONS
    expanded = (
        coarse.assign(_key=1)
        .merge(pd.DataFrame({'ships_sent': SHIPS_OPTIONS, '_key': 1}), on='_key')
        .drop(columns='_key').reset_index(drop=True)
    )

    # Phase B: fleet-speed filter
    _fs_ratio = np.clip(
        np.log(expanded['ships_sent'].values.astype(float)) / math.log(1000.0), 0, None)
    _speed    = 1.0 + (GameConfig.MAX_SPEED-1.0) * _fs_ratio**1.5
    _dist_min = expanded['step_diff'].values * _speed + GameConfig.PLANET_MARGIN + expanded['radius_src'].values
    _dist_prev= _dist_min - _speed

    prev_pos = (
        df_s[['id','step','x','y']]
        .assign(step=lambda d: d['step']+1)
        .rename(columns={'x':'x_prev','y':'y_prev'})
    )
    expanded = (
        expanded
        .assign(fleet_speed=_speed, dist_min=_dist_min, dist_prev=_dist_prev)
        .loc[lambda d: d['dist_tgt_src'] < d['dist_min']+d['fleet_speed']+d['radius']+GameConfig.PLANET_MOVEMENT_SLACK]
        .merge(prev_pos, on=['id','step'], how='left')
        .reset_index(drop=True)
    )
    if expanded.empty:
        return pd.DataFrame()

    # Swept-pair collision (vectorised)
    _dx2  = expanded['x'].values - expanded['x_src'].values
    _dy2  = expanded['y'].values - expanded['y_src'].values
    _d2   = expanded['dist_tgt_src'].values
    _ux   = _dx2 / np.where(_d2 < 1e-9, 1.0, _d2)
    _uy   = _dy2 / np.where(_d2 < 1e-9, 1.0, _d2)
    _xpf  = expanded['x_prev'].fillna(expanded['x']).values
    _ypf  = expanded['y_prev'].fillna(expanded['y']).values
    _fx0  = expanded['x_src'].values + _ux * expanded['dist_prev'].values
    _fy0  = expanded['y_src'].values + _uy * expanded['dist_prev'].values
    _pvx  = expanded['x'].values - _xpf
    _pvy  = expanded['y'].values - _ypf
    _dvx  = _ux*expanded['fleet_speed'].values - _pvx
    _dvy  = _uy*expanded['fleet_speed'].values - _pvy
    _d0x  = _fx0 - _xpf
    _d0y  = _fy0 - _ypf
    _a_   = _dvx**2 + _dvy**2
    _b_   = 2.0*(_d0x*_dvx + _d0y*_dvy)
    _c_   = _d0x**2 + _d0y**2 - expanded['radius'].values**2
    _disc = _b_**2 - 4.0*_a_*_c_
    _sq   = np.sqrt(np.clip(_disc, 0, None))
    _t1   = np.where(_a_ < 1e-12, 0.0, (-_b_-_sq)/(2.0*_a_))
    _t2   = np.where(_a_ < 1e-12, 1.0, (-_b_+_sq)/(2.0*_a_))
    _coll = np.where(_a_ < 1e-12, _c_ <= 0.0, (_disc >= 0.0) & (_t2 >= 0.0) & (_t1 <= 1.0))

    return (
        expanded.assign(collision=_coll)
        .loc[lambda d: d['collision']]
        .reset_index(drop=True)
    )
```

- [ ] **Step 2: Smoke test**

```python
exec(open('96-library.py').read())
import math
R = lambda p: 1 + math.log(p)
obs = Obs(planets=[[0,0,30.,30.,R(2),50,2],[1,-1,40.,40.,R(3),30,3]], angular_velocity=0.03)
df_s, pd_ = get_obs_dataframe(obs, 0)
pa = get_opportunities(df_s, pd_, {0})
assert not pa.empty, "Expected some opportunities"
assert set(pa['ships_sent'].unique()).issubset(set(SHIPS_OPTIONS))
print("get_opportunities OK, rows:", len(pa))
```

Expected: `get_opportunities OK, rows: <positive number>`

- [ ] **Step 3: Commit**

```bash
git add 96-library.py
git commit -m "feat: add get_opportunities(source_planet_ids) with SHIPS_OPTIONS expansion"
```

---

### Task 4: Add `build_edge_features` to `96-library.py`

**Files:**
- Modify: `96-library.py`

- [ ] **Step 1: Append `build_edge_features`**

```python
def build_edge_features(pa: pd.DataFrame) -> dict:
    """Returns dict[(src_id, dst_id) -> np.ndarray(50, float32)].

    feature[s_idx * 10 + (step_diff-1)] = 1.0
    where s_idx = index of ships_sent in SHIPS_OPTIONS.
    """
    result = {}
    if pa.empty:
        return result

    for (src_id, dst_id), grp in pa.groupby(['id_src', 'id']):
        feat = np.zeros(50, dtype=np.float32)
        for _, row in grp.iterrows():
            s_idx = _SHIP_TO_IDX.get(int(row['ships_sent']))
            if s_idx is None:
                continue
            e_idx = int(row['step_diff']) - 1
            if 0 <= e_idx < 10:
                feat[s_idx * 10 + e_idx] = 1.0
        result[(int(src_id), int(dst_id))] = feat

    return result
```

- [ ] **Step 2: Verify edge feature shape and values**

```python
exec(open('96-library.py').read())
import math
R = lambda p: 1 + math.log(p)
obs = Obs(planets=[[0,0,30.,30.,R(2),50,2],[1,-1,40.,40.,R(3),30,3]], angular_velocity=0.03)
df_s, pd_ = get_obs_dataframe(obs, 0)
pa = get_opportunities(df_s, pd_, {0})
ef = build_edge_features(pa)
assert (0, 1) in ef, "Expected edge (0,1)"
feat = ef[(0, 1)]
assert feat.shape == (50,), f"Shape {feat.shape}"
assert feat.sum() > 0, "Expected at least one reachable (ships, ETA) pair"
assert feat.max() == 1.0 and feat.min() == 0.0
print("build_edge_features OK, nonzero slots:", int(feat.sum()))
```

- [ ] **Step 3: Commit**

```bash
git add 96-library.py
git commit -m "feat: add build_edge_features to 96-library.py"
```

---

### Task 5: Add `build_hetero_data` to `96-library.py`

**Files:**
- Modify: `96-library.py`

- [ ] **Step 1: Append `build_hetero_data`**

```python
_LOG1024 = math.log(1024)


def build_hetero_data(obs, step: int, label: int, player_id: int = 0) -> HeteroData:
    """Convert an Obs into a PyG HeteroData graph with label."""
    # ── simulate snapshots ────────────────────────────────────────────────────
    df_s, planet_disp = get_obs_dataframe(obs, step)

    # ── attack opportunities from player's planets ─────────────────────────
    source_ids = {p[0] for p in obs.planets if p[1] == player_id}
    pa = get_opportunities(df_s, planet_disp, source_ids)
    edge_feats = build_edge_features(pa)

    # ── planet node features (22-dim) ─────────────────────────────────────
    planets   = obs.planets
    planet_ids = [p[0] for p in planets]
    pid_to_idx = {pid: i for i, pid in enumerate(planet_ids)}

    # ships time-series per planet: pivot df_s → (planet, step) → ships
    ships_pivot = (
        df_s[df_s['step'].isin(range(step, step+11))]
        .pivot_table(index='id', columns='step', values='ships', aggfunc='first')
        .reindex(index=planet_ids, columns=range(step, step+11), fill_value=0)
    )

    # nature at step0
    nature_at0 = df_s[df_s['step'] == step].set_index('id')['nature']

    planet_feats = []
    for p in planets:
        pid, owner, x, y, radius, ships0, production = p
        nat = nature_at0.get(pid, 'fix')
        is_fix    = 1.0 if nat == 'fix'    else 0.0
        is_moving = 1.0 if nat == 'moving' else 0.0
        is_comet  = 1.0 if nat == 'comet'  else 0.0

        # owner one-hot: index 0 = neutral(-1), 1..4 = players 0..3
        owner_oh = [0.0] * 5
        if owner == -1:
            owner_oh[0] = 1.0
        elif 0 <= owner <= 3:
            owner_oh[owner + 1] = 1.0

        # log-ships time-series (11 values)
        ships_ts = ships_pivot.loc[pid].values.tolist()
        ships_feats = [math.log(max(min(float(s), 1024.0), 1.0)) / _LOG1024
                       for s in ships_ts]

        planet_feats.append([x/100.0, y/100.0, is_fix, is_moving, is_comet,
                              production/5.0] + owner_oh + ships_feats)

    # ── master node features (6-dim) ─────────────────────────────────────
    total_ships = sum(max(p[5], 0) for p in planets) or 1.0
    proportions = [
        sum(p[5] for p in planets if p[1] == pid_) / total_ships
        for pid_ in range(4)
    ]
    master_feat = [
        step / 500.0,
        (obs.angular_velocity - 0.025) / (0.05 - 0.025),
    ] + proportions

    # ── assemble HeteroData ───────────────────────────────────────────────
    data = HeteroData()
    data['master'].x = torch.tensor([master_feat], dtype=torch.float)
    data['planet'].x = torch.tensor(planet_feats, dtype=torch.float)

    n = len(planet_ids)
    data['planet', 'to_master', 'master'].edge_index = torch.tensor(
        [list(range(n)), [0]*n], dtype=torch.long)
    data['master', 'to_planet', 'planet'].edge_index = torch.tensor(
        [[0]*n, list(range(n))], dtype=torch.long)

    if edge_feats:
        srcs = [pid_to_idx[s] for s, _ in edge_feats]
        dsts = [pid_to_idx[d] for _, d in edge_feats]
        attrs = np.stack(list(edge_feats.values()))
        data['planet', 'attacks', 'planet'].edge_index = torch.tensor(
            [srcs, dsts], dtype=torch.long)
        data['planet', 'attacks', 'planet'].edge_attr = torch.tensor(
            attrs, dtype=torch.float)
    else:
        data['planet', 'attacks', 'planet'].edge_index = torch.zeros(
            (2, 0), dtype=torch.long)
        data['planet', 'attacks', 'planet'].edge_attr = torch.zeros(
            (0, 50), dtype=torch.float)

    data.y = torch.tensor([float(label)], dtype=torch.float)
    return data
```

- [ ] **Step 2: Verify tensor shapes**

```python
exec(open('96-library.py').read())
import math
R = lambda p: 1 + math.log(p)
obs = Obs(planets=[[0,0,30.,30.,R(2),50,2],[1,-1,40.,40.,R(3),30,3]], angular_velocity=0.03)
d = build_hetero_data(obs, step=0, label=1)
assert d['master'].x.shape  == (1,  6), d['master'].x.shape
assert d['planet'].x.shape  == (2, 22), d['planet'].x.shape
assert d['planet','attacks','planet'].edge_attr.shape[1] == 50
assert d.y.shape == (1,)
print("build_hetero_data OK")
print("  planet feats:", d['planet'].x.shape)
print("  attack edges:", d['planet','attacks','planet'].edge_index.shape)
print("  edge attr:   ", d['planet','attacks','planet'].edge_attr.shape)
```

- [ ] **Step 3: Commit**

```bash
git add 96-library.py
git commit -m "feat: add build_hetero_data to 96-library.py"
```

---

### Task 6: Add `generate_sample` to `96-library.py` and verify full pipeline

**Files:**
- Modify: `96-library.py`

- [ ] **Step 1: Append `generate_sample`**

```python
def generate_sample(seed: int, player_id: int = 0):
    """Returns (HeteroData, snapshots).

    snapshots is a list of 11 dicts {step, planets, fleets} for make_animation.
    label = 1 if my_planet.ships > neutral_planet.ships at step 0.
    """
    rng = np.random.default_rng(seed)
    angular_velocity = float(rng.uniform(0.025, 0.05))

    # Place my planet — reject if too close to sun at (50,50)
    while True:
        x0 = float(rng.uniform(10, 90))
        y0 = float(rng.uniform(10, 90))
        if math.hypot(x0 - 50, y0 - 50) >= 15:
            break
    prod0   = int(rng.integers(1, 6))
    ships0  = int(rng.integers(10, 101))
    radius0 = 1.0 + math.log(prod0)

    # Place neutral planet within 5–20 units, reject if too close to sun
    while True:
        angle = float(rng.uniform(0, 2 * math.pi))
        dist  = float(rng.uniform(5, 20))
        x1 = x0 + dist * math.cos(angle)
        y1 = y0 + dist * math.sin(angle)
        if (10 <= x1 <= 90 and 10 <= y1 <= 90
                and math.hypot(x1 - 50, y1 - 50) >= 15):
            break
    prod1   = int(rng.integers(1, 6))
    ships1  = int(rng.integers(10, 101))
    radius1 = 1.0 + math.log(prod1)

    planets = [
        [0, player_id, x0, y0, radius0, ships0, prod0],
        [1,         -1, x1, y1, radius1, ships1, prod1],
    ]
    obs = Obs(planets=planets, angular_velocity=angular_velocity)

    label = 1 if ships0 > ships1 else 0

    # Collect snapshots for animation (11 steps, no actions)
    sim = copy.deepcopy(obs)
    snapshots = []
    for i in range(11):
        snapshots.append({
            'step':    i,
            'planets': [p[:] for p in sim.planets],
            'fleets':  [f[:] for f in sim.fleets],
        })
        interpreter(sim, [[], []], i)

    data = build_hetero_data(obs, step=0, label=label, player_id=player_id)
    return data, snapshots
```

- [ ] **Step 2: Verify full pipeline with seed=0**

```python
exec(open('96-library.py').read())
data, snaps = generate_sample(0)
assert data['master'].x.shape  == (1,  6)
assert data['planet'].x.shape  == (2, 22)
assert data['planet','attacks','planet'].edge_attr.shape == (1, 50)
assert data.y.shape == (1,)
assert len(snaps) == 11
assert data.y.item() in (0.0, 1.0)
print(f"generate_sample(0) OK — label={int(data.y.item())}, snaps={len(snaps)}")

# Verify reproducibility
data2, _ = generate_sample(0)
assert torch.allclose(data['planet'].x, data2['planet'].x)
print("Reproducibility OK")
```

- [ ] **Step 3: Commit**

```bash
git add 96-library.py
git commit -m "feat: add generate_sample to 96-library.py — full pipeline complete"
```

---

### Task 7: Create `96-GNN_test.ipynb` — setup and dataset cells

**Files:**
- Create: `96-GNN_test.ipynb`

Use `mcp__jupyter__notebook_create` (or Write tool with raw JSON) to create the notebook, then `mcp__jupyter__notebook_add_cell` for each cell.

- [ ] **Step 1: Create the notebook file**

```json
{
  "nbformat": 4,
  "nbformat_minor": 5,
  "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
  "cells": []
}
```

Or via MCP: `mcp__jupyter__notebook_create(path="96-GNN_test.ipynb")`

- [ ] **Step 2: Cell 0 — Markdown title**

```markdown
# 96 — GNN Edge Classifier Test

Heterogeneous GNN (`HeteroConv` + `NNConv`) for directed attack-edge classification.
2-planet scenario: one owned by player 0, one neutral.
Label = 1 if my ships > neutral ships at step 0.
```

- [ ] **Step 3: Cell 1 — imports**

```python
%run 96-library.py
import torch
import torch.nn as nn
from torch_geometric.nn import SAGEConv, NNConv
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML, display
import copy, math, random
```

- [ ] **Step 4: Cell 2 — animation helpers (copied from `94-93-Load_Test_Enhance.ipynb`)**

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
            ax.text(x, y,   str(ships),          ha='center', va='center', color='white', fontsize=7, fontweight='bold', zorder=4)
            ax.text(x, y+2, str(pid),             ha='center', va='center', color='red',   fontsize=7, fontweight='bold', zorder=4)
            ax.text(x, y-2, '+'+str(production),  ha='center', va='center', color='white', fontsize=5, fontweight='bold', zorder=4)
        for f in snap['fleets']:
            fid, owner, x, y, ang, from_id, ships = f
            ax.plot(x, y, 'D', color=_COLORS.get(owner,'#888888'), markersize=5, zorder=5)
        return []
    ani = animation.FuncAnimation(fig, draw, frames=len(snapshots), interval=interval)
    plt.close()
    return HTML(ani.to_jshtml())
```

- [ ] **Step 5: Cell 3 — generate and inspect one sample**

```python
data0, snaps0 = generate_sample(42)
print("master x:", data0['master'].x.shape)
print("planet x:", data0['planet'].x.shape)
print("attack edge_index:", data0['planet','attacks','planet'].edge_index.shape)
print("attack edge_attr: ", data0['planet','attacks','planet'].edge_attr.shape)
print("label:", int(data0.y.item()))
make_animation(snaps0, title='Sample seed=42', interval=200)
```

- [ ] **Step 6: Cell 4 — generate datasets**

```python
print("Generating train dataset (100 samples)…")
train_dataset = [generate_sample(i)     for i in range(100)]
print("Generating test dataset (10 samples)…")
test_dataset  = [generate_sample(1000+i) for i in range(10)]
train_labels = [int(d.y.item()) for d, _ in train_dataset]
test_labels  = [int(d.y.item()) for d, _ in test_dataset]
print(f"Train label distribution: {train_labels.count(1)} pos / {train_labels.count(0)} neg")
print(f"Test  label distribution: {test_labels.count(1)} pos / {test_labels.count(0)} neg")
```

- [ ] **Step 7: Run cells 0–4 and verify no errors, dataset sizes correct**

Expected output from cell 4:
```
Generating train dataset (100 samples)…
Generating test dataset (10 samples)…
Train label distribution: ~50 pos / ~50 neg
Test  label distribution: ~5 pos / ~5 neg
```

- [ ] **Step 8: Commit**

```bash
git add 96-GNN_test.ipynb
git commit -m "feat: 96-GNN_test.ipynb cells 0-4 — setup, animation helpers, datasets"
```

---

### Task 8: Add `GNNEdgeClassifier` model — cell 5

**Files:**
- Modify: `96-GNN_test.ipynb`

- [ ] **Step 1: Add cell 5 with model definition**

```python
class GNNEdgeClassifier(nn.Module):
    """2-layer heterogeneous GNN for directed edge classification.

    Node types: master (6-dim), planet (22-dim)
    Edge types:
      (planet, to_master, master) — SAGEConv, no edge features
      (master, to_planet, planet) — SAGEConv, no edge features
      (planet, attacks,   planet) — NNConv, 50-dim edge features
    """
    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        H = hidden_dim

        self.master_lin = nn.Linear(6,  H)
        self.planet_lin = nn.Linear(22, H)

        # Layer 1
        self.sage1_pm = SAGEConv(H, H)
        self.sage1_mp = SAGEConv(H, H)
        self.nnconv1  = NNConv(H, H, nn=nn.Linear(50, H * H))

        # Layer 2
        self.sage2_pm = SAGEConv(H, H)
        self.sage2_mp = SAGEConv(H, H)
        self.nnconv2  = NNConv(H, H, nn=nn.Linear(50, H * H))

        # Edge classifier: [h_src | h_dst | edge_attr] → logit
        self.edge_mlp = nn.Sequential(
            nn.Linear(H * 2 + 50, H),
            nn.ReLU(),
            nn.Linear(H, 1),
        )

    def _pass(self, h_p, h_m, data, sage_pm, sage_mp, nnconv):
        ei_pm = data['planet', 'to_master', 'master'].edge_index
        ei_mp = data['master', 'to_planet', 'planet'].edge_index
        ei_pp = data['planet', 'attacks',   'planet'].edge_index
        ea_pp = data['planet', 'attacks',   'planet'].edge_attr
        new_m = sage_pm((h_p, h_m), ei_pm)
        new_p = sage_mp((h_m, h_p), ei_mp) + nnconv(h_p, ei_pp, ea_pp)
        return torch.relu(new_p), torch.relu(new_m)

    def forward(self, data):
        h_m = torch.relu(self.master_lin(data['master'].x))
        h_p = torch.relu(self.planet_lin(data['planet'].x))
        h_p, h_m = self._pass(h_p, h_m, data, self.sage1_pm, self.sage1_mp, self.nnconv1)
        h_p, h_m = self._pass(h_p, h_m, data, self.sage2_pm, self.sage2_mp, self.nnconv2)
        ei = data['planet', 'attacks', 'planet'].edge_index
        ea = data['planet', 'attacks', 'planet'].edge_attr
        edge_in = torch.cat([h_p[ei[0]], h_p[ei[1]], ea], dim=-1)
        return self.edge_mlp(edge_in).squeeze(-1)


model = GNNEdgeClassifier(hidden_dim=64)
n_params = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {n_params:,}")

# Shape check on sample 0
data0, _ = train_dataset[0]
logit = model(data0)
assert logit.shape == (1,), f"Expected (1,), got {logit.shape}"
print(f"Forward pass OK — logit: {logit.item():.4f}")
```

- [ ] **Step 2: Run cell 5 and verify forward pass produces shape (1,)**

Expected:
```
Model parameters: ~270,000
Forward pass OK — logit: <some float>
```

- [ ] **Step 3: Commit**

```bash
git add 96-GNN_test.ipynb
git commit -m "feat: add GNNEdgeClassifier (SAGEConv + NNConv) to notebook"
```

---

### Task 9: Add training loop — cell 6

**Files:**
- Modify: `96-GNN_test.ipynb`

- [ ] **Step 1: Add cell 6**

```python
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.BCEWithLogitsLoss()

model.train()
for epoch in range(50):
    total_loss = 0.0
    for data, _ in train_dataset:
        optimizer.zero_grad()
        logit = model(data)
        loss  = criterion(logit, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    if epoch % 10 == 0:
        print(f"Epoch {epoch:3d} | avg loss {total_loss / len(train_dataset):.4f}")

print("Training complete.")
```

- [ ] **Step 2: Run cell 6 and verify loss decreases over 50 epochs**

Expected output (approximate):
```
Epoch   0 | avg loss ~0.69
Epoch  10 | avg loss <0.69
Epoch  20 | avg loss decreasing
...
Training complete.
```

- [ ] **Step 3: Commit**

```bash
git add 96-GNN_test.ipynb
git commit -m "feat: add training loop (BCEWithLogitsLoss, Adam, 50 epochs)"
```

---

### Task 10: Add evaluation metrics and test animations — cells 7–17

**Files:**
- Modify: `96-GNN_test.ipynb`

- [ ] **Step 1: Add cell 7 — evaluation metrics**

```python
model.eval()
preds, labels_list = [], []
with torch.no_grad():
    for data, _ in test_dataset:
        logit = model(data)
        pred  = int((torch.sigmoid(logit) > 0.5).item())
        preds.append(pred)
        labels_list.append(int(data.y.item()))

print(f"Test Accuracy: {accuracy_score(labels_list, preds):.2f}\n")
print(classification_report(labels_list, preds, target_names=['no-attack','attack']))
```

- [ ] **Step 2: Add cells 8–17 — one animation per test sample**

Each cell follows the same pattern (repeat for i = 0..9):

```python
# Cell 8 (i=0): Test sample 0
i = 0
data_i, snaps_i = test_dataset[i]
with torch.no_grad():
    pred_i = int((torch.sigmoid(model(data_i)) > 0.5).item())
true_i = int(data_i.y.item())
print(f"Sample {i}: pred={pred_i}  true={true_i}  {'✓' if pred_i==true_i else '✗'}")
make_animation(snaps_i, title=f'Test {i} | pred={pred_i} true={true_i}', interval=200)
```

Repeat the same code block in cells 9–17 with `i = 1` through `i = 9`.

- [ ] **Step 3: Run cells 7–17 and verify all 10 animations render**

Each animation should show 11 frames. Cell 7 should print a classification report with F1 scores.

- [ ] **Step 4: Final commit**

```bash
git add 96-GNN_test.ipynb
git commit -m "feat: add evaluation metrics and 10 test animations to 96-GNN_test.ipynb"
```

---

## Self-Review

**Spec coverage check:**

| Spec requirement | Task |
|-----------------|------|
| Master node 6-dim features | Task 5 `build_hetero_data` |
| Planet node 22-dim features | Task 5 `build_hetero_data` |
| 50-dim edge feature via swept-pair (vectorised `_02`) | Tasks 3+4 |
| HeteroConv + NNConv architecture | Task 8 |
| 2-layer conv + edge MLP | Task 8 |
| generate_sample with sun rejection + random ω | Task 6 |
| 100 train / 10 test datasets | Task 7 |
| BCEWithLogitsLoss, Adam, 50 epochs | Task 9 |
| accuracy + classification_report | Task 10 |
| 10 test animations with pred/true labels | Task 10 |
| `%run 96-library.py` instead of 93 | Tasks 1+7 |

**No placeholders found.**

**Type consistency:** `generate_sample` returns `(HeteroData, list[dict])` — used as `(data, snaps)` throughout tasks 7–10. ✓  `build_edge_features` returns `dict[(int,int), ndarray]` — consumed correctly by `build_hetero_data`. ✓  `get_opportunities` returns a DataFrame with columns `id_src`, `id`, `ships_sent`, `step_diff` — consumed by `build_edge_features`. ✓
