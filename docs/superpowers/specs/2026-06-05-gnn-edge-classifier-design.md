# GNN Edge Classifier — Design Spec
Date: 2026-06-05
Notebook: `96-GNN_test.ipynb`

## Goal

Train a heterogeneous GNN (PyTorch Geometric `HeteroConv` + `NNConv`) to classify directed
planet-to-planet attack edges as "relevant to keep" or not, starting from a minimal 2-planet
scenario.

---

## Graph Structure

### Node types

| Type | Count | Features | Dim |
|------|-------|----------|-----|
| `master` | 1 | `step/500`, `(ω−0.025)/(0.05−0.025)`, ship proportions for players 0–3 | 6 |
| `planet` | N | See below | 22 |

**Planet node features (22-dim):**
1. `x / 100`
2. `y / 100`
3–5. `is_fixed`, `is_moving`, `is_comet` (one-hot, 3-dim)
6. `production / 5`
7–11. owner one-hot: index 0 = neutral (−1), 1–4 = players 0–3 (5-dim)
12–22. `log(min(ships_t, 1024)) / log(1024)` for `t = step` to `step+10` (11 values)

### Edge types

| Relation | Dir | Edge features | Notes |
|----------|-----|---------------|-------|
| `(planet, to_master, master)` | p→m | none | fan-in |
| `(master, to_planet, planet)` | m→p | none | broadcast |
| `(planet, attacks, planet)` | directed | 50-dim binary | see below |

**50-dim attack edge feature:**
`ships_options = [1, 2, 4, 8, 16]` (5 values), `eta_options = [1…10]` (10 values).
`feature[s_idx * 10 + e_idx] = 1` iff a fleet of `ships_options[s_idx]` ships launched from
`src` toward `dst` first collides with `dst` at step `e_idx + 1`, using the swept-pair
simulation from `93-Supplier_to_moving.py` (`swept_pair_hit`).
At most one `1` per row of 10 (per `ships_sent` value); zero if unreachable in 10 steps.

---

## Data Generation

**`generate_sample(seed: int) → HeteroData`**

```
rng = np.random.default_rng(seed)
angular_velocity ~ Uniform(0.025, 0.05)
step = 0

My planet (owner=0):
  (x, y) ~ Uniform([10,90]²), reject if dist((x,y),(50,50)) < 15
  ships ~ Uniform(10, 100), production ~ Uniform(1, 5)
  radius = 1 + log(production)

Neutral planet (owner=-1):
  angle ~ Uniform(0, 2π), dist ~ Uniform(5, 20) from my planet
  (x,y) = my_planet + dist*(cos,sin), reject if dist((x,y),(50,50)) < 15
  ships ~ Uniform(10, 100), production ~ Uniform(1, 5), radius = 1 + log(production)

Simulate 11 snapshots (step 0…10) via interpreter loop (no actions)
  → fills ships time-series for planet node features

Compute 50-dim edge feature for (planet_0 → planet_1):
  angle = atan2(y_dst - y_src, x_dst - x_src)
  fleet_pos = src_surface + direction * 0.1
  For s_idx, ships in enumerate([1,2,4,8,16]):
    speed = PhysicsEngine.fleet_speed(ships)
    simulate fleet step by step (up to 10)
    at each step e, check swept_pair_hit(f_old, f_new, p_old_dst, p_new_dst, r_dst)
    first collision step → set feature[s_idx*10 + (e-1)] = 1, break

Label: y = 1 if ships(planet_0, t=0) > ships(planet_1, t=0), else 0

HeteroData assembly:
  data['master'].x          shape (1, 6)
  data['planet'].x          shape (2, 22)
  data[planet→master].edge_index  [[0,1],[0,0]]
  data[master→planet].edge_index  [[0,0],[0,1]]
  data[planet,attacks,planet].edge_index  [[0],[1]]
  data[planet,attacks,planet].edge_attr   shape (1, 50)
  data.y  scalar tensor {0, 1}
```

**Datasets:**
- `train_dataset = [generate_sample(i) for i in range(100)]`
- `test_dataset  = [generate_sample(1000+i) for i in range(10)]`

---

## Model Architecture

```
Input projections
  master_lin:  Linear(6  → 64)
  planet_lin:  Linear(22 → 64)

HeteroConv layer 1
  (planet, to_master, master): SAGEConv(64 → 64)
  (master, to_planet, planet): SAGEConv(64 → 64)
  (planet, attacks,   planet): NNConv(64 → 64, nn=MLP(50 → 64×64))
  aggregation: "sum"  →  ReLU

HeteroConv layer 2  (identical structure)

Edge classifier
  for each (attacks) edge (i → j):
    concat [h_planet[i] ‖ h_planet[j] ‖ edge_attr]  →  dim 178
    Linear(178 → 64) → ReLU → Linear(64 → 1)
    → raw logit
```

---

## Training & Evaluation

| Setting | Value |
|---------|-------|
| Loss | `BCEWithLogitsLoss` |
| Optimizer | `Adam(lr=1e-3)` |
| Epochs | 50 |
| Batch | full-batch (one graph at a time) |
| Log interval | every 10 epochs |

**Evaluation (test set):**
- Threshold logit at 0.5 after sigmoid
- Report `accuracy_score` and `classification_report` (sklearn)

**Test animations:**
- For each of the 10 test samples reuse `make_animation` from `94-93-Load_Test_Enhance.ipynb`
- Title each animation with `pred=X  true=Y`

---

## 96-library.py

A clean rewrite of `93-Supplier_to_moving.py` tailored for GNN graph construction.
The notebook does `%run 96-library.py` instead of `%run 93-Supplier_to_moving.py`.

**Keep unchanged:** `GameConfig`, `PhysicsEngine`, `interpreter`, `Obs`

**Modified `get_opportunities(df_s, planet_disp, source_planet_ids: set)`:**
- Replaces `player_id` ownership filter with explicit `source_planet_ids` — uses step-0
  positions of those IDs as sources, no multi-step stability check
- Replaces range expansion with `SHIPS_OPTIONS = [1, 2, 4, 8, 16]` directly (5 values × all
  reachable `step_diff` in 1–10), avoiding the explode over hundreds of values

**New helpers:**

`build_edge_features(pa, all_planet_ids) → dict[(src_id, dst_id), np.ndarray(50)]`
- Groups `pa` by `(id_src, id)`
- For each row maps `(ships_sent, step_diff)` → `feature[s_idx * 10 + (step_diff-1)] = 1`
  where `s_idx = SHIPS_OPTIONS.index(ships_sent)`
- Any `(src, dst)` pair absent from `pa` gets a zero vector

`build_hetero_data(obs, step, label) → HeteroData`
- Calls `get_obs_dataframe` → `get_opportunities` → `build_edge_features`
- Assembles master node (6-dim), planet nodes (22-dim), all edge types
- Attaches `data.y = label`

`generate_sample(seed) → tuple[HeteroData, list[dict]]`
- Random 2-planet placement with sun-distance rejection (dist to (50,50) > 15)
- `angular_velocity ~ Uniform(0.025, 0.05)`
- Returns `(hetero_data, snapshots)` where snapshots are the 11-step interpreter outputs
  for use in `make_animation`

## Notebook Structure

| Cell | Content |
|------|---------|
| 0 | Markdown title |
| 1 | `%run 96-library.py` + imports |
| 2 | Animation helpers (copied from `94`) |
| 3 | Generate + inspect one sample |
| 4 | Generate train/test datasets |
| 5 | Model definition (`GNNEdgeClassifier`) |
| 6 | Training loop |
| 7 | Evaluation metrics |
| 8–17 | One animation cell per test sample |
