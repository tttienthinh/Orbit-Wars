# GNN Bucket Action Selector — Design Spec
Date: 2026-06-05
Files: `97-library.py`, `97-GNN_bucket.ipynb`

## Goal

A heterogeneous GNN that, given a full game observation, selects which attacks to launch and
exactly how many ships to send — output format `[[src_id, dst_id, eta, ships_to_send], ...]`.
Trained via imitation of the `93-Supplier_to_moving.py` heuristic.

---

## Graph Structure

### Node types

| Type | Count (max) | Features | Dim |
|------|-------------|----------|-----|
| `master` | 1 | step/500, norm ω, ship proportions p0–p3 | 6 |
| `planet` | 44 | x/100, y/100, fix/moving/comet, prod/5, owner 5-hot, log-ships t…t+10 | 22 |
| `action` | 44²×9 ≈ 17K | log(min\_ships)/log(1024), log(max\_ships)/log(1024) | 2 |

**Action node** `(i, j, eta)`: exists iff an owned planet `i` can reach planet `j` in exactly
`eta ∈ {1..9}` steps without crossing the sun. `min_ships` / `max_ships` are the ship-count
bounds that produce this exact ETA, computed by binary search on `swept_pair_hit`.

### Edge types

| Relation | Direction | Count (max) |
|----------|-----------|-------------|
| `(planet, spawns, action)` | planet\_i → action(i,j,eta) | 44²×9 |
| `(action, attacks, planet)` | action(i,j,eta) → planet\_j | 44²×9 |
| `(planet, to_master, master)` | all planets → master | 44 |
| `(master, to_planet, planet)` | master → all planets | 44 |

Total edges ≤ 44²×9×2 + 88 ≈ 35K.

---

## Data Generation

**`generate_sample(seed, player_id=0) → (HeteroData, snapshots, action_index)`**

```
rng = np.random.default_rng(seed)
angular_velocity ~ Uniform(0.025, 0.05)

Place 5–20 planets:
  - 2–4 owned by player_id, 1–2 owned by player 1, rest neutral (-1)
  - (x,y) ~ Uniform([10,90]²), reject if dist((x,y),(50,50)) < 15
  - Reject if overlaps existing planet (dist < r1+r2+1)
  - ships ~ Uniform(10,100), production ~ Uniform(1,5), radius = 1+log(production)

Enumerate action nodes for each (src ∈ owned, dst ≠ src, eta ∈ 1..9):
  - Skip if path src→dst crosses sun
  - Binary-search min_ships: smallest s such that swept_pair_hit is True at step eta
  - Binary-search max_ships: largest s such that swept_pair_hit is False at step eta-1
  - If min_ships ≤ max_ships: create action node

Label action nodes via heuristic:
  - Run StrategyPipeline._04_score_and_decide on obs → heuristic_moves = [[src, angle, ships], ...]
  - For each heuristic move [src_id, angle, ships_sent]:
      dst_id = planet whose position at step=0 has angle atan2(y-y_src, x-x_src) closest to `angle`
      eta = first eta ∈ 1..9 where min_ships(src,dst,eta) ≤ ships_sent ≤ max_ships(src,dst,eta)
      If matching action node exists: label = 1, ships_target = log(ships_sent)/log(1024)
  - All other action nodes → label = 0, ships_target = log(min_ships)/log(1024)

Assemble HeteroData:
  data['master'].x          shape (1, 6)
  data['planet'].x          shape (N_planets, 22)
  data['action'].x          shape (N_actions, 2)
  data['planet','spawns','action'].edge_index     shape (2, N_actions)
  data['action','attacks','planet'].edge_index    shape (2, N_actions)
  data['planet','to_master','master'].edge_index  shape (2, N_planets)
  data['master','to_planet','planet'].edge_index  shape (2, N_planets)
  data['action'].y           shape (N_actions,)   — binary select label
  data['action'].ships_target shape (N_actions,)  — normalised ships regression target

action_index: list[tuple[int,int,int]]  — maps action node index → (src_id, dst_id, eta)
```

**Datasets:**
- `train = [generate_sample(i) for i in range(1000)]`
- `test  = [generate_sample(10000+i) for i in range(20)]`

---

## Model Architecture

```
H = hidden_dim = 64

Input projections:
  master_lin:  Linear(6,  H)
  planet_lin:  Linear(22, H)
  action_lin:  Linear(2,  H)

Layer 1 — planet context into action nodes:
  h_action = ReLU(sage1_pa((h_planet, h_action), spawns_edge_index))
  h_master = ReLU(sage1_pm((h_planet, h_master), to_master_edge_index))
  h_planet = ReLU(sage1_mp((h_master, h_planet), to_planet_edge_index))

Layer 2 — action context into destination planets:
  h_planet = ReLU(sage2_ap((h_action, h_planet), attacks_edge_index))

Layer 3 — enriched planet context back into action nodes:
  h_action = ReLU(sage3_pa((h_planet, h_action), spawns_edge_index))

Output heads (per action node):
  select_head: Linear(H → 1)  → raw logit
  ships_head:  Linear(H → 1) + Sigmoid → ŝ ∈ [0,1]
```

All SAGEConv layers are `SAGEConv(H, H)` (bipartite form via tuple input).

---

## Training & Loss

| Setting | Value |
|---------|-------|
| Optimizer | Adam(lr=1e-3) |
| Epochs | 50 |
| Batch | one graph at a time |
| Select loss | BCEWithLogitsLoss(select_logit, label) |
| Ships loss | MSELoss(ships_pred, ships_target) |
| Total loss | select_loss + ships_loss |
| Log interval | every 10 epochs |

Save weights to `97-model.pt`; load if exists.

---

## Inference & Post-processing

```python
select_prob = sigmoid(select_logit)   # shape (N_actions,)
ships_norm  = sigmoid(ships_head_out) # shape (N_actions,)

candidates = sorted(enumerate(select_prob), key=lambda x: -x[1])

ships_remaining = {p[0]: p[5] for p in obs.planets if p[1] == player_id}
output = []
for k, prob in candidates:
    if prob < 0.5:
        break
    src_id, dst_id, eta = action_index[k]
    ships = round(math.exp(float(ships_norm[k]) * math.log(1024)))
    ships = max(1, min(ships, ships_remaining.get(src_id, 0)))
    if ships_remaining.get(src_id, 0) >= 1:
        output.append([src_id, dst_id, eta, ships])
        ships_remaining[src_id] -= ships
```

Output: `[[src_id, dst_id, eta, ships_to_send], ...]`

---

## File Map

| File | Responsibility |
|------|---------------|
| `97-library.py` | Physics core (from 96), `get_obs_dataframe`, `get_opportunities`, `enumerate_action_nodes`, `build_hetero_data_97`, `generate_sample_97` |
| `97-GNN_bucket.ipynb` | Dataset generation, `GNNActionSelector` model, training loop, evaluation, inference demo |

`97-library.py` imports and reuses `GameConfig`, `PhysicsEngine`, `interpreter`, `Obs`,
`get_obs_dataframe`, `get_opportunities` from `96-library.py` (via `%run` or direct import).
It adds only the new functions needed for the action-node graph construction.

---

## Notebook Structure

| Cell | Content |
|------|---------|
| 0 | Markdown title |
| 1 | `%run 96-library.py` + `%run 97-library.py` + imports |
| 2 | Animation helpers (from 94) |
| 3 | Inspect one sample — print action node count, label balance |
| 4 | Generate train/test datasets |
| 5 | `GNNActionSelector` model definition |
| 6 | Training loop + save to `97-model.pt` |
| 7 | Evaluation: select accuracy, ships MAE on test set |
| 8–27 | One inference demo per test sample (animation + decoded output list) |
