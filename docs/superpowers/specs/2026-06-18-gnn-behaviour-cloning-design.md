# GNN Behaviour Cloning — Design Spec
Date: 2026-06-18

## Goal

Train an Imitation Learning / Behaviour Cloning model on 126-precompute data.
Given the board state at obs step T, predict which attacks in `reachable_max_ships`
the expert would execute (as recorded in `actions_to_copy`).

The trained model runs as the live game agent within the 1-second turn budget (CPU only).

---

## Data

### Source
- Precomputed parquet files in `126-precompute/{episode_id}/`
- ~500 total episodes

### Train / test split
```
TEST_EPISODE_IDS = {
    78867640, 78899068, 78982947, 79033183,
    79126912, 79175592, 79228392, 79320069,
}
```
Train: all other episodes (~492). Augmented 4× with rotations (0°/90°/180°/270°) on (x,y).

### Parquet schemas
| File | Columns |
|------|---------|
| `planete.parquet` | id, step, x, y, production, nature |
| `planete_step.parquet` | id, step (=obs T), future_step, ships, owner |
| `reachable_base_2.parquet` | id_src, step_src, id_tgt, step_tgt, ships_sent |
| `reachable_max_ships.parquet` | id_src, step_src, angle, ships_sent, id_tgt, step_tgt |
| `actions_to_copy.parquet` | step, id_src, angle, ships_sent, id_tgt |

---

## Graph Structure (per obs step T)

### Node types

| Node | Index | Source | Features | Dim |
|------|-------|--------|----------|-----|
| Planet | `id` | `planete` filter `step==T`, dedup by id | `production/5`, `is_fix`, `is_moving`, `is_comet` | 4 |
| PlanetStep | `(id, future_step)` | `planete_step` filter `step==T` JOIN `planete` on `(id, step=future_step)` | `x/100`, `y/100`, `(future_step-T)/20`, `log(ships)/log(1024)`, owner one-hot `{-1,0,1,2,3}` | 9 |
| ActionMaxShips | `(id_src, id_tgt)` | `reachable_max_ships` filter `step_src==T`, owner_src==0 | `log(ships_sent)/log(1024)` | 1 |

### Edge types

**Phase 1 — context edges**

| Edge | Connects | Feature | Filter | Est. count |
|------|---------|---------|--------|-----------|
| PlanetStepEdge | Planet ↔ PlanetStep, same `id` | none | — | ~924 |
| ActionBase2Edge | PlanetStep(`id_src`,`step_src`) ↔ PlanetStep(`id_tgt`,`step_tgt`) | `log(ships_sent)/log(1024)` scalar | `step_src ∈ [T,T+20]`, `step_tgt ≤ T+20`, `ships_sent ∈ {8,64,512}` | ~28K |

**Phase 2 — decision edges**

| Edge | Connects | Feature | Est. count |
|------|---------|---------|-----------|
| SourceEdge | ActionMaxShips ↔ PlanetStep(`id_src`, `T`) | none | 1–250 |
| TargetEdge | ActionMaxShips ↔ PlanetStep(`id_tgt`, `step_tgt`) | none | 1–250 |

`step_tgt` on TargetEdge comes from `reachable_max_ships.step_tgt`.

All edges are undirected (ToUndirected applied, or both directions stored explicitly).

### Labels

ActionMaxShips node `(id_src, id_tgt)` gets label=1 if `(step=T, id_src, id_tgt)` appears
in `actions_to_copy`. Label=0 otherwise.
Owner filter (owner==0 at step T for id_src) is enforced at ActionMaxShips construction time.

---

## Model Architecture

### Hyperparameters
```
H = 64          # hidden dim
L1 = 3          # Phase 1 layers
L2 = 2          # Phase 2 layers
DEVICE = "cpu"
```

### Input projections
```
Linear(4 → H)   # Planet
Linear(9 → H)   # PlanetStep
Linear(1 → H)   # ActionMaxShips  (Phase 2 only)
```

### Phase 1 — HeteroConv × L1 on {Planet, PlanetStep}

Three relations per layer:

| Relation | Conv |
|----------|------|
| planet ↔ planet_step (PlanetStepEdge) | SAGEConv(H→H) |
| planet_step ↔ planet_step (ActionBase2Edge) | ScaledSAGEConv(H→H) |

`ScaledSAGEConv` is a custom 10-line `MessagePassing` where
`message(x_j, edge_weight) = x_j * edge_weight.unsqueeze(-1)`,
then mean aggregation + linear transform (identical to SAGEConv otherwise).
Used only for ActionBase2Edge to incorporate the fleet-size scalar.

After each layer: `ReLU + LayerNorm` per node type.

### Phase 2 — HeteroConv × L2 on {ActionMaxShips, PlanetStep}

PlanetStep embeddings from Phase 1 carry over. ActionMaxShips starts from its
projected 1-dim feature.

Four relations per layer:

| Relation | Conv |
|----------|------|
| planet_step ↔ action_max (SourceEdge) | SAGEConv(H→H) |
| planet_step ↔ action_max (TargetEdge) | SAGEConv(H→H) |

After each layer: `ReLU + LayerNorm` per node type.

### Classifier
```
Linear(H → 1)  applied to each ActionMaxShips node → logit
```

Total trainable parameters: ~150K.

---

## Training

### Data loading
```
EPISODES_PER_EPOCH = 16   # sampled from train pool
N_STEPS_PER_EPISODE = 16  # sampled uniformly per episode
→ ~256 graphs/epoch
batch_size = 4             # PyG DataLoader, num_workers=0 (CPU/Windows)
→ ~64 optimizer steps/epoch
```

### Loss
```
BCEWithLogitsLoss(pos_weight=w)
w = n_negative_total / n_positive_total  # computed once from full train set scan
```

### Optimizer and scheduler
```
AdamW(lr=1e-3, weight_decay=1e-4)
CosineAnnealingLR(T_max=N_EPOCHS)
N_EPOCHS = 100
```

### Augmentation
Rotations 0°/90°/180°/270° applied to (x, y) coordinates of planete data.
Episode is sampled as (episode_id, rotation) pair.

### Metrics (logged to MLflow per epoch)

Computed on train sample (the 16×16 sampled graphs) and full test set (all steps, 8 episodes):

| Metric | Description |
|--------|-------------|
| `loss` | Mean BCE loss over all ActionMaxShips nodes |
| `roc_auc` | sklearn roc_auc_score |
| `n_ams` | Total ActionMaxShips nodes evaluated |
| `n_pos` | Nodes with label=1 |
| `n_tp` | True positives at threshold 0.5 |
| `n_fp` | False positives at threshold 0.5 |

tqdm bar shows `train_auc`, `test_auc`, `train_loss`, `test_loss`.

### Checkpoints
- `model_best.pt` — saved whenever `test_roc_auc` improves
- `model_epoch{N}.pt` — saved every 10 epochs

---

## Graph Construction (detailed, for implementation)

```python
# Planet nodes
planet_df = (
    planete
    .filter(step == T)
    .unique("id", keep="first")
)
# features: production/5, is_fix, is_moving, is_comet

# PlanetStep nodes
ps_df = (
    planete_step.filter(step == T)          # (id, future_step, ships, owner)
    .join(
        planete.rename({"step": "future_step"}),
        on=["id", "future_step"], how="left"
    )
)
# features: x/100, y/100, (future_step-T)/20, log(ships)/log(1024), owner one-hot

# ActionBase2Edge
b2 = (
    reachable_base_2
    .filter(
        (step_src >= T) & (step_src <= T+20) &
        (step_tgt <= T+20) &
        (ships_sent.is_in([8, 64, 512]))
    )
)
# edge_weight: log(ships_sent)/log(1024)

# ActionMaxShips nodes
ams = (
    reachable_max_ships
    .filter(step_src == T)
    .join(ps_df.filter(future_step == T).select(["id", "owner"]),
          left_on="id_src", right_on="id", how="inner")
    .filter(owner == 0)
)
# features: log(ships_sent)/log(1024)
# SourceEdge: ams.id_src → ps_df.filter(future_step==T)
# TargetEdge: ams.(id_tgt, step_tgt) → ps_df.filter(future_step==step_tgt)

# Labels
labels = ams.join(
    actions_to_copy.filter(step == T).select(["id_src", "id_tgt"]),
    on=["id_src", "id_tgt"], how="left"
).with_columns(label = (pl.col("id_tgt_right").is_not_null()).cast(pl.Int64))
```

---

## Inference (live agent)

```python
def agent(obs):
    game_step = obs.step
    df_s, planet_disp = SP._01_get_obs_dataframe(obs, game_step, num_agents)
    df_s = SP._00_remap_owner(df_s, obs, obs.player)

    reach_b2 = SP._02_get_reach(df_s, planet_disp, [8, 64, 512])
    reach_b2 = SP._03_filter_collision(reach_b2).collect()

    reach_max = _get_reach_max_ships(df_s, planet_disp).collect()
    reach_max = SP._03_filter_collision(reach_max.lazy()).collect()
    reach_max = reach_max.filter(pl.col("owner_src") == 0)

    if reach_max.is_empty():
        return []

    data = build_graph(df_s, reach_b2, reach_max, game_step)
    with torch.no_grad():
        logits = model(data)
    mask = torch.sigmoid(logits).squeeze(-1) > THRESHOLD

    return [
        [row["id_src"], row["angle"], row["ships_sent"]]
        for row, pred in zip(reach_max.iter_rows(named=True), mask.tolist())
        if pred
    ]
```

`THRESHOLD = 0.5` initially; tune post-training using ROC curve from MLflow logs.

---

## Files

| File | Purpose |
|------|---------|
| `128-GNN_BC.py` | Training script |
| `128-GNN_BC/` | Model checkpoints + MLflow logs |
| `agent-128/main.py` | Live agent using trained model |
