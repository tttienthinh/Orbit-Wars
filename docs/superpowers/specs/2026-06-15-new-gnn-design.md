# 117-NewGNN — Two-Tower GNN Design

**Date:** 2026-06-15  
**File:** `117-NewGNN.py`  
**Input:** precomputed parquet files from `114-precompute/{episode_id}/`  
**Task:** Binary classification — for each (id_src, id_tgt) pair at game step `t`, predict whether the winner attacked id_tgt from id_src.

---

## 1. Problem Framing

At each game step `t`, the winner controls some set of planets (owner == 0). The GNN predicts which (src, tgt) pairs were actually attacked — using the precomputed reach and planet-state data as context. Labels come from `actions.parquet` (game_step, id_src, id).

Ships_sent is **not** part of the label or the attack node — the model predicts "did an attack happen" not "how many ships".

---

## 2. Graph Structure (per game step `t`)

One `HeteroData` graph is built per unique `step_src == t` in the episode's reach table.

### Node types

| Type | Count | Features | Dim |
|---|---|---|---|
| `planet` | unique planet IDs in window | `(is_fix, is_moving, is_comet, production/5)` | 4 |
| `planet_step` | `(id, step)` from `df_s` where `step == t` (all planets at current step) **plus** `(id, step)` pairs from `reach_t` arrival steps | `(step_norm, x/100, y/100, log_ships/log(1024), owner_−1, owner_0, owner_1, owner_2, owner_3)` | 9 |

### Edge types

| Edge | Direction | Attr | Dim |
|---|---|---|---|
| `has_snapshot` | `planet → planet_step` | none | — |
| `reaches` | `planet_step(id_src, t) → planet_step(id_tgt, arrival_step)` | `log2(ships_sent) / 10` | 1 |

`ToUndirected()` adds `rev_has_snapshot` and `rev_reaches`.

### Reach preprocessing (applied once per episode load)

```python
reach = (
    reach
    .group_by(['id_src', 'step_src', 'id', 'ships_sent'])
    .agg(pl.all().sort_by('step').first())
)
```

This deduplicates to the earliest arrival step per (src_planet, src_step, tgt_planet, ships_sent).

---

## 3. GNN Architecture

```
planet_proj:       Linear(4, H)
planet_step_proj:  Linear(9, H)

conv_1 … conv_N:   HeteroConv({
    planet → planet_step     (has_snapshot):     SAGEConv((H,H), H)
    planet_step → planet     (rev_has_snapshot): SAGEConv((H,H), H)
    planet_step → planet_step (reaches):         GATConv((H,H), H, edge_dim=1, heads=1)
    planet_step → planet_step (rev_reaches):     SAGEConv((H,H), H)
}) + ReLU
```

Parameterized by `hidden_dim` (default 64) and `num_layers` (default 3).

**Forward pass output:** `h_planet` — shape `(N_planets, H)`.

The Planet embedding is a temporal summary of the planet's full 20-step forward simulation, aggregated from its PlanetStep snapshots via `rev_has_snapshot`. This is richer than a single-step snapshot because it encodes where the planet is headed over the window.

---

## 4. Scoring Head (Two-Tower)

No Attack nodes in the graph. Scoring happens outside the GNN on Planet embeddings.

```python
pair_mlp = Sequential(
    Linear(2*H, H),  ReLU(),
    Linear(H, H//2), ReLU(),
    Linear(H//2, 1),
)

# At game step t:
src_ids = df_s.filter((step == t) & (owner == 0))['id']   # owner-0 planets
tgt_ids = df_s.filter(step == t)['id']                    # all planets at t

for id_src in src_ids:
    for id_tgt in tgt_ids:
        if id_src == id_tgt: continue
        score = pair_mlp(cat(h_planet[planet_idx[id_src]],
                             h_planet[planet_idx[id_tgt]]))
        label = 1 if (t, id_src, id_tgt) in actions else 0
```

Index lookup: `planet_idx: dict[planet_id → row_index]` precomputed per graph (simple dict, no step dimension).

---

## 5. Training Setup

- **Graph granularity:** one `HeteroData` per (episode, game_step)
- **Iteration:** loop over episodes in `114-precompute/`, then over unique `step_src` values
- **Loss:** `BCEWithLogitsLoss(pos_weight=N_neg/N_pos)` — handles ~300:1 imbalance for 30-planet games
- **Optimizer:** Adam, lr=1e-3
- **Batching:** one step = one forward pass; gradients accumulated or `torch_geometric.data.Batch` across steps
- **Train/val split:** by episode ID (not step) to prevent data leakage

---

## 6. File: `117-NewGNN.py`

Sections:
1. Imports and constants
2. `build_graph(df_s_t, reach_t) → HeteroData` — builds one per-step graph
3. `OrbitGNN(hidden_dim, num_layers)` — GNN + `pair_mlp` head
4. `build_attack_pairs(df_s_t, actions_set, planet_idx)` — returns tensors of (src_idx, tgt_idx, label)
5. `train_episode(ep_dir, model, optimizer)` — processes one episode
6. `main()` — loads episodes, trains, saves weights
