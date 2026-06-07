# GNN Graph Construction — M6 Hybrid Bipartite (Planet + PlanetStep)

**Date:** 2026-06-07  
**Notebook:** `105-104-AnalyseReach.ipynb`  
**Scope:** Graph construction only (HeteroData object). No model or training loop.

## Goal

Add a cell at the end of `105-104-AnalyseReach.ipynb` that builds a PyTorch Geometric
`HeteroData` graph from the existing `df_s` DataFrame. This is the base bipartite layer
of M6 from the GNN presentation: static Planet nodes grounded by dynamic PlanetStep
snapshot nodes.

## Graph Structure

```
('planet', 'has_snapshot', 'planet_step')
```

- **28 `planet` nodes** — one per unique planet id, indexed 0..27 (sorted by id)
- **588 `planet_step` nodes** — one per row of df_s (28 planets × 21 steps 0..20)
- **588 edges** — each PlanetStep row points back to its base Planet (by id)

## Node Features

### `planet` — 4-dim

| Feature | Computation |
|---------|------------|
| nature_fix | one-hot from `nature` column |
| nature_moving | one-hot from `nature` column |
| nature_comet | one-hot from `nature` column |
| production | `production / 5` |

Source: `df_s[["id", "production", "nature"]].drop_duplicates().sort_values("id").reset_index(drop=True)`

### `planet_step` — 9-dim

| Feature | Computation |
|---------|------------|
| step | `step / m.GameConfig.NB_STEPS_SIM` |
| x | `x / 100` |
| y | `y / 100` |
| ships | `log(ships) / log(1024)` |
| owner_neg1 | one-hot for owner == -1 (neutral) |
| owner_0 | one-hot for owner == 0 (mine) |
| owner_1 | one-hot for owner == 1 |
| owner_2 | one-hot for owner == 2 |
| owner_3 | one-hot for owner == 3 |

Source: `df_s[['id', 'step', 'x', 'y', 'ships', 'owner']]`

## Edge Construction

- Build a lookup dict `{planet_id: planet_node_index}` from the sorted unique-planet table.
- For each row `i` in `df_s` (PlanetStep index 0..587), emit edge `(planet_id_lookup[df_s.id[i]], i)`.
- Edge type: `('planet', 'has_snapshot', 'planet_step')`.
- No edge features.

## Implementation Notes

- Use `pd.get_dummies` for one-hot encoding; reindex to enforce all 3 nature columns and
  all 5 owner columns even if some values are absent in this replay.
- Convert all feature tensors to `torch.float32`.
- Print node/edge counts and feature shapes as a sanity check.
- One self-contained cell; no new functions or files.

## Dependencies

- `torch` and `torch_geometric` must be importable (PyG installed in the kernel).
- `df_s` and `m.GameConfig.NB_STEPS_SIM` available from earlier cells.
