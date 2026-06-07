# GNN Reach Edges — PlanetStep → PlanetStep

**Date:** 2026-06-07  
**Notebook:** `105-104-AnalyseReach.ipynb`  
**Scope:** Add `('planet_step', 'reaches', 'planet_step')` edges to the existing `data` HeteroData object. No model or training loop.

## Goal

Extend the M6 bipartite graph (built in the previous cell) with directed attack-reachability edges between PlanetStep snapshot nodes. Each row of `pa` encodes a fleet that can travel from planet `id_src` at departure step `step_src` to planet `id` arriving at step `step`, carrying `ships_sent` ships.

## Source Data

```python
pa[['id_src', 'step_src', 'ships_sent', 'id', 'step']]  # 29,080 rows
```

All `(id, step)` pairs in `pa` are guaranteed to exist in `ps_df` (28 planets × steps 0–20).

## Edge Definition

| Property | Value |
|----------|-------|
| Relation | `('planet_step', 'reaches', 'planet_step')` |
| Direction | PlanetStep(id_src, step_src) → PlanetStep(id, step) |
| `edge_index` shape | `[2, 29080]` |
| `edge_attr` shape | `[29080, 1]` — `log(ships_sent) / log(1024)` |

## Implementation

### Step 1 — Build index lookup table

```python
ps_idx = ps_df[['id', 'step']].reset_index().rename(columns={'index': 'ps_idx'})
```

`ps_idx` has columns `id`, `step`, `ps_idx` where `ps_idx` is the integer node index (0..587).

### Step 2 — Vectorised index resolution via merge

```python
src = (
    pa[['id_src', 'step_src']]
    .rename(columns={'id_src': 'id', 'step_src': 'step'})
    .merge(ps_idx, on=['id', 'step'])['ps_idx']
    .values
)
dst = pa[['id', 'step']].merge(ps_idx, on=['id', 'step'])['ps_idx'].values
```

No Python-level loop — fully vectorised. Both merges are inner joins; all keys are guaranteed present.

### Step 3 — Edge feature

```python
edge_attr = torch.tensor(
    (np.log(pa['ships_sent'].values) / np.log(1024)).reshape(-1, 1),
    dtype=torch.float32,
)
```

### Step 4 — Attach to existing `data` object

```python
data['planet_step', 'reaches', 'planet_step'].edge_index = torch.tensor(
    np.stack([src, dst]), dtype=torch.long
)
data['planet_step', 'reaches', 'planet_step'].edge_attr = edge_attr
```

## Expected Output

```
(planet_step, reaches, planet_step)={ edge_index=[2, 29080], edge_attr=[29080, 1] }
```

## Notes

- `ps_df` must already exist from the Planet/PlanetStep construction cell and must not have been re-indexed.
- `pa` must already exist from the reach computation cells.
- This cell appends to `data` in-place; re-running requires re-running the construction cell first.
