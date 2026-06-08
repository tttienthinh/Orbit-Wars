# GraphNN Design — `106-Simulate20Next_GNN.py`

## Overview

Create `106-Simulate20Next_GNN.py` as a copy of `104-Simulate20Next.py` with one addition: a static method `StrategyPipeline._05_get_GNN(df_s, pa, safe_attacks)` that builds a PyTorch Geometric `HeteroData` graph from already-computed pipeline DataFrames.

## File

`106-Simulate20Next_GNN.py` = verbatim copy of `104-Simulate20Next.py`, plus:
- `import torch` and `from torch_geometric.data import HeteroData` at the top
- `_05_get_GNN` static method added to `StrategyPipeline`

## Method Signature

```python
@staticmethod
def _05_get_GNN(
    df_s: pd.DataFrame,
    pa: pd.DataFrame,
    safe_attacks: pd.DataFrame,
) -> HeteroData:
```

- `df_s`: full simulation DataFrame from `_01_get_obs_dataframe` (remapped by `_00_remap_owner`)
- `pa`: all reach opportunities from `_02_get_all_opportunities`
- `safe_attacks`: collision-filtered attacks from `_03_filter_collision`

## Graph Schema

### Nodes

| Type | Count | Features (dim) | Source |
|------|-------|----------------|--------|
| `planet` | 28 (one per unique planet id) | fix/moving/comet dummies + production/5 → **4-dim** | `df_s` dedup by `id` |
| `planet_step` | 28 × (NB_STEPS_SIM+1) | step/NB_STEPS_SIM, x/100, y/100, log(ships)/log(1024), owner_neg1/0/1/2/3 → **9-dim** | `df_s` all rows |
| `attack` | `len(safe_attacks[ships_sent ≤ ships_min])` | log(ships_sent)/log(1024) → **1-dim** | filtered `safe_attacks` |

### Edges

| Relation | Src → Dst | Attr | Source |
|----------|-----------|------|--------|
| `has_snapshot` | `planet` → `planet_step` | none | `id` match |
| `reaches` | `planet_step` → `planet_step` | log(ships_sent)/log(1024) [1-dim] | `pa` |
| `AttackSrc` | `planet_step(id_src, step_src)` → `attack` | none | filtered `safe_attacks` |
| `AttackTgt` | `attack` → `planet_step(id, step)` | none | filtered `safe_attacks` |

## Index Lookups

Two lookup tables built inside `_05_get_GNN`:
- `id_to_pos`: planet `id` → row index in `planets_df` (for `has_snapshot` src)
- `ps_idx`: `(id, step)` → row index in `ps_df` (for `reaches` and attack edges)

## Return

A populated `HeteroData` object ready for use with PyTorch Geometric models.
