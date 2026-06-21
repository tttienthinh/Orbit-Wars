# 132-GNN_BC_shift_torch_only — Design Spec

**Date**: 2026-06-20  
**Output file**: `132-GNN_BC_shift_torch_only.py`  
**Weights**: `130-GNN_BC_shift/model_epoch100.pt` (bundled in zip for Kaggle submission)

## Goal

Create a pure-PyTorch Kaggle submission agent that loads the trained `GNNBehaviourCloning` model
from `130-GNN_BC_shift.py` and predicts which (src→tgt) attacks to execute each step.
Pattern follows `112-GNN_torch_only.py` (pure PyTorch, no torch_geometric dependency, zip bundle).

## Architecture — Pure PyTorch Reimplementation

### Sub-modules (parameter names match PyG to allow direct weight loading)

**`_SAGEConvPure(H, H)`**
- `self.lin_l = nn.Linear(H, H, bias=True)` — root/dst transform
- `self.lin_r = nn.Linear(H, H, bias=False)` — mean-aggregated neighbor transform
- Forward: `lin_l(h_dst) + lin_r(mean_agg(h_src, edge_index, num_dst))`

**`_ScaledSAGEConvPure(H, H)`**
- `self.lin_neigh = nn.Linear(H, H, bias=False)`
- `self.lin_root  = nn.Linear(H, H, bias=True)`
- Forward: `lin_neigh(weighted_mean_agg(h, edge_index, edge_attr)) + lin_root(h)`
- Message: `h_src * edge_attr.view(-1,1)`, then mean-aggregated

**`_Phase1ConvPure(H)`** — one HeteroConv layer on {planet, planet_step}
- `self.has_ps     = _SAGEConvPure(H, H)` — planet → planet_step
- `self.rev_has_ps = _SAGEConvPure(H, H)` — planet_step → planet
- `self.reaches    = _ScaledSAGEConvPure(H, H)` — planet_step → planet_step
- Forward returns (new_h_planet, new_h_ps); planet_step gets summed contributions from has_ps + reaches

**`_Phase2ConvPure(H)`** — one HeteroConv layer on {planet_step, action_max}
- `self.src_of     = _SAGEConvPure(H, H)` — planet_step → action_max (from source)
- `self.rev_src_of = _SAGEConvPure(H, H)` — action_max → planet_step
- `self.tgt_of     = _SAGEConvPure(H, H)` — planet_step → action_max (from target)
- `self.rev_tgt_of = _SAGEConvPure(H, H)` — action_max → planet_step
- Forward returns (new_h_ps, new_h_ams); action_max sums src_of + tgt_of contributions

**`GNNBehaviourCloningPure(H=64, L1=3, L2=2)`**
- `self.proj_planet = nn.Linear(4, H)`  — planet features (4-dim)
- `self.proj_ps     = nn.Linear(9, H)`  — planet_step features (9-dim)
- `self.proj_ams    = nn.Linear(1, H)`  — action_max features (1-dim)
- `self.phase1      = nn.ModuleList([_Phase1ConvPure(H)] * L1)`
- `self.norm1_planet = nn.ModuleList([nn.LayerNorm(H)] * L1)`
- `self.norm1_ps    = nn.ModuleList([nn.LayerNorm(H)] * L1)`
- `self.phase2      = nn.ModuleList([_Phase2ConvPure(H)] * L2)`
- `self.norm2_ps    = nn.ModuleList([nn.LayerNorm(H)] * L2)`
- `self.norm2_ams   = nn.ModuleList([nn.LayerNorm(H)] * L2)`
- `self.classifier  = nn.Linear(H, 1)`
- Forward: Phase1 (L1 × LayerNorm+ReLU) → Phase2 (L2 × LayerNorm+ReLU) → classifier → squeeze(-1)

## Weight Remapping

```python
_KEY_MAP = {
    "convs.<planet___has_ps___planet_step>":        "has_ps",
    "convs.<planet_step___rev_has_ps___planet>":    "rev_has_ps",
    "convs.<planet_step___reaches___planet_step>":  "reaches",
    "convs.<planet_step___src_of___action_max>":    "src_of",
    "convs.<action_max___rev_src_of___planet_step>":"rev_src_of",
    "convs.<planet_step___tgt_of___action_max>":    "tgt_of",
    "convs.<action_max___rev_tgt_of___planet_step>":"rev_tgt_of",
}
```

All other keys (proj_planet, proj_ps, proj_ams, norm1_*, norm2_*, classifier) load directly —
PyG `Linear` and `nn.Linear` have identical weight/bias shapes.

## Inference Pipeline (per step T)

1. `_01_get_obs_dataframe(obs, T, num_agents)` → `df_s` (steps T..T+20), `planet_disp`
2. `_00_remap_owner(df_s, obs, pid)` → player_id remapped to 0
3. `_02_get_reach(df_s, planet_disp, ships_list=[8, 64, 512])` + `_03_filter_collision` → `reachable_base_2`
4. `_get_reach_max_ships(df_s, planet_disp)` + `_03_filter_collision` → `reachable_max_ships`
5. Build `planete` = `df_s[["id","step","x","y","production","nature"]]`
6. Build `planete_step` from df_s (rename step→future_step, add obs step column)
7. `build_graph_inference(T, planete, planete_step, rb2, rms)` → `(graph, action_table)`
   - Same logic as `build_graph()` in 130, but no labels
   - Returns `action_table`: DataFrame with `id_src`, `angle`, `ships_sent` indexed by action_max node idx
8. `model(graph)` → logits shape `(N_ams,)`
9. Select where `logit > log(576)` (≈ 6.36) — calibrated to match training positive rate 1/577
10. Deduplicate: per `id_src`, keep the action_max node with highest logit
11. Return `[[id_src, angle, ships_sent], ...]`

## Threshold Calibration

Training used `BCEWithLogitsLoss(pos_weight=576)`, making classes effectively balanced.
Model outputs σ(z) ≈ P(y=1 | x, balanced prior). To select at the true positive rate (1/577):

```
Select when: σ(z) > 576/577 ≈ 0.9983
Equivalent: logit z > log(576) ≈ 6.358
```

This selects ≈ N_ams/577 candidates per step, matching training (1,209 pos / 518,598 total ≈ 0.23%).
Multiple fleets allowed: one per source planet that exceeds the threshold.

## Data Container

`GraphData` (simple class, no torch_geometric):
- `data["planet"].x`, `data["planet_step"].x`, `data["action_max"].x` → feature tensors
- `data.edge_index_dict` → dict keyed by edge type tuples → edge index tensors
- `data[edge_type].edge_attr` → optional edge attribute tensor (for "reaches" only)

## Files

- `132-GNN_BC_shift_torch_only.py` — the agent script (copied functions from 112/113/126)
- `model_epoch100.pt` — copied from `130-GNN_BC_shift/model_epoch100.pt`, bundled in zip

## Functions to copy/adapt

| Source | Function | Used for |
|--------|----------|----------|
| `112-GNN_torch_only.py` | interpreter, `_01_get_obs_dataframe`, `_00_remap_owner`, `_03_filter_collision` | Physics sim + owner remap + collision filter |
| `113-Polars_GNN_Corrected.py` | `_02_get_reach` | Reachability with discrete ships_list |
| `126-precompute.py` | `_get_reach_max_ships` | Max-ships reachability |
| `130-GNN_BC_shift.py` | `build_graph` (adapted → `build_graph_inference`) | Graph construction without labels |
