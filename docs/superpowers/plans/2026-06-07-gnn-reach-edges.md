# GNN Reach Edges Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `('planet_step', 'reaches', 'planet_step')` directed edges (with `log(ships_sent)/log(1024)` features) to the existing `data` HeteroData object in `105-104-AnalyseReach.ipynb`.

**Architecture:** A single new code cell appended after the Planet/PlanetStep construction cell. Uses two vectorised pandas merges to resolve `(planet_id, step)` pairs into integer node indices — no Python-level loops. Attaches `edge_index [2, 29080]` and `edge_attr [29080, 1]` to `data` in-place.

**Tech Stack:** `pandas`, `numpy`, `torch`, `torch_geometric` (all already imported/available in kernel).

---

## File Map

| File | Action |
|------|--------|
| `105-104-AnalyseReach.ipynb` | Insert one new code cell after the existing M6 graph construction cell |

---

### Task 1: Add the reach-edges cell

**Files:**
- Modify: `105-104-AnalyseReach.ipynb` — insert cell after the M6 graph construction cell (the last code cell)

- [ ] **Step 1: Insert a markdown separator cell**

After the last cell in the notebook, insert `cell_type=markdown`:

```markdown
### Reach edges — (`planet_step`, `reaches`, `planet_step`)
```

- [ ] **Step 2: Insert the reach-edges code cell**

After the markdown cell, insert `cell_type=code`:

```python
# ── Build (id, step) → node-index lookup ────────────────────────────────────
ps_idx = (
    ps_df[["id", "step"]]
    .reset_index()
    .rename(columns={"index": "ps_idx"})
)

# ── Resolve source indices (id_src, step_src) ────────────────────────────────
src = (
    pa[["id_src", "step_src"]]
    .rename(columns={"id_src": "id", "step_src": "step"})
    .merge(ps_idx, on=["id", "step"])["ps_idx"]
    .values
)

# ── Resolve destination indices (id, step) ───────────────────────────────────
dst = pa[["id", "step"]].merge(ps_idx, on=["id", "step"])["ps_idx"].values

# ── Edge feature: log(ships_sent) / log(1024) ─────────────────────────────────
edge_attr = torch.tensor(
    (np.log(pa["ships_sent"].values) / np.log(1024)).reshape(-1, 1),
    dtype=torch.float32,
)

# ── Attach to existing HeteroData ────────────────────────────────────────────
data["planet_step", "reaches", "planet_step"].edge_index = torch.tensor(
    np.stack([src, dst]), dtype=torch.long
)
data["planet_step", "reaches", "planet_step"].edge_attr = edge_attr

print(data)
print(f"\nreach edges  : {data['planet_step', 'reaches', 'planet_step'].edge_index.shape}")  # [2, 29080]
print(f"edge features: {data['planet_step', 'reaches', 'planet_step'].edge_attr.shape}")    # [29080, 1]
```

- [ ] **Step 3: Verify expected output**

Run the cell. Expected output:

```
HeteroData(
  planet={ x=[28, 4] },
  planet_step={ x=[588, 9] },
  (planet, has_snapshot, planet_step)={ edge_index=[2, 588] },
  (planet_step, reaches, planet_step)={ edge_index=[2, 29080], edge_attr=[29080, 1] }
)

reach edges  : torch.Size([2, 29080])
edge features: torch.Size([29080, 1])
```

If you see a KeyError on the merge, the `ps_df` index was reset after construction — re-run the M6 construction cell first, then re-run this cell.

- [ ] **Step 4: Commit**

```bash
git add 105-104-AnalyseReach.ipynb
git commit -m "feat: add reach edges (planet_step -> planet_step) to HeteroData"
```
