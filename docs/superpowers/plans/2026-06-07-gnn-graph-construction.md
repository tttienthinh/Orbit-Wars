# GNN Graph Construction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a single cell at the end of `105-104-AnalyseReach.ipynb` that builds a PyTorch Geometric `HeteroData` graph with `planet` and `planet_step` nodes connected by `has_snapshot` edges.

**Architecture:** Vectorised Pandas → feature tensors → `HeteroData`. Planet nodes (28, 4-dim) are built from the static deduplicated planet table; PlanetStep nodes (588, 9-dim) are built from every row of `df_s`. One directed edge per row connects the base planet to its snapshot.

**Tech Stack:** `torch`, `torch_geometric`, `pandas`, `numpy` (all already imported in the notebook).

---

## File Map

| File | Action |
|------|--------|
| `105-104-AnalyseReach.ipynb` | Insert one new code cell at the end (after `cell-14`) |

No new files; no new functions.

---

### Task 1: Add the graph construction cell

**Files:**
- Modify: `105-104-AnalyseReach.ipynb` — insert cell after `cell-14`

- [ ] **Step 1: Insert a markdown header cell**

Insert after `cell-14` with `edit_mode=insert`, `cell_type=markdown`:

```markdown
## M6 — Bipartite Graph (Planet + PlanetStep)
```

- [ ] **Step 2: Insert the graph construction code cell**

Insert after the markdown cell just created, `cell_type=code`:

```python
import torch
from torch_geometric.data import HeteroData

# ── Planet nodes (static, 4-dim) ─────────────────────────────────────────────
planets_df = (
    df_s[["id", "production", "nature"]]
    .drop_duplicates()
    .sort_values("id")
    .reset_index(drop=True)
)

nature_dummies = (
    pd.get_dummies(planets_df["nature"])
    .reindex(columns=["fix", "moving", "comet"], fill_value=0)
    .astype("float32")
)
planet_x = torch.tensor(
    pd.concat(
        [nature_dummies, planets_df[["production"]].div(5)],
        axis=1,
    ).values,
    dtype=torch.float32,
)

# ── PlanetStep nodes (dynamic, 9-dim) ────────────────────────────────────────
ps_df = df_s[["id", "step", "x", "y", "ships", "owner"]].copy().reset_index(drop=True)

owner_dummies = (
    pd.get_dummies(ps_df["owner"])
    .reindex(columns=[-1, 0, 1, 2, 3], fill_value=0)
    .astype("float32")
)
owner_dummies.columns = ["owner_neg1", "owner_0", "owner_1", "owner_2", "owner_3"]

ps_feats = pd.concat(
    [
        ps_df[["step"]].div(m.GameConfig.NB_STEPS_SIM),
        ps_df[["x"]].div(100),
        ps_df[["y"]].div(100),
        np.log(ps_df[["ships"]].clip(lower=1)) / np.log(1024),
        owner_dummies,
    ],
    axis=1,
)
planet_step_x = torch.tensor(ps_feats.values, dtype=torch.float32)

# ── Edges: planet → planet_step (matching id) ─────────────────────────────────
id_to_idx = {pid: i for i, pid in enumerate(planets_df["id"])}
src = torch.tensor([id_to_idx[pid] for pid in ps_df["id"]], dtype=torch.long)
dst = torch.arange(len(ps_df), dtype=torch.long)

# ── Assemble HeteroData ───────────────────────────────────────────────────────
data = HeteroData()
data["planet"].x = planet_x
data["planet_step"].x = planet_step_x
data["planet", "has_snapshot", "planet_step"].edge_index = torch.stack([src, dst])

print(data)
print(f"\nplanet nodes      : {data['planet'].x.shape}")          # [28, 4]
print(f"planet_step nodes : {data['planet_step'].x.shape}")       # [588, 9]
print(f"edges             : {data['planet', 'has_snapshot', 'planet_step'].edge_index.shape}")  # [2, 588]
```

- [ ] **Step 3: Verify expected output**

Run the cell. Expected output:
```
HeteroData(
  planet={ x=[28, 4] },
  planet_step={ x=[588, 9] },
  (planet, has_snapshot, planet_step)={ edge_index=[2, 588] }
)

planet nodes      : torch.Size([28, 4])
planet_step nodes : torch.Size([588, 9])
edges             : torch.Size([2, 588])
```

If `torch_geometric` is not installed: `pip install torch-geometric` in the kernel, then re-run.

- [ ] **Step 4: Commit**

```bash
git add 105-104-AnalyseReach.ipynb
git commit -m "feat: add M6 bipartite HeteroData graph construction cell"
```

---

## Notes

- `ships.clip(lower=1)` prevents `log(0)` when a planet has 0 ships.
- `reindex` on `nature_dummies` and `owner_dummies` guarantees all columns exist even when a category is absent in the current replay (e.g. no comet planets, no player 3).
- The edge direction is `planet → planet_step` (base-to-snapshot), matching M6 `P_i → P_{i,t}` grounding edges.
