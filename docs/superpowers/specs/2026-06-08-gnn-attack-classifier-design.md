# GNN Attack Classifier — Design Spec

**Date:** 2026-06-08
**Files:** `106-Simulate20Next_GNN.py` (model + inference), `107-Train_GNN.py` (training loop)

---

## Goal

Add `OrbitGNN` to `106-Simulate20Next_GNN.py`: a small heterogeneous GNN that takes the
`HeteroData` produced by `_05_get_GNN` and outputs a binary mask over attack nodes,
replacing `_04_score_and_decide` in `agent()`. Train it via behavioral cloning from
`_04_score_and_decide` using games played against agents in `orbit-wars-lab/agents/`,
tracked with MLflow.

---

## Model Architecture — `OrbitGNN`

```
OrbitGNN(hidden_dim=16)

Input projections (Linear, no bias):
  planet_proj:      Linear(4  → 16)
  planet_step_proj: Linear(9  → 16)
  attack_proj:      Linear(1  → 16)

Layer 1 — HeteroConv (aggregation="sum") + ReLU:
  (planet,      has_snapshot,     planet_step): SAGEConv(16→16)
  (planet_step, rev_has_snapshot, planet     ): SAGEConv(16→16)
  (planet_step, reaches,          planet_step): GATConv(16→16, edge_dim=1, heads=1)
  (planet_step, AttackSrc,        attack     ): SAGEConv(16→16)
  (attack,      rev_AttackSrc,    planet_step): SAGEConv(16→16)
  (attack,      AttackTgt,        planet_step): SAGEConv(16→16)
  (planet_step, rev_AttackTgt,    attack     ): SAGEConv(16→16)

Layer 2 — same HeteroConv structure + ReLU

Attack head:
  Linear(16 → 8) → ReLU → Linear(8 → 1)   ← raw logit per attack node
```

**Total params:** ~9 K. Forward pass on typical graph (~600 planet_step, ~200 attack nodes) ≈ 1–3 ms.

**Note on reaches/GATConv:** `_05_get_GNN` only adds reaches edges when `pa` is non-empty, so HeteroConv never calls GATConv for an absent edge type — no edge_attr None-guard needed.

### `forward(data: HeteroData) -> Tensor`

Returns a 1-D tensor of raw logits, shape `[n_attacks]`. Returns empty tensor if
`data["attack"].x` is absent or empty.

```python
def forward(self, data):
    x_dict = {
        "planet":      self.planet_proj(data["planet"].x),
        "planet_step": self.planet_step_proj(data["planet_step"].x),
        "attack":      self.attack_proj(data["attack"].x)
            if "attack" in data.node_types and data["attack"].x.numel() > 0
            else torch.zeros(0, self.hidden_dim),
    }
    edge_index_dict = data.edge_index_dict
    edge_attr_dict = {k: v for k, v in data.edge_attr_dict.items() if v is not None}

    for conv in [self.conv1, self.conv2]:
        x_dict = conv(x_dict, edge_index_dict, edge_attr_dict)
        x_dict = {k: F.relu(v) for k, v in x_dict.items()}

    atk = x_dict.get("attack", torch.zeros(0, self.hidden_dim))
    return self.head(atk).squeeze(-1)   # [n_attacks]
```

---

## Inference Integration (`106-Simulate20Next_GNN.py`)

### Module-level

```python
gnn_model = OrbitGNN(hidden_dim=16)
gnn_model.eval()
```

### Inside `agent(obs)` — replaces `_04_score_and_decide`

```python
data = StrategyPipeline._05_get_GNN(df_s, pa, safe_attacks)

attack_df = safe_attacks.query("ships_sent <= ships_min").reset_index(drop=True)
if not attack_df.empty and "attack" in data.node_types:
    with torch.no_grad():
        logits = gnn_model(data)            # [n_attacks]
    mask = logits.sigmoid() > 0.5
    moves = (
        attack_df[mask.numpy().astype(bool)]
        [["id_src", "final_angle", "ships_sent"]]
        .values.tolist()
    )
else:
    moves = []
```

**Multi-attack budget:** when the GNN selects multiple attacks from the same source
planet, `_interpreter` deducts ships sequentially and silently drops moves that exceed
the remaining budget — no extra guard needed.

---

## Label Construction

`_04_score_and_decide` is extended to optionally return the internal `attacks` dataframe
(the rows it selected) alongside the moves list. A new helper method:

```python
@staticmethod
def _04_get_selected(
    attacks_with_angle: pd.DataFrame,
    player_id: int,
) -> pd.DataFrame:
    """Same logic as _04_score_and_decide but returns the selected rows
    (id_src, id, step, ships_sent) instead of formatted moves."""
```

During training, call `_04_get_selected` on each step to get selected rows, then
label attack nodes:

```python
attack_df = safe_attacks.query("ships_sent <= ships_min").reset_index(drop=True)
selected = _04_get_selected(safe_attacks, player_id=0)  # rows: id_src, id, step, ships_sent
key_cols = ["id_src", "id", "step", "ships_sent"]
selected_keys = set(map(tuple, selected[key_cols].values.tolist()))
y = torch.tensor(
    [1.0 if tuple(r) in selected_keys else 0.0
     for r in attack_df[key_cols].values.tolist()],
    dtype=torch.float32,
)
```

---

## Training Loop (`107-Train_GNN.py`)

### Overview

```
for each iteration:
  1. sample opponent from orbit-wars-lab/agents/
  2. run 1v1 game via kaggle-environments → save replay
  3. iterate over replay steps → collect (HeteroData, y) pairs
  4. every BATCH_SIZE pairs → gradient update
  5. log metrics to MLflow
```

### Data Collection from Replay

```python
from kaggle_environments import make

def collect_samples(replay: dict, player_id: int = 0) -> list[tuple]:
    """Return list of (data, y) pairs from one replay."""
    samples = []
    for step_states in replay["steps"]:
        obs_dict = step_states[player_id]["observation"]
        obs = types.SimpleNamespace(**copy.deepcopy(obs_dict))
        env_step = obs.step

        df_s, planet_disp = StrategyPipeline._01_get_obs_dataframe(obs, env_step, 2)
        df_s = StrategyPipeline._00_remap_owner(df_s, obs, player_id)
        coarse = StrategyPipeline._02_pre_mine(df_s, 0)
        pa = StrategyPipeline._02_get_all_opportunities(coarse, df_s, planet_disp)
        safe_attacks = StrategyPipeline._03_filter_collision(pa)

        attack_df = safe_attacks.query("ships_sent <= ships_min").reset_index(drop=True)
        if attack_df.empty:
            continue

        data = StrategyPipeline._05_get_GNN(df_s, pa, safe_attacks)
        if "attack" not in data.node_types or data["attack"].x.numel() == 0:
            continue

        selected = StrategyPipeline._04_get_selected(safe_attacks, player_id=0)
        key_cols = ["id_src", "id", "step", "ships_sent"]
        selected_keys = set(map(tuple, selected[key_cols].values.tolist()))
        y = torch.tensor(
            [1.0 if tuple(r) in selected_keys else 0.0
             for r in attack_df[key_cols].values.tolist()],
            dtype=torch.float32,
        )
        samples.append((data, y))
    return samples
```

### Batch Learning

Accumulate `BATCH_SIZE = 32` (graph, y) pairs then update:

```python
from torch_geometric.data import Batch

def train_step(model, optimizer, batch_graphs, batch_ys, pos_weight):
    model.train()
    data_batch = Batch.from_data_list(batch_graphs)
    y = torch.cat(batch_ys)                         # [total_attacks_in_batch]
    logits = model(data_batch)
    loss = F.binary_cross_entropy_with_logits(
        logits, y,
        pos_weight=pos_weight,                       # handles class imbalance
    )
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item(), (logits.sigmoid() > 0.5).eq(y.bool()).float().mean().item()
```

`pos_weight` is recomputed per batch: `tensor([(y==0).sum() / max((y==1).sum(), 1)])`.

### Opponent Sampling

```python
from pathlib import Path

AGENTS_DIR = Path("orbit-wars-lab/agents")

def sample_opponent() -> Path:
    """Return a random agent path from orbit-wars-lab/agents/ (mine + external)."""
    candidates = [
        p for p in AGENTS_DIR.rglob("main.py")
        if "baselines" not in str(p)          # skip random/starter baselines
    ]
    return random.choice(candidates).parent
```

### Full Training Loop Skeleton

```python
import mlflow

BATCH_SIZE = 32
LR = 1e-3
SAVE_EVERY = 50   # games

model = OrbitGNN(hidden_dim=16)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

buffer_graphs, buffer_ys = [], []
game_idx = 0

with mlflow.start_run():
    mlflow.log_params({"hidden_dim": 16, "lr": LR, "batch_size": BATCH_SIZE})

    while True:
        opponent_path = sample_opponent()
        env = make("orbit_wars", debug=False)
        env.run(["106-Simulate20Next_GNN.py", str(opponent_path / "main.py")])
        replay = env.toJSON()

        samples = collect_samples(replay, player_id=0)
        for data, y in samples:
            buffer_graphs.append(data)
            buffer_ys.append(y)

            if len(buffer_graphs) >= BATCH_SIZE:
                loss, acc = train_step(model, optimizer, buffer_graphs, buffer_ys,
                                       pos_weight=compute_pos_weight(buffer_ys))
                buffer_graphs, buffer_ys = [], []
                mlflow.log_metrics({"loss": loss, "accuracy": acc}, step=game_idx)

        # Win-rate metric
        winner_is_gnn = (replay["rewards"][0] == 1)
        mlflow.log_metrics({"win": int(winner_is_gnn)}, step=game_idx)

        if game_idx % SAVE_EVERY == 0:
            torch.save(model.state_dict(), f"gnn_weights_{game_idx}.pt")

        game_idx += 1
```

---

## MLflow Metrics

| Metric | Description |
|--------|-------------|
| `loss` | BCEWithLogitsLoss per batch |
| `accuracy` | Fraction of attack nodes correctly classified |
| `win` | 1 if GNN agent won the game, 0 otherwise |

Logged every batch update; `win` logged every game.

---

## File Layout

```
106-Simulate20Next_GNN.py   ← OrbitGNN class + _05_get_GNN + agent() using GNN
                                + _04_get_selected helper
107-Train_GNN.py            ← training loop (MLflow, replay collection, batch updates)
gnn_weights.pt              ← saved checkpoints (not committed)
```

---

## Key Constraints

- `OrbitGNN` forward pass must stay under 10 ms (Kaggle 1-second budget)
- `hidden_dim=16` is fixed; do not increase without benchmarking
- `pos_weight` is recomputed per batch (most steps have zero attacks selected)
- Training agent in `env.run()` uses `106-Simulate20Next_GNN.py` with random-init weights initially
