# GNN Attack Classifier Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `OrbitGNN` (2-layer HeteroConv, hidden_dim=16) to `106-Simulate20Next_GNN.py` replacing `_04_score_and_decide`, and create `107-Train_GNN.py` for behavioral-cloning training with MLflow.

**Architecture:** `OrbitGNN` takes the `HeteroData` from `_05_get_GNN` and outputs a binary mask over attack nodes (sigmoid > 0.5 = launch). Training collects (graph, label) pairs by running 1v1 games against `orbit-wars-lab/agents/` agents, processing each replay step with `_04_get_selected` as teacher. Batches of 32 graphs are updated with BCEWithLogitsLoss, logged to MLflow, and weights saved every 5 games.

**Tech Stack:** Python 3, pandas, numpy, torch, torch_geometric (`HeteroData`, `HeteroConv`, `SAGEConv`, `GATConv`, `Batch`), kaggle-environments, mlflow

---

## File Layout

| File | Action | Responsibility |
|------|--------|----------------|
| `106-Simulate20Next_GNN.py` | Modify | Add `_04_get_selected`, `OrbitGNN`, update `agent()` |
| `107-Train_GNN.py` | Create | Data collection, training loop, MLflow logging |

---

### Task 1: Add `_04_get_selected` to `StrategyPipeline`

**Files:**
- Modify: `106-Simulate20Next_GNN.py` — insert new static method after `_04_score_and_decide`

This method replicates the selection logic of `_04_score_and_decide` but returns a DataFrame
with `[id_src, id, step, ships_sent]` columns instead of a moves list. Training uses this
to build binary labels for attack nodes.

- [ ] **Step 1: Insert `_04_get_selected` after `_04_score_and_decide`**

Find the end of `_04_score_and_decide` (the line `return moves`) and insert the new method
immediately after it, still inside the `StrategyPipeline` class:

```python
    @staticmethod
    def _04_get_selected(
        attacks_with_angle: pd.DataFrame,
        player_id: int,
    ) -> pd.DataFrame:
        """Same selection logic as _04_score_and_decide; returns selected attack rows
        with columns [id_src, id, step, ships_sent] for label construction."""
        _EMPTY = pd.DataFrame(columns=["id_src", "id", "step", "ships_sent"])
        if attacks_with_angle.empty:
            return _EMPTY

        selected_parts: list[pd.DataFrame] = []

        # Comet evasion (mirrors _04_score_and_decide exactly)
        awa_comets = attacks_with_angle[attacks_with_angle["nature_src"] == "comet"]
        if not awa_comets.empty:
            x_off = (awa_comets["x_src"] - GameConfig.CENTER).abs().max() or 0
            y_off = (awa_comets["y_src"] - GameConfig.CENTER).abs().max() or 0
            if max(x_off, y_off) > 45:
                comet_sel = (
                    awa_comets[awa_comets["ships_sent"] <= awa_comets["ships_min"]]
                    .sort_values(["ships_sent", "step"], ascending=[False, True])
                    .groupby("id_src", sort=False)
                    .first()
                    .reset_index()
                )
                if not comet_sel.empty:
                    selected_parts.append(comet_sel[["id_src", "id", "step", "ships_sent"]])
                id_to_avoid = awa_comets["id_src"].unique().tolist()
                attacks_with_angle = attacks_with_angle[
                    ~attacks_with_angle["id_src"].isin(id_to_avoid)
                ]

        if attacks_with_angle.empty:
            return pd.concat(selected_parts, ignore_index=True) if selected_parts else _EMPTY

        planet_id_top_5 = (
            attacks_with_angle
            .sort_values(["step", "ships_sent"])
            .groupby(["id_src", "id"], sort=False)
            .first()
            .reset_index()
            .sort_values(["step", "ships_sent"])
            .groupby("id_src", sort=False)
            .head(5)
            [["id_src", "id"]]
        )

        attacks_joined = (
            planet_id_top_5
            .merge(attacks_with_angle, on=["id_src", "id"], how="left")
            .loc[lambda d: d["owner"] != player_id]
            .assign(
                ships_needed=lambda d: np.where(
                    d["owner"] == -1, d["ships"], d["ships"] + d["production"]
                )
            )
            .loc[lambda d:
                (d["ships_needed"] + 1 <= d["ships_sent"]) &
                (d["ships_sent"] <= d["ships_needed"] + d["production_src"] + 1)
            ]
            .sort_values(["step", "ships_sent"])
            .groupby(["id_src", "id"], sort=False)
            .first()
            .reset_index()
            .assign(time_cost=lambda d: d["ships_needed"] / d["production_src"])
        )

        if attacks_joined.empty:
            return pd.concat(selected_parts, ignore_index=True) if selected_parts else _EMPTY

        attacks_joined = attacks_joined.assign(
            total_time_cost=attacks_joined.groupby("id_src")["time_cost"].transform("sum")
        ).assign(
            score=lambda d: (d["total_time_cost"] - d["time_cost"] - d["step_diff"])
                            * d["production"]
        )

        attacks = (
            attacks_joined
            .sort_values("score", ascending=False)
            .groupby("id_src", sort=False)
            .first()
            .reset_index()
            .loc[lambda d: d["ships_sent"] <= d["ships_min"]]
        )

        if not attacks.empty:
            selected_parts.append(attacks[["id_src", "id", "step", "ships_sent"]])

        return pd.concat(selected_parts, ignore_index=True) if selected_parts else _EMPTY
```

- [ ] **Step 2: Smoke-test `_04_get_selected` matches `_04_score_and_decide`**

Run this in a Python shell (or paste into a `.py` and run):

```python
import importlib.util, copy, json, types, numpy as np, pandas as pd
spec = importlib.util.spec_from_file_location("m106", "106-Simulate20Next_GNN.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

replay = json.load(open("62-logs/77588591.json", encoding="utf-8"))
obs = types.SimpleNamespace(**copy.deepcopy(replay["steps"][0][0]["observation"]))
num_agents = 2
env_step = obs.step
player_id = 1

df_s, planet_disp = m.StrategyPipeline._01_get_obs_dataframe(obs, env_step, num_agents)
df_s = m.StrategyPipeline._00_remap_owner(df_s, obs, player_id)
coarse = m.StrategyPipeline._02_pre_mine(df_s, 0)
pa = m.StrategyPipeline._02_get_all_opportunities(coarse, df_s, planet_disp)
safe_attacks = m.StrategyPipeline._03_filter_collision(pa)

moves = m.StrategyPipeline._04_score_and_decide(safe_attacks, pd.DataFrame(), player_id=0)
selected = m.StrategyPipeline._04_get_selected(safe_attacks, player_id=0)

# The selected DataFrame must have exactly one row per move
assert len(selected) == len(moves), f"len mismatch: {len(selected)} != {len(moves)}"
print(f"_04_get_selected OK: {len(selected)} rows matching {len(moves)} moves")
```

Expected output: `_04_get_selected OK: N rows matching N moves`

- [ ] **Step 3: Commit**

```bash
git add 106-Simulate20Next_GNN.py
git commit -m "feat: add _04_get_selected helper for GNN label construction"
```

---

### Task 2: Add `OrbitGNN` class to `106-Simulate20Next_GNN.py`

**Files:**
- Modify: `106-Simulate20Next_GNN.py` — add imports + class before the `# ── Entry point` comment

- [ ] **Step 1: Add missing imports at top of file**

The file already has `import torch` and `from torch_geometric.data import HeteroData`. Add these
two lines immediately after:

```python
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import HeteroConv, SAGEConv, GATConv
```

- [ ] **Step 2: Add `OrbitGNN` class before `# ── Entry point`**

Find the comment `# ── Entry point` and insert the full class immediately before it:

```python
# ── GNN Model ─────────────────────────────────────────────────────────────────
_REACHES_KEY = ("planet_step", "reaches", "planet_step")


class OrbitGNN(nn.Module):
    def __init__(self, hidden_dim: int = 16):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.planet_proj = nn.Linear(4, hidden_dim, bias=False)
        self.planet_step_proj = nn.Linear(9, hidden_dim, bias=False)
        self.attack_proj = nn.Linear(1, hidden_dim, bias=False)

        def _make_conv():
            return HeteroConv(
                {
                    ("planet", "has_snapshot", "planet_step"):
                        SAGEConv((hidden_dim, hidden_dim), hidden_dim),
                    ("planet_step", "rev_has_snapshot", "planet"):
                        SAGEConv((hidden_dim, hidden_dim), hidden_dim),
                    ("planet_step", "reaches", "planet_step"):
                        GATConv(
                            (hidden_dim, hidden_dim), hidden_dim,
                            edge_dim=1, heads=1, add_self_loops=False,
                        ),
                    ("planet_step", "AttackSrc", "attack"):
                        SAGEConv((hidden_dim, hidden_dim), hidden_dim),
                    ("attack", "rev_AttackSrc", "planet_step"):
                        SAGEConv((hidden_dim, hidden_dim), hidden_dim),
                    ("attack", "AttackTgt", "planet_step"):
                        SAGEConv((hidden_dim, hidden_dim), hidden_dim),
                    ("planet_step", "rev_AttackTgt", "attack"):
                        SAGEConv((hidden_dim, hidden_dim), hidden_dim),
                },
                aggr="sum",
            )

        self.conv1 = _make_conv()
        self.conv2 = _make_conv()
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
        )

    def forward(self, data) -> "torch.Tensor":
        has_attack = (
            "attack" in data.node_types
            and data["attack"].x.numel() > 0
        )

        x_dict = {
            "planet":      self.planet_proj(data["planet"].x),
            "planet_step": self.planet_step_proj(data["planet_step"].x),
        }
        if has_attack:
            x_dict["attack"] = self.attack_proj(data["attack"].x)

        edge_index_dict = data.edge_index_dict

        # Pass edge_attr only for the reaches edge (GATConv uses it; SAGEConv does not)
        reaches_attr = None
        if _REACHES_KEY in edge_index_dict:
            try:
                reaches_attr = data[_REACHES_KEY].edge_attr
            except AttributeError:
                pass
        edge_attr_map = {_REACHES_KEY: reaches_attr} if reaches_attr is not None else {}

        for conv in (self.conv1, self.conv2):
            x_dict = conv(x_dict, edge_index_dict, edge_attr=edge_attr_map)
            x_dict = {k: F.relu(v) for k, v in x_dict.items()}

        atk = x_dict.get("attack")
        if atk is None or atk.numel() == 0:
            return torch.zeros(0)
        return self.head(atk).squeeze(-1)   # [n_attacks]
```

- [ ] **Step 3: Smoke-test `OrbitGNN` instantiates and runs**

```python
import importlib.util, copy, json, types, numpy as np, pandas as pd
spec = importlib.util.spec_from_file_location("m106", "106-Simulate20Next_GNN.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

replay = json.load(open("62-logs/77588591.json", encoding="utf-8"))
obs = types.SimpleNamespace(**copy.deepcopy(replay["steps"][0][0]["observation"]))
df_s, planet_disp = m.StrategyPipeline._01_get_obs_dataframe(obs, obs.step, 2)
df_s = m.StrategyPipeline._00_remap_owner(df_s, obs, 1)
coarse = m.StrategyPipeline._02_pre_mine(df_s, 0)
pa = m.StrategyPipeline._02_get_all_opportunities(coarse, df_s, planet_disp)
safe_attacks = m.StrategyPipeline._03_filter_collision(pa)
data = m.StrategyPipeline._05_get_GNN(df_s, pa, safe_attacks)

import torch
model = m.OrbitGNN(hidden_dim=16)
model.eval()
with torch.no_grad():
    logits = model(data)
n_atk = len(safe_attacks.query("ships_sent <= ships_min"))
assert logits.shape == (n_atk,), f"shape mismatch: {logits.shape} vs ({n_atk},)"
print(f"OrbitGNN OK: logits shape {logits.shape}, n_attacks={n_atk}")
```

Expected output: `OrbitGNN OK: logits shape torch.Size([N]), n_attacks=N`

- [ ] **Step 4: Commit**

```bash
git add 106-Simulate20Next_GNN.py
git commit -m "feat: add OrbitGNN heterogeneous GNN class (2-layer, hidden_dim=16)"
```

---

### Task 3: Wire `OrbitGNN` into `agent()`

**Files:**
- Modify: `106-Simulate20Next_GNN.py` — module-level model, updated `agent()`

- [ ] **Step 1: Add module-level model after the `step / num_agents / player_id` globals**

Find these three lines near the bottom of the file:
```python
step = 0
num_agents = None
player_id = None
```

Add immediately after them (before `def agent`):

```python
gnn_model = OrbitGNN(hidden_dim=16)
_gnn_weights = Path(__file__).with_name("gnn_weights.pt") if "__file__" in dir() else None
if _gnn_weights is not None and _gnn_weights.exists():
    gnn_model.load_state_dict(torch.load(str(_gnn_weights), map_location="cpu"))
gnn_model.eval()
```

Also add `from pathlib import Path` to the top-of-file imports (after the existing imports).

- [ ] **Step 2: Replace `_04_score_and_decide` call inside `agent()` with GNN inference**

Find this block inside `agent()`:

```python
    safe_attacks = StrategyPipeline._03_filter_collision(pa)
    reach        = pd.DataFrame()  # placeholder until _04 uses reach_matrix
    moves        = StrategyPipeline._04_score_and_decide(safe_attacks, reach, player_id=0)
```

Replace it with:

```python
    safe_attacks = StrategyPipeline._03_filter_collision(pa)

    data = StrategyPipeline._05_get_GNN(df_s, pa, safe_attacks)
    attack_df = safe_attacks.query("ships_sent <= ships_min").reset_index(drop=True)
    if not attack_df.empty and "attack" in data.node_types:
        with torch.no_grad():
            logits = gnn_model(data)          # [n_attacks]
        mask = logits.sigmoid() > 0.5
        moves = (
            attack_df[mask.numpy().astype(bool)]
            [["id_src", "final_angle", "ships_sent"]]
            .values.tolist()
        )
    else:
        moves = []
```

- [ ] **Step 3: Smoke-test `agent()` completes without error**

```python
import importlib.util, copy, json, types
spec = importlib.util.spec_from_file_location("m106", "106-Simulate20Next_GNN.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
m.step = 0; m.num_agents = None; m.player_id = None

import json, copy, types
replay = json.load(open("62-logs/77588591.json", encoding="utf-8"))
obs = types.SimpleNamespace(**copy.deepcopy(replay["steps"][0][0]["observation"]))

moves = m.agent(obs)
assert isinstance(moves, list), f"Expected list, got {type(moves)}"
print(f"agent() OK: {len(moves)} moves returned (random-init weights)")
```

Expected output: `agent() OK: N moves returned (random-init weights)` (N may be 0 with random weights)

- [ ] **Step 4: Commit**

```bash
git add 106-Simulate20Next_GNN.py
git commit -m "feat: wire OrbitGNN into agent(), replace _04_score_and_decide"
```

---

### Task 4: Create `107-Train_GNN.py` — data helpers

**Files:**
- Create: `107-Train_GNN.py`

- [ ] **Step 1: Create `107-Train_GNN.py` with imports and `collect_samples`**

```python
"""Training loop for OrbitGNN — behavioral cloning from _04_get_selected.

Usage:
    python 107-Train_GNN.py

Requires:
    pip install mlflow kaggle-environments torch torch-geometric
"""
import copy
import json
import random
import types
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import Batch

import importlib.util as _ilu

# ── Load pipeline from 106 ────────────────────────────────────────────────────
_spec = _ilu.spec_from_file_location("m106", "106-Simulate20Next_GNN.py")
m = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(m)
SP = m.StrategyPipeline
OrbitGNN = m.OrbitGNN

AGENTS_DIR = Path("orbit-wars-lab/agents")
LOGS_DIR = Path("62-logs")
BATCH_SIZE = 32
LR = 1e-3
SAVE_EVERY = 5
WEIGHTS_PATH = Path("gnn_weights.pt")


# ── Data helpers ──────────────────────────────────────────────────────────────

def _process_step(obs_dict: dict) -> "tuple | None":
    """Return (HeteroData, y_tensor) for one observation, or None if no attack nodes."""
    obs = types.SimpleNamespace(**copy.deepcopy(obs_dict))
    player_id = getattr(obs, "player", 0)
    env_step = getattr(obs, "step", 0)

    initial = getattr(obs, "initial_planets", [])
    owners = {p[1] for p in initial if p[1] != -1}
    num_agents = 4 if len(owners) > 2 else 2

    df_s, planet_disp = SP._01_get_obs_dataframe(obs, env_step, num_agents)
    df_s = SP._00_remap_owner(df_s, obs, player_id)
    coarse = SP._02_pre_mine(df_s, 0)
    pa = SP._02_get_all_opportunities(coarse, df_s, planet_disp)
    safe_attacks = SP._03_filter_collision(pa)

    attack_df = safe_attacks.query("ships_sent <= ships_min").reset_index(drop=True)
    if attack_df.empty:
        return None

    data = SP._05_get_GNN(df_s, pa, safe_attacks)
    if "attack" not in data.node_types or data["attack"].x.numel() == 0:
        return None

    selected = SP._04_get_selected(safe_attacks, player_id=0)
    key_cols = ["id_src", "id", "step", "ships_sent"]
    selected_keys = set(map(tuple, selected[key_cols].values.tolist()))
    y = torch.tensor(
        [
            1.0 if tuple(r) in selected_keys else 0.0
            for r in attack_df[key_cols].values.tolist()
        ],
        dtype=torch.float32,
    )
    return data, y


def collect_samples(replay: dict) -> list:
    """Extract (HeteroData, y) pairs from every step of a replay dict."""
    samples = []
    for step_states in replay.get("steps", []):
        if not step_states:
            continue
        state0 = step_states[0]
        if not isinstance(state0, dict):
            continue
        if state0.get("status", "ACTIVE") not in ("ACTIVE", None, ""):
            continue
        obs_dict = state0.get("observation")
        if not obs_dict:
            continue
        try:
            result = _process_step(obs_dict)
        except Exception:
            continue
        if result is not None:
            samples.append(result)
    return samples


def sample_opponent() -> Path:
    """Pick a random agent from orbit-wars-lab/agents/ (mine + external, no baselines)."""
    candidates = [
        p.parent
        for p in AGENTS_DIR.rglob("main.py")
        if "baselines" not in str(p)
    ]
    if not candidates:
        raise RuntimeError(f"No agents found under {AGENTS_DIR}")
    return random.choice(candidates)


def run_game(agent_path: Path, opponent_path: Path) -> dict:
    """Run one 1v1 game (agent_path=player0, opponent_path=player1). Return replay dict."""
    from kaggle_environments import make
    env = make("orbit_wars", debug=False)
    env.run([str(agent_path / "main.py"), str(opponent_path / "main.py")])
    return env.toJSON()
```

- [ ] **Step 2: Verify imports and helpers load cleanly**

```powershell
python -c "import importlib.util; spec = importlib.util.spec_from_file_location('t', '107-Train_GNN.py'); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); print('imports OK')"
```

Expected output: `imports OK`

- [ ] **Step 3: Smoke-test `collect_samples` on existing replay**

```python
import importlib.util
spec = importlib.util.spec_from_file_location("t107", "107-Train_GNN.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

import json
replay = json.load(open("62-logs/77588591.json", encoding="utf-8"))
samples = m.collect_samples(replay)
print(f"collect_samples OK: {len(samples)} (graph, y) pairs from replay")
assert len(samples) > 0, "expected at least one sample"
data, y = samples[0]
print(f"  first sample: {data['attack'].x.shape[0]} attacks, y sum={y.sum().item():.0f}")
```

Expected output: `collect_samples OK: N (graph, y) pairs from replay`

- [ ] **Step 4: Commit**

```bash
git add 107-Train_GNN.py
git commit -m "feat: add 107-Train_GNN.py with collect_samples and data helpers"
```

---

### Task 5: Add training loop with MLflow to `107-Train_GNN.py`

**Files:**
- Modify: `107-Train_GNN.py` — append training utilities and main loop

- [ ] **Step 1: Append training utilities to `107-Train_GNN.py`**

Add these functions at the bottom of the file (after `run_game`):

```python
# ── Training utilities ────────────────────────────────────────────────────────

def compute_pos_weight(batch_ys: list) -> torch.Tensor:
    """BCEWithLogitsLoss pos_weight: ratio of negatives to positives per batch."""
    y_cat = torch.cat(batch_ys)
    n_pos = y_cat.sum().item()
    n_neg = len(y_cat) - n_pos
    if n_pos == 0:
        return torch.tensor(1.0)
    return torch.tensor(n_neg / n_pos)


def train_step(
    model: "OrbitGNN",
    optimizer: "torch.optim.Optimizer",
    batch_graphs: list,
    batch_ys: list,
) -> "tuple[float, float]":
    """One gradient update. Returns (loss, accuracy)."""
    model.train()
    data_batch = Batch.from_data_list(batch_graphs)
    y = torch.cat(batch_ys)
    pos_weight = compute_pos_weight(batch_ys)

    logits = model(data_batch)
    loss = F.binary_cross_entropy_with_logits(logits, y, pos_weight=pos_weight)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    with torch.no_grad():
        preds = (logits.sigmoid() > 0.5)
        acc = preds.eq(y.bool()).float().mean().item()

    return loss.item(), acc


# ── Main training loop ────────────────────────────────────────────────────────

def main() -> None:
    import mlflow

    model = OrbitGNN(hidden_dim=16)
    if WEIGHTS_PATH.exists():
        model.load_state_dict(torch.load(str(WEIGHTS_PATH), map_location="cpu"))
        print(f"Loaded weights from {WEIGHTS_PATH}")
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    agent_106_path = Path(".")   # 106-Simulate20Next_GNN.py is in cwd; kaggle-envs needs dir

    # Bootstrap from existing logs
    bootstrap_samples: list = []
    for log_file in sorted(LOGS_DIR.glob("*.json")):
        try:
            replay = json.loads(log_file.read_text(encoding="utf-8"))
            bootstrap_samples.extend(collect_samples(replay))
        except Exception as e:
            print(f"  skip {log_file.name}: {e}")
    print(f"Bootstrap: {len(bootstrap_samples)} samples from {LOGS_DIR}")

    buffer_graphs: list = [s[0] for s in bootstrap_samples]
    buffer_ys: list = [s[1] for s in bootstrap_samples]
    game_idx = 0
    update_idx = 0

    with mlflow.start_run():
        mlflow.log_params({
            "hidden_dim": 16,
            "lr": LR,
            "batch_size": BATCH_SIZE,
            "save_every": SAVE_EVERY,
        })

        while True:
            # ── Run one game ──────────────────────────────────────────────────
            opponent_path = sample_opponent()
            try:
                replay = run_game(agent_106_path, opponent_path)
            except Exception as e:
                print(f"Game {game_idx} failed ({opponent_path.name}): {e}")
                game_idx += 1
                continue

            # Win rate (player 0 = our GNN agent)
            rewards = replay.get("rewards", [])
            winner_is_gnn = (len(rewards) > 0 and rewards[0] == 1)
            mlflow.log_metrics(
                {"win": int(winner_is_gnn), "opponent": hash(opponent_path.name) % 1000},
                step=game_idx,
            )
            print(
                f"Game {game_idx} vs {opponent_path.name}: "
                f"{'WIN' if winner_is_gnn else 'LOSS'}"
            )

            # ── Collect samples from the just-played game ─────────────────────
            new_samples = collect_samples(replay)
            buffer_graphs.extend(s[0] for s in new_samples)
            buffer_ys.extend(s[1] for s in new_samples)

            # ── Batch updates while buffer has enough samples ─────────────────
            while len(buffer_graphs) >= BATCH_SIZE:
                batch_g = buffer_graphs[:BATCH_SIZE]
                batch_y = buffer_ys[:BATCH_SIZE]
                buffer_graphs = buffer_graphs[BATCH_SIZE:]
                buffer_ys = buffer_ys[BATCH_SIZE:]

                loss, acc = train_step(model, optimizer, batch_g, batch_y)
                mlflow.log_metrics(
                    {"loss": loss, "accuracy": acc},
                    step=update_idx,
                )
                print(f"  update {update_idx}: loss={loss:.4f} acc={acc:.3f}")
                update_idx += 1

            # ── Save weights every SAVE_EVERY games ───────────────────────────
            game_idx += 1
            if game_idx % SAVE_EVERY == 0:
                torch.save(model.state_dict(), str(WEIGHTS_PATH))
                print(f"Saved weights → {WEIGHTS_PATH}  (game {game_idx})")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify the file is syntactically valid**

```powershell
python -m py_compile 107-Train_GNN.py && python -c "print('syntax OK')"
```

Expected output: `syntax OK`

- [ ] **Step 3: Smoke-test `train_step` on a small batch**

```python
import importlib.util, json
spec = importlib.util.spec_from_file_location("t107", "107-Train_GNN.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

import torch

# Build small batch from existing log
replay = json.loads(open("62-logs/77588591.json", encoding="utf-8").read())
samples = m.collect_samples(replay)[:4]
assert len(samples) >= 1, "need at least 1 sample"

model = m.OrbitGNN(hidden_dim=16)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
batch_g = [s[0] for s in samples]
batch_y = [s[1] for s in samples]

loss1, acc1 = m.train_step(model, optimizer, batch_g, batch_y)
loss2, acc2 = m.train_step(model, optimizer, batch_g, batch_y)
print(f"train_step OK: loss1={loss1:.4f} loss2={loss2:.4f}")
# Loss doesn't have to drop on 2 steps from random init, but must be finite
assert loss1 == loss1 and loss2 == loss2, "loss is NaN"
```

Expected output: `train_step OK: loss1=X.XXXX loss2=X.XXXX` (both finite)

- [ ] **Step 4: Commit**

```bash
git add 107-Train_GNN.py
git commit -m "feat: add 107-Train_GNN.py training loop with MLflow and batch updates"
```

---

## Self-Review

**Spec coverage check:**

| Spec requirement | Task |
|-----------------|------|
| `OrbitGNN` 2-layer HeteroConv hidden_dim=16 | Task 2 |
| GATConv(edge_dim=1) for reaches, SAGEConv for others | Task 2 |
| Attack head Linear(16→8)→ReLU→Linear(8→1) | Task 2 |
| Replace `_04_score_and_decide` in `agent()` | Task 3 |
| Module-level `gnn_model` with optional weight loading | Task 3 |
| `_04_get_selected` label construction | Task 1 |
| `collect_samples` from replay | Task 4 |
| `sample_opponent` from orbit-wars-lab/agents | Task 4 |
| `run_game` via kaggle-environments | Task 4 |
| `train_step` with BCEWithLogitsLoss + pos_weight | Task 5 |
| Batch learning via `Batch.from_data_list` | Task 5 |
| MLflow: loss, accuracy, win metrics | Task 5 |
| SAVE_EVERY=5 games | Task 5 |
| Bootstrap from 62-logs | Task 5 |
| Self-play loop (live games) | Task 5 |

All spec requirements covered.
