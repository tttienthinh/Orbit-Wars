# Episode Log Pipeline & ML Model Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build steps 3–4 of the episode log pipeline (combinatorial augmentation + env simulation), a shared feature builder, and two imitation learning models (XGBoost + MLP) with MLFlow tracking.

**Architecture:** Steps 3 and 4 extend `11-download_logs.ipynb` and write to `03-combinations/` and `04-simulate/`. A shared `pipeline/` package builds the feature matrix consumed by standalone training scripts. Each training script is self-contained and logs to a local MLFlow server.

**Tech Stack:** Python 3.10+, kaggle_environments (`orbit_wars`), numpy, pandas, scikit-learn, xgboost, PyTorch, mlflow

---

## File Structure

| File | Responsibility |
|------|----------------|
| `11-download_logs.ipynb` | Step 3 (combinatorial augmentation) + Step 4 (env simulation) cells |
| `pipeline/__init__.py` | Package marker |
| `pipeline/env_sim.py` | `simulate_futures(obs, done_set, nb_future_step)` — env injection + stepping |
| `pipeline/features.py` | `build_slot_map`, `extract_features`, `build_feature_matrix` — V1 and V2 |
| `pipeline/model.py` | `OrbitMLP` — shared PyTorch model class |
| `17-Train_XGBoost.py` | Train 3 XGBoost models, log to MLFlow |
| `17-Play_XGBoost.py` | XGBoost inference: obs → action list |
| `18-Train_ML.py` | Train MLP multi-task model, log to MLFlow |
| `18-Play_ML.py` | MLP inference: obs → action list |

---

## Task 1: Explore env state injection (notebook cell)

**Goal:** Determine the exact API to inject an obs dict into a fresh `orbit_wars` env and step it with custom actions. This informs `pipeline/env_sim.py`.

**Files:**
- Modify: `11-download_logs.ipynb` (add exploratory markdown + code cell, keep it)

- [ ] **Step 1: Add exploration cell**

```python
# ── Task 1 exploration: env state injection ──────────────────────────────────
import kaggle_environments as ke
import json

ep_id = "76319029"
with open(f"11-download_logs/02-augment/episode-{ep_id}.json") as f:
    augmented = json.load(f)
obs_41 = next(o for o in augmented if o["step"] == 41)

env = ke.make("orbit_wars", debug=True)
env.reset()

# Inspect state shape
print(type(env.state[0]))
print(type(env.state[0].observation))
print(list(env.state[0].observation.keys())[:5])
```

- [ ] **Step 2: Run cell, note output.** Record whether `env.state[0].observation` is a plain dict or a `Configuration`/`Namespace`-like object.

- [ ] **Step 3: Try injection**

```python
# Attempt 1: dict-style update
try:
    env.state[0].observation.update(obs_41)
    env.state[1].observation.update(obs_41)
    print("dict-style update works")
except AttributeError:
    print("dict-style update failed — try dot assignment")

# Attempt 2: direct assignment of each key
for key, val in obs_41.items():
    env.state[0].observation[key] = val
    env.state[1].observation[key] = val

env.step([[], []])  # step with empty actions
print("step OK, new step:", env.state[0].observation["step"])
print("planets[0]:", env.state[0].observation["planets"][0])
```

- [ ] **Step 4: Run cell.** Verify `step` increments and `planets` list is non-empty.

- [ ] **Step 5: Note the working injection pattern** in the cell markdown. This pattern is used verbatim in `pipeline/env_sim.py` Task 2.

- [ ] **Step 6: Commit**

```bash
git add 11-download_logs.ipynb
git commit -m "chore: add env state injection exploration"
```

---

## Task 2: `pipeline/env_sim.py` — env simulation utility

**Files:**
- Create: `pipeline/__init__.py`
- Create: `pipeline/env_sim.py`

- [ ] **Step 1: Create `pipeline/__init__.py`** (empty file)

- [ ] **Step 2: Create `pipeline/env_sim.py`**

```python
import copy
import kaggle_environments as ke

ENV_NAME = "orbit_wars"
NB_FUTURE_STEP = 10


def _to_game_action(action):
    """
    The step 2 augmentation adds dest_planet_id → 4-element actions.
    The env only accepts 3-element actions: [from_planet_id, angle, ships].
    """
    return action[:3]


def simulate_futures(obs, done_set, nb_future_step=NB_FUTURE_STEP):
    """
    Given an obs dict and a list of already-decided actions (done_set),
    injects obs into a fresh env, applies done_set (opponent = empty),
    then steps nb_future_step times with empty actions.

    done_set entries may be 3-element [from, angle, ships] or 4-element
    [from, angle, ships, dest] — dest is stripped automatically.

    Returns list of nb_future_step planet snapshots, each a list of
    [id, owner, x, y, radius, ships, production].

    Adjust injection if Task 1 reveals a different API.
    """
    env = ke.make(ENV_NAME, debug=False)
    env.reset()

    for key, val in obs.items():
        env.state[0].observation[key] = val
        env.state[1].observation[key] = val

    game_done_set = [_to_game_action(a) for a in done_set]
    env.step([game_done_set, []])

    snapshots = []
    for _ in range(nb_future_step):
        env.step([[], []])
        snapshots.append(
            copy.deepcopy(list(env.state[0].observation.get("planets", [])))
        )
    return snapshots
```

- [ ] **Step 3: Smoke-test in a Python shell**

```bash
python -c "
from pipeline.env_sim import simulate_futures
import json
with open('11-download_logs/02-augment/episode-76319029.json') as f:
    data = json.load(f)
obs = next(o for o in data if o['step'] == 41)
snaps = simulate_futures(obs, done_set=[], nb_future_step=3)
assert len(snaps) == 3
assert len(snaps[0]) > 0
print('OK — snapshot 0 planet 0:', snaps[0][0])
"
```

Expected: `OK — snapshot 0 planet 0: [0, ...]`

- [ ] **Step 4: Commit**

```bash
git add pipeline/__init__.py pipeline/env_sim.py
git commit -m "feat: pipeline/env_sim — env state injection and future stepping"
```

---

## Task 3: Step 3 in notebook — combinatorial augmentation

**Output:** `11-download_logs/03-combinations/*.json`
Each file is a list of `{obs, done_set, action_to_do}` dicts.

**Files:**
- Modify: `11-download_logs.ipynb`

- [ ] **Step 1: Add test cell**

```python
# ── Step 3: combinatorial augmentation ───────────────────────────────────────
from itertools import combinations as _combinations
import copy

def generate_combinations(obs):
    """
    Expands one obs into all (done_set, action_to_do) pairs plus a stop row.
    obs["action"] = [[from_planet_id, angle, ships, dest_planet_id], ...]
    Returns list of dicts: {obs, done_set, action_to_do}.
    """
    actions = obs["action"]
    nb_action = len(actions)
    rows = []
    for nb_done in range(nb_action):
        for done_indices in _combinations(range(nb_action), nb_done):
            done = [actions[i] for i in done_indices]
            remaining = [actions[i] for i in range(nb_action) if i not in done_indices]
            for action_to_do in remaining:
                rows.append({
                    "obs": obs,
                    "done_set": done,
                    "action_to_do": action_to_do,
                })
    rows.append({"obs": obs, "done_set": actions, "action_to_do": None})
    return rows

# Test: 2 actions → 5 rows: (∅,A),(∅,B),([A],B),([B],A),(stop)
mock_obs = {
    "step": 41,
    "action": [[0, 1.0, 70, 8], [1, 2.0, 30, 5]],
    "planets": [],
    "fleets": [],
}
combos = generate_combinations(mock_obs)
assert len(combos) == 5, f"Expected 5, got {len(combos)}"
assert combos[-1]["action_to_do"] is None
assert combos[0]["done_set"] == []
assert combos[2]["done_set"] == [[0, 1.0, 70, 8]]
assert combos[2]["action_to_do"] == [1, 2.0, 30, 5]
print("generate_combinations OK")
```

- [ ] **Step 2: Run test cell.** Verify: `generate_combinations OK`

- [ ] **Step 3: Add processing cell**

```python
augment_dir = Path("11-download_logs/02-augment")
combo_dir   = Path("11-download_logs/03-combinations")
combo_dir.mkdir(parents=True, exist_ok=True)

for json_file in sorted(augment_dir.glob("*.json")):
    with open(json_file) as f:
        data = json.load(f)
    all_rows = []
    for obs in data:
        if not obs.get("action"):
            continue
        all_rows.extend(copy.deepcopy(generate_combinations(obs)))
    out_path = combo_dir / json_file.name
    with open(out_path, "w") as f:
        json.dump(all_rows, f)
    print(f"Saved {len(all_rows):,} rows → {out_path}")
```

- [ ] **Step 4: Run processing cell.** Verify files appear in `03-combinations/`.

- [ ] **Step 5: Commit**

```bash
git add 11-download_logs.ipynb
git commit -m "feat: step 3 — combinatorial augmentation"
```

---

## Task 4: Step 4 in notebook — env simulation

**Output:** `11-download_logs/04-simulate/*.json`
Each file is a list of rows with an added `future_planets` key (list of 10 planet snapshots).

**Files:**
- Modify: `11-download_logs.ipynb`

- [ ] **Step 1: Add simulation cell**

```python
# ── Step 4: env simulation ────────────────────────────────────────────────────
import sys
sys.path.insert(0, ".")
from pipeline.env_sim import simulate_futures

NB_FUTURE_STEP = 10
```

- [ ] **Step 2: Add sanity check cell**

```python
with open("11-download_logs/03-combinations/episode-76319029.json") as f:
    combos = json.load(f)

row0 = combos[0]
snaps = simulate_futures(row0["obs"], row0["done_set"], NB_FUTURE_STEP)
assert len(snaps) == NB_FUTURE_STEP
assert len(snaps[0]) > 0
print(f"Sanity OK — {NB_FUTURE_STEP} snapshots, {len(snaps[0])} planets each")
print(f"  done_set: {row0['done_set']}")
print(f"  action_to_do: {row0['action_to_do']}")
```

- [ ] **Step 3: Run sanity check.** Verify 10 snapshots with non-empty planet lists.

- [ ] **Step 4: Add processing cell**

```python
combo_dir   = Path("11-download_logs/03-combinations")
simulate_dir = Path("11-download_logs/04-simulate")
simulate_dir.mkdir(parents=True, exist_ok=True)

for json_file in sorted(combo_dir.glob("*.json")):
    with open(json_file) as f:
        rows = json.load(f)
    for row in rows:
        row["future_planets"] = simulate_futures(
            row["obs"], row["done_set"], NB_FUTURE_STEP
        )
    out_path = simulate_dir / json_file.name
    with open(out_path, "w") as f:
        json.dump(rows, f)
    print(f"Saved {len(rows):,} rows → {out_path}")
```

- [ ] **Step 5: Run processing cell.** Verify `04-simulate/` files exist with `future_planets` key present in each row.

- [ ] **Step 6: Commit**

```bash
git add 11-download_logs.ipynb
git commit -m "feat: step 4 — env simulation (NB_FUTURE_STEP=10)"
```

---

## Task 5: `pipeline/features.py` — feature matrix builder

**Files:**
- Create: `pipeline/features.py`

- [ ] **Step 1: Create `pipeline/features.py`**

```python
import json
import math
import numpy as np
from pathlib import Path

NB_PLANETS    = 44   # 40 non-comets + 4 comets
NB_FUTURE_STEP = 10
CENTER_X = CENTER_Y = 50.0


def build_slot_map(obs_planets):
    """
    Classifies planets as comets (orbiting) or non-comets.
    Comets: dist(center, sun) + radius < 50.
    Returns (non_comet_ids sorted asc, comet_ids sorted asc).
    Slot mapping: non_comet[i] → slot i,  comet[j] → slot 40+j.
    """
    comet_ids     = []
    non_comet_ids = []
    for p in obs_planets:
        pid, x, y, radius = p[0], p[2], p[3], p[4]
        dist = math.sqrt((x - CENTER_X) ** 2 + (y - CENTER_Y) ** 2)
        if dist + radius < 50.0:
            comet_ids.append(pid)
        else:
            non_comet_ids.append(pid)
    return sorted(non_comet_ids), sorted(comet_ids)


def planet_id_to_slot(planet_id, non_comet_ids, comet_ids):
    """Returns slot 0-43, or -1 if unknown."""
    if planet_id in comet_ids:
        return 40 + comet_ids.index(planet_id)
    if planet_id in non_comet_ids:
        return non_comet_ids.index(planet_id)
    return -1


def slot_to_planet_id(slot, non_comet_ids, comet_ids):
    """Inverse of planet_id_to_slot."""
    if slot >= 40:
        return comet_ids[slot - 40]
    return non_comet_ids[slot]


def extract_features(future_planets, non_comet_ids, comet_ids, version=1):
    """
    version=1 → 2200 features: 44 * NB_FUTURE_STEP * {owner,ships,x,y,production}
    version=2 → 1540 features: 44 * NB_FUTURE_STEP * {owner,ships,production}
                              + 11 * NB_FUTURE_STEP * {x,y}  (slots 0-10)
    """
    if version == 1:
        feat = np.zeros((NB_FUTURE_STEP, NB_PLANETS, 5), dtype=np.float32)
        for step_i, planets in enumerate(future_planets):
            for p in planets:
                pid, owner, x, y, radius, ships, prod = p[0], p[1], p[2], p[3], p[4], p[5], p[6]
                slot = planet_id_to_slot(pid, non_comet_ids, comet_ids)
                if slot < 0:
                    continue
                feat[step_i, slot] = [owner, ships, x, y, prod]
        return feat.flatten()
    else:
        state = np.zeros((NB_FUTURE_STEP, NB_PLANETS, 3), dtype=np.float32)
        pos   = np.zeros((NB_FUTURE_STEP, 11, 2), dtype=np.float32)
        for step_i, planets in enumerate(future_planets):
            for p in planets:
                pid, owner, x, y, radius, ships, prod = p[0], p[1], p[2], p[3], p[4], p[5], p[6]
                slot = planet_id_to_slot(pid, non_comet_ids, comet_ids)
                if slot < 0:
                    continue
                state[step_i, slot] = [owner, ships, prod]
                if 0 <= slot <= 10:
                    pos[step_i, slot] = [x, y]
        return np.concatenate([state.flatten(), pos.flatten()])


def build_feature_matrix(simulate_dir, version=1):
    """
    Reads all JSON files in simulate_dir (step 4 output).
    Returns (X, y_from, y_to, y_ships):
      X       — float32 array, shape (N, n_features)
      y_from  — float32 array, shape (N,), values 0-43=slot or 44=stop
      y_to    — float32 array, shape (N,), values 0-43=slot or nan when stop
      y_ships — float32 array, shape (N,), nan when stop
    """
    simulate_dir = Path(simulate_dir)
    X_list, y_from_list, y_to_list, y_ships_list = [], [], [], []

    for json_file in sorted(simulate_dir.glob("*.json")):
        with open(json_file) as f:
            rows = json.load(f)
        if not rows:
            continue
        obs_planets = rows[0]["obs"]["planets"]
        non_comet_ids, comet_ids = build_slot_map(obs_planets)

        for row in rows:
            feat = extract_features(
                row["future_planets"], non_comet_ids, comet_ids, version=version
            )
            X_list.append(feat)

            action = row["action_to_do"]
            if action is None:
                y_from_list.append(44)
                y_to_list.append(np.nan)
                y_ships_list.append(np.nan)
            else:
                from_pid, _angle, ships, to_pid = action
                from_slot = planet_id_to_slot(from_pid, non_comet_ids, comet_ids)
                to_slot   = planet_id_to_slot(to_pid,   non_comet_ids, comet_ids)
                y_from_list.append(float(from_slot))
                y_to_list.append(float(to_slot))
                y_ships_list.append(float(ships))

    X       = np.array(X_list,    dtype=np.float32)
    y_from  = np.array(y_from_list,  dtype=np.float32)
    y_to    = np.array(y_to_list,    dtype=np.float32)
    y_ships = np.array(y_ships_list, dtype=np.float32)
    return X, y_from, y_to, y_ships
```

- [ ] **Step 2: Validate shapes**

```bash
python -c "
from pathlib import Path
from pipeline.features import build_feature_matrix
X1, yf, yt, ys = build_feature_matrix(Path('11-download_logs/04-simulate'), version=1)
assert X1.shape[1] == 2200, f'V1: expected 2200, got {X1.shape[1]}'
X2, *_ = build_feature_matrix(Path('11-download_logs/04-simulate'), version=2)
assert X2.shape[1] == 1540, f'V2: expected 1540, got {X2.shape[1]}'
assert (yf == 44).any(), 'No stop rows found'
print(f'V1 shape: {X1.shape}  V2 shape: {X2.shape}  stop rows: {(yf==44).sum()}')
"
```

Expected output contains both shapes and a non-zero stop row count.

- [ ] **Step 3: Commit**

```bash
git add pipeline/features.py
git commit -m "feat: pipeline/features — V1/V2 feature matrix builder"
```

---

## Task 6: `pipeline/model.py` — shared OrbitMLP

**Files:**
- Create: `pipeline/model.py`

- [ ] **Step 1: Create `pipeline/model.py`**

```python
import torch.nn as nn


class OrbitMLP(nn.Module):
    """
    Multi-task MLP for imitation learning.
    Heads: from (45 classes, 44=stop), to (44 classes), ships (regression).
    """
    def __init__(self, input_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.BatchNorm1d(input_dim),
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
        )
        self.head_from  = nn.Linear(256, 45)
        self.head_to    = nn.Linear(256, 44)
        self.head_ships = nn.Linear(256, 1)

    def forward(self, x):
        h = self.encoder(x)
        return self.head_from(h), self.head_to(h), self.head_ships(h).squeeze(-1)
```

- [ ] **Step 2: Smoke-test**

```bash
python -c "
import torch
from pipeline.model import OrbitMLP
m = OrbitMLP(input_dim=2200)
x = torch.randn(4, 2200)
lf, lt, ls = m(x)
assert lf.shape == (4, 45)
assert lt.shape == (4, 44)
assert ls.shape == (4,)
print('OrbitMLP OK')
"
```

Expected: `OrbitMLP OK`

- [ ] **Step 3: Commit**

```bash
git add pipeline/model.py
git commit -m "feat: pipeline/model — shared OrbitMLP"
```

---

## Task 7: `17-Train_XGBoost.py` — XGBoost training + MLFlow

**Files:**
- Create: `17-Train_XGBoost.py`

- [ ] **Step 1: Create `17-Train_XGBoost.py`**

```python
import numpy as np
import mlflow
import mlflow.xgboost
import xgboost as xgb
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from pipeline.features import build_feature_matrix

SIMULATE_DIR    = Path("11-download_logs/04-simulate")
MODEL_DIR       = Path("models/xgboost")
FEATURE_VERSION = 1   # change to 2 for cadran-based features
RANDOM_STATE    = 42
XGB_PARAMS = dict(n_estimators=300, max_depth=6, learning_rate=0.1, random_state=RANDOM_STATE)


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    X, y_from, y_to, y_ships = build_feature_matrix(SIMULATE_DIR, version=FEATURE_VERSION)
    mask_ns = y_from != 44  # non-stop rows

    mlflow.set_experiment("orbit-wars-xgboost")
    with mlflow.start_run():
        mlflow.log_params({
            "feature_version": FEATURE_VERSION,
            "n_samples":   int(len(X)),
            "n_stop_rows": int((~mask_ns).sum()),
            **{f"xgb_{k}": v for k, v in XGB_PARAMS.items()},
        })

        # ── Model 1: from (45-class, all rows) ──────────────────────────────
        Xtr, Xva, ytr, yva = train_test_split(X, y_from, test_size=0.2,
                                               random_state=RANDOM_STATE)
        m_from = xgb.XGBClassifier(num_class=45, objective="multi:softmax",
                                    eval_metric="mlogloss", **XGB_PARAMS)
        m_from.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
        acc_from = accuracy_score(yva, m_from.predict(Xva))
        mlflow.log_metric("from_val_accuracy", acc_from)
        mlflow.xgboost.log_model(m_from, "model_from")
        m_from.save_model(MODEL_DIR / "from.ubj")
        print(f"from  val accuracy : {acc_from:.3f}")

        # ── Model 2: to (44-class, non-stop rows) ───────────────────────────
        X_ns   = X[mask_ns]
        yto_ns = y_to[mask_ns].astype(int)
        Xtr2, Xva2, ytr2, yva2 = train_test_split(X_ns, yto_ns, test_size=0.2,
                                                    random_state=RANDOM_STATE)
        m_to = xgb.XGBClassifier(num_class=44, objective="multi:softmax",
                                  eval_metric="mlogloss", **XGB_PARAMS)
        m_to.fit(Xtr2, ytr2, eval_set=[(Xva2, yva2)], verbose=False)
        acc_to = accuracy_score(yva2, m_to.predict(Xva2))
        mlflow.log_metric("to_val_accuracy", acc_to)
        mlflow.xgboost.log_model(m_to, "model_to")
        m_to.save_model(MODEL_DIR / "to.ubj")
        print(f"to    val accuracy : {acc_to:.3f}")

        # ── Model 3: ships (regressor, conditioned on from + to) ────────────
        yfrom_ns = y_from[mask_ns].astype(int)
        from_oh  = np.eye(44, dtype=np.float32)[np.clip(yfrom_ns, 0, 43)]
        to_oh    = np.eye(44, dtype=np.float32)[yto_ns]
        X_ships  = np.concatenate([X_ns, from_oh, to_oh], axis=1)
        ysh_ns   = y_ships[mask_ns]
        Xtr3, Xva3, ytr3, yva3 = train_test_split(X_ships, ysh_ns, test_size=0.2,
                                                    random_state=RANDOM_STATE)
        m_ships = xgb.XGBRegressor(objective="reg:squarederror", **XGB_PARAMS)
        m_ships.fit(Xtr3, ytr3, eval_set=[(Xva3, yva3)], verbose=False)
        mae = float(np.abs(yva3 - m_ships.predict(Xva3)).mean())
        mlflow.log_metric("ships_val_mae", mae)
        mlflow.xgboost.log_model(m_ships, "model_ships")
        m_ships.save_model(MODEL_DIR / "ships.ubj")
        print(f"ships val MAE      : {mae:.1f}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run training**

```bash
python 17-Train_XGBoost.py
```

Expected: three metric lines printed, files created in `models/xgboost/`.

- [ ] **Step 3: Verify MLFlow**

```bash
mlflow ui --port 5000
```

Navigate to `http://localhost:5000`, confirm experiment "orbit-wars-xgboost" with `from_val_accuracy`, `to_val_accuracy`, `ships_val_mae`.

- [ ] **Step 4: Commit**

```bash
git add 17-Train_XGBoost.py
git commit -m "feat: XGBoost training with MLFlow (from/to/ships)"
```

---

## Task 8: `17-Play_XGBoost.py` — XGBoost inference

**Files:**
- Create: `17-Play_XGBoost.py`

- [ ] **Step 1: Create `17-Play_XGBoost.py`**

```python
import numpy as np
import xgboost as xgb
from pathlib import Path
from pipeline.env_sim import simulate_futures, NB_FUTURE_STEP
from pipeline.features import (
    build_slot_map, extract_features, planet_id_to_slot, slot_to_planet_id,
)

MODEL_DIR       = Path("models/xgboost")
FEATURE_VERSION = 1
MAX_ACTIONS     = 5   # safety cap on action loop


class XGBoostAgent:
    def __init__(self, model_dir=MODEL_DIR, feature_version=FEATURE_VERSION):
        self.feature_version = feature_version
        self.m_from  = xgb.XGBClassifier();  self.m_from.load_model(model_dir / "from.ubj")
        self.m_to    = xgb.XGBClassifier();  self.m_to.load_model(model_dir / "to.ubj")
        self.m_ships = xgb.XGBRegressor();   self.m_ships.load_model(model_dir / "ships.ubj")

    def predict_actions(self, obs, aiming_fn):
        """
        obs        — observation dict (planets, fleets, step, ...)
        aiming_fn  — callable(from_planet_id, to_planet_id, planets) -> angle (float)
        Returns list of [from_planet_id, angle, nb_ships, to_planet_id].
        """
        non_comet_ids, comet_ids = build_slot_map(obs["planets"])
        done_set = []
        actions  = []

        for _ in range(MAX_ACTIONS):
            future_planets = simulate_futures(obs, done_set, NB_FUTURE_STEP)
            feat = extract_features(
                future_planets, non_comet_ids, comet_ids,
                version=self.feature_version,
            ).reshape(1, -1)

            from_slot = int(self.m_from.predict(feat)[0])
            if from_slot == 44:  # stop
                break

            to_slot = int(self.m_to.predict(feat)[0])
            from_oh = np.eye(44, dtype=np.float32)[[min(from_slot, 43)]]
            to_oh   = np.eye(44, dtype=np.float32)[[to_slot]]
            feat_sh = np.concatenate([feat, from_oh, to_oh], axis=1)
            nb_ships = max(1, int(round(float(self.m_ships.predict(feat_sh)[0]))))

            from_pid = slot_to_planet_id(from_slot, non_comet_ids, comet_ids)
            to_pid   = slot_to_planet_id(to_slot,   non_comet_ids, comet_ids)
            angle    = aiming_fn(from_pid, to_pid, obs["planets"])
            action   = [from_pid, angle, nb_ships, to_pid]
            actions.append(action)
            done_set.append(action)

        return actions


if __name__ == "__main__":
    agent = XGBoostAgent()
    print("XGBoostAgent loaded OK")
    print(f"  from model classes : {agent.m_from.n_classes_}")
    print(f"  to   model classes : {agent.m_to.n_classes_}")
```

- [ ] **Step 2: Run smoke test**

```bash
python 17-Play_XGBoost.py
```

Expected:
```
XGBoostAgent loaded OK
  from model classes : 45
  to   model classes : 44
```

- [ ] **Step 3: Commit**

```bash
git add 17-Play_XGBoost.py
git commit -m "feat: XGBoost inference agent"
```

---

## Task 9: `18-Train_ML.py` — MLP training + MLFlow

**Files:**
- Create: `18-Train_ML.py`

- [ ] **Step 1: Create `18-Train_ML.py`**

```python
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import mlflow
from pathlib import Path
from pipeline.features import build_feature_matrix
from pipeline.model import OrbitMLP

SIMULATE_DIR    = Path("11-download_logs/04-simulate")
MODEL_DIR       = Path("models/mlp")
FEATURE_VERSION = 1
BATCH_SIZE      = 256
EPOCHS          = 50
LR              = 1e-3
RANDOM_STATE    = 42


def make_loaders(X, y_from, y_to, y_ships, batch_size, seed):
    mask_ns  = y_from != 44
    X_t      = torch.tensor(X, dtype=torch.float32)
    yf_t     = torch.tensor(y_from, dtype=torch.long)
    yt_t     = torch.tensor(np.nan_to_num(y_to,    nan=0.0), dtype=torch.long)
    ys_t     = torch.tensor(np.nan_to_num(y_ships, nan=0.0), dtype=torch.float32)
    mk_t     = torch.tensor(mask_ns, dtype=torch.bool)

    g = torch.Generator().manual_seed(seed)
    idx   = torch.randperm(len(X_t), generator=g)
    n_val = int(0.2 * len(X_t))
    tr, va = idx[n_val:], idx[:n_val]

    tr_ds = TensorDataset(X_t[tr], yf_t[tr], yt_t[tr], ys_t[tr], mk_t[tr])
    va_ds = TensorDataset(X_t[va], yf_t[va], yt_t[va], ys_t[va], mk_t[va])
    return (DataLoader(tr_ds, batch_size=batch_size, shuffle=True),
            DataLoader(va_ds, batch_size=batch_size))


def eval_epoch(model, loader):
    model.eval()
    ce_from = nn.CrossEntropyLoss()
    acc_f = acc_t = mae_s = 0.0
    n = n_ns = 0
    with torch.no_grad():
        for xb, yf, yt, ys, mk in loader:
            lf, lt, ps = model(xb)
            acc_f += (lf.argmax(1) == yf).float().mean().item()
            n += 1
            if mk.any():
                acc_t += (lt[mk].argmax(1) == yt[mk]).float().mean().item()
                mae_s += (ps[mk] - ys[mk]).abs().mean().item()
                n_ns  += 1
    return acc_f / n, acc_t / max(n_ns, 1), mae_s / max(n_ns, 1)


def main():
    torch.manual_seed(RANDOM_STATE)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    X, y_from, y_to, y_ships = build_feature_matrix(SIMULATE_DIR, version=FEATURE_VERSION)
    tr_loader, va_loader = make_loaders(X, y_from, y_to, y_ships, BATCH_SIZE, RANDOM_STATE)

    model     = OrbitMLP(input_dim=X.shape[1])
    optimizer = optim.Adam(model.parameters(), lr=LR)
    ce_from   = nn.CrossEntropyLoss()
    ce_to     = nn.CrossEntropyLoss()

    mlflow.set_experiment("orbit-wars-mlp")
    with mlflow.start_run():
        mlflow.log_params({
            "feature_version": FEATURE_VERSION,
            "epochs": EPOCHS, "batch_size": BATCH_SIZE, "lr": LR,
            "n_samples": int(len(X)), "input_dim": int(X.shape[1]),
        })

        for epoch in range(EPOCHS):
            model.train()
            for xb, yf, yt, ys, mk in tr_loader:
                lf, lt, ps = model(xb)
                loss = ce_from(lf, yf)
                if mk.any():
                    loss = loss + ce_to(lt[mk], yt[mk])
                    loss = loss + 0.01 * ((ps[mk] - ys[mk]) ** 2).mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if (epoch + 1) % 10 == 0:
                af, at, ms = eval_epoch(model, va_loader)
                mlflow.log_metrics(
                    {"from_val_acc": af, "to_val_acc": at, "ships_val_mae": ms},
                    step=epoch + 1,
                )
                print(f"Epoch {epoch+1:3d}  from={af:.3f}  to={at:.3f}  mae={ms:.1f}")

        model_path = MODEL_DIR / "orbit_mlp.pt"
        torch.save({"state_dict": model.state_dict(), "input_dim": X.shape[1]}, model_path)
        mlflow.log_artifact(str(model_path))
        print(f"Model saved → {model_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run training**

```bash
python 18-Train_ML.py
```

Expected: metrics printed every 10 epochs, model saved to `models/mlp/orbit_mlp.pt`.

- [ ] **Step 3: Verify MLFlow**

Open `http://localhost:5000`, confirm "orbit-wars-mlp" with `from_val_acc`, `to_val_acc`, `ships_val_mae` logged per 10 epochs.

- [ ] **Step 4: Commit**

```bash
git add 18-Train_ML.py
git commit -m "feat: MLP multi-task training with MLFlow"
```

---

## Task 10: `18-Play_ML.py` — MLP inference

**Files:**
- Create: `18-Play_ML.py`

- [ ] **Step 1: Create `18-Play_ML.py`**

```python
import torch
from pathlib import Path
from pipeline.env_sim import simulate_futures, NB_FUTURE_STEP
from pipeline.features import (
    build_slot_map, extract_features, slot_to_planet_id,
)
from pipeline.model import OrbitMLP

MODEL_PATH      = Path("models/mlp/orbit_mlp.pt")
FEATURE_VERSION = 1
MAX_ACTIONS     = 5


class MLPAgent:
    def __init__(self, model_path=MODEL_PATH, feature_version=FEATURE_VERSION):
        self.feature_version = feature_version
        ckpt = torch.load(model_path, map_location="cpu")
        self.input_dim = ckpt["input_dim"]
        self.model = OrbitMLP(input_dim=self.input_dim)
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()

    def predict_actions(self, obs, aiming_fn):
        """
        obs        — observation dict (planets, fleets, step, ...)
        aiming_fn  — callable(from_planet_id, to_planet_id, planets) -> angle (float)
        Returns list of [from_planet_id, angle, nb_ships, to_planet_id].
        """
        non_comet_ids, comet_ids = build_slot_map(obs["planets"])
        done_set = []
        actions  = []

        with torch.no_grad():
            for _ in range(MAX_ACTIONS):
                future_planets = simulate_futures(obs, done_set, NB_FUTURE_STEP)
                feat = extract_features(
                    future_planets, non_comet_ids, comet_ids,
                    version=self.feature_version,
                )
                x = torch.tensor(feat, dtype=torch.float32).unsqueeze(0)
                lf, lt, ps = self.model(x)

                from_slot = int(lf.argmax(1).item())
                if from_slot == 44:  # stop
                    break

                to_slot  = int(lt.argmax(1).item())
                nb_ships = max(1, int(round(ps.item())))

                from_pid = slot_to_planet_id(from_slot, non_comet_ids, comet_ids)
                to_pid   = slot_to_planet_id(to_slot,   non_comet_ids, comet_ids)
                angle    = aiming_fn(from_pid, to_pid, obs["planets"])
                action   = [from_pid, angle, nb_ships, to_pid]
                actions.append(action)
                done_set.append(action)

        return actions


if __name__ == "__main__":
    agent = MLPAgent()
    print("MLPAgent loaded OK")
    print(f"  input_dim : {agent.input_dim}")
```

- [ ] **Step 2: Run smoke test**

```bash
python 18-Play_ML.py
```

Expected:
```
MLPAgent loaded OK
  input_dim : 2200
```

- [ ] **Step 3: Commit**

```bash
git add 18-Play_ML.py
git commit -m "feat: MLP inference agent"
```
