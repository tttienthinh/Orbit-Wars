# Kaggle Pipeline & GPU Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create `19-Pipeline_Kaggle.ipynb` (CPU, processes 4 days of top-10% replays into numpy feature arrays) and `20-Train_ML_Kaggle.ipynb` (GPU, trains improved OrbitMLP and saves `orbit_mlp.pt`), plus update `pipeline/model.py` to match the improved architecture locally.

**Architecture:** Notebook 19 streams episode tarballs from 4 bovard Kaggle datasets, runs all pipeline steps inline (no local imports), and saves `X.npy`/`y_from.npy`/`y_to.npy`/`y_ships.npy` as output. Notebook 20 loads those arrays, trains a wider MLP (1024→512→256, Dropout, `ReduceLROnPlateau`, early stopping), and saves `orbit_mlp.pt` in the same format that `18-Play_ML.py` already loads — no inference code changes needed. `pipeline/model.py` is updated locally so `18-Train_ML.py` also benefits.

**Tech Stack:** Python 3.10, numpy, pandas, torch (CUDA), kaggle_environments, tarfile, itertools, Kaggle Jupyter

---

**Files:**
- Modify: `pipeline/model.py`
- Create: `tests/test_model.py`
- Create: `19-Pipeline_Kaggle.ipynb`
- Create: `20-Train_ML_Kaggle.ipynb`

---

## Task 1: Update `pipeline/model.py` with improved OrbitMLP

**Files:**
- Modify: `pipeline/model.py`
- Create: `tests/test_model.py`

- [ ] **Step 1: Create `tests/test_model.py` with three failing assertions**

```python
# tests/test_model.py
import torch
from pipeline.model import OrbitMLP


def test_output_shapes():
    model = OrbitMLP(input_dim=2200)
    model.eval()
    x = torch.randn(4, 2200)
    lf, lt, ps = model(x)
    assert lf.shape == (4, 45), f"Expected (4,45), got {lf.shape}"
    assert lt.shape == (4, 44), f"Expected (4,44), got {lt.shape}"
    assert ps.shape == (4,),    f"Expected (4,), got {ps.shape}"


def test_has_dropout():
    model = OrbitMLP(input_dim=2200)
    assert any(isinstance(m, torch.nn.Dropout) for m in model.modules()), \
        "Expected Dropout layers in encoder"


def test_param_count_larger():
    model = OrbitMLP(input_dim=2200)
    n = sum(p.numel() for p in model.parameters())
    assert n > 2_000_000, f"Expected >2M params for wider net, got {n:,}"
```

- [ ] **Step 2: Run tests — expect 2 failures**

```
pytest tests/test_model.py -v
```

Expected: `test_output_shapes` PASSES (shapes unchanged), `test_has_dropout` FAILS, `test_param_count_larger` FAILS.

- [ ] **Step 3: Replace `pipeline/model.py` with improved architecture**

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
            nn.Linear(input_dim, 1024), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(1024, 512),       nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(512, 256),        nn.ReLU(),
        )
        self.head_from  = nn.Linear(256, 45)
        self.head_to    = nn.Linear(256, 44)
        self.head_ships = nn.Linear(256, 1)

    def forward(self, x):
        h = self.encoder(x)
        return self.head_from(h), self.head_to(h), self.head_ships(h).squeeze(-1)
```

- [ ] **Step 4: Run tests — expect all 3 to pass**

```
pytest tests/test_model.py -v
```

Expected: 3 PASSED. New param count ≈ 3.7M (was ≈ 1.3M).

- [ ] **Step 5: Commit**

```
git add pipeline/model.py tests/test_model.py
git commit -m "feat: widen OrbitMLP to 1024-512-256 with Dropout"
```

---

## Task 2: Create `19-Pipeline_Kaggle.ipynb`

**Files:**
- Create: `19-Pipeline_Kaggle.ipynb`

The notebook has 6 cells. Create it with `mcp__jupyter__notebook_create`, then add each cell with `mcp__jupyter__notebook_add_cell`.

- [ ] **Step 1: Create the empty notebook**

```
mcp__jupyter__notebook_create path="19-Pipeline_Kaggle.ipynb"
```

- [ ] **Step 2: Add Cell 1 — Config**

Cell type: `code`. Content:

```python
import csv, copy, json, math, tarfile, warnings
import numpy as np
import pandas as pd
from itertools import combinations as _combinations
from pathlib import Path
import kaggle_environments as ke

warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────
DAYS = [
    "2026-05-01",
    "2026-05-02",
    "2026-05-03",
    "2026-05-04",
]
MAX_EPISODES_PER_DAY = 50   # top-N by sum_score per day
NB_FUTURE_STEP       = 10   # future planet snapshots per simulation row
NB_PLANETS           = 44   # 40 non-comets + 4 comets
FEATURE_VERSION      = 1    # 1 → 2200 features (44 × 10 × 5)
MAX_SPEED            = 6.0
PLANET_MARGIN        = 0.1
OUTPUT_DIR           = Path("/kaggle/working")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
```

- [ ] **Step 3: Add Cell 2 — Inlined pipeline functions**

Cell type: `code`. Content:

```python
# ── Step 01: Winner extraction ────────────────────────────────────────────────
def get_winner_data(raw):
    winner_i = int(np.argmax(raw["rewards"]))
    new_data = []
    for step_list in raw["steps"]:
        obs = dict(step_list[winner_i]["observation"])
        obs["action"] = step_list[winner_i]["action"]
        new_data.append(obs)
    return new_data


# ── Step 02: Augment with destination planet ──────────────────────────────────
def _fleet_speed(nb_ships):
    if nb_ships <= 1:
        return 1.0
    ratio = math.log(nb_ships) / math.log(1000.0)
    return 1.0 + (MAX_SPEED - 1.0) * max(0.0, min(1.0, ratio)) ** 1.5


def augment_data(data):
    fleet_rows, planet_rows = [], []
    for obs in data:                          # fixed: was `winner_data` (global bug)
        s = obs["step"]
        for f in obs["fleets"]:
            fleet_rows.append({"step": s, "id": f[0], "owner": f[1],
                                "x": f[2], "y": f[3], "angle": f[4],
                                "from_planet_id": f[5], "ships": f[6]})
        for p in obs["planets"]:
            planet_rows.append({"step": s, "id": p[0], "owner": p[1],
                                 "x": p[2], "y": p[3], "radius": p[4],
                                 "ships": p[5], "production": p[6]})

    if not fleet_rows or not planet_rows:
        return data

    df_fleet   = pd.DataFrame(fleet_rows).drop_duplicates(["step", "id"]).reset_index(drop=True)
    df_planets = pd.DataFrame(planet_rows).drop_duplicates(["step", "id"]).reset_index(drop=True)

    df_fleet_last = (
        df_fleet
        .assign(first_step=lambda df: df.groupby("id")["step"].transform("min"))
        .groupby("id").last().reset_index()
        .assign(
            speed=lambda df: df["ships"].apply(_fleet_speed),
            dx=lambda df: df["speed"] * np.cos(df["angle"]),
            dy=lambda df: df["speed"] * np.sin(df["angle"]),
            next_step=lambda df: df["step"] + 1,
        )
        .merge(df_planets, left_on="next_step", right_on="step",
               suffixes=("_fleet", "_planet"), how="inner")
        .assign(
            t=lambda df: (
                (df["dx"] * (df["x_planet"] - df["x_fleet"]) +
                 df["dy"] * (df["y_planet"] - df["y_fleet"])) / df["speed"] ** 2
            ).clip(lower=0, upper=1),
            distance=lambda df: np.sqrt(
                (df["x_fleet"] + df["dx"] * df["t"] - df["x_planet"]) ** 2 +
                (df["y_fleet"] + df["dy"] * df["t"] - df["y_planet"]) ** 2),
            on_target=lambda df: df["distance"] < df["radius"] + PLANET_MARGIN,
            angle_3=lambda df: df["angle"].round(3),
        )
        .query("on_target")
        .groupby("id_fleet").first()
        .set_index(["first_step", "from_planet_id", "angle_3", "ships_fleet"])
    )
    fleet_arrival = df_fleet_last["id_planet"].to_dict()

    for step_i, obs in enumerate(data):
        if step_i == 0:
            continue
        new_actions = []
        for from_pid, angle, ships in obs["action"]:
            key = (step_i, from_pid, round(angle, 3), ships)
            if key in fleet_arrival:
                new_actions.append([from_pid, angle, ships, fleet_arrival[key]])
        data[step_i - 1]["action"] = new_actions
    return data


# ── Step 03: Combinatorial augmentation ──────────────────────────────────────
def generate_combinations(obs):
    actions = obs["action"]
    nb = len(actions)
    rows = []
    for nb_done in range(nb):
        for done_idx in _combinations(range(nb), nb_done):
            done = [actions[i] for i in done_idx]
            remaining = [actions[i] for i in range(nb) if i not in done_idx]
            for action_to_do in remaining:
                rows.append({"obs": obs, "done_set": done, "action_to_do": action_to_do})
    rows.append({"obs": obs, "done_set": actions, "action_to_do": None})
    return rows


# ── Step 04: Env simulation ───────────────────────────────────────────────────
def simulate_futures(obs, done_set, nb_future_step=NB_FUTURE_STEP):
    env = ke.make("orbit_wars", debug=False)
    env.reset()
    for key, val in obs.items():
        env.state[0].observation[key] = val
        env.state[1].observation[key] = val
    game_actions = [a[:3] for a in done_set]   # strip dest_planet_id if present
    env.step([game_actions, []])
    snapshots = []
    last = copy.deepcopy(list(env.state[0].observation.get("planets", [])))
    for _ in range(nb_future_step):
        if env.done:
            snapshots.append(last)
            continue
        env.step([[], []])
        last = copy.deepcopy(list(env.state[0].observation.get("planets", [])))
        snapshots.append(last)
    return snapshots


# ── Feature extraction ────────────────────────────────────────────────────────
def build_slot_map(obs):
    comet_ids = sorted(obs.get("comet_planet_ids", []))
    comet_set = set(comet_ids)
    all_ids = [p[0] for p in obs["planets"]]
    non_comet_ids = sorted(pid for pid in all_ids if pid not in comet_set)
    return non_comet_ids, comet_ids


def planet_id_to_slot(planet_id, non_comet_ids, comet_ids):
    if planet_id in comet_ids:
        return 40 + comet_ids.index(planet_id)
    if planet_id in non_comet_ids:
        return non_comet_ids.index(planet_id)
    return -1


def extract_features(future_planets, non_comet_ids, comet_ids, version=1):
    feat = np.zeros((NB_FUTURE_STEP, NB_PLANETS, 5), dtype=np.float32)
    for step_i, planets in enumerate(future_planets):
        for p in planets:
            pid, owner, x, y, radius, ships, prod = p[0], p[1], p[2], p[3], p[4], p[5], p[6]
            slot = planet_id_to_slot(pid, non_comet_ids, comet_ids)
            if slot < 0:
                continue
            feat[step_i, slot] = [owner, ships, x, y, prod]
    return feat.flatten()   # shape: (2200,) for version=1
```

- [ ] **Step 4: Add Cell 3 — Main processing loop**

Cell type: `code`. Content:

```python
all_X, all_yf, all_yt, all_ys = [], [], [], []
episode_count = 0

for day in DAYS:
    slug = f"orbit-wars-top10-episodes-{day}"
    root = Path(f"/kaggle/input/{slug}")
    if not root.exists():
        print(f"[SKIP] {root} not found — add it via + Add Data on Kaggle")
        continue

    with open(root / "manifest.csv") as f:
        manifest = list(csv.DictReader(f))

    to_process = manifest[:MAX_EPISODES_PER_DAY]
    print(f"\nDay {day}: processing {len(to_process)} / {len(manifest)} episodes")

    with tarfile.open(root / "episodes.tar.gz", "r:gz") as tar:
        for row_idx, row in enumerate(to_process):
            ep_id = row["episode_id"]
            try:
                member = tar.getmember(f"episodes/{ep_id}.json")
                raw    = json.load(tar.extractfile(member))

                # Steps 01 → 02
                winner_data = get_winner_data(raw)
                winner_data = augment_data(winner_data)

                # Step 03: combinations (skip obs with no actions)
                rows = []
                for obs in winner_data:
                    if obs.get("action"):
                        rows.extend(copy.deepcopy(generate_combinations(obs)))

                if not rows:
                    continue

                # Step 04: simulate futures
                for r in rows:
                    r["future_planets"] = simulate_futures(
                        r["obs"], r["done_set"], NB_FUTURE_STEP
                    )

                # Feature extraction — per-episode slot map
                non_comet_ids, comet_ids = build_slot_map(rows[0]["obs"])
                for r in rows:
                    feat   = extract_features(r["future_planets"], non_comet_ids, comet_ids)
                    action = r["action_to_do"]
                    if action is None:
                        all_X.append(feat); all_yf.append(44.0)
                        all_yt.append(np.nan); all_ys.append(np.nan)
                    elif len(action) == 4:
                        from_pid, _angle, ships, to_pid = action
                        fs = planet_id_to_slot(from_pid, non_comet_ids, comet_ids)
                        ts = planet_id_to_slot(to_pid,   non_comet_ids, comet_ids)
                        all_X.append(feat); all_yf.append(float(fs))
                        all_yt.append(float(ts)); all_ys.append(float(ships))

                episode_count += 1
                if (row_idx + 1) % 10 == 0:
                    print(f"  {row_idx+1}/{len(to_process)} eps | rows so far: {len(all_X):,}")

            except Exception as e:
                print(f"  ERROR ep {ep_id}: {e}")
                continue

print(f"\nDone. {episode_count} episodes processed, {len(all_X):,} total rows.")
```

- [ ] **Step 5: Add Cell 4 — Build feature matrix and save**

Cell type: `code`. Content:

```python
X       = np.array(all_X,  dtype=np.float32)
y_from  = np.array(all_yf, dtype=np.float32)
y_to    = np.array(all_yt, dtype=np.float32)
y_ships = np.array(all_ys, dtype=np.float32)

np.save(OUTPUT_DIR / "X.npy",       X)
np.save(OUTPUT_DIR / "y_from.npy",  y_from)
np.save(OUTPUT_DIR / "y_to.npy",    y_to)
np.save(OUTPUT_DIR / "y_ships.npy", y_ships)

print(f"Saved to {OUTPUT_DIR}")
print(f"  X shape      : {X.shape}")
print(f"  y_from shape : {y_from.shape}")
```

- [ ] **Step 6: Add Cell 5 — Stats**

Cell type: `code`. Content:

```python
stop_mask    = y_from == 44
invalid_mask = np.isnan(y_to) & ~stop_mask   # non-stop rows with missing dest

print(f"Total rows      : {len(X):,}")
print(f"Stop rows       : {stop_mask.sum():,}  ({100*stop_mask.mean():.1f}%)")
print(f"Invalid rows    : {invalid_mask.sum():,}")
print(f"Action rows     : {(~stop_mask & ~invalid_mask).sum():,}")
print(f"\ny_from distribution (slot → count):")
for slot in sorted(np.unique(y_from[~np.isnan(y_from)]).astype(int)):
    count = (y_from == slot).sum()
    label = "stop" if slot == 44 else str(slot)
    print(f"  slot {label:>3}: {count:,}")
```

- [ ] **Step 7: Local sanity check — run notebook cells against existing 2 episodes**

This verifies the inlined code matches the existing `pipeline/` modules.

In a local Python session (not on Kaggle), run:
```python
import json, copy, warnings
import numpy as np
warnings.filterwarnings("ignore")

# Paste Cell 1 config with overrides:
NB_FUTURE_STEP = 10
NB_PLANETS = 44
MAX_SPEED = 6.0
PLANET_MARGIN = 0.1

# Paste Cell 2 functions verbatim.

# Then test against existing episode:
with open("11-download_logs/00-raw/episode-76319029.json") as f:
    raw = json.load(f)

winner = get_winner_data(raw)
augmented = augment_data(winner)
step_41 = next(o for o in augmented if o["step"] == 41)
assert step_41["action"] == [[0, 5.199833393096924, 70, 8]], \
    f"augment_data broken: {step_41['action']}"
print("augment_data OK — dest planet 8 confirmed")

rows = []
for obs in augmented:
    if obs.get("action"):
        rows.extend(copy.deepcopy(generate_combinations(obs)))

# Compare row count with existing 03-combinations output
import json as _json
with open("11-download_logs/03-combinations/episode-76319029.json") as f:
    expected_rows = _json.load(f)
assert len(rows) == len(expected_rows), \
    f"generate_combinations count mismatch: {len(rows)} vs {len(expected_rows)}"
print(f"generate_combinations OK — {len(rows)} rows matches existing 03-combinations")
```

Run with: `python -c "exec(open('tests/test_notebook19_sanity.py').read())"` after saving the above as `tests/test_notebook19_sanity.py`.

Expected output:
```
augment_data OK — dest planet 8 confirmed
generate_combinations OK — N rows matches existing 03-combinations
```

- [ ] **Step 8: Commit**

```
git add 19-Pipeline_Kaggle.ipynb tests/test_notebook19_sanity.py
git commit -m "feat: add 19-Pipeline_Kaggle notebook with inlined pipeline steps 01-04"
```

---

## Task 3: Create `20-Train_ML_Kaggle.ipynb`

**Files:**
- Create: `20-Train_ML_Kaggle.ipynb`

- [ ] **Step 1: Create the empty notebook**

```
mcp__jupyter__notebook_create path="20-Train_ML_Kaggle.ipynb"
```

- [ ] **Step 2: Add Cell 1 — Config**

Cell type: `code`. Content:

```python
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
# Adjust INPUT_DIR to the actual slug shown in Kaggle UI after adding
# Notebook 19's output via "+ Add Input → Notebook Output Files"
INPUT_DIR   = Path("/kaggle/input/orbit-wars-pipeline-output")
OUTPUT_DIR  = Path("/kaggle/working")

BATCH_SIZE  = 1024
EPOCHS      = 150
LR          = 1e-3
PATIENCE_LR = 10    # ReduceLROnPlateau: halve LR after this many non-improving epochs
PATIENCE_ES = 20    # Early stopping: halt after this many non-improving epochs
LR_FACTOR   = 0.5
RANDOM_SEED = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
```

- [ ] **Step 3: Add Cell 2 — Inlined OrbitMLP**

Cell type: `code`. Content:

```python
class OrbitMLP(nn.Module):
    """
    Multi-task MLP for imitation learning.
    Heads: from (45 classes, 44=stop), to (44 classes), ships (regression).
    """
    def __init__(self, input_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.BatchNorm1d(input_dim),
            nn.Linear(input_dim, 1024), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(1024, 512),       nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(512, 256),        nn.ReLU(),
        )
        self.head_from  = nn.Linear(256, 45)
        self.head_to    = nn.Linear(256, 44)
        self.head_ships = nn.Linear(256, 1)

    def forward(self, x):
        h = self.encoder(x)
        return self.head_from(h), self.head_to(h), self.head_ships(h).squeeze(-1)
```

- [ ] **Step 4: Add Cell 3 — Data loading and train/val split**

Cell type: `code`. Content:

```python
X       = np.load(INPUT_DIR / "X.npy")
y_from  = np.load(INPUT_DIR / "y_from.npy")
y_to    = np.load(INPUT_DIR / "y_to.npy")
y_ships = np.load(INPUT_DIR / "y_ships.npy")

# Compute mask BEFORE nan_to_num: stop rows have y_to=NaN, action rows have y_to=slot (0-43)
mask_ns = (y_from != 44) & (~np.isnan(y_to))   # non-stop rows with valid destination

y_to    = np.nan_to_num(y_to,    nan=0.0).astype(np.int64)
y_ships = np.nan_to_num(y_ships, nan=0.0)

input_dim = X.shape[1]
print(f"Loaded  X={X.shape}  y_from={y_from.shape}")
print(f"Non-stop rows: {mask_ns.sum():,} / {len(X):,}")

X_t      = torch.tensor(X,                        dtype=torch.float32)
yf_t     = torch.tensor(y_from.astype(np.int64),  dtype=torch.long)
yt_t     = torch.tensor(y_to,                     dtype=torch.long)
ys_t     = torch.tensor(y_ships.astype(np.float32), dtype=torch.float32)
mk_t     = torch.tensor(mask_ns,                  dtype=torch.bool)

g   = torch.Generator().manual_seed(RANDOM_SEED)
idx = torch.randperm(len(X_t), generator=g)
n_val = int(0.2 * len(X_t))
tr_idx, va_idx = idx[n_val:], idx[:n_val]

tr_ds = TensorDataset(X_t[tr_idx], yf_t[tr_idx], yt_t[tr_idx], ys_t[tr_idx], mk_t[tr_idx])
va_ds = TensorDataset(X_t[va_idx], yf_t[va_idx], yt_t[va_idx], ys_t[va_idx], mk_t[va_idx])

tr_loader = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True)
va_loader = DataLoader(va_ds, batch_size=BATCH_SIZE)
print(f"Train batches: {len(tr_loader)}  Val batches: {len(va_loader)}")
```

- [ ] **Step 5: Add Cell 4 — Training loop with scheduler and early stopping**

Cell type: `code`. Content:

```python
model     = OrbitMLP(input_dim).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LR)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="max", patience=PATIENCE_LR, factor=LR_FACTOR, verbose=True
)
ce_from = nn.CrossEntropyLoss()
ce_to   = nn.CrossEntropyLoss()


def eval_epoch(loader):
    model.eval()
    acc_f = acc_t = mae_s = 0.0
    n = n_ns = 0
    with torch.no_grad():
        for xb, yf, yt, ys, mk in loader:
            xb, yf, yt, ys, mk = xb.to(DEVICE), yf.to(DEVICE), yt.to(DEVICE), ys.to(DEVICE), mk.to(DEVICE)
            lf, lt, ps = model(xb)
            acc_f += (lf.argmax(1) == yf).float().mean().item()
            n += 1
            if mk.any():
                acc_t += (lt[mk].argmax(1) == yt[mk]).float().mean().item()
                mae_s += (ps[mk] - ys[mk]).abs().mean().item()
                n_ns  += 1
    return acc_f / max(n, 1), acc_t / max(n_ns, 1), mae_s / max(n_ns, 1)


best_acc   = -1.0
best_state = None
no_improve = 0

for epoch in range(EPOCHS):
    model.train()
    for xb, yf, yt, ys, mk in tr_loader:
        xb, yf, yt, ys, mk = xb.to(DEVICE), yf.to(DEVICE), yt.to(DEVICE), ys.to(DEVICE), mk.to(DEVICE)
        lf, lt, ps = model(xb)
        loss = ce_from(lf, yf)
        if mk.any():
            loss = loss + ce_to(lt[mk], yt[mk])
            loss = loss + 0.01 * ((ps[mk] - ys[mk]) ** 2).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    af, at, ms = eval_epoch(va_loader)
    scheduler.step(af)

    if af > best_acc:
        best_acc   = af
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        no_improve = 0
    else:
        no_improve += 1

    if (epoch + 1) % 10 == 0 or no_improve == 0:
        print(f"Epoch {epoch+1:3d}  from={af:.3f}  to={at:.3f}  mae={ms:.1f}  "
              f"best={best_acc:.3f}  no_improve={no_improve}")

    if no_improve >= PATIENCE_ES:
        print(f"Early stopping at epoch {epoch+1}")
        break

# Restore best weights
model.load_state_dict(best_state)
print(f"\nTraining complete. Best from_val_acc = {best_acc:.3f}")
```

- [ ] **Step 6: Add Cell 5 — Save checkpoint**

Cell type: `code`. Content:

```python
out_path = OUTPUT_DIR / "orbit_mlp.pt"
torch.save({"state_dict": model.state_dict(), "input_dim": input_dim}, out_path)
print(f"Saved: {out_path}")
print(f"  input_dim : {input_dim}")
print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

# Smoke test: reload and run inference on one batch
ckpt = torch.load(out_path, map_location="cpu")
m2   = OrbitMLP(ckpt["input_dim"])
m2.load_state_dict(ckpt["state_dict"])
m2.eval()
x_test = torch.randn(2, ckpt["input_dim"])
lf, lt, ps = m2(x_test)
assert lf.shape == (2, 45) and lt.shape == (2, 44) and ps.shape == (2,)
print("Reload smoke test passed.")
```

- [ ] **Step 7: Local sanity check — train on existing 794 rows**

Verify the notebook's training code works end-to-end before uploading to Kaggle.

In a local Python session, run:
```python
import numpy as np, torch
from pathlib import Path
from pipeline.features import build_feature_matrix

# Build feature matrix from existing 04-simulate data
X, y_from, y_to, y_ships = build_feature_matrix(Path("11-download_logs/04-simulate"))
print(f"Local data: X={X.shape}")  # expect (794, 2200)

# Paste Cell 2 (OrbitMLP) and Cell 3 (data loading) verbatim,
# pointing INPUT_DIR at "11-download_logs/04-simulate" manually:
# (save npy files first)
import os
os.makedirs("tmp_feat", exist_ok=True)
np.save("tmp_feat/X.npy", X)
np.save("tmp_feat/y_from.npy", y_from)
np.save("tmp_feat/y_to.npy", y_to)
np.save("tmp_feat/y_ships.npy", y_ships)

# Then run training for 5 epochs only (change EPOCHS=5 locally)
# and verify loss decreases.
```

Save as `tests/test_notebook20_sanity.py` and run:
```
python tests/test_notebook20_sanity.py
```

Expected: training runs 5 epochs without error, `from_val_acc` metric is printed.

- [ ] **Step 8: Commit**

```
git add 20-Train_ML_Kaggle.ipynb tests/test_notebook20_sanity.py
git commit -m "feat: add 20-Train_ML_Kaggle notebook with GPU training and early stopping"
```

---

## Kaggle Workflow (after implementation)

1. Go to Kaggle → Create Notebook → name it `19-Pipeline-Orbit-Wars`
2. Add 4 bovard datasets via **+ Add Data**: search `orbit-wars-top10-episodes` and add each of the 4 days
3. Set accelerator to **CPU** (simulation is CPU-bound)
4. Copy-paste the 6 cells from `19-Pipeline_Kaggle.ipynb`
5. Run all cells (~1–2 hours); click **Save Version**
6. Create Notebook `20-Train-Orbit-Wars` → set accelerator to **GPU T4**
7. Add Notebook 19's output via **+ Add Input → Notebook Output Files**
8. Update `INPUT_DIR` to match the path shown in the Kaggle UI
9. Copy-paste the 6 cells from `20-Train_ML_Kaggle.ipynb`
10. Run all cells (~15–30 min with GPU); download `orbit_mlp.pt` from output
11. Locally: `cp orbit_mlp.pt models/mlp/orbit_mlp.pt && python 18-Agent_ML.py --games 20`
