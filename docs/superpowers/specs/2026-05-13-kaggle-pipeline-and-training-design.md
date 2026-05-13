# Kaggle Pipeline & GPU Training Design

## Goal

Build two self-contained Kaggle notebooks (`19-Pipeline_Kaggle.ipynb`, `20-Train_ML_Kaggle.ipynb`) that download top-10% Orbit Wars episode replays, run the full processing pipeline, and train an improved MLP on Kaggle GPU — producing a drop-in replacement for `models/mlp/orbit_mlp.pt`.

## Architecture

```
bovard/orbit-wars-top10-episodes-2026-05-01
bovard/orbit-wars-top10-episodes-2026-05-02   ──► 19-Pipeline_Kaggle.ipynb  (CPU)
bovard/orbit-wars-top10-episodes-2026-05-03        steps 01–04, all code inlined
bovard/orbit-wars-top10-episodes-2026-05-04        outputs: X.npy, y_from.npy,
                                                             y_to.npy, y_ships.npy
                                                        ↓ (Notebook Output → Input)
                                               20-Train_ML_Kaggle.ipynb  (GPU T4/P100)
                                                   improved OrbitMLP, GPU training
                                                   output: orbit_mlp.pt
                                                        ↓ (download locally)
                                               models/mlp/orbit_mlp.pt
                                               ← used by 18-Play_ML.py unchanged
```

**Data handoff**: Notebook 19's output files are added to Notebook 20 via
"+ Add Input → Notebook Output Files" — no public dataset is published.

## Tech Stack

- Python 3.10+ (Kaggle default)
- `kaggle_environments` — orbit_wars env (native on Kaggle, used for simulation)
- `numpy`, `torch` (CUDA) — feature matrix and training
- `tarfile` — streaming episode tarballs without full disk extraction

---

## Notebook 19 — Pipeline (CPU)

### Inputs

Four bovard day-datasets added via "+ Add Data":
- `bovard/orbit-wars-top10-episodes-2026-05-01`
- `bovard/orbit-wars-top10-episodes-2026-05-02`
- `bovard/orbit-wars-top10-episodes-2026-05-03`
- `bovard/orbit-wars-top10-episodes-2026-05-04`

Each mounts at `/kaggle/input/orbit-wars-top10-episodes-YYYY-MM-DD/` and contains:
- `episodes.tar.gz` — replay JSONs at `episodes/{episode_id}.json`
- `manifest.csv` — rows ordered by `sum_score` descending (highest quality first)

### Config (Cell 1)

```python
DAYS = [
    "2026-05-01",
    "2026-05-02",
    "2026-05-03",
    "2026-05-04",
]
MAX_EPISODES_PER_DAY = 50   # top-N by sum_score; change to process more/fewer
NB_FUTURE_STEP = 10         # future planet snapshots per simulation row
FEATURE_VERSION = 1         # 1 → 2200 features (44 planets × 10 steps × 5 fields)
```

### Pipeline Functions (Cell 2 — inlined, no local imports)

All functions from `pipeline/` are copied verbatim into this cell:

- `get_winner_data(raw_episode)` — extracts winner obs+action list
- `augment_data(data)` — adds `dest_planet_id` to each action via fleet tracking
- `generate_combinations(obs)` — expands one obs into (done_set, action_to_do) rows
- `simulate_futures(obs_dict, done_set, n_steps)` — injects obs into orbit_wars env, steps forward
- `build_slot_map(obs)` — derives (non_comet_ids, comet_ids) from obs
- `planet_id_to_slot(planet_id, non_comet_ids, comet_ids)` — maps planet id → slot 0-43
- `extract_features(future_planets, non_comet_ids, comet_ids, version)` — builds flat feature vector

### Main Loop (Cell 3)

```python
for day in DAYS:
    root = Path(f"/kaggle/input/orbit-wars-top10-episodes-{day}")
    with open(root / "manifest.csv") as f:
        manifest = list(csv.DictReader(f))
    episodes_to_process = manifest[:MAX_EPISODES_PER_DAY]

    with tarfile.open(root / "episodes.tar.gz", "r:gz") as tar:
        for row in episodes_to_process:
            member = tar.getmember(f"episodes/{row['episode_id']}.json")
            raw = json.load(tar.extractfile(member))
            # steps 01 → 02 → 03 → 04 → accumulate into all_rows
```

Steps per episode (in order):
1. `get_winner_data(raw)` → winner obs list
2. `augment_data(winner_data)` → adds dest_planet_id
3. `generate_combinations(obs)` for each obs with non-empty action → combo rows
4. `simulate_futures(row["obs"], row["done_set"], NB_FUTURE_STEP)` → `row["future_planets"]`

### Feature Matrix (Cell 4)

```python
# Build arrays from all_rows
X, y_from, y_to, y_ships = build_feature_matrix_from_rows(all_rows, FEATURE_VERSION)

np.save("/kaggle/working/X.npy",       X)
np.save("/kaggle/working/y_from.npy",  y_from)
np.save("/kaggle/working/y_to.npy",    y_to)
np.save("/kaggle/working/y_ships.npy", y_ships)
```

`build_feature_matrix_from_rows(rows, version)` is inlined in the notebook. It accepts a
pre-loaded list of rows (each with `"obs"`, `"future_planets"`, `"action_to_do"` keys) and
returns `(X, y_from, y_to, y_ships)` arrays — identical logic to
`pipeline/features.py:build_feature_matrix` but without the file-reading loop.

```python
def build_feature_matrix_from_rows(rows, version=1):
    if not rows:
        return np.zeros((0, NB_PLANETS * NB_FUTURE_STEP * 5), dtype=np.float32), \
               np.array([]), np.array([]), np.array([])
    non_comet_ids, comet_ids = build_slot_map(rows[0]["obs"])
    X_list, y_from_list, y_to_list, y_ships_list = [], [], [], []
    for row in rows:
        feat = extract_features(row["future_planets"], non_comet_ids, comet_ids, version)
        action = row["action_to_do"]
        if action is None:
            X_list.append(feat); y_from_list.append(44)
            y_to_list.append(np.nan); y_ships_list.append(np.nan)
        elif len(action) == 4:
            from_pid, _angle, ships, to_pid = action
            from_slot = planet_id_to_slot(from_pid, non_comet_ids, comet_ids)
            to_slot   = planet_id_to_slot(to_pid,   non_comet_ids, comet_ids)
            X_list.append(feat); y_from_list.append(float(from_slot))
            y_to_list.append(float(to_slot)); y_ships_list.append(float(ships))
    return (np.array(X_list, dtype=np.float32),
            np.array(y_from_list, dtype=np.float32),
            np.array(y_to_list,   dtype=np.float32),
            np.array(y_ships_list, dtype=np.float32))
```

Note: `build_slot_map` is called once on the first row's obs. This assumes all episodes in the
batch share the same planet layout — which is true within a single game but varies across games.
**Fix**: call `build_slot_map` per-episode, not once globally. The main loop accumulates rows
per episode and calls `build_feature_matrix_from_rows` per episode, then concatenates.

### Stats Cell (Cell 5)

Prints: total rows, stop-row fraction, invalid-action fraction, class counts for y_from and y_to.

### Expected output

~80K rows (50 eps/day × 4 days × ~400 rows/ep), pipeline runtime ~1–2 hours on CPU.
Output files: `X.npy` (shape N×2200), `y_from.npy`, `y_to.npy`, `y_ships.npy` (shape N,).

---

## Notebook 20 — Training (GPU)

### Inputs

Notebook 19's saved output version, added via "+ Add Input → Notebook Output Files".
The exact mount path is shown in the Kaggle UI after adding — typically
`/kaggle/input/<your-username>-<notebook-19-title>/`. Update `INPUT_DIR` to match.

### Config (Cell 1)

```python
INPUT_DIR   = Path("/kaggle/input/orbit-wars-pipeline-output")  # adjust to actual slug shown in UI
OUTPUT_DIR  = Path("/kaggle/working")
BATCH_SIZE  = 1024
EPOCHS      = 150
LR          = 1e-3
PATIENCE_LR = 10     # ReduceLROnPlateau patience (epochs)
PATIENCE_ES = 20     # early stopping patience (epochs)
LR_FACTOR   = 0.5
RANDOM_SEED = 42
```

### Improved MLP Architecture (Cell 2)

Replaces `pipeline/model.py:OrbitMLP`. Inlined as `OrbitMLP` in the notebook.

```python
class OrbitMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.BatchNorm1d(input_dim),
            nn.Linear(input_dim, 1024), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(1024, 512),       nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(512, 256),        nn.ReLU(),
        )
        self.head_from  = nn.Linear(256, 45)   # slots 0-43 + stop=44
        self.head_to    = nn.Linear(256, 44)   # slots 0-43
        self.head_ships = nn.Linear(256, 1)    # regression

    def forward(self, x):
        h = self.encoder(x)
        return self.head_from(h), self.head_to(h), self.head_ships(h).squeeze(-1)
```

### Training Loop (Cell 3)

- Load `X.npy`, `y_from.npy`, `y_to.npy`, `y_ships.npy`; move to GPU via `.cuda()`
- 80/20 train/val split (reproducible via `RANDOM_SEED`)
- Loss: `CrossEntropyLoss` for `head_from` (all rows) + `CrossEntropyLoss` for `head_to` (non-stop rows) + `0.01 × MSELoss` for `head_ships` (non-stop rows)
- Optimizer: `Adam(lr=LR)`
- Scheduler: `ReduceLROnPlateau(optimizer, mode="max", patience=PATIENCE_LR, factor=LR_FACTOR)` tracking `from_val_acc`
- Early stopping: halt if `from_val_acc` has not improved for `PATIENCE_ES` epochs; restore best weights
- Log per-epoch: `from_val_acc`, `to_val_acc`, `ships_val_mae`

### Checkpoint Save (Cell 4)

```python
torch.save(
    {"state_dict": model.state_dict(), "input_dim": input_dim},
    OUTPUT_DIR / "orbit_mlp.pt"
)
```

Same format as existing `18-Train_ML.py` output — `18-Play_ML.py` loads it unchanged.

### Local integration

After downloading `orbit_mlp.pt` from the Kaggle notebook output:
```
cp orbit_mlp.pt models/mlp/orbit_mlp.pt
python 18-Agent_ML.py --games 20
```

---

## Data Flow Summary

| Step | Input | Output | Location |
|------|-------|--------|----------|
| Notebook 19 runs | 4 bovard day-datasets | `X.npy`, `y_from.npy`, `y_to.npy`, `y_ships.npy` | `/kaggle/working/` (private) |
| Notebook 20 runs | Notebook 19 output | `orbit_mlp.pt` | `/kaggle/working/` |
| Local integration | Downloaded `orbit_mlp.pt` | Agent tested via `18-Agent_ML.py` | `models/mlp/` |

## What is NOT changed

- `18-Play_ML.py` — inference code unchanged; loads `orbit_mlp.pt` by path
- `18-Agent_ML.py` — test harness unchanged
- `pipeline/features.py`, `pipeline/env_sim.py` — remain as local reference; notebook code is inlined copies, not imports
- `pipeline/model.py` — **will be updated** to match the improved architecture (1024→512→256 + Dropout) so `18-Train_ML.py` also benefits when run locally
- `orbit-wars-lab/agents/mine/18-ML/main.py` — unchanged

## Out of scope

- Hyperparameter search (fixed config is sufficient for this iteration)
- Uploading the model back to Kaggle as a submission (manual step after local testing)
- Processing all 19 days (4 days is the chosen scope; changing `DAYS` list covers expansion)
