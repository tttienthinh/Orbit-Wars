# Episode Log Pipeline & ML Model Design

**Date:** 2026-05-13
**Scope:** Steps 3–5 of the episode log transformation pipeline, plus ML model architecture for imitation learning from winner replays.

---

## Context

Steps 1 and 2 are already implemented in `11-download_logs.ipynb`:
- **Step 1** (`00-raw` → `01-winner`): extract winner's obs/action sequence
- **Step 2** (`01-winner` → `02-augment`): add destination planet id to each action

This spec covers steps 3–5 and the downstream ML model.

---

## Step 3 — Combinatorial Augmentation (`02-augment` → `03-combinations`)

Pure data expansion, no env simulation. For each obs:

```
actions = obs["action"]   # [[from_planet_id, angle, ships, dest_planet_id], ...]
nb_action = len(actions)

for nb_action_done in range(nb_action):
    for done_set in combinations(actions, nb_action_done):
        for action_to_do in actions not in done_set:
            emit deepcopy({obs, done_set, action_to_do})

# stop row: all actions committed, nothing left
emit deepcopy({obs, done_set=all_actions, action_to_do=None})
```

**Output format:** JSON rows, each containing `{step_obs, done_set, action_to_do}`.
**No env involved here.** Step 3 and Step 4 are kept separate so `NB_FUTURE_STEP` can be changed by re-running Step 4 only.

---

## Step 4 — Env Simulation (`03-combinations` → `04-simulate`)

For each row from Step 3:

```
NB_FUTURE_STEP = 10

inject obs into env  (env.obs = obs, or equivalent state injection)
apply done_set to env  (one env step, opponent action = empty)
for _ in range(NB_FUTURE_STEP):
    step env with opponent action = empty
    record full planet snapshot

emit row + list of NB_FUTURE_STEP planet snapshots
```

**Env state injection** needs to be explored experimentally — likely `env.state = obs` or equivalent internal attribute. The opponent always uses an empty action during simulation.

**To change NB_FUTURE_STEP:** re-run Step 4 only; Step 3 output is reusable.

---

## Step 5 — Feature Table (`04-simulate` → pandas DataFrame)

### Planet ordering

| Slot | Content |
|------|---------|
| 0–39 | Non-comet planets, ordered by planet id |
| 40–43 | Comets, sorted by id: lowest id → slot 40, highest → slot 43 |

Top-left quadrant planets occupy slots 0–10 (11 planets).

### Version 1 — Id-based (2200 features)

```
44 planets × 10 steps × {owner, ships, x, y, production}
= 44 × 10 × 5 = 2200 features
```

All positional info stored per planet per step.

### Version 2 — Cadran-based (1540 features)

```
44 planets × 10 steps × {owner, ships, production}  =  1320 features  (state)
11 planets × 10 steps × {x, y}                       =   220 features  (position)
```

Position stored only for the 11 top-left quadrant planets (slots 0–10). All other planet positions are derivable by board symmetry. This is the already-optimized form; the symmetry reduction is applied immediately rather than as a future step.

### Labels

| Column | Type | Values |
|--------|------|--------|
| `from_planet_id` | int | 0–43 = planet slot, 44 = stop |
| `to_planet_id` | int or NaN | 0–43, NaN when stop |
| `nb_ships` | float or NaN | ship count, NaN when stop |

---

## ML Models

Both options are to be implemented and compared.

### Option A — XGBoost (three models)

**Model 1 — `from`:** 45-class classifier
- Input: full feature vector (Version 1 or 2)
- Classes: 0–43 = planet slot, 44 = stop
- Trained on all rows

**Model 2 — `to`:** 44-class classifier
- Input: full feature vector
- Trained on non-stop rows only (`from_planet_id != 44`)

**Model 3 — `nb_ships`:** regressor
- Input: full feature vector + one-hot(from_planet_id, 44-dim) + one-hot(to_planet_id, 44-dim)
- Trained on non-stop rows only

### Option B — MLP multi-task (one model)

**Architecture:**
```
input: flattened feature vector
→ BatchNorm
→ Dense(512, ReLU)
→ Dense(256, ReLU)
→ [from head: Linear(45) + softmax]
→ [to head:   Linear(44) + softmax]
→ [ships head: Linear(1)]
```

**Loss masking:**
- Stop rows: only `from` head loss is computed
- Non-stop rows: all three heads contribute to loss

**Framework:** PyTorch (straightforward loss masking with masks).

---

## Future Work (out of scope)

- **Per-planet scoring (Option C):** attention-based model that scores each planet independently; better board context capture, more complex training
- `is_orbiting` / `is_comet` as additional features in the table
- Symmetry verification: confirm the 11 top-left quadrant planet positions are sufficient to reconstruct all 44

---

## File Layout

```
11-download_logs/
  00-raw/            existing
  01-winner/         existing (step 1)
  02-augment/        existing (step 2)
  03-combinations/   new (step 3)
  04-simulate/       new (step 4)
```

Step 5 produces a parquet/CSV file; path TBD.
ML model lives in a separate `.py` file using the step 5 table as input.
