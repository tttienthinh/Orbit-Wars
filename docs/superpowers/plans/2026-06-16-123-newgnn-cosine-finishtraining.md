# 123-NewGNN_Cosine_finishTraining Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Modify `123-NewGNN_Cosine_finishTraining.py` to resume from `117-NewGNN/model_epoch82.pt` and continue training for 50 epochs with 16 episodes/epoch instead of 8.

**Architecture:** Five targeted constant/code edits to the copied file — new output dir, resume checkpoint constant, updated epoch/episode counts, `load_state_dict` call in `main()`, and updated MLflow experiment name. No structural changes.

**Tech Stack:** PyTorch, torch-geometric, polars, mlflow

---

### Task 1: Update constants and add RESUME_CKPT

**Files:**
- Modify: `123-NewGNN_Cosine_finishTraining.py:19-33`

- [ ] **Step 1: Change OUT_DIR, add RESUME_CKPT, update N_EPOCHS and EPISODES_PER_EPOCH**

Replace the four constant lines (lines 19, 21, 33) and add `RESUME_CKPT` after `OUT_DIR`:

```python
PRECOMPUTE_DIR = Path("114-precompute")
OUT_DIR        = Path("123-NewGNN_Cosine_finishTraining")
RESUME_CKPT    = Path("117-NewGNN/model_epoch82.pt")
NB_STEPS_SIM   = 20
N_EPOCHS       = 50
HIDDEN_DIM     = 64
NUM_LAYERS     = 3
LR             = 1e-3
WEIGHT_DECAY   = 1e-4
```

And further down:

```python
EPISODES_PER_EPOCH = 16   # doubled from 119; pairs sampled per epoch from the pool of 534×4=2136
```

- [ ] **Step 2: Verify the diff looks correct**

```bash
grep -n "OUT_DIR\|RESUME_CKPT\|N_EPOCHS\|EPISODES_PER_EPOCH" 123-NewGNN_Cosine_finishTraining.py
```

Expected output:
```
19:OUT_DIR        = Path("123-NewGNN_Cosine_finishTraining")
20:RESUME_CKPT    = Path("117-NewGNN/model_epoch82.pt")
22:N_EPOCHS       = 50
33:EPISODES_PER_EPOCH = 16   # doubled from 119; ...
```

---

### Task 2: Load checkpoint in main()

**Files:**
- Modify: `123-NewGNN_Cosine_finishTraining.py` — `main()` function, after model construction

- [ ] **Step 1: Add load_state_dict call**

Find this line in `main()`:
```python
    model     = OrbitGNN(hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS).to(DEVICE)
```

Add immediately after it:
```python
    model.load_state_dict(torch.load(RESUME_CKPT, map_location=DEVICE))
    _log(f"Resumed from {RESUME_CKPT}", log_fh)
```

- [ ] **Step 2: Verify checkpoint file exists**

```bash
python -c "from pathlib import Path; p = Path('117-NewGNN/model_epoch82.pt'); print('EXISTS' if p.exists() else 'MISSING')"
```

Expected: `EXISTS`

---

### Task 3: Update MLflow experiment name

**Files:**
- Modify: `123-NewGNN_Cosine_finishTraining.py` — `main()` function

- [ ] **Step 1: Change experiment name**

Find:
```python
    mlflow.set_experiment("119-NewGNN_Cosine")
```

Replace with:
```python
    mlflow.set_experiment("123-NewGNN_Cosine_finishTraining")
```

- [ ] **Step 2: Smoke-test the script parses and loads the checkpoint**

```bash
python -c "
import torch
from pathlib import Path
import sys
sys.path.insert(0, '.')
from importlib import util
spec = util.spec_from_file_location('m', '123-NewGNN_Cosine_finishTraining.py')
m = util.module_from_spec(spec)
spec.loader.exec_module(m)
model = m.OrbitGNN(hidden_dim=64, num_layers=3)
model.load_state_dict(torch.load('117-NewGNN/model_epoch82.pt', map_location='cpu'))
print('Checkpoint loaded OK, param count:', sum(p.numel() for p in model.parameters()))
"
```

Expected output: `Checkpoint loaded OK, param count: <number>`

- [ ] **Step 3: Commit**

```bash
git add 123-NewGNN_Cosine_finishTraining.py
git commit -m "feat: 123 — resume from 117 epoch82, 50 epochs, 16 eps/epoch"
```
