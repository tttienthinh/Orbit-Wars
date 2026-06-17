# 119-NewGNN-Cosine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce `119-NewGNN_Cosine.py` — a retrained GNN with AdamW+CosineAnnealingLR, hard train/test split, 4× coordinate-rotation augmentation, and per-epoch train+test metrics.

**Architecture:** Single self-contained training script derived from `117-NewGNN.py`. OrbitGNN model and graph-building code are unchanged. A new `_apply_transform()` applies board rotations in-memory to `df_s` before graph construction. A new `evaluate_episode()` mirrors `train_episode()` but without gradients. A new `_compute_metrics()` computes AUC/accuracy/TP/TN from accumulated scores+labels. `main()` is fully replaced.

**Tech Stack:** PyTorch, torch-geometric, Polars, scikit-learn (roc_auc_score), MLflow, Python 3.10+

---

## File Map

| Action | Path | Responsibility |
|--------|------|----------------|
| Create | `119-NewGNN_Cosine.py` | Full training script |
| Create | `119-NewGNN_Cosine/` | Output dir (checkpoints, logs) — created at runtime |

---

### Task 1: Copy base file and update header constants

**Files:**
- Create: `119-NewGNN_Cosine.py` (copy of `117-NewGNN.py`)

- [ ] **Step 1: Copy the file**

  ```powershell
  Copy-Item "117-NewGNN.py" "119-NewGNN_Cosine.py"
  ```

- [ ] **Step 2: Replace the module docstring and all header constants**

  Open `119-NewGNN_Cosine.py`. Replace everything from line 1 through the end of the constants block (lines 1–25) with:

  ```python
  """119-NewGNN_Cosine — Two-tower GNN with AdamW+CosineAnnealingLR, augmentation, train/test split."""
  import math
  import random
  from pathlib import Path

  import mlflow
  import mlflow.pytorch
  import numpy as np
  import polars as pl
  import torch
  import torch.nn as nn
  import torch.nn.functional as F
  from sklearn.metrics import roc_auc_score
  from torch_geometric.data import HeteroData
  from torch_geometric.nn import GATConv, HeteroConv, SAGEConv
  from torch_geometric.transforms import ToUndirected

  PRECOMPUTE_DIR = Path("114-precompute")
  OUT_DIR        = Path("119-NewGNN_Cosine")
  NB_STEPS_SIM   = 20
  N_EPOCHS       = 30
  HIDDEN_DIM     = 64
  NUM_LAYERS     = 3
  LR             = 1e-3
  WEIGHT_DECAY   = 1e-4

  TEST_EPISODE_IDS = {
      78867640, 78899068, 78982947, 79033183,
      79126912, 79175592, 79228392, 79320069,
  }

  TRANSFORMS = ["identity", "rot90", "rot180", "rot270"]
  ```

- [ ] **Step 3: Verify the file parses**

  ```powershell
  python -c "import ast; ast.parse(open('119-NewGNN_Cosine.py').read()); print('OK')"
  ```

  Expected output: `OK`

- [ ] **Step 4: Commit**

  ```powershell
  git add 119-NewGNN_Cosine.py
  git commit -m "feat: scaffold 119-NewGNN_Cosine from 117 with updated constants"
  ```

---

### Task 2: Add `_apply_transform()` and update `train_episode()` signature

**Files:**
- Modify: `119-NewGNN_Cosine.py`

- [ ] **Step 1: Insert `_apply_transform` immediately before `train_episode`**

  Find the line `def train_episode(` and insert the following block directly above it:

  ```python
  def _apply_transform(df_s: pl.DataFrame, transform: str) -> pl.DataFrame:
      """Apply a board rotation to the x/y columns of df_s. Sun at (50,50) is preserved."""
      if transform == "identity":
          return df_s
      elif transform == "rot90":
          return df_s.with_columns([
              (100.0 - pl.col("y")).alias("x"),
              pl.col("x").alias("y"),
          ])
      elif transform == "rot180":
          return df_s.with_columns([
              (100.0 - pl.col("x")).alias("x"),
              (100.0 - pl.col("y")).alias("y"),
          ])
      elif transform == "rot270":
          return df_s.with_columns([
              pl.col("y").alias("x"),
              (100.0 - pl.col("x")).alias("y"),
          ])
      else:
          raise ValueError(f"Unknown transform: {transform!r}")


  ```

- [ ] **Step 2: Update `train_episode` signature and add transform call**

  Change the signature from:
  ```python
  def train_episode(
      ep_dir: Path,
      model: OrbitGNN,
  ) -> tuple[float, list[float], list[float]]:
  ```
  to:
  ```python
  def train_episode(
      ep_dir: Path,
      model: OrbitGNN,
      transform: str = "identity",
  ) -> tuple[float, list[float], list[float]]:
  ```

  Then find the three `pl.read_parquet` lines inside `train_episode`:
  ```python
      df_s    = pl.read_parquet(ep_dir / "df_s.parquet")
      reach   = pl.read_parquet(ep_dir / "reach.parquet")
      actions = pl.read_parquet(ep_dir / "actions.parquet")
  ```
  and add one line immediately after them:
  ```python
      df_s = _apply_transform(df_s, transform)
  ```

- [ ] **Step 3: Write a test for `_apply_transform`**

  Add this function anywhere in the test section of the file (near the other `_test_*` functions):

  ```python
  def _test_apply_transform():
      df = pl.DataFrame({"x": [0.0, 100.0, 50.0], "y": [0.0, 100.0, 50.0],
                         "id": [0, 1, 2], "step": [1, 1, 1]})
      # rot90: new_x = 100-y, new_y = x
      r = _apply_transform(df, "rot90")
      assert r["x"].to_list() == [100.0, 0.0, 50.0], r["x"].to_list()
      assert r["y"].to_list() == [0.0, 100.0, 50.0], r["y"].to_list()
      # rot180: new_x = 100-x, new_y = 100-y
      r = _apply_transform(df, "rot180")
      assert r["x"].to_list() == [100.0, 0.0, 50.0], r["x"].to_list()
      assert r["y"].to_list() == [100.0, 0.0, 50.0], r["y"].to_list()
      # rot270: new_x = y, new_y = 100-x
      r = _apply_transform(df, "rot270")
      assert r["x"].to_list() == [0.0, 100.0, 50.0], r["x"].to_list()
      assert r["y"].to_list() == [100.0, 0.0, 50.0], r["y"].to_list()
      print("_test_apply_transform PASSED")
  ```

- [ ] **Step 4: Run the test**

  ```powershell
  python -c "exec(open('119-NewGNN_Cosine.py').read()); _test_apply_transform()"
  ```

  Expected output: `_test_apply_transform PASSED`

- [ ] **Step 5: Commit**

  ```powershell
  git add 119-NewGNN_Cosine.py
  git commit -m "feat: add _apply_transform and wire into train_episode"
  ```

---

### Task 3: Add `evaluate_episode()`

**Files:**
- Modify: `119-NewGNN_Cosine.py`

- [ ] **Step 1: Insert `evaluate_episode` immediately after `train_episode`**

  Find the blank line after the closing of `train_episode` (the line `    return total_loss / max(n_steps, 1), all_scores, all_labels`) and insert:

  ```python

  def evaluate_episode(
      ep_dir: Path,
      model: OrbitGNN,
  ) -> tuple[float, list[float], list[float]]:
      """Evaluate one episode without gradients. Returns (avg_loss, scores, labels)."""
      df_s    = pl.read_parquet(ep_dir / "df_s.parquet")
      reach   = pl.read_parquet(ep_dir / "reach.parquet")
      actions = pl.read_parquet(ep_dir / "actions.parquet")

      reach = (
          reach
          .group_by(["id_src", "step_src", "id", "ships_sent"])
          .agg(pl.all().sort_by("step").first())
      )

      actions_set: set[tuple[int, int, int]] = set(
          zip(
              actions["game_step"].to_list(),
              actions["id_src"].to_list(),
              actions["id"].to_list(),
          )
      )

      all_scores: list[float] = []
      all_labels: list[float] = []
      total_loss = 0.0
      n_steps    = 0

      model.eval()
      with torch.no_grad():
          for t in sorted(reach["step_src"].unique().to_list()):
              reach_t       = reach.filter(pl.col("step_src") == t)
              arrival_steps = reach_t["step"].unique().to_list()
              df_s_t        = df_s.filter(pl.col("step").is_in([t] + arrival_steps))
              df_s_at_t     = df_s.filter(pl.col("step") == t)

              if df_s_t.is_empty() or df_s_at_t.is_empty():
                  continue

              data, planet_idx = build_graph(df_s_t, reach_t)
              src_idx, tgt_idx, labels = build_attack_pairs(df_s_at_t, actions_set, planet_idx, t)

              if len(src_idx) == 0:
                  continue

              h_planet = model.encode(data)
              logits   = model.score_pairs(h_planet, src_idx, tgt_idx)

              n_pos      = labels.sum().item()
              n_neg      = len(labels) - n_pos
              pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32)
              loss       = F.binary_cross_entropy_with_logits(logits, labels, pos_weight=pos_weight)

              total_loss += loss.item()
              n_steps    += 1
              all_scores.extend(torch.sigmoid(logits).tolist())
              all_labels.extend(labels.tolist())

      return total_loss / max(n_steps, 1), all_scores, all_labels
  ```

- [ ] **Step 2: Verify the file parses**

  ```powershell
  python -c "import ast; ast.parse(open('119-NewGNN_Cosine.py').read()); print('OK')"
  ```

  Expected: `OK`

- [ ] **Step 3: Commit**

  ```powershell
  git add 119-NewGNN_Cosine.py
  git commit -m "feat: add evaluate_episode (no-grad eval)"
  ```

---

### Task 4: Add `_compute_metrics()`

**Files:**
- Modify: `119-NewGNN_Cosine.py`

- [ ] **Step 1: Insert `_compute_metrics` immediately before `_log`**

  Find `def _log(` and insert above it:

  ```python
  def _compute_metrics(scores: list[float], labels: list[float]) -> dict:
      """Compute ROC-AUC, accuracy, TP, TN, pos count, neg count at threshold 0.5."""
      n_pos = int(sum(labels))
      n_neg = len(labels) - n_pos

      if len(set(labels)) > 1:
          auc = roc_auc_score(labels, scores)
      else:
          auc = float("nan")

      tp = int(sum(1 for s, l in zip(scores, labels) if s >= 0.5 and l == 1.0))
      tn = int(sum(1 for s, l in zip(scores, labels) if s <  0.5 and l == 0.0))
      acc = (tp + tn) / max(len(labels), 1)

      return {"auc": auc, "acc": acc, "n_pos": n_pos, "n_neg": n_neg, "tp": tp, "tn": tn}


  ```

- [ ] **Step 2: Write and run a quick inline test**

  ```powershell
  python -c "
  exec(open('119-NewGNN_Cosine.py').read())
  scores = [0.9, 0.1, 0.8, 0.2]
  labels = [1.0, 0.0, 1.0, 0.0]
  m = _compute_metrics(scores, labels)
  assert m['n_pos'] == 2 and m['n_neg'] == 2, m
  assert m['tp'] == 2 and m['tn'] == 2, m
  assert m['acc'] == 1.0, m
  print('_compute_metrics OK', m)
  "
  ```

  Expected output starts with: `_compute_metrics OK`

- [ ] **Step 3: Commit**

  ```powershell
  git add 119-NewGNN_Cosine.py
  git commit -m "feat: add _compute_metrics helper"
  ```

---

### Task 5: Replace `main()`

**Files:**
- Modify: `119-NewGNN_Cosine.py`

- [ ] **Step 1: Delete the old `main()` function entirely**

  In `119-NewGNN_Cosine.py`, locate `def main() -> None:` (currently around line 364) and delete everything from that line through the closing `log_fh.close()` line (just before the first `_test_build_graph` function or `if __name__` block).

- [ ] **Step 2: Insert the new `main()` in its place**

  ```python
  def main() -> None:
      OUT_DIR.mkdir(exist_ok=True)
      log_fh = open(OUT_DIR / "metrics.log", "a", encoding="utf-8")

      ep_dirs    = sorted([d for d in PRECOMPUTE_DIR.iterdir() if d.is_dir()])
      train_dirs = [d for d in ep_dirs if int(d.name) not in TEST_EPISODE_IDS]
      test_dirs  = [d for d in ep_dirs if int(d.name) in     TEST_EPISODE_IDS]

      n_pairs = len(train_dirs) * len(TRANSFORMS)
      _log(f"Episodes: {len(ep_dirs)} total | train={len(train_dirs)} test={len(test_dirs)}", log_fh)
      _log(f"Pairs/epoch: {len(train_dirs)} × {len(TRANSFORMS)} = {n_pairs} (all, shuffled)", log_fh)

      all_pairs = [(ep_dir, t) for ep_dir in train_dirs for t in TRANSFORMS]

      model     = OrbitGNN(hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS)
      optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
      scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=N_EPOCHS, eta_min=1e-6)

      best_test_auc = float("-inf")

      mlflow.set_experiment("119-NewGNN_Cosine")
      with mlflow.start_run():
          mlflow.log_params({
              "hidden_dim":      HIDDEN_DIM,
              "num_layers":      NUM_LAYERS,
              "lr":              LR,
              "weight_decay":    WEIGHT_DECAY,
              "n_epochs":        N_EPOCHS,
              "train_episodes":  len(train_dirs),
              "test_episodes":   len(test_dirs),
              "transforms":      len(TRANSFORMS),
              "pairs_per_epoch": n_pairs,
          })

          for epoch in range(1, N_EPOCHS + 1):
              # ── Training ────────────────────────────────────────────────────
              epoch_pairs = all_pairs.copy()
              random.shuffle(epoch_pairs)

              tr_scores: list[float] = []
              tr_labels: list[float] = []
              tr_loss_sum = 0.0
              tr_steps    = 0

              model.train()
              for ep_dir, transform in epoch_pairs:
                  try:
                      optimizer.zero_grad()
                      loss, scores, labels = train_episode(ep_dir, model, transform)
                      optimizer.step()
                      tr_loss_sum += loss
                      tr_steps    += 1
                      tr_scores.extend(scores)
                      tr_labels.extend(labels)
                  except Exception as e:
                      _log(f"  SKIP train {ep_dir.name}/{transform}: {e}", log_fh)

              scheduler.step()
              current_lr = scheduler.get_last_lr()[0]

              # ── Test evaluation ─────────────────────────────────────────────
              te_scores: list[float] = []
              te_labels: list[float] = []
              te_loss_sum = 0.0
              te_steps    = 0

              for ep_dir in test_dirs:
                  try:
                      loss, scores, labels = evaluate_episode(ep_dir, model)
                      te_loss_sum += loss
                      te_steps    += 1
                      te_scores.extend(scores)
                      te_labels.extend(labels)
                  except Exception as e:
                      _log(f"  SKIP test  {ep_dir.name}: {e}", log_fh)

              # ── Metrics ─────────────────────────────────────────────────────
              avg_tr = tr_loss_sum / max(tr_steps, 1)
              avg_te = te_loss_sum / max(te_steps, 1)
              tr_m   = _compute_metrics(tr_scores, tr_labels)
              te_m   = _compute_metrics(te_scores, te_labels)

              def _fmt(m: dict, loss: float) -> str:
                  auc = f"{m['auc']:.4f}" if not math.isnan(m['auc']) else "  nan"
                  return (f"loss={loss:.4f}  auc={auc}  acc={m['acc']:.4f}"
                          f"  pos={m['n_pos']}  neg={m['n_neg']}  tp={m['tp']}  tn={m['tn']}")

              _log(f"Epoch {epoch:3d}  lr={current_lr:.2e}", log_fh)
              _log(f"  [train] {_fmt(tr_m, avg_tr)}", log_fh)
              _log(f"  [test]  {_fmt(te_m, avg_te)}", log_fh)

              mlflow.log_metrics({
                  "train_loss": avg_tr,
                  "train_auc":  tr_m["auc"]  if not math.isnan(tr_m["auc"])  else 0.0,
                  "train_acc":  tr_m["acc"],
                  "train_tp":   tr_m["tp"],
                  "train_tn":   tr_m["tn"],
                  "test_loss":  avg_te,
                  "test_auc":   te_m["auc"]  if not math.isnan(te_m["auc"])  else 0.0,
                  "test_acc":   te_m["acc"],
                  "test_tp":    te_m["tp"],
                  "test_tn":    te_m["tn"],
                  "lr":         current_lr,
              }, step=epoch)

              # ── Checkpoints ─────────────────────────────────────────────────
              ckpt = OUT_DIR / f"model_epoch{epoch}.pt"
              torch.save(model.state_dict(), ckpt)
              mlflow.log_artifact(str(ckpt), artifact_path="checkpoints")
              _log(f"  -> saved {ckpt}", log_fh)

              if not math.isnan(te_m["auc"]) and te_m["auc"] > best_test_auc:
                  best_test_auc = te_m["auc"]
                  best_path = OUT_DIR / "best_model.pt"
                  torch.save(model.state_dict(), best_path)
                  _log(f"  -> NEW BEST test_auc={best_test_auc:.4f}  saved {best_path}", log_fh)

      _log(f"\nDone. Best test AUC: {best_test_auc:.4f}  Models in {OUT_DIR}/", log_fh)
      log_fh.close()
  ```

- [ ] **Step 3: Verify the file parses**

  ```powershell
  python -c "import ast; ast.parse(open('119-NewGNN_Cosine.py').read()); print('OK')"
  ```

  Expected: `OK`

- [ ] **Step 4: Commit**

  ```powershell
  git add 119-NewGNN_Cosine.py
  git commit -m "feat: replace main() with AdamW+cosine loop, augmentation, train/test metrics"
  ```

---

### Task 6: Update `__main__` block and run smoke test

**Files:**
- Modify: `119-NewGNN_Cosine.py`

- [ ] **Step 1: Update `__main__` to call `_test_apply_transform`**

  Find:
  ```python
  if __name__ == "__main__":
      _test_build_graph()
      _test_orbit_gnn()
      _test_build_attack_pairs()
      _test_train_episode()
      print("\nAll tests passed. Starting training...\n")
      main()
  ```

  Replace with:
  ```python
  if __name__ == "__main__":
      _test_build_graph()
      _test_orbit_gnn()
      _test_build_attack_pairs()
      _test_apply_transform()
      _test_train_episode()
      print("\nAll tests passed. Starting training...\n")
      main()
  ```

- [ ] **Step 2: Run the full test suite (without `main()`)**

  ```powershell
  python -c "
  exec(open('119-NewGNN_Cosine.py').read())
  _test_build_graph()
  _test_orbit_gnn()
  _test_build_attack_pairs()
  _test_apply_transform()
  _test_train_episode()
  print('All tests passed.')
  "
  ```

  Expected output:
  ```
  _test_build_graph PASSED
  _test_orbit_gnn PASSED
  _test_build_attack_pairs PASSED
  _test_apply_transform PASSED
  _test_train_episode PASSED  loss=...  pairs=...  positives=...
  All tests passed.
  ```

- [ ] **Step 3: Commit**

  ```powershell
  git add 119-NewGNN_Cosine.py
  git commit -m "feat: add _test_apply_transform to __main__ test suite"
  ```

---

## Self-Review

**Spec coverage:**
| Spec requirement | Task |
|-----------------|------|
| AdamW + CosineAnnealingLR (T_max=30, 1e-3→1e-6) | Task 5 `main()` |
| weight_decay=1e-4 | Task 5 `main()` |
| 30 epochs | Task 1 `N_EPOCHS=30` |
| Hard test split (8 episodes) | Task 1 `TEST_EPISODE_IDS` |
| 4× augmentation transforms | Task 2 `_apply_transform` |
| All 2136 pairs shuffled and trained each epoch | Task 5 `main()` |
| Train evaluation every epoch | Task 5 `main()` |
| Test evaluation every epoch | Task 5 `main()` |
| ROC AUC, pos/neg, acc, TP, TN | Task 4 `_compute_metrics` |
| model_epochN.pt per epoch | Task 5 `main()` |
| best_model.pt on test AUC improve | Task 5 `main()` |
| MLflow logging | Task 5 `main()` |
| metrics.log file | Task 5 `main()` |
| OUT_DIR = 119-NewGNN_Cosine/ | Task 1 |

**Placeholder scan:** No TBD/TODO present.

**Type consistency:** `_compute_metrics(scores: list[float], labels: list[float]) -> dict` — used consistently in Task 4 and Task 5. `train_episode(ep_dir, model, transform)` signature defined in Task 2 and called in Task 5.
