# 123-NewGNN_Cosine_finishTraining — Design Spec

**Date:** 2026-06-16

## Goal

Continue training the best-performing GNN checkpoint (`117-NewGNN/model_epoch82.pt`) with a doubled episode sample rate to push test AUC higher.

## Changes from 119-NewGNN_Cosine.py

| Setting | Old (119) | New (123) |
|---------|-----------|-----------|
| `OUT_DIR` | `119-NewGNN_Cosine` | `123-NewGNN_Cosine_finishTraining` |
| `RESUME_CKPT` | _(none)_ | `117-NewGNN/model_epoch82.pt` |
| `N_EPOCHS` | 30 | 50 |
| `EPISODES_PER_EPOCH` | 8 | 16 |
| MLflow experiment | `119-NewGNN_Cosine` | `123-NewGNN_Cosine_finishTraining` |

## Architecture

Unchanged: `OrbitGNN(hidden_dim=64, num_layers=3)` — compatible with the 117 checkpoint.

## Training Setup

- **Optimizer:** AdamW, LR=1e-3, weight_decay=1e-4 (fresh state)
- **Scheduler:** CosineAnnealingLR(T_max=50, eta_min=1e-6) — full cosine restart from 1e-3
- **Checkpoint loading:** `model.load_state_dict(torch.load(RESUME_CKPT, map_location=DEVICE))` immediately after model construction, before optimizer creation

## Implementation

One new constant and one `load_state_dict` call in `main()`. All other logic (graph building, augmentation, train/test split, evaluation, MLflow logging, per-epoch checkpointing) is identical to 119.
