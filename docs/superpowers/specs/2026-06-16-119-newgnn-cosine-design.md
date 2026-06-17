# 119-NewGNN-Cosine — Design Spec

## Goal

Retrain the GNN from `117-NewGNN.py` with improved hyperparameters (from autoresearch README), a
hard train/test split, coordinate-rotation augmentation, and richer per-epoch metrics.

---

## Files

| Path | Purpose |
|------|---------|
| `119-NewGNN_Cosine.py` | Training script (copy of 117 with changes below) |
| `119-NewGNN_Cosine/` | Output: checkpoints, logs, MLflow artifacts |
| `119-NewGNN_Cosine/model_epochN.pt` | Checkpoint saved at end of every epoch (never overwritten) |
| `119-NewGNN_Cosine/best_model.pt` | Best checkpoint by test AUC (overwrites on improve) |
| `119-NewGNN_Cosine/metrics.log` | Per-epoch train + test metric lines |

---

## Hyperparameters (from README)

```
optimizer:    AdamW, weight_decay=1e-4
lr schedule:  CosineAnnealingLR(T_max=30), 1e-3 → ~1e-6
epochs:       30
hidden_dim:   64   (unchanged)
num_layers:   3    (unchanged)
```

---

## Train / Test Split

Hard-coded. Never augment test episodes.

```python
TEST_EPISODE_IDS = {
    78867640, 78899068, 78982947, 79033183,
    79126912, 79175592, 79228392, 79320069,
}
```

Training = all remaining episodes in `114-precompute/`.

---

## Data Augmentation

Four coordinate transforms applied to `df_s["x"]` and `df_s["y"]` (the 100×100 board). Each
preserves the sun at (50, 50).

| Name | new_x | new_y |
|------|-------|-------|
| `identity` | x | y |
| `rot90` | 100 − y | x |
| `rot180` | 100 − x | 100 − y |
| `rot270` | y | 100 − x |

Only `df_s` x/y columns are transformed. Reach edges, ship counts, and planet IDs are unchanged.

---

## Training Loop

```
all_pairs = [(ep_dir, transform) for ep_dir in train_dirs
                                 for transform in TRANSFORMS]
# 534 × 4 = 2136 pairs total

for epoch in 1..30:
    shuffle(all_pairs)           # new order each epoch, train on all 2136
    for (ep_dir, transform) in all_pairs:
        optimizer.zero_grad()
        loss, scores, labels = train_episode(ep_dir, model, transform)
        optimizer.step()
        accumulate epoch metrics
    scheduler.step()             # cosine decay step
    eval test set
    log metrics (train + test)
    save model_epochN.pt
    if test_auc > best_test_auc: save best_model.pt (overwrite)
```

---

## Code Changes to `train_episode`

Add optional `transform: str = "identity"` parameter. Before building graphs, apply:

```python
def _apply_transform(df_s: pl.DataFrame, transform: str) -> pl.DataFrame:
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
```

---

## Evaluation Function

New `evaluate_episode(ep_dir, model)` — identical to `train_episode` but with `model.eval()`,
`torch.no_grad()`, no `.backward()`. Returns `(avg_loss, scores, labels)`.

---

## Metrics (printed every epoch for both train and test)

All metrics computed from accumulated scores/labels across all episodes in the set.

```
Epoch  N  [train] loss=X  auc=X  acc=X  pos=X  neg=X  tp=X  tn=X
           [test]  loss=X  auc=X  acc=X  pos=X  neg=X  tp=X  tn=X
```

- **auc**: `roc_auc_score(labels, scores)`
- **acc**: `(tp + tn) / total` at threshold 0.5
- **tp / tn**: from `confusion_matrix` at threshold 0.5
- **pos / neg**: count of positive and negative labels

---

## Self-Review

- No TBD sections.
- Sun at (50,50) is preserved by all 4 rotations — verified by hand.
- Test episodes are never augmented, never in `train_dirs`.
- Best-checkpoint criterion is test AUC (not train), consistent with README guidance.
- 2136 pairs/epoch × 30 epochs = 64,080 total training iterations.
- Each epoch sees every episode in every augmented orientation, in a new random order.
