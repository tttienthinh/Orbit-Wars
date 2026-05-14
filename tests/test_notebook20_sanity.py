"""
Local sanity check for 20-Train_ML_Kaggle.ipynb.
Verifies OrbitMLP (imported from pipeline.model) and training loop work on
existing 794-row dataset.
Run from project root: pytest tests/test_notebook20_sanity.py
"""
import sys
import os
import tempfile
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from pipeline.features import build_feature_matrix
from pipeline.model import OrbitMLP

SIMULATE_DIR = Path(__file__).parent.parent / "11-download_logs" / "04-simulate"

pytestmark = pytest.mark.skipif(
    not SIMULATE_DIR.exists(),
    reason="Local simulate data not available",
)


def _load_dataset():
    X, y_from, y_to, y_ships = build_feature_matrix(SIMULATE_DIR)
    valid = (y_from >= 0) & (np.isnan(y_to) | (y_to >= 0))
    X, y_from, y_to, y_ships = X[valid], y_from[valid], y_to[valid], y_ships[valid]
    return X, y_from, y_to, y_ships


def test_training_loop_loss_decreases():
    """Train OrbitMLP for 3 epochs and verify loss decreases."""
    X, y_from, y_to, y_ships = _load_dataset()

    mask_ns = (y_from != 44) & (~np.isnan(y_to))
    y_to    = np.nan_to_num(y_to,    nan=0.0).astype(np.int64)
    y_ships = np.nan_to_num(y_ships, nan=0.0)

    X_t  = torch.tensor(X,                          dtype=torch.float32)
    yf_t = torch.tensor(y_from.astype(np.int64),    dtype=torch.long)
    yt_t = torch.tensor(y_to,                       dtype=torch.long)
    ys_t = torch.tensor(y_ships.astype(np.float32), dtype=torch.float32)
    mk_t = torch.tensor(mask_ns,                    dtype=torch.bool)

    g = torch.Generator().manual_seed(42)
    idx = torch.randperm(len(X_t), generator=g)
    n_val = int(0.2 * len(X_t))
    tr_idx = idx[n_val:]

    tr_ds = TensorDataset(X_t[tr_idx], yf_t[tr_idx], yt_t[tr_idx], ys_t[tr_idx], mk_t[tr_idx])
    tr_loader = DataLoader(tr_ds, batch_size=64, shuffle=True)

    model     = OrbitMLP(X.shape[1])
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    ce        = nn.CrossEntropyLoss()

    losses = []
    for epoch in range(3):
        model.train()
        epoch_loss = 0.0
        for xb, yf, yt, ys, mk in tr_loader:
            lf, lt, ps = model(xb)
            loss = ce(lf, yf)
            if mk.any():
                loss = loss + ce(lt[mk], yt[mk])
                loss = loss + 0.01 * ((ps[mk] - ys[mk]) ** 2).mean()
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            epoch_loss += loss.item()
        losses.append(epoch_loss)

    assert losses[-1] < losses[0], f"Loss did not decrease: {losses}"


def test_checkpoint_save_load():
    """Verify checkpoint save/load round-trip produces correct output shapes."""
    X, _, _, _ = _load_dataset()
    model = OrbitMLP(X.shape[1])

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        tmp_path = f.name
    try:
        torch.save({"state_dict": model.state_dict(), "input_dim": X.shape[1]}, tmp_path)
        ckpt = torch.load(tmp_path, map_location="cpu", weights_only=False)
        m2 = OrbitMLP(ckpt["input_dim"])
        m2.load_state_dict(ckpt["state_dict"])
        m2.eval()
        lf, lt, ps = m2(torch.randn(2, X.shape[1]))
        assert lf.shape == (2, 45) and lt.shape == (2, 44) and ps.shape == (2,)
    finally:
        os.unlink(tmp_path)


if __name__ == "__main__":
    test_training_loop_loss_decreases()
    print("Loss decreased over 3 epochs — training loop OK")
    test_checkpoint_save_load()
    print("Checkpoint save/load smoke test passed.")
    print("\nAll sanity checks passed.")
