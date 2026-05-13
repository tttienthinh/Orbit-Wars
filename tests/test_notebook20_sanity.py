"""
Local sanity check for 20-Train_ML_Kaggle.ipynb.
Verifies the inlined OrbitMLP and training loop work on existing 794-row dataset.
Run from project root: python tests/test_notebook20_sanity.py
"""
import sys
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from pipeline.features import build_feature_matrix

# ── Inlined OrbitMLP (same as notebook Cell 2) ───────────────────────────────
class OrbitMLP(nn.Module):
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

# ── Load existing 794-row dataset ────────────────────────────────────────────
simulate_dir = Path(__file__).parent.parent / "11-download_logs" / "04-simulate"
X, y_from, y_to, y_ships = build_feature_matrix(simulate_dir)
print(f"Loaded: X={X.shape}")  # expect (794, 2200) or similar

# ── Replicate Cell 3 data prep ───────────────────────────────────────────────
# Filter out rows with invalid slot mappings (-1) that arise from unknown planet IDs
# in local simulate data. Kaggle .npy files may also contain -1; drop them here.
valid = (y_from >= 0) & (np.isnan(y_to) | (y_to >= 0))
X, y_from, y_to, y_ships = X[valid], y_from[valid], y_to[valid], y_ships[valid]
print(f"After filtering invalid slots: {len(X)} rows")

mask_ns = (y_from != 44) & (~np.isnan(y_to))
y_to    = np.nan_to_num(y_to,    nan=0.0).astype(np.int64)
y_ships = np.nan_to_num(y_ships, nan=0.0)

X_t  = torch.tensor(X,                       dtype=torch.float32)
yf_t = torch.tensor(y_from.astype(np.int64), dtype=torch.long)
yt_t = torch.tensor(y_to,                    dtype=torch.long)
ys_t = torch.tensor(y_ships.astype(np.float32), dtype=torch.float32)
mk_t = torch.tensor(mask_ns,                 dtype=torch.bool)

g = torch.Generator().manual_seed(42)
idx = torch.randperm(len(X_t), generator=g)
n_val = int(0.2 * len(X_t))
tr_idx, va_idx = idx[n_val:], idx[:n_val]

tr_ds = TensorDataset(X_t[tr_idx], yf_t[tr_idx], yt_t[tr_idx], ys_t[tr_idx], mk_t[tr_idx])
tr_loader = DataLoader(tr_ds, batch_size=64, shuffle=True)

# ── Train 3 epochs, verify loss decreases ────────────────────────────────────
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
    print(f"Epoch {epoch+1}: loss={epoch_loss:.4f}")

assert losses[-1] < losses[0], f"Loss did not decrease: {losses}"
print("Loss decreased over 3 epochs — training loop OK")

# ── Smoke test checkpoint save/load ──────────────────────────────────────────
import tempfile
with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
    tmp_path = f.name
torch.save({"state_dict": model.state_dict(), "input_dim": X.shape[1]}, tmp_path)
ckpt = torch.load(tmp_path, map_location="cpu", weights_only=False)
m2 = OrbitMLP(ckpt["input_dim"])
m2.load_state_dict(ckpt["state_dict"])
m2.eval()
lf, lt, ps = m2(torch.randn(2, X.shape[1]))
assert lf.shape == (2, 45) and lt.shape == (2, 44) and ps.shape == (2,)
os.unlink(tmp_path)
print("Checkpoint save/load smoke test passed.")

print("\nAll sanity checks passed.")
