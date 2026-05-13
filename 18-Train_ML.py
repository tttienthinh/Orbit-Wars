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
    # mask_ns: rows where y_from is a real non-stop move (not stop=44 and not invalid=-1)
    # and y_to is valid (not NaN and not -1)
    mask_ns  = (y_from != 44) & (y_from != -1) & (~np.isnan(y_to)) & (y_to != -1)
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
    acc_f = acc_t = mae_s = 0.0
    n = n_ns = 0
    with torch.no_grad():
        for xb, yf, yt, ys, mk in loader:
            lf, lt, ps = model(xb)
            # Only count valid y_from labels (not -1)
            valid_f = yf != -1
            if valid_f.any():
                acc_f += (lf[valid_f].argmax(1) == yf[valid_f]).float().mean().item()
                n += 1
            if mk.any():
                acc_t += (lt[mk].argmax(1) == yt[mk]).float().mean().item()
                mae_s += (ps[mk] - ys[mk]).abs().mean().item()
                n_ns  += 1
    return acc_f / max(n, 1), acc_t / max(n_ns, 1), mae_s / max(n_ns, 1)


def main():
    torch.manual_seed(RANDOM_STATE)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    X, y_from, y_to, y_ships = build_feature_matrix(SIMULATE_DIR, version=FEATURE_VERSION)
    tr_loader, va_loader = make_loaders(X, y_from, y_to, y_ships, BATCH_SIZE, RANDOM_STATE)

    model     = OrbitMLP(input_dim=X.shape[1])
    optimizer = optim.Adam(model.parameters(), lr=LR)
    ce_from   = nn.CrossEntropyLoss(ignore_index=-1)
    ce_to     = nn.CrossEntropyLoss(ignore_index=-1)

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
        print(f"Model saved -> {model_path}")


if __name__ == "__main__":
    main()
