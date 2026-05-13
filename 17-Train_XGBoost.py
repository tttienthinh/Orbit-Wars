import numpy as np
import mlflow
import mlflow.xgboost
import xgboost as xgb
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from pipeline.features import build_feature_matrix

SIMULATE_DIR    = Path("11-download_logs/04-simulate")
MODEL_DIR       = Path("models/xgboost")
FEATURE_VERSION = 1   # change to 2 for cadran-based features
RANDOM_STATE    = 42
XGB_PARAMS = dict(n_estimators=300, max_depth=6, learning_rate=0.1, random_state=RANDOM_STATE)


def remap_labels(y):
    """
    Remap arbitrary integer labels to a contiguous 0-based range.
    Returns (y_remapped, classes_array) where classes_array[i] = original label i.
    """
    classes = np.unique(y)
    label_map = {c: i for i, c in enumerate(classes)}
    y_remapped = np.array([label_map[v] for v in y], dtype=np.int64)
    return y_remapped, classes


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    X, y_from, y_to, y_ships = build_feature_matrix(SIMULATE_DIR, version=FEATURE_VERSION)

    # Valid rows: y_from is a known slot (0-43) or stop (44); exclude -1 (unknown)
    mask_valid = y_from != -1
    X       = X[mask_valid]
    y_from  = y_from[mask_valid]
    y_to    = y_to[mask_valid]
    y_ships = y_ships[mask_valid]

    mask_ns = y_from != 44  # non-stop rows (no -1 left at this point)

    mlflow.set_experiment("orbit-wars-xgboost")
    with mlflow.start_run():
        mlflow.log_params({
            "feature_version": FEATURE_VERSION,
            "n_samples":       int(len(X)),
            "n_stop_rows":     int((~mask_ns).sum()),
            "n_invalid_rows":  int((~mask_valid).sum()),
            **{f"xgb_{k}": v for k, v in XGB_PARAMS.items()},
        })

        # ── Model 1: from (45-class, all rows) ──────────────────────────────
        # Remap labels to contiguous range (not all slots appear in data)
        y_from_int = y_from.astype(int)
        y_from_remap, from_classes = remap_labels(y_from_int)
        n_from_classes = len(from_classes)
        print(f"from: {n_from_classes} classes observed (out of 45)")

        Xtr, Xva, ytr, yva = train_test_split(X, y_from_remap, test_size=0.2,
                                               random_state=RANDOM_STATE)
        m_from = xgb.XGBClassifier(num_class=n_from_classes, objective="multi:softmax",
                                    eval_metric="mlogloss", **XGB_PARAMS)
        m_from.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
        acc_from = accuracy_score(yva, m_from.predict(Xva))
        mlflow.log_metric("from_val_accuracy", acc_from)
        mlflow.log_param("from_classes", list(map(int, from_classes)))
        mlflow.xgboost.log_model(m_from, "model_from")
        m_from.save_model(MODEL_DIR / "from.ubj")
        # Save label mapping for inference
        np.save(MODEL_DIR / "from_classes.npy", from_classes)
        print(f"from  val accuracy : {acc_from:.3f}")

        # ── Model 2: to (44-class, non-stop rows, valid to-slot) ────────────
        X_ns    = X[mask_ns]
        yto_ns  = y_to[mask_ns].astype(int)
        # Also filter out -1 in y_to (unknown destination)
        mask_valid_to = yto_ns != -1
        X_ns    = X_ns[mask_valid_to]
        yto_ns  = yto_ns[mask_valid_to]
        yfrom_ns_raw = y_from[mask_ns][mask_valid_to].astype(int)
        ysh_ns_raw   = y_ships[mask_ns][mask_valid_to]

        yto_remap, to_classes = remap_labels(yto_ns)
        n_to_classes = len(to_classes)
        print(f"to:   {n_to_classes} classes observed (out of 44)")

        Xtr2, Xva2, ytr2, yva2 = train_test_split(X_ns, yto_remap, test_size=0.2,
                                                    random_state=RANDOM_STATE)
        m_to = xgb.XGBClassifier(num_class=n_to_classes, objective="multi:softmax",
                                  eval_metric="mlogloss", **XGB_PARAMS)
        m_to.fit(Xtr2, ytr2, eval_set=[(Xva2, yva2)], verbose=False)
        acc_to = accuracy_score(yva2, m_to.predict(Xva2))
        mlflow.log_metric("to_val_accuracy", acc_to)
        mlflow.log_param("to_classes", list(map(int, to_classes)))
        mlflow.xgboost.log_model(m_to, "model_to")
        m_to.save_model(MODEL_DIR / "to.ubj")
        np.save(MODEL_DIR / "to_classes.npy", to_classes)
        print(f"to    val accuracy : {acc_to:.3f}")

        # ── Model 3: ships (regressor, conditioned on from + to) ────────────
        # Remap from slots for one-hot encoding (clip to 0-43 range)
        from_oh = np.eye(44, dtype=np.float32)[np.clip(yfrom_ns_raw, 0, 43)]
        to_oh   = np.eye(44, dtype=np.float32)[np.clip(yto_ns,       0, 43)]
        X_ships = np.concatenate([X_ns, from_oh, to_oh], axis=1)
        ysh_ns  = ysh_ns_raw
        Xtr3, Xva3, ytr3, yva3 = train_test_split(X_ships, ysh_ns, test_size=0.2,
                                                    random_state=RANDOM_STATE)
        m_ships = xgb.XGBRegressor(objective="reg:squarederror", **XGB_PARAMS)
        m_ships.fit(Xtr3, ytr3, eval_set=[(Xva3, yva3)], verbose=False)
        mae = float(np.abs(yva3 - m_ships.predict(Xva3)).mean())
        mlflow.log_metric("ships_val_mae", mae)
        mlflow.xgboost.log_model(m_ships, "model_ships")
        m_ships.save_model(MODEL_DIR / "ships.ubj")
        print(f"ships val MAE      : {mae:.1f}")


if __name__ == "__main__":
    main()
