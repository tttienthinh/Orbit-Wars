"""118-LGBM — LightGBM baseline for attack prediction: (src, tgt, step) → attack yes/no."""
import math
import random
from pathlib import Path

import mlflow
import numpy as np
import polars as pl
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

PRECOMPUTE_DIR = Path("114-precompute")
OUT_DIR        = Path("118-LGBM")
NB_STEPS_SIM   = 20
TRAIN_RATIO    = 0.8
REACH_POW2     = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
LGBM_PARAMS    = dict(n_estimators=500, learning_rate=0.05, num_leaves=63, n_jobs=-1, verbose=-1)
STEP_STRIDE    = 10   # sample every Nth step_src to keep RAM under ~3 GB

_LOG1024 = math.log(1024)

FEATURE_COLS = [
    # Source snapshot @ t (12)
    "src_x", "src_y", "src_ships",
    "src_is_fix", "src_is_moving", "src_is_comet",
    "src_production",
    "src_owner_n1", "src_owner_0", "src_owner_1", "src_owner_2", "src_owner_3",
    # Target snapshot @ t (12)
    "tgt_x", "tgt_y", "tgt_ships",
    "tgt_is_fix", "tgt_is_moving", "tgt_is_comet",
    "tgt_production",
    "tgt_owner_n1", "tgt_owner_0", "tgt_owner_1", "tgt_owner_2", "tgt_owner_3",
    # Travel time per ship size — (arrival_step − t) / 20, NaN if no edge (11)
    *[f"travel_{s}" for s in REACH_POW2],
    # Target trajectory: ships_log + owner for offsets 0..19 (40)
    *[f"tgt_ships_t{i}" for i in range(NB_STEPS_SIM)],
    *[f"tgt_owner_t{i}" for i in range(NB_STEPS_SIM)],
    # Source trajectory: ships_log for offsets 0..19 (20)
    *[f"src_ships_t{i}" for i in range(NB_STEPS_SIM)],
]  # total: 95


def _snap_features(df_at_t: pl.DataFrame, id_alias: str, prefix: str) -> pl.DataFrame:
    """Return 12 snapshot features for all planets at step t, keyed by id_alias."""
    return (
        df_at_t
        .with_columns([
            (pl.col("x") / 100.0).alias(f"{prefix}x"),
            (pl.col("y") / 100.0).alias(f"{prefix}y"),
            (pl.col("ships").cast(pl.Float32).clip(1.0, None).log(math.e) / _LOG1024).alias(f"{prefix}ships"),
            (pl.col("nature") == "fix").cast(pl.Float32).alias(f"{prefix}is_fix"),
            (pl.col("nature") == "moving").cast(pl.Float32).alias(f"{prefix}is_moving"),
            (pl.col("nature") == "comet").cast(pl.Float32).alias(f"{prefix}is_comet"),
            (pl.col("production").cast(pl.Float32) / 5.0).alias(f"{prefix}production"),
            (pl.col("owner") == -1).cast(pl.Float32).alias(f"{prefix}owner_n1"),
            (pl.col("owner") == 0).cast(pl.Float32).alias(f"{prefix}owner_0"),
            (pl.col("owner") == 1).cast(pl.Float32).alias(f"{prefix}owner_1"),
            (pl.col("owner") == 2).cast(pl.Float32).alias(f"{prefix}owner_2"),
            (pl.col("owner") == 3).cast(pl.Float32).alias(f"{prefix}owner_3"),
        ])
        .select([
            pl.col("id").alias(id_alias),
            f"{prefix}x", f"{prefix}y", f"{prefix}ships",
            f"{prefix}is_fix", f"{prefix}is_moving", f"{prefix}is_comet",
            f"{prefix}production",
            f"{prefix}owner_n1", f"{prefix}owner_0", f"{prefix}owner_1",
            f"{prefix}owner_2", f"{prefix}owner_3",
        ])
    )


def _trajectory_pivot(
    df_s_window: pl.DataFrame,
    id_alias: str,
    ships_prefix: str,
    owner_prefix: str | None = None,
) -> pl.DataFrame:
    """Pivot df_s_window (has 'offset' int and 'ships_log' float cols) to wide trajectory.

    Returns DataFrame keyed by id_alias with columns {ships_prefix}0..19
    and optionally {owner_prefix}0..19.
    Missing offsets filled with 0.0 (ships) or -2.0 (owner sentinel).
    """
    ships_piv = df_s_window.pivot(
        values="ships_log", index="id", on="offset", aggregate_function="first"
    )
    for i in range(NB_STEPS_SIM):
        col, new = str(i), f"{ships_prefix}{i}"
        if col in ships_piv.columns:
            ships_piv = ships_piv.with_columns(
                pl.col(col).fill_null(0.0).cast(pl.Float32).alias(new)
            ).drop(col)
        else:
            ships_piv = ships_piv.with_columns(pl.lit(0.0).cast(pl.Float32).alias(new))
    result = ships_piv.rename({"id": id_alias})

    if owner_prefix is not None:
        owner_piv = df_s_window.pivot(
            values="owner", index="id", on="offset", aggregate_function="first"
        )
        for i in range(NB_STEPS_SIM):
            col, new = str(i), f"{owner_prefix}{i}"
            if col in owner_piv.columns:
                owner_piv = owner_piv.with_columns(
                    pl.col(col).fill_null(-2).cast(pl.Float32).alias(new)
                ).drop(col)
            else:
                owner_piv = owner_piv.with_columns(pl.lit(-2.0).cast(pl.Float32).alias(new))
        owner_cols = [f"{owner_prefix}{i}" for i in range(NB_STEPS_SIM)]
        result = result.join(
            owner_piv.select(["id"] + owner_cols).rename({"id": id_alias}),
            on=id_alias,
        )

    return result


def build_episode_features(ep_dir: Path) -> pl.DataFrame:
    df_s    = pl.read_parquet(ep_dir / "df_s.parquet")
    reach   = pl.read_parquet(ep_dir / "reach.parquet")
    actions = pl.read_parquet(ep_dir / "actions.parquet")

    # Keep earliest arrival per (id_src, step_src, id, ships_sent)
    reach = (
        reach
        .group_by(["id_src", "step_src", "id", "ships_sent"])
        .agg(pl.col("step").min())
    )

    step_dfs: list[pl.DataFrame] = []

    all_steps = sorted(reach["step_src"].unique().to_list())
    for t in all_steps[::STEP_STRIDE]:
        df_s_at_t = df_s.filter(pl.col("step") == t)
        if df_s_at_t.is_empty():
            continue

        src_ids = df_s_at_t.filter(pl.col("owner") == 0).select(pl.col("id").alias("id_src"))
        if src_ids.is_empty():
            continue

        # owner-0 planets × all planets, drop self-pairs
        pairs = (
            src_ids
            .join(df_s_at_t.select(pl.col("id").alias("id_tgt")), how="cross")
            .filter(pl.col("id_src") != pl.col("id_tgt"))
        )

        pairs = pairs.join(_snap_features(df_s_at_t, "id_src", "src_"), on="id_src", how="left")
        pairs = pairs.join(_snap_features(df_s_at_t, "id_tgt", "tgt_"), on="id_tgt", how="left")

        # ── Travel time: pivot reach_t → one column per ships_sent ───────────
        reach_t    = reach.filter(pl.col("step_src") == t)
        reach_wide = (
            reach_t
            .pivot(values="step", index=["id_src", "id"], on="ships_sent",
                   aggregate_function="first")
            .rename({"id": "id_tgt"})
        )
        travel_exprs = [
            ((pl.col(str(s)) - t) / NB_STEPS_SIM).cast(pl.Float32).alias(f"travel_{s}")
            if str(s) in reach_wide.columns
            else pl.lit(None).cast(pl.Float32).alias(f"travel_{s}")
            for s in REACH_POW2
        ]
        reach_wide = reach_wide.with_columns(travel_exprs).select(
            ["id_src", "id_tgt"] + [f"travel_{s}" for s in REACH_POW2]
        )
        pairs = pairs.join(reach_wide, on=["id_src", "id_tgt"], how="left")

        # ── Trajectory: target ships+owner, source ships for offsets 0..19 ────
        df_s_window = (
            df_s
            .filter((pl.col("step") >= t) & (pl.col("step") < t + NB_STEPS_SIM))
            .with_columns([
                (pl.col("step") - t).alias("offset"),
                (pl.col("ships").cast(pl.Float32).clip(1.0, None).log(math.e) / _LOG1024)
                .alias("ships_log"),
            ])
        )

        if not df_s_window.is_empty():
            tgt_traj = _trajectory_pivot(df_s_window, "id_tgt", "tgt_ships_t", "tgt_owner_t")
            src_traj = _trajectory_pivot(df_s_window, "id_src", "src_ships_t", owner_prefix=None)
            pairs = pairs.join(tgt_traj, on="id_tgt", how="left")
            pairs = pairs.join(src_traj, on="id_src", how="left")
        else:
            for i in range(NB_STEPS_SIM):
                pairs = pairs.with_columns([
                    pl.lit(0.0).cast(pl.Float32).alias(f"tgt_ships_t{i}"),
                    pl.lit(-2.0).cast(pl.Float32).alias(f"tgt_owner_t{i}"),
                    pl.lit(0.0).cast(pl.Float32).alias(f"src_ships_t{i}"),
                ])

        # ── Labels: 1.0 if (t, id_src, id_tgt) was attacked, else 0.0 ────────
        label_df = (
            actions.filter(pl.col("game_step") == t)
            .rename({"id": "id_tgt"})
            .select(["id_src", "id_tgt"])
            .unique()
            .with_columns(pl.lit(1.0).cast(pl.Float32).alias("label"))
        )
        pairs = (
            pairs
            .join(label_df, on=["id_src", "id_tgt"], how="left")
            .with_columns(pl.col("label").fill_null(0.0))
        )

        step_dfs.append(pairs.with_columns(pl.lit(t).alias("game_step")))

    return pl.concat(step_dfs) if step_dfs else pl.DataFrame()


def _test_snap_features():
    df = pl.DataFrame({
        "id": [0, 1], "step": [5, 5],
        "x": [30.0, 70.0], "y": [30.0, 70.0],
        "ships": [10, 5], "owner": [0, 1],
        "production": [2, 1], "nature": ["fix", "moving"],
    })
    result = _snap_features(df, id_alias="id_src", prefix="src_")
    assert "src_x" in result.columns,        "missing src_x"
    assert "src_owner_n1" in result.columns, "missing src_owner_n1"
    row0 = result.filter(pl.col("id_src") == 0)
    assert abs(row0["src_x"][0] - 0.30) < 1e-4, f"src_x wrong: {row0['src_x'][0]}"
    row1 = result.filter(pl.col("id_src") == 1)
    assert row1["src_is_moving"][0] == 1.0,  "is_moving wrong"
    assert row1["src_is_fix"][0]    == 0.0,  "is_fix wrong"
    print("_test_snap_features PASSED")


def _test_travel_features():
    ep_dirs = sorted([d for d in PRECOMPUTE_DIR.iterdir() if d.is_dir()])
    if not ep_dirs:
        print("_test_travel_features SKIP (no episodes)"); return
    df = build_episode_features(ep_dirs[0])
    assert "travel_1"    in df.columns, "missing travel_1"
    assert "travel_1024" in df.columns, "missing travel_1024"
    valid = df["travel_1"].drop_nulls()
    assert (valid >= 0.0).all() and (valid <= 1.0).all(), \
        f"travel_1 out of [0,1]: min={valid.min()}, max={valid.max()}"
    print(f"_test_travel_features PASSED  rows={len(df)}  travel_1_non_null={len(valid)}")


def _test_trajectory_features():
    ep_dirs = sorted([d for d in PRECOMPUTE_DIR.iterdir() if d.is_dir()])
    if not ep_dirs:
        print("_test_trajectory_features SKIP"); return
    df = build_episode_features(ep_dirs[0])
    for col in ["tgt_ships_t0", "tgt_owner_t0", "src_ships_t0",
                "tgt_ships_t19", "tgt_owner_t19", "src_ships_t19"]:
        assert col in df.columns, f"missing {col}"
    assert df["tgt_ships_t0"].dtype == pl.Float32, "expected Float32"
    assert df["src_ships_t0"].dtype == pl.Float32, "expected Float32"
    print(f"_test_trajectory_features PASSED  rows={len(df)}")


def _test_labels():
    ep_dirs = sorted([d for d in PRECOMPUTE_DIR.iterdir() if d.is_dir()])
    if not ep_dirs:
        print("_test_labels SKIP"); return
    df = build_episode_features(ep_dirs[0])

    missing = [c for c in FEATURE_COLS if c not in df.columns]
    assert not missing, f"Missing FEATURE_COLS: {missing}"

    label_vals = set(df["label"].unique().to_list())
    assert label_vals.issubset({0.0, 1.0}), f"Unexpected label values: {label_vals}"

    n_pos = int(df["label"].sum())
    assert n_pos > 0,       "Expected at least one positive label"
    assert n_pos < len(df), "Expected at least one negative label"

    print(f"_test_labels PASSED  rows={len(df)}  positives={n_pos}  "
          f"pos_rate={n_pos/len(df):.4f}")


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)

    ep_dirs = sorted([d for d in PRECOMPUTE_DIR.iterdir() if d.is_dir()])
    ep_ids  = [int(d.name) for d in ep_dirs]
    random.seed(42)
    random.shuffle(ep_ids)   # mix easy (low ID) and hard (high ID) in both sets

    split      = int(TRAIN_RATIO * len(ep_ids))
    train_dirs = [PRECOMPUTE_DIR / str(eid) for eid in ep_ids[:split]]
    test_dirs  = [PRECOMPUTE_DIR / str(eid) for eid in ep_ids[split:]]
    print(f"Episodes: {len(ep_ids)} total | {len(train_dirs)} train | {len(test_dirs)} test")

    def build_dataset(dirs: list[Path], desc: str) -> tuple[np.ndarray, np.ndarray]:
        X_parts: list[np.ndarray] = []
        y_parts: list[np.ndarray] = []
        for d in tqdm(dirs, desc=desc):
            try:
                frame = build_episode_features(d)
                if not frame.is_empty():
                    X_parts.append(
                        frame.select(FEATURE_COLS).fill_null(float("nan"))
                             .to_numpy().astype(np.float32)
                    )
                    y_parts.append(frame["label"].to_numpy().astype(np.float32))
            except Exception as e:
                print(f"  {d.name} failed: {e}")
        if not X_parts:
            return (np.zeros((0, len(FEATURE_COLS)), dtype=np.float32),
                    np.zeros(0, dtype=np.float32))
        return np.concatenate(X_parts, axis=0), np.concatenate(y_parts)

    X_train, y_train = build_dataset(train_dirs, "Train")
    X_test,  y_test  = build_dataset(test_dirs,  "Test")

    n_pos      = int(y_train.sum())
    n_neg      = len(y_train) - n_pos
    pos_weight = n_neg / max(n_pos, 1)
    print(f"Train: {len(X_train)} pairs | {n_pos} pos | pos_weight={pos_weight:.1f}")
    print(f"Test:  {len(X_test)} pairs  | {int(y_test.sum())} pos")

    mlflow.set_experiment("118-LGBM")
    with mlflow.start_run():
        mlflow.log_params({
            "train_ratio":    TRAIN_RATIO,
            "train_episodes": len(train_dirs),
            "test_episodes":  len(test_dirs),
            "n_features":     len(FEATURE_COLS),
            "pos_weight":     round(pos_weight, 4),
            "step_stride":    STEP_STRIDE,
            **{k: v for k, v in LGBM_PARAMS.items() if k != "verbose"},
        })

        model = LGBMClassifier(scale_pos_weight=pos_weight, **LGBM_PARAMS)
        model.fit(X_train, y_train, feature_name=FEATURE_COLS)

        train_auc = roc_auc_score(y_train, model.predict_proba(X_train)[:, 1])
        test_auc  = roc_auc_score(y_test,  model.predict_proba(X_test)[:, 1])
        pos_rate  = float(y_train.mean())

        print(f"\nTrain AUC: {train_auc:.4f}  |  Test AUC: {test_auc:.4f}")

        mlflow.log_metrics({
            "train_auc":     train_auc,
            "test_auc":      test_auc,
            "n_train_pairs": len(X_train),
            "n_test_pairs":  len(X_test),
            "positive_rate": pos_rate,
        })

        model_path = OUT_DIR / "model.txt"
        model.booster_.save_model(str(model_path))
        mlflow.log_artifact(str(model_path))

        importances = sorted(
            zip(FEATURE_COLS, model.feature_importances_), key=lambda x: -x[1]
        )
        print("\nTop 20 features:")
        for name, imp in importances[:20]:
            print(f"  {name:<30s} {imp}")

    print(f"\nDone. Model saved to {model_path}")


if __name__ == "__main__":
    _test_snap_features()
    _test_travel_features()
    _test_trajectory_features()
    _test_labels()
    print("\nAll tests passed. Starting training...\n")
    main()
