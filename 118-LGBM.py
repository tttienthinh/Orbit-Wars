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

    for t in sorted(reach["step_src"].unique().to_list()):
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
            .pivot(values="step", index=["id_src", "id"], columns="ships_sent",
                   aggregate_function="first")
            .rename({"id": "id_tgt"})
        )
        for s in REACH_POW2:
            col = str(s)
            new = f"travel_{s}"
            if col in reach_wide.columns:
                reach_wide = reach_wide.with_columns(
                    ((pl.col(col) - t) / NB_STEPS_SIM).cast(pl.Float32).alias(new)
                ).drop(col)
            else:
                reach_wide = reach_wide.with_columns(
                    pl.lit(None).cast(pl.Float32).alias(new)
                )
        pairs = pairs.join(reach_wide, on=["id_src", "id_tgt"], how="left")

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
