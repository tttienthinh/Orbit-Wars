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
