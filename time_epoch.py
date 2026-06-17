"""One-epoch timing profiler for 119-NewGNN_Cosine training."""
import time
import random
from pathlib import Path
from collections import defaultdict

import polars as pl
import torch
import torch.nn.functional as F

# Import functions without triggering __main__ block
_ns = {"__name__": "imported"}
exec(open("119-NewGNN_Cosine.py").read(), _ns)

OrbitGNN        = _ns["OrbitGNN"]
build_graph     = _ns["build_graph"]
build_attack_pairs = _ns["build_attack_pairs"]
_apply_transform = _ns["_apply_transform"]
HIDDEN_DIM      = _ns["HIDDEN_DIM"]
NUM_LAYERS      = _ns["NUM_LAYERS"]
LR              = _ns["LR"]
WEIGHT_DECAY    = _ns["WEIGHT_DECAY"]

PRECOMPUTE_DIR     = Path("114-precompute")
TEST_EPISODE_IDS_L = {78867640, 78899068, 78982947, 79033183,
                      79126912, 79175592, 79228392, 79320069}
TRANSFORMS_L       = ["identity", "rot90", "rot180", "rot270"]
EPISODES_PER_EPOCH = 8

random.seed(42)
ep_dirs    = sorted([d for d in PRECOMPUTE_DIR.iterdir() if d.is_dir()])
train_dirs = [d for d in ep_dirs if int(d.name) not in TEST_EPISODE_IDS_L]
test_dirs  = [d for d in ep_dirs if int(d.name) in     TEST_EPISODE_IDS_L]
all_pairs  = [(ep_dir, t) for ep_dir in train_dirs for t in TRANSFORMS_L]

model     = OrbitGNN(hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

epoch_pairs = random.sample(all_pairs, EPISODES_PER_EPOCH)
timings     = defaultdict(float)
step_counts = defaultdict(int)

print(f"Profiling 1 epoch ({EPISODES_PER_EPOCH} train pairs + {len(test_dirs)} test eps)...\n")

# ── Training ──────────────────────────────────────────────────────────────────
model.train()
for ep_dir, transform in epoch_pairs:
    t0 = time.perf_counter()
    df_s    = pl.read_parquet(ep_dir / "df_s.parquet")
    reach   = pl.read_parquet(ep_dir / "reach.parquet")
    actions = pl.read_parquet(ep_dir / "actions.parquet")
    timings["1_load_parquet"] += time.perf_counter() - t0

    t0 = time.perf_counter()
    df_s = _apply_transform(df_s, transform)
    timings["2_transform"] += time.perf_counter() - t0

    t0 = time.perf_counter()
    reach = (reach.group_by(["id_src", "step_src", "id", "ships_sent"])
                  .agg(pl.all().sort_by("step").first()))
    actions_set = set(zip(actions["game_step"].to_list(),
                          actions["id_src"].to_list(),
                          actions["id"].to_list()))
    timings["3_preprocess"] += time.perf_counter() - t0

    optimizer.zero_grad()
    for t in sorted(reach["step_src"].unique().to_list()):
        reach_t       = reach.filter(pl.col("step_src") == t)
        arrival_steps = reach_t["step"].unique().to_list()
        df_s_t        = df_s.filter(pl.col("step").is_in([t] + arrival_steps))
        df_s_at_t     = df_s.filter(pl.col("step") == t)
        if df_s_t.is_empty() or df_s_at_t.is_empty():
            continue

        t0 = time.perf_counter()
        data, planet_idx = build_graph(df_s_t, reach_t)
        src_idx, tgt_idx, labels = build_attack_pairs(df_s_at_t, actions_set, planet_idx, t)
        timings["4_build_graph"] += time.perf_counter() - t0

        if len(src_idx) == 0:
            continue

        t0 = time.perf_counter()
        h_planet = model.encode(data)
        logits   = model.score_pairs(h_planet, src_idx, tgt_idx)
        timings["5_encode_gnn"] += time.perf_counter() - t0

        t0 = time.perf_counter()
        n_pos = labels.sum().item()
        n_neg = len(labels) - n_pos
        pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32)
        loss = F.binary_cross_entropy_with_logits(logits, labels, pos_weight=pos_weight)
        loss.backward()
        timings["6_loss_backward"] += time.perf_counter() - t0

        step_counts["train_steps"] += 1

    t0 = time.perf_counter()
    optimizer.step()
    timings["7_optimizer_step"] += time.perf_counter() - t0

# ── Test ──────────────────────────────────────────────────────────────────────
model.eval()
with torch.no_grad():
    for ep_dir in test_dirs:
        t0 = time.perf_counter()
        df_s    = pl.read_parquet(ep_dir / "df_s.parquet")
        reach   = pl.read_parquet(ep_dir / "reach.parquet")
        actions = pl.read_parquet(ep_dir / "actions.parquet")
        timings["8_test_load"] += time.perf_counter() - t0

        t0 = time.perf_counter()
        reach = (reach.group_by(["id_src", "step_src", "id", "ships_sent"])
                      .agg(pl.all().sort_by("step").first()))
        actions_set = set(zip(actions["game_step"].to_list(),
                              actions["id_src"].to_list(),
                              actions["id"].to_list()))
        timings["9_test_preprocess"] += time.perf_counter() - t0

        for t in sorted(reach["step_src"].unique().to_list()):
            reach_t       = reach.filter(pl.col("step_src") == t)
            arrival_steps = reach_t["step"].unique().to_list()
            df_s_t        = df_s.filter(pl.col("step").is_in([t] + arrival_steps))
            df_s_at_t     = df_s.filter(pl.col("step") == t)
            if df_s_t.is_empty() or df_s_at_t.is_empty():
                continue

            t0 = time.perf_counter()
            data, planet_idx = build_graph(df_s_t, reach_t)
            build_attack_pairs(df_s_at_t, actions_set, planet_idx, t)
            timings["A_test_build_graph"] += time.perf_counter() - t0

            if len(src_idx) == 0:
                continue

            t0 = time.perf_counter()
            h_planet = model.encode(data)
            timings["B_test_encode"] += time.perf_counter() - t0

            step_counts["test_steps"] += 1

# ── Report ────────────────────────────────────────────────────────────────────
total = sum(timings.values())
print(f"{'Phase':<28} {'Time(s)':>8}  {'%':>6}")
print("-" * 48)
for k in sorted(timings):
    label = k[2:]
    pct = 100 * timings[k] / total
    print(f"  {label:<26} {timings[k]:>8.3f}  {pct:>5.1f}%")
print("-" * 48)
print(f"  {'TOTAL':<26} {total:>8.3f}  100.0%")
print()
print(f"Train game-steps: {step_counts['train_steps']}")
print(f"Test  game-steps: {step_counts['test_steps']}")
