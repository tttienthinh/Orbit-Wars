"""Pre-compute GNN inputs from Kaggle winning replays.

Reads all episode JSONs from 110-replays/53402535/ and writes 3 parquet files
per episode to 114-precompute/{episode_id}/:

    df_s.parquet     (id, step, x, y, ships, owner, production, nature)
    reach.parquet    (id_src, step_src, id, step, ships_sent)
                     ships_sent in [1,2,4,8,16,32,64,128,256,512,1024],
                     all source planets, collision-filtered
    actions.parquet  (game_step, id_src, angle, ships_sent, id)
                     id = target planet resolved via fleet trajectory + swept_pair_hit

Usage:
    python 114-precompute.py
"""
import json
import math
import types
from pathlib import Path

import polars as pl
from tqdm import tqdm

import importlib.util as _ilu

# ── Load pipeline from 113 ────────────────────────────────────────────────────
_spec = _ilu.spec_from_file_location("m113", "113-Polars_GNN_Corrected.py")
_m = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_m)
SP = _m.StrategyPipeline

REPLAYS_DIR = Path("110-replays/53402535")
OUT_ROOT = Path("114-precompute")
META_PATH = REPLAYS_DIR / "_metadata.json"

DF_S_COLS = ["id", "step", "x", "y", "ships", "owner", "production", "nature"]
REACH_COLS = ["id_src", "step_src", "id", "step", "ships_sent"]
ATK_COLS = ["id_src", "step_src", "step", "id", "ships_sent", "angle"]
ACT_SCHEMA = {
    "game_step": pl.Int64, "id_src": pl.Int64, "angle": pl.Float64,
    "ships_sent": pl.Int64, "id": pl.Int64,
}

# Powers of 2 fleet sizes for reach edges and action-candidate enumeration
REACH_POW2 = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]

fleet_speed = _m.PhysicsEngine.fleet_speed


def swept_pair_hit(A, B, P0, P1, r):
    d0x, d0y = A[0] - P0[0], A[1] - P0[1]
    dvx = (B[0] - A[0]) - (P1[0] - P0[0])
    dvy = (B[1] - A[1]) - (P1[1] - P0[1])
    a = dvx * dvx + dvy * dvy
    b = 2.0 * (d0x * dvx + d0y * dvy)
    c = d0x * d0x + d0y * d0y - r * r
    if a < 1e-12:
        return c <= 0.0
    disc = b * b - 4.0 * a * c
    if disc < 0.0:
        return False
    sq = math.sqrt(disc)
    t1 = (-b - sq) / (2.0 * a)
    t2 = (-b + sq) / (2.0 * a)
    return t2 >= 0.0 and t1 <= 1.0


def _winner_idx(ep_meta: dict) -> int | None:
    agents = ep_meta.get("agents") or []
    rewards = [(i, a.get("reward")) for i, a in enumerate(agents) if isinstance(a, dict)]
    valid = [(i, r) for i, r in rewards if isinstance(r, (int, float))]
    if not valid:
        return None
    max_r = max(r for _, r in valid)
    if max_r <= 0:
        return None
    winners = [i for i, r in valid if r == max_r]
    return winners[0] if len(winners) == 1 else None


def _process_episode(
    ep_json: dict, player_idx: int
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    df_s_parts: list[pl.DataFrame] = []
    reach_parts: list[pl.DataFrame] = []
    atk_parts: list[pl.DataFrame] = []
    act_rows: list[dict] = []

    for game_step, step_states in enumerate(ep_json.get("steps", [])):
        if not step_states or len(step_states) <= player_idx:
            continue
        state = step_states[player_idx]
        if not isinstance(state, dict):
            continue
        if state.get("status", "ACTIVE") not in ("ACTIVE", None, ""):
            continue
        obs_dict = state.get("observation")
        if not obs_dict:
            continue

        obs = types.SimpleNamespace(**obs_dict)
        pid = obs.player

        initial = getattr(obs, "initial_planets", [])
        owners = {p[1] for p in initial if p[1] != -1}
        num_agents = 4 if len(owners) > 2 else 2

        try:
            df_s_full, planet_disp = SP._01_get_obs_dataframe(obs, game_step, num_agents)
            df_s_full = SP._00_remap_owner(df_s_full, obs, pid)

            # Single geometry pass: all planets × REACH_POW2 ships values.
            # This covers both reach (filter to REACH_POW2) and actions (filter
            # to owner==0, ships_sent<=ships_min) without running two pipelines.
            raw = SP._02_get_reach(df_s_full, planet_disp, REACH_POW2).collect()
            filtered = SP._03_filter_collision(raw.lazy()).collect()
        except Exception as e:
            print(f"  step {game_step} skipped: {e}")
            continue

        # ── df_s ─────────────────────────────────────────────────────────────
        df_s_parts.append(df_s_full.select(DF_S_COLS))

        # ── reach: REACH_POW2 values, all source planets ─────────────────────
        if not filtered.is_empty():
            reach_df = filtered.filter(pl.col("ships_sent").is_in(REACH_POW2))
            if not reach_df.is_empty():
                reach_parts.append(reach_df.select(REACH_COLS))

        # ── attacks: mine planets only, ships_sent <= ships_min ───────────────
        if not filtered.is_empty():
            atk_df = (
                filtered
                .filter(
                    (pl.col("owner_src") == 0) &
                    (pl.col("ships_sent") <= pl.col("ships_min"))
                )
                .select(["id_src", "step_src", "step", "id", "ships_sent", "final_angle"])
                .rename({"final_angle": "angle"})
            )
            if not atk_df.is_empty():
                atk_parts.append(atk_df)

        # ── actions: map launch angle → target planet id ──────────────────────
        # Use simple geometric lookup: find the planet whose current position
        # is most aligned with the action launch angle from the source planet.
        # This works regardless of the player's angle convention (direct aim vs
        # orbital intercept) since planet separation >> orbital displacement.
        base_pos: dict[int, tuple[float, float]] = {
            row[0]: (row[1], row[2])
            for row in df_s_full
            .group_by("id")
            .agg(pl.first("x"), pl.first("y"))
            .rows()
        }
        _TWO_PI = 2.0 * math.pi

        raw_action = state.get("action") or []
        if isinstance(raw_action, list):
            for move in raw_action:
                if isinstance(move, list) and len(move) == 3:
                    id_src = int(move[0])
                    angle = float(move[1])
                    ships = int(move[2])
                    id_tgt = None
                    src = base_pos.get(id_src)
                    if src is not None:
                        sx, sy = src
                        best_diff = 0.25  # ~14 degree threshold
                        for tgt_id, (tx, ty) in base_pos.items():
                            if tgt_id == id_src:
                                continue
                            tgt_angle = math.atan2(ty - sy, tx - sx)
                            diff = abs(((angle - tgt_angle + math.pi) % _TWO_PI) - math.pi)
                            if diff < best_diff:
                                best_diff = diff
                                id_tgt = tgt_id
                    act_rows.append({
                        "game_step": game_step,
                        "id_src": id_src,
                        "angle": angle,
                        "ships_sent": ships,
                        "id": id_tgt,
                    })

    # ── Batch dedup and assemble output ───────────────────────────────────────
    df_s_out = (
        pl.concat(df_s_parts).unique(subset=["id", "step"], keep="first")
        if df_s_parts
        else pl.DataFrame(schema={c: pl.Int64 for c in DF_S_COLS})
    )
    reach_out = (
        pl.concat(reach_parts).unique(keep="first")
        if reach_parts
        else pl.DataFrame(schema={c: pl.Int64 for c in REACH_COLS})
    )
    atk_out = (
        pl.concat(atk_parts)
        if atk_parts
        else pl.DataFrame(schema={c: pl.Int64 for c in ATK_COLS})
    )
    act_out = (
        pl.DataFrame(act_rows, schema=ACT_SCHEMA)
        if act_rows
        else pl.DataFrame(schema=ACT_SCHEMA)
    )

    return df_s_out, reach_out, atk_out, act_out


def main() -> None:
    meta: list[dict] = json.loads(META_PATH.read_text(encoding="utf-8"))
    meta_by_id: dict[int, dict] = {int(ep["id"]): ep for ep in meta}

    episode_files = sorted(REPLAYS_DIR.glob("episode_*.json"))
    print(f"Found {len(episode_files)} episode files")

    OUT_ROOT.mkdir(exist_ok=True)

    skipped = done = errors = 0
    bar = tqdm(episode_files, unit="ep", dynamic_ncols=True)
    for ep_path in bar:
        ep_id = int(ep_path.stem.split("_")[1])
        out_dir = OUT_ROOT / str(ep_id)
        all_files = ("df_s.parquet", "reach.parquet", "attacks.parquet", "actions.parquet")

        if out_dir.exists() and all((out_dir / f).exists() for f in all_files):
            skipped += 1
            bar.set_postfix(done=done, skipped=skipped, errors=errors)
            continue

        ep_meta = meta_by_id.get(ep_id)
        player_idx = _winner_idx(ep_meta) if ep_meta else None
        if player_idx is None:
            errors += 1
            bar.set_postfix(done=done, skipped=skipped, errors=errors)
            continue

        try:
            ep_json = json.loads(ep_path.read_text(encoding="utf-8"))
            df_s, reach, attacks, actions = _process_episode(ep_json, player_idx)

            out_dir.mkdir(exist_ok=True)
            df_s.write_parquet(out_dir / "df_s.parquet")
            reach.write_parquet(out_dir / "reach.parquet")
            attacks.write_parquet(out_dir / "attacks.parquet")
            actions.write_parquet(out_dir / "actions.parquet")
            done += 1
        except Exception as e:
            print(f"\n  episode {ep_id} failed: {e}")
            errors += 1

        bar.set_postfix(done=done, skipped=skipped, errors=errors)

    print(f"\nDone. {done} processed, {skipped} skipped, {errors} errors -> {OUT_ROOT}")


if __name__ == "__main__":
    main()
