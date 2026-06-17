"""Pre-compute GNN inputs from Kaggle winning replays.

Reads all episode JSONs from 110-replays/53402535/ and writes 5 parquet files
per episode to 126-precompute/{episode_id}/:

    planete.parquet             (id, step, x, y, production, nature)
    planete_step.parquet        (id, step, future_step, ships, owner)
                                step=obs game_step, future_step in [step, step+20]
    reachable_base_2.parquet    (id_src, step_src, id_tgt, step_tgt, ships_sent)
                                ships_sent in [1,2,4,8,16,32,64,128,256,512,1024],
                                collision-filtered
    reachable_max_ships.parquet (id_src, step_src, angle, ships_sent, id_tgt, step_tgt)
                                ships_sent = ships at source at step_src,
                                one row per (id_src, step_src, id_tgt), min step_tgt
    actions_to_copy.parquet     (step, id_src, angle, ships_sent, id_tgt)
                                id_tgt resolved via fleet trajectory + swept_pair_hit

Usage:
    python 126-precompute.py [--episode EPISODE_ID]
"""
import argparse
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
GameConfig = _m.GameConfig

REPLAYS_ROOT = Path("110-replays")
OUT_ROOT = Path("126-precompute")

PLANETE_COLS = ["id", "step", "x", "y", "production", "nature"]
PLANETE_STEP_COLS = ["id", "step", "future_step", "ships", "owner"]
REACH_B2_COLS = ["id_src", "step_src", "id_tgt", "step_tgt", "ships_sent"]
REACH_MAX_COLS = ["id_src", "step_src", "angle", "ships_sent", "id_tgt", "step_tgt"]
ACT_SCHEMA = {
    "step": pl.Int64, "id_src": pl.Int64, "angle": pl.Float64,
    "ships_sent": pl.Int64, "id_tgt": pl.Int64,
}

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


def _get_reach_max_ships(
    df_s: pl.DataFrame,
    planet_disp: pl.DataFrame,
) -> pl.LazyFrame:
    """Like _02_get_reach but ships_sent = ships at source at step_src (no explode)."""
    df_s_lf = df_s.lazy()
    planet_disp_lf = planet_disp.lazy()

    all_base_lf = (
        df_s_lf
        .group_by("id", maintain_order=True)
        .agg([
            pl.first("step").alias("step_src"),
            pl.first("x").alias("x_src"),
            pl.first("y").alias("y_src"),
            pl.first("radius").alias("radius_src"),
            pl.first("ships").alias("ships_at_src"),
            pl.first("production").alias("production_src"),
            pl.first("nature").alias("nature_src"),
            pl.first("owner").alias("owner_src"),
        ])
        .rename({"id": "id_src"})
    )

    dx = pl.col("x") - pl.col("x_src")
    dy = pl.col("y") - pl.col("y_src")
    l2 = dx.pow(2) + dy.pow(2)
    dist_tgt_src = l2.sqrt()
    step_diff = (pl.col("step") - pl.col("step_src")).cast(pl.Float64)

    dot = (GameConfig.CENTER - pl.col("x_src")) * dx + (GameConfig.CENTER - pl.col("y_src")) * dy
    t_sun = (dot / pl.when(l2 == 0).then(pl.lit(1.0)).otherwise(l2)).clip(0.0, 1.0)
    proj_dist_sun = (
        (GameConfig.CENTER - pl.col("x_src") - t_sun * dx).pow(2) +
        (GameConfig.CENTER - pl.col("y_src") - t_sun * dy).pow(2)
    ).sqrt()
    crossing_sun = pl.when(l2 == 0).then(
        ((GameConfig.CENTER - pl.col("x_src")).pow(2) +
         (GameConfig.CENTER - pl.col("y_src")).pow(2)).sqrt()
    ).otherwise(proj_dist_sun) < (GameConfig.SUN_RADIUS + GameConfig.PLANET_MARGIN)

    coarse_lf = (
        all_base_lf
        .join(df_s_lf, how="cross")
        .filter(
            (pl.col("step") > pl.col("step_src")) & (pl.col("id") != pl.col("id_src"))
        )
        .join(planet_disp_lf, on=["id", "step"], how="left")
        .with_columns([
            dist_tgt_src.alias("dist_tgt_src"),
            step_diff.alias("step_diff"),
        ])
        .filter(
            (pl.col("dist_tgt_src") <
             (pl.col("step_diff") + 1) * GameConfig.MAX_SPEED
             + pl.col("radius_src") + GameConfig.PLANET_MARGIN + pl.col("radius")
             + pl.col("planet_disp").fill_null(0.0))
            & ~crossing_sun
        )
        .with_columns(pl.col("ships_at_src").alias("ships_sent"))
    )

    fleet_speed_expr = 1.0 + (GameConfig.MAX_SPEED - 1.0) * (
        pl.col("ships_sent").cast(pl.Float64).log(base=math.e) / math.log(1000.0)
    ).clip(lower_bound=0.0).pow(1.5)
    dist_min_expr = pl.col("step_diff") * fleet_speed_expr + GameConfig.PLANET_MARGIN + pl.col("radius_src")
    dist_prev_expr = dist_min_expr - fleet_speed_expr

    prev_pos_lf = (
        df_s_lf.select(["id", "step", "x", "y"])
        .rename({"x": "x_prev", "y": "y_prev"})
        .with_columns((pl.col("step") + 1).alias("step"))
    )

    unit_x = (pl.col("x") - pl.col("x_src")) / pl.when(
        pl.col("dist_tgt_src") < 1e-9
    ).then(pl.lit(1.0)).otherwise(pl.col("dist_tgt_src"))
    unit_y = (pl.col("y") - pl.col("y_src")) / pl.when(
        pl.col("dist_tgt_src") < 1e-9
    ).then(pl.lit(1.0)).otherwise(pl.col("dist_tgt_src"))
    fleet_x0 = pl.col("x_src") + unit_x * pl.col("dist_prev")
    fleet_y0 = pl.col("y_src") + unit_y * pl.col("dist_prev")
    planet_vx = pl.col("x") - pl.col("x_prev").fill_null(pl.col("x"))
    planet_vy = pl.col("y") - pl.col("y_prev").fill_null(pl.col("y"))
    dvx_sp = unit_x * pl.col("fleet_speed") - planet_vx
    dvy_sp = unit_y * pl.col("fleet_speed") - planet_vy
    d0x_sp = fleet_x0 - pl.col("x_prev").fill_null(pl.col("x"))
    d0y_sp = fleet_y0 - pl.col("y_prev").fill_null(pl.col("y"))
    a_sp = dvx_sp.pow(2) + dvy_sp.pow(2)
    b_sp = 2.0 * (d0x_sp * dvx_sp + d0y_sp * dvy_sp)
    c_sp = d0x_sp.pow(2) + d0y_sp.pow(2) - pl.col("radius").pow(2)
    disc_sp = b_sp.pow(2) - 4.0 * a_sp * c_sp
    sq_sp = disc_sp.clip(lower_bound=0.0).sqrt()
    t1_expr = pl.when(a_sp < 1e-12).then(pl.lit(0.0)).otherwise((-b_sp - sq_sp) / (2.0 * a_sp))
    t2_expr = pl.when(a_sp < 1e-12).then(pl.lit(1.0)).otherwise((-b_sp + sq_sp) / (2.0 * a_sp))
    collision = pl.when(a_sp < 1e-12).then(c_sp <= 0.0).otherwise(
        (disc_sp >= 0.0) & (t2_expr >= 0.0) & (t1_expr <= 1.0)
    )

    x_prev_f = pl.col("x_prev").fill_null(pl.col("x"))
    y_prev_f = pl.col("y_prev").fill_null(pl.col("y"))

    reach_lf = (
        coarse_lf
        .with_columns([
            fleet_speed_expr.alias("fleet_speed"),
            dist_min_expr.alias("dist_min"),
            dist_prev_expr.alias("dist_prev"),
        ])
        .filter(
            pl.col("dist_tgt_src") < pl.col("dist_min") + pl.col("fleet_speed")
            + pl.col("radius") + GameConfig.PLANET_MOVEMENT_SLACK
        )
        .join(prev_pos_lf, on=["id", "step"], how="left")
        .with_columns([
            t1_expr.alias("t1"),
            t2_expr.alias("t2"),
            collision.alias("collision"),
        ])
        .filter(pl.col("collision"))
        .with_columns([
            pl.col("t1").clip(0.0, 1.0).alias("t1_eff"),
            pl.col("t2").clip(0.0, 1.0).alias("t2_eff"),
        ])
        .with_columns([
            (x_prev_f + pl.col("t1_eff") * (pl.col("x") - x_prev_f)).alias("p_t1_x"),
            (y_prev_f + pl.col("t1_eff") * (pl.col("y") - y_prev_f)).alias("p_t1_y"),
            (x_prev_f + pl.col("t2_eff") * (pl.col("x") - x_prev_f)).alias("p_t2_x"),
            (y_prev_f + pl.col("t2_eff") * (pl.col("y") - y_prev_f)).alias("p_t2_y"),
        ])
        .with_columns([
            pl.arctan2(pl.col("p_t1_y") - pl.col("y_src"), pl.col("p_t1_x") - pl.col("x_src")).alias("angle_t1"),
            pl.arctan2(pl.col("p_t2_y") - pl.col("y_src"), pl.col("p_t2_x") - pl.col("x_src")).alias("angle_t2"),
            ((pl.col("p_t1_x") - pl.col("x_src")).pow(2) + (pl.col("p_t1_y") - pl.col("y_src")).pow(2)).sqrt().alias("d_s_t1"),
            ((pl.col("p_t2_x") - pl.col("x_src")).pow(2) + (pl.col("p_t2_y") - pl.col("y_src")).pow(2)).sqrt().alias("d_s_t2"),
        ])
        .with_columns([
            (pl.col("dist_prev") + pl.col("t1_eff") * pl.col("fleet_speed")).alias("d_f_t1"),
            (pl.col("dist_prev") + pl.col("t2_eff") * pl.col("fleet_speed")).alias("d_f_t2"),
        ])
        .with_columns([
            ((pl.col("d_s_t1").pow(2) + pl.col("d_f_t1").pow(2) - pl.col("radius").pow(2))
             / (2.0 * pl.col("d_s_t1") * pl.col("d_f_t1"))).clip(-1.0, 1.0).arccos().alias("angle_radius_t1"),
            ((pl.col("d_s_t2").pow(2) + pl.col("d_f_t2").pow(2) - pl.col("radius").pow(2))
             / (2.0 * pl.col("d_s_t2") * pl.col("d_f_t2"))).clip(-1.0, 1.0).arccos().alias("angle_radius_t2"),
        ])
        .with_columns([
            pl.min_horizontal(
                pl.col("angle_t1") - pl.col("angle_radius_t1"),
                pl.col("angle_t2") - pl.col("angle_radius_t2"),
            ).mod(2 * math.pi).alias("angle_min"),
            pl.max_horizontal(
                pl.col("angle_t1") + pl.col("angle_radius_t1"),
                pl.col("angle_t2") + pl.col("angle_radius_t2"),
            ).mod(2 * math.pi).alias("angle_max"),
            pl.arctan2(
                pl.col("angle_t1").sin() + pl.col("angle_t2").sin(),
                pl.col("angle_t1").cos() + pl.col("angle_t2").cos(),
            ).alias("angle"),
        ])
        .sort("step")
    )

    return reach_lf


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
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    planete_parts: list[pl.DataFrame] = []
    planete_step_parts: list[pl.DataFrame] = []
    reach_b2_parts: list[pl.DataFrame] = []
    reach_max_parts: list[pl.DataFrame] = []
    raw_actions: list[tuple] = []
    planet_snap: dict[int, dict] = {}
    fleet_snap: dict[int, dict] = {}

    for game_step, step_states in enumerate(ep_json.get("steps", [])):
        if not step_states or len(step_states) <= player_idx:
            continue
        state = step_states[player_idx]
        if not isinstance(state, dict):
            continue
        obs_dict = state.get("observation")
        if not obs_dict:
            continue

        obs = types.SimpleNamespace(**obs_dict)

        planet_snap[game_step] = {
            p[0]: (p[2], p[3], p[4])
            for p in getattr(obs, "planets", [])
        }
        fleet_snap[game_step] = {
            f[0]: (f[2], f[3], f[4], f[5], f[6])
            for f in getattr(obs, "fleets", [])
        }

        if state.get("status", "ACTIVE") not in ("ACTIVE", None, ""):
            continue

        pid = obs.player

        initial = getattr(obs, "initial_planets", [])
        owners = {p[1] for p in initial if p[1] != -1}
        num_agents = 4 if len(owners) > 2 else 2

        try:
            df_s_full, planet_disp = SP._01_get_obs_dataframe(obs, game_step, num_agents)
            df_s_full = SP._00_remap_owner(df_s_full, obs, pid)

            # ── planete: spatial info, deduped later ──────────────────────────
            planete_parts.append(df_s_full.select(PLANETE_COLS))

            # ── planete_step: dynamic state, tag with obs game_step ───────────
            planete_step_parts.append(
                df_s_full.select(["id", "step", "ships", "owner"])
                .filter(pl.col("step") >= game_step)
                .with_columns(pl.lit(game_step).cast(pl.Int64).alias("obs_step"))
                .rename({"obs_step": "step", "step": "future_step"})
                .select(PLANETE_STEP_COLS)
            )

            # ── reachable_base_2: pow2 fleet sizes, collision-filtered ────────
            raw_b2 = SP._02_get_reach(df_s_full, planet_disp, REACH_POW2).collect()
            filtered_b2 = SP._03_filter_collision(raw_b2.lazy()).collect()
            if not filtered_b2.is_empty():
                reach_b2_parts.append(
                    filtered_b2.select(["id_src", "step_src", "id", "step", "ships_sent"])
                    .rename({"id": "id_tgt", "step": "step_tgt"})
                )

            # ── reachable_max_ships: ships_at_src, collision-filtered ─────────
            raw_max = _get_reach_max_ships(df_s_full, planet_disp).collect()
            filtered_max = SP._03_filter_collision(raw_max.lazy()).collect()
            if not filtered_max.is_empty():
                reach_max_parts.append(
                    filtered_max
                    .select(["id_src", "step_src", "angle", "ships_sent", "id", "step"])
                    .rename({"id": "id_tgt", "step": "step_tgt"})
                    .sort("step_tgt")
                    .group_by(["id_src", "step_src", "id_tgt"], maintain_order=True)
                    .first()
                    .select(REACH_MAX_COLS)
                )

        except Exception as e:
            print(f"  step {game_step} skipped: {e}")
            continue

        # ── raw actions ───────────────────────────────────────────────────────
        raw_action = state.get("action") or []
        if isinstance(raw_action, list):
            for move in raw_action:
                if isinstance(move, list) and len(move) == 3:
                    raw_actions.append((
                        game_step, int(move[0]), float(move[1]), int(move[2])
                    ))

    # ── Pass 2: build fleet lifetime index ───────────────────────────────────
    all_fids: set[int] = set()
    for step_fleets in fleet_snap.values():
        all_fids.update(step_fleets.keys())

    fleet_life: dict[int, dict] = {}
    for fid in all_fids:
        born = last_step = None
        last_x = last_y = 0.0
        from_pid = angle_f = ships_f = None
        for step in sorted(fleet_snap):
            if fid in fleet_snap[step]:
                fx, fy, fa, fpid, fships = fleet_snap[step][fid]
                if born is None:
                    born = step
                    from_pid, angle_f, ships_f = fpid, fa, fships
                last_step = step
                last_x, last_y = fx, fy
        if born is not None:
            fleet_life[fid] = {
                "born": born, "from_pid": from_pid,
                "angle": angle_f, "ships": ships_f,
                "last_step": last_step, "last_x": last_x, "last_y": last_y,
            }

    fleet_by_sig: dict[tuple, int] = {}
    for fid, info in fleet_life.items():
        key = (info["born"], info["from_pid"], info["angle"], info["ships"])
        fleet_by_sig.setdefault(key, fid)

    # ── Pass 2: resolve each action's target planet ───────────────────────────
    act_rows: list[dict] = []
    for game_step, id_src, angle, ships_sent in raw_actions:
        id_tgt = None

        matched_fid = fleet_by_sig.get((game_step, id_src, angle, ships_sent))

        if matched_fid is not None:
            info = fleet_life[matched_fid]
            t_last = info["last_step"]
            next_step = t_last + 1
            if next_step in fleet_snap and matched_fid not in fleet_snap[next_step]:
                if next_step in planet_snap:
                    spd = fleet_speed(ships_sent)
                    fl_old = (info["last_x"], info["last_y"])
                    fl_new = (
                        info["last_x"] + spd * math.cos(angle),
                        info["last_y"] + spd * math.sin(angle),
                    )
                    for pid_t, (px0, py0, r0) in planet_snap[t_last].items():
                        if pid_t in planet_snap[next_step]:
                            px1, py1 = planet_snap[next_step][pid_t][:2]
                        else:
                            px1, py1 = px0, py0
                        if swept_pair_hit(fl_old, fl_new, (px0, py0), (px1, py1), r0):
                            id_tgt = pid_t
                            break

        act_rows.append({
            "step": game_step, "id_src": id_src,
            "angle": angle, "ships_sent": ships_sent, "id_tgt": id_tgt,
        })

    # ── Assemble outputs ──────────────────────────────────────────────────────
    planete_out = (
        pl.concat(planete_parts).unique(subset=["id", "step"], keep="first")
        if planete_parts
        else pl.DataFrame(schema={
            "id": pl.Int64, "step": pl.Int64,
            "x": pl.Float64, "y": pl.Float64,
            "production": pl.Int64, "nature": pl.Utf8,
        })
    )
    planete_step_out = (
        pl.concat(planete_step_parts).unique(subset=["id", "step", "future_step"], keep="first")
        if planete_step_parts
        else pl.DataFrame(schema={
            "id": pl.Int64, "step": pl.Int64, "future_step": pl.Int64,
            "ships": pl.Int64, "owner": pl.Int64,
        })
    )
    reach_b2_out = (
        pl.concat(reach_b2_parts).unique(keep="first")
        if reach_b2_parts
        else pl.DataFrame(schema={c: pl.Int64 for c in REACH_B2_COLS})
    )
    reach_max_out = (
        pl.concat(reach_max_parts)
        .sort("step_tgt")
        .group_by(["id_src", "step_src", "id_tgt"], maintain_order=True)
        .first()
        .select(REACH_MAX_COLS)
        if reach_max_parts
        else pl.DataFrame(schema={
            "id_src": pl.Int64, "step_src": pl.Int64,
            "angle": pl.Float64, "ships_sent": pl.Int64,
            "id_tgt": pl.Int64, "step_tgt": pl.Int64,
        })
    )
    act_out = (
        pl.DataFrame(act_rows, schema=ACT_SCHEMA)
        if act_rows
        else pl.DataFrame(schema=ACT_SCHEMA)
    )

    return planete_out, planete_step_out, reach_b2_out, reach_max_out, act_out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode", type=int, default=None,
                        help="Process only this episode ID")
    args = parser.parse_args()

    # Collect episodes + metadata from all submission subdirs
    meta_by_id: dict[int, dict] = {}
    episode_files: list[Path] = []
    for sub in sorted(REPLAYS_ROOT.iterdir()):
        if not sub.is_dir():
            continue
        meta_path = sub / "_metadata.json"
        if not meta_path.exists():
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        for ep in meta:
            meta_by_id[int(ep["id"])] = ep
        episode_files.extend(sorted(sub.glob("episode_*.json")))

    if args.episode is not None:
        episode_files = [f for f in episode_files
                         if int(f.stem.split("_")[1]) == args.episode]
    print(f"Found {len(episode_files)} episode files across {len(list(REPLAYS_ROOT.iterdir()))} submission dir(s)")

    OUT_ROOT.mkdir(exist_ok=True)

    all_files = (
        "planete.parquet", "planete_step.parquet",
        "reachable_base_2.parquet", "reachable_max_ships.parquet",
        "actions_to_copy.parquet",
    )

    skipped = done = errors = 0
    bar = tqdm(episode_files, unit="ep", dynamic_ncols=True)
    for ep_path in bar:
        ep_id = int(ep_path.stem.split("_")[1])
        out_dir = OUT_ROOT / str(ep_id)

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
            planete, planete_step, reach_b2, reach_max, actions = _process_episode(
                ep_json, player_idx
            )

            out_dir.mkdir(exist_ok=True)
            planete.write_parquet(out_dir / "planete.parquet")
            planete_step.write_parquet(out_dir / "planete_step.parquet")
            reach_b2.write_parquet(out_dir / "reachable_base_2.parquet")
            reach_max.write_parquet(out_dir / "reachable_max_ships.parquet")
            actions.write_parquet(out_dir / "actions_to_copy.parquet")
            done += 1
        except Exception as e:
            print(f"\n  episode {ep_id} failed: {e}")
            errors += 1

        bar.set_postfix(done=done, skipped=skipped, errors=errors)

    print(f"\nDone. {done} processed, {skipped} skipped, {errors} errors -> {OUT_ROOT}")


if __name__ == "__main__":
    main()
