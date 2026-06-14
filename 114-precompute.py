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
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    df_s_parts: list[pl.DataFrame] = []
    reach_parts: list[pl.DataFrame] = []
    raw_actions: list[tuple] = []
    planet_snap: dict[int, dict] = {}   # step -> {pid: (x, y, radius)}
    fleet_snap: dict[int, dict] = {}    # step -> {fid: (x, y, angle, from_pid, ships)}

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

        # ── Always snapshot planets/fleets (needed even for DONE steps) ──────
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

            raw = SP._02_get_reach(df_s_full, planet_disp, REACH_POW2).collect()
            filtered = SP._03_filter_collision(raw.lazy()).collect()
        except Exception as e:
            print(f"  step {game_step} skipped: {e}")
            continue

        # ── df_s ─────────────────────────────────────────────────────────────
        df_s_parts.append(df_s_full.select(DF_S_COLS))

        # ── reach: all REACH_POW2, all source planets ─────────────────────────
        if not filtered.is_empty():
            reach_parts.append(filtered.select(REACH_COLS))

        # ── raw actions (target resolved in Pass 2 below) ─────────────────────
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

    # ── Pass 2: resolve each action's target planet ───────────────────────────
    act_rows: list[dict] = []
    for game_step, id_src, angle, ships_sent in raw_actions:
        id_tgt = None

        matched_fid = None
        for fid in sorted(fleet_life):
            info = fleet_life[fid]
            if (info["born"] == game_step + 1
                    and info["from_pid"] == id_src
                    and info["angle"] == angle
                    and info["ships"] == ships_sent):
                matched_fid = fid
                break

        if matched_fid is not None:
            info = fleet_life[matched_fid]
            t_last = info["last_step"]
            next_step = t_last + 1
            # Fleet absent from next observed step → it hit something that tick
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
                            px1, py1 = px0, py0  # planet expired (comet), hold in place
                        if swept_pair_hit(fl_old, fl_new, (px0, py0), (px1, py1), r0):
                            id_tgt = pid_t
                            break

        act_rows.append({
            "game_step": game_step, "id_src": id_src,
            "angle": angle, "ships_sent": ships_sent, "id": id_tgt,
        })

    # ── Assemble outputs ──────────────────────────────────────────────────────
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
    act_out = (
        pl.DataFrame(act_rows, schema=ACT_SCHEMA)
        if act_rows
        else pl.DataFrame(schema=ACT_SCHEMA)
    )

    return df_s_out, reach_out, act_out


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
