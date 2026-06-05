import math, copy
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData

# Pull in all symbols from 96 and 90 (interpreter, PhysicsEngine, Obs, GameConfig,
# get_obs_dataframe, get_opportunities, StrategyPipeline, etc.)
exec(open('96-library.py').read())
exec(open('90-Simulate10Next_Conqueror2_Supplier_prod_per_step.py').read())

_LOG1024 = math.log(1024.0)
_MAX_SHIPS_SEARCH = 1024


def _fleet_hits_at_step(ships: int, src_x: float, src_y: float, r_src: float,
                         df_s: pd.DataFrame, dst_id: int, target_step: int,
                         base_step: int = 0) -> bool:
    """True iff a fleet of `ships` ships launched from (src_x, src_y) aimed at
    dst's position at base_step+target_step collides with dst at that exact step.

    Uses the same swept-pair intercept logic as get_opportunities:
    - fleet travels in a straight line toward dst's position at target_step
    - swept_pair_hit checks fleet segment vs planet segment between steps
    """
    speed = PhysicsEngine.fleet_speed(ships)
    dst_rows = df_s[df_s['id'] == dst_id]

    dst_at = dst_rows[dst_rows['step'] == base_step + target_step]
    if dst_at.empty:
        return False

    tgt_x = float(dst_at['x'].values[0])
    tgt_y = float(dst_at['y'].values[0])
    r_dst = float(dst_at['radius'].values[0])

    dx, dy = tgt_x - src_x, tgt_y - src_y
    dist_to_tgt = math.hypot(dx, dy)
    if dist_to_tgt < 1e-9:
        return False

    ux, uy = dx / dist_to_tgt, dy / dist_to_tgt
    start_dist = r_src + GameConfig.PLANET_MARGIN

    f_old = (src_x + ux * (start_dist + (target_step - 1) * speed),
             src_y + uy * (start_dist + (target_step - 1) * speed))
    f_new = (src_x + ux * (start_dist + target_step * speed),
             src_y + uy * (start_dist + target_step * speed))

    # Planet p_old: position at target_step-1 (for step 1, use step 0)
    prev_abs = base_step + target_step - 1
    dst_prev = dst_rows[dst_rows['step'] == prev_abs]
    if not dst_prev.empty:
        p_old = (float(dst_prev['x'].values[0]), float(dst_prev['y'].values[0]))
    else:
        p_old = (tgt_x, tgt_y)

    return PhysicsEngine.swept_pair_hit(f_old, f_new, p_old, (tgt_x, tgt_y), r_dst)


def _find_eta_range(src_x: float, src_y: float, r_src: float,
                    df_s: pd.DataFrame, dst_id: int, eta: int,
                    base_step: int = 0):
    """Return (min_ships, max_ships) for ETA bucket eta, or (None, None) if empty.

    min_ships: fewest ships that arrive at step eta (fast enough).
    max_ships: most ships that arrive at step eta but NOT step eta-1 (not too fast).
    Binary search over [1, 1024].  More ships = faster = smaller ETA.
    """
    # min_ships: smallest s where fleet hits at step eta
    if not _fleet_hits_at_step(_MAX_SHIPS_SEARCH, src_x, src_y, r_src, df_s, dst_id, eta, base_step):
        return None, None  # even max ships can't reach in eta steps
    if _fleet_hits_at_step(1, src_x, src_y, r_src, df_s, dst_id, eta, base_step):
        min_ships = 1
    else:
        lo, hi = 1, _MAX_SHIPS_SEARCH
        while lo < hi - 1:
            mid = (lo + hi) // 2
            if _fleet_hits_at_step(mid, src_x, src_y, r_src, df_s, dst_id, eta, base_step):
                hi = mid
            else:
                lo = mid
        min_ships = hi

    # max_ships: largest s that does NOT arrive at step eta-1
    if eta == 1:
        max_ships = _MAX_SHIPS_SEARCH
    else:
        if _fleet_hits_at_step(1, src_x, src_y, r_src, df_s, dst_id, eta - 1, base_step):
            return None, None  # slowest fleet already arrives at eta-1; bucket empty
        if not _fleet_hits_at_step(_MAX_SHIPS_SEARCH, src_x, src_y, r_src, df_s, dst_id, eta - 1, base_step):
            max_ships = _MAX_SHIPS_SEARCH
        else:
            lo2, hi2 = 1, _MAX_SHIPS_SEARCH
            while lo2 < hi2 - 1:
                mid = (lo2 + hi2) // 2
                if _fleet_hits_at_step(mid, src_x, src_y, r_src, df_s, dst_id, eta - 1, base_step):
                    hi2 = mid
                else:
                    lo2 = mid
            max_ships = lo2  # last that does NOT hit at eta-1

    if min_ships > max_ships:
        return None, None
    return min_ships, max_ships


def enumerate_action_nodes(obs, df_s: pd.DataFrame, player_id: int = 0,
                            base_step: int = 0) -> list:
    """Return list of (src_id, dst_id, eta, min_ships, max_ships) for all
    feasible attack opportunities owned by player_id.

    One entry per valid ETA bucket (1..9) for each (src, dst) pair.
    Paths crossing the sun are skipped.
    """
    step0_rows = df_s[df_s['step'] == base_step].set_index('id')

    action_nodes = []
    owned_ids = [p[0] for p in obs.planets if p[1] == player_id]

    for src_id in owned_ids:
        if src_id not in step0_rows.index:
            continue
        src_row = step0_rows.loc[src_id]
        src_x, src_y, r_src = float(src_row['x']), float(src_row['y']), float(src_row['radius'])

        for dst_id, dst_row in step0_rows.iterrows():
            if dst_id == src_id:
                continue
            dst_x0, dst_y0 = float(dst_row['x']), float(dst_row['y'])
            if not _path_clears_sun(src_x, src_y, dst_x0, dst_y0):
                continue

            for eta in range(1, 10):  # 1..9
                min_s, max_s = _find_eta_range(src_x, src_y, r_src, df_s, dst_id, eta, base_step)
                if min_s is not None:
                    action_nodes.append((src_id, dst_id, eta, min_s, max_s))

    return action_nodes
