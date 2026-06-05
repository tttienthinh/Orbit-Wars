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
