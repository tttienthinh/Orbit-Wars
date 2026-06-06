import math, copy
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData

exec(open('90-Simulate10Next_Conqueror2_Supplier_prod_per_step.py').read(), globals())


class Obs:
    def __init__(self, planets, initial_planets=None, fleets=None,
                 next_fleet_id=100, comets=None, comet_planet_ids=None,
                 angular_velocity=0.0):
        self.planets          = [list(p) for p in planets]
        self.initial_planets  = [list(p) for p in (initial_planets if initial_planets is not None else planets)]
        self.fleets           = [list(f) for f in (fleets or [])]
        self.next_fleet_id    = next_fleet_id
        self.comets           = comets or []
        self.comet_planet_ids = comet_planet_ids or []
        self.angular_velocity = angular_velocity


_LOG1024 = math.log(1024.0)


def get_obs_dataframe(obs, step: int, num_agents: int = 2):
    sim = copy.deepcopy(obs)
    no_actions = [[] for _ in range(num_agents)]
    rows = []
    for i in range(GameConfig.NB_STEPS_SIM + 1):
        for p in sim.planets:
            pid, owner, x, y, radius, ships, production = p[0], p[1], p[2], p[3], p[4], p[5], p[6]
            r = math.hypot(x - GameConfig.CENTER, y - GameConfig.CENTER)
            if pid in sim.comet_planet_ids:
                nature = "comet"
            elif r + radius < GameConfig.ROTATION_RADIUS_LIMIT:
                nature = "moving"
            else:
                nature = "fix"
            rows.append({"step": step+i, "id": pid, "x": x, "y": y,
                         "radius": radius, "ships": ships,
                         "production": production, "owner": owner, "nature": nature})
        interpreter(sim, no_actions, step+i, num_agents)

    df_s = pd.DataFrame(rows).sort_values("step").reset_index(drop=True)

    prev_pos = (
        df_s[["id","step","x","y"]]
        .assign(step=lambda d: d["step"]+1)
        .rename(columns={"x":"x_prev","y":"y_prev"})
    )
    planet_disp = (
        df_s[["id","step","x","y"]]
        .merge(prev_pos, on=["id","step"], how="left")
        .assign(planet_disp=lambda d: np.sqrt(
            (d["x"]-d["x_prev"].fillna(d["x"]))**2 +
            (d["y"]-d["y_prev"].fillna(d["y"]))**2))
        [["id","step","planet_disp"]]
    )
    return df_s, planet_disp


def _path_clears_sun(x0, y0, x1, y1):
    return PhysicsEngine.point_to_segment_distance(
        (GameConfig.CENTER, GameConfig.CENTER), (x0, y0), (x1, y1)
    ) > GameConfig.SUN_RADIUS + GameConfig.PLANET_MARGIN
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


def label_action_nodes(action_nodes: list, heuristic_moves: list,
                        obs, df_s: pd.DataFrame, base_step: int = 0):
    """Label each action node from heuristic_moves.

    heuristic_moves: [[src_id, angle, ships_sent], ...] from StrategyPipeline._04_score_and_decide

    Returns:
        labels:        np.ndarray shape (N,) float32 — 1.0 if heuristic selected this action
        ships_targets: np.ndarray shape (N,) float32 — log(ships)/log(1024) normalised
    """
    n = len(action_nodes)
    labels = np.zeros(n, dtype=np.float32)
    ships_targets = np.array(
        [math.log(max(an[3], 1)) / _LOG1024 for an in action_nodes],
        dtype=np.float32,
    )  # default: log(min_ships) / log(1024)

    if not heuristic_moves:
        return labels, ships_targets

    step0 = df_s[df_s['step'] == base_step].set_index('id')

    for move in heuristic_moves:
        src_id, move_angle, ships_sent = int(move[0]), float(move[1]), int(move[2])

        # Find dst: planet with smallest angular difference from move_angle
        best_dst_id, best_adiff = None, float('inf')
        if src_id not in step0.index:
            continue
        src_row = step0.loc[src_id]
        sx, sy = float(src_row['x']), float(src_row['y'])

        for pid, row in step0.iterrows():
            if pid == src_id:
                continue
            a = math.atan2(float(row['y']) - sy, float(row['x']) - sx)
            diff = abs((a - move_angle + math.pi) % (2 * math.pi) - math.pi)
            if diff < best_adiff:
                best_adiff, best_dst_id = diff, pid

        if best_dst_id is None:
            continue

        # Find matching action node: same src, same dst, eta where min<=ships_sent<=max
        s_norm = math.log(max(ships_sent, 1)) / _LOG1024
        for k, (asrc, adst, aeta, amin, amax) in enumerate(action_nodes):
            if asrc == src_id and adst == best_dst_id and amin <= ships_sent <= amax:
                labels[k] = 1.0
                ships_targets[k] = s_norm
                break  # first matching eta bucket wins

    return labels, ships_targets


def build_hetero_data_97(obs, step: int = 0, player_id: int = 0):
    """Build HeteroData with action nodes labelled by heuristic.

    Returns (HeteroData, action_index) where action_index[k] = (src_id, dst_id, eta).
    """
    df_s, planet_disp = get_obs_dataframe(obs, step)

    # Heuristic labels via 90-Simulate StrategyPipeline
    # _02_get_all_opportunities produces angle/angle_min/angle_max required by _03
    pa   = StrategyPipeline._02_get_all_opportunities(df_s, planet_disp, player_id)
    safe = StrategyPipeline._03_filter_collision(pa)
    heuristic_moves = StrategyPipeline._04_score_and_decide(safe, player_id)

    # Action nodes
    action_nodes = enumerate_action_nodes(obs, df_s, player_id, base_step=step)
    action_index = [(an[0], an[1], an[2]) for an in action_nodes]

    labels, ships_targets = label_action_nodes(action_nodes, heuristic_moves, obs, df_s, base_step=step)

    # ── Planet node features (22-dim) ────────────────────────────────────────
    planets    = obs.planets
    planet_ids = [p[0] for p in planets]
    pid_to_idx = {pid: i for i, pid in enumerate(planet_ids)}

    ships_pivot = (
        df_s[df_s['step'].isin(range(step, step + 11))]
        .pivot_table(index='id', columns='step', values='ships', aggfunc='first')
        .reindex(index=planet_ids, columns=range(step, step + 11), fill_value=0)
    )
    nature_at0 = df_s[df_s['step'] == step].set_index('id')['nature']

    planet_feats = []
    for p in planets:
        pid, owner, x, y, radius, _, production = p
        nat = nature_at0.get(pid, 'fix')
        owner_oh = [0.0] * 5
        if owner == -1:
            owner_oh[0] = 1.0
        elif 0 <= owner <= 3:
            owner_oh[owner + 1] = 1.0
        ships_ts = ships_pivot.loc[pid].values.tolist()
        ships_feats = [math.log(max(min(float(s), 1024.0), 1.0)) / _LOG1024 for s in ships_ts]
        planet_feats.append([
            x / 100.0, y / 100.0,
            1.0 if nat == 'fix' else 0.0,
            1.0 if nat == 'moving' else 0.0,
            1.0 if nat == 'comet' else 0.0,
            production / 5.0,
        ] + owner_oh + ships_feats)

    # ── Master node features (6-dim) ─────────────────────────────────────────
    total_ships = sum(max(p[5], 0) for p in planets) or 1.0
    proportions = [sum(max(p[5], 0) for p in planets if p[1] == pid_) / total_ships
                   for pid_ in range(4)]
    master_feat = [step / 500.0, (obs.angular_velocity - 0.025) / (0.05 - 0.025)] + proportions

    # ── Action node features (3-dim) ─────────────────────────────────────────
    # an = (src_id, dst_id, eta, min_ships, max_ships)
    action_feats = [
        [math.log(max(an[3], 1)) / _LOG1024, math.log(max(an[4], 1)) / _LOG1024, an[2] / 9.0]
        for an in action_nodes
    ]

    # ── Assemble HeteroData ───────────────────────────────────────────────────
    data = HeteroData()
    data['master'].x = torch.tensor([master_feat], dtype=torch.float)
    data['planet'].x = torch.tensor(planet_feats, dtype=torch.float)

    n_planets = len(planet_ids)
    data['planet', 'to_master', 'master'].edge_index = torch.tensor(
        [list(range(n_planets)), [0] * n_planets], dtype=torch.long)
    data['master', 'to_planet', 'planet'].edge_index = torch.tensor(
        [[0] * n_planets, list(range(n_planets))], dtype=torch.long)

    if action_nodes:
        data['action'].x = torch.tensor(action_feats, dtype=torch.float)
        data['action'].y = torch.tensor(labels, dtype=torch.float)
        data['action'].ships_target = torch.tensor(ships_targets, dtype=torch.float)

        spawns_src   = [pid_to_idx[an[0]] for an in action_nodes]
        attacks_dst  = [pid_to_idx[an[1]] for an in action_nodes]
        n_actions    = len(action_nodes)
        action_range = list(range(n_actions))

        data['planet', 'spawns', 'action'].edge_index = torch.tensor(
            [spawns_src, action_range], dtype=torch.long)
        data['action', 'attacks', 'planet'].edge_index = torch.tensor(
            [action_range, attacks_dst], dtype=torch.long)
    else:
        data['action'].x = torch.zeros((0, 3), dtype=torch.float)
        data['action'].y = torch.zeros(0, dtype=torch.float)
        data['action'].ships_target = torch.zeros(0, dtype=torch.float)
        data['planet', 'spawns', 'action'].edge_index = torch.zeros((2, 0), dtype=torch.long)
        data['action', 'attacks', 'planet'].edge_index = torch.zeros((2, 0), dtype=torch.long)

    return data, action_index


def _place_planets_randomly(rng, n_owned, n_enemy, n_neutral, player_id=0):
    """Place planets on the board, avoiding sun and each other."""
    planets = []
    pid = 0

    def try_add(owner, attempts=500):
        nonlocal pid
        for _ in range(attempts):
            x = float(rng.uniform(10, 90))
            y = float(rng.uniform(10, 90))
            if math.hypot(x - 50, y - 50) < 15:
                continue
            prod = int(rng.integers(1, 6))
            radius = 1.0 + math.log(prod)
            if any(math.hypot(x - p[2], y - p[3]) < radius + p[4] + 1.0 for p in planets):
                continue
            ships = int(rng.integers(10, 101))
            planets.append([pid, owner, x, y, radius, ships, prod])
            pid += 1
            return True
        return False

    for _ in range(n_owned):
        try_add(player_id)
    for _ in range(n_enemy):
        try_add(1)
    for _ in range(n_neutral):
        try_add(-1)
    return planets


def generate_sample_97(seed: int, player_id: int = 0):
    """Returns (HeteroData, snapshots, action_index).

    snapshots: 11 dicts {step, planets, fleets} for make_animation.
    action_index: list[(src_id, dst_id, eta)] mapping action node k to game action.
    """
    rng = np.random.default_rng(seed)
    angular_velocity = float(rng.uniform(0.025, 0.05))

    n_total   = int(rng.integers(5, 21))
    n_owned   = int(rng.integers(2, min(5, n_total - 2) + 1))
    n_enemy   = int(rng.integers(1, min(3, n_total - n_owned - 1) + 1))
    n_neutral = n_total - n_owned - n_enemy

    planets = _place_planets_randomly(rng, n_owned, n_enemy, n_neutral, player_id)
    obs = Obs(planets=planets, angular_velocity=angular_velocity)

    # Snapshots for animation
    sim = copy.deepcopy(obs)
    snapshots = []
    for i in range(11):
        snapshots.append({
            'step':    i,
            'planets': [p[:] for p in sim.planets],
            'fleets':  [f[:] for f in sim.fleets],
        })
        interpreter(sim, [[], []], i)

    data, action_index = build_hetero_data_97(obs, step=0, player_id=player_id)
    return data, snapshots, action_index
