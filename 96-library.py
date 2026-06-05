import math
import copy
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData

# ── Configuration ─────────────────────────────────────────────────────────────
class GameConfig:
    CENTER = 50.0
    SUN_RADIUS = 10.0
    ROTATION_RADIUS_LIMIT = 50.0
    MAX_SPEED = 6.0
    NB_STEPS_SIM = 10
    PLANET_MARGIN = 0.1
    PLANET_MOVEMENT_SLACK = 3.0


CENTER = GameConfig.CENTER
SUN_RADIUS = GameConfig.SUN_RADIUS
ROTATION_RADIUS_LIMIT = GameConfig.ROTATION_RADIUS_LIMIT
BOARD_SIZE = 100.0
MAX_NB_STEP = 500

SHIPS_OPTIONS = [1, 2, 4, 8, 16]
_SHIP_TO_IDX  = {s: i for i, s in enumerate(SHIPS_OPTIONS)}


# ── Physics helpers ───────────────────────────────────────────────────────────
class PhysicsEngine:
    @staticmethod
    def distance(p1, p2):
        return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

    @staticmethod
    def point_to_segment_distance(p, v, w):
        """Minimum distance from point p to line segment v-w."""
        l2 = (v[0] - w[0]) ** 2 + (v[1] - w[1]) ** 2
        if l2 == 0.0:
            return PhysicsEngine.distance(p, v)
        t = max(
            0, min(1, ((p[0] - v[0]) * (w[0] - v[0]) + (p[1] - v[1]) * (w[1] - v[1])) / l2)
        )
        projection = (v[0] + t * (w[0] - v[0]), v[1] + t * (w[1] - v[1]))
        return PhysicsEngine.distance(p, projection)

    @staticmethod
    def swept_pair_hit(A, B, P0, P1, r):
        """True iff a fleet moving A->B and a planet moving P0->P1 come within r
        of each other for some t in [0, 1]."""
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

    @staticmethod
    def fleet_speed(ships):
        if ships <= 1:
            return 1.0
        ratio = math.log(ships) / math.log(1000.0)
        return 1.0 + (GameConfig.MAX_SPEED - 1.0) * max(0.0, min(1.0, ratio)) ** 1.5


# ── Obs ───────────────────────────────────────────────────────────────────────
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


# ── interpreter ───────────────────────────────────────────────────────────────
def interpreter(obs, actions, step, num_agents=2):
    obs0 = obs

    expired_comet_pids = []
    for group in obs0.comets:
        idx = group["path_index"]
        for i, pid in enumerate(group["planet_ids"]):
            if idx >= len(group["paths"][i]):
                expired_comet_pids.append(pid)
    if expired_comet_pids:
        expired_set = set(expired_comet_pids)
        obs0.planets = [p for p in obs0.planets if p[0] not in expired_set]
        obs0.initial_planets = [
            p for p in obs0.initial_planets if p[0] not in expired_set
        ]
        obs0.comet_planet_ids = [
            pid for pid in obs0.comet_planet_ids if pid not in expired_set
        ]
        for group in obs0.comets:
            group["planet_ids"] = [
                pid for pid in group["planet_ids"] if pid not in expired_set
            ]
        obs0.comets = [g for g in obs0.comets if g["planet_ids"]]

    def process_moves(player_id, action):
        if not action or not isinstance(action, list):
            return
        for move in action:
            if len(move) != 3:
                continue
            from_id, angle, ships = move
            ships = int(ships)
            from_planet = next((p for p in obs0.planets if p[0] == from_id), None)
            if from_planet and from_planet[1] == player_id:
                if from_planet[5] >= ships and ships > 0:
                    from_planet[5] -= ships
                    start_x = from_planet[2] + math.cos(angle) * (from_planet[4] + 0.1)
                    start_y = from_planet[3] + math.sin(angle) * (from_planet[4] + 0.1)
                    obs0.fleets.append([
                        obs0.next_fleet_id, player_id,
                        start_x, start_y, angle, from_id, ships,
                    ])
                    obs0.next_fleet_id += 1

    for i in range(num_agents):
        process_moves(i, actions[i])

    for planet in obs0.planets:
        if planet[1] != -1:
            planet[5] += planet[6]

    fleets_to_remove = []
    combat_lists = {p[0]: [] for p in obs0.planets}

    angular_velocity = obs0.angular_velocity
    comet_pid_set = set(obs0.comet_planet_ids)
    initial_by_id = {p[0]: p for p in obs0.initial_planets}

    planet_paths = {}
    for planet in obs0.planets:
        if planet[0] in comet_pid_set:
            continue
        p_old = (planet[2], planet[3])
        p_new = p_old
        initial_p = initial_by_id.get(planet[0])
        if initial_p is not None:
            dx_p = initial_p[2] - CENTER
            dy_p = initial_p[3] - CENTER
            r_p = math.sqrt(dx_p ** 2 + dy_p ** 2)
            if r_p + planet[4] < ROTATION_RADIUS_LIMIT:
                initial_angle = math.atan2(dy_p, dx_p)
                current_angle = initial_angle + angular_velocity * step
                p_new = (
                    CENTER + r_p * math.cos(current_angle),
                    CENTER + r_p * math.sin(current_angle),
                )
        planet_paths[planet[0]] = (p_old, p_new)

    for fleet in obs0.fleets:
        angle = fleet[4]
        ships = fleet[6]
        speed = PhysicsEngine.fleet_speed(ships)
        f_old = (fleet[2], fleet[3])
        fleet[2] += math.cos(angle) * speed
        fleet[3] += math.sin(angle) * speed
        f_new = (fleet[2], fleet[3])

        hit_planet = False
        for planet in obs0.planets:
            path = planet_paths.get(planet[0])
            if path is None:
                continue
            p_old, p_new = path
            if PhysicsEngine.swept_pair_hit(f_old, f_new, p_old, p_new, planet[4]):
                combat_lists[planet[0]].append(fleet)
                fleets_to_remove.append(fleet)
                hit_planet = True
                break
        if hit_planet:
            continue
        if not (0 <= fleet[2] <= BOARD_SIZE and 0 <= fleet[3] <= BOARD_SIZE):
            fleets_to_remove.append(fleet)
            continue
        if PhysicsEngine.point_to_segment_distance((CENTER, CENTER), f_old, f_new) < SUN_RADIUS:
            fleets_to_remove.append(fleet)
            continue

    for planet in obs0.planets:
        path = planet_paths.get(planet[0])
        if path is not None:
            planet[2], planet[3] = path[1]

    expired_comet_pids = []
    for group in obs0.comets:
        group["path_index"] += 1
        idx = group["path_index"]
        for i, pid in enumerate(group["planet_ids"]):
            planet = next((p for p in obs0.planets if p[0] == pid), None)
            if planet is None:
                continue
            p_path = group["paths"][i]
            if idx >= len(p_path):
                expired_comet_pids.append(pid)
            else:
                c_old = (planet[2], planet[3])
                planet[2] = p_path[idx][0]
                planet[3] = p_path[idx][1]
                if c_old[0] >= 0:
                    c_new = (planet[2], planet[3])
                    for fleet in obs0.fleets:
                        if fleet not in fleets_to_remove:
                            if PhysicsEngine.point_to_segment_distance((fleet[2], fleet[3]), c_old, c_new) < planet[4]:
                                combat_lists[planet[0]].append(fleet)
                                fleets_to_remove.append(fleet)

    if expired_comet_pids:
        expired_set = set(expired_comet_pids)
        obs0.planets = [p for p in obs0.planets if p[0] not in expired_set]
        obs0.initial_planets = [p for p in obs0.initial_planets if p[0] not in expired_set]
        obs0.comet_planet_ids = [pid for pid in obs0.comet_planet_ids if pid not in expired_set]
        for group in obs0.comets:
            group["planet_ids"] = [pid for pid in group["planet_ids"] if pid not in expired_set]
        obs0.comets = [g for g in obs0.comets if g["planet_ids"]]

    obs0.fleets = [f for f in obs0.fleets if f not in fleets_to_remove]

    for pid, planet_fleets in combat_lists.items():
        planet = next((p for p in obs0.planets if p[0] == pid), None)
        if not planet or not planet_fleets:
            continue
        player_ships = {}
        for fleet in planet_fleets:
            owner = fleet[1]
            player_ships[owner] = player_ships.get(owner, 0) + fleet[6]
        if not player_ships:
            continue
        sorted_players = sorted(player_ships.items(), key=lambda item: item[1], reverse=True)
        top_player, top_ships = sorted_players[0]
        if len(sorted_players) > 1:
            second_ships = sorted_players[1][1]
            survivor_ships = top_ships - second_ships
            if sorted_players[0][1] == sorted_players[1][1]:
                survivor_ships = 0
            survivor_owner = top_player if survivor_ships > 0 else -1
        else:
            survivor_owner = top_player
            survivor_ships = top_ships
        if survivor_ships > 0:
            if planet[1] == survivor_owner:
                planet[5] += survivor_ships
            else:
                planet[5] -= survivor_ships
                if planet[5] < 0:
                    planet[1] = survivor_owner
                    planet[5] = abs(planet[5])

    obs1 = {
        "planets": obs0.planets,
        "initial_planets": obs0.initial_planets,
        "fleets": obs0.fleets,
        "next_fleet_id": obs0.next_fleet_id,
        "comets": obs0.comets,
        "comet_planet_ids": obs0.comet_planet_ids,
    }

    terminated = False
    if step >= MAX_NB_STEP - 2:
        terminated = True
    alive_players = set()
    for p in obs0.planets:
        if p[1] != -1:
            alive_players.add(p[1])
    for f in obs0.fleets:
        alive_players.add(f[1])
    if len(alive_players) <= 1:
        terminated = True

    return obs1


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


def get_opportunities(df_s: pd.DataFrame, planet_disp: pd.DataFrame,
                      source_planet_ids: set) -> pd.DataFrame:
    step0 = int(df_s['step'].min())
    src_rows = df_s[(df_s['step'] == step0) & (df_s['id'].isin(source_planet_ids))]
    if src_rows.empty:
        return pd.DataFrame()

    mine_base = (
        src_rows
        .rename(columns={'id':'id_src','x':'x_src','y':'y_src','radius':'radius_src',
                         'ships':'ships_min','production':'production_src',
                         'nature':'nature_src','owner':'owner_src','step':'step_src'})
        [['id_src','x_src','y_src','radius_src','ships_min',
          'production_src','nature_src','owner_src','step_src']]
        .reset_index(drop=True)
    )

    # Phase A: cross join src x all (dst, step) pairs
    coarse = (
        mine_base.assign(_key=1)
        .merge(df_s.assign(_key=1), on='_key').drop(columns='_key')
        .loc[lambda d: (d['step'] > d['step_src']) & (d['id'] != d['id_src'])]
        .merge(planet_disp, on=['id','step'], how='left')
        .reset_index(drop=True)
        .assign(
            dist_tgt_src=lambda d: np.sqrt((d['x']-d['x_src'])**2+(d['y']-d['y_src'])**2),
            step_diff=lambda d: (d['step']-d['step_src']).astype(float),
        )
    )

    # Sun-crossing filter
    _dx = coarse['x'].values - coarse['x_src'].values
    _dy = coarse['y'].values - coarse['y_src'].values
    _l2 = _dx**2 + _dy**2
    _dot = (GameConfig.CENTER-coarse['x_src'].values)*_dx + (GameConfig.CENTER-coarse['y_src'].values)*_dy
    _t   = np.clip(_dot / np.where(_l2==0, 1.0, _l2), 0.0, 1.0)
    _proj = np.sqrt(
        (GameConfig.CENTER-coarse['x_src'].values - _t*_dx)**2 +
        (GameConfig.CENTER-coarse['y_src'].values - _t*_dy)**2)
    _sun_dist = np.where(_l2==0,
        np.sqrt((GameConfig.CENTER-coarse['x_src'].values)**2+(GameConfig.CENTER-coarse['y_src'].values)**2),
        _proj)
    _crossing = _sun_dist < (GameConfig.SUN_RADIUS + GameConfig.PLANET_MARGIN)

    coarse = (
        coarse.assign(_cross=_crossing)
        .loc[lambda d:
            (d['dist_tgt_src'] < (d['step_diff']+1)*GameConfig.MAX_SPEED
             + d['radius_src'] + GameConfig.PLANET_MARGIN + d['radius']
             + d['planet_disp'].fillna(0.0))
            & ~d['_cross']]
        .drop(columns='_cross').reset_index(drop=True)
    )
    if coarse.empty:
        return pd.DataFrame()

    # Ships expansion: only SHIPS_OPTIONS
    expanded = (
        coarse.assign(_key=1)
        .merge(pd.DataFrame({'ships_sent': SHIPS_OPTIONS, '_key': 1}), on='_key')
        .drop(columns='_key').reset_index(drop=True)
    )

    # Phase B: fleet-speed filter
    _fs_ratio = np.clip(
        np.log(expanded['ships_sent'].values.astype(float)) / math.log(1000.0), 0, None)
    _speed    = 1.0 + (GameConfig.MAX_SPEED-1.0) * _fs_ratio**1.5
    _dist_min = expanded['step_diff'].values * _speed + GameConfig.PLANET_MARGIN + expanded['radius_src'].values
    _dist_prev= _dist_min - _speed

    prev_pos = (
        df_s[['id','step','x','y']]
        .assign(step=lambda d: d['step']+1)
        .rename(columns={'x':'x_prev','y':'y_prev'})
    )
    expanded = (
        expanded
        .assign(fleet_speed=_speed, dist_min=_dist_min, dist_prev=_dist_prev)
        .loc[lambda d: d['dist_tgt_src'] < d['dist_min']+d['fleet_speed']+d['radius']+GameConfig.PLANET_MOVEMENT_SLACK]
        .merge(prev_pos, on=['id','step'], how='left')
        .reset_index(drop=True)
    )
    if expanded.empty:
        return pd.DataFrame()

    # Swept-pair collision (vectorised)
    _dx2  = expanded['x'].values - expanded['x_src'].values
    _dy2  = expanded['y'].values - expanded['y_src'].values
    _d2   = expanded['dist_tgt_src'].values
    _ux   = _dx2 / np.where(_d2 < 1e-9, 1.0, _d2)
    _uy   = _dy2 / np.where(_d2 < 1e-9, 1.0, _d2)
    _xpf  = expanded['x_prev'].fillna(expanded['x']).values
    _ypf  = expanded['y_prev'].fillna(expanded['y']).values
    _fx0  = expanded['x_src'].values + _ux * expanded['dist_prev'].values
    _fy0  = expanded['y_src'].values + _uy * expanded['dist_prev'].values
    _pvx  = expanded['x'].values - _xpf
    _pvy  = expanded['y'].values - _ypf
    _dvx  = _ux*expanded['fleet_speed'].values - _pvx
    _dvy  = _uy*expanded['fleet_speed'].values - _pvy
    _d0x  = _fx0 - _xpf
    _d0y  = _fy0 - _ypf
    _a_   = _dvx**2 + _dvy**2
    _b_   = 2.0*(_d0x*_dvx + _d0y*_dvy)
    _c_   = _d0x**2 + _d0y**2 - expanded['radius'].values**2
    _disc = _b_**2 - 4.0*_a_*_c_
    _sq   = np.sqrt(np.clip(_disc, 0, None))
    _t1   = np.where(_a_ < 1e-12, 0.0, (-_b_-_sq)/(2.0*_a_))
    _t2   = np.where(_a_ < 1e-12, 1.0, (-_b_+_sq)/(2.0*_a_))
    _coll = np.where(_a_ < 1e-12, _c_ <= 0.0, (_disc >= 0.0) & (_t2 >= 0.0) & (_t1 <= 1.0))

    return (
        expanded.assign(collision=_coll)
        .loc[lambda d: d['collision']]
        .reset_index(drop=True)
    )


def build_edge_features(pa: pd.DataFrame) -> dict:
    """Returns dict[(src_id, dst_id) -> np.ndarray(50, float32)].

    feature[s_idx * 10 + (step_diff-1)] = 1.0
    where s_idx = index of ships_sent in SHIPS_OPTIONS.
    """
    result = {}
    if pa.empty:
        return result

    for (src_id, dst_id), grp in pa.groupby(['id_src', 'id']):
        feat = np.zeros(50, dtype=np.float32)
        for _, row in grp.iterrows():
            s_idx = _SHIP_TO_IDX.get(int(row['ships_sent']))
            if s_idx is None:
                continue
            e_idx = int(row['step_diff']) - 1
            if 0 <= e_idx < 10:
                feat[s_idx * 10 + e_idx] = 1.0
        result[(int(src_id), int(dst_id))] = feat

    return result


_LOG1024 = math.log(1024)


def build_hetero_data(obs, step: int, label: int, player_id: int = 0) -> HeteroData:
    """Convert an Obs into a PyG HeteroData graph with label."""
    # simulate snapshots
    df_s, planet_disp = get_obs_dataframe(obs, step)

    # attack opportunities from player's planets
    source_ids = {p[0] for p in obs.planets if p[1] == player_id}
    pa = get_opportunities(df_s, planet_disp, source_ids)
    edge_feats = build_edge_features(pa)

    # planet node features (22-dim)
    planets    = obs.planets
    planet_ids = [p[0] for p in planets]
    pid_to_idx = {pid: i for i, pid in enumerate(planet_ids)}

    # ships time-series per planet
    ships_pivot = (
        df_s[df_s['step'].isin(range(step, step+11))]
        .pivot_table(index='id', columns='step', values='ships', aggfunc='first')
        .reindex(index=planet_ids, columns=range(step, step+11), fill_value=0)
    )

    # nature at step0
    nature_at0 = df_s[df_s['step'] == step].set_index('id')['nature']

    planet_feats = []
    for p in planets:
        pid, owner, x, y, radius, ships0, production = p
        nat = nature_at0.get(pid, 'fix')
        is_fix    = 1.0 if nat == 'fix'    else 0.0
        is_moving = 1.0 if nat == 'moving' else 0.0
        is_comet  = 1.0 if nat == 'comet'  else 0.0

        # owner one-hot: index 0 = neutral(-1), 1..4 = players 0..3
        owner_oh = [0.0] * 5
        if owner == -1:
            owner_oh[0] = 1.0
        elif 0 <= owner <= 3:
            owner_oh[owner + 1] = 1.0

        # log-ships time-series (11 values)
        ships_ts = ships_pivot.loc[pid].values.tolist()
        ships_feats = [math.log(max(min(float(s), 1024.0), 1.0)) / _LOG1024
                       for s in ships_ts]

        planet_feats.append([x/100.0, y/100.0, is_fix, is_moving, is_comet,
                              production/5.0] + owner_oh + ships_feats)

    # master node features (6-dim)
    total_ships = sum(max(p[5], 0) for p in planets) or 1.0
    proportions = [
        sum(p[5] for p in planets if p[1] == pid_) / total_ships
        for pid_ in range(4)
    ]
    master_feat = [
        step / 500.0,
        (obs.angular_velocity - 0.025) / (0.05 - 0.025),
    ] + proportions

    # assemble HeteroData
    data = HeteroData()
    data['master'].x = torch.tensor([master_feat], dtype=torch.float)
    data['planet'].x = torch.tensor(planet_feats, dtype=torch.float)

    n = len(planet_ids)
    data['planet', 'to_master', 'master'].edge_index = torch.tensor(
        [list(range(n)), [0]*n], dtype=torch.long)
    data['master', 'to_planet', 'planet'].edge_index = torch.tensor(
        [[0]*n, list(range(n))], dtype=torch.long)

    if edge_feats:
        srcs  = [pid_to_idx[s] for s, _ in edge_feats]
        dsts  = [pid_to_idx[d] for _, d in edge_feats]
        attrs = np.stack(list(edge_feats.values()))
        data['planet', 'attacks', 'planet'].edge_index = torch.tensor(
            [srcs, dsts], dtype=torch.long)
        data['planet', 'attacks', 'planet'].edge_attr = torch.tensor(
            attrs, dtype=torch.float)
    else:
        data['planet', 'attacks', 'planet'].edge_index = torch.zeros(
            (2, 0), dtype=torch.long)
        data['planet', 'attacks', 'planet'].edge_attr = torch.zeros(
            (0, 50), dtype=torch.float)

    data.y = torch.tensor([float(label)], dtype=torch.float)
    return data


def generate_sample(seed: int, player_id: int = 0):
    """Returns (HeteroData, snapshots).

    snapshots is a list of 11 dicts {step, planets, fleets} for make_animation.
    label = 1 if my_planet.ships > neutral_planet.ships at step 0.
    """
    rng = np.random.default_rng(seed)
    angular_velocity = float(rng.uniform(0.025, 0.05))

    # Place my planet — reject if too close to sun at (50,50)
    while True:
        x0 = float(rng.uniform(10, 90))
        y0 = float(rng.uniform(10, 90))
        if math.hypot(x0 - 50, y0 - 50) >= 15:
            break
    prod0   = int(rng.integers(1, 6))
    ships0  = int(rng.integers(10, 101))
    radius0 = 1.0 + math.log(prod0)

    # Place neutral planet within 5-20 units, reject if too close to sun
    while True:
        angle = float(rng.uniform(0, 2 * math.pi))
        dist  = float(rng.uniform(5, 20))
        x1 = x0 + dist * math.cos(angle)
        y1 = y0 + dist * math.sin(angle)
        if (10 <= x1 <= 90 and 10 <= y1 <= 90
                and math.hypot(x1 - 50, y1 - 50) >= 15):
            break
    prod1   = int(rng.integers(1, 6))
    ships1  = int(rng.integers(10, 101))
    radius1 = 1.0 + math.log(prod1)

    planets = [
        [0, player_id, x0, y0, radius0, ships0, prod0],
        [1,         -1, x1, y1, radius1, ships1, prod1],
    ]
    obs = Obs(planets=planets, angular_velocity=angular_velocity)

    label = 1 if ships0 > ships1 else 0

    # Collect snapshots for animation (11 steps, no actions)
    sim = copy.deepcopy(obs)
    snapshots = []
    for i in range(11):
        snapshots.append({
            'step':    i,
            'planets': [p[:] for p in sim.planets],
            'fleets':  [f[:] for f in sim.fleets],
        })
        interpreter(sim, [[], []], i)

    data = build_hetero_data(obs, step=0, label=label, player_id=player_id)
    return data, snapshots
