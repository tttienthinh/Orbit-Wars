import math
import copy
import pandas as pd
import numpy as np


# ── Configuration ─────────────────────────────────────────────────────────────
class GameConfig:
    CENTER = 50.0
    SUN_RADIUS = 10.0
    ROTATION_RADIUS_LIMIT = 50.0
    MAX_SPEED = 6.0
    NB_STEPS_SIM = 10
    PLANET_MARGIN = 0.1
    PLANET_MOVEMENT_SLACK = 3.0


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


CENTER = GameConfig.CENTER
SUN_RADIUS = GameConfig.SUN_RADIUS
ROTATION_RADIUS_LIMIT = GameConfig.ROTATION_RADIUS_LIMIT

BOARD_SIZE = 100.0
MAX_NB_STEP = 500


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


# ── Strategy Pipeline ─────────────────────────────────────────────────────────
class StrategyPipeline:

    @staticmethod
    def _01_get_obs_dataframe(obs, step: int, num_agents: int) -> tuple:
        sim = copy.deepcopy(obs)
        no_actions = [[] for _ in range(num_agents)]
        rows = []
        for i in range(GameConfig.NB_STEPS_SIM + 1):
            for p in sim.planets:
                pid, owner, x, y, radius, ships, production = (
                    p[0], p[1], p[2], p[3], p[4], p[5], p[6]
                )
                r = math.hypot(x - GameConfig.CENTER, y - GameConfig.CENTER)
                if pid in sim.comet_planet_ids:
                    nature = "comet"
                elif r + radius < GameConfig.ROTATION_RADIUS_LIMIT:
                    nature = "moving"
                else:
                    nature = "fix"
                rows.append({
                    "step": step + i,
                    "id": pid,
                    "x": x,
                    "y": y,
                    "radius": radius,
                    "ships": ships,
                    "production": production,
                    "owner": owner,
                    "nature": nature,
                })
            interpreter(sim, no_actions, step + i, num_agents)

        df_s = pd.DataFrame(rows).sort_values("step").reset_index(drop=True)

        prev_pos = (
            df_s[["id", "step", "x", "y"]]
            .assign(step=lambda d: d["step"] + 1)
            .rename(columns={"x": "x_prev", "y": "y_prev"})
        )
        planet_disp = (
            df_s[["id", "step", "x", "y"]]
            .merge(prev_pos, on=["id", "step"], how="left")
            .assign(
                planet_disp=lambda d: np.sqrt(
                    (d["x"] - d["x_prev"].fillna(d["x"])) ** 2 +
                    (d["y"] - d["y_prev"].fillna(d["y"])) ** 2
                )
            )
            [["id", "step", "planet_disp"]]
        )
        return df_s, planet_disp

    @staticmethod
    def _sun_crossing_filter(coarse: pd.DataFrame) -> pd.DataFrame:
        _dx = coarse["x"].values - coarse["x_src"].values
        _dy = coarse["y"].values - coarse["y_src"].values
        _l2 = _dx ** 2 + _dy ** 2
        _dot = (
            (GameConfig.CENTER - coarse["x_src"].values) * _dx
            + (GameConfig.CENTER - coarse["y_src"].values) * _dy
        )
        _t_sun = np.clip(_dot / np.where(_l2 == 0, 1.0, _l2), 0.0, 1.0)
        _proj = np.sqrt(
            (GameConfig.CENTER - coarse["x_src"].values - _t_sun * _dx) ** 2
            + (GameConfig.CENTER - coarse["y_src"].values - _t_sun * _dy) ** 2
        )
        _sun_dist = np.where(
            _l2 == 0,
            np.sqrt(
                (GameConfig.CENTER - coarse["x_src"].values) ** 2
                + (GameConfig.CENTER - coarse["y_src"].values) ** 2
            ),
            _proj,
        )
        _crossing = _sun_dist < (GameConfig.SUN_RADIUS + GameConfig.PLANET_MARGIN)
        return coarse[~_crossing].reset_index(drop=True)

    @staticmethod
    def _02_pre_mine(df_s: pd.DataFrame, player_id: int) -> pd.DataFrame:
        nb_steps_sim = GameConfig.NB_STEPS_SIM
        mine_base = (
            df_s
            .assign(is_mine=(df_s["owner"] == player_id).astype(int))
            .groupby("id", sort=False)
            .agg(
                step_src=("step", "first"),
                x_src=("x", "first"),
                y_src=("y", "first"),
                radius_src=("radius", "first"),
                ships_min=("ships", "min"),
                production_src=("production", "first"),
                nature_src=("nature", "first"),
                owner_src=("owner", "first"),
                row_count=("id", "count"),
                is_mine=("is_mine", "sum"),
            )
            .reset_index()
            .loc[lambda d: (d["row_count"] == d["is_mine"]) & (d["owner_src"] == player_id)]
            .rename(columns={"id": "id_src"})
            .reset_index(drop=True)
        )

        if mine_base.empty:
            return pd.DataFrame()

        coarse = (
            mine_base.assign(_key=1)
            .merge(df_s.assign(_key=1), on="_key")
            .drop(columns="_key")
            .loc[lambda d: (d["step"] > d["step_src"]) & (d["id"] != d["id_src"])]
            .reset_index(drop=True)
            .assign(
                dist_tgt_src=lambda d: np.sqrt(
                    (d["x"] - d["x_src"]) ** 2 + (d["y"] - d["y_src"]) ** 2
                ),
                step_diff=lambda d: (d["step"] - d["step_src"]).astype(float),
            )
        )

        coarse = StrategyPipeline._sun_crossing_filter(coarse)

        if coarse.empty:
            return pd.DataFrame()

        coarse = coarse.assign(
            ships_sent=lambda d: [
                list(range(1, int(sm) + int(ps) * nb_steps_sim + 1))
                for sm, ps in zip(d["ships_min"], d["production_src"])
            ]
        )

        return coarse

    @staticmethod
    def _02_pre_all(df_s: pd.DataFrame, ships_list: list) -> pd.DataFrame:
        if df_s.empty:
            return pd.DataFrame()

        all_base = (
            df_s
            .groupby("id", sort=False)
            .agg(
                step_src=("step", "first"),
                x_src=("x", "first"),
                y_src=("y", "first"),
                radius_src=("radius", "first"),
                ships_min=("ships", "min"),
                production_src=("production", "first"),
                nature_src=("nature", "first"),
                owner_src=("owner", "first"),
            )
            .reset_index()
            .rename(columns={"id": "id_src"})
            .reset_index(drop=True)
        )

        coarse = (
            all_base.assign(_key=1)
            .merge(df_s.assign(_key=1), on="_key")
            .drop(columns="_key")
            .loc[lambda d: (d["step"] > d["step_src"]) & (d["id"] != d["id_src"])]
            .reset_index(drop=True)
            .assign(
                dist_tgt_src=lambda d: np.sqrt(
                    (d["x"] - d["x_src"]) ** 2 + (d["y"] - d["y_src"]) ** 2
                ),
                step_diff=lambda d: (d["step"] - d["step_src"]).astype(float),
            )
        )

        coarse = StrategyPipeline._sun_crossing_filter(coarse)

        if coarse.empty:
            return pd.DataFrame()

        coarse = coarse.assign(ships_sent=[list(ships_list) for _ in range(len(coarse))])

        return coarse

    @staticmethod
    def _02_get_all_opportunities(
        coarse: pd.DataFrame,
        df_s: pd.DataFrame,
        planet_disp: pd.DataFrame,
    ) -> pd.DataFrame:
        if coarse.empty:
            return pd.DataFrame()

        coarse = (
            coarse
            .merge(planet_disp, on=["id", "step"], how="left")
            .loc[lambda d:
                d["dist_tgt_src"] <
                (d["step_diff"] + 1) * GameConfig.MAX_SPEED
                + d["radius_src"] + GameConfig.PLANET_MARGIN + d["radius"]
                + d["planet_disp"].fillna(0.0)
                ]
            .reset_index(drop=True)
        )

        if coarse.empty:
            return pd.DataFrame()

        # Ships_sent expansion
        expanded = (
            coarse
            .explode("ships_sent")
            .assign(ships_sent=lambda d: d["ships_sent"].astype("int64"))
            .reset_index(drop=True)
        )

        # Phase B: fleet-speed filter
        _fs_ratio = np.clip(
            np.log(expanded["ships_sent"].values.astype(float)) / math.log(1000.0),
            0, None,
        )
        _fleet_speed_b = 1.0 + (GameConfig.MAX_SPEED - 1.0) * _fs_ratio ** 1.5
        _dist_min_b = expanded["step_diff"].values * _fleet_speed_b + GameConfig.PLANET_MARGIN + expanded["radius_src"].values
        _dist_prev_b = _dist_min_b - _fleet_speed_b

        prev_pos = (
            df_s[["id", "step", "x", "y"]]
            .assign(step=lambda d: d["step"] + 1)
            .rename(columns={"x": "x_prev", "y": "y_prev"})
        )

        expanded = (
            expanded
            .assign(fleet_speed=_fleet_speed_b, dist_min=_dist_min_b, dist_prev=_dist_prev_b)
            .loc[lambda d: d["dist_tgt_src"] < d["dist_min"] + d["fleet_speed"] + d["radius"] + GameConfig.PLANET_MOVEMENT_SLACK]
            .merge(prev_pos, on=["id", "step"], how="left")
            .reset_index(drop=True)
        )

        if expanded.empty:
            return pd.DataFrame()

        # Swept-pair collision (vectorised)
        _dx2 = expanded["x"].values - expanded["x_src"].values
        _dy2 = expanded["y"].values - expanded["y_src"].values
        _dist2 = expanded["dist_tgt_src"].values
        _ux = _dx2 / np.where(_dist2 < 1e-9, 1.0, _dist2)
        _uy = _dy2 / np.where(_dist2 < 1e-9, 1.0, _dist2)

        _xpf = expanded["x_prev"].fillna(expanded["x"]).values
        _ypf = expanded["y_prev"].fillna(expanded["y"]).values

        _fx0 = expanded["x_src"].values + _ux * expanded["dist_prev"].values
        _fy0 = expanded["y_src"].values + _uy * expanded["dist_prev"].values
        _pvx = expanded["x"].values - _xpf
        _pvy = expanded["y"].values - _ypf
        _dvx = _ux * expanded["fleet_speed"].values - _pvx
        _dvy = _uy * expanded["fleet_speed"].values - _pvy
        _d0x = _fx0 - _xpf
        _d0y = _fy0 - _ypf
        _a   = _dvx ** 2 + _dvy ** 2
        _b_  = 2.0 * (_d0x * _dvx + _d0y * _dvy)
        _c_  = _d0x ** 2 + _d0y ** 2 - expanded["radius"].values ** 2
        _disc = _b_ ** 2 - 4.0 * _a * _c_
        _sq  = np.sqrt(np.clip(_disc, 0, None))
        _t1  = np.where(_a < 1e-12, 0.0, (-_b_ - _sq) / (2.0 * _a))
        _t2  = np.where(_a < 1e-12, 1.0, (-_b_ + _sq) / (2.0 * _a))
        _coll = np.where(_a < 1e-12, _c_ <= 0.0, (_disc >= 0.0) & (_t2 >= 0.0) & (_t1 <= 1.0))

        pa = (
            expanded
            .assign(t1=_t1, t2=_t2, collision=_coll)
            .loc[lambda d: d["collision"]]
            .reset_index(drop=True)
        )

        if pa.empty:
            return pa

        # Angle geometry
        _xpf_pa = pa["x_prev"].fillna(pa["x"]).values
        _ypf_pa = pa["y_prev"].fillna(pa["y"]).values
        _t1e = np.clip(pa["t1"].values, 0.0, 1.0)
        _t2e = np.clip(pa["t2"].values, 0.0, 1.0)

        pa = (
            pa
            .assign(
                t1_eff=_t1e,
                t2_eff=_t2e,
                p_t1_x=_xpf_pa + _t1e * (pa["x"].values - _xpf_pa),
                p_t1_y=_ypf_pa + _t1e * (pa["y"].values - _ypf_pa),
                p_t2_x=_xpf_pa + _t2e * (pa["x"].values - _xpf_pa),
                p_t2_y=_ypf_pa + _t2e * (pa["y"].values - _ypf_pa),
            )
            .assign(
                angle_t1=lambda d: np.arctan2(d["p_t1_y"] - d["y_src"], d["p_t1_x"] - d["x_src"]),
                angle_t2=lambda d: np.arctan2(d["p_t2_y"] - d["y_src"], d["p_t2_x"] - d["x_src"]),
                d_s_t1=lambda d: np.sqrt((d["p_t1_x"] - d["x_src"]) ** 2 + (d["p_t1_y"] - d["y_src"]) ** 2),
                d_s_t2=lambda d: np.sqrt((d["p_t2_x"] - d["x_src"]) ** 2 + (d["p_t2_y"] - d["y_src"]) ** 2),
            )
            .assign(
                d_f_t1=lambda d: d["dist_prev"] + d["t1_eff"] * d["fleet_speed"],
                d_f_t2=lambda d: d["dist_prev"] + d["t2_eff"] * d["fleet_speed"],
            )
            .assign(
                angle_radius_t1=lambda d: np.arccos(np.clip(
                    (d["d_s_t1"] ** 2 + d["d_f_t1"] ** 2 - d["radius"] ** 2)
                    / (2.0 * d["d_s_t1"] * d["d_f_t1"]),
                    -1.0, 1.0,
                )),
                angle_radius_t2=lambda d: np.arccos(np.clip(
                    (d["d_s_t2"] ** 2 + d["d_f_t2"] ** 2 - d["radius"] ** 2)
                    / (2.0 * d["d_s_t2"] * d["d_f_t2"]),
                    -1.0, 1.0,
                )),
            )
            .assign(
                angle_min=lambda d: np.minimum(
                    d["angle_t1"] - d["angle_radius_t1"],
                    d["angle_t2"] - d["angle_radius_t2"],
                ) % (2 * math.pi),
                angle_max=lambda d: np.maximum(
                    d["angle_t1"] + d["angle_radius_t1"],
                    d["angle_t2"] + d["angle_radius_t2"],
                ) % (2 * math.pi),
                angle=lambda d: np.arctan2(
                    np.sin(d["angle_t1"]) + np.sin(d["angle_t2"]),
                    np.cos(d["angle_t1"]) + np.cos(d["angle_t2"]),
                ),
            )
            .sort_values("step")
            .reset_index(drop=True)
        )

        return pa

    @staticmethod
    def _03_filter_collision(pa: pd.DataFrame) -> pd.DataFrame:
        if pa.empty:
            return pa

        pa_left = pa[["id_src", "ships_sent", "step", "id", "angle", "angle_min", "angle_max"]].copy()
        pa_obs = (
            pa[["id_src", "ships_sent", "step", "id", "angle_min", "angle_max"]]
            .rename(columns={
                "step": "step_obs", "id": "id_obs",
                "angle_min": "angle_min_obs", "angle_max": "angle_max_obs",
            })
        )

        blocked_joined = (
            pa_left
            .merge(pa_obs, on=["id_src", "ships_sent"])
            .loc[lambda d: (d["step_obs"] < d["step"]) & (d["id_obs"] != d["id"])]
            .reset_index(drop=True)
        )

        if not blocked_joined.empty:
            _anorm = blocked_joined["angle"].values % (2 * math.pi)
            _wraps = (blocked_joined["angle_min_obs"] > blocked_joined["angle_max_obs"]).values
            _in_cone = np.where(
                _wraps,
                (_anorm >= blocked_joined["angle_min_obs"].values) | (_anorm <= blocked_joined["angle_max_obs"].values),
                (_anorm >= blocked_joined["angle_min_obs"].values) & (_anorm <= blocked_joined["angle_max_obs"].values),
            )
            blocked = (
                blocked_joined[_in_cone]
                [["id_src", "ships_sent", "step", "id"]]
                .drop_duplicates()
            )
        else:
            blocked = pd.DataFrame(columns=["id_src", "ships_sent", "step", "id"])

        attacks_with_angle = (
            pa
            .merge(blocked.assign(_blocked=True), on=["id_src", "ships_sent", "step", "id"], how="left")
            .loc[lambda d: d["_blocked"].isna()]
            .drop(columns="_blocked")
            .assign(final_angle=lambda d: d["angle"])
            .reset_index(drop=True)
        )

        return attacks_with_angle

    @staticmethod
    def _04_score_and_decide(
        attacks_with_angle: pd.DataFrame,
        reach_matrix: pd.DataFrame,  # reserved for threat-model scoring; unused in this iteration
        player_id: int,
    ) -> list:
        if attacks_with_angle.empty:
            return []

        moves = []

        # Comet evasion
        awa_comets = attacks_with_angle[attacks_with_angle["nature_src"] == "comet"]
        if not awa_comets.empty:
            x_off = (awa_comets["x_src"] - GameConfig.CENTER).abs().max() or 0
            y_off = (awa_comets["y_src"] - GameConfig.CENTER).abs().max() or 0
            if max(x_off, y_off) > 45:
                moves += (
                    awa_comets[awa_comets["ships_sent"] <= awa_comets["ships_min"]]
                    .sort_values(["ships_sent", "step"], ascending=[False, True])
                    .groupby("id_src", sort=False)
                    .first()
                    .reset_index()
                    [["id_src", "final_angle", "ships_sent"]]
                    .values.tolist()
                )
                id_to_avoid = awa_comets["id_src"].unique().tolist()
                attacks_with_angle = attacks_with_angle[~attacks_with_angle["id_src"].isin(id_to_avoid)]

        planet_id_top_5 = (
            attacks_with_angle
            .sort_values(["step", "ships_sent"])
            .groupby(["id_src", "id"], sort=False)
            .first()
            .reset_index()
            .sort_values(["step", "ships_sent"])
            .groupby("id_src", sort=False)
            .head(5)
            [["id_src", "id"]]
        )

        attacks_joined = (
            planet_id_top_5
            .merge(attacks_with_angle, on=["id_src", "id"], how="left")
            .loc[lambda d: d["owner"] != player_id]
            .assign(
                ships_needed=lambda d: np.where(d["owner"] == -1, d["ships"], d["ships"] + d["production"])
            )
            .loc[lambda d:
                (d["ships_needed"] + 1 <= d["ships_sent"]) &
                (d["ships_sent"] <= d["ships_needed"] + d["production_src"] + 1)
            ]
            .sort_values(["step", "ships_sent"])
            .groupby(["id_src", "id"], sort=False)
            .first()
            .reset_index()
            .assign(time_cost=lambda d: d["ships_needed"] / d["production_src"])
        )

        if attacks_joined.empty:
            return moves

        attacks_joined = attacks_joined.assign(
            total_time_cost=attacks_joined.groupby("id_src")["time_cost"].transform("sum")
        ).assign(
            score=lambda d: (d["total_time_cost"] - d["time_cost"] - d["step_diff"]) * d["production"]
        )

        attacks = (
            attacks_joined
            .sort_values("score", ascending=False)
            .groupby("id_src", sort=False)
            .first()
            .reset_index()
            .loc[lambda d: d["ships_sent"] <= d["ships_min"]]
        )

        moves += attacks[["id_src", "final_angle", "ships_sent"]].values.tolist()
        return moves


# ── Player-ID normalisation ───────────────────────────────────────────────────
def remap_player_ids(obs, my_player_id: int):
    """Return a deep-copied Obs with player IDs normalised.

    my_player_id → 0; remaining active players ranked 1-3 by descending total
    ships (planet garrison + in-flight fleets).  Planet/comet IDs (p[0]) are
    unchanged so action [from_planet_id, angle, ships] needs no reverse mapping.
    """
    obs = copy.deepcopy(obs)

    ships_by_player = {}
    for p in obs.planets:
        if p[1] != -1:
            ships_by_player[p[1]] = ships_by_player.get(p[1], 0) + p[5]
    for f in obs.fleets:
        ships_by_player[f[1]] = ships_by_player.get(f[1], 0) + f[6]

    opponents = sorted(
        [(pid, s) for pid, s in ships_by_player.items() if pid != my_player_id],
        key=lambda x: x[1],
        reverse=True,
    )
    id_map = {my_player_id: 0}
    for new_id, (old_id, _) in enumerate(opponents, start=1):
        id_map[old_id] = new_id

    for p in obs.planets:
        if p[1] != -1:
            p[1] = id_map.get(p[1], p[1])
    for p in obs.initial_planets:
        if p[1] != -1:
            p[1] = id_map.get(p[1], p[1])
    for f in obs.fleets:
        f[1] = id_map.get(f[1], f[1])

    return obs


# ── Entry point ───────────────────────────────────────────────────────────────
step = 0
num_agents = None
player_id = None


def agent(obs):
    global step, num_agents, player_id

    if num_agents is None:
        initial = (
            obs.initial_planets if hasattr(obs, "initial_planets")
            else obs["initial_planets"]
        )
        owners = {p[1] for p in initial if p[1] != -1}
        num_agents = 4 if len(owners) > 2 else 2
    if player_id is None:
        player_id = obs.get("player", 0) if isinstance(obs, dict) else obs.player

    obs = remap_player_ids(obs, player_id)

    df_s, planet_disp = StrategyPipeline._01_get_obs_dataframe(obs, step, num_agents)
    coarse_mine = StrategyPipeline._02_pre_mine(df_s, 0)
    pa          = StrategyPipeline._02_get_all_opportunities(coarse_mine, df_s, planet_disp)
    safe_attacks = StrategyPipeline._03_filter_collision(pa)
    reach        = pd.DataFrame()  # placeholder until _04 uses reach_matrix
    moves        = StrategyPipeline._04_score_and_decide(safe_attacks, reach, player_id=0)

    step += 1
    return moves
