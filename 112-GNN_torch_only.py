"""GNN agent — behavioral cloning of the Polars heuristic.

Pure PyTorch — no torch_geometric dependency.
Loads weights from After50Games.pt via key remapping from the PyG state dict format.
No _04_get_selected: attack selection is done entirely by the GNN.
"""
import math
import copy
import polars as pl
import numpy as np
from pathlib import Path
import torch
import torch.nn.functional as F
from torch import nn


# ── Configuration ─────────────────────────────────────────────────────────────
class GameConfig:
    CENTER = 50.0
    SUN_RADIUS = 10.0
    ROTATION_RADIUS_LIMIT = 50.0
    MAX_SPEED = 6.0
    NB_STEPS_SIM = 20
    PLANET_MARGIN = 0.1
    PLANET_MOVEMENT_SLACK = 3.0


# ── Physics helpers ───────────────────────────────────────────────────────────
class PhysicsEngine:
    @staticmethod
    def distance(p1, p2):
        return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

    @staticmethod
    def point_to_segment_distance(p, v, w):
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


def _interpreter(obs, actions, step, num_agents=2):
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
        obs0.initial_planets = [p for p in obs0.initial_planets if p[0] not in expired_set]
        obs0.comet_planet_ids = [pid for pid in obs0.comet_planet_ids if pid not in expired_set]
        for group in obs0.comets:
            group["planet_ids"] = [pid for pid in group["planet_ids"] if pid not in expired_set]
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
                            if PhysicsEngine.point_to_segment_distance(
                                (fleet[2], fleet[3]), c_old, c_new
                            ) < planet[4]:
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

    obs0.fleets = [f for f in obs0.fleets if f not in fleets_to_remove]

    alive_players = set()
    for p in obs0.planets:
        if p[1] != -1:
            alive_players.add(p[1])
    for f in obs0.fleets:
        alive_players.add(f[1])


# ── Graph data container (no torch_geometric) ─────────────────────────────────
class GraphData:
    def __init__(self):
        self._nodes = {}
        self._edges = {}
        self.reaches_attr = None

    @property
    def node_types(self):
        return [k for k, v in self._nodes.items() if v.numel() > 0]

    def __getitem__(self, key):
        if isinstance(key, str):
            return self._nodes.get(key, torch.zeros(0))
        return self._edges.get(key, torch.zeros(2, 0, dtype=torch.long))

    def __setitem__(self, key, val):
        if isinstance(key, str):
            self._nodes[key] = val
        else:
            self._edges[key] = val

    @property
    def edge_index_dict(self):
        return self._edges


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
            _interpreter(sim, no_actions, step + i, num_agents)

        df_s = pl.DataFrame(rows).sort("step")

        prev_pos = (
            df_s.lazy()
            .select(["id", "step", "x", "y"])
            .rename({"x": "x_prev", "y": "y_prev"})
            .with_columns((pl.col("step") + 1).alias("step"))
        )
        planet_disp = (
            df_s.lazy()
            .select(["id", "step", "x", "y"])
            .join(prev_pos, on=["id", "step"], how="left")
            .with_columns(
                ((pl.col("x") - pl.col("x_prev").fill_null(pl.col("x"))).pow(2) +
                 (pl.col("y") - pl.col("y_prev").fill_null(pl.col("y"))).pow(2)
                 ).sqrt().alias("planet_disp")
            )
            .select(["id", "step", "planet_disp"])
            .collect()
        )
        return df_s, planet_disp

    @staticmethod
    def _00_remap_owner(df_s: pl.DataFrame, obs, player_id: int) -> pl.DataFrame:
        ships_by_player: dict = {}
        for p in obs.planets:
            if p[1] != -1:
                ships_by_player[p[1]] = ships_by_player.get(p[1], 0) + p[5]
        for f in obs.fleets:
            ships_by_player[f[1]] = ships_by_player.get(f[1], 0) + f[6]
        opponents = sorted(
            [(pid, s) for pid, s in ships_by_player.items() if pid != player_id],
            key=lambda x: x[1], reverse=True,
        )
        id_map = {player_id: 0}
        for new_id, (old_id, _) in enumerate(opponents, start=1):
            id_map[old_id] = new_id
        return df_s.with_columns(
            pl.col("owner").map_elements(lambda x: id_map.get(x, x), return_dtype=pl.Int64)
        )

    @staticmethod
    def _02_get_all_opportunities(
        df_s: pl.DataFrame,
        planet_disp: pl.DataFrame,
        player_id: int,
    ) -> pl.LazyFrame:
        df_s_lf = df_s.lazy()
        planet_disp_lf = planet_disp.lazy()
        nb_steps_sim = GameConfig.NB_STEPS_SIM

        mine_base_lf = (
            df_s_lf
            .with_columns(
                pl.when(pl.col("owner") == player_id).then(1).otherwise(0).alias("is_mine")
            )
            .group_by("id", maintain_order=True)
            .agg([
                pl.first("step").alias("step_src"),
                pl.first("x").alias("x_src"),
                pl.first("y").alias("y_src"),
                pl.first("radius").alias("radius_src"),
                pl.min("ships").alias("ships_min"),
                pl.first("production").alias("production_src"),
                pl.first("nature").alias("nature_src"),
                pl.first("owner").alias("owner_src"),
                pl.len().alias("row_count"),
                pl.sum("is_mine").alias("is_mine"),
            ])
            .filter(
                (pl.col("row_count") == pl.col("is_mine")) & (pl.col("owner_src") == player_id)
            )
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
            mine_base_lf
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
            .with_columns(
                pl.int_ranges(
                    1,
                    pl.col("ships_min") + pl.col("production_src") * nb_steps_sim + 1,
                    dtype=pl.Int64,
                ).alias("ships_sent")
            )
            .explode("ships_sent")
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

        pa_lf = (
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

        return pa_lf

    @staticmethod
    def _03_filter_collision(pa_lf: pl.LazyFrame) -> pl.LazyFrame:
        angle_norm = pl.col("angle") % (2 * math.pi)
        wraps = pl.col("angle_min_obs") > pl.col("angle_max_obs")
        in_cone = pl.when(wraps).then(
            (angle_norm >= pl.col("angle_min_obs")) | (angle_norm <= pl.col("angle_max_obs"))
        ).otherwise(
            (angle_norm >= pl.col("angle_min_obs")) & (angle_norm <= pl.col("angle_max_obs"))
        )

        blocked_lf = (
            pa_lf.select(["id_src", "ships_sent", "step", "id", "angle", "angle_min", "angle_max"])
            .join(
                pa_lf.select(["id_src", "ships_sent", "step", "id", "angle_min", "angle_max"])
                     .rename({"step": "step_obs", "id": "id_obs",
                              "angle_min": "angle_min_obs", "angle_max": "angle_max_obs"}),
                on=["id_src", "ships_sent"],
                how="inner",
            )
            .filter(
                (pl.col("step_obs") < pl.col("step")) & (pl.col("id_obs") != pl.col("id"))
            )
            .filter(in_cone)
            .select(["id_src", "ships_sent", "step", "id"])
            .unique()
        )

        return (
            pa_lf
            .join(blocked_lf, on=["id_src", "ships_sent", "step", "id"], how="anti")
            .with_columns(pl.col("angle").alias("final_angle"))
        )

    @staticmethod
    def _05_get_GNN(
        df_s: pl.DataFrame,
        pa: pl.DataFrame,
        safe_attacks: pl.DataFrame,
    ) -> GraphData:
        planets_df = (
            df_s.select(["id", "production", "nature"])
            .unique(subset=["id"])
            .sort("id")
        )
        nature_arr = planets_df["nature"].to_numpy()
        planet_feat = np.column_stack([
            (nature_arr == "fix").astype(np.float32),
            (nature_arr == "moving").astype(np.float32),
            (nature_arr == "comet").astype(np.float32),
            planets_df["production"].to_numpy().astype(np.float32) / 5.0,
        ])
        planet_x = torch.tensor(planet_feat, dtype=torch.float32)

        ps_df = df_s.select(["id", "step", "x", "y", "ships", "owner"])
        owner_arr = ps_df["owner"].to_numpy()
        ps_feat = np.column_stack([
            ps_df["step"].to_numpy().astype(np.float32) / GameConfig.NB_STEPS_SIM,
            ps_df["x"].to_numpy().astype(np.float32) / 100.0,
            ps_df["y"].to_numpy().astype(np.float32) / 100.0,
            np.log(np.clip(ps_df["ships"].to_numpy().astype(np.float32), 1, None)) / np.log(1024),
            (owner_arr == -1).astype(np.float32),
            (owner_arr == 0).astype(np.float32),
            (owner_arr == 1).astype(np.float32),
            (owner_arr == 2).astype(np.float32),
            (owner_arr == 3).astype(np.float32),
        ])
        planet_step_x = torch.tensor(ps_feat, dtype=torch.float32)

        planet_id_to_idx = {pid: i for i, pid in enumerate(planets_df["id"].to_list())}
        src_snap = torch.tensor(
            [planet_id_to_idx[pid] for pid in ps_df["id"].to_list()], dtype=torch.long
        )
        dst_snap = torch.arange(len(ps_df), dtype=torch.long)

        ps_idx = ps_df.with_row_index("ps_idx").select(["id", "step", "ps_idx"])

        data = GraphData()
        data["planet"] = planet_x
        data["planet_step"] = planet_step_x
        # has_snapshot edges (bidirectional via ToUndirected equivalent)
        data[("planet", "has_snapshot", "planet_step")] = torch.stack([src_snap, dst_snap])
        data[("planet_step", "rev_has_snapshot", "planet")] = torch.stack([dst_snap, src_snap])

        if not pa.is_empty():
            reach_df = (
                pa.select(["id_src", "step_src", "id", "step", "ships_sent"])
                .join(
                    ps_idx.rename({"id": "id_src", "step": "step_src", "ps_idx": "src_r"}),
                    on=["id_src", "step_src"], how="inner",
                )
                .join(
                    ps_idx.rename({"ps_idx": "dst_r"}),
                    on=["id", "step"], how="inner",
                )
            )
            if not reach_df.is_empty():
                ships_arr = reach_df["ships_sent"].to_numpy().astype(float)
                src_r = reach_df["src_r"].to_numpy()
                dst_r = reach_df["dst_r"].to_numpy()
                # Make reaches undirected (same as ToUndirected on a self-loop edge type)
                ei_reaches = torch.tensor(
                    np.stack([
                        np.concatenate([src_r, dst_r]),
                        np.concatenate([dst_r, src_r]),
                    ]),
                    dtype=torch.long,
                )
                log_ships = (np.log(np.clip(ships_arr, 1, None)) / np.log(1024)).reshape(-1, 1)
                data[("planet_step", "reaches", "planet_step")] = ei_reaches
                data.reaches_attr = torch.tensor(
                    np.tile(log_ships, (2, 1)), dtype=torch.float32
                )

        attack_df = (
            safe_attacks.filter(pl.col("ships_sent") <= pl.col("ships_min"))
            .select(["id_src", "step_src", "step", "id", "ships_sent"])
        )
        if not attack_df.is_empty():
            atk_df = (
                attack_df
                .join(
                    ps_idx.rename({"id": "id_src", "step": "step_src", "ps_idx": "src_atk"}),
                    on=["id_src", "step_src"], how="inner",
                )
                .join(
                    ps_idx.rename({"ps_idx": "dst_tgt"}),
                    on=["id", "step"], how="inner",
                )
            )
            if not atk_df.is_empty():
                ships_arr = atk_df["ships_sent"].to_numpy().astype(float)
                n_atk = len(atk_df)
                data["attack"] = torch.tensor(
                    (np.log(np.clip(ships_arr, 1, None)) / np.log(1024)).reshape(-1, 1),
                    dtype=torch.float32,
                )
                src_atk = torch.tensor(atk_df["src_atk"].to_numpy(), dtype=torch.long)
                dst_tgt = torch.tensor(atk_df["dst_tgt"].to_numpy(), dtype=torch.long)
                arange_atk = torch.arange(n_atk, dtype=torch.long)
                data[("planet_step", "AttackSrc", "attack")] = torch.stack([src_atk, arange_atk])
                data[("attack", "rev_AttackSrc", "planet_step")] = torch.stack([arange_atk, src_atk])
                data[("attack", "AttackTgt", "planet_step")] = torch.stack([arange_atk, dst_tgt])
                data[("planet_step", "rev_AttackTgt", "attack")] = torch.stack([dst_tgt, arange_atk])

        return data


# ── Pure PyTorch GNN (matches PyG OrbitGNN parameter names via remapping) ─────

def _mean_agg(h_src: torch.Tensor, edge_index: torch.Tensor, num_dst: int) -> torch.Tensor:
    src_i, dst_i = edge_index
    D = h_src.shape[1]
    agg = h_src.new_zeros(num_dst, D)
    cnt = h_src.new_zeros(num_dst, 1)
    agg.scatter_add_(0, dst_i.unsqueeze(1).expand(-1, D), h_src[src_i])
    cnt.scatter_add_(0, dst_i.unsqueeze(1), h_src.new_ones(len(dst_i), 1))
    return agg / cnt.clamp(min=1)


class _SAGEConvPure(nn.Module):
    """SAGEConv: out[dst] = lin_l(h_dst) + lin_r(mean_agg(h_src))"""
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.lin_l = nn.Linear(in_dim, out_dim, bias=True)
        self.lin_r = nn.Linear(in_dim, out_dim, bias=False)

    def forward(
        self,
        h_src: torch.Tensor,
        h_dst: torch.Tensor,
        edge_index: torch.Tensor,
        num_dst: int,
    ) -> torch.Tensor:
        agg = _mean_agg(h_src, edge_index, num_dst)
        return self.lin_l(h_dst) + self.lin_r(agg)


class _GATConvPure(nn.Module):
    """GATConv(in, out, edge_dim=1, heads=1, add_self_loops=False) — same param names as PyG."""
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.lin_src = nn.Linear(in_dim, out_dim, bias=False)
        self.lin_dst = nn.Linear(in_dim, out_dim, bias=False)
        self.lin_edge = nn.Linear(1, out_dim, bias=False)
        self.att_src = nn.Parameter(torch.zeros(1, 1, out_dim))
        self.att_dst = nn.Parameter(torch.zeros(1, 1, out_dim))
        self.att_edge = nn.Parameter(torch.zeros(1, 1, out_dim))
        self.bias = nn.Parameter(torch.zeros(out_dim))

    def forward(
        self,
        h: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        num_nodes: int,
    ) -> torch.Tensor:
        src_i, dst_i = edge_index
        z_s = self.lin_src(h)           # [N, D]
        z_d = self.lin_dst(h)           # [N, D]
        z_e = self.lin_edge(edge_attr)  # [E, D]

        a_s = self.att_src.view(-1)
        a_d = self.att_dst.view(-1)
        a_e = self.att_edge.view(-1)

        alpha = (
            (z_s[src_i] * a_s).sum(-1)
            + (z_d[dst_i] * a_d).sum(-1)
            + (z_e * a_e).sum(-1)
        )
        alpha = F.leaky_relu(alpha, 0.2)

        # Per-destination softmax (global-max stabilization)
        alpha = (alpha - alpha.max().detach()).exp()
        alpha_sum = h.new_zeros(num_nodes)
        alpha_sum.scatter_add_(0, dst_i, alpha)
        alpha = alpha / (alpha_sum[dst_i] + 1e-16)

        D = z_s.shape[1]
        out = h.new_zeros(num_nodes, D)
        out.scatter_add_(
            0,
            dst_i.unsqueeze(1).expand(-1, D),
            alpha.unsqueeze(1) * z_s[src_i],
        )
        return out + self.bias


class _HeteroConvPure(nn.Module):
    """Replaces PyG HeteroConv(aggr='sum') — attribute names match after key remapping."""
    def __init__(self, d: int):
        super().__init__()
        self.p_ps = _SAGEConvPure(d, d)        # planet  → planet_step
        self.ps_p = _SAGEConvPure(d, d)        # planet_step → planet
        self.ps_ps_gat = _GATConvPure(d, d)    # planet_step → planet_step (reaches)
        self.ps_atk = _SAGEConvPure(d, d)      # planet_step → attack (AttackSrc)
        self.atk_ps_r = _SAGEConvPure(d, d)    # attack → planet_step (rev_AttackSrc)
        self.atk_ps_t = _SAGEConvPure(d, d)    # attack → planet_step (AttackTgt)
        self.ps_atk_r = _SAGEConvPure(d, d)    # planet_step → attack (rev_AttackTgt)

    def forward(
        self,
        x_dict: dict,
        ei_dict: dict,
        reaches_attr,
    ) -> dict:
        h_p = x_dict["planet"]
        h_ps = x_dict["planet_step"]
        h_atk = x_dict.get("attack")
        n_p, n_ps = h_p.shape[0], h_ps.shape[0]
        n_atk = h_atk.shape[0] if h_atk is not None else 0

        out_p = torch.zeros_like(h_p)
        out_ps = torch.zeros_like(h_ps)
        out_atk = torch.zeros_like(h_atk) if h_atk is not None else None

        ei = ei_dict.get(("planet", "has_snapshot", "planet_step"))
        if ei is not None and ei.shape[1] > 0:
            out_ps = out_ps + self.p_ps(h_p, h_ps, ei, n_ps)

        ei = ei_dict.get(("planet_step", "rev_has_snapshot", "planet"))
        if ei is not None and ei.shape[1] > 0:
            out_p = out_p + self.ps_p(h_ps, h_p, ei, n_p)

        ei = ei_dict.get(("planet_step", "reaches", "planet_step"))
        if ei is not None and ei.shape[1] > 0:
            ea = reaches_attr if reaches_attr is not None else torch.ones(ei.shape[1], 1)
            out_ps = out_ps + self.ps_ps_gat(h_ps, ei, ea, n_ps)

        if h_atk is not None and out_atk is not None:
            ei = ei_dict.get(("planet_step", "AttackSrc", "attack"))
            if ei is not None and ei.shape[1] > 0:
                out_atk = out_atk + self.ps_atk(h_ps, h_atk, ei, n_atk)

            ei = ei_dict.get(("attack", "rev_AttackSrc", "planet_step"))
            if ei is not None and ei.shape[1] > 0:
                out_ps = out_ps + self.atk_ps_r(h_atk, h_ps, ei, n_ps)

            ei = ei_dict.get(("attack", "AttackTgt", "planet_step"))
            if ei is not None and ei.shape[1] > 0:
                out_ps = out_ps + self.atk_ps_t(h_atk, h_ps, ei, n_ps)

            ei = ei_dict.get(("planet_step", "rev_AttackTgt", "attack"))
            if ei is not None and ei.shape[1] > 0:
                out_atk = out_atk + self.ps_atk_r(h_ps, h_atk, ei, n_atk)

        result = {"planet": out_p, "planet_step": out_ps}
        if out_atk is not None:
            result["attack"] = out_atk
        return result


class OrbitGNN(nn.Module):
    def __init__(self, hidden_dim: int = 16):
        super().__init__()
        d = hidden_dim
        self.planet_proj = nn.Linear(4, d, bias=False)
        self.planet_step_proj = nn.Linear(9, d, bias=False)
        self.attack_proj = nn.Linear(1, d, bias=False)
        self.conv1 = _HeteroConvPure(d)
        self.conv2 = _HeteroConvPure(d)
        self.head = nn.Sequential(nn.Linear(d, 8), nn.ReLU(), nn.Linear(8, 1))

    def forward(self, data: GraphData) -> torch.Tensor:
        planet_x = data["planet"]
        ps_x = data["planet_step"]
        atk_x = data["attack"]
        has_attack = "attack" in data.node_types and atk_x.numel() > 0

        x_dict = {
            "planet": self.planet_proj(planet_x),
            "planet_step": self.planet_step_proj(ps_x),
        }
        if has_attack:
            x_dict["attack"] = self.attack_proj(atk_x)

        ei_dict = data.edge_index_dict
        reaches_attr = data.reaches_attr

        for conv in (self.conv1, self.conv2):
            x_dict = conv(x_dict, ei_dict, reaches_attr)
            x_dict = {k: F.relu(v) for k, v in x_dict.items()}

        atk = x_dict.get("attack")
        if atk is None or atk.numel() == 0:
            return torch.zeros(0)
        return self.head(atk).squeeze(-1)


# ── Weight loading (remaps PyG HeteroConv key names to our attribute names) ───
_KEY_MAP = {
    "convs.<planet___has_snapshot___planet_step>": "p_ps",
    "convs.<planet_step___rev_has_snapshot___planet>": "ps_p",
    "convs.<planet_step___reaches___planet_step>": "ps_ps_gat",
    "convs.<planet_step___AttackSrc___attack>": "ps_atk",
    "convs.<attack___rev_AttackSrc___planet_step>": "atk_ps_r",
    "convs.<attack___AttackTgt___planet_step>": "atk_ps_t",
    "convs.<planet_step___rev_AttackTgt___attack>": "ps_atk_r",
}


def _load_pyg_weights(model: OrbitGNN, path) -> None:
    sd = torch.load(str(path), map_location="cpu", weights_only=True)
    new_sd = {}
    for k, v in sd.items():
        new_k = k
        for pyg_name, our_name in _KEY_MAP.items():
            new_k = new_k.replace(pyg_name, our_name)
        new_sd[new_k] = v
    model.load_state_dict(new_sd)


# ── Load model ────────────────────────────────────────────────────────────────
_here = Path(__file__).parent if "__file__" in dir() else Path(".")
_weights = _here / "After50Games.pt"

gnn_model = OrbitGNN(hidden_dim=16)
if _weights.exists():
    _load_pyg_weights(gnn_model, _weights)
gnn_model.eval()

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

    df_s, planet_disp = StrategyPipeline._01_get_obs_dataframe(obs, step, num_agents)
    df_s = StrategyPipeline._00_remap_owner(df_s, obs, player_id)

    pa_df = StrategyPipeline._02_get_all_opportunities(df_s, planet_disp, 0).collect()
    safe_df = StrategyPipeline._03_filter_collision(pa_df.lazy()).collect()

    data = StrategyPipeline._05_get_GNN(df_s, pa_df, safe_df)
    attack_df = safe_df.filter(pl.col("ships_sent") <= pl.col("ships_min"))

    if not attack_df.is_empty() and "attack" in data.node_types:
        with torch.no_grad():
            logits = gnn_model(data)
        mask_np = (logits.sigmoid() > 0.5).numpy().astype(bool)
        selected_attacks = attack_df.filter(pl.Series(values=mask_np, dtype=pl.Boolean))
        moves = [list(r) for r in selected_attacks.select(["id_src", "final_angle", "ships_sent"]).rows()]
    else:
        moves = []

    step += 1
    return moves
