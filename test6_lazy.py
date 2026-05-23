import kaggle_environments as ke  # must be imported BEFORE polars
import math, copy, random
import numpy as np
import pandas as pd
import polars as pl

# ── Constants ─────────────────────────────────────────────────────────────────
CENTER = 50.0
SUN_RADIUS = 10.0
ROTATION_RADIUS_LIMIT = 50.0
MAX_SPEED = 6.0
NB_STEPS_SIM = 10
PLANET_MARGIN = 0.1

# ── Obs class ─────────────────────────────────────────────────────────────────
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

# ── Interpreter ───────────────────────────────────────────────────────────────
from collections import namedtuple

Planet = namedtuple("Planet", ["id", "owner", "x", "y", "radius", "ships", "production"])
Fleet  = namedtuple("Fleet",  ["id", "owner", "x", "y", "angle", "from_planet_id", "ships"])

BOARD_SIZE          = 100.0
COMET_RADIUS        = 1.0
COMET_PRODUCTION    = 1
PLANET_CLEARANCE    = 7
MIN_PLANET_GROUPS   = 5
MAX_PLANET_GROUPS   = 10
MIN_STATIC_GROUPS   = 3
COMET_SPAWN_STEPS   = [50, 150, 250, 350, 450]
CENTER_X            = 50.0
CENTER_Y            = 50.0
MAX_NB_STEP         = 500


def distance(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def point_to_segment_distance(p, v, w):
    l2 = (v[0] - w[0]) ** 2 + (v[1] - w[1]) ** 2
    if l2 == 0.0:
        return distance(p, v)
    t = max(0, min(1, ((p[0] - v[0]) * (w[0] - v[0]) + (p[1] - v[1]) * (w[1] - v[1])) / l2))
    projection = (v[0] + t * (w[0] - v[0]), v[1] + t * (w[1] - v[1]))
    return distance(p, projection)


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
                    obs0.fleets.append([obs0.next_fleet_id, player_id, start_x, start_y, angle, from_id, ships])
                    obs0.next_fleet_id += 1

    for i in range(num_agents):
        process_moves(i, actions[i])

    for planet in obs0.planets:
        if planet[1] != -1:
            planet[5] += planet[6]

    max_speed = MAX_SPEED
    fleets_to_remove = []
    combat_lists = {p[0]: [] for p in obs0.planets}

    for fleet in obs0.fleets:
        angle  = fleet[4]
        ships  = fleet[6]
        speed  = 1.0 + (max_speed - 1.0) * (math.log(ships) / math.log(1000)) ** 1.5
        speed  = min(speed, max_speed)
        old_pos = (fleet[2], fleet[3])
        fleet[2] += math.cos(angle) * speed
        fleet[3] += math.sin(angle) * speed
        new_pos = (fleet[2], fleet[3])

        hit_planet = False
        for planet in obs0.planets:
            planet_pos = (planet[2], planet[3])
            if point_to_segment_distance(planet_pos, old_pos, new_pos) < planet[4]:
                combat_lists[planet[0]].append(fleet)
                fleets_to_remove.append(fleet)
                hit_planet = True
                break
        if hit_planet:
            continue
        if not (0 <= fleet[2] <= BOARD_SIZE and 0 <= fleet[3] <= BOARD_SIZE):
            fleets_to_remove.append(fleet)
            continue
        if point_to_segment_distance((CENTER, CENTER), old_pos, new_pos) < SUN_RADIUS:
            fleets_to_remove.append(fleet)
            continue

    angular_velocity = obs0.angular_velocity
    comet_pid_set    = set(obs0.comet_planet_ids)
    initial_by_id    = {p[0]: p for p in obs0.initial_planets}

    def sweep_fleets(planet, old_pos, new_pos):
        if old_pos == new_pos:
            return
        for fleet in obs0.fleets:
            if fleet not in fleets_to_remove:
                if point_to_segment_distance((fleet[2], fleet[3]), old_pos, new_pos) < planet[4]:
                    combat_lists[planet[0]].append(fleet)
                    fleets_to_remove.append(fleet)

    for planet in obs0.planets:
        if planet[0] in comet_pid_set:
            continue
        initial_p = initial_by_id.get(planet[0])
        if not initial_p:
            continue
        dx = initial_p[2] - CENTER
        dy = initial_p[3] - CENTER
        r  = math.sqrt(dx**2 + dy**2)
        old_pos = (planet[2], planet[3])
        if r + planet[4] < ROTATION_RADIUS_LIMIT:
            initial_angle  = math.atan2(dy, dx)
            current_angle  = initial_angle + angular_velocity * step
            planet[2]      = CENTER + r * math.cos(current_angle)
            planet[3]      = CENTER + r * math.sin(current_angle)
        sweep_fleets(planet, old_pos, (planet[2], planet[3]))

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
                old_pos   = (planet[2], planet[3])
                planet[2] = p_path[idx][0]
                planet[3] = p_path[idx][1]
                if old_pos[0] >= 0:
                    sweep_fleets(planet, old_pos, (planet[2], planet[3]))

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
        "planets":          obs0.planets,
        "initial_planets":  obs0.initial_planets,
        "fleets":           obs0.fleets,
        "next_fleet_id":    obs0.next_fleet_id,
        "comets":           obs0.comets,
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


# ── Physics helpers ────────────────────────────────────────────────────────────
def _fleet_speed(ships):
    if ships <= 1:
        return 1.0
    ratio = math.log(ships) / math.log(1000.0)
    return 1.0 + (MAX_SPEED - 1.0) * max(0.0, min(1.0, ratio)) ** 1.5


def _simulate(obs, global_step, num_agents, n_steps=NB_STEPS_SIM):
    sim = copy.deepcopy(obs)
    no_actions = [[] for _ in range(num_agents)]
    rows = []
    for i in range(n_steps + 1):
        for p in sim.planets:
            pid, owner, x, y, radius, ships, production = (
                p[0], p[1], p[2], p[3], p[4], p[5], p[6]
            )
            r = math.hypot(x - CENTER, y - CENTER)
            if pid in sim.comet_planet_ids:
                nature = "comet"
            elif r + radius < ROTATION_RADIUS_LIMIT:
                nature = "moving"
            else:
                nature = "fix"
            rows.append({
                "step":       global_step + i,
                "id":         pid,
                "x":          x,
                "y":          y,
                "radius":     radius,
                "ships":      ships,
                "production": production,
                "owner":      owner,
                "nature":     nature,
            })
        interpreter(sim, no_actions, global_step + i, num_agents)
    return pd.DataFrame(rows)


# ── IntervalProcessor (pandas) ────────────────────────────────────────────────
class IntervalProcessor:
    def merge_intervals(intervals):
        if not intervals:
            return []
        intervals = sorted(intervals, key=lambda x: x[0])
        merged = [intervals[0]]
        for current in intervals[1:]:
            prev_min, prev_max = merged[-1]
            curr_min, curr_max = current
            if curr_min <= prev_max:
                merged[-1] = (prev_min, max(prev_max, curr_max))
            else:
                merged.append(current)
        return merged

    def create_cumulative_obstacles(possible_attacks, min_step=0):
        max_step = int(possible_attacks["step"].max())
        attack_map = {}
        group_cols = ["id_src", "ships_sent", "step"]
        for (id_src, ship, step), group in possible_attacks.groupby(group_cols):
            unwrapped_intervals = []
            for _, row in group.iterrows():
                amin, amax = row["angle_min"], row["angle_max"]
                if amin > amax:
                    unwrapped_intervals.append((amin, 2 * np.pi))
                    unwrapped_intervals.append((0.0, amax))
                else:
                    unwrapped_intervals.append((amin, amax))
            attack_map[(id_src, ship, step)] = unwrapped_intervals

        unique_combinations = (
            possible_attacks[["id_src", "ships_sent"]].drop_duplicates().values
        )
        cumulative_records = []
        for id_src, ship in unique_combinations:
            current_intervals = []
            merged = []
            for step in range(min_step, max_step + 1):
                if (id_src, ship, step) in attack_map:
                    current_intervals.extend(attack_map[(id_src, ship, step)])
                    merged = IntervalProcessor.merge_intervals(current_intervals)
                cumulative_records.append({
                    "step":          step + 1,
                    "id_src":        id_src,
                    "ships_sent":    ship,
                    "obstacle_list": merged,
                })
        return pd.DataFrame(cumulative_records)

    def subtract_intervals(target_min, target_max, blocked_intervals):
        safe_intervals = [(target_min, target_max)]
        for b_min, b_max in blocked_intervals:
            next_safe = []
            for s_min, s_max in safe_intervals:
                if b_max <= s_min or b_min >= s_max:
                    next_safe.append((s_min, s_max))
                else:
                    if b_min > s_min:
                        next_safe.append((s_min, b_min))
                    if b_max < s_max:
                        next_safe.append((b_max, s_max))
            safe_intervals = next_safe
            if not safe_intervals:
                break
        return safe_intervals

    def compute_free_angles(row):
        amin = row["angle_min"]
        amax = row["angle_max"]
        obstacles = row["obstacle_list"]
        if not isinstance(obstacles, list) or len(obstacles) == 0:
            return [(amin, amax)]
        if amin > amax:
            targets = [(amin, 2 * np.pi), (0.0, amax)]
        else:
            targets = [(amin, amax)]
        all_free_intervals = []
        for t_min, t_max in targets:
            free_pieces = IntervalProcessor.subtract_intervals(t_min, t_max, obstacles)
            all_free_intervals.extend(free_pieces)
        has_end_piece   = any(abs(f[1] - 2 * np.pi) < 1e-9 for f in all_free_intervals)
        has_start_piece = any(abs(f[0] - 0.0) < 1e-9 for f in all_free_intervals)
        if has_end_piece and has_start_piece and len(all_free_intervals) > 1:
            end_idx   = next(i for i, f in enumerate(all_free_intervals) if abs(f[1] - 2 * np.pi) < 1e-9)
            start_idx = next(i for i, f in enumerate(all_free_intervals) if abs(f[0] - 0.0) < 1e-9)
            wrapped_interval = (all_free_intervals[end_idx][0], all_free_intervals[start_idx][1])
            indices_to_remove = {end_idx, start_idx}
            all_free_intervals = [f for i, f in enumerate(all_free_intervals) if i not in indices_to_remove]
            all_free_intervals.append(wrapped_interval)
        return all_free_intervals

    def interval_to_final_angle(angle_list_series):
        def process_row_intervals(intervals):
            if not isinstance(intervals, list) or len(intervals) == 0:
                return np.nan
            widest_span  = -1.0
            best_midpoint = np.nan
            for amin, amax in intervals:
                if amin <= amax:
                    span = amax - amin
                else:
                    span = (2 * np.pi - amin) + amax
                if span > widest_span:
                    widest_span = span
                    if amin <= amax:
                        midpoint = (amin + amax) / 2.0
                    else:
                        midpoint = amin + (span / 2.0)
                    best_midpoint = midpoint % (2 * np.pi)
            return best_midpoint
        return angle_list_series.map(process_row_intervals)


def take_action(df, player_id, nb_steps_sim=NB_STEPS_SIM, return_df=False):
    mine_across_sim = (
        df
        .assign(is_mine=lambda d: (d["owner"] == player_id).astype(int))
        .groupby("id")
        .agg(
            step_src=("step", "first"),
            x_src=("x", "first"),
            y_src=("y", "first"),
            radius_src=("radius", "first"),
            ships_min=("ships", "min"),
            production_src=("production", "first"),
            nature_src=("nature", "first"),
            owner_src=("owner", "first"),
            row_count=("ships", "size"),
            is_mine=("is_mine", "sum"),
        )
        .query("row_count == is_mine and owner_src==@player_id")
        .reset_index(drop=False)
        .rename(columns={"id": "id_src"})
    )

    expanded_mine = (
        mine_across_sim
        .assign(ships_sent=(mine_across_sim["ships_min"] + mine_across_sim["production_src"] * NB_STEPS_SIM).apply(lambda n: list(range(1, n + 1))))
        .explode("ships_sent")
        .astype({"ships_sent": int})
        .reset_index(drop=True)
    )

    df_src_tgt = (
        expanded_mine
        .merge(df, how="cross")
        .query("step > step_src and id != id_src")
    )

    possible_attacks = (
        df_src_tgt
        .assign(
            dist_tgt_src=lambda d: ((d["x"] - d["x_src"]) ** 2 + (d["y"] - d["y_src"]) ** 2) ** 0.5,
            step_diff=lambda d: d["step"] - d["step_src"],
            fleet_speed=lambda d: 1.0 + (MAX_SPEED - 1.0) * (np.log(d["ships_sent"]) / math.log(1000)) ** 1.5,
            dist_fleet_src_min=lambda d: d["step_diff"] * d["fleet_speed"] + PLANET_MARGIN + d["radius_src"],
            dist_fleet_src_max=lambda d: (d["step_diff"] + 1) * d["fleet_speed"] + PLANET_MARGIN + d["radius_src"],
            collision=lambda d: (
                ((d["dist_tgt_src"] - d["radius"] < d["dist_fleet_src_min"]) & (d["dist_fleet_src_min"] < d["dist_tgt_src"] + d["radius"])) |
                ((d["dist_tgt_src"] - d["radius"] < d["dist_fleet_src_max"]) & (d["dist_fleet_src_max"] < d["dist_tgt_src"] + d["radius"]))
            ),
        )
        .query("collision")
        .assign(
            crossing_sun=lambda d: d.apply(
                lambda row: point_to_segment_distance(
                    (CENTER, CENTER),
                    (row["x_src"], row["y_src"]),
                    (row["x"], row["y"]),
                ) < SUN_RADIUS + PLANET_MARGIN,
                axis=1,
            ).astype(bool)
        )
        .query("not crossing_sun")
        .assign(
            angle=lambda d: np.arctan2(d["y"] - d["y_src"], d["x"] - d["x_src"]),
            radius_angle=lambda d: np.maximum(
                np.arccos(((d["dist_tgt_src"]**2 + d["dist_fleet_src_min"]**2 - d["radius"]**2) / (2 * d["dist_tgt_src"] * d["dist_fleet_src_min"])).clip(-1, 1)),
                np.arccos(((d["dist_tgt_src"]**2 + d["dist_fleet_src_max"]**2 - d["radius"]**2) / (2 * d["dist_tgt_src"] * d["dist_fleet_src_max"])).clip(-1, 1))
            ),
            angle_min=lambda d: np.mod(d["angle"] - d["radius_angle"], 2 * math.pi),
            angle_max=lambda d: np.mod(d["angle"] + d["radius_angle"], 2 * math.pi),
        )
        .sort_values("step", ascending=True)
    )

    if possible_attacks.empty:
        if return_df:
            return [], possible_attacks
        return []

    df_obstacles = IntervalProcessor.create_cumulative_obstacles(possible_attacks)

    attacks_with_angle = (
        possible_attacks
        .merge(df_obstacles, how="left", on=["id_src", "step", "ships_sent"])
        .assign(angle_list=lambda d: d.apply(IntervalProcessor.compute_free_angles, axis=1))
        .query("angle_list.str.len() > 0")
    )

    planet_id_top_5_id_src = (
        attacks_with_angle
        .sort_values(["step", "ships_sent"], ascending=True)
        .groupby(["id_src", "id"], as_index=False)
        .first()
        .sort_values(["step", "ships_sent"], ascending=True)
        .groupby("id_src", as_index=False)
        .head(5)
        [["id_src", "id"]]
    )

    attacks_with_angle_comets = attacks_with_angle.query("nature_src == 'comet'")
    moves = []
    if not attacks_with_angle_comets.empty and max(
        max((attacks_with_angle_comets["x_src"] - CENTER).abs()),
        max((attacks_with_angle_comets["y_src"] - CENTER).abs())
    ) > 45:
        moves += (
            attacks_with_angle_comets
            .query("ships_sent <= ships_min")
            .sort_values(["ships_sent", "step"], ascending=[False, True])
            .groupby("id_src", as_index=False)
            .first()
            [["id_src", "angle", "ships_sent"]]
            .values
            .tolist()
        )
        id_to_avoid = attacks_with_angle_comets["id_src"].unique().tolist()
        attacks_with_angle = attacks_with_angle.query("id_src not in @id_to_avoid")

    attacks = (
        planet_id_top_5_id_src
        .merge(attacks_with_angle, how="left", on=["id_src", "id"])
        .query("@player_id != owner")
        .assign(
            ships_needed=lambda d: np.where(d["owner"] == -1, d["ships"], d["ships"] + d["production"])
        )
        .query("ships_needed + 1 <= ships_sent and ships_sent <= ships_needed + production_src + 1")
        .sort_values(["step", "ships_sent"], ascending=True)
        .groupby(["id_src", "id"], as_index=False)
        .first()
        .assign(
            time_cost=lambda d: d["ships_needed"] / d["production_src"],
            total_time_cost=lambda d: d.groupby("id_src")["time_cost"].transform("sum"),
            score=lambda d: (d["total_time_cost"] - d["time_cost"] - d["step_diff"]) * d["production"],
        )
        .sort_values("score", ascending=False)
        .groupby("id_src", as_index=False)
        .first()
        .query("ships_sent <= ships_min")
        .assign(final_angle=lambda d: IntervalProcessor.interval_to_final_angle(d["angle_list"]))
    )
    moves += attacks[["id_src", "final_angle", "ships_sent"]].values.tolist()
    if return_df:
        return moves, possible_attacks
    return moves


# ── IntervalProcessorPolars ───────────────────────────────────────────────────
class IntervalProcessorPolars:

    @staticmethod
    def merge_intervals(intervals):
        if not intervals:
            return []
        intervals = sorted(intervals, key=lambda x: x[0])
        merged = [list(intervals[0])]
        for current in intervals[1:]:
            prev_min, prev_max = merged[-1]
            curr_min, curr_max = current
            if curr_min <= prev_max:
                merged[-1] = [prev_min, max(prev_max, curr_max)]
            else:
                merged.append(list(current))
        return [tuple(x) for x in merged]

    @staticmethod
    def subtract_intervals(target_min, target_max, blocked_intervals):
        safe_intervals = [(target_min, target_max)]
        for b_min, b_max in blocked_intervals:
            next_safe = []
            for s_min, s_max in safe_intervals:
                if b_max <= s_min or b_min >= s_max:
                    next_safe.append((s_min, s_max))
                else:
                    if b_min > s_min:
                        next_safe.append((s_min, b_min))
                    if b_max < s_max:
                        next_safe.append((b_max, s_max))
            safe_intervals = next_safe
            if not safe_intervals:
                break
        return safe_intervals

    @staticmethod
    def create_cumulative_obstacles(possible_attacks: pl.DataFrame, min_step: int = 0) -> pl.DataFrame:
        max_step = int(possible_attacks["step"].max())
        attack_map = {}
        for row in possible_attacks.select(["id_src", "ships_sent", "step", "angle_min", "angle_max"]).to_dicts():
            key = (row["id_src"], row["ships_sent"], row["step"])
            amin, amax = row["angle_min"], row["angle_max"]
            if amin > amax:
                attack_map.setdefault(key, []).extend([(amin, 2 * np.pi), (0.0, amax)])
            else:
                attack_map.setdefault(key, []).append((amin, amax))

        unique_combinations = (
            possible_attacks.select(["id_src", "ships_sent"]).unique(maintain_order=True).to_numpy()
        )

        steps_col, id_srcs_col, ships_col, obstacles_col = [], [], [], []
        for id_src, ship in unique_combinations:
            current_intervals = []
            merged = []
            for step in range(min_step, max_step + 1):
                if (id_src, ship, step) in attack_map:
                    current_intervals.extend(attack_map[(id_src, ship, step)])
                    merged = IntervalProcessorPolars.merge_intervals(current_intervals)
                steps_col.append(step + 1)
                id_srcs_col.append(id_src)
                ships_col.append(ship)
                obstacles_col.append([[a, b] for a, b in merged])

        return pl.DataFrame(
            {"step": steps_col, "id_src": id_srcs_col, "ships_sent": ships_col, "obstacle_list": obstacles_col},
            schema={"step": pl.Int64, "id_src": pl.Int64, "ships_sent": pl.Int64,
                    "obstacle_list": pl.List(pl.List(pl.Float64))},
        )

    @staticmethod
    def compute_free_angles(row) -> list:
        if hasattr(row, "as_py"):
            row = row.as_py()
        amin     = row["angle_min"]
        amax     = row["angle_max"]
        raw_obs  = row["obstacle_list"]
        if hasattr(raw_obs, "to_list"):
            raw_obs = raw_obs.to_list()
        obstacles = raw_obs or []

        if not obstacles:
            return [[amin, amax]]

        targets = [(amin, 2 * np.pi), (0.0, amax)] if amin > amax else [(amin, amax)]
        all_free = []
        for t_min, t_max in targets:
            all_free.extend(IntervalProcessorPolars.subtract_intervals(t_min, t_max, obstacles))

        has_end   = any(abs(f[1] - 2 * np.pi) < 1e-9 for f in all_free)
        has_start = any(abs(f[0] - 0.0) < 1e-9 for f in all_free)

        if has_end and has_start and len(all_free) > 1:
            end_idx   = next(i for i, f in enumerate(all_free) if abs(f[1] - 2 * np.pi) < 1e-9)
            start_idx = next(i for i, f in enumerate(all_free) if abs(f[0] - 0.0) < 1e-9)
            wrapped   = (all_free[end_idx][0], all_free[start_idx][1])
            all_free  = [f for i, f in enumerate(all_free) if i not in {end_idx, start_idx}]
            all_free.append(wrapped)

        return [[a, b] for a, b in all_free]

    @staticmethod
    def interval_to_final_angle(series: pl.Series) -> pl.Series:
        def _best(intervals):
            if intervals is None:
                return float("nan")
            if hasattr(intervals, "to_list"):
                intervals = intervals.to_list()
            if not intervals:
                return float("nan")
            widest, best = -1.0, float("nan")
            for interval in intervals:
                amin, amax = interval[0], interval[1]
                span = (amax - amin) if amin <= amax else (2 * np.pi - amin + amax)
                if span > widest:
                    widest = span
                    mid  = (amin + amax) / 2.0 if amin <= amax else amin + span / 2.0
                    best = mid % (2 * np.pi)
            return best
        return series.map_elements(_best, return_dtype=pl.Float64)


# ── take_action_lazy ──────────────────────────────────────────────────────────
def take_action_lazy(df: pd.DataFrame, player_id: int,
                     nb_steps_sim: int = NB_STEPS_SIM,
                     return_df: bool = False):
    df_lf = pl.from_pandas(df).sort("step").lazy()

    mine_across_sim = (
        df_lf
        .with_columns(
            pl.when(pl.col("owner") == player_id).then(1).otherwise(0).alias("is_mine")
        )
        .group_by("id", maintain_order=True)
        .agg(
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
        )
        .filter(
            (pl.col("row_count") == pl.col("is_mine")) &
            (pl.col("owner_src") == player_id)
        )
        .rename({"id": "id_src"})
        .collect()
    )

    if mine_across_sim.is_empty():
        return ([], pl.DataFrame()) if return_df else []

    dx_vw = pl.col("x") - pl.col("x_src")
    dy_vw = pl.col("y") - pl.col("y_src")
    l2    = dx_vw.pow(2) + dy_vw.pow(2)
    dot   = (CENTER - pl.col("x_src")) * dx_vw + (CENTER - pl.col("y_src")) * dy_vw
    t     = (dot / pl.when(l2 == 0).then(pl.lit(1.0)).otherwise(l2)).clip(0.0, 1.0)
    dist_sun_proj   = ((CENTER - (pl.col("x_src") + t * dx_vw)).pow(2) +
                       (CENTER - (pl.col("y_src") + t * dy_vw)).pow(2)).sqrt()
    dist_sun_direct = ((CENTER - pl.col("x_src")).pow(2) +
                       (CENTER - pl.col("y_src")).pow(2)).sqrt()
    dist_to_sun     = pl.when(l2 == 0).then(dist_sun_direct).otherwise(dist_sun_proj)
    crossing_sun_expr = dist_to_sun < (SUN_RADIUS + PLANET_MARGIN)

    dist_tgt_src_expr = ((pl.col("x") - pl.col("x_src")).pow(2) +
                         (pl.col("y") - pl.col("y_src")).pow(2)).sqrt()
    step_diff_expr    = pl.col("step") - pl.col("step_src")
    fleet_speed_expr  = (
        1.0 + (MAX_SPEED - 1.0) *
        (pl.col("ships_sent").cast(pl.Float64).log(base=math.e) / math.log(1000.0)).pow(1.5)
    )
    dist_min_expr = step_diff_expr * fleet_speed_expr + PLANET_MARGIN + pl.col("radius_src")
    dist_max_expr = (step_diff_expr + 1) * fleet_speed_expr + PLANET_MARGIN + pl.col("radius_src")
    collision_expr = (
        ((dist_tgt_src_expr - pl.col("radius") < dist_min_expr) &
         (dist_min_expr < dist_tgt_src_expr + pl.col("radius"))) |
        ((dist_tgt_src_expr - pl.col("radius") < dist_max_expr) &
         (dist_max_expr < dist_tgt_src_expr + pl.col("radius")))
    )

    possible_attacks = (
        mine_across_sim.lazy()
        .with_columns(
            pl.int_ranges(
                1,
                pl.col("ships_min") + pl.col("production_src") * NB_STEPS_SIM + 1,
                dtype=pl.Int64,
            ).alias("ships_sent")
        )
        .explode("ships_sent")
        .join(df_lf, how="cross")
        .filter(
            (pl.col("step") > pl.col("step_src")) &
            (pl.col("id") != pl.col("id_src"))
        )
        .with_columns([
            dist_tgt_src_expr.alias("dist_tgt_src"),
            step_diff_expr.alias("step_diff"),
            fleet_speed_expr.alias("fleet_speed"),
            dist_min_expr.alias("dist_fleet_src_min"),
            dist_max_expr.alias("dist_fleet_src_max"),
            collision_expr.alias("collision"),
        ])
        .filter(pl.col("collision"))
        .with_columns(crossing_sun_expr.alias("crossing_sun"))
        .filter(~pl.col("crossing_sun"))
        .with_columns(
            pl.arctan2(pl.col("y") - pl.col("y_src"), pl.col("x") - pl.col("x_src")).alias("angle")
        )
        .with_columns(
            pl.max_horizontal(
                ((pl.col("dist_tgt_src").pow(2) + pl.col("dist_fleet_src_min").pow(2) -
                  pl.col("radius").pow(2)) /
                 (2 * pl.col("dist_tgt_src") * pl.col("dist_fleet_src_min"))).clip(-1.0, 1.0).arccos(),
                ((pl.col("dist_tgt_src").pow(2) + pl.col("dist_fleet_src_max").pow(2) -
                  pl.col("radius").pow(2)) /
                 (2 * pl.col("dist_tgt_src") * pl.col("dist_fleet_src_max"))).clip(-1.0, 1.0).arccos(),
            ).alias("radius_angle")
        )
        .with_columns([
            ((pl.col("angle") - pl.col("radius_angle")) % (2 * math.pi)).alias("angle_min"),
            ((pl.col("angle") + pl.col("radius_angle")) % (2 * math.pi)).alias("angle_max"),
        ])
        .sort("step")
        .collect()
    )

    if possible_attacks.is_empty():
        return ([], possible_attacks) if return_df else []

    df_obstacles = IntervalProcessorPolars.create_cumulative_obstacles(possible_attacks)

    attacks_with_angle = (
        possible_attacks.lazy()
        .join(df_obstacles.lazy(), on=["id_src", "step", "ships_sent"], how="left")
        .with_columns(
            pl.struct(["angle_min", "angle_max", "obstacle_list"])
            .map_elements(
                IntervalProcessorPolars.compute_free_angles,
                return_dtype=pl.List(pl.List(pl.Float64)),
            )
            .alias("angle_list")
        )
        .filter(pl.col("angle_list").list.len() > 0)
        .collect()
    )

    # Compute top-5 target planets per source BEFORE the comet filter,
    # matching the pandas take_action behaviour where planet_id_top_5_id_src
    # is derived from the full attacks_with_angle prior to comet exclusion.
    planet_id_top_5 = (
        attacks_with_angle.lazy()
        .sort(["step", "ships_sent"])
        .group_by(["id_src", "id"], maintain_order=True)
        .first()
        .sort(["step", "ships_sent"])
        .group_by("id_src", maintain_order=True)
        .head(5)
        .select(["id_src", "id"])
        .collect()
    )

    awa_comets = attacks_with_angle.filter(pl.col("nature_src") == "comet")
    moves = []
    if not awa_comets.is_empty():
        x_off = (awa_comets["x_src"] - CENTER).abs().max()
        y_off = (awa_comets["y_src"] - CENTER).abs().max()
        if max(x_off, y_off) > 45:
            comet_rows = (
                awa_comets
                .filter(pl.col("ships_sent") <= pl.col("ships_min"))
                .sort(["ships_sent", "step"], descending=[True, False])
                .group_by("id_src", maintain_order=True)
                .first()
                .select(["id_src", "angle", "ships_sent"])
                .rows()
            )
            moves += [list(r) for r in comet_rows]
            avoid = awa_comets["id_src"].unique().to_list()
            attacks_with_angle = attacks_with_angle.filter(~pl.col("id_src").is_in(avoid))

    attacks = (
        planet_id_top_5.lazy()
        .join(attacks_with_angle.lazy(), on=["id_src", "id"], how="left")
        .filter(pl.col("owner") != player_id)
        .with_columns(
            pl.when(pl.col("owner") == -1)
            .then(pl.col("ships"))
            .otherwise(pl.col("ships") + pl.col("production"))
            .alias("ships_needed")
        )
        .filter(
            (pl.col("ships_needed") + 1 <= pl.col("ships_sent")) &
            (pl.col("ships_sent") <= pl.col("ships_needed") + pl.col("production_src") + 1)
        )
        .sort(["step", "ships_sent"])
        .group_by(["id_src", "id"], maintain_order=True)
        .first()
        .with_columns(
            (pl.col("ships_needed") / pl.col("production_src")).alias("time_cost")
        )
        .with_columns(
            pl.col("time_cost").sum().over("id_src").alias("total_time_cost")
        )
        .with_columns(
            ((pl.col("total_time_cost") - pl.col("time_cost") - pl.col("step_diff")) *
             pl.col("production")).alias("score")
        )
        .sort("score", descending=True)
        .group_by("id_src", maintain_order=True)
        .first()
        .filter(pl.col("ships_sent") <= pl.col("ships_min"))
        .with_columns(
            pl.col("angle_list").map_batches(
                IntervalProcessorPolars.interval_to_final_angle,
                return_dtype=pl.Float64,
            ).alias("final_angle")
        )
        .collect()
    )

    moves += [list(r) for r in attacks.select(["id_src", "final_angle", "ships_sent"]).rows()]
    return (moves, possible_attacks) if return_df else moves


# ── Test 6 helpers ─────────────────────────────────────────────────────────────
def assert_moves_equal(moves_pd, moves_pl, label=""):
    key = lambda m: (int(m[0]), int(m[2]))
    pd_s = sorted(moves_pd, key=key)
    pl_s = sorted(moves_pl, key=key)
    assert len(pd_s) == len(pl_s), f"{label}: length {len(pd_s)} != {len(pl_s)}"
    for i, (a, b) in enumerate(zip(pd_s, pl_s)):
        assert int(a[0]) == int(b[0])
        assert int(a[2]) == int(b[2])
        assert abs(float(a[1]) - float(b[1])) < 1e-6
    print(f"{label}: OK")


def random_agent_fn(obs):
    player     = obs.player
    my_planets = [p for p in obs.planets if p[1] == player]
    if not my_planets:
        return []
    planet = random.choice(my_planets)
    ships  = planet[5] // 2
    if ships < 1:
        return []
    return [[planet[0], random.uniform(0, 2 * math.pi), ships]]


# ── Run ────────────────────────────────────────────────────────────────────────
SEED    = 42
N_STEPS = 100

random.seed(SEED)
env = ke.make("orbit_wars", debug=False)
env.reset(2)

for env_step in range(N_STEPS):
    obs0 = env.state[0].observation
    obs1 = env.state[1].observation

    df = _simulate(copy.deepcopy(obs0), global_step=env_step, num_agents=2, n_steps=NB_STEPS_SIM)

    moves_pd = take_action(df, player_id=0, nb_steps_sim=NB_STEPS_SIM)
    moves_lz = take_action_lazy(df, player_id=0, nb_steps_sim=NB_STEPS_SIM)
    assert_moves_equal(moves_pd, moves_lz, f"step {env_step}")

    rng_action = random_agent_fn(obs1)
    env.step([moves_pd, rng_action])

    if env.state[0].status != "ACTIVE":
        break

print(f"Test 6 passed: {env_step + 1} steps, all moves identical")
