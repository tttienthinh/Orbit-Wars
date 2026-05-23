import math
import copy
import pandas as pd

# ── Constants ─────────────────────────────────────────────────────────────────
CENTER = 50.0
SUN_RADIUS = 10.0
ROTATION_RADIUS_LIMIT = 50.0
MAX_SPEED = 6.0
NB_STEPS_SIM = 10
PLANET_MARGIN = 0.1

# ── Interpreter (verbatim from 32-board_from_kaggle.ipynb cell 0) ─────────────
from collections import namedtuple

Planet = namedtuple(
    "Planet", ["id", "owner", "x", "y", "radius", "ships", "production"]
)
Fleet = namedtuple(
    "Fleet", ["id", "owner", "x", "y", "angle", "from_planet_id", "ships"]
)

BOARD_SIZE = 100.0
COMET_RADIUS = 1.0
COMET_PRODUCTION = 1
PLANET_CLEARANCE = 7
MIN_PLANET_GROUPS = 5
MAX_PLANET_GROUPS = 10
MIN_STATIC_GROUPS = 3
COMET_SPAWN_STEPS = [50, 150, 250, 350, 450]
CENTER_X = 50.0
CENTER_Y = 50.0
MAX_NB_STEP = 500


def distance(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def point_to_segment_distance(p, v, w):
    """Minimum distance from point p to line segment v-w."""
    l2 = (v[0] - w[0]) ** 2 + (v[1] - w[1]) ** 2
    if l2 == 0.0:
        return distance(p, v)
    t = max(
        0, min(1, ((p[0] - v[0]) * (w[0] - v[0]) + (p[1] - v[1]) * (w[1] - v[1])) / l2)
    )
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

    max_speed = MAX_SPEED
    fleets_to_remove = []
    combat_lists = {p[0]: [] for p in obs0.planets}

    for fleet in obs0.fleets:
        angle = fleet[4]
        ships = fleet[6]
        speed = 1.0 + (max_speed - 1.0) * (math.log(ships) / math.log(1000)) ** 1.5
        speed = min(speed, max_speed)
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
    comet_pid_set = set(obs0.comet_planet_ids)
    initial_by_id = {p[0]: p for p in obs0.initial_planets}

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
        r = math.sqrt(dx**2 + dy**2)
        old_pos = (planet[2], planet[3])
        if r + planet[4] < ROTATION_RADIUS_LIMIT:
            initial_angle = math.atan2(dy, dx)
            current_angle = initial_angle + angular_velocity * step
            planet[2] = CENTER + r * math.cos(current_angle)
            planet[3] = CENTER + r * math.sin(current_angle)
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
                old_pos = (planet[2], planet[3])
                planet[2] = p_path[idx][0]
                planet[3] = p_path[idx][1]
                if old_pos[0] >= 0:
                    sweep_fleets(planet, old_pos, (planet[2], planet[3]))

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


# ── Physics helpers ───────────────────────────────────────────────────────────

def _fleet_speed(ships):
    if ships <= 1:
        return 1.0
    ratio = math.log(ships) / math.log(1000.0)
    return 1.0 + (MAX_SPEED - 1.0) * max(0.0, min(1.0, ratio)) ** 1.5


def _simulate(obs, global_step, num_agents, n_steps=NB_STEPS_SIM):
    sim = copy.deepcopy(obs)
    no_actions = [[] for _ in range(num_agents)]
    rows = []
    for i in range(n_steps+1):
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
                "step": global_step + i,
                "id": pid,
                "x": x,
                "y": y,
                "radius": radius,
                "ships": ships,
                "production": production,
                "owner": owner,
                "nature": nature,
            })
        interpreter(sim, no_actions, global_step + i, num_agents)
    return pd.DataFrame(rows)


def _eta(src_x, src_y, src_r, tgt, angular_velocity, ships=1):
    """ETA in steps for a fleet of `ships` to reach tgt from (src_x, src_y)."""
    tx, ty, tr = tgt[2], tgt[3], tgt[4]
    speed = _fleet_speed(ships)
    if math.hypot(tx - CENTER, ty - CENTER) + tr >= ROTATION_RADIUS_LIMIT:
        dist = max(0.0, math.hypot(tx - src_x, ty - src_y) - src_r - tr)
        return max(1, math.ceil(dist / speed))
    base_angle = math.atan2(ty - CENTER, tx - CENTER)
    r = math.hypot(tx - CENTER, ty - CENTER)
    for t in range(1, 101):
        fx = CENTER + r * math.cos(base_angle + angular_velocity * t)
        fy = CENTER + r * math.sin(base_angle + angular_velocity * t)
        if math.hypot(fx - src_x, fy - src_y) - t * speed < src_r + tr:
            return t
    return 9999


def _aim_angle(src_x, src_y, src_r, tgt, angular_velocity, ships):
    """Angle to fire at tgt, accounting for orbital intercept."""
    tx, ty, tr = tgt[2], tgt[3], tgt[4]
    speed = _fleet_speed(ships)
    if math.hypot(tx - CENTER, ty - CENTER) + tr >= ROTATION_RADIUS_LIMIT:
        return math.atan2(ty - src_y, tx - src_x)
    base_angle = math.atan2(ty - CENTER, tx - CENTER)
    r = math.hypot(tx - CENTER, ty - CENTER)
    for t in range(1, 101):
        fx = CENTER + r * math.cos(base_angle + angular_velocity * t)
        fy = CENTER + r * math.sin(base_angle + angular_velocity * t)
        if math.hypot(fx - src_x, fy - src_y) - t * speed < src_r + tr:
            return math.atan2(fy - src_y, fx - src_x)
    return math.atan2(ty - src_y, tx - src_x)

import numpy as np
import polars as pl

class IntervalProcessorPolarsOptimized:

    # ── Pure-Python helpers (identical to IntervalProcessor) ──────────────────

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
    def create_cumulative_obstacles_numpy(possible_attacks: pl.DataFrame, min_step: int = 0) -> pl.DataFrame:
        """
        Optimizes obstacle timeline construction via zipped array sweeps instead of to_dicts().
        """
        if possible_attacks.is_empty():
            return pl.DataFrame(schema={"step": pl.Int64, "id_src": pl.Int64, "ships_sent": pl.Int64, "obstacle_list": pl.List(pl.List(pl.Float64))})

        max_step = int(possible_attacks["step"].max())

        # Zero-overhead column parsing using NumPy views
        id_srcs = possible_attacks["id_src"].to_numpy()
        ships = possible_attacks["ships_sent"].to_numpy()
        steps = possible_attacks["step"].to_numpy()
        amins = possible_attacks["angle_min"].to_numpy()
        amaxs = possible_attacks["angle_max"].to_numpy()

        attack_map = {}
        TWO_PI = 2 * np.pi

        for i in range(len(id_srcs)):
            key = (id_srcs[i], ships[i], steps[i])
            amin, amax = amins[i], amaxs[i]
            
            if key not in attack_map:
                attack_map[key] = []
                
            if amin > amax:
                attack_map[key].append((amin, TWO_PI))
                attack_map[key].append((0.0, amax))
            else:
                attack_map[key].append((amin, amax))

        # Pull unique rows purely using Polars tracking speed
        unique_combinations = (
            possible_attacks.select(["id_src", "ships_sent"])
            .unique(maintain_order=True)
            .to_numpy()
        )

        steps_col, id_srcs_col, ships_col, obstacles_col = [], [], [], []

        for id_src, ship in unique_combinations:
            current_intervals = []
            merged = []
            for step in range(min_step, max_step + 1):
                key = (id_src, ship, step)
                if key in attack_map:
                    current_intervals.extend(attack_map[key])
                    
                    # Inlined fast-sorting merge pattern
                    current_intervals.sort(key=lambda x: x[0])
                    curr_merged = [list(current_intervals[0])]
                    for curr in current_intervals[1:]:
                        if curr[0] <= curr_merged[-1][1]:
                            curr_merged[-1][1] = max(curr_merged[-1][1], curr[1])
                        else:
                            curr_merged.append(list(curr))
                    merged = curr_merged

                steps_col.append(step + 1)
                id_srcs_col.append(id_src)
                ships_col.append(ship)
                obstacles_col.append(merged.copy()) # Store native primitives

        return pl.DataFrame(
            {"step": steps_col, "id_src": id_srcs_col, "ships_sent": ships_col, "obstacle_list": obstacles_col},
            schema={"step": pl.Int64, "id_src": pl.Int64, "ships_sent": pl.Int64, "obstacle_list": pl.List(pl.List(pl.Float64))},
        )

    @staticmethod
    def compute_free_angles_vectorized(df: pl.DataFrame) -> list:
        """
        Eliminates pl.struct map_elements by running flat loops over zipped column lists.
        """
        amins = df["angle_min"].to_list()
        amaxs = df["angle_max"].to_list()
        obstacles_col = df["obstacle_list"].to_list()

        results = []
        TWO_PI = 2 * np.pi
        EPSILON = 1e-9

        for amin, amax, obstacles in zip(amins, amaxs, obstacles_col):
            if not obstacles:
                results.append([[amin, amax]])
                continue

            targets = [(amin, TWO_PI), (0.0, amax)] if amin > amax else [(amin, amax)]
            safe_intervals = targets

            # Flat procedural interval subtraction loop
            for b_min, b_max in obstacles:
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

            # Handle edge wrap-around re-stitching
            if len(safe_intervals) > 1:
                has_end, has_start = False, False
                end_idx, start_idx = -1, -1
                for idx, (f_min, f_max) in enumerate(safe_intervals):
                    if abs(f_max - TWO_PI) < EPSILON:
                        has_end, end_idx = True, idx
                    if abs(f_min - 0.0) < EPSILON:
                        has_start, start_idx = True, idx

                if has_end and has_start:
                    wrapped = [safe_intervals[end_idx][0], safe_intervals[start_idx][1]]
                    safe_intervals = [f for i, f in enumerate(safe_intervals) if i != end_idx and i != start_idx]
                    safe_intervals.append(wrapped)

            results.append([[float(a), float(b)] for a, b in safe_intervals])

        return results

    @staticmethod
    def interval_to_final_angle_fast(series: pl.Series) -> pl.Series:
        """
        Fixed midpoint extractor handling incoming Series data structures natively.
        """
        TWO_PI = 2 * np.pi
        
        def _best(intervals):
            # If Polars passes down a nested Series structure, safely unpack it
            if hasattr(intervals, "to_list"):
                intervals = intervals.to_list()
                
            if intervals is None or len(intervals) == 0:
                return float("nan")
                
            widest, best = -1.0, float("nan")
            for interval in intervals:
                amin, amax = interval[0], interval[1]
                span = (amax - amin) if amin <= amax else (TWO_PI - amin + amax)
                if span > widest:
                    widest = span
                    mid = (amin + amax) / 2.0 if amin <= amax else amin + span / 2.0
                    best = mid % TWO_PI
            return best

        # Apply using map_elements now that the element wrapper safely handles structural unpacking
        return series.map_elements(_best, return_dtype=pl.Float64)

def take_action_lazy_optimized(df: pd.DataFrame, player_id: int, nb_steps_sim: int = NB_STEPS_SIM, return_df: bool = False):
    df_lf = pl.from_pandas(df).sort("step").lazy()

    # 1. Source Planet Analysis
    mine_across_sim = (
        df_lf
        .with_columns(pl.when(pl.col("owner") == player_id).then(1).otherwise(0).alias("is_mine"))
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
        .filter((pl.col("row_count") == pl.col("is_mine")) & (pl.col("owner_src") == player_id))
        .rename({"id": "id_src"})
        .collect()
    )

    if mine_across_sim.is_empty():
        return ([], pl.DataFrame()) if return_df else []

    # Vectorized geometric expressions for Lazy engine context
    dx_vw = pl.col("x") - pl.col("x_src")
    dy_vw = pl.col("y") - pl.col("y_src")
    l2 = dx_vw.pow(2) + dy_vw.pow(2)
    dot = (CENTER - pl.col("x_src")) * dx_vw + (CENTER - pl.col("y_src")) * dy_vw
    t = (dot / pl.when(l2 == 0).then(pl.lit(1.0)).otherwise(l2)).clip(0.0, 1.0)
    
    dist_sun_proj = ((CENTER - (pl.col("x_src") + t * dx_vw)).pow(2) + (CENTER - (pl.col("y_src") + t * dy_vw)).pow(2)).sqrt()
    dist_sun_direct = ((CENTER - pl.col("x_src")).pow(2) + (CENTER - pl.col("y_src")).pow(2)).sqrt()
    dist_to_sun = pl.when(l2 == 0).then(dist_sun_direct).otherwise(dist_sun_proj)
    crossing_sun_expr = dist_to_sun < (SUN_RADIUS + PLANET_MARGIN)

    dist_tgt_src_expr = l2.sqrt()
    step_diff_expr = pl.col("step") - pl.col("step_src")
    fleet_speed_expr = 1.0 + (MAX_SPEED - 1.0) * (pl.col("ships_sent").cast(pl.Float64).log(base=math.e) / math.log(1000.0)).pow(1.5)
    
    dist_min_expr = step_diff_expr * fleet_speed_expr + PLANET_MARGIN + pl.col("radius_src")
    dist_max_expr = (step_diff_expr + 1) * fleet_speed_expr + PLANET_MARGIN + pl.col("radius_src")
    
    collision_expr = (
        ((dist_tgt_src_expr - pl.col("radius") < dist_min_expr) & (dist_min_expr < dist_tgt_src_expr + pl.col("radius"))) |
        ((dist_tgt_src_expr - pl.col("radius") < dist_max_expr) & (dist_max_expr < dist_tgt_src_expr + pl.col("radius")))
    )

    # 2. Lazy cross-join execution track
    possible_attacks = (
        mine_across_sim.lazy()
        .with_columns(pl.int_ranges(1, pl.col("ships_min") + pl.col("production_src") * nb_steps_sim + 1, dtype=pl.Int64).alias("ships_sent"))
        .explode("ships_sent")
        .join(df_lf, how="cross")
        .filter((pl.col("step") > pl.col("step_src")) & (pl.col("id") != pl.col("id_src")))
        .with_columns([
            dist_tgt_src_expr.alias("dist_tgt_src"),
            step_diff_expr.alias("step_diff"),
            fleet_speed_expr.alias("fleet_speed"),
            dist_min_expr.alias("dist_fleet_src_min"),
            dist_max_expr.alias("dist_fleet_src_max"),
            collision_expr.alias("collision"),
        ])
        .filter(pl.col("collision"))
        .filter(~crossing_sun_expr)
        .with_columns(pl.arctan2(pl.col("y") - pl.col("y_src"), pl.col("x") - pl.col("x_src")).alias("angle"))
        .with_columns(
            pl.max_horizontal(
                ((pl.col("dist_tgt_src").pow(2) + pl.col("dist_fleet_src_min").pow(2) - pl.col("radius").pow(2)) / (2 * pl.col("dist_tgt_src") * pl.col("dist_fleet_src_min"))).clip(-1.0, 1.0).arccos(),
                ((pl.col("dist_tgt_src").pow(2) + pl.col("dist_fleet_src_max").pow(2) - pl.col("radius").pow(2)) / (2 * pl.col("dist_tgt_src") * pl.col("dist_fleet_src_max"))).clip(-1.0, 1.0).arccos(),
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

    # 3. Create Obstacles (Numpy Accelerated Step)
    df_obstacles = IntervalProcessorPolarsOptimized.create_cumulative_obstacles_numpy(possible_attacks)

    # 4. Join and Vectorize Free Angle Computations
    attacks_with_angle = possible_attacks.join(df_obstacles, on=["id_src", "step", "ships_sent"], how="left")
    
    # Fast non-struct batch assignment execution path
    attacks_with_angle = attacks_with_angle.with_columns(
        pl.Series(
            name="angle_list", 
            values=IntervalProcessorPolarsOptimized.compute_free_angles_vectorized(attacks_with_angle),
            dtype=pl.List(pl.List(pl.Float64))
        )
    ).filter(pl.col("angle_list").list.len() > 0)

    # 5. Comet filtering routines
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

    # Top-5 extraction logic
    planet_id_top_5 = (
        attacks_with_angle.lazy()
        .sort(["step", "ships_sent"])
        .group_by(["id_src", "id"], maintain_order=True)
        .first()
        .sort(["step", "ships_sent"])
        .group_by("id_src", maintain_order=True)
        .head(5)
        .select(["id_src", "id"])
    )

    # 6. Final Score Strategy calculation 
    attacks = (
        planet_id_top_5
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
        .with_columns((pl.col("ships_needed") / pl.col("production_src")).alias("time_cost"))
        .with_columns(pl.col("time_cost").sum().over("id_src").alias("total_time_cost"))
        .with_columns(((pl.col("total_time_cost") - pl.col("time_cost") - pl.col("step_diff")) * pl.col("production")).alias("score"))
        .sort("score", descending=True)
        .group_by("id_src", maintain_order=True)
        .first()
        .filter(pl.col("ships_sent") <= pl.col("ships_min"))
        .with_columns(
            pl.col("angle_list").map_batches(
                IntervalProcessorPolarsOptimized.interval_to_final_angle_fast,
                return_dtype=pl.Float64,
            ).alias("final_angle")
        )
        .collect()
    )

    moves += [list(r) for r in attacks.select(["id_src", "final_angle", "ships_sent"]).rows()]
    return (moves, possible_attacks) if return_df else moves


# ── Agent ─────────────────────────────────────────────────────────────────────

step = 0
num_agents = None
player_id = None

def nearest_planet_sniper(obs):
    global step
    global num_agents
    global player_id

    print(f"Agent called step: {step} remainingOverageTime: {obs.get('remainingOverageTime', 0)}")
    if num_agents is None:
        initial = (
            obs.initial_planets if hasattr(obs, "initial_planets")
            else obs["initial_planets"]
        )
        owners = {p[1] for p in initial if p[1] != -1}
        num_agents = 4 if len(owners) > 2 else 2
    if player_id is None:
        player_id = obs.get("player", 0) if isinstance(obs, dict) else obs.player

    df = _simulate(obs, step, num_agents, n_steps=NB_STEPS_SIM)
    moves = take_action_lazy_optimized(df, player_id=player_id, nb_steps_sim=NB_STEPS_SIM)

    step += 1
    return moves


agent = nearest_planet_sniper
