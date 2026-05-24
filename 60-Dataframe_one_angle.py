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


def is_crossing_sun_vectorized(x_src, y_src, x, y, center, threshold):
    """Vectorized point-to-segment distance from sun center to each fleet path."""
    import numpy as np
    vx, vy = x_src, y_src
    wx, wy = x, y
    px, py = center, center

    l2 = (vx - wx) ** 2 + (vy - wy) ** 2
    l2_safe = np.where(l2 == 0, 1.0, l2)

    dot_prod = (px - vx) * (wx - vx) + (py - vy) * (wy - vy)

    t = np.clip(dot_prod / l2_safe, 0.0, 1.0)
    t = np.where(l2 == 0, 0.0, t)

    proj_x = vx + t * (wx - vx)
    proj_y = vy + t * (wy - vy)

    dist = np.sqrt((px - proj_x) ** 2 + (py - proj_y) ** 2)
    return dist < threshold


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


def _filter_blocked_attacks(possible_attacks):
    """Drop attacks whose direct angle to target is blocked by an earlier obstacle.

    For each (id_src, ships_sent) trajectory, an obstacle at step_obs blocks a
    target at step_tgt when step_obs < step_tgt and the direct angle to the target
    falls inside the obstacle's collision cone [angle_min_obs, angle_max_obs].
    The cone may wrap around 2π (angle_min > angle_max); handled via np.where.
    The target's raw atan2 angle (in [-π, π]) is normalised to [0, 2π] so that
    negative angles compare correctly against cone bounds in [0, 2π].
    """
    pairs = (
        possible_attacks[["id_src", "ships_sent", "step", "id", "angle", "angle_min", "angle_max"]]
        .merge(
            possible_attacks[["id_src", "ships_sent", "step", "id", "angle_min", "angle_max"]]
            .rename(columns={
                "step": "step_obs",
                "id": "id_obs",
                "angle_min": "angle_min_obs",
                "angle_max": "angle_max_obs",
            }),
            on=["id_src", "ships_sent"],
        )
        .query("step_obs < step and id_obs != id")
    )

    if pairs.empty:
        return possible_attacks.assign(final_angle=lambda d: d["angle"])

    angle_norm = np.mod(pairs["angle"].values, 2 * np.pi)
    pairs = pairs.assign(
        is_blocked=np.where(
            pairs["angle_min_obs"].values > pairs["angle_max_obs"].values,
            (angle_norm >= pairs["angle_min_obs"].values) | (angle_norm <= pairs["angle_max_obs"].values),
            (angle_norm >= pairs["angle_min_obs"].values) & (angle_norm <= pairs["angle_max_obs"].values),
        )
    )

    blocked = (
        pairs.query("is_blocked")[["id_src", "ships_sent", "step", "id"]]
        .drop_duplicates()
    )

    return (
        possible_attacks
        .merge(blocked, on=["id_src", "ships_sent", "step", "id"], how="left", indicator=True)
        .query("_merge == 'left_only'")
        .drop(columns="_merge")
        .assign(final_angle=lambda d: d["angle"])
    )


def take_action(df, player_id, nb_steps_sim=NB_STEPS_SIM, return_df=False):
    mine_across_sim = (
        df
        .assign(
            is_mine=lambda d: (d["owner"]==player_id).astype(int)
        )
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
        .merge(
            df,
            how="cross"
        )
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
            collision=lambda d: ((d["dist_tgt_src"] - d["radius"] < d["dist_fleet_src_min"]) & (d["dist_fleet_src_min"] < d["dist_tgt_src"] + d["radius"])) | ((d["dist_tgt_src"] - d["radius"] < d["dist_fleet_src_max"]) & (d["dist_fleet_src_max"] < d["dist_tgt_src"] + d["radius"])),
        )
        .query("collision")
        .assign(
            crossing_sun=lambda d: is_crossing_sun_vectorized(
                d["x_src"].values,
                d["y_src"].values,
                d["x"].values,
                d["y"].values,
                CENTER,
                SUN_RADIUS + PLANET_MARGIN,
            )
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

    attacks_with_angle = _filter_blocked_attacks(possible_attacks)

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

    # Comets handling
    attacks_with_angle_comets = attacks_with_angle.query("nature_src == 'comet'")
    moves = []
    if not attacks_with_angle_comets.empty and max(max((attacks_with_angle_comets["x_src"]-CENTER).abs()), max((attacks_with_angle_comets["y_src"]-CENTER).abs())) > 45:
        moves += (
            attacks_with_angle_comets
            .query("ships_sent <= ships_min")
            .sort_values(["ships_sent", "step"], ascending=[False, True])
            .groupby("id_src", as_index=False)
            .first()
            [["id_src", "final_angle", "ships_sent"]]
            .values
            .tolist()
        )
        id_to_avoid = attacks_with_angle_comets["id_src"].unique().tolist()
        attacks_with_angle = attacks_with_angle.query("id_src not in @id_to_avoid")

    attacks = (
        planet_id_top_5_id_src
        .merge(
            attacks_with_angle,
            how="left",
            on=["id_src", "id"]
        )
        .query("@player_id != owner")
        .assign(
            ships_needed = lambda d: np.where(
                d["owner"] == -1,
                d["ships"],
                d["ships"] + d["production"]
            )
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
    )
    for row in attacks.itertuples():
        print(f"From {row.id_src}, To {row.id} at step {row.step} with {row.ships_sent} ships (target has min {row.ships_min})")
    moves += (
        attacks
        [["id_src", "final_angle", "ships_sent"]]
        .values
        .tolist()
    )
    if return_df:
        return moves, possible_attacks
    return moves
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
    moves = take_action(df, player_id=player_id, nb_steps_sim=NB_STEPS_SIM)


    step += 1
    return moves


agent = nearest_planet_sniper
