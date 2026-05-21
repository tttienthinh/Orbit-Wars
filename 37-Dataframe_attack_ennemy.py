import math
import copy
import pandas as pd

# ── Constants ─────────────────────────────────────────────────────────────────
CENTER = 50.0
SUN_RADIUS = 10.0
ROTATION_RADIUS_LIMIT = 50.0
MAX_SPEED = 6.0
NB_STEPS_SIM = 50
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

def take_action(df, player_id, nb_steps_sim=NB_STEPS_SIM):
    mine_across_sim = (
        df
        .query("owner == @player_id")
        .groupby("id")
        .agg(
            step_src=("step", "first"),
            x_src=("x", "first"),
            y_src=("y", "first"),
            radius_src=("radius", "first"),
            ships_min=("ships", "min"),
            production_src=("production", "first"),
            nature_src=("nature", "first"),
            row_count=("ships", "size"),
        )
        .query("row_count >= @nb_steps_sim + 1 and ships_min > 0")
        .reset_index(drop=False)
        .rename(columns={"id": "id_src"})
    )

    df_src_tgt = (
        mine_across_sim
        .merge(
            df,
            how="cross"
        )
        .query("step > step_src and id != id_src and @player_id != owner")
    )

    possible_attacks = (
        df_src_tgt
        .assign(
            dist_tgt_src=lambda d: ((d["x"] - d["x_src"]) ** 2 + (d["y"] - d["y_src"]) ** 2) ** 0.5 - d["radius_src"] - d["radius"] - PLANET_MARGIN,
            step_diff=lambda d: d["step"] - d["step_src"] + 1, # For a one ship fleet, the speed is 1 unit per step
            ships_needed=lambda d: d["ships"] + 1, # Need at least one more ship than the target to win
            fleet_speed=lambda d: 1.0 + (MAX_SPEED - 1.0) * ((np.log(d["ships_needed"])) / math.log(1000)) ** 1.5,
            possible_attack=lambda d: (d["dist_tgt_src"] / d["fleet_speed"] <= d["step_diff"]),
        )
        .query("possible_attack and ships_needed <= ships_min") # Only consider attacks we can win with the ships we have at the source during the sim
        .assign(
            crossing_sun=lambda d: d.apply(
                lambda row: point_to_segment_distance(
                    (CENTER, CENTER),
                    (row["x_src"], row["y_src"]),
                    (row["x"], row["y"]),
                ) < SUN_RADIUS + PLANET_MARGIN,
                axis=1,
            )
        )
        .query("not crossing_sun")
        .sort_values(["step_src", "id_src", "step", "id"], ascending=[True, True, True, True])
        .groupby(["step_src", "id_src", "id"], as_index=False)
        .first()
        .assign(
            angle=lambda d: np.arctan2(d["y"] - d["y_src"], d["x"] - d["x_src"]),
        )
        .sort_values("step", ascending=True)
    )
    for row in possible_attacks.itertuples():
        print(f"From {row.id_src}, To {row.id} at step {row.step} with {row.ships_needed} ships (target has min {row.ships_min})")
    moves = (
        possible_attacks
        [["id_src", "angle", "ships_needed"]]
        .values
        .tolist()
    )
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
