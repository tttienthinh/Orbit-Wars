import math
import copy
import pandas as pd
import polars as pl

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



def take_action(df, player_id, nb_steps_sim=NB_STEPS_SIM, return_df=False):
    # ── Section 1: entry ───────────────────────────────────────────────────────
    df_lf = pl.from_pandas(df).sort("step").lazy()

    # ── Section 2: mine analysis + ships_sent expansion ────────────────────────
    mine_lf = (
        df_lf
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
        .with_columns(
            pl.int_ranges(
                1,
                pl.col("ships_min") + pl.col("production_src") * nb_steps_sim + 1,
                dtype=pl.Int64,
            ).alias("ships_sent")
        )
        .explode("ships_sent")
    )

    # ── Section 3: cross-join + collision + sun (all lazy, all inlined) ────────
    dx = pl.col("x") - pl.col("x_src")
    dy = pl.col("y") - pl.col("y_src")
    l2 = dx.pow(2) + dy.pow(2)
    dist_tgt_src = l2.sqrt()
    step_diff   = pl.col("step") - pl.col("step_src")
    fleet_speed = 1.0 + (MAX_SPEED - 1.0) * (
        pl.col("ships_sent").cast(pl.Float64).log(base=math.e) / math.log(1000.0)
    ).pow(1.5)
    dist_min = step_diff * fleet_speed + PLANET_MARGIN + pl.col("radius_src")
    dist_max = (step_diff + 1) * fleet_speed + PLANET_MARGIN + pl.col("radius_src")
    collision = (
        ((dist_tgt_src - pl.col("radius") < dist_min) & (dist_min < dist_tgt_src + pl.col("radius"))) |
        ((dist_tgt_src - pl.col("radius") < dist_max) & (dist_max < dist_tgt_src + pl.col("radius")))
    )
    dot   = (CENTER - pl.col("x_src")) * dx + (CENTER - pl.col("y_src")) * dy
    t_sun = (dot / pl.when(l2 == 0).then(pl.lit(1.0)).otherwise(l2)).clip(0.0, 1.0)
    proj_dist_sun = (
        (CENTER - pl.col("x_src") - t_sun * dx).pow(2) +
        (CENTER - pl.col("y_src") - t_sun * dy).pow(2)
    ).sqrt()
    crossing_sun = pl.when(l2 == 0).then(
        ((CENTER - pl.col("x_src")).pow(2) + (CENTER - pl.col("y_src")).pow(2)).sqrt()
    ).otherwise(proj_dist_sun) < (SUN_RADIUS + PLANET_MARGIN)

    pa_lf = (
        mine_lf
        .join(df_lf, how="cross")
        .filter((pl.col("step") > pl.col("step_src")) & (pl.col("id") != pl.col("id_src")))
        .with_columns([
            dist_tgt_src.alias("dist_tgt_src"),
            step_diff.alias("step_diff"),
            fleet_speed.alias("fleet_speed"),
            dist_min.alias("dist_fleet_src_min"),
            dist_max.alias("dist_fleet_src_max"),
            collision.alias("collision"),
        ])
        .filter(pl.col("collision"))
        .filter(~crossing_sun)
        .with_columns(pl.arctan2(dy, dx).alias("angle"))
        .with_columns(
            pl.max_horizontal(
                ((pl.col("dist_tgt_src").pow(2) + pl.col("dist_fleet_src_min").pow(2) - pl.col("radius").pow(2))
                 / (2 * pl.col("dist_tgt_src") * pl.col("dist_fleet_src_min"))).clip(-1.0, 1.0).arccos(),
                ((pl.col("dist_tgt_src").pow(2) + pl.col("dist_fleet_src_max").pow(2) - pl.col("radius").pow(2))
                 / (2 * pl.col("dist_tgt_src") * pl.col("dist_fleet_src_max"))).clip(-1.0, 1.0).arccos(),
            ).alias("radius_angle")
        )
        .with_columns([
            ((pl.col("angle") - pl.col("radius_angle")) % (2 * math.pi)).alias("angle_min"),
            ((pl.col("angle") + pl.col("radius_angle")) % (2 * math.pi)).alias("angle_max"),
        ])
        .sort("step")
    )

    # ── Section 4: blocking self-join + THE ONE collect() ──────────────────────
    angle_norm = pl.col("angle") % (2 * math.pi)
    wraps      = pl.col("angle_min_obs") > pl.col("angle_max_obs")
    in_cone    = pl.when(wraps).then(
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
        .filter((pl.col("step_obs") < pl.col("step")) & (pl.col("id_obs") != pl.col("id")))
        .filter(in_cone)
        .select(["id_src", "ships_sent", "step", "id"])
        .unique()
    )

    attacks_with_angle = (
        pa_lf
        .join(blocked_lf, on=["id_src", "ships_sent", "step", "id"], how="anti")
        .with_columns(pl.col("angle").alias("final_angle"))
        .collect()
    )

    # ── Section 5: comet handling + scoring (eager) ────────────────────────────
    if attacks_with_angle.is_empty():
        return ([], attacks_with_angle) if return_df else []

    awa_comets = attacks_with_angle.filter(pl.col("nature_src") == "comet")
    moves = []
    if not awa_comets.is_empty():
        x_off = (awa_comets["x_src"] - CENTER).abs().max() or 0
        y_off = (awa_comets["y_src"] - CENTER).abs().max() or 0
        if max(x_off, y_off) > 45:
            moves += [list(r) for r in (
                awa_comets
                .filter(pl.col("ships_sent") <= pl.col("ships_min"))
                .sort(["ships_sent", "step"], descending=[True, False])
                .group_by("id_src", maintain_order=True)
                .first()
                .select(["id_src", "final_angle", "ships_sent"])
                .rows()
            )]
            id_to_avoid = awa_comets["id_src"].unique().to_list()
            attacks_with_angle = attacks_with_angle.filter(~pl.col("id_src").is_in(id_to_avoid))

    planet_id_top_5 = (
        attacks_with_angle
        .sort(["step", "ships_sent"])
        .group_by(["id_src", "id"], maintain_order=True)
        .first()
        .sort(["step", "ships_sent"])
        .group_by("id_src", maintain_order=True)
        .head(5)
        .select(["id_src", "id"])
    )

    attacks = (
        planet_id_top_5
        .join(attacks_with_angle, on=["id_src", "id"], how="left")
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
        .with_columns(
            ((pl.col("total_time_cost") - pl.col("time_cost") - pl.col("step_diff")) * pl.col("production"))
            .alias("score")
        )
        .sort("score", descending=True)
        .group_by("id_src", maintain_order=True)
        .first()
        .filter(pl.col("ships_sent") <= pl.col("ships_min"))
    )

    for row in attacks.iter_rows(named=True):
        print(f"From {row['id_src']}, To {row['id']} at step {row['step']} "
              f"with {row['ships_sent']} ships (target has min {row['ships_min']})")

    moves += [list(r) for r in attacks.select(["id_src", "final_angle", "ships_sent"]).rows()]
    return (moves, attacks_with_angle) if return_df else moves

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
