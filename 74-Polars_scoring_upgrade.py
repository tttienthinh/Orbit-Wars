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
# Max orbital displacement per tick (generous: 0.05 rad/tick * 45 unit radius ≈ 2.25)
PLANET_MOVEMENT_SLACK = 3.0

# ── Scoring weights ────────────────────────────────────────────────────────────
# All constants below are tunable. Each line notes the effect of increasing (↑) or
# decreasing (↓) the value.
#
# Production value
PROD_MULT           = 10.0  # ↑ favours high-production targets over all else
TIME_PROD_MULT      =  1.0  # ↑ urgency: rewards capturing earlier (more ticks to collect)
#
# Enemy flip bonus — applied only when target.owner is an enemy (not neutral, not own)
ENEMY_MULT          = 10.0  # ↑ strongly prefers attacking enemies over claiming neutral planets
COMPOUND_MULT       =  5.0  # ↑ extra reward for flipping a well-surrounded enemy (cluster break)
#
# Target strategic context
MINE_NEAR_TGT_MULT  =  5.0  # ↑ confident attacks where other own planets can finish the job
ENEMY_NEAR_TGT_MULT =  3.0  # ↑ values targets near many enemy planets (cuts supply lines)
#
# Source quality
PROD_SRC_MULT       =  2.0  # ↑ prefers attacking from high-production sources (faster recovery)
ORBIT_BONUS         = 10.0  # ↑ activates orbiting (inner-ring) source planets more aggressively
PROXIMITY_MULT      =  8.0  # ↑ front-line sources (near enemies) attack more boldly
#
# Cost / risk
DIST_MULT           =  0.5  # ↑ stay closer to home; ↓ reach across the board
SHIPS_MULT          =  0.3  # ↑ more conservative fleet sizes; ↓ sends bigger fleets
ETA_MULT            =  1.0  # ↑ prioritises fast captures; ↓ patient long-range attacks
OVEREXTEND_MULT     =  0.5  # ↑ protects source from being drained; ↓ allows bold over-sends
#
# Proximity threshold for n_enemy_nearby_target (board units)
PROXIMITY_DIST      = 20.0  # ↑ counts more enemies as "near" a target; ↓ only immediate neighbours

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
        speed = 1.0 + (max_speed - 1.0) * (math.log(ships) / math.log(1000)) ** 1.5
        speed = min(speed, max_speed)
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
            if swept_pair_hit(f_old, f_new, p_old, p_new, planet[4]):
                combat_lists[planet[0]].append(fleet)
                fleets_to_remove.append(fleet)
                hit_planet = True
                break
        if hit_planet:
            continue
        if not (0 <= fleet[2] <= BOARD_SIZE and 0 <= fleet[3] <= BOARD_SIZE):
            fleets_to_remove.append(fleet)
            continue
        if point_to_segment_distance((CENTER, CENTER), f_old, f_new) < SUN_RADIUS:
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
                            if point_to_segment_distance((fleet[2], fleet[3]), c_old, c_new) < planet[4]:
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
    prev_pos_lf = (
        df_lf.select(["id", "step", "x", "y"])
        .rename({"x": "x_prev", "y": "y_prev"})
        .with_columns((pl.col("step") + 1).alias("step"))
    )
    planet_disp_lf = (
        df_lf.select(["id", "step", "x", "y"])
        .join(prev_pos_lf, on=["id", "step"], how="left")
        .with_columns(
            ((pl.col("x") - pl.col("x_prev").fill_null(pl.col("x"))).pow(2) +
             (pl.col("y") - pl.col("y_prev").fill_null(pl.col("y"))).pow(2)
            ).sqrt().alias("planet_disp")
        )
        .select(["id", "step", "planet_disp"])
    )

    # ── Section 2: mine analysis — no ships_sent expansion yet ─────────────────
    mine_base_lf = (
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
    )

    # ── Section 3: three-phase collision filter ────────────────────────────────
    # Shared lazy expressions (evaluated from raw cross-join columns)
    dx = pl.col("x") - pl.col("x_src")
    dy = pl.col("y") - pl.col("y_src")
    l2 = dx.pow(2) + dy.pow(2)
    dist_tgt_src = l2.sqrt()
    step_diff = pl.col("step") - pl.col("step_src")

    dot = (CENTER - pl.col("x_src")) * dx + (CENTER - pl.col("y_src")) * dy
    t_sun = (dot / pl.when(l2 == 0).then(pl.lit(1.0)).otherwise(l2)).clip(0.0, 1.0)
    proj_dist_sun = (
        (CENTER - pl.col("x_src") - t_sun * dx).pow(2) +
        (CENTER - pl.col("y_src") - t_sun * dy).pow(2)
    ).sqrt()
    crossing_sun = pl.when(l2 == 0).then(
        ((CENTER - pl.col("x_src")).pow(2) + (CENTER - pl.col("y_src")).pow(2)).sqrt()
    ).otherwise(proj_dist_sun) < (SUN_RADIUS + PLANET_MARGIN)

    # Phase A: planet-level cross-join (N_src × N_tgt × N_steps, no ships_sent).
    # Filter with actual per-tick planet displacement from simulation data.
    coarse_lf = (
        mine_base_lf
        .join(df_lf, how="cross")
        .filter((pl.col("step") > pl.col("step_src")) & (pl.col("id") != pl.col("id_src")))
        .join(planet_disp_lf, on=["id", "step"], how="left")
        .with_columns([
            dist_tgt_src.alias("dist_tgt_src"),
            step_diff.alias("step_diff"),
        ])
        .filter(
            (pl.col("dist_tgt_src") < (pl.col("step_diff") + 1) * MAX_SPEED
             + pl.col("radius_src") + PLANET_MARGIN + pl.col("radius")
             + pl.col("planet_disp").fill_null(0.0))
            & ~crossing_sun
        )
    )

    # Ships_sent expansion: only for (src, target, step) triples that passed Phase A.
    expanded_lf = (
        coarse_lf
        .with_columns(
            pl.int_ranges(
                1,
                pl.col("ships_min") + pl.col("production_src") * nb_steps_sim + 1,
                dtype=pl.Int64,
            ).alias("ships_sent")
        )
        .explode("ships_sent")
    )

    # Phase B: fleet_speed-specific second filter, prev_pos join, swept-pair, angle cone.
    fleet_speed = 1.0 + (MAX_SPEED - 1.0) * (
        pl.col("ships_sent").cast(pl.Float64).log(base=math.e) / math.log(1000.0)
    ).pow(1.5)
    dist_min = pl.col("step_diff") * fleet_speed + PLANET_MARGIN + pl.col("radius_src")
    dist_prev = dist_min - fleet_speed

    unit_x = (pl.col("x") - pl.col("x_src")) / pl.when(pl.col("dist_tgt_src") < 1e-9).then(pl.lit(1.0)).otherwise(pl.col("dist_tgt_src"))
    unit_y = (pl.col("y") - pl.col("y_src")) / pl.when(pl.col("dist_tgt_src") < 1e-9).then(pl.lit(1.0)).otherwise(pl.col("dist_tgt_src"))
    fleet_x0 = pl.col("x_src") + unit_x * pl.col("dist_prev")
    fleet_y0 = pl.col("y_src") + unit_y * pl.col("dist_prev")
    planet_vx = pl.col("x") - pl.col("x_prev")
    planet_vy = pl.col("y") - pl.col("y_prev")
    dvx_sp = unit_x * pl.col("fleet_speed") - planet_vx
    dvy_sp = unit_y * pl.col("fleet_speed") - planet_vy
    d0x_sp = fleet_x0 - pl.col("x_prev")
    d0y_sp = fleet_y0 - pl.col("y_prev")
    a_sp   = dvx_sp.pow(2) + dvy_sp.pow(2)
    b_sp   = 2.0 * (d0x_sp * dvx_sp + d0y_sp * dvy_sp)
    c_sp   = d0x_sp.pow(2) + d0y_sp.pow(2) - pl.col("radius").pow(2)
    disc_sp = b_sp.pow(2) - 4.0 * a_sp * c_sp
    sq_sp  = disc_sp.clip(lower_bound=0.0).sqrt()
    t1_expr = pl.when(a_sp < 1e-12).then(pl.lit(0.0)).otherwise((-b_sp - sq_sp) / (2.0 * a_sp))
    t2_expr = pl.when(a_sp < 1e-12).then(pl.lit(1.0)).otherwise((-b_sp + sq_sp) / (2.0 * a_sp))
    collision = pl.when(a_sp < 1e-12).then(c_sp <= 0.0).otherwise(
        (disc_sp >= 0.0) & (t2_expr >= 0.0) & (t1_expr <= 1.0)
    )

    pa_lf = (
        expanded_lf
        .with_columns([
            fleet_speed.alias("fleet_speed"),
            dist_min.alias("dist_min"),
            dist_prev.alias("dist_prev"),
        ])
        .filter(
            pl.col("dist_tgt_src") < pl.col("dist_min") + pl.col("fleet_speed")
            + pl.col("radius") + PLANET_MOVEMENT_SLACK
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
            (pl.col("x_prev") + pl.col("t1_eff") * (pl.col("x") - pl.col("x_prev"))).alias("p_t1_x"),
            (pl.col("y_prev") + pl.col("t1_eff") * (pl.col("y") - pl.col("y_prev"))).alias("p_t1_y"),
            (pl.col("x_prev") + pl.col("t2_eff") * (pl.col("x") - pl.col("x_prev"))).alias("p_t2_x"),
            (pl.col("y_prev") + pl.col("t2_eff") * (pl.col("y") - pl.col("y_prev"))).alias("p_t2_y"),
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
                (pl.col("angle_t1").sin() + pl.col("angle_t2").sin()),
                (pl.col("angle_t1").cos() + pl.col("angle_t2").cos()),
            ).alias("angle"),
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

    # Pre-filter to affordable attacks and deduplicate to cheapest (src, target) pair
    best_per_pair = (
        attacks_with_angle
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
    )

    if best_per_pair.is_empty():
        return (moves, attacks_with_angle) if return_df else moves

    # ── Context features ───────────────────────────────────────────────────────

    # n_enemy_nearby_src: among src's 5 nearest reachable targets, how many are enemy-owned.
    # High value → source is a front-line planet surrounded by enemies.
    src_neighbor_stats = (
        best_per_pair
        .sort(["id_src", "step", "ships_sent"])
        .group_by("id_src", maintain_order=True)
        .head(5)
        .with_columns(
            pl.when(
                (pl.col("owner") != player_id) & (pl.col("owner") != -1)
            ).then(pl.lit(1)).otherwise(pl.lit(0)).alias("_is_enemy")
        )
        .group_by("id_src", maintain_order=True)
        .agg(pl.col("_is_enemy").sum().alias("n_enemy_nearby_src"))
    )

    # n_mine_nearby_target: how many distinct own source planets can reach each target.
    # High value → multiple own planets can finish the job if the first fleet falls short.
    tgt_support_stats = (
        best_per_pair
        .group_by("id", maintain_order=True)
        .agg(pl.col("id_src").n_unique().alias("n_mine_nearby_target"))
    )

    # n_enemy_nearby_target: enemy planets within PROXIMITY_DIST of each target (step-0 snapshot).
    # High value → target is deep in enemy territory; capturing it disrupts their supply lines.
    step0_min = df["step"].min()
    step0_pl = pl.from_pandas(df[df["step"] == step0_min][["id", "x", "y", "owner"]])
    enemy_pos = (
        step0_pl
        .filter((pl.col("owner") != player_id) & (pl.col("owner") != -1))
        .select(["id", "x", "y"])
        .rename({"id": "id_enemy", "x": "x_enemy", "y": "y_enemy"})
    )
    target_pos = (
        step0_pl
        .join(best_per_pair.select("id").unique(), on="id")
        .select(["id", "x", "y"])
    )
    if enemy_pos.is_empty() or target_pos.is_empty():
        tgt_enemy_stats = (
            target_pos.select("id")
            .with_columns(pl.lit(0).cast(pl.Int32).alias("n_enemy_nearby_target"))
        )
    else:
        tgt_enemy_stats = (
            target_pos
            .join(enemy_pos, how="cross")
            .with_columns(
                ((pl.col("x") - pl.col("x_enemy")).pow(2) +
                 (pl.col("y") - pl.col("y_enemy")).pow(2)).sqrt().alias("_dist")
            )
            .filter(
                (pl.col("id") != pl.col("id_enemy")) &
                (pl.col("_dist") <= PROXIMITY_DIST)
            )
            .group_by("id", maintain_order=True)
            .agg(pl.len().alias("n_enemy_nearby_target"))
        )

    # ── Full scoring formula ───────────────────────────────────────────────────
    # Each constant is defined at the top of this file with tuning notes.
    # Increase a constant to make that feature matter more; decrease to downweight it.
    is_enemy = (pl.col("owner") != player_id) & (pl.col("owner") != -1)
    attacks = (
        best_per_pair
        .join(src_neighbor_stats, on="id_src", how="left")
        .join(tgt_support_stats, on="id", how="left")
        .join(tgt_enemy_stats, on="id", how="left")
        .with_columns([
            pl.col("n_enemy_nearby_src").fill_null(0).cast(pl.Float64),
            pl.col("n_mine_nearby_target").fill_null(0).cast(pl.Float64),
            pl.col("n_enemy_nearby_target").fill_null(0).cast(pl.Float64),
        ])
        .with_columns(
            (
                # Production value: capture high-production planets first
                PROD_MULT * pl.col("production")
                # Urgency: capturing earlier leaves more ticks to collect output
                + TIME_PROD_MULT * (NB_STEPS_SIM - pl.col("step_diff")).clip(lower_bound=0) * pl.col("production")
                # Enemy flip bonus: attacking enemies beats claiming neutral planets
                + ENEMY_MULT * pl.when(is_enemy).then(pl.lit(1.0)).otherwise(pl.lit(0.0))
                # Cluster break: extra reward for flipping an enemy deep in enemy territory
                + COMPOUND_MULT * pl.when(is_enemy).then(pl.col("n_enemy_nearby_target")).otherwise(pl.lit(0.0))
                # Backup support: other own planets can finish the job if needed
                + MINE_NEAR_TGT_MULT * pl.col("n_mine_nearby_target")
                # Supply line disruption: target is near many enemy planets (enemy hub)
                + ENEMY_NEAR_TGT_MULT * pl.col("n_enemy_nearby_target")
                # Source quality: high-production sources recover ships faster after attack
                + PROD_SRC_MULT * pl.col("production_src")
                # Orbiting (inner-ring) source planets are hard to defend — use them aggressively
                + ORBIT_BONUS * pl.when(pl.col("nature_src") == "moving").then(pl.lit(1.0)).otherwise(pl.lit(0.0))
                # Front-line bonus: source surrounded by enemies should strike, not sit idle
                + PROXIMITY_MULT * pl.col("n_enemy_nearby_src")
                # Distance cost: longer flights tie up ships longer
                - DIST_MULT * pl.col("dist_tgt_src")
                # Fleet size cost: bigger fleets leave source more exposed
                - SHIPS_MULT * pl.col("ships_sent")
                # ETA cost: prefer faster captures over patient long-range attacks
                - ETA_MULT * pl.col("step_diff")
                # Overextend penalty: draining source beyond ships_needed makes it vulnerable
                - OVEREXTEND_MULT * (pl.col("ships_sent") - pl.col("ships_needed")).clip(lower_bound=0)
            ).alias("score")
        )
        .filter(pl.col("score") > 0)
        .sort("score", descending=True)
        .group_by("id_src", maintain_order=True)
        .first()
        .filter(pl.col("ships_sent") <= pl.col("ships_min"))
    )

    for row in attacks.iter_rows(named=True):
        print(f"From {row['id_src']}, To {row['id']} at step {row['step']} "
              f"with {row['ships_sent']} ships (score={row['score']:.1f})")

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
