import math
import copy
import polars as pl


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


# ── Strategy pipeline (Polars-based) ─────────────────────────────────────────
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
    def _02_get_all_opportunities(
        df_s: pl.DataFrame,
        planet_disp: pl.DataFrame,
        player_id: int,
    ) -> pl.LazyFrame:
        df_s_lf = df_s.lazy()
        planet_disp_lf = planet_disp.lazy()

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

        # Phase A: planet-level cross-join with sun-crossing filter
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
        )

        # Ships_sent expansion
        nb_steps_sim = GameConfig.NB_STEPS_SIM
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

        # Phase B: fleet-speed filter
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

        # Swept-pair collision (quadratic discriminant)
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

        # Angle geometry
        x_prev_f = pl.col("x_prev").fill_null(pl.col("x"))
        y_prev_f = pl.col("y_prev").fill_null(pl.col("y"))

        pa_lf = (
            expanded_lf
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
    def _04_score_and_decide(safe_lf: pl.LazyFrame, player_id: int) -> list:
        attacks_with_angle = safe_lf.collect()
        if attacks_with_angle.is_empty():
            return []

        moves = []

        # Comet evasion
        awa_comets = attacks_with_angle.filter(pl.col("nature_src") == "comet")
        if not awa_comets.is_empty():
            x_off = (awa_comets["x_src"] - GameConfig.CENTER).abs().max() or 0
            y_off = (awa_comets["y_src"] - GameConfig.CENTER).abs().max() or 0
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

        if attacks_with_angle.is_empty():
            return moves

        # Top-5 targets per source planet
        top5_ids = (
            attacks_with_angle
            .sort(["step", "ships_sent"])
            .group_by(["id_src", "id"], maintain_order=True)
            .first()
            .sort(["step", "ships_sent"])
            .group_by("id_src", maintain_order=True)
            .head(5)
            .select(["id_src", "id"])
            .with_columns(pl.lit(True).alias("is_top5"))
        )

        mine_src_ids = attacks_with_angle["id_src"].unique().to_list()

        # Classify Supplier / Conqueror
        src_nature = (
            top5_ids
            .with_columns(pl.col("id").is_in(mine_src_ids).alias("target_is_mine"))
            .group_by("id_src")
            .agg([
                pl.col("target_is_mine").sum().alias("mine_count"),
                pl.len().alias("total_count"),
            ])
            .with_columns(
                pl.when(pl.col("mine_count") == pl.col("total_count"))
                .then(pl.lit("Supplier"))
                .otherwise(pl.lit("Conqueror"))
                .alias("status")
            )
        )
        conqueror_ids = src_nature.filter(pl.col("status") == "Conqueror")["id_src"].to_list()
        supplier_ids = src_nature.filter(pl.col("status") == "Supplier")["id_src"].to_list()

        # ── Conqueror: attack enemy/neutral planets ──────────────────────────────
        attacks_conqueror = pl.DataFrame()
        attacks_conqueror_2 = pl.DataFrame()
        conqueror_needs = None

        if conqueror_ids:
            # All valid enemy/neutral attacks (no ships-range filter yet)
            _c_1_or_2 = (
                attacks_with_angle
                .filter(pl.col("id_src").is_in(conqueror_ids))
                .join(top5_ids.select(["id_src", "id", "is_top5"]), on=["id_src", "id"], how="left")
                .with_columns(pl.col("is_top5").fill_null(False))
                .filter(pl.col("is_top5"))
                .filter(pl.col("owner") != player_id)
                .with_columns(
                    pl.when(pl.col("owner") == -1)
                    .then(pl.col("ships"))
                    .otherwise(pl.col("ships") + pl.col("production"))
                    .alias("ships_needed")
                )
            )

            # Single-attacker path: apply ships-range filter and score
            _c = (
                _c_1_or_2
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
                    (pl.col("production") / (pl.col("time_cost") + pl.col("step_diff"))).alias("score")
                )
            )

            if not _c.is_empty():
                conqueror_needs = (
                    _c
                    .group_by("id_src", maintain_order=True)
                    .agg([
                        pl.col("ships_min").min().alias("ship_min"),
                        pl.col("ships_sent").sum().alias("all_need"),
                        pl.col("ships_sent").min().alias("lowest_need"),
                        pl.len().alias("nb_need"),
                        pl.col("score").max().alias("max_score"),
                    ])
                )

                attacks_conqueror = (
                    _c
                    .sort("score", descending=True)
                    .group_by("id_src", maintain_order=True)
                    .first()
                    .filter(pl.col("ships_sent") <= pl.col("ships_min"))
                )

                # Conqueror2: two source planets attacking the same target together
                if not _c_1_or_2.is_empty():
                    right = (
                        _c_1_or_2
                        .select(["id", "id_src", "step", "ships_needed", "ships_sent",
                                 "ships_min", "production_src", "step_diff"])
                        .rename({
                            "id_src": "id_src_2",
                            "step": "step_2",
                            "ships_needed": "ships_needed_2",
                            "ships_sent": "ships_sent_2",
                            "ships_min": "ships_min_2",
                            "production_src": "production_src_2",
                            "step_diff": "step_diff_2",
                        })
                    )

                    _c2 = (
                        _c_1_or_2
                        .join(right, on="id", how="inner")
                        .filter(pl.col("id_src") != pl.col("id_src_2"))
                        .filter(pl.col("step") < pl.col("step_2"))
                        .filter(
                            (pl.max_horizontal(pl.col("ships_needed"), pl.col("ships_needed_2")) + 1
                             <= pl.col("ships_sent") + pl.col("ships_sent_2")) &
                            (pl.col("ships_sent") + pl.col("ships_sent_2")
                             <= pl.max_horizontal(pl.col("ships_needed"), pl.col("ships_needed_2"))
                             + pl.col("production_src_2") + 1)
                        )
                        .filter(pl.col("ships_sent") <= pl.col("ships_min"))
                        .filter(pl.col("ships_sent_2") <= pl.col("ships_min_2") + pl.col("production_src_2"))
                        .join(
                            conqueror_needs.select(["id_src", "max_score"]),
                            on="id_src",
                            how="left",
                        )
                        .join(
                            conqueror_needs.select(["id_src", "max_score"])
                            .rename({"id_src": "id_src_2", "max_score": "max_score_2"}),
                            on="id_src_2",
                            how="left",
                        )
                        .with_columns([
                            (pl.col("ships_sent") / pl.col("production_src")).alias("time_cost"),
                            (pl.col("ships_sent_2") / pl.col("production_src_2")).alias("time_cost_2"),
                        ])
                        .with_columns(
                            (pl.col("production") / (
                                pl.col("step_diff_2") +
                                (pl.col("time_cost").pow(2) + pl.col("time_cost_2").pow(2)).sqrt()
                            )).alias("score")
                        )
                        .filter(
                            (pl.col("score") > pl.col("max_score").fill_null(0.0)) &
                            (pl.col("score") > pl.col("max_score_2").fill_null(0.0))
                        )
                        .sort(["step_2", "ships_sent_2"])
                        .group_by(["id_src", "id"], maintain_order=True)
                        .first()
                        .sort(["step_2", "ships_sent_2"])
                        .head(1)
                    )
                    if not _c2.is_empty():
                        attacks_conqueror_2 = _c2

        # ── Supplier: reinforce own planets ─────────────────────────────────────
        attacks_supplier = pl.DataFrame()
        if supplier_ids and conqueror_needs is not None:
            _s = (
                attacks_with_angle
                .filter(pl.col("id_src").is_in(supplier_ids))
                .join(top5_ids.select(["id_src", "id", "is_top5"]), on=["id_src", "id"], how="left")
                .with_columns(pl.col("is_top5").fill_null(False))
                .filter(pl.col("is_top5"))
                .with_columns(pl.col("id").is_in(supplier_ids).alias("target_is_supplier"))
                .filter(~pl.col("target_is_supplier"))
                .join(
                    conqueror_needs.rename({"id_src": "id"}),
                    on="id",
                    how="right",
                )
                .filter((pl.col("lowest_need") - pl.col("ships_min")) * 1.5 < pl.col("ships_sent"))
                .filter(
                    (pl.col("ships_min") * 0.75 < pl.col("ships_sent")) &
                    (pl.col("ships_sent") < pl.col("ships_min"))
                )
                .filter(pl.col("id_src").is_not_null())
                .sort(["all_need", "ships_sent"], descending=[True, False])
                .group_by("id_src", maintain_order=True)
                .first()
            )
            attacks_supplier = _s

        # ── Combine and emit ─────────────────────────────────────────────────────
        parts = [df for df in [attacks_conqueror, attacks_conqueror_2, attacks_supplier] if not df.is_empty()]
        if not parts:
            return moves

        attacks = pl.concat(parts, how="diagonal")
        for row in attacks.iter_rows(named=True):
            print(f"From {row['id_src']}, To {row['id']} at step {row['step']} "
                  f"with {row['ships_sent']} ships (target has min {row['ships_min']})")

        moves += [list(r) for r in attacks.select(["id_src", "final_angle", "ships_sent"]).rows()]
        return moves


# ── Entry point ───────────────────────────────────────────────────────────────
step = 0
num_agents = None
player_id = None


def agent(obs):
    global step, num_agents, player_id
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
    df_s, planet_disp = StrategyPipeline._01_get_obs_dataframe(obs, step, num_agents)
    pa_lf = StrategyPipeline._02_get_all_opportunities(df_s, planet_disp, player_id)
    safe_lf = StrategyPipeline._03_filter_collision(pa_lf)
    moves = StrategyPipeline._04_score_and_decide(safe_lf, player_id)
    step += 1
    return moves
