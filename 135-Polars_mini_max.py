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


def evaluate(obs, player_id: int) -> tuple:
    planets = obs.planets if hasattr(obs, "planets") else obs["planets"]
    fleets  = obs.fleets  if hasattr(obs, "fleets")  else obs["fleets"]

    opponents = {p[1] for p in planets if p[1] not in (-1, player_id)}
    opponents |= {f[1] for f in fleets  if f[1] not in (-1, player_id)}

    my_prod  = sum(p[6] for p in planets if p[1] == player_id)
    my_ships = (sum(p[5] for p in planets if p[1] == player_id)
              + sum(f[6] for f in fleets  if f[1] == player_id))

    if not opponents:
        return (my_prod, my_ships)

    opp_prod  = max(sum(p[6] for p in planets if p[1] == opp) for opp in opponents)
    opp_ships = max(
        sum(p[5] for p in planets if p[1] == opp)
      + sum(f[6] for f in fleets  if f[1] == opp)
        for opp in opponents
    )
    return (my_prod - opp_prod, my_ships - opp_ships)


def _simulate(obs, move, n_steps: int, start_step: int,
              num_agents: int, player_id: int):
    sim = copy.deepcopy(obs)
    actions = [[] for _ in range(num_agents)]
    if move is not None:
        actions[player_id] = [move]
    interpreter(sim, actions, start_step, num_agents)
    no_actions = [[] for _ in range(num_agents)]
    for k in range(1, n_steps):
        interpreter(sim, no_actions, start_step + k, num_agents)
    return sim


def build_df_s_n(cache, obs, current_step: int, nb_steps: int):
    planets = obs.planets if hasattr(obs, "planets") else obs["planets"]
    fleets  = obs.fleets  if hasattr(obs, "fleets")  else obs["fleets"]

    ship_state  = {p[0]: p[5] for p in planets}
    owner_state = {p[0]: p[1] for p in planets}
    production  = {p[0]: p[6] for p in planets}
    radius_map  = {p[0]: p[4] for p in planets}

    arrivals_by_step = {}
    for fleet in fleets:
        fid     = fleet[0]
        arrival = cache._fleet_arrival.get(fid)
        if arrival is None:
            arrival = cache._compute_fleet_arrival(fleet, current_step)
        if arrival is not None:
            planet_id, arrival_step = arrival
            arrivals_by_step.setdefault(arrival_step, []).append((fleet, planet_id))

    rows = []
    for k in range(nb_steps + 1):
        game_step = current_step + k
        for pid in list(ship_state.keys()):
            pos = cache._planet_pos(pid, game_step)
            if pos is None:
                continue
            x, y = pos
            meta = cache._planet_meta.get(pid)
            if meta is None:
                continue
            rows.append({
                "step": game_step,
                "id": pid,
                "x": x,
                "y": y,
                "radius": radius_map[pid],
                "ships": ship_state[pid],
                "production": production[pid],
                "owner": owner_state[pid],
                "nature": meta["nature"],
            })

        if k == nb_steps:
            break

        new_ships = dict(ship_state)
        new_owner = dict(owner_state)
        for pid, owner in owner_state.items():
            if owner != -1:
                new_ships[pid] += production[pid]

        planet_arrivals = {}
        for fleet, planet_id in arrivals_by_step.get(game_step, []):
            if planet_id in ship_state:
                planet_arrivals.setdefault(planet_id, []).append(fleet)

        for planet_id, fleet_list in planet_arrivals.items():
            player_ships = {}
            for fleet in fleet_list:
                fowner = fleet[1]
                player_ships[fowner] = player_ships.get(fowner, 0) + fleet[6]
            sorted_players = sorted(player_ships.items(), key=lambda x: x[1], reverse=True)
            top_player, top_ships = sorted_players[0]
            if len(sorted_players) > 1:
                second_ships = sorted_players[1][1]
                survivor_ships = 0 if sorted_players[0][1] == sorted_players[1][1] else top_ships - second_ships
                survivor_owner = top_player if survivor_ships > 0 else -1
            else:
                survivor_owner = top_player
                survivor_ships = top_ships
            if survivor_ships > 0:
                if new_owner[planet_id] == survivor_owner:
                    new_ships[planet_id] += survivor_ships
                else:
                    new_ships[planet_id] -= survivor_ships
                    if new_ships[planet_id] < 0:
                        new_owner[planet_id] = survivor_owner
                        new_ships[planet_id] = abs(new_ships[planet_id])

        ship_state  = new_ships
        owner_state = new_owner

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
                pl.first("ships").alias("ships_src"),
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

        # All-in: always send current ships at source
        expanded_lf = coarse_lf.with_columns(pl.col("ships_src").alias("ships_sent"))

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
    def _04_minimax_search(safe_lf: pl.LazyFrame, obs, cache, player_id: int) -> list:
        attacks_with_angle = safe_lf.collect()
        moves_out = []

        # ── Comet evasion (preserved from original _04) ───────────────────
        if not attacks_with_angle.is_empty():
            awa_comets = attacks_with_angle.filter(pl.col("nature_src") == "comet")
            if not awa_comets.is_empty():
                x_off = (awa_comets["x_src"] - GameConfig.CENTER).abs().max() or 0
                y_off = (awa_comets["y_src"] - GameConfig.CENTER).abs().max() or 0
                if max(x_off, y_off) > 45:
                    moves_out += [list(r) for r in (
                        awa_comets
                        .sort(["ships_sent", "step"], descending=[True, False])
                        .group_by("id_src", maintain_order=True)
                        .first()
                        .select(["id_src", "final_angle", "ships_sent"])
                        .rows()
                    )]
                    id_to_avoid = awa_comets["id_src"].unique().to_list()
                    attacks_with_angle = attacks_with_angle.filter(
                        ~pl.col("id_src").is_in(id_to_avoid)
                    )

        # ── Step-0 candidates: top-5 per source (plus "do nothing") ──────
        NB_STEPS_5 = 5
        num_agents   = cache.num_agents
        current_step = cache.step

        step0_candidates = [None]  # None means "do nothing"
        if not attacks_with_angle.is_empty():
            top5_df = (
                attacks_with_angle
                .sort(["step", "ships_sent"])
                .group_by(["id_src", "id"], maintain_order=True)
                .first()
                .sort(["step", "ships_sent"])
                .group_by("id_src", maintain_order=True)
                .head(5)
            )
            for row in top5_df.select(["id_src", "final_angle", "ships_sent"]).iter_rows():
                step0_candidates.append(list(row))

        # ── Base cache: "do nothing" → step-5 candidate table ────────────
        obs_base5 = _simulate(obs, None, NB_STEPS_5, current_step, num_agents, player_id)
        df_s5_base, pd5_base = build_df_s_n(
            cache, obs_base5, current_step + NB_STEPS_5, NB_STEPS_5
        )
        pa5_base   = StrategyPipeline._02_get_all_opportunities(df_s5_base, pd5_base, player_id)
        safe5_base = StrategyPipeline._03_filter_collision(pa5_base).collect()

        base_planets5 = {
            p[0]: (p[1], p[5])
            for p in (obs_base5.planets if hasattr(obs_base5, "planets") else obs_base5["planets"])
        }

        best_score: tuple | None = None
        best_c0 = None

        for c0 in step0_candidates:
            # Simulate this step-0 candidate to step 5
            obs_c5 = _simulate(obs, c0, NB_STEPS_5, current_step, num_agents, player_id)
            c5_planets = obs_c5.planets if hasattr(obs_c5, "planets") else obs_c5["planets"]

            # Detect which of MY planets changed (ownership or ship count)
            changed_ids: set = set()
            for p in c5_planets:
                pid, owner = p[0], p[1]
                if owner != player_id:
                    continue
                base = base_planets5.get(pid)
                if base is None or base[0] != owner or base[1] != p[5]:
                    changed_ids.add(pid)

            # Build merged step-5 candidate table
            if changed_ids:
                df_s5_c, pd5_c = build_df_s_n(
                    cache, obs_c5, current_step + NB_STEPS_5, NB_STEPS_5
                )
                pa5_c   = StrategyPipeline._02_get_all_opportunities(df_s5_c, pd5_c, player_id)
                safe5_c = StrategyPipeline._03_filter_collision(pa5_c).collect()

                changed_list = list(changed_ids)
                new_rows  = safe5_c.filter(pl.col("id_src").is_in(changed_list)) if not safe5_c.is_empty() else safe5_c
                keep_rows = safe5_base.filter(~pl.col("id_src").is_in(changed_list)) if not safe5_base.is_empty() else safe5_base
                parts = [df for df in [keep_rows, new_rows] if not df.is_empty()]
                merged5 = pl.concat(parts) if parts else pl.DataFrame()
            else:
                merged5 = safe5_base

            # Enumerate step-5 candidates (plus "do nothing")
            step5_candidates = [None]
            if not merged5.is_empty():
                for row in merged5.select(["id_src", "final_angle", "ships_sent"]).iter_rows():
                    step5_candidates.append(list(row))

            # Evaluate each leaf
            best_score_5: tuple | None = None
            for c5 in step5_candidates:
                obs_leaf = _simulate(
                    obs_c5, c5, NB_STEPS_5, current_step + NB_STEPS_5, num_agents, player_id
                )
                score = evaluate(obs_leaf, player_id)
                if best_score_5 is None or score > best_score_5:
                    best_score_5 = score

            if best_score is None or best_score_5 > best_score:
                best_score  = best_score_5
                best_c0     = c0

        if best_c0 is not None:
            moves_out.append(best_c0)

        if moves_out:
            print(f"Minimax best move: {moves_out[-1]}  score={best_score}")
        return moves_out

    @staticmethod
    def _04_score_and_decide(safe_lf: pl.LazyFrame, player_id: int) -> list:
        attacks_with_angle = safe_lf.collect()
        if attacks_with_angle.is_empty():
            return []

        moves = []

        # Comet evasion — all-in flee (no ships_min guard)
        awa_comets = attacks_with_angle.filter(pl.col("nature_src") == "comet")
        if not awa_comets.is_empty():
            x_off = (awa_comets["x_src"] - GameConfig.CENTER).abs().max() or 0
            y_off = (awa_comets["y_src"] - GameConfig.CENTER).abs().max() or 0
            if max(x_off, y_off) > 45:
                moves += [list(r) for r in (
                    awa_comets
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

        # Unified 72-style scoring
        attacks = (
            attacks_with_angle
            .with_columns(
                pl.when(pl.col("owner") == -1)
                .then(pl.col("ships"))
                .otherwise(pl.col("ships") + pl.col("production"))
                .alias("ships_needed")
            )
            .filter(pl.col("ships_needed") < pl.col("ships_sent"))
            .sort(["step", "ships_sent"])
            .group_by(["id_src", "id"], maintain_order=True)
            .first()
            .join(top5_ids, on=["id_src", "id"], how="left")
            .with_columns(pl.col("is_top5").fill_null(False))
            .with_columns(
                (pl.col("ships_needed") / pl.col("production_src")).alias("time_cost")
            )
            .with_columns(
                pl.col("time_cost").sum().over("id_src").alias("total_time_cost")
            )
            .with_columns(
                (
                    (pl.col("total_time_cost") - pl.col("time_cost") - pl.col("step_diff"))
                    * pl.col("production")
                    - pl.when(~pl.col("is_top5")).then(pl.lit(100.0)).otherwise(pl.lit(0.0))
                    - pl.when(pl.col("owner") == player_id).then(pl.lit(100.0)).otherwise(pl.lit(0.0))
                ).alias("score")
            )
            .filter(pl.col("score") > 0)
            .sort("score", descending=True)
            .group_by("id_src", maintain_order=True)
            .first()
        )

        if attacks.is_empty():
            return moves

        for row in attacks.iter_rows(named=True):
            print(f"From {row['id_src']}, To {row['id']} at step {row['step']} "
                  f"with {row['ships_sent']} ships (target has min {row['ships_min']})")

        moves += [list(r) for r in attacks.select(["id_src", "final_angle", "ships_sent"]).rows()]
        return moves


# ── GameCache ─────────────────────────────────────────────────────────────────
class GameCache:
    def __init__(self, obs, step: int, num_agents: int, player_id: int):
        self.step = step
        self.num_agents = num_agents
        self.player_id = player_id
        self.angular_velocity = obs.angular_velocity if hasattr(obs, "angular_velocity") else obs["angular_velocity"]
        self._init_planet_meta(obs)
        self._pos_window: dict = {}
        self._fill_pos_window(obs, step)
        self._fleet_arrival: dict = {}
        fleets = obs.fleets if hasattr(obs, "fleets") else obs["fleets"]
        for fleet in fleets:
            fid = fleet[0]
            self._fleet_arrival[fid] = self._compute_fleet_arrival(fleet, step)

    # ── Planet metadata ───────────────────────────────────────────────────────

    def _init_planet_meta(self, obs):
        self._comet_pid_set = set(obs.comet_planet_ids if hasattr(obs, "comet_planet_ids") else obs["comet_planet_ids"])

        self._comet_path_by_pid = {}
        self._comet_idx_by_pid = {}
        comets = obs.comets if hasattr(obs, "comets") else obs["comets"]
        for group in comets:
            for i, pid in enumerate(group["planet_ids"]):
                self._comet_path_by_pid[pid] = group["paths"][i]
                self._comet_idx_by_pid[pid] = group["path_index"]

        initial_planets = obs.initial_planets if hasattr(obs, "initial_planets") else obs["initial_planets"]
        initial_by_id = {p[0]: p for p in initial_planets}

        self._planet_meta = {}
        planets = obs.planets if hasattr(obs, "planets") else obs["planets"]
        for p in planets:
            pid, owner, x, y, radius, ships, production = p[0], p[1], p[2], p[3], p[4], p[5], p[6]
            r = math.hypot(x - CENTER, y - CENTER)
            if pid in self._comet_pid_set:
                nature = "comet"
            elif r + radius < ROTATION_RADIUS_LIMIT:
                nature = "moving"
            else:
                nature = "fix"
            meta = {"nature": nature, "radius": radius, "production": production}
            if nature == "moving":
                ip = initial_by_id.get(pid)
                if ip is not None:
                    dx, dy = ip[2] - CENTER, ip[3] - CENTER
                    meta["r"] = math.sqrt(dx * dx + dy * dy)
                    meta["theta0"] = math.atan2(dy, dx)
            elif nature == "fix":
                meta["pos"] = (x, y)
            self._planet_meta[pid] = meta

    def _fill_pos_window(self, obs, step: int):
        planets = obs.planets if hasattr(obs, "planets") else obs["planets"]
        for p in planets:
            pid = p[0]
            self._pos_window[(pid, step)] = (p[2], p[3])
        for k in range(1, GameConfig.NB_STEPS_SIM + 2):
            game_step = step + k
            for pid in self._planet_meta:
                pos = self._planet_pos_analytical(pid, game_step)
                if pos is not None:
                    self._pos_window[(pid, game_step)] = pos

    def _planet_pos_analytical(self, pid: int, game_step: int):
        meta = self._planet_meta.get(pid)
        if meta is None:
            return None
        nature = meta["nature"]
        if nature == "moving":
            theta = meta["theta0"] + self.angular_velocity * (game_step - 1)
            return (CENTER + meta["r"] * math.cos(theta), CENTER + meta["r"] * math.sin(theta))
        if nature == "fix":
            return meta["pos"]
        path = self._comet_path_by_pid.get(pid)
        idx = self._comet_idx_by_pid.get(pid)
        if path is None or idx is None:
            return None
        comet_idx = idx + (game_step - self.step)
        if 0 <= comet_idx < len(path):
            return (path[comet_idx][0], path[comet_idx][1])
        return None

    def _planet_pos(self, pid: int, game_step: int):
        cached = self._pos_window.get((pid, game_step))
        if cached is not None:
            return cached
        return self._planet_pos_analytical(pid, game_step)

    # ── Fleet arrival (computed fresh each call) ──────────────────────────────

    def _compute_fleet_arrival(self, fleet, current_step: int):
        _, owner, fx, fy, angle, src_id, ships = fleet
        speed = PhysicsEngine.fleet_speed(ships)
        dx_f = math.cos(angle) * speed
        dy_f = math.sin(angle) * speed

        non_comet_pids = [pid for pid in self._planet_meta if pid not in self._comet_pid_set]
        comet_pids = list(self._comet_pid_set)

        for k in range(GameConfig.NB_STEPS_SIM):
            f_old = (fx + k * dx_f, fy + k * dy_f)
            f_new = (fx + (k + 1) * dx_f, fy + (k + 1) * dy_f)
            game_step = current_step + k

            for pid in non_comet_pids:
                p_old = self._planet_pos(pid, game_step)
                p_new = self._planet_pos(pid, game_step + 1)
                if p_old is None or p_new is None:
                    continue
                radius = self._planet_meta[pid]["radius"]
                if PhysicsEngine.swept_pair_hit(f_old, f_new, p_old, p_new, radius):
                    return (pid, game_step)

            if not (0 <= f_new[0] <= BOARD_SIZE and 0 <= f_new[1] <= BOARD_SIZE):
                return None
            if PhysicsEngine.point_to_segment_distance((CENTER, CENTER), f_old, f_new) < SUN_RADIUS:
                return None

            for pid in comet_pids:
                c_old = self._planet_pos(pid, game_step)
                c_new = self._planet_pos(pid, game_step + 1)
                if c_old is None or c_new is None:
                    continue
                radius = self._planet_meta[pid]["radius"]
                if PhysicsEngine.point_to_segment_distance(f_new, c_old, c_new) < radius:
                    return (pid, game_step)

        return None

    # ── Advance ───────────────────────────────────────────────────────────────

    def advance(self, obs, step: int):
        old_step = self.step
        self.step = step

        # Fleet arrival cache: evict departed fleets, compute new ones
        fleets = obs.fleets if hasattr(obs, "fleets") else obs["fleets"]
        current_fleet_ids = {f[0] for f in fleets}
        for fid in list(self._fleet_arrival.keys()):
            if fid not in current_fleet_ids:
                del self._fleet_arrival[fid]
        for fleet in fleets:
            fid = fleet[0]
            if fid not in self._fleet_arrival:
                self._fleet_arrival[fid] = self._compute_fleet_arrival(fleet, step)

        # Sync expired comets; evict their window entries
        current_comet_pids = set(obs.comet_planet_ids if hasattr(obs, "comet_planet_ids") else obs["comet_planet_ids"])
        for pid in list(self._comet_pid_set - current_comet_pids):
            self._planet_meta.pop(pid, None)
            self._comet_path_by_pid.pop(pid, None)
            self._comet_idx_by_pid.pop(pid, None)
            for s in range(old_step, old_step + GameConfig.NB_STEPS_SIM + 3):
                self._pos_window.pop((pid, s), None)
        self._comet_pid_set = current_comet_pids

        # Evict the just-passed step from the window
        for pid in self._planet_meta:
            self._pos_window.pop((pid, old_step), None)

        # Add the new far-end step
        new_step = step + GameConfig.NB_STEPS_SIM + 1
        for pid in self._planet_meta:
            pos = self._planet_pos_analytical(pid, new_step)
            if pos is not None:
                self._pos_window[(pid, new_step)] = pos

    # ── Build df_s analytically ───────────────────────────────────────────────

    def build_df_s(self, obs, current_step: int):
        planets = obs.planets if hasattr(obs, "planets") else obs["planets"]
        fleets = obs.fleets if hasattr(obs, "fleets") else obs["fleets"]

        ship_state = {p[0]: p[5] for p in planets}
        owner_state = {p[0]: p[1] for p in planets}
        production = {p[0]: p[6] for p in planets}
        radius_map = {p[0]: p[4] for p in planets}

        arrivals_by_step = {}
        for fleet in fleets:
            fid = fleet[0]
            arrival = self._fleet_arrival.get(fid)
            if arrival is not None:
                planet_id, arrival_step = arrival
                arrivals_by_step.setdefault(arrival_step, []).append((fleet, planet_id))

        rows = []
        for k in range(GameConfig.NB_STEPS_SIM + 1):
            game_step = current_step + k

            for pid in list(ship_state.keys()):
                pos = self._planet_pos(pid, game_step)
                if pos is None:
                    continue
                x, y = pos
                meta = self._planet_meta.get(pid)
                if meta is None:
                    continue
                rows.append({
                    "step": game_step,
                    "id": pid,
                    "x": x,
                    "y": y,
                    "radius": radius_map[pid],
                    "ships": ship_state[pid],
                    "production": production[pid],
                    "owner": owner_state[pid],
                    "nature": meta["nature"],
                })

            if k == GameConfig.NB_STEPS_SIM:
                break

            # Production before combat (matches interpreter order)
            new_ships = dict(ship_state)
            new_owner = dict(owner_state)
            for pid, owner in owner_state.items():
                if owner != -1:
                    new_ships[pid] += production[pid]

            planet_arrivals = {}
            for fleet, planet_id in arrivals_by_step.get(game_step, []):
                if planet_id in ship_state:
                    planet_arrivals.setdefault(planet_id, []).append(fleet)

            for planet_id, fleet_list in planet_arrivals.items():
                player_ships = {}
                for fleet in fleet_list:
                    fowner = fleet[1]
                    player_ships[fowner] = player_ships.get(fowner, 0) + fleet[6]
                sorted_players = sorted(player_ships.items(), key=lambda x: x[1], reverse=True)
                top_player, top_ships = sorted_players[0]
                if len(sorted_players) > 1:
                    second_ships = sorted_players[1][1]
                    if sorted_players[0][1] == sorted_players[1][1]:
                        survivor_ships = 0
                    else:
                        survivor_ships = top_ships - second_ships
                    survivor_owner = top_player if survivor_ships > 0 else -1
                else:
                    survivor_owner = top_player
                    survivor_ships = top_ships
                if survivor_ships > 0:
                    if new_owner[planet_id] == survivor_owner:
                        new_ships[planet_id] += survivor_ships
                    else:
                        new_ships[planet_id] -= survivor_ships
                        if new_ships[planet_id] < 0:
                            new_owner[planet_id] = survivor_owner
                            new_ships[planet_id] = abs(new_ships[planet_id])

            ship_state = new_ships
            owner_state = new_owner

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


# ── Entry point ───────────────────────────────────────────────────────────────
CACHE: GameCache | None = None


def agent(obs):
    global CACHE
    print(f"Agent called step: {CACHE.step if CACHE else 0} "
          f"remainingOverageTime: {obs.get('remainingOverageTime', 0) if isinstance(obs, dict) else 0}")
    if CACHE is None:
        initial = obs.initial_planets if hasattr(obs, "initial_planets") else obs["initial_planets"]
        owners = {p[1] for p in initial if p[1] != -1}
        num_agents = 4 if len(owners) > 2 else 2
        player_id = obs.get("player", 0) if isinstance(obs, dict) else obs.player
        CACHE = GameCache(obs, step=0, num_agents=num_agents, player_id=player_id)
    else:
        CACHE.advance(obs, CACHE.step + 1)
    df_s, planet_disp = CACHE.build_df_s(obs, CACHE.step)
    pa_lf = StrategyPipeline._02_get_all_opportunities(df_s, planet_disp, CACHE.player_id)
    safe_lf = StrategyPipeline._03_filter_collision(pa_lf)
    moves = StrategyPipeline._04_minimax_search(safe_lf, obs, CACHE, CACHE.player_id)
    return moves
