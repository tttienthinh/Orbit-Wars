import math
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


class StrategyPipeline:
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


class Board:
    def __init__(self, obs, step: int, num_agents: int, player_id: int):
        self.step = step
        self.num_agents = num_agents
        self.player_id = player_id
        self.angular_velocity = obs.angular_velocity if hasattr(obs, "angular_velocity") else obs["angular_velocity"]

        self._init_nature_and_pos(obs)
        self.df_fleet = pl.DataFrame(schema={
            "id": pl.Int64, "owner": pl.Int64, "ships": pl.Int64,
            "id_tgt": pl.Int64, "step_tgt": pl.Int64,
        })
        self.df_fleet_sim = pl.DataFrame(schema={
            "id_src": pl.Int64, "step_src": pl.Int64, "ships_sent": pl.Int64,
            "owner": pl.Int64, "id_tgt": pl.Int64, "step_tgt": pl.Int64,
        })
        self.df_planete_ships = pl.DataFrame(schema={
            "id": pl.Int64, "step": pl.Int64, "ships": pl.Int64,
            "owner": pl.Int64, "recompute": pl.Boolean,
        })

    def _init_nature_and_pos(self, obs):
        planets  = obs.planets         if hasattr(obs, "planets")         else obs["planets"]
        initial  = obs.initial_planets if hasattr(obs, "initial_planets") else obs["initial_planets"]
        comets   = obs.comets          if hasattr(obs, "comets")          else obs["comets"]
        cpids    = obs.comet_planet_ids if hasattr(obs, "comet_planet_ids") else obs["comet_planet_ids"]

        self._comet_pid_set = set(cpids)
        self._comet_path_by_pid: dict = {}
        self._comet_idx_by_pid:  dict = {}
        for group in comets:
            for i, pid in enumerate(group["planet_ids"]):
                self._comet_path_by_pid[pid] = group["paths"][i]
                self._comet_idx_by_pid[pid]  = group["path_index"]

        initial_by_id = {p[0]: p for p in initial}
        nature_rows = []
        for p in planets:
            pid, owner, x, y, radius, ships, production = p[0], p[1], p[2], p[3], p[4], p[5], p[6]
            r = math.hypot(x - CENTER, y - CENTER)
            if pid in self._comet_pid_set:
                nature = "comet"
            elif r + radius < ROTATION_RADIUS_LIMIT:
                nature = "moving"
            else:
                nature = "fix"
            nature_rows.append({"id": pid, "radius": radius, "production": production, "nature": nature})

        self._planet_meta: dict = {}
        for row in nature_rows:
            pid = row["id"]
            meta: dict = {"nature": row["nature"]}
            if row["nature"] == "moving":
                ip = initial_by_id.get(pid)
                if ip is not None:
                    dx, dy = ip[2] - CENTER, ip[3] - CENTER
                    meta["r"]      = math.sqrt(dx * dx + dy * dy)
                    meta["theta0"] = math.atan2(dy, dx)
            elif row["nature"] == "fix":
                p_obj = next(pp for pp in planets if pp[0] == pid)
                meta["pos"] = (p_obj[2], p_obj[3])
            meta["radius"] = row["radius"]
            self._planet_meta[pid] = meta

        self.df_planete_nature = pl.DataFrame(
            nature_rows,
            schema={"id": pl.Int64, "radius": pl.Float64, "production": pl.Int64, "nature": pl.Utf8},
        )

        # Build initial position window: steps self.step .. self.step+NB_STEPS_SIM
        pos_rows = []
        for pid in self._planet_meta:
            for k in range(GameConfig.NB_STEPS_SIM + 1):
                game_step = self.step + k
                pos = self._planet_pos_analytical(pid, game_step)
                if pos is not None:
                    pos_rows.append({"id": pid, "step": game_step, "x": pos[0], "y": pos[1]})
        self.df_planete_pos = pl.DataFrame(
            pos_rows,
            schema={"id": pl.Int64, "step": pl.Int64, "x": pl.Float64, "y": pl.Float64},
        )

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
        idx  = self._comet_idx_by_pid.get(pid)
        if path is None or idx is None:
            return None
        comet_idx = idx + (game_step - self.step)
        if 0 <= comet_idx < len(path):
            return (path[comet_idx][0], path[comet_idx][1])
        return None


BOARD: "Board | None" = None


def agent(obs):
    raise NotImplementedError


if __name__ == "__main__":
    import time
    from types import SimpleNamespace

    def _make_obs():
        # Planet 0: r=20, radius=31 → r+radius=51 >= 50 → "fix" (static)
        # Planet 1 and 2 are "moving" (orbiting)
        planets = [
            [0, 0, 30.0, 50.0, 31.0, 100, 2],
            [1, 1, 70.0, 50.0, 5.0,   20, 2],
            [2, -1, 50.0, 20.0, 4.0,   0, 1],
        ]
        return SimpleNamespace(
            planets=[list(p) for p in planets],
            initial_planets=[list(p) for p in planets],
            fleets=[],
            comets=[],
            comet_planet_ids=[],
            angular_velocity=0.01,
            next_fleet_id=0,
            player=0,
        )

    obs0 = _make_obs()
    board = Board(obs0, step=0, num_agents=2, player_id=0)

    # df_planete_nature
    assert board.df_planete_nature.shape == (3, 4), \
        f"Expected (3,4), got {board.df_planete_nature.shape}"
    assert set(board.df_planete_nature.columns) == {"id", "radius", "production", "nature"}
    natures = dict(zip(
        board.df_planete_nature["id"].to_list(),
        board.df_planete_nature["nature"].to_list(),
    ))
    assert natures[0] == "fix", f"Planet 0 should be fix, got {natures[0]}"

    # df_planete_pos: 3 planets × 11 steps = 33 rows
    assert board.df_planete_pos.shape[0] == 33, \
        f"Expected 33 pos rows, got {board.df_planete_pos.shape[0]}"
    assert board.df_planete_pos["step"].min() == 0
    assert board.df_planete_pos["step"].max() == 10

    # _planet_pos_analytical for a static planet returns constant position
    pos0 = board._planet_pos_analytical(0, 5)
    assert pos0 == (30.0, 50.0), f"Expected (30.0, 50.0), got {pos0}"

    print("Task 2 PASSED")
