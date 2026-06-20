"""132-GNN_BC_shift_torch_only — pure-PyTorch inference agent for GNNBehaviourCloning.

Architecture matches 130-GNN_BC_shift.py (H=64, L1=3, L2=2).
Weights loaded from model_epoch100.pt in same directory.
No torch_geometric dependency.
"""
import math
import copy
import types
from pathlib import Path

import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Constants ─────────────────────────────────────────────────────────────────
CENTER                = 50.0
SUN_RADIUS            = 10.0
ROTATION_RADIUS_LIMIT = 50.0
MAX_SPEED             = 6.0
PLANET_MARGIN         = 0.1
PLANET_MOVEMENT_SLACK = 3.0
BOARD_SIZE            = 100.0
NB_STEPS_SIM          = 20
REACH_SHIPS           = [8, 64, 512]
LOG_1024              = math.log(1024.0)
LOGIT_THRESHOLD       = math.log(576.0)  # calibrated to training positive rate 1/577


# ── PyTorch primitives ────────────────────────────────────────────────────────

def _mean_agg(h_src: torch.Tensor, edge_index: torch.Tensor, num_dst: int) -> torch.Tensor:
    src_i, dst_i = edge_index
    D = h_src.shape[1]
    agg = h_src.new_zeros(num_dst, D)
    cnt = h_src.new_zeros(num_dst, 1)
    agg.scatter_add_(0, dst_i.unsqueeze(1).expand(-1, D), h_src[src_i])
    cnt.scatter_add_(0, dst_i.unsqueeze(1), h_src.new_ones(len(dst_i), 1))
    return agg / cnt.clamp(min=1)


def _scaled_mean_agg(
    h: torch.Tensor,
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
    num_nodes: int,
) -> torch.Tensor:
    src_i, dst_i = edge_index
    D = h.shape[1]
    msgs = h[src_i] * edge_attr.view(-1, 1)
    agg = h.new_zeros(num_nodes, D)
    cnt = h.new_zeros(num_nodes, 1)
    agg.scatter_add_(0, dst_i.unsqueeze(1).expand(-1, D), msgs)
    cnt.scatter_add_(0, dst_i.unsqueeze(1), h.new_ones(len(dst_i), 1))
    return agg / cnt.clamp(min=1)


class _SAGEConvPure(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.lin_l = nn.Linear(in_ch, out_ch, bias=True)
        self.lin_r = nn.Linear(in_ch, out_ch, bias=False)

    def forward(self, h_src, h_dst, edge_index, num_dst):
        return self.lin_l(h_dst) + self.lin_r(_mean_agg(h_src, edge_index, num_dst))


class _ScaledSAGEConvPure(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.lin_neigh = nn.Linear(in_ch, out_ch, bias=False)
        self.lin_root  = nn.Linear(in_ch, out_ch, bias=True)

    def forward(self, h, edge_index, edge_attr, num_nodes):
        return self.lin_neigh(_scaled_mean_agg(h, edge_index, edge_attr, num_nodes)) + self.lin_root(h)


class _Phase1ConvPure(nn.Module):
    def __init__(self, H: int):
        super().__init__()
        self.has_ps     = _SAGEConvPure(H, H)
        self.rev_has_ps = _SAGEConvPure(H, H)
        self.reaches    = _ScaledSAGEConvPure(H, H)

    def forward(self, h_planet, h_ps, ei_has_ps, ei_rev_has_ps, ei_reaches, reaches_attr):
        n_planet = h_planet.shape[0]
        n_ps     = h_ps.shape[0]
        new_ps = self.has_ps(h_planet, h_ps, ei_has_ps, n_ps)
        if ei_reaches.shape[1] > 0:
            new_ps = new_ps + self.reaches(h_ps, ei_reaches, reaches_attr, n_ps)
        new_planet = self.rev_has_ps(h_ps, h_planet, ei_rev_has_ps, n_planet)
        return new_planet, new_ps


class _Phase2ConvPure(nn.Module):
    def __init__(self, H: int):
        super().__init__()
        self.src_of     = _SAGEConvPure(H, H)
        self.rev_src_of = _SAGEConvPure(H, H)
        self.tgt_of     = _SAGEConvPure(H, H)
        self.rev_tgt_of = _SAGEConvPure(H, H)

    def forward(self, h_ps, h_ams, ei_src_of, ei_rev_src_of, ei_tgt_of, ei_rev_tgt_of):
        n_ps  = h_ps.shape[0]
        n_ams = h_ams.shape[0]
        new_ams = (
            self.src_of(h_ps, h_ams, ei_src_of, n_ams) +
            self.tgt_of(h_ps, h_ams, ei_tgt_of, n_ams)
        )
        new_ps = (
            self.rev_src_of(h_ams, h_ps, ei_rev_src_of, n_ps) +
            self.rev_tgt_of(h_ams, h_ps, ei_rev_tgt_of, n_ps)
        )
        return new_ps, new_ams


class GNNBehaviourCloningPure(nn.Module):
    def __init__(self, H: int = 64, L1: int = 3, L2: int = 2):
        super().__init__()
        self.proj_planet  = nn.Linear(4, H)
        self.proj_ps      = nn.Linear(9, H)
        self.proj_ams     = nn.Linear(1, H)
        self.phase1       = nn.ModuleList([_Phase1ConvPure(H) for _ in range(L1)])
        self.norm1_planet = nn.ModuleList([nn.LayerNorm(H) for _ in range(L1)])
        self.norm1_ps     = nn.ModuleList([nn.LayerNorm(H) for _ in range(L1)])
        self.phase2       = nn.ModuleList([_Phase2ConvPure(H) for _ in range(L2)])
        self.norm2_ps     = nn.ModuleList([nn.LayerNorm(H) for _ in range(L2)])
        self.norm2_ams    = nn.ModuleList([nn.LayerNorm(H) for _ in range(L2)])
        self.classifier   = nn.Linear(H, 1)

    def forward(self, data) -> torch.Tensor:
        h_planet = self.proj_planet(data.planet_x)
        h_ps     = self.proj_ps(data.ps_x)
        h_ams    = self.proj_ams(data.ams_x)

        for i, conv in enumerate(self.phase1):
            h_planet, h_ps = conv(
                h_planet, h_ps,
                data.ei_has_ps, data.ei_rev_has_ps,
                data.ei_reaches, data.reaches_attr,
            )
            h_planet = F.relu(self.norm1_planet[i](h_planet))
            h_ps     = F.relu(self.norm1_ps[i](h_ps))

        for i, conv in enumerate(self.phase2):
            h_ps, h_ams = conv(
                h_ps, h_ams,
                data.ei_src_of, data.ei_rev_src_of,
                data.ei_tgt_of, data.ei_rev_tgt_of,
            )
            h_ps  = F.relu(self.norm2_ps[i](h_ps))
            h_ams = F.relu(self.norm2_ams[i](h_ams))

        return self.classifier(h_ams).squeeze(-1)


# ── Weight loading ────────────────────────────────────────────────────────────

_KEY_MAP = {
    "convs.<planet___has_ps___planet_step>":         "has_ps",
    "convs.<planet_step___rev_has_ps___planet>":     "rev_has_ps",
    "convs.<planet_step___reaches___planet_step>":   "reaches",
    "convs.<planet_step___src_of___action_max>":     "src_of",
    "convs.<action_max___rev_src_of___planet_step>": "rev_src_of",
    "convs.<planet_step___tgt_of___action_max>":     "tgt_of",
    "convs.<action_max___rev_tgt_of___planet_step>": "rev_tgt_of",
}


def _load_weights(model: nn.Module, path) -> tuple:
    sd = torch.load(str(path), map_location="cpu", weights_only=True)
    new_sd = {}
    for k, v in sd.items():
        new_k = k
        for pyg_name, our_name in _KEY_MAP.items():
            new_k = new_k.replace(pyg_name, our_name)
        new_sd[new_k] = v
    missing, unexpected = model.load_state_dict(new_sd, strict=True)
    return list(missing), list(unexpected)


# ── Physics helpers ───────────────────────────────────────────────────────────

def _fleet_speed(ships: float) -> float:
    if ships <= 1:
        return 1.0
    ratio = math.log(ships) / math.log(1000.0)
    return 1.0 + (MAX_SPEED - 1.0) * max(0.0, min(1.0, ratio)) ** 1.5


def _point_seg_dist(p, v, w):
    l2 = (v[0] - w[0]) ** 2 + (v[1] - w[1]) ** 2
    if l2 == 0.0:
        return math.hypot(p[0] - v[0], p[1] - v[1])
    t = max(0.0, min(1.0, ((p[0]-v[0])*(w[0]-v[0]) + (p[1]-v[1])*(w[1]-v[1])) / l2))
    proj = (v[0] + t*(w[0]-v[0]), v[1] + t*(w[1]-v[1]))
    return math.hypot(p[0] - proj[0], p[1] - proj[1])


def _swept_pair_hit(A, B, P0, P1, r):
    d0x, d0y = A[0]-P0[0], A[1]-P0[1]
    dvx = (B[0]-A[0]) - (P1[0]-P0[0])
    dvy = (B[1]-A[1]) - (P1[1]-P0[1])
    a = dvx*dvx + dvy*dvy
    b = 2.0*(d0x*dvx + d0y*dvy)
    c = d0x*d0x + d0y*d0y - r*r
    if a < 1e-12:
        return c <= 0.0
    disc = b*b - 4.0*a*c
    if disc < 0.0:
        return False
    sq = math.sqrt(disc)
    t1 = (-b - sq) / (2.0*a)
    t2 = (-b + sq) / (2.0*a)
    return t2 >= 0.0 and t1 <= 1.0


# ── Interpreter (verbatim from 112-GNN_torch_only.py) ────────────────────────

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
            r_p  = math.sqrt(dx_p**2 + dy_p**2)
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
        speed = _fleet_speed(ships)
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
            if _swept_pair_hit(f_old, f_new, p_old, p_new, planet[4]):
                combat_lists[planet[0]].append(fleet)
                fleets_to_remove.append(fleet)
                hit_planet = True
                break
        if hit_planet:
            continue
        if not (0 <= fleet[2] <= BOARD_SIZE and 0 <= fleet[3] <= BOARD_SIZE):
            fleets_to_remove.append(fleet)
            continue
        if _point_seg_dist((CENTER, CENTER), f_old, f_new) < SUN_RADIUS:
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
                            if _point_seg_dist((fleet[2], fleet[3]), c_old, c_new) < planet[4]:
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

    for pid_c, planet_fleets in combat_lists.items():
        planet = next((p for p in obs0.planets if p[0] == pid_c), None)
        if not planet or not planet_fleets:
            continue
        player_ships: dict = {}
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


# ── Observation → DataFrame ───────────────────────────────────────────────────

def _01_get_obs_dataframe(obs, step: int, num_agents: int):
    sim = copy.deepcopy(obs)
    no_actions = [[] for _ in range(num_agents)]
    rows = []
    for i in range(NB_STEPS_SIM + 1):
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
                "step": step + i, "id": pid, "x": x, "y": y,
                "radius": radius, "ships": ships,
                "production": production, "owner": owner, "nature": nature,
            })
        _interpreter(sim, no_actions, step + i, num_agents)

    df_s = pl.DataFrame(rows).sort("step")

    prev_pos = (
        df_s.lazy().select(["id", "step", "x", "y"])
        .rename({"x": "x_prev", "y": "y_prev"})
        .with_columns((pl.col("step") + 1).alias("step"))
    )
    planet_disp = (
        df_s.lazy().select(["id", "step", "x", "y"])
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


# ── Reachability helpers (shared geometry) ────────────────────────────────────

def _reach_geometry_tail(coarse_lf, df_s_lf, fleet_speed_expr, dist_min_expr, dist_prev_expr):
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

    return (
        coarse_lf
        .with_columns([
            fleet_speed_expr.alias("fleet_speed"),
            dist_min_expr.alias("dist_min"),
            dist_prev_expr.alias("dist_prev"),
        ])
        .filter(
            pl.col("dist_tgt_src") < pl.col("dist_min") + pl.col("fleet_speed")
            + pl.col("radius") + PLANET_MOVEMENT_SLACK
        )
        .join(prev_pos_lf, on=["id", "step"], how="left")
        .with_columns([t1_expr.alias("t1"), t2_expr.alias("t2"), collision.alias("collision")])
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
            ((pl.col("p_t1_x")-pl.col("x_src")).pow(2)+(pl.col("p_t1_y")-pl.col("y_src")).pow(2)).sqrt().alias("d_s_t1"),
            ((pl.col("p_t2_x")-pl.col("x_src")).pow(2)+(pl.col("p_t2_y")-pl.col("y_src")).pow(2)).sqrt().alias("d_s_t2"),
        ])
        .with_columns([
            (pl.col("dist_prev") + pl.col("t1_eff") * pl.col("fleet_speed")).alias("d_f_t1"),
            (pl.col("dist_prev") + pl.col("t2_eff") * pl.col("fleet_speed")).alias("d_f_t2"),
        ])
        .with_columns([
            ((pl.col("d_s_t1").pow(2)+pl.col("d_f_t1").pow(2)-pl.col("radius").pow(2))
             / (2.0*pl.col("d_s_t1")*pl.col("d_f_t1"))).clip(-1.0, 1.0).arccos().alias("angle_radius_t1"),
            ((pl.col("d_s_t2").pow(2)+pl.col("d_f_t2").pow(2)-pl.col("radius").pow(2))
             / (2.0*pl.col("d_s_t2")*pl.col("d_f_t2"))).clip(-1.0, 1.0).arccos().alias("angle_radius_t2"),
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


def _build_coarse(df_s_lf, planet_disp_lf, owner_filter=None):
    agg = [
        pl.first("step").alias("step_src"),
        pl.first("x").alias("x_src"),
        pl.first("y").alias("y_src"),
        pl.first("radius").alias("radius_src"),
        pl.min("ships").alias("ships_min"),
        pl.first("ships").alias("ships_at_src"),
        pl.first("production").alias("production_src"),
        pl.first("nature").alias("nature_src"),
        pl.first("owner").alias("owner_src"),
    ]
    base_lf = (
        df_s_lf.group_by("id", maintain_order=True).agg(agg)
        .rename({"id": "id_src"})
    )
    if owner_filter is not None:
        base_lf = base_lf.filter(pl.col("owner_src") == owner_filter)

    dx = pl.col("x") - pl.col("x_src")
    dy = pl.col("y") - pl.col("y_src")
    l2 = dx.pow(2) + dy.pow(2)
    dist_tgt_src = l2.sqrt()
    step_diff = (pl.col("step") - pl.col("step_src")).cast(pl.Float64)

    dot = (CENTER - pl.col("x_src")) * dx + (CENTER - pl.col("y_src")) * dy
    t_sun = (dot / pl.when(l2 == 0).then(pl.lit(1.0)).otherwise(l2)).clip(0.0, 1.0)
    proj_dist_sun = (
        (CENTER - pl.col("x_src") - t_sun * dx).pow(2) +
        (CENTER - pl.col("y_src") - t_sun * dy).pow(2)
    ).sqrt()
    crossing_sun = pl.when(l2 == 0).then(
        ((CENTER - pl.col("x_src")).pow(2) + (CENTER - pl.col("y_src")).pow(2)).sqrt()
    ).otherwise(proj_dist_sun) < (SUN_RADIUS + PLANET_MARGIN)

    return (
        base_lf.join(df_s_lf, how="cross")
        .filter((pl.col("step") > pl.col("step_src")) & (pl.col("id") != pl.col("id_src")))
        .join(planet_disp_lf, on=["id", "step"], how="left")
        .with_columns([dist_tgt_src.alias("dist_tgt_src"), step_diff.alias("step_diff")])
        .filter(
            (pl.col("dist_tgt_src") < (pl.col("step_diff") + 1) * MAX_SPEED
             + pl.col("radius_src") + PLANET_MARGIN + pl.col("radius")
             + pl.col("planet_disp").fill_null(0.0))
            & ~crossing_sun
        )
    )


def _02_get_reach(
    df_s: pl.DataFrame,
    planet_disp: pl.DataFrame,
    ships_list: list,
) -> pl.LazyFrame:
    df_s_lf = df_s.lazy()
    planet_disp_lf = planet_disp.lazy()

    coarse_lf = (
        _build_coarse(df_s_lf, planet_disp_lf)
        .with_columns(pl.lit(ships_list).alias("ships_sent"))
        .explode("ships_sent")
    )

    fs_expr = 1.0 + (MAX_SPEED - 1.0) * (
        pl.col("ships_sent").cast(pl.Float64).log(base=math.e) / math.log(1000.0)
    ).clip(lower_bound=0.0).pow(1.5)
    dm_expr = pl.col("step_diff") * fs_expr + PLANET_MARGIN + pl.col("radius_src")
    dp_expr = dm_expr - fs_expr

    return _reach_geometry_tail(coarse_lf, df_s_lf, fs_expr, dm_expr, dp_expr)


def _get_reach_max_ships(
    df_s: pl.DataFrame,
    planet_disp: pl.DataFrame,
) -> pl.LazyFrame:
    df_s_lf = df_s.lazy()
    planet_disp_lf = planet_disp.lazy()

    coarse_lf = (
        _build_coarse(df_s_lf, planet_disp_lf)
        .with_columns(pl.col("ships_at_src").alias("ships_sent"))
    )

    fs_expr = 1.0 + (MAX_SPEED - 1.0) * (
        pl.col("ships_sent").cast(pl.Float64).log(base=math.e) / math.log(1000.0)
    ).clip(lower_bound=0.0).pow(1.5)
    dm_expr = pl.col("step_diff") * fs_expr + PLANET_MARGIN + pl.col("radius_src")
    dp_expr = dm_expr - fs_expr

    return _reach_geometry_tail(coarse_lf, df_s_lf, fs_expr, dm_expr, dp_expr)


def _03_filter_collision(pa_lf: pl.LazyFrame) -> pl.LazyFrame:
    angle_norm = pl.col("angle") % (2 * math.pi)
    wraps    = pl.col("angle_min_obs") > pl.col("angle_max_obs")
    in_cone  = pl.when(wraps).then(
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

    return (
        pa_lf
        .join(blocked_lf, on=["id_src", "ships_sent", "step", "id"], how="anti")
        .with_columns(pl.col("angle").alias("final_angle"))
    )


# ── Graph construction for inference ─────────────────────────────────────────

def build_graph_inference(
    T: int,
    planete: pl.DataFrame,
    planete_step: pl.DataFrame,
    reachable_base_2: pl.DataFrame,
    reachable_max_ships: pl.DataFrame,
):
    """Returns (graph_namespace, action_table) or (None, None) if no action_max nodes."""
    # Planet nodes
    planet_df = planete.filter(pl.col("step") == T).unique("id", keep="first").sort("id")
    if planet_df.is_empty():
        return None, None

    planet_ids = planet_df["id"].to_list()
    max_pid    = max(planet_ids)
    pid_to_idx = np.full(max_pid + 1, -1, dtype=np.int64)
    for i, pid in enumerate(planet_ids):
        pid_to_idx[pid] = i

    nature_arr  = planet_df["nature"].to_numpy()
    planet_feat = np.column_stack([
        planet_df["production"].to_numpy().astype(np.float32) / 5.0,
        (nature_arr == "fix").astype(np.float32),
        (nature_arr == "moving").astype(np.float32),
        (nature_arr == "comet").astype(np.float32),
    ])

    # PlanetStep nodes
    ps_df_raw = planete_step.filter(pl.col("step") == T)
    if ps_df_raw.is_empty():
        return None, None

    planete_renamed = planete.rename({"step": "future_step"})
    ps_df = ps_df_raw.join(planete_renamed, on=["id", "future_step"], how="left")

    ps_ids          = ps_df["id"].to_numpy()
    ps_future_steps = ps_df["future_step"].to_numpy()
    ps_key_to_idx: dict = {
        (int(ps_ids[i]), int(ps_future_steps[i])): i
        for i in range(len(ps_df))
    }

    owner_arr  = ps_df["owner"].to_numpy()
    x_arr      = ps_df["x"].to_numpy().astype(np.float32)
    y_arr      = ps_df["y"].to_numpy().astype(np.float32)
    ships_arr  = np.log(np.clip(ps_df["ships"].to_numpy().astype(np.float32), 1, None)) / LOG_1024
    delta_arr  = (ps_future_steps.astype(np.float32) - T) / NB_STEPS_SIM
    ps_feat = np.column_stack([
        x_arr / 100.0, y_arr / 100.0, delta_arr, ships_arr,
        (owner_arr == -1).astype(np.float32),
        (owner_arr ==  0).astype(np.float32),
        (owner_arr ==  1).astype(np.float32),
        (owner_arr ==  2).astype(np.float32),
        (owner_arr ==  3).astype(np.float32),
    ])

    # Planet <-> PlanetStep edges
    pid_arr    = ps_df["id"].to_numpy()
    valid_pid  = pid_arr <= max_pid
    p_side_raw = np.where(valid_pid, pid_arr, 0)
    p_side_idx = pid_to_idx[p_side_raw]
    valid_mask = valid_pid & (p_side_idx >= 0)
    has_ps_src = p_side_idx[valid_mask].astype(np.int64)
    has_ps_dst = np.where(valid_mask)[0].astype(np.int64)

    ei_has_ps     = torch.tensor(np.stack([has_ps_src, has_ps_dst]), dtype=torch.long)
    ei_rev_has_ps = torch.tensor(np.stack([has_ps_dst, has_ps_src]), dtype=torch.long)

    # ActionBase2 edges (PlanetStep <-> PlanetStep)
    b2 = reachable_base_2.filter(
        (pl.col("step_src") >= T) & (pl.col("step_src") <= T + NB_STEPS_SIM) &
        (pl.col("step_tgt") <= T + NB_STEPS_SIM) &
        (pl.col("ships_sent").is_in(REACH_SHIPS))
    )
    b2_src_list: list = []
    b2_dst_list: list = []
    b2_w_list:   list = []
    if not b2.is_empty():
        b2_id_src   = b2["id_src"].to_numpy()
        b2_step_src = b2["step_src"].to_numpy()
        b2_id_tgt   = b2["id_tgt"].to_numpy()
        b2_step_tgt = b2["step_tgt"].to_numpy()
        b2_ships    = b2["ships_sent"].to_numpy().astype(np.float32)
        for i in range(len(b2)):
            sk = (int(b2_id_src[i]), int(b2_step_src[i]))
            tk = (int(b2_id_tgt[i]), int(b2_step_tgt[i]))
            si = ps_key_to_idx.get(sk, -1)
            ti = ps_key_to_idx.get(tk, -1)
            if si >= 0 and ti >= 0:
                b2_src_list.append(si); b2_dst_list.append(ti)
                b2_w_list.append(math.log(max(float(b2_ships[i]), 1.0)) / LOG_1024)

    if b2_src_list:
        ei_reaches   = torch.tensor(np.stack([np.array(b2_src_list, dtype=np.int64),
                                              np.array(b2_dst_list, dtype=np.int64)]), dtype=torch.long)
        reaches_attr = torch.tensor(b2_w_list, dtype=torch.float32)
    else:
        ei_reaches   = torch.zeros(2, 0, dtype=torch.long)
        reaches_attr = torch.zeros(0)

    # ActionMaxShips nodes (only my planets at step T)
    ps_now        = ps_df.filter(pl.col("future_step") == T)
    my_planet_ids = ps_now.filter(pl.col("owner") == 0)["id"].to_list()
    if not my_planet_ids:
        return None, None

    ams_raw = reachable_max_ships.filter(
        (pl.col("step_src") == T) & (pl.col("id_src").is_in(my_planet_ids))
    )
    if ams_raw.is_empty():
        return None, None

    ams_ships    = ams_raw["ships_sent"].to_numpy().astype(np.float32)
    ams_feat_all = (np.log(np.clip(ams_ships, 1, None)) / LOG_1024).reshape(-1, 1)

    ams_id_src   = ams_raw["id_src"].to_numpy()
    ams_id_tgt   = ams_raw["id_tgt"].to_numpy()
    ams_step_tgt = ams_raw["step_tgt"].to_numpy()

    src_ps_list: list = []
    tgt_ps_list: list = []
    valid_rows:  list = []
    for i in range(len(ams_raw)):
        sk = (int(ams_id_src[i]), T)
        tk = (int(ams_id_tgt[i]), int(ams_step_tgt[i]))
        sp = ps_key_to_idx.get(sk, -1)
        tp = ps_key_to_idx.get(tk, -1)
        if sp >= 0 and tp >= 0:
            src_ps_list.append(sp); tgt_ps_list.append(tp)
            valid_rows.append(i)

    if not valid_rows:
        return None, None

    vidx     = np.array(valid_rows, dtype=np.int64)
    ams_feat = ams_feat_all[vidx]
    n_ams    = len(valid_rows)
    src_ps   = np.array(src_ps_list, dtype=np.int64)
    tgt_ps   = np.array(tgt_ps_list, dtype=np.int64)
    aidx     = np.arange(n_ams, dtype=np.int64)

    graph = types.SimpleNamespace(
        planet_x      = torch.tensor(planet_feat, dtype=torch.float32),
        ps_x          = torch.tensor(ps_feat,     dtype=torch.float32),
        ams_x         = torch.tensor(ams_feat,    dtype=torch.float32),
        ei_has_ps     = ei_has_ps,
        ei_rev_has_ps = ei_rev_has_ps,
        ei_reaches    = ei_reaches,
        reaches_attr  = reaches_attr,
        ei_src_of     = torch.tensor(np.stack([src_ps, aidx]), dtype=torch.long),
        ei_rev_src_of = torch.tensor(np.stack([aidx, src_ps]), dtype=torch.long),
        ei_tgt_of     = torch.tensor(np.stack([tgt_ps, aidx]), dtype=torch.long),
        ei_rev_tgt_of = torch.tensor(np.stack([aidx, tgt_ps]), dtype=torch.long),
    )

    action_table = ams_raw[vidx.tolist()].select(["id_src", "angle", "ships_sent"])
    return graph, action_table


# ── Model loading ─────────────────────────────────────────────────────────────

_here         = Path(__file__).parent if "__file__" in dir() else Path(".")
_weights_path = _here / "model_epoch100.pt"

_model = GNNBehaviourCloningPure(H=64, L1=3, L2=2)
if _weights_path.exists():
    _missing, _unexpected = _load_weights(_model, _weights_path)
    if _missing or _unexpected:
        import sys
        print(f"[132] weight warnings: missing={_missing} unexpected={_unexpected}", file=sys.stderr)
_model.eval()


# ── Agent entry point ─────────────────────────────────────────────────────────

_agent_step:       dict = {}
_agent_num_agents: dict = {}
_agent_player_id:  dict = {}


def agent(obs):
    pid = obs.player if hasattr(obs, "player") else obs.get("player", 0)

    if pid not in _agent_step:
        _agent_step[pid] = 0

    if pid not in _agent_num_agents:
        initial = (
            obs.initial_planets if hasattr(obs, "initial_planets")
            else obs.get("initial_planets", [])
        )
        owners = {p[1] for p in initial if p[1] != -1}
        _agent_num_agents[pid] = 4 if len(owners) > 2 else 2

    T          = _agent_step[pid]
    num_agents = _agent_num_agents[pid]
    _agent_step[pid] += 1

    df_s, planet_disp = _01_get_obs_dataframe(obs, T, num_agents)
    df_s = _00_remap_owner(df_s, obs, pid)

    rb2_raw = _02_get_reach(df_s, planet_disp, REACH_SHIPS).collect()
    if rb2_raw.is_empty():
        return []
    rb2 = (
        _03_filter_collision(rb2_raw.lazy()).collect()
        .select(["id_src", "step_src", "id", "step", "ships_sent"])
        .rename({"id": "id_tgt", "step": "step_tgt"})
    )

    rms_raw = _get_reach_max_ships(df_s, planet_disp).collect()
    if rms_raw.is_empty():
        return []
    rms = (
        _03_filter_collision(rms_raw.lazy()).collect()
        .select(["id_src", "step_src", "final_angle", "ships_sent", "id", "step"])
        .rename({"final_angle": "angle", "id": "id_tgt", "step": "step_tgt"})
        .sort("step_tgt")
        .group_by(["id_src", "step_src", "id_tgt"], maintain_order=True)
        .first()
        .select(["id_src", "step_src", "angle", "ships_sent", "id_tgt", "step_tgt"])
    )

    planete = df_s.select(["id", "step", "x", "y", "production", "nature"])
    planete_step = (
        df_s.select(["id", "step", "ships", "owner"])
        .with_columns(pl.lit(T).cast(pl.Int64).alias("obs_step"))
        .rename({"obs_step": "step", "step": "future_step"})
        .select(["id", "step", "future_step", "ships", "owner"])
    )

    graph, action_table = build_graph_inference(T, planete, planete_step, rb2, rms)
    if graph is None:
        return []

    with torch.no_grad():
        logits = _model(graph)

    logits_np = logits.numpy()
    selected  = logits_np > LOGIT_THRESHOLD
    if not selected.any():
        return []

    best: dict = {}
    for i, (is_sel, lv) in enumerate(zip(selected, logits_np)):
        if not is_sel:
            continue
        id_src = int(action_table["id_src"][i])
        if id_src not in best or lv > best[id_src][0]:
            best[id_src] = (lv, i)

    moves = []
    for _, (_, row_idx) in best.items():
        row = action_table.row(row_idx, named=True)
        moves.append([int(row["id_src"]), float(row["angle"]), int(row["ships_sent"])])

    return moves


# ── Smoke test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # 1. Weight loading
    _tm = GNNBehaviourCloningPure(H=64, L1=3, L2=2)
    _wp = Path("130-GNN_BC_shift") / "model_epoch100.pt"
    if not _wp.exists():
        print(f"SKIP weight test: {_wp} not found")
    else:
        miss, unexp = _load_weights(_tm, _wp)
        assert not miss,  f"Missing keys: {miss}"
        assert not unexp, f"Unexpected keys: {unexp}"
        print("Weight loading: OK")

    # 2. Forward pass with synthetic graph
    N_p, N_ps, N_ams = 3, 6, 2
    _d = types.SimpleNamespace(
        planet_x     = torch.randn(N_p, 4),
        ps_x         = torch.randn(N_ps, 9),
        ams_x        = torch.randn(N_ams, 1),
        ei_has_ps    = torch.tensor([[0,0,1,1,2,2],[0,1,2,3,4,5]], dtype=torch.long),
        ei_rev_has_ps= torch.tensor([[0,1,2,3,4,5],[0,0,1,1,2,2]], dtype=torch.long),
        ei_reaches   = torch.tensor([[0,2],[1,3]], dtype=torch.long),
        reaches_attr = torch.tensor([0.3, 0.7]),
        ei_src_of    = torch.tensor([[0,1],[0,1]], dtype=torch.long),
        ei_rev_src_of= torch.tensor([[0,1],[0,1]], dtype=torch.long),
        ei_tgt_of    = torch.tensor([[2,3],[0,1]], dtype=torch.long),
        ei_rev_tgt_of= torch.tensor([[0,1],[2,3]], dtype=torch.long),
    )
    _tm.eval()
    with torch.no_grad():
        _logits = _tm(_d)
    assert _logits.shape == (N_ams,), f"Bad shape: {_logits.shape}"
    print(f"Forward pass: OK  logits={_logits.tolist()}")

    # 3. Pipeline smoke test (requires a replay JSON)
    import json
    ep_dirs = sorted([d for d in Path("126-precompute").iterdir() if d.is_dir()]) if Path("126-precompute").exists() else []
    if ep_dirs:
        ep_id = int(ep_dirs[0].name)
        ep_json_path = None
        for sub in sorted(Path("110-replays").iterdir()):
            if not sub.is_dir():
                continue
            cand = sub / f"episode_{ep_id}.json"
            if cand.exists():
                ep_json_path = cand
                break
        if ep_json_path:
            ep_json = json.loads(ep_json_path.read_text(encoding="utf-8"))
            for step_idx in range(min(3, len(ep_json["steps"]))):
                step_state = ep_json["steps"][step_idx]
                if not step_state:
                    continue
                obs_dict = step_state[0]["observation"]
                obs_ns   = types.SimpleNamespace(**obs_dict)
                # Reset per-game state
                _agent_step.clear(); _agent_num_agents.clear()
                _agent_step[obs_ns.player] = step_idx
                moves = agent(obs_ns)
                print(f"  step {step_idx}: agent() returned {len(moves)} moves: {moves}")
            print("End-to-end agent test: OK")
        else:
            print("Pipeline SKIP (no episode JSON found)")
    else:
        print("Pipeline SKIP (no 126-precompute/ dirs)")

    print("All checks passed.")
