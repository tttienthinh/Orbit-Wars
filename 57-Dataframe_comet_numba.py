import math
import copy
import pandas as pd
import sys
import subprocess

# 1. Catch the Kaggle environment incompatibility
try:
    from numba import njit
except ImportError:
    # 2. Upgrade Numba to a version that supports NumPy 2.x+, silently
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "numba>=0.60.0", "--quiet", "--disable-pip-version-check"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )
    # 3. Now it is safe to import
    from numba import njit


import numpy as np
import math

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
    """Vectorized calculation of point-to-segment distance for entire arrays."""
    vx, vy = x_src, y_src
    wx, wy = x, y
    px, py = center, center
    
    # Squared length of the line segment (l2)
    l2 = (vx - wx)**2 + (vy - wy)**2
    
    # Prevent division by zero. We temporarily replace 0 with 1 
    # (we will force the result to 0 anyway for these cases in the next steps)
    l2_safe = np.where(l2 == 0, 1.0, l2)
    
    # Calculate the dot product
    dot_prod = (px - vx) * (wx - vx) + (py - vy) * (wy - wy)
    
    # Calculate t and clamp it strictly between 0.0 and 1.0
    t = np.clip(dot_prod / l2_safe, 0.0, 1.0)
    
    # If the segment length was 0 (v == w), force t to 0 so the projection is just v
    t = np.where(l2 == 0, 0.0, t)
    
    # Find the projection coordinates on the line
    proj_x = vx + t * (wx - vx)
    proj_y = vy + t * (wy - vy)
    
    # Calculate the true distance to the projection
    dist = np.sqrt((px - proj_x)**2 + (py - proj_y)**2)
    
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


@njit
def fast_visibility_pipeline(id_src, ships_sent, step, amin, amax):
    n = len(id_src)
    final_angles = np.full(n, np.nan, dtype=np.float64)
    has_free = np.zeros(n, dtype=np.bool_)
    
    PI_2 = 2 * math.pi
    
    # State tracking for the current trajectory
    blocked_starts = np.empty(1000, dtype=np.float64)
    blocked_ends = np.empty(1000, dtype=np.float64)
    num_blocked = 0
    
    # Pending blocks (targets at step N block targets at step N+1)
    pending_starts = np.empty(1000, dtype=np.float64)
    pending_ends = np.empty(1000, dtype=np.float64)
    num_pending = 0
    
    # Temp buffers for interval subtraction
    vis_starts = np.empty(100, dtype=np.float64)
    vis_ends = np.empty(100, dtype=np.float64)
    new_vis_starts = np.empty(100, dtype=np.float64)
    new_vis_ends = np.empty(100, dtype=np.float64)
    
    curr_step = -1
    
    for i in range(n):
        # 1. Handle Trajectory / Step Changes
        if i == 0 or id_src[i] != id_src[i-1] or ships_sent[i] != ships_sent[i-1]:
            num_blocked = 0
            num_pending = 0
            curr_step = step[i]
            
        elif step[i] > curr_step:
            # Flush pending obstacles into blocked and merge them
            for p in range(num_pending):
                blocked_starts[num_blocked] = pending_starts[p]
                blocked_ends[num_blocked] = pending_ends[p]
                num_blocked += 1
            num_pending = 0
            curr_step = step[i]
            
            # Merge overlapping blocked intervals for speed
            if num_blocked > 1:
                b_starts = blocked_starts[:num_blocked].copy()
                b_ends = blocked_ends[:num_blocked].copy()
                sort_idx = np.argsort(b_starts)
                
                m_num = 0
                c_s = b_starts[sort_idx[0]]
                c_e = b_ends[sort_idx[0]]
                
                for b in range(1, num_blocked):
                    n_s = b_starts[sort_idx[b]]
                    n_e = b_ends[sort_idx[b]]
                    if n_s <= c_e:
                        if n_e > c_e: c_e = n_e # Merge overlap
                    else:
                        blocked_starts[m_num], blocked_ends[m_num] = c_s, c_e
                        m_num += 1
                        c_s, c_e = n_s, n_e
                        
                blocked_starts[m_num], blocked_ends[m_num] = c_s, c_e
                num_blocked = m_num + 1

        # 2. Initialize Target Intervals (handling Modulo wrap)
        t_min, t_max = amin[i], amax[i]
        if t_min > t_max:
            vis_starts[0], vis_ends[0] = t_min, PI_2
            vis_starts[1], vis_ends[1] = 0.0, t_max
            num_vis = 2
            
            pending_starts[num_pending], pending_ends[num_pending] = t_min, PI_2
            num_pending += 1
            pending_starts[num_pending], pending_ends[num_pending] = 0.0, t_max
            num_pending += 1
        else:
            vis_starts[0], vis_ends[0] = t_min, t_max
            num_vis = 1
            
            pending_starts[num_pending], pending_ends[num_pending] = t_min, t_max
            num_pending += 1

        # 3. Subtract Blocked Intervals
        for b in range(num_blocked):
            bs, be = blocked_starts[b], blocked_ends[b]
            new_num = 0
            for v in range(num_vis):
                vs, ve = vis_starts[v], vis_ends[v]
                
                if be <= vs or bs >= ve: # No overlap
                    new_vis_starts[new_num], new_vis_ends[new_num] = vs, ve
                    new_num += 1
                elif bs <= vs and be >= ve: # Fully blocked
                    pass 
                elif bs > vs and be < ve: # Split
                    new_vis_starts[new_num], new_vis_ends[new_num] = vs, bs
                    new_num += 1
                    new_vis_starts[new_num], new_vis_ends[new_num] = be, ve
                    new_num += 1
                elif bs <= vs and be > vs and be < ve: # Left block
                    new_vis_starts[new_num], new_vis_ends[new_num] = be, ve
                    new_num += 1
                elif bs > vs and bs < ve and be >= ve: # Right block
                    new_vis_starts[new_num], new_vis_ends[new_num] = vs, bs
                    new_num += 1
            
            num_vis = new_num
            for v in range(num_vis):
                vis_starts[v], vis_ends[v] = new_vis_starts[v], new_vis_ends[v]

        # 4. Find Widest Angle and Midpoint
        if num_vis > 0:
            best_min, best_max, max_width = 0.0, 0.0, -1.0
            
            # Check for wrap-around restitch (touches 0 and 2*PI)
            idx_0, idx_max = -1, -1
            for v in range(num_vis):
                if vis_starts[v] <= 1e-9: idx_0 = v
                if vis_ends[v] >= (PI_2 - 1e-9): idx_max = v
                
            if idx_0 != -1 and idx_max != -1 and idx_0 != idx_max:
                stitched_w = (PI_2 - vis_starts[idx_max]) + vis_ends[idx_0]
                if stitched_w > max_width:
                    max_width = stitched_w
                    best_min, best_max = vis_starts[idx_max], vis_ends[idx_0]
                    
            for v in range(num_vis):
                if idx_0 != -1 and idx_max != -1 and (v == idx_0 or v == idx_max):
                    continue
                width = vis_ends[v] - vis_starts[v]
                if width > max_width:
                    max_width = width
                    best_min, best_max = vis_starts[v], vis_ends[v]
            
            if max_width > 0:
                has_free[i] = True
                if best_min <= best_max:
                    midpoint = (best_min + best_max) / 2.0
                else:
                    midpoint = best_min + ((PI_2 - best_min + best_max) / 2.0)
                final_angles[i] = midpoint % PI_2

    return has_free, final_angles


class IntervalProcessor:
    def merge_intervals(intervals):
        """Merges a list of [min, max] angle intervals where min <= max."""
        if not intervals:
            return []

        # Sort intervals by their start angle
        intervals = sorted(intervals, key=lambda x: x[0])
        merged = [intervals[0]]

        for current in intervals[1:]:
            prev_min, prev_max = merged[-1]
            curr_min, curr_max = current

            # If current interval overlaps with or touches the previous one
            if curr_min <= prev_max:
                merged[-1] = (prev_min, max(prev_max, curr_max))
            else:
                merged.append(current)

        return merged


    def create_cumulative_obstacles(possible_attacks, min_step=0):
        # 1. Determine the maximum step in the entire dataset
        max_step = int(possible_attacks["step"].max())

        # 2. Extract and unwrap intervals per (id_src, ships_sent, step) combination
        attack_map = {}
        group_cols = ["id_src", "ships_sent", "step"]

        for (id_src, ship, step), group in possible_attacks.groupby(group_cols):
            unwrapped_intervals = []

            for _, row in group.iterrows():
                amin, amax = row["angle_min"], row["angle_max"]

                if amin > amax:
                    # Modulo overlap caught: Split into two valid intervals
                    unwrapped_intervals.append((amin, 2 * np.pi))
                    unwrapped_intervals.append((0.0, amax))
                else:
                    # Normal interval
                    unwrapped_intervals.append((amin, amax))

            # Store using a triple-compound key (id_src, ship, step)
            attack_map[(id_src, ship, step)] = unwrapped_intervals

        # 3. Get all unique combinations of id_src and ships_sent to build continuous timelines
        unique_combinations = (
            possible_attacks[["id_src", "ships_sent"]].drop_duplicates().values
        )

        cumulative_records = []

        # Loop through each independent identifier track
        for id_src, ship in unique_combinations:
            current_intervals = []
            merged = []

            # Run the sequential simulation loop from min_step up to max_step (inclusive)
            for step in range(min_step, max_step + 1):

                # Accumulate new obstacles if they exist for this specific path
                if (id_src, ship, step) in attack_map:
                    current_intervals.extend(attack_map[(id_src, ship, step)])
                    # Merge everything tracked up to this point
                    merged = IntervalProcessor.merge_intervals(current_intervals)

                # Append the record maintaining your step shifting format
                cumulative_records.append(
                    {
                        "step": step + 1,
                        "id_src": id_src,
                        "ships_sent": ship,
                        "obstacle_list": merged,
                    }
                )

        return pd.DataFrame(cumulative_records)
    

    def subtract_intervals(target_min, target_max, blocked_intervals):
        """Subtracts a list of blocked intervals from a single valid [min, max] target interval."""
        # We start assuming the entire target interval is free/safe
        safe_intervals = [(target_min, target_max)]

        for b_min, b_max in blocked_intervals:
            next_safe = []
            for s_min, s_max in safe_intervals:
                # Case 1: Blocked interval completely misses the safe interval
                if b_max <= s_min or b_min >= s_max:
                    next_safe.append((s_min, s_max))

                # Case 2: Blocked interval overlaps or cuts into the safe interval
                else:
                    # Keep left piece if it exists
                    if b_min > s_min:
                        next_safe.append((s_min, b_min))
                    # Keep right piece if it exists
                    if b_max < s_max:
                        next_safe.append((b_max, s_max))

            safe_intervals = next_safe
            if not safe_intervals:
                break

        return safe_intervals


    def compute_free_angles(row):
        """Processes a single row to handle modulo wrapping and find the difference."""
        amin = row["angle_min"]
        amax = row["angle_max"]
        obstacles = row["obstacle_list"]

        # If there are no obstacles, the entire target interval is completely free
        if not isinstance(obstacles, list) or len(obstacles) == 0:
            return [(amin, amax)]

        # 1. Handle wrapping: Split target interval if it crosses the 2*pi boundary
        if amin > amax:
            targets = [(amin, 2 * np.pi), (0.0, amax)]
        else:
            targets = [(amin, amax)]

        # 2. Subtract obstacles from all active targets
        all_free_intervals = []
        for t_min, t_max in targets:
            free_pieces = IntervalProcessor.subtract_intervals(t_min, t_max, obstacles)
            all_free_intervals.extend(free_pieces)

        # 3. Re-wrap adjacent boundary pieces if necessary
        # (If one piece ends at 2*pi and another starts at 0, they are the same interval)
        has_end_piece = any(abs(f[1] - 2 * np.pi) < 1e-9 for f in all_free_intervals)
        has_start_piece = any(abs(f[0] - 0.0) < 1e-9 for f in all_free_intervals)

        if has_end_piece and has_start_piece and len(all_free_intervals) > 1:
            # Find them
            end_idx = next(
                i
                for i, f in enumerate(all_free_intervals)
                if abs(f[1] - 2 * np.pi) < 1e-9
            )
            start_idx = next(
                i for i, f in enumerate(all_free_intervals) if abs(f[0] - 0.0) < 1e-9
            )

            # Merge them back into a single modulo-wrapped interval: (min_of_end_piece, max_of_start_piece)
            wrapped_interval = (
                all_free_intervals[end_idx][0],
                all_free_intervals[start_idx][1],
            )

            # Remove old pieces and add the cleanly wrapped one
            indices_to_remove = {end_idx, start_idx}
            all_free_intervals = [
                f for i, f in enumerate(all_free_intervals) if i not in indices_to_remove
            ]
            all_free_intervals.append(wrapped_interval)

        return all_free_intervals

    def interval_to_final_angle(angle_list_series):
        """Takes a pandas Series of interval lists, finds the widest interval in each row,

        and returns its midpoint/average angle modulo 2*pi.
        """

        def process_row_intervals(intervals):
            if not isinstance(intervals, list) or len(intervals) == 0:
                return np.nan

            widest_span = -1.0
            best_midpoint = np.nan

            for amin, amax in intervals:
                # 1. Calculate the true span length considering modulo 2*pi wrap-around
                if amin <= amax:
                    span = amax - amin
                else:
                    span = (2 * np.pi - amin) + amax

                # 2. Track the maximum width seen so far
                if span > widest_span:
                    widest_span = span

                    # 3. Calculate the midpoint angle safely
                    if amin <= amax:
                        midpoint = (amin + amax) / 2.0
                    else:
                        midpoint = amin + (span / 2.0)

                    # Keep the final angle bound strictly between 0 and 2*pi
                    best_midpoint = midpoint % (2 * np.pi)

            return best_midpoint

        # Map the helper over the entire pandas Series
        return angle_list_series.map(process_row_intervals)

    


        
def take_action(df, player_id, nb_steps_sim=NB_STEPS_SIM, return_df=False):
    mine_across_sim = (
        df
        .assign(
            is_mine=lambda d: (d["owner"]==player_id).astype(int)
        )
        # .query("owner == @player_id")
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
        # .query("row_count >= @nb_steps_sim + 1 and ships_min > 0")
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
            step_diff=lambda d: d["step"] - d["step_src"],  # For a one ship fleet, the speed is 1 unit per step
            fleet_speed=lambda d: 1.0 + (MAX_SPEED - 1.0) * (np.log(d["ships_sent"]) / math.log(1000)) ** 1.5,
            dist_fleet_src_min=lambda d: d["step_diff"] * d["fleet_speed"] + PLANET_MARGIN + d["radius_src"],
            dist_fleet_src_max=lambda d: (d["step_diff"] + 1) * d["fleet_speed"] + PLANET_MARGIN + d["radius_src"],
            collision=lambda d: ((d["dist_tgt_src"] - d["radius"] < d["dist_fleet_src_min"]) & (d["dist_fleet_src_min"] < d["dist_tgt_src"] + d["radius"])) | ((d["dist_tgt_src"] - d["radius"] < d["dist_fleet_src_max"]) & (d["dist_fleet_src_max"] < d["dist_tgt_src"] + d["radius"])),
        )
        .query("collision") # Only consider attacks we can win with the ships we have at the source during the sim
        .assign(
            crossing_sun=lambda d: is_crossing_sun_vectorized(
                d["x_src"].values, 
                d["y_src"].values, 
                d["x"].values, 
                d["y"].values, 
                CENTER, 
                SUN_RADIUS + PLANET_MARGIN
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
        return []# 1. Sort values exactly how Numba expects to see the groups
    possible_attacks = possible_attacks.sort_values(["id_src", "ships_sent", "step"], ascending=True)

    # 2. Extract primitive NumPy arrays and run the fast Numba kernel
    has_free, final_angles = fast_visibility_pipeline(
        possible_attacks["id_src"].values,
        possible_attacks["ships_sent"].values,
        possible_attacks["step"].values,
        possible_attacks["angle_min"].values,
        possible_attacks["angle_max"].values
    )

    # 3. Reassign directly to the dataframe
    possible_attacks["has_free_angle"] = has_free
    possible_attacks["final_angle"] = final_angles

    # 4. Filter to only attacks that actually have a free angle
    attacks_with_angle = possible_attacks.query("has_free_angle").copy()

    # (You no longer need IntervalProcessor AT ALL)

    # ... Your existing code continues exactly as before:
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
            .query("ships_sent <= ships_min")  # Ensure we have enough ships to attack
            .sort_values(["ships_sent", "step"], ascending=[False, True])
            .groupby("id_src", as_index=False)
            .first()
            [["id_src", "angle", "ships_sent"]]
            .values
            .tolist()
        )
        id_to_avoid = attacks_with_angle_comets["id_src"].unique().tolist() 
        attacks_with_angle = attacks_with_angle.query("id_src not in @id_to_avoid")  # Remove the comet attack source from the main attack dataframe to avoid double attacking from it
        

    attacks = (
        planet_id_top_5_id_src
        # Nearest top 5
        .merge(
            attacks_with_angle,
            how="left",
            on=["id_src", "id"]
        )
        # Non-player filtering
        .query("@player_id != owner")
        # ships_needed filtering
        .assign(
            ships_needed = lambda d: np.where(
                d["owner"] == -1,
                d["ships"],
                d["ships"] + d["production"]
            )
        )
        .query("ships_needed + 1 <= ships_sent and ships_sent <= ships_needed + production_src + 1")  # Ensure we are looking at future steps CAN BE LATER REMOVED TO CONSIDER PAIR ATTACKS
        .sort_values(["step", "ships_sent"], ascending=True)
        .groupby(["id_src", "id"], as_index=False)
        .first()
        # Score aiming sorted
        .assign(
            time_cost=lambda d: d["ships_needed"] / d["production_src"],
            total_time_cost=lambda d: d.groupby("id_src")["time_cost"].transform("sum"),
            score=lambda d: (d["total_time_cost"] - d["time_cost"] - d["step_diff"]) * d["production"],
        )
        .sort_values("score", ascending=False)
        .groupby("id_src", as_index=False)
        .first()
        .query("ships_sent <= ships_min")  # Ensure we have enough ships to attack
        # .assign(
        #     final_angle = lambda d: IntervalProcessor.interval_to_final_angle(d["angle_list"])
        # )
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
    return moves #, attacks_with_angle, df_src_tgt
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
