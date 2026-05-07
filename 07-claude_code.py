"""
Orbit Wars - Rule-Based Agent (07)

Strategy:
  Early (t<40):  Hyper-aggressive expansion. Near-zero garrison.
  Mid  (t40-150):Active defense + counter-attack weakened enemy planets.
  Late (t>150):  Press or recover.

Key fixes over prior versions:
  * intercept_time includes target.radius → finds hits for slow early fleets
  * Two-pass intercept: compute travel time, then refine with actual fleet size
  * Counter-attack: detect enemy planets that just dispatched fleets (from_planet_id)
                    and score them higher (exposed garrison)
  * Enemy fallback only sends if we can actually capture (no wasted ships)
  * Surface-to-surface sun avoidance (much less restrictive than center-to-center)
"""

import math
from collections import defaultdict
from kaggle_environments.envs.orbit_wars.orbit_wars import Planet

CENTER_X  = 50.0
CENTER_Y  = 50.0
SUN_R     = 10.0
SUN_SAFE  = 0.5
MAX_SPEED = 6.0

NEUTRAL_BONUS       = 1.15
EXPOSED_BONUS       = 2.0   # score multiplier for enemy planets that just sent fleets
INTERCEPT_LIMIT     = 200   # wider search window for fast-orbiting planets


# ---------------------------------------------------------------------------
# Garrison by phase
# ---------------------------------------------------------------------------

def garrison_for(step, ships):
    if step < 30:
        return max(2, int(ships * 0.05))
    if step < 80:
        return max(5, int(ships * 0.15))
    if step < 200:
        return max(8, int(ships * 0.25))
    return max(10, int(ships * 0.30))


# ---------------------------------------------------------------------------
# Physics
# ---------------------------------------------------------------------------

def dist(x0, y0, x1, y1):
    return math.sqrt((x0 - x1) ** 2 + (y0 - y1) ** 2)


def dist_to_center(x, y):
    return dist(x, y, CENTER_X, CENTER_Y)


def is_orbiting(planet):
    return dist_to_center(planet.x, planet.y) + planet.radius < 50.0


def fleet_speed(ships):
    if ships <= 1:
        return 1.0
    ratio = math.log(max(ships, 1)) / math.log(1000.0)
    ratio = max(0.0, min(1.0, ratio))
    return 1.0 + (MAX_SPEED - 1.0) * (ratio ** 1.5)


def orbital_position(x, y, angular_velocity, t):
    r     = dist_to_center(x, y)
    angle = math.atan2(y - CENTER_Y, x - CENTER_X) + angular_velocity * t
    return CENTER_X + r * math.cos(angle), CENTER_Y + r * math.sin(angle)


def intercept_time(ox, oy, tx, ty, angular_velocity, ships, target_radius=0.0):
    """
    Find the earliest turn t where a fleet of `ships` can reach the orbiting
    planet.  Including target_radius is essential: the fleet hits the planet
    surface, not its centre.
    """
    speed = fleet_speed(ships)
    for t in range(1, INTERCEPT_LIMIT + 1):
        px, py = orbital_position(tx, ty, angular_velocity, t)
        if dist(ox, oy, px, py) <= speed * t + target_radius:
            return t, px, py
    return None, None, None


def _pt_seg_dist(px, py, x1, y1, x2, y2):
    dx, dy = x2 - x1, y2 - y1
    lsq    = dx * dx + dy * dy
    if lsq < 1e-9:
        return dist(px, py, x1, y1)
    t = max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / lsq))
    return dist(px, py, x1 + t * dx, y1 + t * dy)


def path_hits_sun(mine, aim_x, aim_y, target_radius=0.0):
    """Surface-to-surface segment check."""
    d = dist(mine.x, mine.y, aim_x, aim_y)
    if d < 1e-6:
        return False
    dx, dy = (aim_x - mine.x) / d, (aim_y - mine.y) / d
    lx = mine.x + dx * mine.radius
    ly = mine.y + dy * mine.radius
    travel = max(0.0, d - mine.radius - target_radius)
    ex = lx + dx * travel
    ey = ly + dy * travel
    return _pt_seg_dist(CENTER_X, CENTER_Y, lx, ly, ex, ey) < SUN_R + SUN_SAFE


# ---------------------------------------------------------------------------
# Fleet ledgers
# ---------------------------------------------------------------------------

def _fleet_target(raw, planets):
    fx, fy, fangle, fships = raw[2], raw[3], raw[4], raw[6]
    speed  = fleet_speed(fships)
    dir_x  = math.cos(fangle)
    dir_y  = math.sin(fangle)
    best   = None
    b_eta  = float("inf")
    for p in planets:
        dx    = p.x - fx
        dy    = p.y - fy
        along = dx * dir_x + dy * dir_y
        if along <= 0:
            continue
        cross_sq = dx * dx + dy * dy - along * along
        if cross_sq < (p.radius + 1.0) ** 2:
            eta = along / speed
            if eta < b_eta:
                b_eta = eta
                best  = p.id
    return best, b_eta


def build_ledgers(raw_fleets, planets, my_player):
    friendly_inc  = defaultdict(int)
    enemy_inc     = defaultdict(int)
    enemy_eta     = defaultdict(lambda: float("inf"))
    exposed_ships = defaultdict(int)   # enemy ships sent away from each enemy planet

    for raw in (raw_fleets or []):
        pid, eta = _fleet_target(raw, planets)
        ships    = int(raw[6])
        owner    = raw[1]
        from_pid = raw[5]    # from_planet_id

        if pid is not None:
            if owner == my_player:
                friendly_inc[pid] += ships
            else:
                enemy_inc[pid] += ships
                if eta < enemy_eta[pid]:
                    enemy_eta[pid] = eta

        if owner != my_player and from_pid is not None:
            exposed_ships[from_pid] += ships   # enemy planet that dispatched fleet

    return friendly_inc, enemy_inc, enemy_eta, exposed_ships


# ---------------------------------------------------------------------------
# Shot geometry — two-pass for accuracy
# ---------------------------------------------------------------------------

def compute_shot(mine, target, angular_velocity, actual_ships=None):
    """
    Returns (base_garrison, travel_turns, aim_angle) or (None, None, None).
    base_garrison = target.ships now; caller adds production during travel.
    actual_ships: if known, use for fleet-speed computation (more accurate).
    """
    speed_ships = actual_ships or max(target.ships + 1, 5)

    if is_orbiting(target):
        t, px, py = intercept_time(mine.x, mine.y, target.x, target.y,
                                   angular_velocity, speed_ships,
                                   target_radius=target.radius)
        if t is None:
            return None, None, None
        if path_hits_sun(mine, px, py):
            return None, None, None
        return target.ships, t, math.atan2(py - mine.y, px - mine.x)
    else:
        if path_hits_sun(mine, target.x, target.y, target_radius=target.radius):
            return None, None, None
        travel = max(1, int(dist(mine.x, mine.y, target.x, target.y) /
                             fleet_speed(speed_ships)))
        return target.ships, travel, math.atan2(target.y - mine.y, target.x - mine.x)


# ---------------------------------------------------------------------------
# Defense
# ---------------------------------------------------------------------------

def plan_defense(my_planets, enemy_inc, enemy_eta, friendly_inc,
                 angular_velocity, step, reserved):
    defense_moves = []
    used          = set()

    threats = []
    for mp in my_planets:
        einc   = enemy_inc.get(mp.id, 0)
        finc   = friendly_inc.get(mp.id, 0)
        margin = mp.ships + finc - einc
        if margin < 0:
            threats.append((enemy_eta.get(mp.id, float("inf")), -margin + 1, mp))
    threats.sort()

    for eta, needed, threatened in threats:
        best_src  = None
        best_dist = float("inf")
        for src in my_planets:
            if src.id == threatened.id or src.id in used:
                continue
            garrison = garrison_for(step, src.ships)
            sendable = src.ships - garrison - reserved[src.id]
            if sendable < needed:
                continue
            travel = dist(src.x, src.y, threatened.x, threatened.y) / fleet_speed(needed)
            if travel > eta + 2:
                continue
            d = dist(src.x, src.y, threatened.x, threatened.y)
            if d < best_dist:
                best_dist = d
                best_src  = src

        if best_src is None:
            continue
        if path_hits_sun(best_src, threatened.x, threatened.y, threatened.radius):
            continue
        angle = math.atan2(threatened.y - best_src.y, threatened.x - best_src.x)
        defense_moves.append([best_src.id, angle, needed])
        reserved[best_src.id] += needed
        used.add(best_src.id)

    return defense_moves, used


# ---------------------------------------------------------------------------
# Offense scoring
# ---------------------------------------------------------------------------

def score_attack(mine, target, angular_velocity, is_neutral,
                 friendly_inc, sendable, exposed_ships):
    # First pass: rough travel estimate
    base, travel, angle = compute_shot(mine, target, angular_velocity)
    if base is None:
        return None

    if is_neutral:
        needed = base + 1
    else:
        needed = base + travel * target.production + 1

    effective = max(0, needed - friendly_inc.get(target.id, 0))
    if effective == 0 or effective > sendable:
        return None

    # Second pass: refine aim with actual fleet size (matters for orbiting)
    if is_orbiting(target):
        base2, travel, angle = compute_shot(mine, target, angular_velocity,
                                            actual_ships=effective)
        if base2 is None:
            return None
        if not is_neutral:
            needed = base2 + travel * target.production + 1
            effective = max(0, needed - friendly_inc.get(target.id, 0))
            if effective == 0 or effective > sendable:
                return None

    score = (target.production + 1) / (effective * (travel + 1))
    if is_neutral:
        score *= NEUTRAL_BONUS
    # Counter-attack bonus: enemy planet whose fleet just left is exposed
    if not is_neutral and exposed_ships.get(target.id, 0) > 0:
        score *= EXPOSED_BONUS

    return score, effective, angle


# ---------------------------------------------------------------------------
# Main agent
# ---------------------------------------------------------------------------

def agent(obs):
    moves = []

    player  = obs.get("player", 0)  if isinstance(obs, dict) else obs.player
    step    = obs.get("step",   0)  if isinstance(obs, dict) else obs.step
    raw_p   = obs.get("planets", []) if isinstance(obs, dict) else obs.planets
    raw_f   = obs.get("fleets",  []) if isinstance(obs, dict) else obs.fleets
    ang_vel = obs.get("angular_velocity", 0) if isinstance(obs, dict) else obs.angular_velocity
    comet_ids = set(obs.get("comet_planet_ids", []) if isinstance(obs, dict) else obs.comet_planet_ids)

    planets    = [Planet(*p) for p in raw_p]
    my_planets = [p for p in planets if p.owner == player]
    neutrals   = [p for p in planets if p.owner == -1 and p.id not in comet_ids]
    enemies    = [p for p in planets if p.owner not in (-1, player) and p.id not in comet_ids]

    if not neutrals and not enemies:
        return moves

    friendly_inc, enemy_inc, enemy_eta, exposed_ships = build_ledgers(
        raw_f, planets, player
    )

    reserved = defaultdict(int)

    # 1. Defense
    defense_moves, defense_sources = plan_defense(
        my_planets, enemy_inc, enemy_eta, friendly_inc, ang_vel, step, reserved
    )
    moves.extend(defense_moves)

    # 2. Offense
    candidates = []
    for mine in my_planets:
        if mine.id in defense_sources:
            continue
        garrison = garrison_for(step, mine.ships)
        sendable = mine.ships - garrison - reserved[mine.id]
        if sendable <= 0:
            continue
        for target in neutrals + enemies:
            is_neutral = (target.owner == -1)
            result = score_attack(mine, target, ang_vel, is_neutral,
                                  friendly_inc, sendable, exposed_ships)
            if result is None:
                continue
            score, needed, angle = result
            candidates.append((score, mine, target, needed, angle))

    candidates.sort(key=lambda c: c[0], reverse=True)

    used_sources = set(defense_sources)
    claimed      = set()

    for score, mine, target, needed, angle in candidates:
        if mine.id in used_sources or target.id in claimed:
            continue
        moves.append([mine.id, angle, needed])
        used_sources.add(mine.id)
        claimed.add(target.id)

    # 3. Enemy fallback — only send if we can actually capture
    for mine in my_planets:
        if mine.id in used_sources:
            continue
        garrison = garrison_for(step, mine.ships)
        sendable = mine.ships - garrison - reserved[mine.id]
        if sendable <= 0:
            continue

        best = None
        for enemy in sorted(enemies, key=lambda e: e.ships):
            if enemy.id in claimed:
                continue
            _, travel, angle = compute_shot(mine, enemy, ang_vel)
            if travel is None:
                continue
            needed = enemy.ships + travel * enemy.production + 1
            if needed > sendable:
                continue
            score = (enemy.production + 1) / (needed * (travel + 1))
            if exposed_ships.get(enemy.id, 0) > 0:
                score *= EXPOSED_BONUS
            if best is None or score > best[0]:
                best = (score, angle, needed, enemy.id)

        if best is not None:
            _, angle, needed, eid = best
            moves.append([mine.id, angle, needed])
            used_sources.add(mine.id)
            claimed.add(eid)

    return moves
