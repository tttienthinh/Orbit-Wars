"""
Orbit Wars - Rule-Based Agent (07)

Strategy:
  Early (t<40):  Hyper-aggressive expansion. Near-zero garrison.
  Mid  (t40-150):Active defense + counter-attack exposed enemies +
                 multi-fleet coordination for strong targets.
  Late (t>150):  Press or recover. Desperate mode if losing.

Key features:
  * intercept_time includes target.radius (finds hits for small slow fleets)
  * Two-pass intercept: refine aim angle with actual fleet size
  * Counter-attack bonus for enemy planets that just dispatched fleets
  * Surface-to-surface sun avoidance
  * Desperate garrison: lower when behind in planet count
  * Multi-fleet coordination: simulate sequential arrivals so fleet A softens
    and fleet B captures — enables taking strong enemy planets
"""

import math
from collections import defaultdict
from kaggle_environments.envs.orbit_wars.orbit_wars import Planet

CENTER_X  = 50.0
CENTER_Y  = 50.0
SUN_R     = 10.0
SUN_SAFE  = 0.5
MAX_SPEED = 6.0

NEUTRAL_BONUS   = 1.15
EXPOSED_BONUS   = 2.0
INTERCEPT_LIMIT = 200


# ---------------------------------------------------------------------------
# Garrison — lower when we're behind
# ---------------------------------------------------------------------------

def garrison_for(step, ships, planet_ratio=1.0):
    """
    planet_ratio = my_planets / total_planets (lower = more desperate).
    When below 0.35, we're in desperation mode: very low garrison.
    """
    desperate = planet_ratio < 0.35
    if step < 30 or desperate:
        return max(2, int(ships * 0.05))
    if step < 80:
        return max(5, int(ships * 0.15))
    if step < 200:
        return max(8, int(ships * 0.22))
    return max(10, int(ships * 0.28))


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
    d = dist(mine.x, mine.y, aim_x, aim_y)
    if d < 1e-6:
        return False
    dx, dy  = (aim_x - mine.x) / d, (aim_y - mine.y) / d
    lx, ly  = mine.x + dx * mine.radius, mine.y + dy * mine.radius
    travel  = max(0.0, d - mine.radius - target_radius)
    ex, ey  = lx + dx * travel, ly + dy * travel
    return _pt_seg_dist(CENTER_X, CENTER_Y, lx, ly, ex, ey) < SUN_R + SUN_SAFE


# ---------------------------------------------------------------------------
# Fleet ledgers
# ---------------------------------------------------------------------------

def _fleet_target(raw, planets):
    fx, fy, fangle, fships = raw[2], raw[3], raw[4], raw[6]
    speed  = fleet_speed(fships)
    dir_x  = math.cos(fangle)
    dir_y  = math.sin(fangle)
    best, b_eta = None, float("inf")
    for p in planets:
        dx = p.x - fx; dy = p.y - fy
        along = dx * dir_x + dy * dir_y
        if along <= 0:
            continue
        if dx * dx + dy * dy - along * along < (p.radius + 1.0) ** 2:
            eta = along / speed
            if eta < b_eta:
                b_eta = eta; best = p.id
    return best, b_eta


def build_ledgers(raw_fleets, planets, my_player):
    friendly_inc  = defaultdict(int)
    enemy_inc     = defaultdict(int)
    enemy_eta     = defaultdict(lambda: float("inf"))
    exposed_ships = defaultdict(int)
    for raw in (raw_fleets or []):
        pid, eta = _fleet_target(raw, planets)
        ships    = int(raw[6])
        owner    = raw[1]
        from_pid = raw[5]
        if pid is not None:
            if owner == my_player:
                friendly_inc[pid] += ships
            else:
                enemy_inc[pid] += ships
                if eta < enemy_eta[pid]:
                    enemy_eta[pid] = eta
        if owner != my_player and from_pid is not None:
            exposed_ships[from_pid] += ships
    return friendly_inc, enemy_inc, enemy_eta, exposed_ships


# ---------------------------------------------------------------------------
# Shot geometry
# ---------------------------------------------------------------------------

def compute_shot(mine, target, angular_velocity, actual_ships=None):
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
                 angular_velocity, step, planet_ratio, reserved):
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
        best_src, best_d = None, float("inf")
        for src in my_planets:
            if src.id == threatened.id or src.id in used:
                continue
            garrison = garrison_for(step, src.ships, planet_ratio)
            sendable = src.ships - garrison - reserved[src.id]
            if sendable < needed:
                continue
            travel = dist(src.x, src.y, threatened.x, threatened.y) / fleet_speed(needed)
            if travel > eta + 2:
                continue
            d = dist(src.x, src.y, threatened.x, threatened.y)
            if d < best_d:
                best_d = d; best_src = src
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
# Single-planet offense scoring
# ---------------------------------------------------------------------------

def score_attack(mine, target, angular_velocity, is_neutral,
                 friendly_inc, sendable, exposed_ships, planet_ratio):
    base, travel, angle = compute_shot(mine, target, angular_velocity)
    if base is None:
        return None
    needed = base + 1 if is_neutral else base + travel * target.production + 1
    effective = max(0, needed - friendly_inc.get(target.id, 0))
    if effective == 0 or effective > sendable:
        return None
    # Refine aim for orbiting targets using actual fleet size
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
    if not is_neutral and exposed_ships.get(target.id, 0) > 0:
        score *= EXPOSED_BONUS
    return score, effective, angle


# ---------------------------------------------------------------------------
# Multi-fleet coordination
# ---------------------------------------------------------------------------

def coordinated_attacks(my_planets, enemies, angular_velocity, step,
                        planet_ratio, friendly_inc, exposed_ships,
                        used_sources, reserved):
    """
    For free planets that have no single-planet valid attack, try to coordinate
    2+ sequential fleet arrivals to soften-then-capture a strong enemy planet.
    Returns (list of move triples, set of claimed enemy planet ids).
    """
    new_moves  = []
    claimed    = set()
    free       = [p for p in my_planets if p.id not in used_sources]
    if len(free) < 2:
        return new_moves, claimed

    # Try each enemy from weakest to strongest
    for enemy in sorted(enemies, key=lambda e: e.ships):
        if enemy.id in claimed:
            continue

        shooters = []
        for src in free:
            if src.id in used_sources:
                continue
            garrison = garrison_for(step, src.ships, planet_ratio)
            sendable = src.ships - garrison - reserved[src.id]
            if sendable < 3:
                continue
            _, travel, angle = compute_shot(src, enemy, angular_velocity)
            if travel is None:
                continue
            shooters.append((travel, sendable, src, angle))

        if len(shooters) < 2:
            continue

        shooters.sort()

        garrison    = enemy.ships
        prev_travel = 0
        attack_plan = []
        already_inc = friendly_inc.get(enemy.id, 0)

        for travel, sendable, src, angle in shooters:
            if src.id in used_sources:
                continue
            garrison    = max(0, garrison - already_inc) + \
                          (travel - prev_travel) * enemy.production
            already_inc = 0

            if garrison <= 0:
                break

            send = min(sendable, garrison + 1)
            attack_plan.append((src.id, angle, send))
            garrison    = max(0, garrison - send)
            prev_travel = travel

            if garrison <= 0:
                break

        if garrison <= 0 and len(attack_plan) >= 2:
            for src_id, angle, send in attack_plan:
                new_moves.append([src_id, angle, send])
                used_sources.add(src_id)
                reserved[src_id] += send
            claimed.add(enemy.id)
            free = [p for p in free if p.id not in used_sources]

    return new_moves, claimed


# ---------------------------------------------------------------------------
# Main agent
# ---------------------------------------------------------------------------

def agent(obs):
    moves = []

    player    = obs.get("player", 0) if isinstance(obs, dict) else obs.player
    step      = obs.get("step",   0) if isinstance(obs, dict) else obs.step
    raw_p     = obs.get("planets", []) if isinstance(obs, dict) else obs.planets
    raw_f     = obs.get("fleets",  []) if isinstance(obs, dict) else obs.fleets
    ang_vel   = obs.get("angular_velocity", 0) if isinstance(obs, dict) else obs.angular_velocity
    comet_ids = set(obs.get("comet_planet_ids", []) if isinstance(obs, dict) else obs.comet_planet_ids)

    planets    = [Planet(*p) for p in raw_p]
    my_planets = [p for p in planets if p.owner == player]
    neutrals   = [p for p in planets if p.owner == -1 and p.id not in comet_ids]
    enemies    = [p for p in planets if p.owner not in (-1, player) and p.id not in comet_ids]

    if not neutrals and not enemies:
        return moves

    total_contested = len(planets) - len([p for p in planets if p.id in comet_ids])
    planet_ratio    = len(my_planets) / max(total_contested, 1)

    friendly_inc, enemy_inc, enemy_eta, exposed_ships = build_ledgers(
        raw_f, planets, player
    )
    reserved = defaultdict(int)

    # 1. Defense
    defense_moves, defense_sources = plan_defense(
        my_planets, enemy_inc, enemy_eta, friendly_inc,
        ang_vel, step, planet_ratio, reserved
    )
    moves.extend(defense_moves)

    # 2. Single-planet offense
    candidates = []
    for mine in my_planets:
        if mine.id in defense_sources:
            continue
        garrison = garrison_for(step, mine.ships, planet_ratio)
        sendable = mine.ships - garrison - reserved[mine.id]
        if sendable <= 0:
            continue
        for target in neutrals + enemies:
            is_neutral = (target.owner == -1)
            result = score_attack(mine, target, ang_vel, is_neutral,
                                  friendly_inc, sendable, exposed_ships,
                                  planet_ratio)
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

    # 3. Multi-fleet coordination (attack enemies that need > single-planet force)
    coord_moves, coord_claimed = coordinated_attacks(
        my_planets, enemies, ang_vel, step, planet_ratio,
        friendly_inc, exposed_ships, used_sources, reserved
    )
    moves.extend(coord_moves)
    claimed |= coord_claimed

    # 4. Enemy fallback — only if we can solo-capture
    for mine in my_planets:
        if mine.id in used_sources:
            continue
        garrison = garrison_for(step, mine.ships, planet_ratio)
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
