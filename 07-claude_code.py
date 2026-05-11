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

NEUTRAL_BONUS      = 1.15
EXPOSED_BONUS      = 3.0    # enemy just launched fleet — planet is exposed
BUILDUP_BONUS      = 1.5    # enemy has large buildup — preemptively disrupt
BUILDUP_THRESHOLD  = 60     # ships threshold to trigger BUILDUP_BONUS
INTERCEPT_LIMIT    = 200


# ---------------------------------------------------------------------------
# Garrison — lower when we're behind
# ---------------------------------------------------------------------------

def garrison_for(step, ships, planet_ratio=1.0, net_threat=0):
    """
    planet_ratio = my_planets / total_planets (lower = more desperate).
    net_threat   = max(0, enemy_ships_incoming - friendly_ships_incoming).
    Reactive: keep enough to survive incoming threat.
    """
    desperate = planet_ratio < 0.35
    if step < 10:
        base = 1
    elif step < 30 or desperate:
        base = max(2, int(ships * 0.05))
    elif step < 80:
        base = max(5, int(ships * 0.15))
    elif step < 200:
        base = max(8, int(ships * 0.22))
    else:
        base = max(10, int(ships * 0.28))
    if net_threat > 0:
        base = max(base, net_threat + 1)
    return base


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
    """Multi-source defense: pool ships from several planets to repel large waves."""
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
        remaining = needed
        # Sort sources by proximity so closest ships arrive first
        sources = sorted(
            [s for s in my_planets if s.id != threatened.id and s.id not in used],
            key=lambda s: dist(s.x, s.y, threatened.x, threatened.y)
        )
        for src in sources:
            if remaining <= 0:
                break
            src_einc   = enemy_inc.get(src.id, 0)
            src_finc   = friendly_inc.get(src.id, 0)
            src_threat = max(0, src_einc - src_finc)
            garrison   = garrison_for(step, src.ships, planet_ratio,
                                      net_threat=src_threat)
            sendable   = src.ships - garrison - reserved[src.id]
            if sendable <= 0:
                continue
            travel = dist(src.x, src.y, threatened.x, threatened.y) / fleet_speed(sendable)
            if travel > eta + 2:
                continue
            if path_hits_sun(src, threatened.x, threatened.y, threatened.radius):
                continue
            send = min(sendable, remaining)
            angle = math.atan2(threatened.y - src.y, threatened.x - src.x)
            defense_moves.append([src.id, angle, send])
            reserved[src.id] += send
            used.add(src.id)
            remaining -= send
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
                        planet_ratio, friendly_inc, enemy_inc, exposed_ships,
                        used_sources, reserved):
    """
    For free planets that have no single-planet valid attack, try to coordinate
    2+ sequential fleet arrivals to soften-then-capture a strong enemy planet.
    Returns (list of move triples, set of claimed enemy planet ids).
    """
    new_moves    = []
    claimed      = set()
    coord_used   = set()  # internal: planets committed in this coordination pass
    # All planets can join coordinated attacks — reserved tracks committed ships
    free         = list(my_planets)
    if len(free) < 2:
        return new_moves, claimed

    # Try each enemy from weakest to strongest — easier plans succeed first
    for enemy in sorted(enemies, key=lambda e: e.ships):
        if enemy.id in claimed:
            continue

        shooters = []
        for src in free:
            if src.id in coord_used:  # blocked only if already committed THIS pass
                continue
            src_einc2  = enemy_inc.get(src.id, 0)
            src_finc2  = friendly_inc.get(src.id, 0)
            src_threat = max(0, src_einc2 - src_finc2)
            garrison   = garrison_for(step, src.ships, planet_ratio,
                                      net_threat=src_threat)
            sendable   = src.ships - garrison - reserved[src.id]
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
            if src.id in coord_used:
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
                coord_used.add(src_id)
                used_sources.add(src_id)
                reserved[src_id] += send
            claimed.add(enemy.id)
            free = [p for p in free if p.id not in coord_used]

    return new_moves, claimed


# ---------------------------------------------------------------------------
# Frontier reinforcement
# ---------------------------------------------------------------------------

def plan_reinforcement(my_planets, enemies, reserved, step, planet_ratio,
                       enemy_inc, friendly_inc):
    """
    Push ships from rear planets to frontier planets so frontline has
    more ships to absorb incoming waves and stage counter-attacks.
    Only runs mid/late game (step > 40) to avoid disrupting early expansion.
    """
    if step < 40 or not enemies:
        return []

    # Frontier = our planet closest to any enemy
    def min_enemy_dist(p):
        return min(dist(p.x, p.y, e.x, e.y) for e in enemies)

    sorted_by_enemy = sorted(my_planets, key=min_enemy_dist)
    n = len(sorted_by_enemy)
    if n < 2:
        return []

    # Bottom half by enemy proximity = frontier; top half = rear
    frontier = sorted_by_enemy[:max(1, n // 2)]
    rear     = sorted_by_enemy[max(1, n // 2):]

    # Only reinforce if frontier is significantly weaker than rear
    frontier_avg = sum(p.ships for p in frontier) / len(frontier)
    rear_avg     = sum(p.ships for p in rear) / len(rear)
    if rear_avg < frontier_avg * 1.4:
        return []

    moves = []
    virtual_ships = {p.id: p.ships for p in frontier}  # track expected ship count after reinforcement
    for src in rear:
        src_threat = max(0, enemy_inc.get(src.id, 0) - friendly_inc.get(src.id, 0))
        garrison   = garrison_for(step, src.ships, planet_ratio, net_threat=src_threat)
        sendable   = src.ships - garrison - reserved[src.id]
        if sendable < 15:
            continue
        send = int(sendable * 0.45)
        if send < 10:
            continue
        # Pick the frontier planet with fewest expected ships (distribute evenly)
        dest = min(frontier, key=lambda p: virtual_ships[p.id])
        if path_hits_sun(src, dest.x, dest.y, dest.radius):
            continue
        angle = math.atan2(dest.y - src.y, dest.x - src.x)
        moves.append([src.id, angle, send])
        reserved[src.id] += send
        virtual_ships[dest.id] += send
    return moves


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

    # 2. Single-planet offense — multi-fleet allowed per planet via reserved tracking
    # Steeper travel penalty early game to maximize close-range expansion speed.
    travel_exp = 1.5 if step < 40 else 1.0

    candidates = []
    for mine in my_planets:
        einc       = enemy_inc.get(mine.id, 0)
        finc       = friendly_inc.get(mine.id, 0)
        net_threat = max(0, einc - finc)
        garrison   = garrison_for(step, mine.ships, planet_ratio, net_threat=net_threat)
        sendable   = mine.ships - garrison - reserved[mine.id]
        if sendable <= 0:
            continue
        for target in neutrals + enemies:
            is_neutral = (target.owner == -1)
            base, travel, angle = compute_shot(mine, target, ang_vel)
            if base is None:
                continue
            needed = base + 1 if is_neutral else base + travel * target.production + 1
            effective = max(0, needed - friendly_inc.get(target.id, 0))
            if effective == 0 or effective > sendable:
                continue
            # Two-pass: refine angle for orbiting targets
            if is_orbiting(target):
                base2, travel, angle = compute_shot(mine, target, ang_vel,
                                                    actual_ships=effective)
                if base2 is None:
                    continue
                if not is_neutral:
                    needed = base2 + travel * target.production + 1
                    effective = max(0, needed - friendly_inc.get(target.id, 0))
                    if effective == 0 or effective > sendable:
                        continue
            score = (target.production + 1) / (effective * (travel + 1) ** travel_exp)
            if is_neutral:
                score *= NEUTRAL_BONUS
            if not is_neutral:
                if exposed_ships.get(target.id, 0) > 0:
                    score *= EXPOSED_BONUS
                elif target.ships >= BUILDUP_THRESHOLD:
                    score *= BUILDUP_BONUS
            candidates.append((score, mine.id, target.id, effective, angle))

    candidates.sort(key=lambda c: c[0], reverse=True)

    # Build lookup maps for dynamic re-checks
    planet_by_id = {p.id: p for p in my_planets}
    claimed      = set()
    used_sources = set()  # track for coordinated_attacks only

    for score, src_id, tgt_id, needed, angle in candidates:
        if tgt_id in claimed:
            continue
        mine = planet_by_id.get(src_id)
        if mine is None:
            continue
        einc2      = enemy_inc.get(src_id, 0)
        finc2      = friendly_inc.get(src_id, 0)
        net_threat = max(0, einc2 - finc2)
        garrison   = garrison_for(step, mine.ships, planet_ratio, net_threat=net_threat)
        still_sendable = mine.ships - garrison - reserved[src_id]
        if needed > still_sendable:
            continue
        moves.append([src_id, angle, needed])
        reserved[src_id] += needed
        claimed.add(tgt_id)
        used_sources.add(src_id)

    # 3. Multi-fleet coordination (attack enemies that need > single-planet force)
    coord_moves, coord_claimed = coordinated_attacks(
        my_planets, enemies, ang_vel, step, planet_ratio,
        friendly_inc, enemy_inc, exposed_ships, used_sources, reserved
    )
    moves.extend(coord_moves)
    claimed |= coord_claimed

    return moves
