"""
Orbit Wars - Rule-Based Agent (07)

Rules over 05 (nearest-planet sniper with orbital intercept):

  1. Garrison          - keep ships at home, only spend the surplus.
  2. Target scoring    - rank by production / (cost * travel); prefer value.
  3. Neutral preference- slight 1.2x bonus; does NOT block enemy attacks.
  4. Production-adjust - enemy ships keep growing in transit; account for it.
  5. Fleet tracking    - subtract friendly ships already en route so we don't
                         over-send to the same planet across turns.
  6. No dup targeting  - one attacker per target per turn.
  7. Sun avoidance     - skip any path whose straight line crosses the sun.
  8. Enemy fallback    - planets with no viable neutral target attack enemies.
"""

import math
from collections import defaultdict
from kaggle_environments.envs.orbit_wars.orbit_wars import Planet

CENTER_X = 50.0
CENTER_Y = 50.0
SUN_R    = 10.0
SUN_SAFE = 1.5      # clearance margin around sun
MAX_SPEED = 6.0

GARRISON_RATIO  = 0.30
GARRISON_MIN    = 8
NEUTRAL_BONUS   = 1.20          # mild preference for neutral planets
INTERCEPT_LIMIT = 100           # max turns searched for intercept


# ---------------------------------------------------------------------------
# Geometry / physics helpers
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
    r = dist_to_center(x, y)
    angle = math.atan2(y - CENTER_Y, x - CENTER_X) + angular_velocity * t
    return CENTER_X + r * math.cos(angle), CENTER_Y + r * math.sin(angle)


def intercept_time(ox, oy, tx, ty, angular_velocity, ships):
    speed = fleet_speed(ships)
    for t in range(1, INTERCEPT_LIMIT + 1):
        px, py = orbital_position(tx, ty, angular_velocity, t)
        if dist(ox, oy, px, py) <= speed * t:
            return t, px, py
    return None, None, None


def _point_seg_dist(px, py, x1, y1, x2, y2):
    dx, dy = x2 - x1, y2 - y1
    len_sq = dx * dx + dy * dy
    if len_sq < 1e-9:
        return dist(px, py, x1, y1)
    t = ((px - x1) * dx + (py - y1) * dy) / len_sq
    t = max(0.0, min(1.0, t))
    return dist(px, py, x1 + t * dx, y1 + t * dy)


def path_hits_sun(x1, y1, x2, y2):
    return _point_seg_dist(CENTER_X, CENTER_Y, x1, y1, x2, y2) < SUN_R + SUN_SAFE


# ---------------------------------------------------------------------------
# Fleet tracking — count friendly ships already heading to each planet
# ---------------------------------------------------------------------------

def build_friendly_incoming(raw_fleets, planets, player):
    """Returns {planet_id: ships_incoming} for our own in-flight fleets."""
    incoming = defaultdict(int)
    if not raw_fleets:
        return incoming

    Fleet = None  # lazy import of namedtuple shape via index
    for raw in raw_fleets:
        # raw = [id, owner, x, y, angle, from_planet_id, ships]
        owner = raw[1]
        if owner != player:
            continue
        fx, fy, fangle, fships = raw[2], raw[3], raw[4], raw[6]
        speed = fleet_speed(fships)
        dir_x = math.cos(fangle)
        dir_y = math.sin(fangle)
        best_planet_id = None
        best_eta = float("inf")
        for p in planets:
            dx = p.x - fx
            dy = p.y - fy
            along = dx * dir_x + dy * dir_y
            if along <= 0:
                continue
            d_sq = dx * dx + dy * dy
            cross_sq = d_sq - along * along
            if cross_sq < (p.radius + 1.0) ** 2:
                eta = along / speed
                if eta < best_eta:
                    best_eta = eta
                    best_planet_id = p.id
        if best_planet_id is not None:
            incoming[best_planet_id] += int(fships)

    return incoming


# ---------------------------------------------------------------------------
# Targeting helpers
# ---------------------------------------------------------------------------

def ships_needed_for(mine, target, angular_velocity, is_neutral):
    """
    Ships required to capture target accounting for production during travel.
    Returns (ships_needed, travel_turns, aim_angle).
    Returns (None, None, None) if path is blocked by the sun.
    """
    if is_orbiting(target):
        guess = max(target.ships + 1, 10)
        t, px, py = intercept_time(mine.x, mine.y, target.x, target.y,
                                   angular_velocity, guess)
        if t is None:
            return None, None, None
        if path_hits_sun(mine.x, mine.y, px, py):
            return None, None, None
        travel = t
        angle = math.atan2(py - mine.y, px - mine.x)
    else:
        if path_hits_sun(mine.x, mine.y, target.x, target.y):
            return None, None, None
        travel = max(1, int(dist(mine.x, mine.y, target.x, target.y) /
                             fleet_speed(max(target.ships + 1, 10))))
        angle = math.atan2(target.y - mine.y, target.x - mine.x)

    if is_neutral:
        needed = target.ships + 1
    else:
        # Enemy keeps producing during transit
        needed = target.ships + travel * target.production + 1

    return needed, travel, angle


def score_target(mine, target, angular_velocity, is_neutral, already_incoming):
    """
    Score this (mine → target) pair. Higher = better.
    Returns (score, ships_needed, travel, angle) or None if infeasible.
    """
    needed, travel, angle = ships_needed_for(mine, target, angular_velocity, is_neutral)
    if needed is None:
        return None

    # Discount by friendly ships already heading there
    effective_needed = max(0, needed - already_incoming.get(target.id, 0))

    if effective_needed == 0:
        return None  # already covered by in-flight fleets

    score = (target.production + 1) / (effective_needed * (travel + 1))
    if is_neutral:
        score *= NEUTRAL_BONUS

    return score, effective_needed, travel, angle


# ---------------------------------------------------------------------------
# Main agent
# ---------------------------------------------------------------------------

def agent(obs):
    moves = []

    player  = obs.get("player", 0) if isinstance(obs, dict) else obs.player
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

    friendly_incoming = build_friendly_incoming(raw_f, planets, player)

    # Build scored candidate list: (score, mine, target, ships_needed, travel, angle)
    candidates = []
    for mine in my_planets:
        garrison  = max(GARRISON_MIN, int(mine.ships * GARRISON_RATIO))
        sendable  = mine.ships - garrison
        if sendable <= 0:
            continue

        for target in neutrals + enemies:
            is_neutral = (target.owner == -1)
            result = score_target(mine, target, ang_vel, is_neutral, friendly_incoming)
            if result is None:
                continue
            score, needed, travel, angle = result
            if needed <= sendable:
                candidates.append((score, mine, target, needed, travel, angle))

    candidates.sort(key=lambda c: c[0], reverse=True)

    used_sources  = set()  # one move per source planet per turn
    claimed       = set()  # one attacker per target per turn

    for score, mine, target, needed, travel, angle in candidates:
        if mine.id in used_sources:
            continue
        if target.id in claimed:
            continue
        moves.append([mine.id, angle, needed])
        used_sources.add(mine.id)
        claimed.add(target.id)

    # Enemy fallback: source planets that got no move, attack weakest reachable enemy
    for mine in my_planets:
        if mine.id in used_sources:
            continue
        garrison = max(GARRISON_MIN, int(mine.ships * GARRISON_RATIO))
        sendable = mine.ships - garrison
        if sendable <= 0:
            continue

        best = None
        for enemy in enemies:
            if enemy.id in claimed:
                continue
            needed, travel, angle = ships_needed_for(mine, enemy, ang_vel, False)
            if needed is None:
                continue
            # Send whatever we can — even a weakening attack is useful
            send = min(sendable, needed)
            if send <= 0:
                continue
            score = (enemy.production + 1) / (enemy.ships + 1) / (travel + 1)
            if best is None or score > best[0]:
                best = (score, angle, send, enemy.id)

        if best is not None:
            _, angle, send, eid = best
            moves.append([mine.id, angle, send])
            used_sources.add(mine.id)
            claimed.add(eid)

    return moves
