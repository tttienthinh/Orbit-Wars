"""
Orbit Wars - Rule-Based Agent (07)

Improvements over 05 (nearest-planet sniper with orbital intercept):

  Rule 1 - Garrison: always keep some ships at home so planets aren't left
           defenceless.

  Rule 2 - Target scoring: pick by production/cost rather than raw distance.
           A nearby low-value planet is worse than a slightly farther rich one.

  Rule 3 - Neutral preference: neutral planets cost no production penalty
           during travel, so they get a scoring bonus over enemy planets.

  Rule 4 - Production-adjusted cost: enemy planets keep producing while our
           fleet is in transit, so ships_needed = garrison + travel_turns *
           production + 1.

  Rule 5 - No duplicate targeting: once a target is assigned this turn, other
           planets won't pile on (avoids wasted ships).
"""

import math
from kaggle_environments.envs.orbit_wars.orbit_wars import Planet

CENTER_X = 50.0
CENTER_Y = 50.0
MAX_SPEED = 6.0

# Tuning knobs
GARRISON_RATIO = 0.30       # keep this fraction of ships at home
GARRISON_MIN = 8            # always keep at least this many ships
NEUTRAL_SCORE_BONUS = 1.5   # scoring multiplier for neutral targets
INTERCEPT_HORIZON = 100     # max turns to search for moving-planet intercept


# ---------------------------------------------------------------------------
# Utility helpers (same physics as 05)
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
    ratio = math.log(ships) / math.log(1000.0)
    ratio = max(0.0, min(1.0, ratio))
    return 1.0 + (MAX_SPEED - 1.0) * (ratio ** 1.5)


def orbital_position(x, y, angular_velocity, t):
    r = dist_to_center(x, y)
    angle = math.atan2(y - CENTER_Y, x - CENTER_X) + angular_velocity * t
    return CENTER_X + r * math.cos(angle), CENTER_Y + r * math.sin(angle)


def intercept_time(ox, oy, tx, ty, angular_velocity, ships, horizon=INTERCEPT_HORIZON):
    """Binary-search for the earliest turn t where fleet can reach planet."""
    speed = fleet_speed(ships)
    for t in range(1, horizon + 1):
        px, py = orbital_position(tx, ty, angular_velocity, t)
        if dist(ox, oy, px, py) <= speed * t:
            return t, px, py
    return None, None, None


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def estimate_ships_needed(mine, target, angular_velocity, is_neutral):
    """Ships required to capture target, accounting for travel and production."""
    if is_orbiting(target):
        n_ships_guess = max(target.ships + 1, 10)
        t, px, py = intercept_time(mine.x, mine.y, target.x, target.y,
                                   angular_velocity, n_ships_guess)
        travel = t if t is not None else INTERCEPT_HORIZON
    else:
        travel = max(1, int(dist(mine.x, mine.y, target.x, target.y) /
                             fleet_speed(max(target.ships + 1, 10))))

    if is_neutral:
        return target.ships + 1, travel
    else:
        # Enemy keeps producing during transit
        garrison_on_arrival = target.ships + travel * target.production
        return max(garrison_on_arrival + 1, target.ships + 1), travel


def score_target(mine, target, angular_velocity, is_neutral):
    """Higher score = better target. Returns (score, ships_needed, travel_turns)."""
    ships_needed, travel = estimate_ships_needed(mine, target, angular_velocity, is_neutral)

    if ships_needed <= 0:
        return -1, ships_needed, travel

    # Value: production per unit cost and time
    # Prefer high production, low cost, short travel
    score = (target.production + 1) / (ships_needed * (travel + 1))

    if is_neutral:
        score *= NEUTRAL_SCORE_BONUS

    return score, ships_needed, travel


# ---------------------------------------------------------------------------
# Main agent
# ---------------------------------------------------------------------------

def agent(obs):
    moves = []

    player = obs.get("player", 0) if isinstance(obs, dict) else obs.player
    raw_planets = obs.get("planets", []) if isinstance(obs, dict) else obs.planets
    angular_velocity = obs.get("angular_velocity", 0) if isinstance(obs, dict) else obs.angular_velocity
    comet_ids = set(obs.get("comet_planet_ids", []) if isinstance(obs, dict) else obs.comet_planet_ids)

    planets = [Planet(*p) for p in raw_planets]
    my_planets = [p for p in planets if p.owner == player]
    targets = [p for p in planets if p.owner != player and p.id not in comet_ids]

    if not targets:
        return moves

    claimed_targets = set()  # Rule 5: no duplicate targeting this turn

    # Score all (source, target) pairs and attack best first
    candidates = []
    for mine in my_planets:
        garrison = max(GARRISON_MIN, int(mine.ships * GARRISON_RATIO))
        sendable = mine.ships - garrison
        if sendable <= 0:
            continue

        for t in targets:
            is_neutral = (t.owner == -1)
            score, ships_needed, travel = score_target(mine, t, angular_velocity, is_neutral)
            if score > 0 and ships_needed <= sendable:
                candidates.append((score, mine, t, ships_needed, travel, is_neutral))

    # Sort descending by score; greedily assign best moves
    candidates.sort(key=lambda c: c[0], reverse=True)

    used_sources = set()  # one move per source planet per turn

    for score, mine, target, ships_needed, travel, is_neutral in candidates:
        if mine.id in used_sources:
            continue
        if target.id in claimed_targets:
            continue

        # Compute aim angle, accounting for orbital motion
        if is_orbiting(target):
            t_hit, px, py = intercept_time(mine.x, mine.y, target.x, target.y,
                                           angular_velocity, ships_needed)
            if t_hit is None:
                continue
            angle = math.atan2(py - mine.y, px - mine.x)
        else:
            angle = math.atan2(target.y - mine.y, target.x - mine.x)

        moves.append([mine.id, angle, ships_needed])
        used_sources.add(mine.id)
        claimed_targets.add(target.id)

    return moves
