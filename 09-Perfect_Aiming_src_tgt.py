"""
Orbit Wars - Perfect Aiming Agent

Strategy: for each owned planet, target the nearest unowned planet.
Predicts where orbiting planets will be when the fleet arrives.
"""

import math
from kaggle_environments.envs.orbit_wars.orbit_wars import Planet

CENTER_X = 50.0
CENTER_Y = 50.0
MAX_SPEED = 6.0


def fleet_speed(nb_ships):
    if nb_ships <= 1:
        return 1.0
    ratio = math.log(nb_ships) / math.log(1000.0)
    return 1.0 + (MAX_SPEED - 1.0) * max(0.0, min(1.0, ratio)) ** 1.5


def is_planet_moving(planet):
    orbital_radius = math.hypot(planet.x - CENTER_X, planet.y - CENTER_Y)
    return orbital_radius + planet.radius < 50


def _predict_position(x, y, angular_velocity, t):
    r = math.hypot(x - CENTER_X, y - CENTER_Y)
    angle = math.atan2(y - CENTER_Y, x - CENTER_X) + angular_velocity * t
    return CENTER_X + r * math.cos(angle), CENTER_Y + r * math.sin(angle)


def _find_intercept(src_x, src_y, src_radius, tgt_x, tgt_y, tgt_radius, angular_velocity, speed, t_max=100):
    for t in range(t_max):
        future_x, future_y = _predict_position(tgt_x, tgt_y, angular_velocity, t)
        dist = math.hypot(future_x - src_x, future_y - src_y)
        if dist - t * speed < src_radius + tgt_radius:
            return t, future_x, future_y
    return None


def aiming(from_id, to_id, planets, nb_ships, angular_velocity=0.0):
    """Return (angle, eta) to fire nb_ships from planet from_id toward planet to_id, or None."""
    planet_by_id = {p.id: p for p in planets}
    src = planet_by_id[from_id]
    tgt = planet_by_id[to_id]
    speed = fleet_speed(nb_ships)

    if not is_planet_moving(tgt) or angular_velocity == 0.0:
        dx, dy = tgt.x - src.x, tgt.y - src.y
        travel_dist = max(0.0, math.hypot(dx, dy) - src.radius - tgt.radius)
        eta = max(1, math.ceil(travel_dist / speed))
        angle = math.atan2(dy, dx)
        return angle, eta

    result = _find_intercept(src.x, src.y, src.radius, tgt.x, tgt.y, tgt.radius, angular_velocity, speed)
    if result is None:
        return None

    t, future_x, future_y = result
    angle = math.atan2(future_y - src.y, future_x - src.x)
    return angle, t


def agent(obs):
    moves = []
    player = obs.get("player", 0) if isinstance(obs, dict) else obs.player
    raw_planets = obs.get("planets", []) if isinstance(obs, dict) else obs.planets
    angular_velocity = obs.get("angular_velocity", 0.0) if isinstance(obs, dict) else getattr(obs, "angular_velocity", 0.0)
    comet_planet_ids = set(obs.get("comet_planet_ids", []) if isinstance(obs, dict) else getattr(obs, "comet_planet_ids", []))

    planets = [Planet(*p) for p in raw_planets]
    my_planets = [p for p in planets if p.owner == player]
    targets = [p for p in planets if p.owner != player and p.id not in comet_planet_ids]

    if not targets:
        return moves

    for mine in my_planets:
        nearest = min(targets, key=lambda t: math.hypot(t.x - mine.x, t.y - mine.y))

        result = aiming(mine.id, nearest.id, planets, 1, angular_velocity)
        if result is None:
            continue
        _, eta = result
        production_accrued = nearest.production * eta if nearest.owner != -1 else 0
        ships_needed = nearest.ships + production_accrued + 1

        if mine.ships < ships_needed:
            continue

        result = aiming(mine.id, nearest.id, planets, ships_needed, angular_velocity)
        if result is None:
            continue
        angle, _ = result
        moves.append([mine.id, angle, ships_needed])

    return moves
