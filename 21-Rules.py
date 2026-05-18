"""
Orbit Wars - Perfect Aiming Agent

Strategy: for each owned planet, target the nearest unowned planet.
Predicts where orbiting planets will be when the fleet arrives.
"""

import math

CENTER_X = 50.0
CENTER_Y = 50.0
MAX_SPEED = 6.0


class Planet:
    __slots__ = ("id", "owner", "x", "y", "radius", "ships", "production")

    def __init__(self, id, owner, x, y, radius, ships, production, comet_planet_ids):
        self.id = id
        self.owner = owner
        self.x = x
        self.y = y
        self.radius = radius
        self.ships = ships
        self.production = production

        # Comets
        self.comet_planet_ids = comet_planet_ids
        self.is_comet = self.id in comet_planet_ids

        # 




    # -- physics helpers --

    def is_moving(self):
        orbital_radius = math.hypot(self.x - CENTER_X, self.y - CENTER_Y)
        return orbital_radius + self.radius < 50

    def predict_position(self, angular_velocity, t):
        r = math.hypot(self.x - CENTER_X, self.y - CENTER_Y)
        angle = math.atan2(self.y - CENTER_Y, self.x - CENTER_X) + angular_velocity * t
        return CENTER_X + r * math.cos(angle), CENTER_Y + r * math.sin(angle)

    def _find_intercept(self, tgt, angular_velocity, speed, t_max=100):
        for t in range(t_max):
            future_x, future_y = tgt.predict_position(angular_velocity, t)
            dist = math.hypot(future_x - self.x, future_y - self.y)
            if dist - t * speed < self.radius + tgt.radius:
                return t, future_x, future_y
        return None

    def aiming(self, tgt, nb_ships, angular_velocity=0.0):
        """Return (angle, eta) to fire nb_ships toward tgt, or None."""
        speed = _fleet_speed(nb_ships)

        if not tgt.is_moving() or angular_velocity == 0.0:
            dx, dy = tgt.x - self.x, tgt.y - self.y
            travel_dist = max(0.0, math.hypot(dx, dy) - self.radius - tgt.radius)
            eta = max(1, math.ceil(travel_dist / speed))
            return math.atan2(dy, dx), eta

        result = self._find_intercept(tgt, angular_velocity, speed)
        if result is None:
            return None
        t, future_x, future_y = result
        return math.atan2(future_y - self.y, future_x - self.x), t


def _fleet_speed(nb_ships):
    if nb_ships <= 1:
        return 1.0
    ratio = math.log(nb_ships) / math.log(1000.0)
    return 1.0 + (MAX_SPEED - 1.0) * max(0.0, min(1.0, ratio)) ** 1.5


def agent(obs):
    moves = []
    player = obs.get("player", 0) if isinstance(obs, dict) else obs.player
    raw_planets = obs.get("planets", []) if isinstance(obs, dict) else obs.planets
    angular_velocity = obs.get("angular_velocity", 0.0) if isinstance(obs, dict) else getattr(obs, "angular_velocity", 0.0)
    comet_planet_ids = set(obs.get("comet_planet_ids", []) if isinstance(obs, dict) else getattr(obs, "comet_planet_ids", []))

    planets = [Planet(*p, comet_planet_ids) for p in raw_planets]
    my_planets = [p for p in planets if p.owner == player]
    targets = [p for p in planets if p.owner != player and p.id not in comet_planet_ids]

    if not targets:
        return moves

    for mine in my_planets:
        nearest = min(targets, key=lambda t: math.hypot(t.x - mine.x, t.y - mine.y))

        result = mine.aiming(nearest, 1, angular_velocity)
        if result is None:
            continue
        _, eta = result
        production_accrued = nearest.production * eta if nearest.owner != -1 else 0
        ships_needed = nearest.ships + production_accrued + 1

        if mine.ships < ships_needed:
            continue

        result = mine.aiming(nearest, ships_needed, angular_velocity)
        if result is None:
            continue
        angle, _ = result
        moves.append([mine.id, angle, ships_needed])

    return moves
