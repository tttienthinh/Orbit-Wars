import math
from kaggle_environments.envs.orbit_wars.orbit_wars import Planet
# https://www.kaggle.com/code/vincentcaujolle/orbit-wars-pathfinding
# kaggle competitions submit orbit-wars -f 02-Getting_started_only_non_moving.py -m "02-Getting_started_only_non_moving"

CENTER_X = 50.0
CENTER_Y = 50.0

MAX_SPEED = 6.0


def get_distance(x0, y0, x1=CENTER_X, y1=CENTER_Y):
    return math.sqrt((x0-x1) ** 2 + (y0-y1) ** 2)


def is_planet_moving(planet):
    orbital_radius = get_distance(planet.x, planet.y)
    planet_radius = planet.radius
    return orbital_radius + planet_radius < 50


def fleet_speed(ships):
    """
    1.0 + (maxSpeed - 1.0) * (log(ships) / log(1000)) ^ 1.5
    """
    if ships <= 1:
        return 1.0
    ratio = math.log(ships) / math.log(1000.0)
    ratio = max(0.0, min(1.0, ratio))
    return 1.0 + (MAX_SPEED - 1.0) * (ratio**1.5)



def position_to_angle(x, y):
    scaled_x = (x-CENTER_X) 
    scaled_y = (y-CENTER_Y) 
    return math.atan2(scaled_y, scaled_x)


def angle_to_position(angle, orbital_radius=1):
    x = math.cos(angle) * orbital_radius
    y = math.sin(angle) * orbital_radius
    return x, y


def next_position(x, y, angular_velocity, t=1):
    orbital_radius = get_distance(x, y)
    angle = position_to_angle(x, y)
    new_angle = angle + angular_velocity * t
    return angle_to_position(new_angle, orbital_radius=orbital_radius)


def distance_at_t(x0, y0, x1, y1, angular_velocity, t=1):
    new_x1, new_y1 = next_position(x1, y1, angular_velocity, t)
    return get_distance(x0, y0, new_x1, new_y1)


def get_first_t(x0, y0, x1, y1, angular_velocity, speed, t_max=100):
    for t in range(t_max):
        if distance_at_t(x0, y0, x1, y1, angular_velocity, t=t) - t * speed < 0:
            return True, t
    return False, t_max




def nearest_planet_sniper(obs):
    moves = []
    player = obs.get("player", 0) if isinstance(obs, dict) else obs.player
    raw_planets = obs.get("planets", []) if isinstance(obs, dict) else obs.planets
    planets = [Planet(*p) for p in raw_planets]
    angular_velocity = obs.get("angular_velocity", 0) if isinstance(obs, dict) else obs.angular_velocity

    # Separate our planets from targets
    my_planets = [p for p in planets if p.owner == player]
    targets = [p for p in planets if p.owner != player]

    if not targets:
        return moves

    for mine in my_planets:
        # Find the nearest planet we don't own
        nearest = None
        min_dist = float('inf')
        for t in targets:
            dist = math.sqrt((mine.x - t.x)**2 + (mine.y - t.y)**2)
            if dist < min_dist and not is_planet_moving(planet=t):
                min_dist = dist
                nearest = t

        if nearest is None:
            continue

        # How many ships do we need? Target's garrison + 1
        ships_needed = max(nearest.ships + 1, 20)

        # Only send if we have enough
        if mine.ships >= ships_needed:
            # Calculate angle from our planet to the target
            angle = math.atan2(nearest.y - mine.y, nearest.x - mine.x)
            moves.append([mine.id, angle, ships_needed])

    return moves