import math
from kaggle_environments.envs.orbit_wars.orbit_wars import Planet

# kaggle competitions submit orbit-wars -f 02-Getting_started_only_non_moving.py -m "02-Getting_started_only_non_moving"

episodeSteps = 500
actTimeout = 1
shipSpeed = 6
sunRadius = 10
boardSize = 100
cometSpeed = 4

def get_planet_initial_position_from_id(planet_id, initial_planets):
    for id, owner, x, y, radius, ships, production in initial_planets:
        if planet_id == id:
            return x, y


def nearest_planet_sniper(obs):
    moves = []
    player = obs.get("player", 0) if isinstance(obs, dict) else obs.player
    raw_planets = obs.get("planets", []) if isinstance(obs, dict) else obs.planets
    planets = [Planet(*p) for p in raw_planets]
    initial_planets = obs.get("initial_planets", []) if isinstance(obs, dict) else obs.initial_planets

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
            if dist < min_dist:
                i_x, i_y = get_planet_initial_position_from_id(planet_id=t.id, initial_planets=initial_planets)
                if t.x == i_x and t.y == i_y: 
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