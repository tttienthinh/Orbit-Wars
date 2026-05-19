"""
Orbit Wars - Perfect Aiming Agent (Board refactor)

Identical logic to 24-Rules_Target_big_prod.py, restructured:
- Planet: raw data + physics + setup methods (no decision logic)
- Board: owns game state, simulation, and move generation
"""
import math
import kaggle_environments as ke

CENTER_X = 50.0
CENTER_Y = 50.0
MAX_SPEED = 6.0
NB_FORECAST_STEPS = 20
NEIGHBOURHOOD = 5


def _fleet_speed(nb_ships):
    if nb_ships <= 1:
        return 1.0
    ratio = math.log(nb_ships) / math.log(1000.0)
    return 1.0 + (MAX_SPEED - 1.0) * max(0.0, min(1.0, ratio)) ** 1.5


class OrbitWarsSimulator:
    MAX_SPEED = 6.0
    BOARD_SIZE = 100.0
    CENTER = 50.0
    SUN_RADIUS = 10.0
    ROTATION_RADIUS_LIMIT = 50.0

    def __init__(self, obs):
        def _get(attr, default):
            return obs.get(attr, default) if isinstance(obs, dict) else getattr(obs, attr, default)

        self.angular_velocity = _get("angular_velocity", 0.0)
        self.comet_pid_set = set(_get("comet_planet_ids", []))

        raw_planets = _get("planets", [])
        self.planets = [list(p) for p in raw_planets]
        self.initial_planets = [list(p) for p in raw_planets]
        self.initial_by_id = {p[0]: p for p in self.initial_planets}

        self.fleets = [list(f) for f in _get("fleets", [])]

        self.comets = [
            {
                "planet_ids": list(g["planet_ids"]),
                "paths": [[list(pos) for pos in path] for path in g["paths"]],
                "path_index": g["path_index"],
            }
            for g in _get("comets", [])
        ]

        self.sim_step = 0

    @staticmethod
    def _pt_seg_dist(p, v, w):
        """Minimum distance from point p to segment v→w."""
        l2 = (v[0] - w[0]) ** 2 + (v[1] - w[1]) ** 2
        if l2 == 0.0:
            return math.hypot(p[0] - v[0], p[1] - v[1])
        t = max(0.0, min(1.0, (
            (p[0] - v[0]) * (w[0] - v[0]) + (p[1] - v[1]) * (w[1] - v[1])
        ) / l2))
        proj_x = v[0] + t * (w[0] - v[0])
        proj_y = v[1] + t * (w[1] - v[1])
        return math.hypot(p[0] - proj_x, p[1] - proj_y)

    def step(self):
        self.sim_step += 1

        # Phase 1: comet pre-expiry (remove any comets already past their path end)
        pre_expired = {
            pid
            for g in self.comets
            for i, pid in enumerate(g["planet_ids"])
            if g["path_index"] >= len(g["paths"][i])
        }
        self._expire_comets(pre_expired)

        # Phase 2: production
        for planet in self.planets:
            if planet[1] != -1:
                planet[5] += planet[6]

        # Phase 3: fleet movement + continuous collision
        fleets_to_remove = set()
        combat_lists = {p[0]: [] for p in self.planets}

        for fleet in self.fleets:
            angle = fleet[4]
            ships = fleet[6]
            speed = _fleet_speed(ships)
            old_pos = (fleet[2], fleet[3])
            fleet[2] += math.cos(angle) * speed
            fleet[3] += math.sin(angle) * speed
            new_pos = (fleet[2], fleet[3])

            hit = False
            for planet in self.planets:
                if self._pt_seg_dist((planet[2], planet[3]), old_pos, new_pos) < planet[4]:
                    combat_lists[planet[0]].append(fleet)
                    fleets_to_remove.add(id(fleet))
                    hit = True
                    break
            if hit:
                continue

            if not (0 <= fleet[2] <= self.BOARD_SIZE and 0 <= fleet[3] <= self.BOARD_SIZE):
                fleets_to_remove.add(id(fleet))
                continue

            if self._pt_seg_dist(
                (self.CENTER, self.CENTER), old_pos, new_pos
            ) < self.SUN_RADIUS:
                fleets_to_remove.add(id(fleet))

        # Phases 4–6 stubbed until next tasks
        self.fleets = [f for f in self.fleets if id(f) not in fleets_to_remove]

        return [
            {"id": p[0], "owner": p[1], "x": p[2], "y": p[3],
             "radius": p[4], "ships": p[5], "production": p[6]}
            for p in self.planets
        ]

    def _sweep(self, planet, old_pos, new_pos, fleets_to_remove, combat_lists):
        if old_pos == new_pos:
            return
        for fleet in self.fleets:
            if id(fleet) not in fleets_to_remove:
                if self._pt_seg_dist(
                    (fleet[2], fleet[3]), old_pos, new_pos
                ) < planet[4]:
                    combat_lists[planet[0]].append(fleet)
                    fleets_to_remove.add(id(fleet))

    def _expire_comets(self, expired_pids):
        if not expired_pids:
            return
        self.planets = [p for p in self.planets if p[0] not in expired_pids]
        self.initial_by_id = {
            pid: p for pid, p in self.initial_by_id.items()
            if pid not in expired_pids
        }
        self.comet_pid_set -= expired_pids
        for g in self.comets:
            g["planet_ids"] = [pid for pid in g["planet_ids"] if pid not in expired_pids]
        self.comets = [g for g in self.comets if g["planet_ids"]]

    def run(self, n):
        raise NotImplementedError


class Planet:
    NB_NEARBY_OUT_OF = 5

    def __init__(self, id, owner, x, y, radius, ships, production, comet_planet_ids, angular_velocity):
        # -- raw data (immutable after construction) --
        self.id = id
        self.owner = owner
        self.x = x
        self.y = y
        self.radius = radius
        self.ships = ships
        self.production = production
        self.angular_velocity = angular_velocity
        self.is_moving = math.hypot(self.x - CENTER_X, self.y - CENTER_Y) + self.radius < 50
        self.comet_planet_ids = comet_planet_ids
        self.is_comet = self.id in comet_planet_ids

        # -- computed state (declared here, populated by Board) --
        self.nearby_planets = []
        self.nb_nearby_mine = None
        self.nb_nearby_enemy = None
        self.nb_nearby_neutral = None
        self.nature = None
        self.nexts = []  # list of dicts {id, owner, x, y, radius, ships, production}

    # -- setup (called by Board in sequence) --

    def compute_neighbors(self, planets_no_comets):
        candidates = []
        for p in planets_no_comets:
            if p.id == self.id:
                continue
            result = self.aiming(p, 1)
            if result is None:
                continue
            _, eta = result
            candidates.append((eta, p))
        self.nearby_planets = [p for _, p in sorted(candidates, key=lambda item: item[0])]

    def compute_counts(self):
        self.nb_nearby_mine = 0
        self.nb_nearby_enemy = 0
        self.nb_nearby_neutral = 0
        for p in self.nearby_planets[:self.NB_NEARBY_OUT_OF]:
            if p.owner == -1:
                self.nb_nearby_neutral += 1
            elif p.owner == self.owner:
                self.nb_nearby_mine += 1
            else:
                self.nb_nearby_enemy += 1

    def assign_nature(self):
        if self.nb_nearby_enemy >= 1:
            self.nature = "Conqueror"
        elif self.nb_nearby_neutral >= 1:
            self.nature = "Explorer"
        else:
            self.nature = "Supplier"

    # -- physics --

    def predict_position(self, t):
        r = math.hypot(self.x - CENTER_X, self.y - CENTER_Y)
        angle = math.atan2(self.y - CENTER_Y, self.x - CENTER_X) + self.angular_velocity * t
        return CENTER_X + r * math.cos(angle), CENTER_Y + r * math.sin(angle)

    def _find_intercept(self, tgt, speed, t_max=100):
        for t in range(t_max):
            future_x, future_y = tgt.predict_position(t)
            dist = math.hypot(future_x - self.x, future_y - self.y)
            if dist - t * speed < self.radius + tgt.radius:
                return t, future_x, future_y
        return None

    def aiming(self, tgt, nb_ships):
        """Return (angle, eta) to fire nb_ships toward tgt, or None."""
        speed = _fleet_speed(nb_ships)
        if not tgt.is_moving or self.angular_velocity == 0.0:
            dx, dy = tgt.x - self.x, tgt.y - self.y
            travel_dist = max(0.0, math.hypot(dx, dy) - self.radius - tgt.radius)
            eta = max(1, math.ceil(travel_dist / speed))
            return math.atan2(dy, dx), eta
        result = self._find_intercept(tgt, speed)
        if result is None:
            return None
        t, future_x, future_y = result
        return math.atan2(future_y - self.y, future_x - self.x), t


class Board:
    def __init__(self, obs):
        print(obs.remainingOverageTime)

        # -- extract raw obs fields --
        self.player = obs.get("player", 0) if isinstance(obs, dict) else obs.player
        self.angular_velocity = (obs.get("angular_velocity", 0.0) if isinstance(obs, dict)
                                 else getattr(obs, "angular_velocity", 0.0))
        self.comet_planet_ids = set(obs.get("comet_planet_ids", []) if isinstance(obs, dict)
                                    else getattr(obs, "comet_planet_ids", []))
        raw_planets = obs.get("planets", []) if isinstance(obs, dict) else obs.planets
        self.fleets = obs.fleets

        # -- build planets --
        self.planets = [Planet(*p, self.comet_planet_ids, self.angular_velocity) for p in raw_planets]
        self.planets_no_comets = [p for p in self.planets if not p.is_comet]
        self.planets_dico = {p.id: p for p in self.planets}
        self.my_planets = [p for p in self.planets if p.owner == self.player]
        self.targets = [p for p in self.planets_no_comets if p.owner != self.player]

        # -- forward simulation + planet setup (skipped when game is already won) --
        if self.targets:
            self._run_simulation(obs)
            for p in self.my_planets:
                p.compute_neighbors(self.planets_no_comets)
                p.compute_counts()
                p.assign_nature()

    def _run_simulation(self, obs):
        max_planet_id = max(p.id for p in self.planets)
        num_agents = 2 if max_planet_id <= 1 else 4

        env = ke.make("orbit_wars", debug=True)
        env.reset(num_agents)
        for i in range(num_agents):
            env.state[i].action = []
            env.state[i].reward = 0
            env.state[i].observation.remainingOverageTime = obs.remainingOverageTime
            env.state[i].observation.player = i
            env.state[i].observation.angular_velocity = obs.angular_velocity
            env.state[i].observation.next_fleet_id = obs.next_fleet_id
            env.state[i].observation.comet_planet_ids = obs.comet_planet_ids.copy()
            env.state[i].observation.planets = [planet.copy() for planet in obs.planets]
            env.state[i].observation.fleets = [fleet.copy() for fleet in obs.fleets]
            # Use current positions as the sim's "initial" so orbiting angles are correct.
            # The rotation formula is: angle = atan2(initial_y, initial_x) + ω*step
            # If we keep game-start initial_planets, step=1 in the sim gives the wrong angle.
            env.state[i].observation.initial_planets = [planet.copy() for planet in obs.planets]

        for step in range(NB_FORECAST_STEPS):
            step_observation = env.step([[]] * num_agents)
            next_obs = step_observation[self.player].observation
            for id, owner, x, y, radius, ships, production in next_obs.planets:
                self.planets_dico[id].nexts.append({
                    "id": id, "owner": owner,
                    "x": x, "y": y,
                    "radius": radius, "ships": ships, "production": production,
                })

    def get_moves(self):
        if not self.targets:
            return []
        moves = []
        for planet in self.my_planets:
            moves += self._create_actions(planet)
        return moves

    def _create_one_action(self, planet, target, ships_needed=None) -> list:
        result = planet.aiming(target, 1)
        if result is not None:
            _, eta = result
            production_accrued = target.production * eta if target.owner != -1 else 0
            if ships_needed is None:
                ships_needed = target.ships + production_accrued + 1
            if planet.ships >= ships_needed:
                result = planet.aiming(target, ships_needed)
                if result is not None:
                    angle, _ = result
                    return [[planet.id, angle, ships_needed]]
        return []

    def _create_actions(self, planet) -> list:
        actions = []
        if planet.is_comet:
            if abs(planet.x - 50) > 45 or abs(planet.y - 50) > 45:
                for target in planet.nearby_planets[:planet.NB_NEARBY_OUT_OF]:
                    action = self._create_one_action(planet, target, planet.ships)
                    if action != []:
                        actions += action
                        return actions

        elif planet.nature == "Supplier":
            for target in planet.nearby_planets[:planet.NB_NEARBY_OUT_OF]:
                if target.nexts[NB_FORECAST_STEPS - 1]['owner'] != planet.owner or planet.ships > target.nexts[NB_FORECAST_STEPS - 1]['ships'] * 2:
                    action = self._create_one_action(planet, target, planet.ships // 4)
                    if action != []:
                        actions += action
                        return actions

        elif planet.nature == "Explorer":
            # Priority 1: intercept planets being claimed by an enemy
            for target in planet.nearby_planets[:NEIGHBOURHOOD]:
                if target.nexts[NB_FORECAST_STEPS - 1]['owner'] not in (-1, planet.owner):
                    action = self._create_one_action(planet, target)
                    if action != []:
                        return action
            # Priority 2: highest Exploring score among neutral neighbours
            return self._explorer_score_action(planet)

        elif planet.nature == "Conqueror":
            for target in planet.nearby_planets[:planet.NB_NEARBY_OUT_OF]:
                if target.nexts[NB_FORECAST_STEPS - 1]['owner'] == -1 and planet.ships > target.nexts[NB_FORECAST_STEPS - 1]['ships'] * 2:
                    action = self._create_one_action(planet, target, planet.ships // 2)
                    if action != []:
                        actions += action
                        return actions
            for target in planet.nearby_planets[:planet.NB_NEARBY_OUT_OF]:
                if target.nexts[NB_FORECAST_STEPS - 1]['owner'] != planet.owner and planet.ships > target.nexts[NB_FORECAST_STEPS - 1]['ships'] * 2:
                    action = self._create_one_action(planet, target, planet.ships // 2)
                    if action != []:
                        actions += action
                        return actions

        return actions

    def _explorer_score_action(self, planet) -> list:
        neutral = [t for t in planet.nearby_planets[:NEIGHBOURHOOD] if t.owner == -1]
        if not neutral or planet.production == 0:
            return []
        T = sum(t.ships for t in neutral) / planet.production
        best = max(neutral, key=lambda t: t.production * (T - t.ships / planet.production))
        return self._create_one_action(planet, best)


def agent(obs):
    return Board(obs).get_moves()
