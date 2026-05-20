import math
import kaggle_environments.envs.orbit_wars.orbit_wars as ow
import numpy as np

# https://www.kaggle.com/code/djenkivanov/orbit-wars-agent-ow-proto-passed-1-000

fleet_trajectories = []
reinforcement_trajectories = []
moving_planets = []
planets_coords = {}
steps = 0

MAX_SPEED = 6.0
# could use RL in future to tune these vars to optimal values
MIN_SHIPS_MINE_ATTACK = 5
MIN_SHIPS_TARGET_COOP_ATTACK = 20
COOP_PLANET_CAP = 8
COLLIDE_TICK_THOLD = 1

FORMULA_DIST = 100
FORMULA_PROD_MULT = 15
FORMULA_ENEMY_BONUS_MULT = 10
FORMULA_TOTAL_SHIPS_PERCENT = 0.7


def get_custom_score(m, t):
    dist = math.sqrt((m.x - t.x)**2 + (m.y - t.y)**2)

    min_ships = t.ships + 1
    fleet_speed = 1.0 + (MAX_SPEED - 1.0) * (math.log(max(1, min_ships)) / math.log(1000)) ** 1.5
    eta = dist / fleet_speed

    enemy_produced = 0
    enemy_bonus = 0
    if t.owner != -1:
        enemy_produced = eta * t.production
        enemy_bonus = t.production

    total_ships = min_ships + enemy_produced

    # + close targets
    # + high production
    # + if planet is owned by enemy (capturing planet is more valuable because we gain ships, they lose ships)
    # - lot of enemies and enemies produced by arrival
    # - slow arrivals
    
    return (
        (FORMULA_DIST - dist)
        + (FORMULA_PROD_MULT * t.production)
        + (FORMULA_ENEMY_BONUS_MULT * enemy_bonus)
        - (FORMULA_TOTAL_SHIPS_PERCENT * total_ships)
        - (2 * eta)
    )

            

def get_planets_under_attack(mine, fleets, player, vel):
    mov_pl_traj = {}
    under_attack = {}
    seen = set()
    fleets = [f for f in fleets if f.owner != player]
    for m in mine:
        if m.id in moving_planets:
            mov_pl_traj[m.id] = get_planet_trajectories(m, vel)

    for f in fleets:
        fleet_speed = 1.0 + (MAX_SPEED - 1.0) * (math.log(f.ships) / math.log(1000)) ** 1.5
        prev_x = f.x
        prev_y = f.y

        for tick in range(1, 61):
            next_x = f.x + math.cos(f.angle) * fleet_speed * tick
            next_y = f.y + math.sin(f.angle) * fleet_speed * tick

            for m in mine:
                if m.id in moving_planets:
                    m_x, m_y = mov_pl_traj[m.id][tick-1] # tick is 1 based, index 0 based, so -1
                else:
                    m_x, m_y = m.x, m.y
                    
                if collides(prev_x, prev_y, next_x, next_y, m_x, m_y, m.radius): 
                    if (m.id, f.id) not in seen:
                        if m.id not in under_attack:
                            under_attack[m.id] = {
                                "planet": m,
                                "fleets": []
                            }
                            
                        under_attack[m.id]["fleets"].append({
                            "fleet": f,
                            "arrive_tick": tick
                        })
                        seen.add((m.id, f.id))
            
            prev_x = next_x
            prev_y = next_y            
                    
    return under_attack
    


def refresh_local_obs(obs):
    planets = [ow.Planet(*p) for p in obs.get("planets", [])]
    mine = [p for p in planets if p.owner == obs.get("player", [])]
    targets = [p for p in planets if p.owner != obs.get("player", [])]
    player = obs.get("player", -2)
    fleets = [ow.Fleet(*f) for f in obs.get("fleets", [])]

    return {
        "planets": planets,
        "mine": mine,
        "targets": targets,
        "player": player,
        "fleets": fleets
    }

def sun_collision(m, fleet_speed, angle, ticks=61):
    prev_x = m.x
    prev_y = m.y

    for tick in range(1, ticks):
        x = m.x + math.cos(angle) * fleet_speed * tick
        y = m.y + math.sin(angle) * fleet_speed * tick

        if collides(prev_x, prev_y, x, y, 50, 50, 10):
            return True

        prev_x = x
        prev_y = y
            
    return False


def calculate_req_ships_moving(attacking_planets, t, base_ships, vel):
    MAX_SPEED = 6.0
    required_ships = base_ships
    planet_trajectories = get_planet_trajectories(t, vel)
    
    for _ in range(3):
        remainder = required_ships
        max_tick = 0

        for a_p in attacking_planets:
            p = a_p["planet"]
            p_ships = min(a_p["ships"], remainder)

            if p_ships > 0:
                p_ships = min(a_p["ships"], max(p_ships, MIN_SHIPS_MINE_ATTACK))

            if p_ships <= 0:
                continue
            
            fleet_speed = 1.0 + (MAX_SPEED - 1.0) * (math.log(max(1, p_ships)) / math.log(1000)) ** 1.5

            found_tick = 0
            for tick, (tx, ty) in enumerate(planet_trajectories, start=1):
                dist = math.sqrt((p.x - tx)**2 + (p.y - ty)**2)
                turns_to_arrive = math.floor(dist / fleet_speed)

                if abs(turns_to_arrive - tick) <= 1:
                    found_tick = tick
                    break

            if found_tick > max_tick:
                max_tick = found_tick

            remainder -= p_ships

        new_req = base_ships + (max_tick * t.production)
        if new_req == required_ships:
            break
        required_ships = new_req
        
    return required_ships

def calculate_req_ships(attacking_planets, t, base_ships):
    required_ships = base_ships
    
    for _ in range(3):
        remainder = required_ships
        max_tick = 0
        
        for a_p in attacking_planets:
            p = a_p["planet"]
            p_ships = min(a_p["ships"], remainder)
            
            if p_ships > 0:
                p_ships = min(a_p["ships"], max(p_ships, MIN_SHIPS_MINE_ATTACK))

            if p_ships <= 0:
                continue
            
            ships_for_speed = max(1, p_ships)
            fleet_speed = 1.0 + (MAX_SPEED - 1.0) * (math.log(ships_for_speed) / math.log(1000)) ** 1.5
            dist = math.sqrt((p.x - t.x)**2 + (p.y - t.y)**2)
            tick_arrival = math.floor(dist / fleet_speed)
            
            if tick_arrival > max_tick:
                max_tick = tick_arrival

            remainder -= p_ships

        new_req = base_ships + (max_tick * t.production)
        
        if new_req == required_ships:
            break
            
        required_ships = new_req
    
    return required_ships


def calculate_angle(m, t):
    return math.atan2(t.y - m.y, t.x - m.x)
    

def find_angle_to_moving_planet(p, t, ships, vel):
    fleet_speed = 1.0 + (MAX_SPEED - 1.0) * (math.log(ships) / math.log(1000)) ** 1.5
    planet_trajectories = get_planet_trajectories(t, vel)

    for tick, (tx, ty) in enumerate(planet_trajectories, start=1):
        dx = tx - p.x
        dy = ty - p.y
        dist_to_target = math.sqrt(dx**2 + dy**2) - p.radius

        travel_dist = fleet_speed * tick
        miss_dist = abs(travel_dist - dist_to_target)

        if miss_dist > t.radius:
            continue
        
        angle = math.atan2(dy, dx)
        
        if sun_collision(p, fleet_speed, angle):
            return None, None

        return angle, tick

    return None, None


def collides(x1, y1, x2, y2, cx, cy, r):
    vec_x = x2 - x1
    vec_y = y2 - y1

    vec_to_cx = cx - x1
    vec_to_cy = cy - y1

    vec_length_sq = vec_x**2 + vec_y**2

    if vec_length_sq == 0:
        dx = x1 - cx
        dy = y1 - cy
        return dx**2 + dy**2 <= r**2

    closest_point = (vec_to_cx * vec_x + vec_to_cy * vec_y) / vec_length_sq
    closest_point = max(0, min(1, closest_point))

    closest_x = x1 + closest_point * vec_x
    closest_y = y1 + closest_point * vec_y

    dx = closest_x - cx
    dy = closest_y - cy
    return dx**2 + dy**2 <= r**2


def get_closest_planets_to_target(mine, t):
    planets = []
    for m in mine:
        dist = math.sqrt((m.x - t.x)**2 + (m.y - t.y)**2)
        planets.append((m, dist))
    planets = sorted(planets, key=lambda k: k[1])
    return planets
    

def update_fleet_trajectories(fleets):
    for f_t in fleet_trajectories[:]:
        found = False
        for f in fleets:
            if f.from_planet_id == f_t["mine"].id and abs(f.angle - f_t["angle"]) < 1e-3:
                found = True
                break

        if found:
            f_t["arrive_tick"] = max(0, f_t["arrive_tick"] - 1)

        if not found:
            fleet_trajectories.remove(f_t)


def update_reinforcement_trajectories(planets):
    planet_ids = {p.id for p in planets}
    
    for r_t in reinforcement_trajectories[:]:
        r_t["arrive_tick"] -= 1

        if r_t["arrive_tick"] <= 0:
            reinforcement_trajectories.remove(r_t)
            continue


def get_planet_trajectories(p, vel):
    planet_trajectories = []
    angle = math.atan2(p.y - 50, p.x - 50)
    r = math.sqrt((p.x - 50)**2 + (p.y - 50)**2)
    for tick in range(1, 61): # max 60 ticks
        angle_t = angle + vel * tick
        x_t = 50 + r * math.cos(angle_t)
        y_t = 50 + r * math.sin(angle_t)
        planet_trajectories.append((x_t, y_t))

    return planet_trajectories
    

def fill_moving_planets(obs):
    planets = [ow.Planet(*p) for p in obs.get("planets", [])]
    initial_by_id = {i[0]: ow.Planet(*i) for i in obs.get("initial_planets", [])}
    for p in planets:
        i = initial_by_id[p.id]
        if (p.x, p.y) != (i.x, i.y):
            if p.id not in moving_planets:
                moving_planets.append(p.id)

def get_reinforcement_plans(mine, under_attack):
    reinforcement_plans = {}
    
    for p in mine:
        if p.id in under_attack:
            attacking_fleets = sorted(
                under_attack[p.id]["fleets"],
                key=lambda att: att["arrive_tick"]
            )
            
            incoming_reinforcements = sorted(
                [r for r in reinforcement_trajectories if r["target"].id == p.id],
                key=lambda r: r["arrive_tick"]
            )
            
            p_available_ships = p.ships
            previous_tick = 0
            r_idx = 0

            for att in attacking_fleets:
                att_arrive_tick = att["arrive_tick"]

                p_available_ships += (att_arrive_tick - previous_tick) * p.production
                
                while (
                    r_idx < len(incoming_reinforcements)
                    and incoming_reinforcements[r_idx]["arrive_tick"] <= att_arrive_tick
                ):
                    p_available_ships += incoming_reinforcements[r_idx]["total_ships"]
                    r_idx += 1

                enemy_ships = att["fleet"].ships
                p_available_ships -= enemy_ships
                previous_tick = att_arrive_tick
                
                if p_available_ships < 0:
                    reinforcements_needed = max(MIN_SHIPS_MINE_ATTACK, abs(p_available_ships))
                    reinforcement_plans[p] = {
                        "ships_needed": reinforcements_needed,
                        "needed_by_tick": att_arrive_tick
                    }
                    break
                
    return reinforcement_plans


def agent(obs):
    global steps
    global fleet_trajectories
    global reinforcement_trajectories
    moves = []
    
    if steps < 2:
        steps += 1
        return []
    if steps == 2:
        fill_moving_planets(obs)
        steps = 3  

    lobs = refresh_local_obs(obs)
    update_fleet_trajectories(lobs.get("fleets", []))
    update_reinforcement_trajectories(lobs.get("planets", []))
    comet_planet_ids = obs.get("comet_planet_ids", [])    
    under_attack = get_planets_under_attack(lobs.get("mine", []), lobs.get("fleets", []), lobs.get("player", -2), obs.angular_velocity)
    exhausted_planets_id = set()
    
    if not lobs.get("targets", []):
        return []

    reinforcement_plans = get_reinforcement_plans(lobs.get("mine", []), under_attack)
    for p, plan in reinforcement_plans.items():
        already_reinforced = any(
            r["target"].id == p.id and r["arrive_tick"] >= 0
            for r in reinforcement_trajectories
        )

        if already_reinforced:
            continue
            
        ships_needed = plan["ships_needed"]
        needed_by_tick = plan["needed_by_tick"]
        nearest_planets = get_closest_planets_to_target(lobs.get("mine", []), p)
        
        for row in nearest_planets:
            p_np, _ = row
            
            if p_np.id == p.id or p_np.id in exhausted_planets_id:
                continue

            p_np_available_ships = p_np.ships

            reserved_reinforcement_ships = sum(
                r["total_ships"]
                for r in reinforcement_trajectories
                if r["mine"].id == p_np.id
            )
            
            p_np_available_ships -= reserved_reinforcement_ships

            if p_np.id in under_attack:
                enemy_ships = sum(
                    att["fleet"].ships
                    for att in under_attack[p_np.id]["fleets"]
                )
                p_np_available_ships = max(0, p_np_available_ships - enemy_ships)

            sent_reinforcements = max(MIN_SHIPS_MINE_ATTACK, ships_needed)

            if p_np_available_ships < sent_reinforcements:
                continue
            angle_np = None
            if p.id not in moving_planets:
                angle_np = math.atan2(p.y - p_np.y, p.x - p_np.x)
                fleet_speed = 1.0 + (MAX_SPEED - 1.0) * (math.log(sent_reinforcements) / math.log(1000)) ** 1.5                
                dist = math.sqrt((p.x - p_np.x)**2 + (p.y - p_np.y)**2)
                arrive_tick = math.floor(dist / fleet_speed)

                if arrive_tick > needed_by_tick:
                    continue
                
            else:
                angle_np, arrive_tick = find_angle_to_moving_planet(p_np, p, sent_reinforcements, obs.angular_velocity)

            if angle_np is None or arrive_tick is None:
                continue
            
            moves.append([p_np.id, angle_np, sent_reinforcements])
            exhausted_planets_id.add(p_np.id)
            reinforcement_trajectories.append({
                "mine": p_np,
                "target": p,
                "angle": angle_np,
                "total_ships": sent_reinforcements,
                "arrive_tick": arrive_tick
            })
            break

    for m in sorted(lobs.get("mine", []), key=lambda p: p.ships, reverse=True):
        if m.id in exhausted_planets_id:
            continue

        if m.ships < MIN_SHIPS_MINE_ATTACK:
            continue

        candidate_targets = []
        for t in lobs.get("targets", []):
            if t.id in comet_planet_ids:
                continue

            score = get_custom_score(m, t)
            candidate_targets.append((m, t, score))

        candidate_targets = sorted(candidate_targets, key=lambda x: x[2], reverse=True)
        
    
        for m, t, s in candidate_targets[:3]:
            m_available_ships = m.ships
    
            if m.id in under_attack:
                enemy_ships = sum(
                    att["fleet"].ships
                    for att in under_attack[m.id]["fleets"]
                )
                m_available_ships = max(0, m.ships - enemy_ships)
    
            if m_available_ships < MIN_SHIPS_MINE_ATTACK:
                continue
            
            nearest_planets = get_closest_planets_to_target(lobs.get("mine", []), t)
            safe_nearest_planets = []
            for p, dist in nearest_planets: # check which planets are fit to attack and are not vulnerable
                if p.id == m.id or p.id in exhausted_planets_id:
                    continue
                
                available_ships = p.ships
                
                if p.id in under_attack:
                    enemy_ships = sum(
                        att["fleet"].ships 
                        for att in under_attack[p.id]["fleets"]
                    )
                    available_ships = max(0, p.ships - enemy_ships)
    
                if available_ships < MIN_SHIPS_MINE_ATTACK:
                    continue
                
                safe_nearest_planets.append((p, dist, available_ships))
            
            owned_count = len(lobs.get("mine", []))
            total_count = len(lobs.get("planets", []))
    
            en_route = 0
            if fleet_trajectories:
                en_route = sum(
                    f["total_ships"]
                    for f in fleet_trajectories
                    if f["target"].id == t.id
                )
    
            needed_now = t.ships + 1
            if t.owner != -1:
                needed_now += 3 * t.production
            
            if owned_count < total_count * 0.75: # release all havoc when targets less than ~25%
                if en_route >= needed_now:
                    continue
            
            base_ships = max(MIN_SHIPS_MINE_ATTACK, needed_now - en_route)
            
            extra_ships = 0
            fleet_speed = 0
            angle = None
            arrive_tick = None
    
            if m_available_ships >= base_ships: # single attack
                if t.id in moving_planets: # single moving planet
                    total_ships = base_ships
                    
                    for _ in range(3):
                        angle, arrive_tick = find_angle_to_moving_planet(m, t, total_ships, obs.angular_velocity)

                        if angle is None:
                            break

                        if t.owner != -1:
                            new_total_ships = base_ships + arrive_tick * t.production
                        else:
                            new_total_ships = base_ships

                        if new_total_ships > m_available_ships:
                            angle = None
                            break
                        
                        if new_total_ships == total_ships:
                            break

                        total_ships = new_total_ships
                    extra_ships = total_ships - base_ships
                        
                else: # single static planet
                    angle = calculate_angle(m, t) # single static unowned
                    total_ships = base_ships
                    dist = math.sqrt((t.x - m.x)**2 + (t.y - m.y)**2)
                    fleet_speed = 1.0 + (MAX_SPEED - 1.0) * (math.log(max(1, total_ships)) / math.log(1000)) ** 1.5
                    arrive_tick = math.floor(dist / fleet_speed)
                    
                    if t.owner != -1: # single static owned
                        for _ in range(3):
                            fleet_speed = 1.0 + (MAX_SPEED - 1.0) * (math.log(max(1, total_ships)) / math.log(1000)) ** 1.5
                            turns_to_arrive = math.floor(dist / fleet_speed)
                            
                            extra_ships = turns_to_arrive * t.production
                            new_total_ships = base_ships + extra_ships

                            if new_total_ships > m_available_ships:
                                angle = None
                                arrive_tick = None
                                break

                            arrive_tick = turns_to_arrive
                            
                            if new_total_ships == total_ships:
                                break
                            
                            total_ships = new_total_ships

                        extra_ships = total_ships - base_ships
                        
                if angle is not None and arrive_tick is not None:
                    fleet_speed = 1.0 + (MAX_SPEED - 1.0) * (math.log(max(1, total_ships)) / math.log(1000)) ** 1.5

                    collides_sun = sun_collision(m, fleet_speed, angle)
                    if collides_sun:
                        continue
                        
                    moves.append([m.id, angle, total_ships])
                    exhausted_planets_id.add(m.id)
                    fleet_trajectories.append({
                        "mine": m,
                        "target": t,
                        "angle": angle,
                        "total_ships": total_ships,
                        "arrive_tick": arrive_tick
                    })
            
            elif m_available_ships < base_ships and len(lobs.get("mine", [])) > 1 and t.ships >= MIN_SHIPS_TARGET_COOP_ATTACK: # coop attack
                accum = m_available_ships
                attacking_planets = [{"planet": m, "ships": m_available_ships}]
                coop_sent = False
                
                for p, dist, p_available_ships in safe_nearest_planets:
                    if coop_sent:
                        break
                    
                    attacking_planets.append({"planet": p, "ships": p_available_ships})
                    accum += p_available_ships
    
                    if len(attacking_planets) > COOP_PLANET_CAP:
                        break
                    
                    if accum < base_ships:
                        continue
                        
                    if t.id not in moving_planets: # coop static planet
                        if t.owner == -1: # coop static unowned
                            remainder = base_ships
                            planned = []
                            for a_p in attacking_planets:
                                p = a_p["planet"]
                                p_ships = min(a_p["ships"], remainder)
                                
                                if p_ships > 0:
                                    p_ships = min(a_p["ships"], max(p_ships, MIN_SHIPS_MINE_ATTACK))
    
                                if p_ships <= 0:
                                    continue
                                
                                angle = calculate_angle(p, t)
                                dist = math.sqrt((p.x - t.x)**2 + (p.y - t.y)**2)
                                fleet_speed = 1.0 + (MAX_SPEED - 1.0) * (math.log(p_ships) / math.log(1000)) ** 1.5
                                arrive_tick = math.floor(dist / fleet_speed)
                                
                                collides_sun = sun_collision(p, fleet_speed=fleet_speed, angle=angle)
                                if collides_sun:
                                    break
    
                                remainder -= p_ships
                                    
                                planned.append([p, angle, p_ships, arrive_tick])
    
                            if remainder > 0:
                                continue
                                
                            for move in planned:
                                fleet_trajectories.append({
                                    "mine": move[0],
                                    "target": t,
                                    "angle": move[1],
                                    "total_ships": move[2],
                                    "arrive_tick": move[3]
                                })
                                exhausted_planets_id.add(move[0].id)
                                move[0] = move[0].id
                                moves.append(move)
    
                            coop_sent = True
                            break
                                
                        else: # coop static owned
                            required_ships = calculate_req_ships(attacking_planets, t, base_ships)
                            remainder = required_ships
                            
                            if accum < required_ships: 
                                continue
                                
                            planned = []
                            for a_p in attacking_planets:
                                p = a_p["planet"]
                                p_ships = min(a_p["ships"], remainder)
                                
                                if p_ships > 0:
                                    p_ships = min(a_p["ships"], max(p_ships, MIN_SHIPS_MINE_ATTACK))
    
                                if p_ships <= 0:
                                    continue
                                    
                                angle = calculate_angle(p, t)
                                dist = math.sqrt((p.x - t.x)**2 + (p.y - t.y)**2)
                                fleet_speed = 1.0 + (MAX_SPEED - 1.0) * (math.log(p_ships) / math.log(1000)) ** 1.5
                                arrive_tick = math.floor(dist / fleet_speed)
                                
                                collides_sun = sun_collision(p, fleet_speed=fleet_speed, angle=angle)
                                if collides_sun:
                                    continue
    
                                remainder -= p_ships
                                
                                planned.append([p, angle, p_ships, arrive_tick])
    
                            if remainder > 0:
                                continue
                            
                            for move in planned:
                                fleet_trajectories.append({
                                    "mine": move[0],
                                    "target": t,
                                    "angle": move[1],
                                    "total_ships": move[2],
                                    "arrive_tick": move[3]
                                })
                                exhausted_planets_id.add(move[0].id)
                                move[0] = move[0].id
                                moves.append(move)
    
                            coop_sent = True
                            break
                    
                    else: # coop moving planet
                        planet_trajectories = get_planet_trajectories(t, obs.angular_velocity)
                        if t.owner == -1: # coop moving unowned
                            remainder = base_ships
                            planned = []
                            for a_p in attacking_planets:
                                p = a_p["planet"]
                                p_ships = min(a_p["ships"], remainder)
                                
                                if p_ships > 0:
                                    p_ships = min(a_p["ships"], max(p_ships, MIN_SHIPS_MINE_ATTACK))
    
                                if p_ships <= 0:
                                    continue
    
                                angle, arrive_tick = find_angle_to_moving_planet(p, t, p_ships, obs.angular_velocity)
                                
                                if angle is None or arrive_tick is None:
                                    continue
                                    
                                planned.append([p, angle, p_ships, arrive_tick])
                                remainder -= p_ships
    
                            if remainder > 0:
                                continue
    
                            for move in planned:
                                fleet_trajectories.append({
                                    "mine": move[0],
                                    "target": t,
                                    "angle": move[1],
                                    "total_ships": move[2],
                                    "arrive_tick": move[3]
                                })
                                exhausted_planets_id.add(move[0].id)
                                move[0] = move[0].id
                                moves.append(move)
    
                            coop_sent = True
                            break
                    
                        else: # coop moving owned
                            required_ships = calculate_req_ships_moving(attacking_planets, t, base_ships, obs.angular_velocity)
                            remainder = required_ships
                            planned = []
    
                            if accum < required_ships:
                                continue
                            
                            for a_p in attacking_planets:
                                p = a_p["planet"]
                                p_ships = min(a_p["ships"], remainder)
                                
                                if p_ships > 0:
                                    p_ships = min(a_p["ships"], max(p_ships, MIN_SHIPS_MINE_ATTACK))   
                                    
                                if p_ships <= 0:
                                    continue
                                
                                fleet_speed = 1.0 + (MAX_SPEED - 1.0) * (math.log(max(1, p_ships)) / math.log(1000)) ** 1.5
    
                                angle, arrive_tick = find_angle_to_moving_planet(p, t, p_ships, obs.angular_velocity)
    
                                if angle is None or arrive_tick is None:
                                    continue
                                
                                remainder -= p_ships
    
                                planned.append([p, angle, p_ships, arrive_tick])
    
                            if remainder > 0:
                                continue
                            
                            for move in planned:
                                fleet_trajectories.append({
                                    "mine": move[0],
                                    "target": t,
                                    "angle": move[1],
                                    "total_ships": move[2],
                                    "arrive_tick": move[3]
                                })
                                exhausted_planets_id.add(move[0].id)
                                move[0] = move[0].id
                                moves.append(move)
    
                            coop_sent = True
                            break
    
    return moves