import math
import time
from kaggle_environments.envs.orbit_wars.orbit_wars import Planet

step = 0

def nearest_planet_sniper(obs):
    global step

    time.sleep(2)

    print(f"Agent called step: {step} remainingOverageTime: {obs.get('remainingOverageTime', 0)}")
    moves = []
    step += 1
    return moves