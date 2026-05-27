import math
import time
from kaggle_environments.envs.orbit_wars.orbit_wars import Planet

class Obj:
    def __init__(self, step=0):
        self.step = step

obj = Obj()

def nearest_planet_sniper(obs, test):
    global obj
    print(f"{obj.step}")
    obj.step += 1

    return []