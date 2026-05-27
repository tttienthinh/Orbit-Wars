import math
import time

_states = {}

def nearest_planet_sniper(obs):
    pid = obs.get("player", 0) if isinstance(obs, dict) else obs.player
    if pid not in _states:
        _states[pid] = {"step": 0}
    s = _states[pid]

    print(f"Agent step: {s['step']}")
    s["step"] += 1
    return []

agent = nearest_planet_sniper