"""
explore_env2.py — Test obs injection approaches into kaggle orbit_wars env.

Findings:
- env.state[0] is a kaggle_environments.utils.Struct (subclass of dict)
- Both dict-style (obs[key] = val) and setattr work for injection
- The env's framework OVERWRITES obs['step'] with len(env.steps) after each step
  (see kaggle_environments/core.py:602)
- BUT the injected 'step' IS used during physics computation (planet rotation angles)
- The game state (planets, fleets) correctly reflects the injected obs
"""
import kaggle_environments as ke
import json
import warnings, logging
warnings.filterwarnings("ignore")
logging.disable(logging.CRITICAL)

with open("11-download_logs/02-augment/episode-76319029.json") as f:
    data = json.load(f)
obs_41 = next(o for o in data if o["step"] == 41)

# Keys present in env observation (from explore_env.py output)
OBS_KEYS = ['remainingOverageTime', 'step', 'planets', 'fleets', 'player',
            'angular_velocity', 'initial_planets', 'next_fleet_id', 'comets', 'comet_planet_ids']

# Try approach A: update via dict-style assignment per key
env = ke.make("orbit_wars", debug=False)
env.reset()
try:
    for key in OBS_KEYS:
        if key in obs_41:
            env.state[0].observation[key] = obs_41[key]
            env.state[1].observation[key] = obs_41[key]
    env.step([[], []])
    new_step = env.state[0].observation["step"]
    n_planets = len(env.state[0].observation["planets"])
    print(f"Approach A WORKS: new step={new_step}, planets={n_planets}")
except Exception as e:
    print(f"Approach A FAILED: {e}")
    import traceback; traceback.print_exc()

# Try approach B: direct attribute assignment
env2 = ke.make("orbit_wars", debug=False)
env2.reset()
try:
    for key in OBS_KEYS:
        if key in obs_41:
            setattr(env2.state[0].observation, key, obs_41[key])
            setattr(env2.state[1].observation, key, obs_41[key])
    env2.step([[], []])
    new_step = env2.state[0].observation.step
    n_planets = len(env2.state[0].observation.planets)
    print(f"Approach B WORKS: new step={new_step}, planets={n_planets}")
except Exception as e:
    print(f"Approach B FAILED: {e}")
    import traceback; traceback.print_exc()

# Demonstrate full workflow: inject obs_41, apply done_action, step N more times
print("\n--- Full workflow demonstration ---")
env3 = ke.make("orbit_wars", debug=False)
env3.reset()
for key in OBS_KEYS:
    if key in obs_41:
        env3.state[0].observation[key] = obs_41[key]
        env3.state[1].observation[key] = obs_41[key]

done_action = obs_41.get("action", [])
print(f"Injected step={obs_41['step']}, planets={len(obs_41['planets'])}, fleets={len(obs_41['fleets'])}")
print(f"Applying done_action: {done_action}")

env3.step([done_action, []])  # Apply done actions for player 0, empty for player 1
print(f"After done_action step: env step={env3.state[0].observation['step']}, "
      f"planets={len(env3.state[0].observation['planets'])}, "
      f"fleets={len(env3.state[0].observation['fleets'])}")

# Step N more times with empty actions, recording planets at each step
planet_records = []
N = 5
for i in range(N):
    env3.step([[], []])
    obs = env3.state[0].observation
    planet_records.append({
        "env_step": obs["step"],
        "planets": [p[:] for p in obs["planets"]],
    })

print(f"\nPlanet[0] ships over next {N} steps (production rate={obs_41['planets'][0][6]}):")
for rec in planet_records:
    print(f"  env_step={rec['env_step']}: planet[0] ships={rec['planets'][0][5]}")

print("\nConclusion: Approach A (dict-style assignment) is the canonical injection pattern.")
print("NOTE: obs['step'] is reset to 1 by the framework after first step, then increments normally.")
print("NOTE: The injected 'step' value IS used for planet rotation physics during the step call.")
