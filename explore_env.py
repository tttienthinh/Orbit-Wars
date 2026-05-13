import kaggle_environments as ke
import json

with open("11-download_logs/02-augment/episode-76319029.json") as f:
    data = json.load(f)
obs_41 = next(o for o in data if o["step"] == 41)

env = ke.make("orbit_wars", debug=False)
env.reset()

# Print state structure
s = env.state[0]
print("state[0] type:", type(s))
print("state[0] keys:", list(s.keys()) if hasattr(s, 'keys') else dir(s))
obs = s.observation
print("observation type:", type(obs))
print("observation keys:", list(obs.keys()) if hasattr(obs, 'keys') else dir(obs))
print("observation['step']:", obs.get("step", "N/A") if hasattr(obs, 'get') else getattr(obs, 'step', 'N/A'))

# Extra: show all state[0] attributes
print("\n--- state[0] full dump ---")
for k in (s.keys() if hasattr(s, 'keys') else []):
    print(f"  {k}: {type(s[k]).__name__} = {repr(s[k])[:120]}")

print("\n--- obs full dump ---")
for k in (obs.keys() if hasattr(obs, 'keys') else []):
    val = obs[k] if hasattr(obs, '__getitem__') else getattr(obs, k)
    print(f"  {k}: {type(val).__name__} = {repr(val)[:120]}")
