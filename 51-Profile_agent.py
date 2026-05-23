import importlib.util
import time
import random
import math

import kaggle_environments as ke
import plotly.graph_objects as go
from line_profiler import LineProfiler

# ── Load 44-Dataframe_comet.py as a fresh module (resets global state) ────────
spec = importlib.util.spec_from_file_location(
    "agent44",
    "44-Dataframe_comet.py",
)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
m.step = 0
m.num_agents = None
m.player_id = None

# ── Random opponent (same as Test 6) ─────────────────────────────────────────
SEED = 42
N_STEPS = 100
random.seed(SEED)


def random_agent_fn(obs):
    player = obs.player
    my_planets = [p for p in obs.planets if p[1] == player]
    if not my_planets:
        return []
    planet = random.choice(my_planets)
    ships = planet[5] // 2
    if ships < 1:
        return []
    return [[planet[0], random.uniform(0, 2 * math.pi), ships]]


# ── Line profiler: instrument the four key functions ─────────────────────────
lp = LineProfiler()
lp.add_function(m._simulate)
lp.add_function(m.take_action)
lp.add_function(m.IntervalProcessor.create_cumulative_obstacles)
profiled_agent = lp(m.nearest_planet_sniper)

# ── Game loop ─────────────────────────────────────────────────────────────────
env = ke.make("orbit_wars", debug=False)
env.reset(2)

step_numbers: list[int] = []
step_times_ms: list[float] = []

for env_step in range(N_STEPS):
    obs0 = env.state[0].observation
    obs1 = env.state[1].observation

    t0 = time.perf_counter()
    action0 = profiled_agent(obs0)
    dt_ms = (time.perf_counter() - t0) * 1000
    step_numbers.append(env_step)
    step_times_ms.append(dt_ms)
    print(f"Step {env_step:3d}: {dt_ms:7.2f} ms")

    action1 = random_agent_fn(obs1)
    env.step([action0, action1])
    if env.state[0].status != "ACTIVE":
        break

# ── Final result ──────────────────────────────────────────────────────────────
obs0 = env.state[0].observation
p0 = sum(p[5] for p in obs0.planets if p[1] == 0)
p1 = sum(p[5] for p in obs0.planets if p[1] == 1)
n = len(step_numbers)
print(f"\nPlayer 0 (our agent): {p0} ships")
print(f"Player 1 (random):    {p1} ships")
winner = "Our agent wins" if p0 > p1 else "Random wins" if p1 > p0 else "Tie"
print(f"Result after {n} steps: {winner}")
mean_ms = sum(step_times_ms) / n
print(
    f"Step times — min: {min(step_times_ms):.1f} ms  "
    f"max: {max(step_times_ms):.1f} ms  "
    f"mean: {mean_ms:.1f} ms"
)

# ── Plotly: agent call time per step ─────────────────────────────────────────
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=step_numbers,
    y=step_times_ms,
    mode="lines+markers",
    name="Agent call time",
    line=dict(color="royalblue", width=1.5),
    marker=dict(size=5),
))
fig.add_hline(
    y=mean_ms,
    line_dash="dash",
    line_color="orange",
    annotation_text=f"Mean {mean_ms:.1f} ms",
    annotation_position="top right",
)
fig.update_layout(
    title="44-Dataframe_comet agent call time per step (vs random)",
    xaxis_title="Step",
    yaxis_title="Time (ms)",
    template="plotly_white",
    hovermode="x unified",
)
fig.show()

# ── Line profiler results ─────────────────────────────────────────────────────
print("\n" + "=" * 72)
print("LINE PROFILER RESULTS  (time unit: ms)")
print("=" * 72)
lp.print_stats(output_unit=1e-3)
