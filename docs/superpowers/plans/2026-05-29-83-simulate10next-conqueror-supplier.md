# 83-Simulate10Next_Conqueror_Supplier Notebook Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create `83-Simulate10Next_Conqueror_Supplier.ipynb` — a test notebook for `StrategyPipeline` that exposes all intermediate DataFrames (`df_s`, `pa`, `safe`, `action`) per pipeline step for interactive inspection.

**Architecture:** `%run 82-Simulate10Next_Conqueror_Supplier.py` loads the pipeline; `Obs` class and visualization helpers are defined inline. Each of the 5 tests follows a fixed 5-cell pattern (heading → step01 → step02 → step03 → step04+animation), except Test 05 which uses `kaggle_environments`.

**Tech Stack:** Jupyter notebook, `mcp__jupyter__*` tools, `matplotlib`, `pandas`, `kaggle_environments`

---

## File Map

| File | Action | Responsibility |
|------|--------|---------------|
| `83-Simulate10Next_Conqueror_Supplier.ipynb` | **Create** | All test cells |
| `82-Simulate10Next_Conqueror_Supplier.py` | Read-only | Source of `StrategyPipeline`, `interpreter`, `agent`, `GameConfig` |

---

### Task 1: Create notebook with setup cells

**Files:**
- Create: `83-Simulate10Next_Conqueror_Supplier.ipynb`

- [ ] **Step 1: Create the notebook**

```python
mcp__jupyter__notebook_create(path="83-Simulate10Next_Conqueror_Supplier.ipynb")
```

- [ ] **Step 2: Add title markdown cell (cell 0)**

```python
mcp__jupyter__notebook_add_cell(
    path="83-Simulate10Next_Conqueror_Supplier.ipynb",
    cell_type="markdown",
    source="# 83 — Simulate10Next: Conqueror & Supplier Tests\n\nTest notebook for `StrategyPipeline` from `82-Simulate10Next_Conqueror_Supplier.py`.\nEach test exposes `df_s`, `pa`, `safe`, and `action` for interactive inspection.",
    position="end"
)
```

- [ ] **Step 3: Add %run cell (cell 1)**

```python
mcp__jupyter__notebook_add_cell(
    path="83-Simulate10Next_Conqueror_Supplier.ipynb",
    cell_type="code",
    source="%run 82-Simulate10Next_Conqueror_Supplier.py",
    position="end"
)
```

- [ ] **Step 4: Add helpers cell (cell 2)**

```python
mcp__jupyter__notebook_add_cell(
    path="83-Simulate10Next_Conqueror_Supplier.ipynb",
    cell_type="code",
    source="""import copy, math, random
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
import kaggle_environments as ke

_COLORS = {0: 'steelblue', 1: 'tomato', -1: '#888888'}


class Obs:
    def __init__(self, planets, initial_planets=None, fleets=None,
                 next_fleet_id=100, comets=None, comet_planet_ids=None,
                 angular_velocity=0.0):
        self.planets          = [list(p) for p in planets]
        self.initial_planets  = [list(p) for p in (initial_planets if initial_planets is not None else planets)]
        self.fleets           = [list(f) for f in (fleets or [])]
        self.next_fleet_id    = next_fleet_id
        self.comets           = comets or []
        self.comet_planet_ids = comet_planet_ids or []
        self.angular_velocity = angular_velocity


def simulate_with_action(obs, action0, n_steps, current_step=0):
    snapshots = []
    for i, step in enumerate(range(current_step, current_step + n_steps)):
        snapshots.append({
            'step':    step,
            'planets': [p[:] for p in obs.planets],
            'fleets':  [f[:] for f in obs.fleets],
        })
        interpreter(obs, [action0 if i == 0 else [], []], step)
    return snapshots


def make_animation(snapshots, title='', interval=150):
    fig, ax = plt.subplots(figsize=(6, 6))
    fig.patch.set_facecolor('#111122')

    def draw(frame):
        snap = snapshots[frame]
        ax.cla()
        ax.set_xlim(0, 100)
        ax.set_ylim(100, 0)
        ax.set_aspect('equal')
        ax.set_facecolor('#111122')
        ax.tick_params(colors='#aaaaaa')
        for sp in ax.spines.values():
            sp.set_edgecolor('#444444')
        ax.set_title(f"{title}  (step {snap['step']})", color='white', fontsize=11)
        ax.add_patch(plt.Circle((50, 50), 10, color='gold', zorder=2, alpha=0.9))
        for p in snap['planets']:
            pid, owner, x, y, radius, ships, production = p
            c = _COLORS.get(owner, '#888888')
            ax.add_patch(plt.Circle((x, y), radius, color=c, alpha=0.85, zorder=3))
            ax.text(x, y,   str(ships),         ha='center', va='center', color='white', fontsize=7, fontweight='bold', zorder=4)
            ax.text(x, y+2, str(pid),            ha='center', va='center', color='red',   fontsize=7, fontweight='bold', zorder=4)
            ax.text(x, y-2, "+"+str(production), ha='center', va='center', color='white', fontsize=5, fontweight='bold', zorder=4)
        for f in snap['fleets']:
            fid, owner, x, y, angle, from_id, ships = f
            c = _COLORS.get(owner, '#888888')
            ax.plot(x, y, 'D', color=c, markersize=5, zorder=5)
            ax.text(x + 1.5, y + 1.5, str(ships), color=c, fontsize=5, zorder=6)
        return []

    ani = animation.FuncAnimation(fig, draw, frames=len(snapshots), interval=interval)
    plt.close()
    return HTML(ani.to_jshtml())""",
    position="end"
)
```

- [ ] **Step 5: Commit**

```bash
git add "83-Simulate10Next_Conqueror_Supplier.ipynb"
git commit -m "feat: add 83 notebook with setup cells"
```

---

### Task 2: Test 01 — 1 Supplier, 1 Conqueror, 1 Enemy

**Files:**
- Modify: `83-Simulate10Next_Conqueror_Supplier.ipynb`

Planet 1 (x=30) is closer to the enemy (x=70) than planet 0 (x=20). The pipeline should choose planet 1 as Conqueror.

- [ ] **Step 1: Add heading cell**

```python
mcp__jupyter__notebook_add_cell(
    path="83-Simulate10Next_Conqueror_Supplier.ipynb",
    cell_type="markdown",
    source="## Test 01 — 1 Supplier, 1 Conqueror, 1 Enemy\n\nPlanet 1 (x=30) should attack planet 2 (x=70). Planet 0 (x=20) stays as supplier.",
    position="end"
)
```

- [ ] **Step 2: Add step 01 cell**

```python
mcp__jupyter__notebook_add_cell(
    path="83-Simulate10Next_Conqueror_Supplier.ipynb",
    cell_type="code",
    source="""obs01 = Obs(
    planets=[
        [0, 0, 20.0, 20.0, 1 + math.log(3), 50, 3],  # Supplier
        [1, 0, 30.0, 20.0, 1 + math.log(3), 50, 3],  # Conqueror
        [2, 1, 70.0, 20.0, 1.0,              1,  1],  # Enemy
    ],
    angular_velocity=0.0,
)
df_s01, pd01 = StrategyPipeline._01_get_obs_dataframe(obs01, step=0, num_agents=2)
df_s01""",
    position="end"
)
```

- [ ] **Step 3: Add step 02 cell**

```python
mcp__jupyter__notebook_add_cell(
    path="83-Simulate10Next_Conqueror_Supplier.ipynb",
    cell_type="code",
    source="""pa01 = StrategyPipeline._02_get_all_opportunities(df_s01, pd01, player_id=0)
pa01""",
    position="end"
)
```

- [ ] **Step 4: Add step 03 cell**

```python
mcp__jupyter__notebook_add_cell(
    path="83-Simulate10Next_Conqueror_Supplier.ipynb",
    cell_type="code",
    source="""safe01 = StrategyPipeline._03_filter_collision(pa01)
safe01""",
    position="end"
)
```

- [ ] **Step 5: Add step 04 + animation cell**

```python
mcp__jupyter__notebook_add_cell(
    path="83-Simulate10Next_Conqueror_Supplier.ipynb",
    cell_type="code",
    source="""action01 = StrategyPipeline._04_score_and_decide(safe01, player_id=0)
print("Action:", action01)
snaps01 = simulate_with_action(copy.deepcopy(obs01), action01, 20)
make_animation(snaps01, title='Test 01 — 1 Supplier, 1 Conqueror, 1 Enemy', interval=200)""",
    position="end"
)
```

- [ ] **Step 6: Commit**

```bash
git add "83-Simulate10Next_Conqueror_Supplier.ipynb"
git commit -m "feat: add test 01 (1 supplier, 1 conqueror, 1 enemy)"
```

---

### Task 3: Test 02 — 1 Supplier, 2 Conquerors (one more in need)

**Files:**
- Modify: `83-Simulate10Next_Conqueror_Supplier.ipynb`

Two enemy planets: planet 2 (x=70, 1 ship) easy target, planet 4 (x=20,y=70, 100 ships) hard target. Pipeline should pick one conqueror per enemy based on scoring.

- [ ] **Step 1: Add heading cell**

```python
mcp__jupyter__notebook_add_cell(
    path="83-Simulate10Next_Conqueror_Supplier.ipynb",
    cell_type="markdown",
    source="## Test 02 — 1 Supplier, 2 Conquerors (one more in need)\n\nPlanet 1 attacks planet 2 (easy). Planet 3 attacks planet 4 (heavy — 100 ships). Planet 0 stays as supplier.",
    position="end"
)
```

- [ ] **Step 2: Add step 01 cell**

```python
mcp__jupyter__notebook_add_cell(
    path="83-Simulate10Next_Conqueror_Supplier.ipynb",
    cell_type="code",
    source="""obs02 = Obs(
    planets=[
        [0, 0, 20.0, 20.0, 1 + math.log(3), 50,  3],
        [1, 0, 30.0, 20.0, 1 + math.log(3), 50,  3],
        [2, 1, 70.0, 20.0, 1 + math.log(3), 1,   1],
        [3, 0, 20.0, 30.0, 1 + math.log(3), 50,  3],
        [4, 1, 20.0, 70.0, 1 + math.log(3), 100, 1],
    ],
    angular_velocity=0.0,
)
df_s02, pd02 = StrategyPipeline._01_get_obs_dataframe(obs02, step=0, num_agents=2)
df_s02""",
    position="end"
)
```

- [ ] **Step 3: Add step 02 cell**

```python
mcp__jupyter__notebook_add_cell(
    path="83-Simulate10Next_Conqueror_Supplier.ipynb",
    cell_type="code",
    source="""pa02 = StrategyPipeline._02_get_all_opportunities(df_s02, pd02, player_id=0)
pa02""",
    position="end"
)
```

- [ ] **Step 4: Add step 03 cell**

```python
mcp__jupyter__notebook_add_cell(
    path="83-Simulate10Next_Conqueror_Supplier.ipynb",
    cell_type="code",
    source="""safe02 = StrategyPipeline._03_filter_collision(pa02)
safe02""",
    position="end"
)
```

- [ ] **Step 5: Add step 04 + animation cell**

```python
mcp__jupyter__notebook_add_cell(
    path="83-Simulate10Next_Conqueror_Supplier.ipynb",
    cell_type="code",
    source="""action02 = StrategyPipeline._04_score_and_decide(safe02, player_id=0)
print("Action:", action02)
snaps02 = simulate_with_action(copy.deepcopy(obs02), action02, 20)
make_animation(snaps02, title='Test 02 — 1 Supplier, 2 Conquerors (one more in need)', interval=200)""",
    position="end"
)
```

- [ ] **Step 6: Commit**

```bash
git add "83-Simulate10Next_Conqueror_Supplier.ipynb"
git commit -m "feat: add test 02 (2 conquerors, one heavily defended)"
```

---

### Task 4: Test 03 — 4 Suppliers, 2 Conquerors (one more in need)

**Files:**
- Modify: `83-Simulate10Next_Conqueror_Supplier.ipynb`

Same enemy setup as Test 02 but with 4 additional supplier planets. Validates pipeline still selects correct conquerors among more source options.

- [ ] **Step 1: Add heading cell**

```python
mcp__jupyter__notebook_add_cell(
    path="83-Simulate10Next_Conqueror_Supplier.ipynb",
    cell_type="markdown",
    source="## Test 03 — 4 Suppliers, 2 Conquerors (one more in need)\n\nSame enemies as Test 02. Planets 5, 6, 7 added as extra suppliers. Pipeline should still route correctly.",
    position="end"
)
```

- [ ] **Step 2: Add step 01 cell**

```python
mcp__jupyter__notebook_add_cell(
    path="83-Simulate10Next_Conqueror_Supplier.ipynb",
    cell_type="code",
    source="""obs03 = Obs(
    planets=[
        [0, 0, 20.0, 20.0, 1 + math.log(3), 50,  3],
        [1, 0, 30.0, 20.0, 1 + math.log(3), 50,  3],
        [2, 1, 70.0, 20.0, 1 + math.log(3), 1,   1],
        [3, 0, 20.0, 30.0, 1 + math.log(3), 50,  3],
        [4, 1, 20.0, 70.0, 1 + math.log(3), 100, 1],
        [5, 0, 10.0, 10.0, 1 + math.log(3), 50,  3],
        [6, 0, 10.0, 25.0, 1 + math.log(3), 50,  3],
        [7, 0, 25.0, 10.0, 1 + math.log(3), 50,  3],
    ],
    angular_velocity=0.0,
)
df_s03, pd03 = StrategyPipeline._01_get_obs_dataframe(obs03, step=0, num_agents=2)
df_s03""",
    position="end"
)
```

- [ ] **Step 3: Add step 02 cell**

```python
mcp__jupyter__notebook_add_cell(
    path="83-Simulate10Next_Conqueror_Supplier.ipynb",
    cell_type="code",
    source="""pa03 = StrategyPipeline._02_get_all_opportunities(df_s03, pd03, player_id=0)
pa03""",
    position="end"
)
```

- [ ] **Step 4: Add step 03 cell**

```python
mcp__jupyter__notebook_add_cell(
    path="83-Simulate10Next_Conqueror_Supplier.ipynb",
    cell_type="code",
    source="""safe03 = StrategyPipeline._03_filter_collision(pa03)
safe03""",
    position="end"
)
```

- [ ] **Step 5: Add step 04 + animation cell**

```python
mcp__jupyter__notebook_add_cell(
    path="83-Simulate10Next_Conqueror_Supplier.ipynb",
    cell_type="code",
    source="""action03 = StrategyPipeline._04_score_and_decide(safe03, player_id=0)
print("Action:", action03)
snaps03 = simulate_with_action(copy.deepcopy(obs03), action03, 20)
make_animation(snaps03, title='Test 03 — 4 Suppliers, 2 Conquerors (one more in need)', interval=200)""",
    position="end"
)
```

- [ ] **Step 6: Commit**

```bash
git add "83-Simulate10Next_Conqueror_Supplier.ipynb"
git commit -m "feat: add test 03 (4 suppliers, 2 conquerors)"
```

---

### Task 5: Test 04 — 4 Suppliers, 1 Conqueror Orbiting

**Files:**
- Modify: `83-Simulate10Next_Conqueror_Supplier.ipynb`

Planets 4 (ours, x=25,y=25) and 5 (enemy, x=30,y=30) are both within orbit radius (dist from center ≈35 and ≈28, both < 50−radius). `angular_velocity=0.05` makes the orbit visible. The pipeline must track the orbiting enemy's projected position.

- [ ] **Step 1: Add heading cell**

```python
mcp__jupyter__notebook_add_cell(
    path="83-Simulate10Next_Conqueror_Supplier.ipynb",
    cell_type="markdown",
    source="## Test 04 — 4 Suppliers, 1 Conqueror Orbiting\n\nPlanets 4 and 5 orbit (dist_from_center < 50 − radius). `angular_velocity=0.05`. The pipeline must intercept the moving enemy planet.",
    position="end"
)
```

- [ ] **Step 2: Add step 01 cell**

```python
mcp__jupyter__notebook_add_cell(
    path="83-Simulate10Next_Conqueror_Supplier.ipynb",
    cell_type="code",
    source="""obs04 = Obs(
    planets=[
        [0, 0, 20.0, 20.0, 1 + math.log(3), 50, 3],
        [1, 0, 10.0, 10.0, 1 + math.log(3), 50, 3],
        [2, 0, 10.0, 25.0, 1 + math.log(3), 50, 3],
        [3, 0, 25.0, 10.0, 1 + math.log(3), 50, 3],
        [4, 0, 25.0, 25.0, 1.0,              50, 1],  # orbiting (dist≈35 < 49)
        [5, 1, 30.0, 30.0, 1.0,              50, 1],  # orbiting enemy (dist≈28 < 49)
    ],
    angular_velocity=0.05,
)
df_s04, pd04 = StrategyPipeline._01_get_obs_dataframe(obs04, step=0, num_agents=2)
df_s04""",
    position="end"
)
```

- [ ] **Step 3: Add step 02 cell**

```python
mcp__jupyter__notebook_add_cell(
    path="83-Simulate10Next_Conqueror_Supplier.ipynb",
    cell_type="code",
    source="""pa04 = StrategyPipeline._02_get_all_opportunities(df_s04, pd04, player_id=0)
pa04""",
    position="end"
)
```

- [ ] **Step 4: Add step 03 cell**

```python
mcp__jupyter__notebook_add_cell(
    path="83-Simulate10Next_Conqueror_Supplier.ipynb",
    cell_type="code",
    source="""safe04 = StrategyPipeline._03_filter_collision(pa04)
safe04""",
    position="end"
)
```

- [ ] **Step 5: Add step 04 + animation cell**

```python
mcp__jupyter__notebook_add_cell(
    path="83-Simulate10Next_Conqueror_Supplier.ipynb",
    cell_type="code",
    source="""action04 = StrategyPipeline._04_score_and_decide(safe04, player_id=0)
print("Action:", action04)
snaps04 = simulate_with_action(copy.deepcopy(obs04), action04, 20)
make_animation(snaps04, title='Test 04 — 4 Suppliers, 1 Conqueror Orbiting', interval=200)""",
    position="end"
)
```

- [ ] **Step 6: Commit**

```bash
git add "83-Simulate10Next_Conqueror_Supplier.ipynb"
git commit -m "feat: add test 04 (orbiting conqueror)"
```

---

### Task 6: Test 05 — vs Random Agent

**Files:**
- Modify: `83-Simulate10Next_Conqueror_Supplier.ipynb`

Uses `kaggle_environments` for a real game loop. Reset the `agent()` globals (`step`, `num_agents`, `player_id`) before starting since `%run` set them at startup.

- [ ] **Step 1: Add heading cell**

```python
mcp__jupyter__notebook_add_cell(
    path="83-Simulate10Next_Conqueror_Supplier.ipynb",
    cell_type="markdown",
    source="## Test 05 — Our Agent vs Random Agent\n\nFull game via `kaggle_environments`. Player 0 uses `agent()` from `82-...py`, player 1 uses a random policy.",
    position="end"
)
```

- [ ] **Step 2: Add random_agent + game loop cell**

```python
mcp__jupyter__notebook_add_cell(
    path="83-Simulate10Next_Conqueror_Supplier.ipynb",
    cell_type="code",
    source="""def random_agent_fn(obs):
    player = obs.player
    my_planets = [p for p in obs.planets if p[1] == player]
    if not my_planets:
        return []
    planet = random.choice(my_planets)
    ships = planet[5] // 2
    if ships < 1:
        return []
    return [[planet[0], random.uniform(0, 2 * math.pi), ships]]


# Reset agent globals so this cell is re-runnable
import importlib, sys
step = 0
num_agents = None
player_id = None

SEED = 42
N_STEPS = 100
random.seed(SEED)

env = ke.make("orbit_wars", debug=False)
env.reset(2)

snaps05 = []
for env_step in range(N_STEPS):
    obs0 = env.state[0].observation
    obs1 = env.state[1].observation
    snaps05.append({
        'step':    env_step,
        'planets': [list(p) for p in obs0.planets],
        'fleets':  [list(f) for f in obs0.fleets],
    })
    action0 = agent(obs0)
    action1 = random_agent_fn(obs1)
    env.step([action0, action1])
    if env.state[0].status != "ACTIVE":
        break

obs0 = env.state[0].observation
snaps05.append({
    'step':    len(snaps05),
    'planets': [list(p) for p in obs0.planets],
    'fleets':  [list(f) for f in obs0.fleets],
})
p0 = sum(p[5] for p in obs0.planets if p[1] == 0)
p1 = sum(p[5] for p in obs0.planets if p[1] == 1)
winner = "Our agent wins" if p0 > p1 else "Random wins" if p1 > p0 else "Tie"
print(f"After {len(snaps05) - 1} steps: {winner}  (player0={p0}, player1={p1})")

make_animation(snaps05, title='Test 05 — Our Agent vs Random', interval=100)""",
    position="end"
)
```

- [ ] **Step 3: Commit**

```bash
git add "83-Simulate10Next_Conqueror_Supplier.ipynb"
git commit -m "feat: add test 05 (vs random agent via kaggle_environments)"
```

---

## Self-Review

### Spec coverage

| Spec requirement | Task |
|-----------------|------|
| `%run 82-...py` for loading | Task 1 step 3 |
| `Obs` class inline | Task 1 step 4 |
| `simulate_with_action` + `make_animation` inline | Task 1 step 4 |
| 5 cells per test (heading, step01–04, animation) | Tasks 2–5 |
| Test 01 planet config | Task 2 |
| Test 02 planet config | Task 3 |
| Test 03 planet config | Task 4 |
| Test 04 with `angular_velocity=0.05` | Task 5 |
| Test 05 via `kaggle_environments` | Task 6 |
| Intermediate DataFrames (`df_s`, `pa`, `safe`) displayed per step | Tasks 2–5 steps 2–4 |

### No placeholders

All cell `source` strings contain complete, literal code. No TBDs. ✓

### Type consistency

- `Obs` class used identically in all tests ✓
- `StrategyPipeline._01_` returns `(df_s, planet_disp)` — matched in all step 01 cells ✓
- `_02_` takes `(df_s, planet_disp, player_id)` — consistent ✓
- `_03_` takes `pa` — consistent ✓
- `_04_` takes `(safe, player_id)` — consistent ✓
- Variable names per test use numeric suffix (`obs01`, `df_s01`, `pa01`, `safe01`, `action01`, `snaps01`) — no collision ✓
- `copy.deepcopy(obs0N)` used in animation cell to avoid mutating the test obs ✓
