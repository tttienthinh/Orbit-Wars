# Polars Lazy Port Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Polars lazy evaluation to `take_action_polars`, producing `take_action_lazy` in `48-Polars_lazy.ipynb` and `49-Polars_lazy.py`, verified identical to the pandas reference and submitted to Kaggle.

**Architecture:** `take_action_lazy` splits the pipeline into 4 lazy chains separated by unavoidable `.collect()` materializations. The key win is Chain 2 (cross-join + filter) where Polars' optimizer applies predicate pushdown, reducing intermediate rows before the collision/crossing_sun filters. `create_cumulative_obstacles` and the comet branch remain eager because they require Python conditionals over materialized data.

**Tech Stack:** Python 3, Polars 1.35.2 (lazy API), pandas, kaggle-environments, mcp__jupyter__* tools for notebook authoring.

---

## File Structure

- **Create:** `48-Polars_lazy.ipynb` — notebook with `take_action_lazy`, 8 equality tests, 3-way timing, Test 6 full game
- **Create:** `49-Polars_lazy.py` — Kaggle submission (`46-Polars_comet.py` + `take_action_lazy` + agent uses it)

---

## Lazy chain structure reference (for implementer)

The 4 chains and their forced collection points:

| Chain | Start | End `.collect()` | Reason forced |
|---|---|---|---|
| 1 | `pl.from_pandas(df).sort("step").lazy()` | after `mine_across_sim` | `.is_empty()` Python guard |
| 2 | `mine_across_sim.lazy()` cross-joined with `df_lf` | after `possible_attacks` | `create_cumulative_obstacles` needs `.to_dicts()` |
| 3 | `possible_attacks.lazy()` joined with `df_obstacles.lazy()` | after `attacks_with_angle` | comet branch needs `.is_empty()` and `abs().max()` |
| 4 | `attacks_with_angle.lazy()` (top-5 + score merged) | after `attacks` | extract final moves |

---

## Task 1: Create `48-Polars_lazy.ipynb` skeleton (cells 0–4)

**Files:**
- Create: `48-Polars_lazy.ipynb`

- [ ] **Step 1: Create notebook and add cell 0 (imports)**

Use `mcp__jupyter__notebook_create` to create `48-Polars_lazy.ipynb`, then add the imports cell:

```python
import math, copy
import numpy as np
import pandas as pd
import polars as pl
```

- [ ] **Step 2: Copy cells 1–4 from `47-Polars_comet.ipynb`**

Read cells 1, 2, 3 from `47-Polars_comet.ipynb` (Obs class; constants+simulate; IntervalProcessor+take_action) and add them as cells 1, 2, 3 in the new notebook.

- [ ] **Step 3: Copy IntervalProcessorPolars + take_action_polars (cells 5 and 8 from 47)**

Read cell 5 (`IntervalProcessorPolars` class) and cell 8 (`take_action_polars`) from `47-Polars_comet.ipynb`. Add them as two separate cells (cells 4 and 5) in `48-Polars_lazy.ipynb`. These are the reference implementations used in timing comparisons.

- [ ] **Step 4: Commit skeleton**

```bash
git add "48-Polars_lazy.ipynb"
git commit -m "feat: add 48-Polars_lazy.ipynb skeleton"
```

---

## Task 2: Implement `take_action_lazy`

**Files:**
- Modify: `48-Polars_lazy.ipynb`

- [ ] **Step 1: Add markdown header cell**

Add a markdown cell: `## take_action_lazy — Polars lazy chains`

- [ ] **Step 2: Add the `take_action_lazy` function cell**

Add the following code cell (this is the complete implementation):

```python
def take_action_lazy(df: pd.DataFrame, player_id: int,
                     nb_steps_sim: int = NB_STEPS_SIM,
                     return_df: bool = False):
    # Keep a lazy handle on the input; df_lf is reused in Chain 2
    df_lf = pl.from_pandas(df).sort("step").lazy()

    # ── Chain 1: source planets (collect for is_empty guard) ─────────────────
    mine_across_sim = (
        df_lf
        .with_columns(
            pl.when(pl.col("owner") == player_id).then(1).otherwise(0).alias("is_mine")
        )
        .group_by("id", maintain_order=True)
        .agg(
            pl.first("step").alias("step_src"),
            pl.first("x").alias("x_src"),
            pl.first("y").alias("y_src"),
            pl.first("radius").alias("radius_src"),
            pl.min("ships").alias("ships_min"),
            pl.first("production").alias("production_src"),
            pl.first("nature").alias("nature_src"),
            pl.first("owner").alias("owner_src"),
            pl.len().alias("row_count"),
            pl.sum("is_mine").alias("is_mine"),
        )
        .filter(
            (pl.col("row_count") == pl.col("is_mine")) &
            (pl.col("owner_src") == player_id)
        )
        .rename({"id": "id_src"})
        .collect()
    )

    if mine_across_sim.is_empty():
        return ([], pl.DataFrame()) if return_df else []

    # ── Chain 2: expand → cross-join → filter → compute attacks ──────────────
    # Polars lazy optimizer applies predicate pushdown on the cross-join,
    # reducing intermediate rows before collision/crossing_sun filters.
    dx_vw = pl.col("x") - pl.col("x_src")
    dy_vw = pl.col("y") - pl.col("y_src")
    l2    = dx_vw.pow(2) + dy_vw.pow(2)
    dot   = (CENTER - pl.col("x_src")) * dx_vw + (CENTER - pl.col("y_src")) * dy_vw
    t     = (dot / pl.when(l2 == 0).then(pl.lit(1.0)).otherwise(l2)).clip(0.0, 1.0)
    dist_sun_proj   = ((CENTER - (pl.col("x_src") + t * dx_vw)).pow(2) +
                       (CENTER - (pl.col("y_src") + t * dy_vw)).pow(2)).sqrt()
    dist_sun_direct = ((CENTER - pl.col("x_src")).pow(2) +
                       (CENTER - pl.col("y_src")).pow(2)).sqrt()
    dist_to_sun     = pl.when(l2 == 0).then(dist_sun_direct).otherwise(dist_sun_proj)
    crossing_sun_expr = dist_to_sun < (SUN_RADIUS + PLANET_MARGIN)

    dist_tgt_src_expr = ((pl.col("x") - pl.col("x_src")).pow(2) +
                         (pl.col("y") - pl.col("y_src")).pow(2)).sqrt()
    step_diff_expr    = pl.col("step") - pl.col("step_src")
    fleet_speed_expr  = (
        1.0 + (MAX_SPEED - 1.0) *
        (pl.col("ships_sent").cast(pl.Float64).log(base=math.e) / math.log(1000.0)).pow(1.5)
    )
    dist_min_expr = step_diff_expr * fleet_speed_expr + PLANET_MARGIN + pl.col("radius_src")
    dist_max_expr = (step_diff_expr + 1) * fleet_speed_expr + PLANET_MARGIN + pl.col("radius_src")
    collision_expr = (
        ((dist_tgt_src_expr - pl.col("radius") < dist_min_expr) &
         (dist_min_expr < dist_tgt_src_expr + pl.col("radius"))) |
        ((dist_tgt_src_expr - pl.col("radius") < dist_max_expr) &
         (dist_max_expr < dist_tgt_src_expr + pl.col("radius")))
    )

    possible_attacks = (
        mine_across_sim.lazy()
        .with_columns(
            pl.int_ranges(
                1,
                pl.col("ships_min") + pl.col("production_src") * NB_STEPS_SIM + 1,
                dtype=pl.Int64,
            ).alias("ships_sent")
        )
        .explode("ships_sent")
        .join(df_lf, how="cross")
        .filter(
            (pl.col("step") > pl.col("step_src")) &
            (pl.col("id") != pl.col("id_src"))
        )
        .with_columns([
            dist_tgt_src_expr.alias("dist_tgt_src"),
            step_diff_expr.alias("step_diff"),
            fleet_speed_expr.alias("fleet_speed"),
            dist_min_expr.alias("dist_fleet_src_min"),
            dist_max_expr.alias("dist_fleet_src_max"),
            collision_expr.alias("collision"),
        ])
        .filter(pl.col("collision"))
        .with_columns(crossing_sun_expr.alias("crossing_sun"))
        .filter(~pl.col("crossing_sun"))
        .with_columns(
            pl.arctan2(pl.col("y") - pl.col("y_src"), pl.col("x") - pl.col("x_src")).alias("angle")
        )
        .with_columns(
            pl.max_horizontal(
                ((pl.col("dist_tgt_src").pow(2) + pl.col("dist_fleet_src_min").pow(2) -
                  pl.col("radius").pow(2)) /
                 (2 * pl.col("dist_tgt_src") * pl.col("dist_fleet_src_min"))).clip(-1.0, 1.0).arccos(),
                ((pl.col("dist_tgt_src").pow(2) + pl.col("dist_fleet_src_max").pow(2) -
                  pl.col("radius").pow(2)) /
                 (2 * pl.col("dist_tgt_src") * pl.col("dist_fleet_src_max"))).clip(-1.0, 1.0).arccos(),
            ).alias("radius_angle")
        )
        .with_columns([
            ((pl.col("angle") - pl.col("radius_angle")) % (2 * math.pi)).alias("angle_min"),
            ((pl.col("angle") + pl.col("radius_angle")) % (2 * math.pi)).alias("angle_max"),
        ])
        .sort("step")
        .collect()
    )

    if possible_attacks.is_empty():
        return ([], possible_attacks) if return_df else []

    # ── Python: cumulative obstacle intervals (unavoidably eager) ─────────────
    df_obstacles = IntervalProcessorPolars.create_cumulative_obstacles(possible_attacks)

    # ── Chain 3: free angles (collect for comet branch conditionals) ──────────
    attacks_with_angle = (
        possible_attacks.lazy()
        .join(df_obstacles.lazy(), on=["id_src", "step", "ships_sent"], how="left")
        .with_columns(
            pl.struct(["angle_min", "angle_max", "obstacle_list"])
            .map_elements(
                IntervalProcessorPolars.compute_free_angles,
                return_dtype=pl.List(pl.List(pl.Float64)),
            )
            .alias("angle_list")
        )
        .filter(pl.col("angle_list").list.len() > 0)
        .collect()
    )

    # ── Comet branch (needs Python conditionals over materialized data) ────────
    awa_comets = attacks_with_angle.filter(pl.col("nature_src") == "comet")
    moves = []
    if not awa_comets.is_empty():
        x_off = (awa_comets["x_src"] - CENTER).abs().max()
        y_off = (awa_comets["y_src"] - CENTER).abs().max()
        if max(x_off, y_off) > 45:
            comet_rows = (
                awa_comets
                .filter(pl.col("ships_sent") <= pl.col("ships_min"))
                .sort(["ships_sent", "step"], descending=[True, False])
                .group_by("id_src", maintain_order=True)
                .first()
                .select(["id_src", "angle", "ships_sent"])
                .rows()
            )
            moves += [list(r) for r in comet_rows]
            avoid = awa_comets["id_src"].unique().to_list()
            attacks_with_angle = attacks_with_angle.filter(~pl.col("id_src").is_in(avoid))

    # ── Chain 4: top-5 + score + final angle (single lazy chain) ─────────────
    attacks = (
        attacks_with_angle.lazy()
        .sort(["step", "ships_sent"])
        .group_by(["id_src", "id"], maintain_order=True)
        .first()
        .sort(["step", "ships_sent"])
        .group_by("id_src", maintain_order=True)
        .head(5)
        .select(["id_src", "id"])
        .join(attacks_with_angle.lazy(), on=["id_src", "id"], how="left")
        .filter(pl.col("owner") != player_id)
        .with_columns(
            pl.when(pl.col("owner") == -1)
            .then(pl.col("ships"))
            .otherwise(pl.col("ships") + pl.col("production"))
            .alias("ships_needed")
        )
        .filter(
            (pl.col("ships_needed") + 1 <= pl.col("ships_sent")) &
            (pl.col("ships_sent") <= pl.col("ships_needed") + pl.col("production_src") + 1)
        )
        .sort(["step", "ships_sent"])
        .group_by(["id_src", "id"], maintain_order=True)
        .first()
        .with_columns(
            (pl.col("ships_needed") / pl.col("production_src")).alias("time_cost")
        )
        .with_columns(
            pl.col("time_cost").sum().over("id_src").alias("total_time_cost")
        )
        .with_columns(
            ((pl.col("total_time_cost") - pl.col("time_cost") - pl.col("step_diff")) *
             pl.col("production")).alias("score")
        )
        .sort("score", descending=True)
        .group_by("id_src", maintain_order=True)
        .first()
        .filter(pl.col("ships_sent") <= pl.col("ships_min"))
        .with_columns(
            pl.col("angle_list").map_batches(
                IntervalProcessorPolars.interval_to_final_angle,
                return_dtype=pl.Float64,
            ).alias("final_angle")
        )
        .collect()
    )

    for row in attacks.rows(named=True):
        print(f"From {row['id_src']}, To {row['id']} at step {row['step']} "
              f"with {row['ships_sent']} ships (target has min {row['ships_min']})")

    moves += [list(r) for r in attacks.select(["id_src", "final_angle", "ships_sent"]).rows()]
    return (moves, possible_attacks) if return_df else moves
```

- [ ] **Step 3: Commit**

```bash
git add "48-Polars_lazy.ipynb"
git commit -m "feat: implement take_action_lazy with 4 lazy chains"
```

---

## Task 3: Smoke test + `assert_moves_equal` helper

**Files:**
- Modify: `48-Polars_lazy.ipynb`

- [ ] **Step 1: Add smoke test cell**

```python
obs_smoke = Obs(
    planets=[[0, 0, 10.0, 10.0, 5.0, 6, 3], [1, -1, 30.0, 10.0, 5.0, 5, 2]],
    angular_velocity=0.0,
)
df_smoke = _simulate(obs_smoke, global_step=0, num_agents=2, n_steps=10)
result = take_action_lazy(df_smoke, player_id=0)
print("Smoke test passed, result:", result)
assert result == [[0, result[0][1], 6]], f"Unexpected result: {result}"
```

Expected output: `Smoke test passed, result: [[0, <some_angle>, 6]]`

- [ ] **Step 2: Run the smoke test cell to verify it passes**

Execute via `mcp__jupyter__*`. Expected: no exception, prints a move list with one entry `[0, <angle>, 6]`.

- [ ] **Step 3: Add `assert_moves_equal` helper cell**

```python
def assert_moves_equal(moves_pd, moves_pl, label=""):
    """Assert two move lists match: id_src and ships_sent exact, angle within 1e-6."""
    key = lambda m: (int(m[0]), int(m[2]))
    pd_s = sorted(moves_pd, key=key)
    pl_s = sorted(moves_pl, key=key)
    assert len(pd_s) == len(pl_s), f"{label}: length {len(pd_s)} != {len(pl_s)}"
    for i, (a, b) in enumerate(zip(pd_s, pl_s)):
        assert int(a[0]) == int(b[0]), f"{label}[{i}]: id_src {a[0]} != {b[0]}"
        assert int(a[2]) == int(b[2]), f"{label}[{i}]: ships_sent {a[2]} != {b[2]}"
        assert abs(float(a[1]) - float(b[1])) < 1e-6, f"{label}[{i}]: angle {a[1]} != {b[1]}"
    print(f"{label}: ✓  moves={pl_s}")
```

- [ ] **Step 4: Commit**

```bash
git add "48-Polars_lazy.ipynb"
git commit -m "test: add smoke test and assert_moves_equal helper"
```

---

## Task 4: Equality tests 1–5, 7, 8

**Files:**
- Modify: `48-Polars_lazy.ipynb`

Each test compares `take_action(df, ...)` (pandas ground truth) against `take_action_lazy(df, ...)`. All 7 cells must be executed and show `✓`.

- [ ] **Step 1: Add and execute Test 1**

```python
## Test 1 — Planet production (no targets → [])
obs1 = Obs(planets=[[0, 0, 10.0, 10.0, 5.0, 1, 3]], angular_velocity=0.0)
df1 = _simulate(obs1, global_step=0, num_agents=2, n_steps=NB_STEPS_SIM)
assert_moves_equal(
    take_action(df1, player_id=0, nb_steps_sim=NB_STEPS_SIM),
    take_action_lazy(df1, player_id=0, nb_steps_sim=NB_STEPS_SIM),
    "Test 1"
)
```
Expected: `Test 1: ✓  moves=[]`

- [ ] **Step 2: Add and execute Test 2**

```python
## Test 2 — Attack neutral (6 vs 5 → should attack)
obs2 = Obs(
    planets=[[0, 0, 10.0, 10.0, 5.0, 6, 3], [1, -1, 30.0, 10.0, 5.0, 5, 2]],
    angular_velocity=0.0,
)
df2 = _simulate(obs2, global_step=0, num_agents=2, n_steps=NB_STEPS_SIM)
assert_moves_equal(
    take_action(df2, player_id=0, nb_steps_sim=NB_STEPS_SIM),
    take_action_lazy(df2, player_id=0, nb_steps_sim=NB_STEPS_SIM),
    "Test 2"
)
```
Expected: `Test 2: ✓  moves=[[0, <angle>, 6]]`

- [ ] **Step 3: Add and execute Test 3**

```python
## Test 3 — Equal ships (5 vs 5 → do nothing)
obs3 = Obs(
    planets=[[0, 0, 10.0, 10.0, 5.0, 5, 3], [1, -1, 30.0, 10.0, 5.0, 5, 2]],
    angular_velocity=0.0,
)
df3 = _simulate(obs3, global_step=0, num_agents=2, n_steps=5)
assert_moves_equal(
    take_action(df3, player_id=0, nb_steps_sim=5),
    take_action_lazy(df3, player_id=0, nb_steps_sim=5),
    "Test 3"
)
```
Expected: `Test 3: ✓  moves=[]`

- [ ] **Step 4: Add and execute Test 4**

```python
## Test 4 — Enemy fleet inbound (do nothing)
obs4 = Obs(
    planets=[[0, 0, 10.0, 10.0, 5.0, 6, 3], [1, -1, 30.0, 10.0, 5.0, 5, 2]],
    fleets=[[0, 1, 10.0, 30.0, 3 * math.pi / 2, 0, 24]],
    next_fleet_id=1,
    angular_velocity=0.0,
)
df4 = _simulate(obs4, global_step=0, num_agents=2, n_steps=10)
assert_moves_equal(
    take_action(df4, player_id=0, nb_steps_sim=10),
    take_action_lazy(df4, player_id=0, nb_steps_sim=10),
    "Test 4"
)
```
Expected: `Test 4: ✓`

- [ ] **Step 5: Add and execute Test 5**

```python
## Test 5 — Attack enemy (50 vs 5 → attack)
obs5 = Obs(
    planets=[[0, 0, 10.0, 10.0, 5.0, 50, 3], [1, 1, 30.0, 10.0, 5.0, 5, 2]],
    angular_velocity=0.0,
)
df5 = _simulate(obs5, global_step=0, num_agents=2, n_steps=5)
assert_moves_equal(
    take_action(df5, player_id=0, nb_steps_sim=5),
    take_action_lazy(df5, player_id=0, nb_steps_sim=5),
    "Test 5"
)
```
Expected: `Test 5: ✓`

- [ ] **Step 6: Add and execute Test 7**

```python
## Test 7 — Target planet behind (orbital intercept)
obs7 = Obs(
    planets=[
        [0, 0, 10.0, 10.0, 5.0, 10, 3],
        [1, -1, 30.0, 10.0, 5.0, 5, 1],
        [2, -1, 50.0, 15.0, 10.0, 1, 10],
    ],
    angular_velocity=0.0,
)
df7 = _simulate(copy.deepcopy(obs7), global_step=0, num_agents=2, n_steps=NB_STEPS_SIM)
assert_moves_equal(
    take_action(df7, player_id=0, nb_steps_sim=NB_STEPS_SIM, return_df=False),
    take_action_lazy(df7, player_id=0, nb_steps_sim=NB_STEPS_SIM),
    "Test 7"
)
```
Expected: `Test 7: ✓`

- [ ] **Step 7: Add and execute Test 8**

```python
## Test 8 — Moving planet (angular_velocity = π/20)
n_steps = 50
obs8 = Obs(
    planets=[[0, 0, 10.0, 10.0, 5.0, 50, 3], [1, -1, 30.0, 5.0, 5.0, 5, 2],
             [2, -1, 30.0, 50.0, 5.0, 5, 2]],
    angular_velocity=math.pi / 20,
)
df8 = _simulate(obs8, global_step=0, num_agents=2, n_steps=n_steps)
assert_moves_equal(
    take_action(df8, player_id=0, nb_steps_sim=n_steps),
    take_action_lazy(df8, player_id=0, nb_steps_sim=n_steps),
    "Test 8"
)
```
Expected: `Test 8: ✓`

- [ ] **Step 8: Commit**

```bash
git add "48-Polars_lazy.ipynb"
git commit -m "test: add equality tests 1-5, 7, 8 for take_action_lazy"
```

---

## Task 5: 3-way timing comparison

**Files:**
- Modify: `48-Polars_lazy.ipynb`

- [ ] **Step 1: Add timing cell for Test 2 scenario (small case)**

```python
## Timing — Test 2 scenario (10-step sim, 1 source, 1 target)
import timeit

obs_t = Obs(
    planets=[[0, 0, 10.0, 10.0, 5.0, 6, 3], [1, -1, 30.0, 10.0, 5.0, 5, 2]],
    angular_velocity=0.0,
)
df_t = _simulate(obs_t, global_step=0, num_agents=2, n_steps=NB_STEPS_SIM)

t_pd = timeit.timeit(lambda: take_action(df_t, player_id=0), number=20) / 20
t_pl = timeit.timeit(lambda: take_action_polars(df_t, player_id=0), number=20) / 20
t_lz = timeit.timeit(lambda: take_action_lazy(df_t, player_id=0), number=20) / 20
print(f"pandas eager:  {t_pd*1000:.2f} ms  (1.0×)")
print(f"polars eager:  {t_pl*1000:.2f} ms  ({t_pd/t_pl:.1f}×)")
print(f"polars lazy:   {t_lz*1000:.2f} ms  ({t_pd/t_lz:.1f}×)")
```

- [ ] **Step 2: Add timing cell for Test 8 scenario (larger case)**

```python
## Timing — Test 8 scenario (50-step sim, 1 source, 2 targets)
obs_t8 = Obs(
    planets=[[0, 0, 10.0, 10.0, 5.0, 50, 3], [1, -1, 30.0, 5.0, 5.0, 5, 2],
             [2, -1, 30.0, 50.0, 5.0, 5, 2]],
    angular_velocity=math.pi / 20,
)
df_t8 = _simulate(obs_t8, global_step=0, num_agents=2, n_steps=50)

t_pd8 = timeit.timeit(lambda: take_action(df_t8, player_id=0, nb_steps_sim=50), number=10) / 10
t_pl8 = timeit.timeit(lambda: take_action_polars(df_t8, player_id=0, nb_steps_sim=50), number=10) / 10
t_lz8 = timeit.timeit(lambda: take_action_lazy(df_t8, player_id=0, nb_steps_sim=50), number=10) / 10
print(f"pandas eager:  {t_pd8*1000:.2f} ms  (1.0×)")
print(f"polars eager:  {t_pl8*1000:.2f} ms  ({t_pd8/t_pl8:.1f}×)")
print(f"polars lazy:   {t_lz8*1000:.2f} ms  ({t_pd8/t_lz8:.1f}×)")
```

- [ ] **Step 3: Execute both timing cells and record results**

Run both cells. No assertion needed — the output is informational. Confirm they complete without error.

- [ ] **Step 4: Commit**

```bash
git add "48-Polars_lazy.ipynb"
git commit -m "test: add 3-way timing comparison (pandas / polars eager / polars lazy)"
```

---

## Task 6: Test 6 — Full game equality

**Files:**
- Modify: `48-Polars_lazy.ipynb`

Run this test last — it takes ~1 minute.

- [ ] **Step 1: Add Test 6 cell**

```python
## Test 6 — Full game: pandas take_action vs take_action_lazy (100 steps)
import random
import kaggle_environments as ke

SEED = 42
N_STEPS = 100

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

random.seed(SEED)
env = ke.make("orbit_wars", debug=False)
env.reset(2)

for env_step in range(N_STEPS):
    obs0 = env.state[0].observation
    obs1 = env.state[1].observation

    df = _simulate(copy.deepcopy(obs0), global_step=env_step, num_agents=2, n_steps=NB_STEPS_SIM)

    moves_pd = take_action(df, player_id=0, nb_steps_sim=NB_STEPS_SIM)
    moves_lz = take_action_lazy(df, player_id=0, nb_steps_sim=NB_STEPS_SIM)
    assert_moves_equal(moves_pd, moves_lz, f"step {env_step}")

    rng_action = random_agent_fn(obs1)
    env.step([moves_pd, rng_action])

    if env.state[0].status != "ACTIVE":
        break

print(f"Test 6 passed: {env_step + 1} steps, all moves identical ✓")
```

**Important:** `import kaggle_environments as ke` must come **before** `import polars as pl` in the kernel to avoid DLL conflicts on Windows. Since imports are in cell 0 (`import polars as pl`) and cell 1+ don't re-import, this test cell must import ke before the kernel has loaded polars. The safest approach: add `import kaggle_environments as ke` at the **top** of cell 0 (before `import polars as pl`), then re-run all cells. If a DLL conflict arises anyway, run the test in a separate script: `python test6_lazy.py` and paste output into a markdown cell.

- [ ] **Step 2: Execute Test 6 cell**

Execute and wait for completion. Expected output:
```
step 0: ✓  moves=...
step 1: ✓  moves=...
...
Test 6 passed: 100 steps, all moves identical ✓
```

If a DLL conflict occurs, create `test6_lazy.py` in the project root:

```python
import kaggle_environments as ke
import math, copy, random
import numpy as np
import pandas as pd
import polars as pl

# --- paste full contents of cells 1-6 of 48-Polars_lazy.ipynb here ---
# (Obs, constants+simulate, IntervalProcessor+take_action,
#  IntervalProcessorPolars+take_action_polars, take_action_lazy)

SEED = 42
N_STEPS = 100

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

def assert_moves_equal(moves_pd, moves_pl, label=""):
    key = lambda m: (int(m[0]), int(m[2]))
    pd_s = sorted(moves_pd, key=key)
    pl_s = sorted(moves_pl, key=key)
    assert len(pd_s) == len(pl_s), f"{label}: length {len(pd_s)} != {len(pl_s)}"
    for i, (a, b) in enumerate(zip(pd_s, pl_s)):
        assert int(a[0]) == int(b[0])
        assert int(a[2]) == int(b[2])
        assert abs(float(a[1]) - float(b[1])) < 1e-6
    print(f"{label}: ✓")

random.seed(SEED)
env = ke.make("orbit_wars", debug=False)
env.reset(2)

for env_step in range(N_STEPS):
    obs0 = env.state[0].observation
    obs1 = env.state[1].observation
    df = _simulate(copy.deepcopy(obs0), global_step=env_step, num_agents=2, n_steps=NB_STEPS_SIM)
    moves_pd = take_action(df, player_id=0, nb_steps_sim=NB_STEPS_SIM)
    moves_lz = take_action_lazy(df, player_id=0, nb_steps_sim=NB_STEPS_SIM)
    assert_moves_equal(moves_pd, moves_lz, f"step {env_step}")
    rng_action = random_agent_fn(obs1)
    env.step([moves_pd, rng_action])
    if env.state[0].status != "ACTIVE":
        break

print(f"Test 6 passed: {env_step + 1} steps, all moves identical ✓")
```

Run: `python test6_lazy.py`

- [ ] **Step 3: Commit**

```bash
git add "48-Polars_lazy.ipynb"
git commit -m "test: add Test 6 full game equality for take_action_lazy"
```

---

## Task 7: Create `49-Polars_lazy.py` and submit

**Files:**
- Create: `49-Polars_lazy.py`

- [ ] **Step 1: Copy `46-Polars_comet.py` to `49-Polars_lazy.py`**

```bash
cp "46-Polars_comet.py" "49-Polars_lazy.py"
```

- [ ] **Step 2: Insert `take_action_lazy` after `take_action_polars`**

In `49-Polars_lazy.py`, find the line `# ── Agent ─────` and insert the full `take_action_lazy` function (identical to the notebook cell from Task 2 Step 2) immediately before that line.

- [ ] **Step 3: Change agent body to use `take_action_lazy`**

Find in `49-Polars_lazy.py`:
```python
    moves = take_action_polars(df, player_id=player_id, nb_steps_sim=NB_STEPS_SIM)
```
Replace with:
```python
    moves = take_action_lazy(df, player_id=player_id, nb_steps_sim=NB_STEPS_SIM)
```

- [ ] **Step 4: Verify smoke test**

```bash
python -c "
import sys
sys.path.insert(0, '.')
exec(open('49-Polars_lazy.py').read())
import math, copy
obs = Obs(planets=[[0,0,10.0,10.0,5.0,6,3],[1,-1,30.0,10.0,5.0,5,2]], angular_velocity=0.0)
df = _simulate(obs, 0, 2, n_steps=10)
result = take_action_lazy(df, player_id=0)
print('Agent output:', result)
assert len(result) == 1
print('OK')
"
```
Expected: `Agent output: [[0, <angle>, 6]]` followed by `OK`.

- [ ] **Step 5: Commit**

```bash
git add "49-Polars_lazy.py"
git commit -m "feat: add 49-Polars_lazy.py submission with take_action_lazy agent"
```

- [ ] **Step 6: Submit to Kaggle**

Use the `/sub` skill with argument `49-Polars_lazy.py`:

```bash
kaggle competitions submit orbit-wars -f "49-Polars_lazy.py" -m "49-Polars_lazy.py"
```

Expected: `Successfully submitted to Orbit Wars`
