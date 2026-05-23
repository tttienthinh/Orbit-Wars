# Polars Port: IntervalProcessorPolars + take_action_polars Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port `IntervalProcessor` and `take_action` from `44-Dataframe_comet.py` to Polars equivalents (`IntervalProcessorPolars`, `take_action_polars`), verify identical output against all test cases from `43-Dataframe_comet.ipynb`, and ship as `46-Polars_comet.py`.

**Architecture:** `take_action_polars` accepts the same pandas DataFrame as `take_action` (from `_simulate`), converts it to Polars immediately, and runs the full pipeline in Polars. The key optimization is replacing the per-row `apply(axis=1)` for `crossing_sun` with a fully vectorized Polars expression. `IntervalProcessorPolars` keeps the pure-Python list helpers unchanged and ports only the DataFrame I/O methods. Both `IntervalProcessor`/`take_action` and their Polars counterparts coexist in the notebook for side-by-side comparison.

**Tech Stack:** Python 3, polars 1.35.2, pandas, numpy, kaggle-environments (Test 6 only)

---

## File Structure

| File | Action | Responsibility |
|---|---|---|
| `45-Polars_comet.ipynb` | **Create** | Development notebook: both implementations, equality tests, timing |
| `46-Polars_comet.py` | **Create** | Kaggle submission: full agent using `take_action_polars` |
| `44-Dataframe_comet.py` | **No change** | Reference implementation |

---

### Task 1: Create `45-Polars_comet.ipynb` skeleton

**Files:**
- Create: `45-Polars_comet.ipynb`

Use `mcp__jupyter__notebook_create` then `mcp__jupyter__notebook_add_cell` (type `"code"` or `"markdown"`) for each cell below.

- [ ] **Step 1: Create the notebook**

```
mcp__jupyter__notebook_create path="45-Polars_comet.ipynb"
```

- [ ] **Step 2: Add Cell 0 — imports**

```python
import math, copy
import numpy as np
import pandas as pd
import polars as pl
```

- [ ] **Step 3: Add Cell 1 — Obs helper class (needed for test setups)**

```python
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
```

- [ ] **Step 4: Add Cell 2 — constants + interpreter + `_simulate` (copy verbatim from `44-Dataframe_comet.py` lines 5–290)**

The cell must include: `CENTER`, `SUN_RADIUS`, `ROTATION_RADIUS_LIMIT`, `MAX_SPEED`, `NB_STEPS_SIM = 10`, `PLANET_MARGIN`, `BOARD_SIZE`, `MAX_NB_STEP`, `distance`, `point_to_segment_distance`, `interpreter`, `_fleet_speed`, `_simulate`.

- [ ] **Step 5: Add Cell 3 — reference `IntervalProcessor` + `take_action` (copy verbatim from `44-Dataframe_comet.py` lines 325–705)**

This is the pandas reference we compare against.

- [ ] **Step 6: Add markdown cell — section header**

```markdown
## IntervalProcessorPolars + take_action_polars
```

- [ ] **Step 7: Commit notebook skeleton**

```bash
git add "45-Polars_comet.ipynb"
git commit -m "feat: add 45-Polars_comet.ipynb skeleton with reference implementation"
```

---

### Task 2: Implement `IntervalProcessorPolars`

**Files:**
- Modify: `45-Polars_comet.ipynb` (add implementation cell)

- [ ] **Step 1: Add failing assertion cell to the notebook (run it — it will fail with NameError)**

```python
# Sanity check — will fail until IntervalProcessorPolars is defined
assert IntervalProcessorPolars.merge_intervals([(0.5, 1.0), (0.2, 0.6)]) == [(0.2, 1.0)]
print("merge_intervals OK")
```

Run the cell. Expected: `NameError: name 'IntervalProcessorPolars' is not defined`

- [ ] **Step 2: Add implementation cell — `IntervalProcessorPolars`**

```python
class IntervalProcessorPolars:

    # ── Pure-Python helpers (identical to IntervalProcessor) ──────────────────

    @staticmethod
    def merge_intervals(intervals):
        if not intervals:
            return []
        intervals = sorted(intervals, key=lambda x: x[0])
        merged = [list(intervals[0])]
        for current in intervals[1:]:
            prev_min, prev_max = merged[-1]
            curr_min, curr_max = current
            if curr_min <= prev_max:
                merged[-1] = [prev_min, max(prev_max, curr_max)]
            else:
                merged.append(list(current))
        return [tuple(x) for x in merged]

    @staticmethod
    def subtract_intervals(target_min, target_max, blocked_intervals):
        safe_intervals = [(target_min, target_max)]
        for b_min, b_max in blocked_intervals:
            next_safe = []
            for s_min, s_max in safe_intervals:
                if b_max <= s_min or b_min >= s_max:
                    next_safe.append((s_min, s_max))
                else:
                    if b_min > s_min:
                        next_safe.append((s_min, b_min))
                    if b_max < s_max:
                        next_safe.append((b_max, s_max))
            safe_intervals = next_safe
            if not safe_intervals:
                break
        return safe_intervals

    # ── Polars-aware methods ───────────────────────────────────────────────────

    @staticmethod
    def create_cumulative_obstacles(possible_attacks: pl.DataFrame, min_step: int = 0) -> pl.DataFrame:
        """Same logic as IntervalProcessor.create_cumulative_obstacles but I/O is Polars."""
        max_step = int(possible_attacks["step"].max())

        # Single pass to build attack_map (avoids groupby + iterrows)
        attack_map = {}
        for row in possible_attacks.select(["id_src", "ships_sent", "step", "angle_min", "angle_max"]).to_dicts():
            key = (row["id_src"], row["ships_sent"], row["step"])
            amin, amax = row["angle_min"], row["angle_max"]
            if amin > amax:
                attack_map.setdefault(key, []).extend([(amin, 2 * np.pi), (0.0, amax)])
            else:
                attack_map.setdefault(key, []).append((amin, amax))

        unique_combinations = (
            possible_attacks.select(["id_src", "ships_sent"]).unique().to_numpy()
        )

        steps_col, id_srcs_col, ships_col, obstacles_col = [], [], [], []

        for id_src, ship in unique_combinations:
            current_intervals = []
            merged = []
            for step in range(min_step, max_step + 1):
                if (id_src, ship, step) in attack_map:
                    current_intervals.extend(attack_map[(id_src, ship, step)])
                    merged = IntervalProcessorPolars.merge_intervals(current_intervals)
                steps_col.append(step + 1)
                id_srcs_col.append(id_src)
                ships_col.append(ship)
                # Store as list-of-lists for Polars List(List(Float64)) schema
                obstacles_col.append([[a, b] for a, b in merged])

        return pl.DataFrame(
            {"step": steps_col, "id_src": id_srcs_col, "ships_sent": ships_col, "obstacle_list": obstacles_col},
            schema={"step": pl.Int64, "id_src": pl.Int64, "ships_sent": pl.Int64,
                    "obstacle_list": pl.List(pl.List(pl.Float64))},
        )

    @staticmethod
    def compute_free_angles(row: dict) -> list:
        """Row function for map_elements on struct(angle_min, angle_max, obstacle_list)."""
        amin = row["angle_min"]
        amax = row["angle_max"]
        obstacles = row["obstacle_list"] or []

        if not obstacles:
            return [[amin, amax]]

        targets = [(amin, 2 * np.pi), (0.0, amax)] if amin > amax else [(amin, amax)]

        all_free = []
        for t_min, t_max in targets:
            all_free.extend(IntervalProcessorPolars.subtract_intervals(t_min, t_max, obstacles))

        has_end   = any(abs(f[1] - 2 * np.pi) < 1e-9 for f in all_free)
        has_start = any(abs(f[0] - 0.0) < 1e-9 for f in all_free)

        if has_end and has_start and len(all_free) > 1:
            end_idx   = next(i for i, f in enumerate(all_free) if abs(f[1] - 2 * np.pi) < 1e-9)
            start_idx = next(i for i, f in enumerate(all_free) if abs(f[0] - 0.0) < 1e-9)
            wrapped   = (all_free[end_idx][0], all_free[start_idx][1])
            all_free  = [f for i, f in enumerate(all_free) if i not in {end_idx, start_idx}]
            all_free.append(wrapped)

        return [[a, b] for a, b in all_free]

    @staticmethod
    def interval_to_final_angle(series: pl.Series) -> pl.Series:
        def _best(intervals):
            if not intervals:
                return float("nan")
            widest, best = -1.0, float("nan")
            for interval in intervals:
                amin, amax = interval[0], interval[1]
                span = (amax - amin) if amin <= amax else (2 * np.pi - amin + amax)
                if span > widest:
                    widest = span
                    mid = (amin + amax) / 2.0 if amin <= amax else amin + span / 2.0
                    best = mid % (2 * np.pi)
            return best
        return series.map_elements(_best, return_dtype=pl.Float64)
```

- [ ] **Step 3: Re-run the failing assertion cell**

Expected: `merge_intervals OK`

- [ ] **Step 4: Add unit-test cell for all pure-Python helpers**

```python
# subtract_intervals
assert IntervalProcessorPolars.subtract_intervals(0.0, 1.0, [(0.3, 0.7)]) == [(0.0, 0.3), (0.7, 1.0)]
assert IntervalProcessorPolars.subtract_intervals(0.0, 1.0, []) == [(0.0, 1.0)]
assert IntervalProcessorPolars.subtract_intervals(0.0, 1.0, [(0.0, 1.0)]) == []
print("subtract_intervals OK")

# interval_to_final_angle — midpoint of widest interval
s = pl.Series([[[0.0, 1.0], [2.0, 4.0]]])
result = IntervalProcessorPolars.interval_to_final_angle(s)
assert abs(result[0] - 3.0) < 1e-9, f"Expected 3.0, got {result[0]}"
print("interval_to_final_angle OK")
```

Run the cell. Expected: both `OK` lines printed.

- [ ] **Step 5: Commit**

```bash
git add "45-Polars_comet.ipynb"
git commit -m "feat: add IntervalProcessorPolars class with all static methods"
```

---

### Task 3: Implement `take_action_polars` — mine + expand + cross join

**Files:**
- Modify: `45-Polars_comet.ipynb`

- [ ] **Step 1: Add skeleton cell for `take_action_polars`** (just the function signature + first conversion)

```python
def take_action_polars(df: pd.DataFrame, player_id: int,
                       nb_steps_sim: int = NB_STEPS_SIM,
                       return_df: bool = False):
    df_pl = pl.from_pandas(df).sort("step")  # sort ensures first() == min-step row

    # ── Step A: source planets ────────────────────────────────────────────────
    mine_across_sim = (
        df_pl
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
    )

    if mine_across_sim.is_empty():
        return ([], pl.DataFrame()) if return_df else []

    # ── Step B: expand ships_sent 1..ships_min+production*NB_STEPS_SIM ────────
    expanded_mine = (
        mine_across_sim
        .with_columns(
            pl.int_ranges(
                1,
                pl.col("ships_min") + pl.col("production_src") * NB_STEPS_SIM + 1,
                dtype=pl.Int64,
            ).alias("ships_sent")
        )
        .explode("ships_sent")
    )

    # ── Step C: cross join with all simulation rows, filter future steps ───────
    df_src_tgt = (
        expanded_mine
        .join(df_pl, how="cross")
        .filter(
            (pl.col("step") > pl.col("step_src")) &
            (pl.col("id") != pl.col("id_src"))
        )
    )

    # Placeholder — will be filled in Task 4
    return []
```

- [ ] **Step 2: Add quick smoke-test cell**

```python
obs_smoke = Obs(
    planets=[[0, 0, 10.0, 10.0, 5.0, 6, 3], [1, -1, 30.0, 10.0, 5.0, 5, 2]],
    angular_velocity=0.0,
)
df_smoke = _simulate(obs_smoke, global_step=0, num_agents=2, n_steps=10)
result = take_action_polars(df_smoke, player_id=0)
print("Smoke test passed, result:", result)  # [] expected (placeholder)
```

Run cell. Expected: no errors, prints `[]`.

- [ ] **Step 3: Commit**

```bash
git add "45-Polars_comet.ipynb"
git commit -m "feat: add take_action_polars skeleton with mine/expand/cross-join steps"
```

---

### Task 4: Implement `take_action_polars` — `possible_attacks` with vectorized `crossing_sun`

**Files:**
- Modify: `45-Polars_comet.ipynb`

- [ ] **Step 1: Replace the placeholder `return []` in `take_action_polars` with the full `possible_attacks` block**

Find the line `# Placeholder — will be filled in Task 4` and replace it + `return []` with:

```python
    # ── Step D: compute distances, collision window, crossing_sun ─────────────
    dist_tgt_src_expr = (
        (pl.col("x") - pl.col("x_src")).pow(2) +
        (pl.col("y") - pl.col("y_src")).pow(2)
    ).sqrt()
    step_diff_expr = pl.col("step") - pl.col("step_src")
    fleet_speed_expr = (
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

    # Vectorised point-to-segment distance: sun=(CENTER,CENTER) to segment (src→tgt)
    dx_vw = pl.col("x") - pl.col("x_src")
    dy_vw = pl.col("y") - pl.col("y_src")
    l2    = dx_vw.pow(2) + dy_vw.pow(2)
    dot   = (CENTER - pl.col("x_src")) * dx_vw + (CENTER - pl.col("y_src")) * dy_vw
    t_raw = dot / pl.when(l2 == 0).then(pl.lit(1.0)).otherwise(l2)
    t     = t_raw.clip(0.0, 1.0)
    dist_sun_proj = (
        ((CENTER - (pl.col("x_src") + t * dx_vw)).pow(2) +
         (CENTER - (pl.col("y_src") + t * dy_vw)).pow(2))
        .sqrt()
    )
    dist_sun_direct = (
        ((CENTER - pl.col("x_src")).pow(2) + (CENTER - pl.col("y_src")).pow(2)).sqrt()
    )
    dist_to_sun  = pl.when(l2 == 0).then(dist_sun_direct).otherwise(dist_sun_proj)
    crossing_sun_expr = dist_to_sun < (SUN_RADIUS + PLANET_MARGIN)

    possible_attacks = (
        df_src_tgt
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
    )

    if possible_attacks.is_empty():
        return ([], possible_attacks) if return_df else []

    # Placeholder — will be filled in Task 5
    return ([], possible_attacks) if return_df else []
```

- [ ] **Step 2: Add `possible_attacks` verification cell**

```python
obs_v = Obs(
    planets=[[0, 0, 10.0, 10.0, 5.0, 6, 3], [1, -1, 30.0, 10.0, 5.0, 5, 2]],
    angular_velocity=0.0,
)
df_v = _simulate(obs_v, global_step=0, num_agents=2, n_steps=10)

_, pa_pd = take_action(df_v, player_id=0, nb_steps_sim=10, return_df=True)
_, pa_pl = take_action_polars(df_v, player_id=0, nb_steps_sim=10, return_df=True)

pa_pd_sorted = pa_pd.sort_values(["id_src", "step", "ships_sent"]).reset_index(drop=True)
pa_pl_sorted = pa_pl.sort(["id_src", "step", "ships_sent"]).to_pandas().reset_index(drop=True)

assert len(pa_pd_sorted) == len(pa_pl_sorted), f"Row count differs: {len(pa_pd_sorted)} vs {len(pa_pl_sorted)}"
for col in ["id_src", "step", "ships_sent"]:
    assert (pa_pd_sorted[col].values == pa_pl_sorted[col].values).all(), f"Column {col} differs"
for col in ["angle_min", "angle_max", "dist_tgt_src"]:
    assert np.allclose(pa_pd_sorted[col].values, pa_pl_sorted[col].values, atol=1e-9), f"Column {col} differs"
print(f"possible_attacks matches: {len(pa_pl_sorted)} rows ✓")
```

Run cell. Expected: `possible_attacks matches: N rows ✓`

- [ ] **Step 3: Commit**

```bash
git add "45-Polars_comet.ipynb"
git commit -m "feat: add vectorized crossing_sun and possible_attacks in take_action_polars"
```

---

### Task 5: Implement `take_action_polars` — interval processing + final attack selection

**Files:**
- Modify: `45-Polars_comet.ipynb`

- [ ] **Step 1: Replace the second placeholder in `take_action_polars` with the full pipeline completion**

Find `# Placeholder — will be filled in Task 5` and replace it + `return ([], possible_attacks) if return_df else []` with:

```python
    # ── Step E: cumulative obstacle intervals ──────────────────────────────────
    df_obstacles = IntervalProcessorPolars.create_cumulative_obstacles(possible_attacks)

    # ── Step F: free angles per attack row ────────────────────────────────────
    attacks_with_angle = (
        possible_attacks
        .join(df_obstacles, on=["id_src", "step", "ships_sent"], how="left")
        .with_columns(
            pl.struct(["angle_min", "angle_max", "obstacle_list"])
            .map_elements(
                IntervalProcessorPolars.compute_free_angles,
                return_dtype=pl.List(pl.List(pl.Float64)),
            )
            .alias("angle_list")
        )
        .filter(pl.col("angle_list").list.len() > 0)
    )

    # ── Step G: top-5 reachable targets per source planet ─────────────────────
    planet_id_top_5 = (
        attacks_with_angle
        .sort(["step", "ships_sent"])
        .group_by(["id_src", "id"], maintain_order=True)
        .first()
        .sort(["step", "ships_sent"])
        .group_by("id_src", maintain_order=True)
        .head(5)
        .select(["id_src", "id"])
    )

    # ── Step H: comet branch ──────────────────────────────────────────────────
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

    # ── Step I: score and select one attack per source ─────────────────────────
    attacks = (
        planet_id_top_5
        .join(attacks_with_angle, on=["id_src", "id"], how="left")
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
            IntervalProcessorPolars.interval_to_final_angle(pl.col("angle_list")).alias("final_angle")
        )
    )

    for row in attacks.rows(named=True):
        print(f"From {row['id_src']}, To {row['id']} at step {row['step']} "
              f"with {row['ships_sent']} ships (target has min {row['ships_min']})")

    moves += [list(r) for r in attacks.select(["id_src", "final_angle", "ships_sent"]).rows()]

    return (moves, possible_attacks) if return_df else moves
```

- [ ] **Step 2: Smoke-test the full function**

```python
obs_full = Obs(
    planets=[[0, 0, 10.0, 10.0, 5.0, 6, 3], [1, -1, 30.0, 10.0, 5.0, 5, 2]],
    angular_velocity=0.0,
)
df_full = _simulate(obs_full, global_step=0, num_agents=2, n_steps=10)
moves_pd = take_action(df_full, player_id=0, nb_steps_sim=10)
moves_pl = take_action_polars(df_full, player_id=0, nb_steps_sim=10)
print("pandas:", moves_pd)
print("polars:", moves_pl)
```

Expected: both print the same move list (one attack). Verify `id_src` and `ships_sent` match exactly; `final_angle` within `1e-6`.

- [ ] **Step 3: Commit**

```bash
git add "45-Polars_comet.ipynb"
git commit -m "feat: complete take_action_polars with interval processing and attack selection"
```

---

### Task 6: Equality tests — Tests 1–5, 7, 8

**Files:**
- Modify: `45-Polars_comet.ipynb`

Add one markdown cell + two code cells per test (setup + assertion). Tests 1, 3, 4 expect `[]`. Tests 2, 5, 7, 8 expect non-empty moves.

- [ ] **Step 1: Add assert helper cell**

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

- [ ] **Step 2: Add Test 1 — single owned planet, no attacks expected**

```python
## Test 1 — Planet production (no targets → [])
obs1 = Obs(planets=[[0, 0, 10.0, 10.0, 5.0, 1, 3]], angular_velocity=0.0)
df1 = _simulate(obs1, global_step=0, num_agents=2, n_steps=NB_STEPS_SIM)
assert_moves_equal(
    take_action(df1, player_id=0, nb_steps_sim=NB_STEPS_SIM),
    take_action_polars(df1, player_id=0, nb_steps_sim=NB_STEPS_SIM),
    "Test 1"
)
```

Run. Expected: `Test 1: ✓  moves=[]`

- [ ] **Step 3: Add Test 2 — attack neutral**

```python
## Test 2 — Attack neutral (6 vs 5 → should attack)
obs2 = Obs(
    planets=[[0, 0, 10.0, 10.0, 5.0, 6, 3], [1, -1, 30.0, 10.0, 5.0, 5, 2]],
    angular_velocity=0.0,
)
df2 = _simulate(obs2, global_step=0, num_agents=2, n_steps=NB_STEPS_SIM)
assert_moves_equal(
    take_action(df2, player_id=0, nb_steps_sim=NB_STEPS_SIM),
    take_action_polars(df2, player_id=0, nb_steps_sim=NB_STEPS_SIM),
    "Test 2"
)
```

Run. Expected: `Test 2: ✓  moves=[[0, <angle>, <ships>]]`

- [ ] **Step 4: Add Test 3 — equal ships, do nothing**

```python
## Test 3 — Equal ships (5 vs 5 → do nothing)
obs3 = Obs(
    planets=[[0, 0, 10.0, 10.0, 5.0, 5, 3], [1, -1, 30.0, 10.0, 5.0, 5, 2]],
    angular_velocity=0.0,
)
df3 = _simulate(obs3, global_step=0, num_agents=2, n_steps=5)
assert_moves_equal(
    take_action(df3, player_id=0, nb_steps_sim=5),
    take_action_polars(df3, player_id=0, nb_steps_sim=5),
    "Test 3"
)
```

Run. Expected: `Test 3: ✓  moves=[]`

- [ ] **Step 5: Add Test 4 — enemy fleet inbound, do nothing**

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
    take_action_polars(df4, player_id=0, nb_steps_sim=10),
    "Test 4"
)
```

Run. Expected: `Test 4: ✓  moves=[]`

- [ ] **Step 6: Add Test 5 — attack enemy with overwhelming force**

```python
## Test 5 — Attack enemy (50 vs 5 → attack)
obs5 = Obs(
    planets=[[0, 0, 10.0, 10.0, 5.0, 50, 3], [1, 1, 30.0, 10.0, 5.0, 5, 2]],
    angular_velocity=0.0,
)
df5 = _simulate(obs5, global_step=0, num_agents=2, n_steps=5)
assert_moves_equal(
    take_action(df5, player_id=0, nb_steps_sim=5),
    take_action_polars(df5, player_id=0, nb_steps_sim=5),
    "Test 5"
)
```

Run. Expected: `Test 5: ✓  moves=[[0, <angle>, <ships>]]`

- [ ] **Step 7: Add Test 7 — target planet behind large planet**

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
    take_action_polars(df7, player_id=0, nb_steps_sim=NB_STEPS_SIM),
    "Test 7"
)
```

Run. Expected: `Test 7: ✓`

- [ ] **Step 8: Add Test 8 — moving planet with angular_velocity**

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
    take_action_polars(df8, player_id=0, nb_steps_sim=n_steps),
    "Test 8"
)
```

Run. Expected: `Test 8: ✓`

- [ ] **Step 9: Commit**

```bash
git add "45-Polars_comet.ipynb"
git commit -m "test: add equality assertions for Tests 1-5, 7, 8 in 45-Polars_comet.ipynb"
```

---

### Task 7: Add timing comparison cells

**Files:**
- Modify: `45-Polars_comet.ipynb`

- [ ] **Step 1: Add timing cell for Test 2 (typical case)**

```python
## Timing comparison — Test 2 (10-step sim, 1 source, 1 target)
import timeit

obs_t = Obs(
    planets=[[0, 0, 10.0, 10.0, 5.0, 6, 3], [1, -1, 30.0, 10.0, 5.0, 5, 2]],
    angular_velocity=0.0,
)
df_t = _simulate(obs_t, global_step=0, num_agents=2, n_steps=NB_STEPS_SIM)

t_pd = timeit.timeit(lambda: take_action(df_t, player_id=0), number=20) / 20
t_pl = timeit.timeit(lambda: take_action_polars(df_t, player_id=0), number=20) / 20
print(f"pandas:  {t_pd*1000:.2f} ms")
print(f"polars:  {t_pl*1000:.2f} ms")
print(f"speedup: {t_pd/t_pl:.1f}×")
```

- [ ] **Step 2: Add timing cell for Test 8 (larger sim — 50 steps, 3 planets)**

```python
## Timing comparison — Test 8 (50-step sim, 1 source, 2 targets)
obs_t8 = Obs(
    planets=[[0, 0, 10.0, 10.0, 5.0, 50, 3], [1, -1, 30.0, 5.0, 5.0, 5, 2],
             [2, -1, 30.0, 50.0, 5.0, 5, 2]],
    angular_velocity=math.pi / 20,
)
df_t8 = _simulate(obs_t8, global_step=0, num_agents=2, n_steps=50)

t_pd8 = timeit.timeit(lambda: take_action(df_t8, player_id=0, nb_steps_sim=50), number=10) / 10
t_pl8 = timeit.timeit(lambda: take_action_polars(df_t8, player_id=0, nb_steps_sim=50), number=10) / 10
print(f"pandas:  {t_pd8*1000:.2f} ms")
print(f"polars:  {t_pl8*1000:.2f} ms")
print(f"speedup: {t_pd8/t_pl8:.1f}×")
```

- [ ] **Step 3: Commit**

```bash
git add "45-Polars_comet.ipynb"
git commit -m "perf: add timing comparison cells to 45-Polars_comet.ipynb"
```

---

### Task 8: Test 6 — full game vs random agent (slow)

**Files:**
- Modify: `45-Polars_comet.ipynb`

Run this task last. Requires `kaggle_environments` (`import kaggle_environments as ke`).

- [ ] **Step 1: Add Test 6 — full game equality cell**

```python
## Test 6 — Full game (our agent pandas vs polars, step-by-step comparison)
import random
import kaggle_environments as ke

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

env_pd = ke.make("orbit_wars", debug=False)
env_pd.reset(2)
env_pl = ke.make("orbit_wars", debug=False)
env_pl.reset(2)

# Replay the same game steps with both agents and compare moves each step
random.seed(SEED)
for env_step in range(N_STEPS):
    obs0_pd = env_pd.state[0].observation
    obs0_pl = env_pl.state[0].observation
    obs1_pd = env_pd.state[1].observation
    obs1_pl = env_pl.state[1].observation

    df_pd = _simulate(copy.deepcopy(obs0_pd), global_step=env_step, num_agents=2, n_steps=NB_STEPS_SIM)
    df_pl_in = _simulate(copy.deepcopy(obs0_pl), global_step=env_step, num_agents=2, n_steps=NB_STEPS_SIM)

    moves_pd = take_action(df_pd, player_id=0, nb_steps_sim=NB_STEPS_SIM)
    moves_pl = take_action_polars(df_pl_in, player_id=0, nb_steps_sim=NB_STEPS_SIM)
    assert_moves_equal(moves_pd, moves_pl, f"step {env_step}")

    rng_action_pd = random_agent_fn(obs1_pd)
    rng_action_pl = random_agent_fn(obs1_pl)
    env_pd.step([moves_pd, rng_action_pd])
    env_pl.step([moves_pl, rng_action_pl])

    if env_pd.state[0].status != "ACTIVE":
        break

print(f"Test 6 passed: {env_step + 1} steps, all moves identical ✓")
```

Run. Expected: `Test 6 passed: N steps, all moves identical ✓`

- [ ] **Step 2: Commit**

```bash
git add "45-Polars_comet.ipynb"
git commit -m "test: add Test 6 full-game equality check in 45-Polars_comet.ipynb"
```

---

### Task 9: Assemble `46-Polars_comet.py`

**Files:**
- Create: `46-Polars_comet.py`

- [ ] **Step 1: Create `46-Polars_comet.py`**

Copy `44-Dataframe_comet.py` exactly, then apply the following changes:

1. After `import numpy as np` (line 325), add `import polars as pl`

2. After the closing of `take_action` function (after line 705), add the full `IntervalProcessorPolars` class and `take_action_polars` function as implemented in the notebook (Tasks 2 and 5).

3. In the `nearest_planet_sniper` function, replace:
```python
    df = _simulate(obs, step, num_agents, n_steps=NB_STEPS_SIM)
    # moves = take_action(df, player_id=player_id, nb_steps_sim=NB_STEPS_SIM)


    step += 1
    # return moves
    return []
```
with:
```python
    df = _simulate(obs, step, num_agents, n_steps=NB_STEPS_SIM)
    moves = take_action_polars(df, player_id=player_id, nb_steps_sim=NB_STEPS_SIM)

    step += 1
    return moves
```

- [ ] **Step 2: Verify the file runs without import errors**

```bash
python -c "import importlib.util; spec = importlib.util.spec_from_file_location('agent46', '46-Polars_comet.py'); m = importlib.util.load_from_spec(spec); spec.loader.exec_module(m); print('46 loads OK')"
```

Expected: `46 loads OK`

- [ ] **Step 3: Quick smoke test of `46-Polars_comet.py` agent function**

```python
# Run in Python REPL or notebook cell
import importlib.util, sys
spec = importlib.util.spec_from_file_location("agent46", "46-Polars_comet.py")
m = importlib.util.load_from_spec(spec)
spec.loader.exec_module(m)

# Build a minimal obs and call the agent
obs_test = m.Obs(
    planets=[[0, 0, 10.0, 10.0, 5.0, 6, 3], [1, -1, 30.0, 10.0, 5.0, 5, 2]],
    angular_velocity=0.0,
)
obs_test.player = 0
obs_test.remainingOverageTime = 60
result = m.agent(obs_test)
print("Agent output:", result)
assert isinstance(result, list)
print("46-Polars_comet.py smoke test ✓")
```

Expected: `Agent output: [[0, <angle>, <ships>]]` and `smoke test ✓`

- [ ] **Step 4: Commit**

```bash
git add "46-Polars_comet.py"
git commit -m "feat: add 46-Polars_comet.py — full Kaggle agent using take_action_polars"
```

---

## Self-Review Notes

- **Spec coverage:** All spec sections covered — `IntervalProcessorPolars` (Task 2), `take_action_polars` pipeline (Tasks 3–5), notebook with test equality + timing (Tasks 6–7), Test 6 (Task 8), submission file (Task 9). ✓
- **Polars `group_by` ordering:** All group-by operations use `maintain_order=True` or are preceded by an explicit `.sort(...)` to match pandas `"first"` semantics. ✓
- **`NB_STEPS_SIM` global vs parameter:** `expanded_mine` uses `NB_STEPS_SIM` global (same as pandas version) to match behavior exactly. ✓
- **`obstacle_list` dtype:** Explicitly typed as `pl.List(pl.List(pl.Float64))` in `create_cumulative_obstacles` schema. ✓
- **`compute_free_angles` returns list-of-lists:** All return sites use `[[a, b] for a, b in ...]` so `map_elements` receives the correct nested-list type. ✓
- **Edge case `l2 == 0`:** Guarded with `pl.when(l2 == 0).then(...)` in vectorized `crossing_sun`. ✓
