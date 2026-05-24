# 62-One_angle_polars Polars Conversion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convert `take_action` in `62-One_angle_polars.py` from pandas to Polars using a single `.lazy()` / `.collect()` pair, inlining `is_crossing_sun_vectorized` and `_filter_blocked_attacks` as lazy Polars expressions.

**Architecture:** One `pl.from_pandas(df).lazy()` entry → full pipeline (mine agg → cross-join → collision/sun filter → blocking self-join anti-join) → single `.collect()` → eager comet handling and scoring. CSE (default in Polars 1.35.2) caches `pa_lf` which appears 3× in the DAG.

**Tech Stack:** Python 3.12, Polars 1.35.2, pandas (only for `_simulate` return type), pytest

---

## File Map

| Action | Path |
|---|---|
| Modify | `62-One_angle_polars.py` |
| Create | `tests/test_62_polars_conversion.py` |

---

### Task 1: Write failing tests

**Files:**
- Create: `tests/test_62_polars_conversion.py`

These tests will FAIL before the conversion because `take_action` still uses pandas; they verify the converted Polars function produces identical outputs.

- [ ] **Step 1: Create the test file**

```python
# tests/test_62_polars_conversion.py
import sys
import os
import importlib.util
import math
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_62():
    path = os.path.join(ROOT, "62-One_angle_polars.py")
    spec = importlib.util.spec_from_file_location("mod62", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["mod62"] = mod
    spec.loader.exec_module(mod)
    return mod


def _make_df(rows_per_planet=11):
    """Two-planet fixture: planet 1 (mine) at (20,80), planet 2 (enemy) at (20,50).
    Path is vertical x=20; sun at (50,50) is 30 units away — no crossing.
    At step_diff=10, ships_sent≈12 reaches planet 2 (distance 30, radius 5).
    """
    rows = []
    for step in range(rows_per_planet):
        rows.append({
            "step": step, "id": 1, "x": 20.0, "y": 80.0, "radius": 5.0,
            "ships": 50, "production": 3, "owner": 0, "nature": "fix",
        })
        rows.append({
            "step": step, "id": 2, "x": 20.0, "y": 50.0, "radius": 5.0,
            "ships": 10, "production": 1, "owner": 1, "nature": "fix",
        })
    return pd.DataFrame(rows)


def _make_sun_blocked_df(rows_per_planet=11):
    """Planets aligned through sun: (40,50) → (60,50). Path passes through sun at (50,50).
    All attacks should be filtered out → moves = [].
    """
    rows = []
    for step in range(rows_per_planet):
        rows.append({
            "step": step, "id": 1, "x": 40.0, "y": 50.0, "radius": 3.0,
            "ships": 50, "production": 3, "owner": 0, "nature": "fix",
        })
        rows.append({
            "step": step, "id": 2, "x": 60.0, "y": 50.0, "radius": 3.0,
            "ships": 5, "production": 1, "owner": 1, "nature": "fix",
        })
    return pd.DataFrame(rows)


def _normalize(moves):
    """Normalize move list to Python native types for stable comparison."""
    return [[int(m[0]), float(m[1]), int(m[2])] for m in moves]


mod62 = load_62()


def test_take_action_produces_moves():
    """Mine planet attacks enemy planet — expect at least one move."""
    df = _make_df()
    moves = mod62.take_action(df, player_id=0)
    assert len(moves) >= 1
    for m in moves:
        assert len(m) == 3, "each move is [id_src, angle, ships_sent]"
        id_src, angle, ships_sent = m
        assert int(id_src) == 1
        assert isinstance(float(angle), float)
        assert int(ships_sent) > 0


def test_take_action_angle_is_downward():
    """Planet 1 at (20,80) attacking planet 2 at (20,50): angle should be ≈ -π/2."""
    df = _make_df()
    moves = mod62.take_action(df, player_id=0)
    assert len(moves) >= 1
    _, angle, _ = moves[0]
    assert math.isclose(float(angle), -math.pi / 2, abs_tol=0.1)


def test_take_action_no_mine_returns_empty():
    """player_id with no owned planets returns empty move list."""
    df = _make_df()
    moves = mod62.take_action(df, player_id=99)
    assert moves == []


def test_take_action_sun_blocked_path_returns_empty():
    """Path directly through sun is rejected by crossing_sun filter."""
    df = _make_sun_blocked_df()
    moves = mod62.take_action(df, player_id=0)
    assert moves == []


def test_take_action_return_df_flag():
    """return_df=True returns (moves, DataFrame) tuple."""
    import polars as pl
    df = _make_df()
    result = mod62.take_action(df, player_id=0, return_df=True)
    assert isinstance(result, tuple) and len(result) == 2
    moves, attacks_df = result
    assert isinstance(moves, list)
    assert isinstance(attacks_df, pl.DataFrame)
```

- [ ] **Step 2: Run tests — expect failures**

```
cd "C:\Users\trant\Documents\Programmation\Orbit Wars"
python -m pytest tests/test_62_polars_conversion.py -v
```

Expected: `test_take_action_return_df_flag` FAILS (returns `pd.DataFrame`, not `pl.DataFrame`). Others may pass (pandas version is functional). Note which pass/fail — these are the baseline.

---

### Task 2: Add polars import; remove helper functions

**Files:**
- Modify: `62-One_angle_polars.py`

- [ ] **Step 1: Add `import polars as pl` near the other imports**

In `62-One_angle_polars.py`, the imports block ends around line 3 (`import pandas as pd`). Add after `import numpy as np` (line 347):

```python
import polars as pl
```

- [ ] **Step 2: Delete `is_crossing_sun_vectorized` (lines 52–71)**

Remove the entire function:

```python
def is_crossing_sun_vectorized(x_src, y_src, x, y, center, threshold):
    """Vectorized point-to-segment distance from sun center to each fleet path."""
    import numpy as np
    ...
    return dist < threshold
```

- [ ] **Step 3: Delete `_filter_blocked_attacks` (lines 350–398)**

Remove the entire function:

```python
def _filter_blocked_attacks(possible_attacks):
    """Drop attacks whose direct angle to target is blocked by an earlier obstacle.
    ...
    """
    pairs = (
        ...
    )
    ...
    return (...)
```

- [ ] **Step 4: Run existing tests to verify nothing else broke**

```
python -m pytest tests/ -v --ignore=tests/test_62_polars_conversion.py
```

Expected: all pre-existing tests still PASS (the deleted functions are only used by `take_action` in file 62).

---

### Task 3: Replace `take_action` — Sections 1–3 (entry + mine + cross-join)

**Files:**
- Modify: `62-One_angle_polars.py` (replace the existing `take_action` function, lines ~401–550)

- [ ] **Step 1: Replace the `take_action` function signature and Sections 1–3**

Delete the entire existing `take_action` function and replace with:

```python
def take_action(df, player_id, nb_steps_sim=NB_STEPS_SIM, return_df=False):
    # ── Section 1: entry ───────────────────────────────────────────────────────
    df_lf = pl.from_pandas(df).sort("step").lazy()

    # ── Section 2: mine analysis + ships_sent expansion ────────────────────────
    mine_lf = (
        df_lf
        .with_columns(
            pl.when(pl.col("owner") == player_id).then(1).otherwise(0).alias("is_mine")
        )
        .group_by("id", maintain_order=True)
        .agg([
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
        ])
        .filter(
            (pl.col("row_count") == pl.col("is_mine")) & (pl.col("owner_src") == player_id)
        )
        .rename({"id": "id_src"})
        .with_columns(
            pl.int_ranges(
                1,
                pl.col("ships_min") + pl.col("production_src") * nb_steps_sim + 1,
                dtype=pl.Int64,
            ).alias("ships_sent")
        )
        .explode("ships_sent")
    )

    # ── Section 3: cross-join + collision + sun (all lazy, all inlined) ────────
    dx = pl.col("x") - pl.col("x_src")
    dy = pl.col("y") - pl.col("y_src")
    l2 = dx.pow(2) + dy.pow(2)
    dist_tgt_src = l2.sqrt()
    step_diff   = pl.col("step") - pl.col("step_src")
    fleet_speed = 1.0 + (MAX_SPEED - 1.0) * (
        pl.col("ships_sent").cast(pl.Float64).log(base=math.e) / math.log(1000.0)
    ).pow(1.5)
    dist_min = step_diff * fleet_speed + PLANET_MARGIN + pl.col("radius_src")
    dist_max = (step_diff + 1) * fleet_speed + PLANET_MARGIN + pl.col("radius_src")
    collision = (
        ((dist_tgt_src - pl.col("radius") < dist_min) & (dist_min < dist_tgt_src + pl.col("radius"))) |
        ((dist_tgt_src - pl.col("radius") < dist_max) & (dist_max < dist_tgt_src + pl.col("radius")))
    )
    dot   = (CENTER - pl.col("x_src")) * dx + (CENTER - pl.col("y_src")) * dy
    t_sun = (dot / pl.when(l2 == 0).then(pl.lit(1.0)).otherwise(l2)).clip(0.0, 1.0)
    proj_dist_sun = (
        (CENTER - pl.col("x_src") - t_sun * dx).pow(2) +
        (CENTER - pl.col("y_src") - t_sun * dy).pow(2)
    ).sqrt()
    crossing_sun = pl.when(l2 == 0).then(
        ((CENTER - pl.col("x_src")).pow(2) + (CENTER - pl.col("y_src")).pow(2)).sqrt()
    ).otherwise(proj_dist_sun) < (SUN_RADIUS + PLANET_MARGIN)

    pa_lf = (
        mine_lf
        .join(df_lf, how="cross")
        .filter((pl.col("step") > pl.col("step_src")) & (pl.col("id") != pl.col("id_src")))
        .with_columns([
            dist_tgt_src.alias("dist_tgt_src"),
            step_diff.alias("step_diff"),
            fleet_speed.alias("fleet_speed"),
            dist_min.alias("dist_fleet_src_min"),
            dist_max.alias("dist_fleet_src_max"),
            collision.alias("collision"),
        ])
        .filter(pl.col("collision"))
        .filter(~crossing_sun)
        .with_columns(pl.arctan2(dy, dx).alias("angle"))
        .with_columns(
            pl.max_horizontal(
                ((pl.col("dist_tgt_src").pow(2) + pl.col("dist_fleet_src_min").pow(2) - pl.col("radius").pow(2))
                 / (2 * pl.col("dist_tgt_src") * pl.col("dist_fleet_src_min"))).clip(-1.0, 1.0).arccos(),
                ((pl.col("dist_tgt_src").pow(2) + pl.col("dist_fleet_src_max").pow(2) - pl.col("radius").pow(2))
                 / (2 * pl.col("dist_tgt_src") * pl.col("dist_fleet_src_max"))).clip(-1.0, 1.0).arccos(),
            ).alias("radius_angle")
        )
        .with_columns([
            ((pl.col("angle") - pl.col("radius_angle")) % (2 * math.pi)).alias("angle_min"),
            ((pl.col("angle") + pl.col("radius_angle")) % (2 * math.pi)).alias("angle_max"),
        ])
        .sort("step")
    )
```

*(Do NOT close the function yet — Sections 4–5 come in the next task.)*

- [ ] **Step 2: Verify module loads without syntax errors**

```
python -c "import importlib.util, sys; spec = importlib.util.spec_from_file_location('m', '62-One_angle_polars.py'); m = importlib.util.module_from_spec(spec); sys.modules['m'] = m; spec.loader.exec_module(m); print('OK')"
```

Expected: prints `OK`. The function body is syntactically valid but incomplete (returns `None` until Task 4 adds Sections 4–5). Do NOT run the full test suite yet.

---

### Task 4: Complete `take_action` — Sections 4–5 (blocking + comet + scoring)

**Files:**
- Modify: `62-One_angle_polars.py` (append Sections 4–5 inside the open `take_action` function)

- [ ] **Step 1: Append Sections 4–5 to `take_action`**

Immediately after the `pa_lf = (... .sort("step") )` block, add:

```python
    # ── Section 4: blocking self-join + THE ONE collect() ──────────────────────
    angle_norm = pl.col("angle") % (2 * math.pi)
    wraps      = pl.col("angle_min_obs") > pl.col("angle_max_obs")
    in_cone    = pl.when(wraps).then(
        (angle_norm >= pl.col("angle_min_obs")) | (angle_norm <= pl.col("angle_max_obs"))
    ).otherwise(
        (angle_norm >= pl.col("angle_min_obs")) & (angle_norm <= pl.col("angle_max_obs"))
    )

    blocked_lf = (
        pa_lf.select(["id_src", "ships_sent", "step", "id", "angle", "angle_min", "angle_max"])
        .join(
            pa_lf.select(["id_src", "ships_sent", "step", "id", "angle_min", "angle_max"])
                 .rename({"step": "step_obs", "id": "id_obs",
                          "angle_min": "angle_min_obs", "angle_max": "angle_max_obs"}),
            on=["id_src", "ships_sent"],
            how="inner",
        )
        .filter((pl.col("step_obs") < pl.col("step")) & (pl.col("id_obs") != pl.col("id")))
        .filter(in_cone)
        .select(["id_src", "ships_sent", "step", "id"])
        .unique()
    )

    attacks_with_angle = (
        pa_lf
        .join(blocked_lf, on=["id_src", "ships_sent", "step", "id"], how="anti")
        .with_columns(pl.col("angle").alias("final_angle"))
        .collect()
    )

    # ── Section 5: comet handling + scoring (eager) ────────────────────────────
    if attacks_with_angle.is_empty():
        return ([], attacks_with_angle) if return_df else []

    awa_comets = attacks_with_angle.filter(pl.col("nature_src") == "comet")
    moves = []
    if not awa_comets.is_empty():
        x_off = (awa_comets["x_src"] - CENTER).abs().max() or 0
        y_off = (awa_comets["y_src"] - CENTER).abs().max() or 0
        if max(x_off, y_off) > 45:
            moves += [list(r) for r in (
                awa_comets
                .filter(pl.col("ships_sent") <= pl.col("ships_min"))
                .sort(["ships_sent", "step"], descending=[True, False])
                .group_by("id_src", maintain_order=True)
                .first()
                .select(["id_src", "final_angle", "ships_sent"])
                .rows()
            )]
            id_to_avoid = awa_comets["id_src"].unique().to_list()
            attacks_with_angle = attacks_with_angle.filter(~pl.col("id_src").is_in(id_to_avoid))

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
        .with_columns((pl.col("ships_needed") / pl.col("production_src")).alias("time_cost"))
        .with_columns(pl.col("time_cost").sum().over("id_src").alias("total_time_cost"))
        .with_columns(
            ((pl.col("total_time_cost") - pl.col("time_cost") - pl.col("step_diff")) * pl.col("production"))
            .alias("score")
        )
        .sort("score", descending=True)
        .group_by("id_src", maintain_order=True)
        .first()
        .filter(pl.col("ships_sent") <= pl.col("ships_min"))
    )

    for row in attacks.iter_rows(named=True):
        print(f"From {row['id_src']}, To {row['id']} at step {row['step']} "
              f"with {row['ships_sent']} ships (target has min {row['ships_min']})")

    moves += [list(r) for r in attacks.select(["id_src", "final_angle", "ships_sent"]).rows()]
    return (moves, attacks_with_angle) if return_df else moves
```

- [ ] **Step 2: Run all 62 tests**

```
python -m pytest tests/test_62_polars_conversion.py -v
```

Expected: ALL 5 tests PASS.

If `test_take_action_sun_blocked_path_returns_empty` fails, verify the sun geometry: path from (40,50) to (60,50) passes through (50,50) which is the sun center. The `crossing_sun` expression computes `dist_to_sun=0 < SUN_RADIUS+PLANET_MARGIN=10.1` → filter removes it. If it still fails, print `pa_lf.collect()` with just the sun filter to debug.

- [ ] **Step 3: Run full test suite**

```
python -m pytest tests/ -v
```

Expected: all pre-existing tests still PASS. The deleted functions (`is_crossing_sun_vectorized`, `_filter_blocked_attacks`) are only referenced inside `take_action` in file 62 — no other file imports them.

- [ ] **Step 4: Commit**

```
git add 62-One_angle_polars.py tests/test_62_polars_conversion.py
git commit -m "feat: convert 62-One_angle_polars take_action from pandas to Polars (1 lazy/collect)"
```
