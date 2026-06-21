# 134-Polars_ships_src All-In Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create `134-Polars_ships_src.py` by copying `133-Polars_speed.py` and replacing the ships_sent expansion loop with a single all-in value (`ships_src = pl.first("ships")`), then simplifying scoring to the 72-style unified formula.

**Architecture:** Copy-and-patch — start from `133-Polars_speed.py`, apply five surgical edits. No new files, no new classes. The only behavioral difference is `ships_sent` is always the current ships at the source planet rather than an expanded range from 1 to ships_min+.

**Tech Stack:** Python 3.10+, Polars, math

## Global Constraints

- Do not modify `133-Polars_speed.py` — write only to `134-Polars_ships_src.py`
- `agent(obs)` must remain the entry point with identical signature
- `GameCache`, `interpreter`, `PhysicsEngine` are untouched
- `_03_filter_collision` is untouched

---

### Task 1: Copy file and apply structural changes to `_02_get_all_opportunities`

Changes: add `ships_src` to `mine_base_lf`; replace `int_ranges + explode` with single `with_columns`.

**Files:**
- Create: `134-Polars_ships_src.py` (copy of `133-Polars_speed.py`)

- [ ] **Step 1: Copy the source file**

```bash
cp "133-Polars_speed.py" "134-Polars_ships_src.py"
```

- [ ] **Step 2: Add `ships_src` to `mine_base_lf` aggregation**

In `_02_get_all_opportunities`, find the `mine_base_lf` `.agg([...])` block. It currently ends with:

```python
            pl.first("owner").alias("owner_src"),
            pl.len().alias("row_count"),
            pl.sum("is_mine").alias("is_mine"),
```

Add `ships_src` right after `ships_min`:

```python
            pl.first("ships").alias("ships_src"),
            pl.first("owner").alias("owner_src"),
            pl.len().alias("row_count"),
            pl.sum("is_mine").alias("is_mine"),
```

The full aggregation list becomes:
```python
        .agg([
            pl.first("step").alias("step_src"),
            pl.first("x").alias("x_src"),
            pl.first("y").alias("y_src"),
            pl.first("radius").alias("radius_src"),
            pl.min("ships").alias("ships_min"),
            pl.first("ships").alias("ships_src"),
            pl.first("production").alias("production_src"),
            pl.first("nature").alias("nature_src"),
            pl.first("owner").alias("owner_src"),
            pl.len().alias("row_count"),
            pl.sum("is_mine").alias("is_mine"),
        ])
```

- [ ] **Step 3: Replace the ships_sent expansion block**

Find and replace this block (the `# Ships_sent expansion` comment through the `.explode` call):

```python
        # Ships_sent expansion
        nb_steps_sim = GameConfig.NB_STEPS_SIM
        expanded_lf = (
            coarse_lf
            .with_columns(
                pl.int_ranges(
                    1,
                    pl.col("ships_min") + pl.col("production_src") * nb_steps_sim + 1,
                    dtype=pl.Int64,
                ).alias("ships_sent")
            )
            .explode("ships_sent")
        )
```

Replace with:

```python
        # All-in: always send current ships at source
        expanded_lf = coarse_lf.with_columns(pl.col("ships_src").alias("ships_sent"))
```

- [ ] **Step 4: Smoke-test the module imports and `_02` runs without crash**

Run from the project root:

```python
python - <<'EOF'
import types, math
import importlib.util, sys

spec = importlib.util.spec_from_file_location("agent134", "134-Polars_ships_src.py")
mod = importlib.util.load_from_spec(spec)
spec.loader.exec_module(mod)

# Minimal mock observation
obs = types.SimpleNamespace(
    planets=[[0, 0, 30.0, 50.0, 3.0, 50, 2], [1, 1, 70.0, 50.0, 3.0, 30, 2]],
    initial_planets=[[0, 0, 30.0, 50.0, 3.0, 50, 2], [1, 1, 70.0, 50.0, 3.0, 30, 2]],
    fleets=[],
    next_fleet_id=0,
    comets=[],
    comet_planet_ids=[],
    angular_velocity=0.02,
    player=0,
)
import polars as pl
df_s, planet_disp = mod.GameCache(obs, 0, 2, 0).build_df_s(obs, 0)
pa_lf = mod.StrategyPipeline._02_get_all_opportunities(df_s, planet_disp, 0)
print("columns:", pa_lf.columns)
print("ships_sent sample:", pa_lf.select("ships_sent").head(3).collect())
print("PASS")
EOF
```

Expected: prints `PASS` and `ships_sent` shows a single integer value (not a list/range). No exception.

- [ ] **Step 5: Commit**

```bash
git add 134-Polars_ships_src.py
git commit -m "feat: 134 — copy + ships_src all-in expansion (task 1)"
```

---

### Task 2: Simplify `_04_score_and_decide` to 72-style unified scoring

Replace the Conqueror / Supplier / Conqueror2 logic and simplify comet evasion.

**Files:**
- Modify: `134-Polars_ships_src.py` — `_04_score_and_decide` method

- [ ] **Step 1: Replace the entire body of `_04_score_and_decide`**

The current method spans from `def _04_score_and_decide(safe_lf, player_id)` through `return moves`. Replace it entirely with:

```python
    @staticmethod
    def _04_score_and_decide(safe_lf: pl.LazyFrame, player_id: int) -> list:
        attacks_with_angle = safe_lf.collect()
        if attacks_with_angle.is_empty():
            return []

        moves = []

        # Comet evasion — all-in flee (no ships_min guard)
        awa_comets = attacks_with_angle.filter(pl.col("nature_src") == "comet")
        if not awa_comets.is_empty():
            x_off = (awa_comets["x_src"] - GameConfig.CENTER).abs().max() or 0
            y_off = (awa_comets["y_src"] - GameConfig.CENTER).abs().max() or 0
            if max(x_off, y_off) > 45:
                moves += [list(r) for r in (
                    awa_comets
                    .sort(["ships_sent", "step"], descending=[True, False])
                    .group_by("id_src", maintain_order=True)
                    .first()
                    .select(["id_src", "final_angle", "ships_sent"])
                    .rows()
                )]
                id_to_avoid = awa_comets["id_src"].unique().to_list()
                attacks_with_angle = attacks_with_angle.filter(~pl.col("id_src").is_in(id_to_avoid))

        if attacks_with_angle.is_empty():
            return moves

        # Top-5 targets per source planet
        top5_ids = (
            attacks_with_angle
            .sort(["step", "ships_sent"])
            .group_by(["id_src", "id"], maintain_order=True)
            .first()
            .sort(["step", "ships_sent"])
            .group_by("id_src", maintain_order=True)
            .head(5)
            .select(["id_src", "id"])
            .with_columns(pl.lit(True).alias("is_top5"))
        )

        # Unified 72-style scoring
        attacks = (
            attacks_with_angle
            .with_columns(
                pl.when(pl.col("owner") == -1)
                .then(pl.col("ships"))
                .otherwise(pl.col("ships") + pl.col("production"))
                .alias("ships_needed")
            )
            .filter(pl.col("ships_needed") < pl.col("ships_sent"))
            .sort(["step", "ships_sent"])
            .group_by(["id_src", "id"], maintain_order=True)
            .first()
            .join(top5_ids, on=["id_src", "id"], how="left")
            .with_columns(pl.col("is_top5").fill_null(False))
            .with_columns(
                (pl.col("ships_needed") / pl.col("production_src")).alias("time_cost")
            )
            .with_columns(
                pl.col("time_cost").sum().over("id_src").alias("total_time_cost")
            )
            .with_columns(
                (
                    (pl.col("total_time_cost") - pl.col("time_cost") - pl.col("step_diff"))
                    * pl.col("production")
                    - pl.when(~pl.col("is_top5")).then(pl.lit(100.0)).otherwise(pl.lit(0.0))
                    - pl.when(pl.col("owner") == player_id).then(pl.lit(100.0)).otherwise(pl.lit(0.0))
                ).alias("score")
            )
            .filter(pl.col("score") > 0)
            .sort("score", descending=True)
            .group_by("id_src", maintain_order=True)
            .first()
        )

        if attacks.is_empty():
            return moves

        for row in attacks.iter_rows(named=True):
            print(f"From {row['id_src']}, To {row['id']} at step {row['step']} "
                  f"with {row['ships_sent']} ships (target has min {row['ships_min']})")

        moves += [list(r) for r in attacks.select(["id_src", "final_angle", "ships_sent"]).rows()]
        return moves
```

- [ ] **Step 2: Full agent smoke test**

```python
python - <<'EOF'
import types, importlib.util

spec = importlib.util.spec_from_file_location("agent134", "134-Polars_ships_src.py")
mod = importlib.util.load_from_spec(spec)
spec.loader.exec_module(mod)

obs = types.SimpleNamespace(
    planets=[[0, 0, 30.0, 50.0, 3.0, 50, 2], [1, 1, 70.0, 50.0, 3.0, 30, 2]],
    initial_planets=[[0, 0, 30.0, 50.0, 3.0, 50, 2], [1, 1, 70.0, 50.0, 3.0, 30, 2]],
    fleets=[],
    next_fleet_id=0,
    comets=[],
    comet_planet_ids=[],
    angular_velocity=0.02,
    player=0,
)

moves = mod.agent(obs)
print(f"moves: {moves}")
assert isinstance(moves, list), "agent must return a list"
if moves:
    assert all(len(m) == 3 for m in moves), "each move must be [from_id, angle, ships]"
print("PASS")
EOF
```

Expected: prints `PASS`. `moves` is a list (may be empty or contain valid `[from_id, angle, ships]` triples). No exception.

- [ ] **Step 3: Verify `ships_sent` in moves equals `ships_src` (all-in)**

If the smoke test above printed any moves, verify ships_sent matches the source planet's current ship count:

```python
python - <<'EOF'
import types, importlib.util

spec = importlib.util.spec_from_file_location("agent134", "134-Polars_ships_src.py")
mod = importlib.util.load_from_spec(spec)
spec.loader.exec_module(mod)

obs = types.SimpleNamespace(
    planets=[[0, 0, 30.0, 50.0, 3.0, 100, 2], [1, 1, 70.0, 50.0, 3.0, 5, 2]],
    initial_planets=[[0, 0, 30.0, 50.0, 3.0, 100, 2], [1, 1, 70.0, 50.0, 3.0, 5, 2]],
    fleets=[],
    next_fleet_id=0,
    comets=[],
    comet_planet_ids=[],
    angular_velocity=0.02,
    player=0,
)

moves = mod.agent(obs)
print(f"moves: {moves}")
# Source planet 0 has 100 ships — if it attacks, ships_sent must be 100
for m in moves:
    if m[0] == 0:
        assert m[2] == 100, f"expected ships_sent=100 (all-in), got {m[2]}"
print("PASS — all-in confirmed")
EOF
```

Expected: prints `PASS — all-in confirmed`.

- [ ] **Step 4: Commit**

```bash
git add 134-Polars_ships_src.py
git commit -m "feat: 134 — unified 72-style scoring, remove Conqueror/Supplier (task 2)"
```
