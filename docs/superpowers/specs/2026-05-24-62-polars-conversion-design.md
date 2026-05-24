# Design: Convert 62-One_angle_polars.py from pandas to Polars

**Date:** 2026-05-24
**File:** `62-One_angle_polars.py`
**Reference:** `55-Polars_lazy_optimised.py`

---

## Goal

Convert `take_action` in `62-One_angle_polars.py` from pandas to Polars, using a single
`.lazy()` / `.collect()` pair. Reduce lazy/collect calls compared to `55-Polars_lazy_optimised.py`
(which has 3–4 collects). Keep 62's simpler blocking logic (cross-merge cone check) rather than
55's interval-subtraction `IntervalProcessorPolarsOptimized`.

---

## Scope

- **Changed:** `take_action` function and its two helpers (`is_crossing_sun_vectorized`,
  `_filter_blocked_attacks`)
- **Unchanged:** `_simulate` (stays pandas, returns `pd.DataFrame`), `interpreter`,
  all physics helpers, agent entry point

---

## Key Decisions

| Decision | Choice | Reason |
|---|---|---|
| `_simulate` return type | Keep `pd.DataFrame` | Limits change to `take_action` only |
| Lazy/collect count | 1 lazy, 1 collect | Cross-join is ~350k rows; lazy enables pipelining |
| CSE for self-join | Default (on in Polars 1.35.2) | `pa_lf` used 3× in DAG; no explicit flag needed |
| `if pairs.empty` guard | Removed | Anti-join on empty `blocked_lf` returns all rows |
| `is_crossing_sun_vectorized` | Inlined as Polars expressions | Eliminates numpy dependency in hot path |
| `_filter_blocked_attacks` | Inlined as lazy branches | No reason to keep as separate function |
| Top-5 computation | After comet removal | Avoids null rows in left join |

---

## Architecture: Single Lazy Pipeline

```
pl.from_pandas(df).lazy()                          ← entry, one lazy()
  mine_lf:  group_by + agg + filter + explode
  pa_lf:    cross-join × df_lf
            filter (step > step_src, id != id_src)
            with_columns (collision geometry)
            filter (collision)
            filter (~crossing_sun)                 ← inlined Polars expr
            with_columns (angle, angle_min/max)
            sort("step")
  blocked_lf: pa_lf self-join (inner)              ← CSE caches pa_lf
              filter (step_obs < step, cone check)
              unique
  pa_lf.join(blocked_lf, how="anti")              ← pa_lf reused (CSE)
       .with_columns(final_angle = angle)
       .collect()                                   ← THE ONE collect()

comet handling  (eager, small frame)
scoring + top-5 (eager, small frame)
return moves
```

---

## Section 1: Entry Point

```python
def take_action(df: pd.DataFrame, player_id: int,
                nb_steps_sim: int = NB_STEPS_SIM, return_df: bool = False):
    df_lf = pl.from_pandas(df).lazy()
```

---

## Section 2: Mine Analysis + Expansion (lazy)

```python
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
```

---

## Section 3: Cross-Join + Collision + Sun (lazy, inlined expressions)

Shared geometry expressions defined before the pipeline so they can be reused:

```python
dx = pl.col("x") - pl.col("x_src")
dy = pl.col("y") - pl.col("y_src")
l2 = dx.pow(2) + dy.pow(2)
dist_tgt_src = l2.sqrt()
step_diff    = pl.col("step") - pl.col("step_src")
fleet_speed  = 1.0 + (MAX_SPEED - 1.0) * (
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

---

## Section 4: Blocking Self-Join + Collect

`pa_lf` appears 3× in the DAG. CSE (on by default in Polars 1.35.2) caches it.
`if pairs.empty` guard removed — anti-join on empty `blocked_lf` is a no-op.

```python
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
```

---

## Section 5: Comet Handling + Scoring (eager)

```python
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

---

## Removed

- `is_crossing_sun_vectorized()` — replaced by inline Polars expressions in Section 3
- `_filter_blocked_attacks()` — inlined as lazy branches in Section 4

---

## Lazy/Collect Count Comparison

| Version | lazy() calls | collect() calls |
|---|---|---|
| 55-Polars_lazy_optimised | 2 | 3–4 |
| 62-One_angle_polars (this) | 1 | 1 |
