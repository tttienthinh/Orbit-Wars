# Per-Step Ships-Sent Expansion — Design Spec

**Date:** 2026-06-07  
**File:** `100-Rewrite_physics.ipynb` (new notebook, StrategyPipeline copied from `81-Simulate10Next.py`)

---

## Goal

Replace the single-step `ships_sent` expansion in `_02_get_all_opportunities` with a per-launch-step model: for each step `s` from current to current+10, a fleet may be launched from a source planet with up to `ships_min_s` ships, where `ships_min_s = min(ships_src at steps s, s+1, …, 10)`.

This removes the artificial `ships_min + production_src * nb_steps_sim` ceiling and correctly pairs launch time with arrival time.

---

## Column Naming Convention

All columns use consistent `_src` / `_tgt` suffixes throughout the pipeline.

| Side | Columns |
|------|---------|
| Source (launch) | `id_src`, `step_src`, `x_src`, `y_src`, `radius_src`, `ships_min`, `production_src`, `nature_src`, `owner_src` |
| Target (arrival) | `id_tgt`, `step_tgt`, `x_tgt`, `y_tgt`, `radius_tgt`, `ships_tgt`, `production_tgt`, `owner_tgt`, `nature_tgt` |
| Derived | `step_diff = step_tgt - step_src`, `dist_tgt_src` |

---

## Section 1 — `mine_per_step` (replaces `mine_base`)

Instead of aggregating source planets to one row each, produce one row per `(source_planet, step_s)` for planets that are always owned by `player_id` across all simulated steps.

```python
always_mine_ids = (
    df_s.groupby("id")
    .filter(lambda g: (g["owner"] == player_id).all())
    ["id"].unique()
)

mine_per_step = (
    df_s
    .loc[df_s["id"].isin(always_mine_ids)]
    .sort_values(["id", "step"])
    .assign(
        ships_min=lambda d: d.groupby("id")["ships"]
                             .transform(lambda s: s[::-1].cummin()[::-1])
    )
    .rename(columns={
        "id": "id_src",
        "step": "step_src",
        "x": "x_src",
        "y": "y_src",
        "radius": "radius_src",
        "production": "production_src",
        "nature": "nature_src",
        "owner": "owner_src",
    })
)
```

`ships_min` at row `(id_src, step_src)` = minimum ships at `id_src` from `step_src` through the end of simulation. This is the safe ship budget for a fleet launched at `step_src`.

---

## Section 2 — Cross-Join

```python
coarse = (
    mine_per_step.assign(_key=1)
    .merge(
        df_s.rename(columns={
            "id": "id_tgt", "step": "step_tgt",
            "x": "x_tgt", "y": "y_tgt",
            "radius": "radius_tgt", "ships": "ships_tgt",
            "production": "production_tgt",
            "owner": "owner_tgt", "nature": "nature_tgt",
        }).assign(_key=1),
        on="_key"
    )
    .drop(columns="_key")
    .loc[lambda d: (d["step_tgt"] > d["step_src"]) & (d["id_tgt"] != d["id_src"])]
    .assign(
        step_diff=lambda d: (d["step_tgt"] - d["step_src"]).astype(float),
        dist_tgt_src=lambda d: np.sqrt(
            (d["x_tgt"] - d["x_src"]) ** 2 + (d["y_tgt"] - d["y_src"]) ** 2
        ),
    )
)
```

`step_diff = step_tgt - step_src` is the fleet travel time (same semantics as before, now accurate across launch steps). Sun-crossing filter, swept-pair collision, and angle geometry downstream are unchanged — they operate on per-row `(x_src, y_src, x_tgt, y_tgt)`.

Row count scales as `N_src_steps × N_tgt_steps` but distance/speed filters prune aggressively.

---

## Section 3 — Ships_sent Expansion

```python
expanded = (
    coarse
    .assign(
        ships_sent=lambda d: [
            list(range(1, int(sm) + 1))
            for sm in d["ships_min"]
        ]
    )
    .explode("ships_sent")
    .assign(ships_sent=lambda d: d["ships_sent"].astype("int64"))
)
```

No `production_src * nb_steps_sim` term. The upper bound is exactly `ships_min` — safe to commit at `step_src`.

---

## Downstream Impact

- `_03_filter_collision`: rename `id` → `id_tgt`, `step` → `step_tgt` in the join keys. Logic unchanged.
- `_04_score_and_decide`: rename `owner` → `owner_tgt`, `ships` → `ships_tgt`, `production` → `production_tgt`. Logic unchanged.
- `planet_disp` join: uses `id_tgt` + `step_tgt` as join keys (renamed from `id` + `step`).
- `prev_pos` join for swept-pair: uses `id_tgt` + `step_tgt` (renamed).

---

## Scope

- Changes are contained to `_02_get_all_opportunities` and minor renames in `_03` / `_04`.
- `_01_get_obs_dataframe` is unchanged.
- `interpreter` and `PhysicsEngine` are unchanged.
- Work happens in `100-Rewrite_physics.ipynb` — `81-Simulate10Next.py` is the reference, not modified.
