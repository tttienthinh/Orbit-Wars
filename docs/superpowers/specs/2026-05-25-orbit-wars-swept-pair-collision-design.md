# Swept-Pair Collision Update — `66-One_angle_polars_updated.py`

**Date:** 2026-05-25  
**Context:** kaggle-environments 1.29.1 fixed a fleet-tunneling bug in the orbit_wars interpreter using continuous swept-pair collision detection. This spec updates `66-One_angle_polars_updated.py` to match.

---

## Background

### The bug (fixed in 1.29.1)

The old interpreter moved fleets first (treating planets as static), then rotated planets (sweeping through stationary fleets). Fast fleets could skip through a rotating planet entirely within one tick — 13 such events were observed in a single 339-step game.

### The fix

`swept_pair_hit(A, B, P0, P1, r)` solves the quadratic `|R(t)|² ≤ r²` where `R(t) = (A-P0) + t*((B-A)-(P1-P0))` is the fleet's position relative to the planet at fractional tick time `t ∈ [0,1]`. Both objects are treated as moving linearly across the tick.

---

## Scope

File to modify: `66-One_angle_polars_updated.py`  
Reference: `kaggle_env_orbit_wars/1_29_3.py`

---

## Part 1 — Interpreter

### Changes

1. **Add** `swept_pair_hit(A, B, P0, P1, r)` — copy verbatim from `1_29_3.py` (lines 46–64).

2. **Before the fleet loop** — precompute `planet_paths: dict[pid → (old_pos, new_pos)]`:
   - Static planets: `old_pos == new_pos == (planet[2], planet[3])`
   - Rotating planets (`r + radius < ROTATION_RADIUS_LIMIT`): compute `new_pos` using `initial_angle + angular_velocity * step`
   - Comets: skip (handled separately)

3. **Inside the fleet loop** — replace:
   ```python
   # OLD
   if point_to_segment_distance(planet_pos, old_pos, new_pos) < planet[4]:
   ```
   with:
   ```python
   # NEW
   p_old, p_new = planet_paths[planet[0]]
   if swept_pair_hit(fleet_old, fleet_new, p_old, p_new, planet[4]):
   ```

4. **After the fleet loop** — apply precomputed positions:
   ```python
   for planet in obs0.planets:
       if planet[0] in planet_paths:
           planet[2], planet[3] = planet_paths[planet[0]][1]
   ```

5. **Remove** `sweep_fleets()` helper — its role is now absorbed by `swept_pair_hit` inside the fleet loop (which accounts for planet movement, handling both fleet-hits-planet and planet-sweeps-fleet).

---

## Part 2a — t1, t2 in `take_action`

### Prev-pos join

Add a shifted lookup table so each `(id, step)` row can access the planet's position at `step-1`:

```python
prev_pos_lf = (
    df_lf.select(["id", "step", "x", "y"])
    .rename({"x": "x_prev", "y": "y_prev"})
    .with_columns((pl.col("step") + 1).alias("step"))
)
```

Join into the cross-join pipeline on `["id", "step"]`.

### Swept-pair coefficients (Polars expressions)

Fleet velocity vector (unit direction scaled by fleet_speed):

```
unit_x = dx / dist_tgt_src
unit_y = dy / dist_tgt_src
fleet_vx = unit_x * fleet_speed
fleet_vy = unit_y * fleet_speed
```

Planet velocity vector (displacement per tick from prev-pos join):

```
planet_vx = x - x_prev
planet_vy = y - y_prev
```

Fleet position at start of tick (t=0) relative to planet start:

```
d0x = (x_src + unit_x * dist_min) - x_prev
d0y = (y_src + unit_y * dist_min) - y_prev
```

Relative velocity:

```
dvx = fleet_vx - planet_vx
dvy = fleet_vy - planet_vy
```

Quadratic coefficients:

```
a = dvx² + dvy²
b = 2*(d0x*dvx + d0y*dvy)
c = d0x² + d0y² - radius²
disc = b² - 4*a*c
```

### t1, t2 and collision flag

Normal case (`a ≥ 1e-12`):

```
sq = sqrt(max(disc, 0))
t1 = (-b - sq) / (2*a)
t2 = (-b + sq) / (2*a)
collision = (disc ≥ 0) & (t2 ≥ 0) & (t1 ≤ 1)
```

Degenerate case (`a < 1e-12`, relative velocity ≈ zero):

```
collision = (c ≤ 0)
t1 = 0.0,  t2 = 1.0
```

Use `pl.when(a < 1e-12).then(...).otherwise(...)` to branch in Polars.

Replace the current `dist_min/dist_max` collision expression entirely.

### Contact-time clipping

Before computing planet positions at contact:

```
t1_eff = clip(t1, 0.0, 1.0)
t2_eff = clip(t2, 0.0, 1.0)
```

---

## Part 2b — Cone from t1_eff, t2_eff

### Planet position at contact

```
p_t1_x = x_prev + t1_eff * (x - x_prev)
p_t1_y = y_prev + t1_eff * (y - y_prev)
p_t2_x = x_prev + t2_eff * (x - x_prev)
p_t2_y = y_prev + t2_eff * (y - y_prev)
```

### Aim angles

```
angle_t1 = arctan2(p_t1_y - y_src, p_t1_x - x_src)
angle_t2 = arctan2(p_t2_y - y_src, p_t2_x - x_src)
```

### angle_radius at each contact point

```
d_s_t1 = sqrt((p_t1_x - x_src)² + (p_t1_y - y_src)²)
d_f_t1 = dist_min + t1_eff * fleet_speed

angle_radius_t1 = arccos(clip(
    (d_s_t1² + d_f_t1² - radius²) / (2 * d_s_t1 * d_f_t1),
    -1.0, 1.0
))
```

Same for t2_eff.

### Cone bounds

```
angle_min = min(angle_t1 - angle_radius_t1, angle_t2 - angle_radius_t2) % (2π)
angle_max = max(angle_t1 + angle_radius_t1, angle_t2 + angle_radius_t2) % (2π)
```

### Aim angle (replaces `arctan2(dy, dx)`)

```
final_angle = (angle_t1 + angle_t2) / 2
```

### Blocking (unchanged structure)

The `blocked_lf` self-join and `wraps` + `in_cone` logic remain structurally identical. The only change: `angle` used in `in_cone` is now `final_angle` (midpoint of contact arc) instead of the raw `arctan2` direction.

---

## What does NOT change

- `_simulate` structure (still calls interpreter tick-by-tick)
- Section 4 (scoring, `ships_needed`, `time_cost`)
- Section 5 (comet handling)
- `blocked_lf` join structure and `wraps/in_cone` logic
- `nearest_planet_sniper` agent wrapper

---

## Invariants

- `swept_pair_hit` is pure and stateless — safe to call per fleet per planet
- t1 ≤ t2 always holds when `disc ≥ 0` and `a > 0`
- `t1_eff, t2_eff ∈ [0, 1]` by construction
- `angle_min ≤ angle_max` in the non-wrapping case; `wraps` flag handles the straddling case
