# 114-precompute — Design Spec

## Goal

Modify `114-precompute.py` to output to `114-precompute/` with three changes:
1. Remove `attacks.parquet` (redundant with expanded reach)
2. Expand `reach.parquet` ships_sent to all powers of 2 `[1, 2, 4, ..., 1024]`
3. Improve `actions.parquet` target resolution using actual fleet trajectory from the replay instead of a static angle-to-nearest heuristic

## Output directory

`OUT_ROOT = Path("114-precompute")`

The `all_files` skip check drops `"attacks.parquet"`:
```python
all_files = ("df_s.parquet", "reach.parquet", "actions.parquet")
```

## reach.parquet

**Schema unchanged:** `id_src, step_src, id, step, ships_sent`

**Change:** `ships_sent` expands from `[4, 16, 64, 256]` to all powers of 2:
```python
REACH_POW2 = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
```

All source planets (no owner filter), collision-filtered via `_03_filter_collision`. No `ships_sent <= ships_min` constraint (that was attacks-only). The single geometry pass now uses `REACH_POW2` instead of `ATTACK_POW2`.

Since `reach` and `attacks` now use the same ships list, the `SHIPS_LIST` / `ATTACK_POW2` split is removed — one constant `REACH_POW2` drives both the `_02_get_reach` call and the `reach_df` filter.

## attacks.parquet

Removed entirely.

## actions.parquet — two-pass approach

**Schema unchanged:** `game_step, id_src, angle, ships_sent, id`

The current heuristic matches launch angle to the nearest planet's *current* position — it ignores orbital motion, so it misidentifies targets for orbiting planets. The replacement uses the replay's actual fleet positions and `swept_pair_hit` from the game engine.

### Pass 1 — data collection (existing step loop)

Alongside the existing `df_s` / `reach` processing, each step now also collects:

```
planet_snap: dict[int, dict[int, tuple[float, float, float]]]
    step -> {planet_id -> (x, y, radius)}
    Built from raw obs.planets (not from df_s — needs raw radius).

fleet_snap: dict[int, dict[int, tuple]]
    step -> {fleet_id -> (x, y, angle, from_planet_id, ships)}
    Built from obs.fleets.

raw_actions: list[tuple[int, int, float, int]]
    (game_step, id_src, angle, ships_sent) — recorded without target lookup.
```

### Pass 2 — target resolution (after all steps)

Build `fleet_life` from `fleet_snap`:
```
fleet_life: dict[fleet_id, {
    born:      int,   # first step fleet appears in obs
    from_pid:  int,
    angle:     float,
    ships:     int,
    last_step: int,   # last step fleet appears in obs
    last_x:    float,
    last_y:    float,
}]
```

For each `(game_step, id_src, angle, ships_sent)` in `raw_actions`:

1. **Match fleet**: find `fid` in `fleet_life` where `born == game_step + 1`, `from_pid == id_src`, `angle == action_angle`, `ships == ships_sent`.
2. **Check disappearance**: if `last_step + 1 >= total_steps`, the fleet survived to game end → `id_tgt = None`.
3. **Find target**: fleet was removed during step `last_step + 1`'s movement phase.
   - `speed = fleet_speed(ships_sent)`
   - `fleet_old = (last_x, last_y)`
   - `fleet_new = (last_x + speed*cos(angle), last_y + speed*sin(angle))`
   - For each planet `pid` in `planet_snap[last_step]`:
     - `p_old = planet_snap[last_step][pid][:2]`
     - `p_new = planet_snap[last_step + 1][pid][:2]` (if planet still exists at `last_step+1`)
     - `radius = planet_snap[last_step][pid][2]`
     - If `swept_pair_hit(fleet_old, fleet_new, p_old, p_new, radius)` → `id_tgt = pid`, break
   - If no planet matched (OOB or sun hit): `id_tgt = None`

`act_out` is built from resolved rows at the end of `_process_episode`, replacing the per-step `act_rows` accumulator.

### Edge cases

- A planet at `last_step` may not exist at `last_step+1` (comet expired). Skip those planets for `p_new`; use `p_old` as both positions or skip entirely.
- Multiple fleet matches (same from_pid, angle, ships in same born-step): iterate in fleet_id order; the game engine assigns IDs monotonically so the first match is correct.
- If no matching fleet is found (fleet was invalid / not spawned): `id_tgt = None`, still emit the action row.

## df_s.parquet

Unchanged.

## Implementation location

All changes are in `114-precompute.py`. `113-Polars_GNN_Corrected.py` is not modified.
