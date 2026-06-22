# 142 — First Minimax Then Scoring

**Date:** 2026-06-23
**File:** `142-First_Minimax_Then_Scoring.py`
**Base:** `140-BoardCache_optimised.py`

## Goal

Replace the full batched 2-ply minimax in `_04_minimax_search` with a lighter sequential
search that uses a net-production **delta** as the leaf evaluator instead of a ship-count tuple.

## What Changes vs 140

| Component | Change |
|---|---|
| `_02_get_all_opportunities` | Unchanged — already uses `ships_sent = ships_src` |
| `_03_filter_collision` | Unchanged |
| `_02_get_all_opportunities_batched` | Removed — not used |
| `_03_filter_collision_batched` | Removed — not used |
| `_04_minimax_search` | Replaced by `_04_first_minimax_then_scoring` |
| `_net_prod` helper | New — extracts `my_prod − opp_prod` from a horizon dict |
| `Board`, `agent()` | Unchanged except agent calls the new `_04` method |

## Helper: `_net_prod`

```python
@staticmethod
def _net_prod(horizon: dict, player_id: int) -> int:
    opponents = {v[1] for v in horizon.values() if v[1] not in (-1, player_id)}
    my_prod = sum(v[2] for v in horizon.values() if v[1] == player_id)
    if not opponents:
        return my_prod
    opp_prod = max(sum(v[2] for v in horizon.values() if v[1] == opp) for opp in opponents)
    return my_prod - opp_prod
```

Lives on `StrategyPipeline` alongside `_evaluate_dict`.

## `_04_first_minimax_then_scoring` Logic

### Inputs
- `safe_lf` — step-0 Polars LazyFrame from `_02+_03` (ships_sent = ships_src)
- `obs` — current observation
- `board` — Board instance

### Phase 0 — Comet evasion
Identical to `_04_minimax_search`: if a comet source is far from center (> 45 units),
launch it immediately and exclude its `id_src` from further search.

### Phase 1 — Step-0 candidates
Collect rows from `safe_lf.collect()` as `(src_id, tgt_id, step_tgt, angle, ships_sent)`.
Prepend a `None` sentinel representing "do nothing".

### Phase 2 — Per-candidate evaluation loop

For each candidate `c0`:

```
1. Build ship state:
   - c0 is None  → df_ships_c0 = df_ships_full  (no allocation)
   - c0 is move  → df_ships_c0 = board._recompute_from_sim(
                       df_ships_full, [src_id, angle, ships_sent], tgt_id, step_tgt)

2. Build step-5 opportunity set:
   (df_s5, pd5) = board.build_df_s_slice(df_ships_c0, step_from=board.step + 5)
   pa5_lf       = _02_get_all_opportunities(df_s5, pd5, player_id)
   safe5        = _03_filter_collision(pa5_lf).collect()

3. Score:
   horizon_c0  = board.extract_horizon_dict(df_ships_c0)
   baseline_np = _net_prod(horizon_c0, player_id)

   score_c0 = max(
       0,   # "do nothing at step 5" always available
       max over each row in safe5:
           _net_prod(
               _apply_sim_fleet(horizon_c0, (row.id_src, row.id, row.step, row.ships_sent)),
               player_id
           ) - baseline_np
   )
```

### Phase 3 — Select & return

```
best_c0 = argmax(score_c0)
if best_c0 is None:   return moves_out          (comet moves only / do nothing)
else:                 return moves_out + [[src_id, angle, ships_sent]]
```

## Scoring Semantics

`score_c0` measures: *"if I make step-0 move c0, how much can I improve my net
production at the next reachable step-5 window?"*

- Capturing a neutral planet with production P → delta = +P
- Capturing an opponent's planet with production P → delta = +2P (we gain P, they lose P)
- No capturable planet at step 5 → delta = 0

The `do-nothing` sentinel always scores ≥ 0, so a step-0 attack only wins when it
strictly opens a better step-5 window than the baseline.

## Removed Complexity vs 140

- No batched `c0_id` tagging
- No `build_df_s_slice_batched` / `build_df_s_slice` with stacked slices
- No `best_c5_base` carry-over logic
- No ship-count tuple comparison `(my_prod - opp_prod, my_ships - opp_ships)`

## Smoke Test (main block)

Same structure as `140`: construct `_make_obs()`, run `agent()` twice (step 0 and step 1),
assert result is a list of valid 3-element moves, assert elapsed < 5 s.
