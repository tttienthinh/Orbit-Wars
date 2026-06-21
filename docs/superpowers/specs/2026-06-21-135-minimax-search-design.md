# 135 — Minimax Search Agent Design

**Date:** 2026-06-21
**File:** `135-Polars_mini_max.py` (copy of `134-Polars_ships_src.py` with `_04` replaced)

---

## Goal

Replace the greedy single-step scorer (`_04_score_and_decide`) with a 2-level lookahead search that maximises `(net_production, net_ships)` lexicographically, where:

```
net_production = my_production − max(opponent_production)
net_ships      = my_ships      − max(opponent_ships)
```

`my_ships` / `opponent_ships` include ships on owned planets, comets, and in-flight fleets.

---

## Decision Structure

The 10-step simulation window is split into two decision points and two simulation corridors:

```
Step 0:   I choose one move (or do nothing)
Steps 1–4: pure simulation (no new actions)
Step 5:   I choose one move (or do nothing)
Steps 6–9: pure simulation (no new actions)
Step 10:  evaluate (net_production, net_ships)
```

**Opponents do nothing** in this version (they hold their current fleets and produce ships normally, but issue no new moves).

---

## Architecture

Everything in `134` is unchanged except `_04_score_and_decide`, which becomes `_04_minimax_search`.

### `_04_minimax_search(safe_lf, obs, cache, player_id) -> list`

Orchestrates the search and returns the best step-0 move list (zero or one move).

#### Phase 1 — Step-0 candidates + "do nothing" base cache

1. Collect `attacks_with_angle` from `safe_lf` (the existing `_02`/`_03` output). Each row is a candidate step-0 individual move `(id_src, final_angle, ships_sent)`. Add an explicit "do nothing" candidate.
2. Run the **base simulation**: deep copy obs, run `interpreter` × 4 with no actions (steps 1–4). This produces `obs_base5` (state at step 5 if no step-0 move is made).
3. Build the **step-5 base cache**: run `build_df_s(obs_base5, step=5)` with `NB_STEPS_SIM=5`, then run `_02_get_all_opportunities` + `_03_filter_collision` on the result. Cache the output DataFrame keyed by `id_src`.

#### Phase 2 — Per step-0 candidate evaluation

For each step-0 candidate `c0` (including "do nothing"):

1. **Simulate to step 5**: deep copy obs, apply `c0` via `interpreter` at step 0 (passing `[[c0]]` as the action for player_id, `[]` for others), then run `interpreter` × 4 with no actions.
2. **Detect changed planets**: compare `obs_c5.planets` against `obs_base5.planets`. A planet is "changed" if its `owner` or `ships` count differs. Collect the set of changed `id_src` values among my planets.
3. **Build merged step-5 candidate table**:
   - Start from the cached base table.
   - Drop rows where `id_src` is in the changed set.
   - Re-run `build_df_s(obs_c5, step=5)` + `_02` + `_03` only for planets in the changed set; append those rows.
4. **Evaluate step-5 leaves**: for each step-5 candidate `c5` (plus "do nothing"):
   - Deep copy `obs_c5`, apply `c5`, run `interpreter` × 4 (steps 6–9).
   - Call `evaluate(obs_leaf, player_id)` → `(net_prod, net_ships)`.
5. **Best step-5 score**: `max` over all `c5` candidates using lexicographic `(net_prod, net_ships)` comparison.

#### Phase 3 — Pick best step-0 action

Pick the step-0 candidate `c0` with the highest best-step-5 score (lexicographic). Return it as a move list (empty list for "do nothing", otherwise `[[id_src, final_angle, ships_sent]]`).

---

## Evaluation Function

```python
def evaluate(obs, player_id: int) -> tuple[float, float]:
    planets = obs.planets if hasattr(obs, "planets") else obs["planets"]
    fleets  = obs.fleets  if hasattr(obs, "fleets")  else obs["fleets"]

    opponents = {p[1] for p in planets if p[1] not in (-1, player_id)}
    opponents |= {f[1] for f in fleets  if f[1] not in (-1, player_id)}

    my_prod  = sum(p[6] for p in planets if p[1] == player_id)
    my_ships = (sum(p[5] for p in planets if p[1] == player_id)
              + sum(f[6] for f in fleets  if f[1] == player_id))

    if not opponents:
        return (my_prod, my_ships)

    opp_prod  = max(sum(p[6] for p in planets if p[1] == opp) for opp in opponents)
    opp_ships = max(
        sum(p[5] for p in planets if p[1] == opp)
      + sum(f[6] for f in fleets  if f[1] == opp)
        for opp in opponents
    )
    return (my_prod - opp_prod, my_ships - opp_ships)
```

---

## Simulation Helper

```python
def _simulate(obs, move: list | None, n_steps: int, start_step: int,
              num_agents: int, player_id: int):
    """Deep-copy obs, optionally apply one move at start_step, then sim n_steps."""
    import copy
    sim = copy.deepcopy(obs)
    actions = [[] for _ in range(num_agents)]
    if move is not None:
        actions[player_id] = [move]
    interpreter(sim, actions, start_step, num_agents)
    no_actions = [[] for _ in range(num_agents)]
    for k in range(1, n_steps):
        interpreter(sim, no_actions, start_step + k, num_agents)
    return sim
```

---

## Step-5 Candidate Generation

`build_df_s` is called with the step-5 obs and a local constant `NB_STEPS_5 = 5` so it snapshots steps 5–10. Implement a module-level `build_df_s_n(cache, obs, current_step, nb_steps)` wrapper that mirrors `build_df_s` but uses `nb_steps` instead of `GameConfig.NB_STEPS_SIM`. The existing `_02`/`_03` pipeline runs on this smaller frame unchanged (it just sees fewer rows).

For the incremental update (changed planets only), after building the full step-5 `df_s` from the changed-state obs, filter the `_02` cross-join to only rows where `id_src` is in the changed set before merging with the base cache.

---

## "Do Nothing" at Step 5

Always include "do nothing" as a step-5 candidate. It evaluates the natural outcome of the step-0 action without any further intervention.

---

## Comet Evasion

Preserve the existing comet-evasion logic from `_04_score_and_decide`: if a comet source planet is far from centre, issue an all-in flee move at step 0 and skip the minimax for that planet.

---

## Complexity Estimate

- Step-0 candidates: ~10 (one per owned planet) + 1 "do nothing" = ~11
- Step-5 candidates per branch: ~10 + 1 = ~11
- Leaf evaluations: 11 × 11 = ~121
- Per leaf: 4 interpreter calls + deepcopy
- Changed-planet re-computation: typically 1–2 planets per branch

Expected wall time: well under 1 second.

---

## What Does Not Change

- `GameCache`, `_planet_pos_analytical`, `_fill_pos_window`, `advance`, `build_df_s`
- `_02_get_all_opportunities`, `_03_filter_collision`
- `interpreter`, `PhysicsEngine`, `GameConfig`
- `agent()` entry point (calls `_04_minimax_search` instead of `_04_score_and_decide`)
