# 135 Minimax Search Agent — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Copy `134-Polars_ships_src.py` to `135-Polars_mini_max.py` and replace the greedy scorer with a 2-level lookahead search that picks the step-0 move maximising `(net_production, net_ships)` at step 10.

**Architecture:** Two decision points (step 0 and step 5) separated by 4-step pure-simulation corridors. Opponents do nothing. A "do nothing" base simulation at step 0 pre-builds the step-5 candidate table; non-trivial step-0 moves only recompute candidates for planets whose ownership or ship count changed.

**Tech Stack:** Python 3.11+, Polars, `copy.deepcopy`, existing `interpreter` / `GameCache` / `StrategyPipeline._02` / `_03`.

## Global Constraints

- Single file: `135-Polars_mini_max.py` — no new modules.
- Opponents issue zero new moves in the search tree.
- Step-0 candidates capped at top-5 per source planet (same as original `_04`).
- Step-5 candidates: every valid (src, tgt) pair after simulating the step-0 choice.
- Lexicographic tuple comparison decides all ties: `(net_production, net_ships)`.
- `interpreter` is called as `interpreter(sim, actions, game_step, num_agents)` and mutates `sim` in place.

---

### Task 1: Copy file and wire stub

**Files:**
- Create: `135-Polars_mini_max.py` (copy of `134-Polars_ships_src.py`)

**Interfaces:**
- Produces: `agent(obs) -> list` — same signature as 134

- [ ] **Step 1: Copy the file**

```bash
cp "134-Polars_ships_src.py" "135-Polars_mini_max.py"
```

- [ ] **Step 2: Replace `_04_score_and_decide` call in `agent()` with a stub**

In `135-Polars_mini_max.py`, find the `agent()` function (line ~909) and change:

```python
    moves = StrategyPipeline._04_score_and_decide(safe_lf, CACHE.player_id)
```

to:

```python
    moves = StrategyPipeline._04_minimax_search(safe_lf, obs, CACHE, CACHE.player_id)
```

- [ ] **Step 3: Add the stub method to `StrategyPipeline`**

Add this method directly after `_03_filter_collision` (around line 538):

```python
    @staticmethod
    def _04_minimax_search(safe_lf: pl.LazyFrame, obs, cache, player_id: int) -> list:
        # stub — replace in Task 5
        return []
```

- [ ] **Step 4: Verify the file imports cleanly**

```bash
python -c "import importlib.util, sys; spec = importlib.util.spec_from_file_location('m', '135-Polars_mini_max.py'); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); print('OK')"
```

Expected output: `OK`

- [ ] **Step 5: Commit**

```bash
git add 135-Polars_mini_max.py
git commit -m "feat: 135 — copy 134 and wire _04_minimax_search stub"
```

---

### Task 2: `evaluate()` module-level function

**Files:**
- Modify: `135-Polars_mini_max.py` — add `evaluate()` after the `interpreter` function (around line 270, before `StrategyPipeline`)

**Interfaces:**
- Consumes: mutated obs object (has `.planets` and `.fleets` list attributes), `player_id: int`
- Produces: `evaluate(obs, player_id) -> tuple[float, float]` — `(net_production, net_ships)`

- [ ] **Step 1: Add `evaluate()` in `135-Polars_mini_max.py`**

Insert this block after the `interpreter` function and before the `StrategyPipeline` class:

```python
def evaluate(obs, player_id: int) -> tuple:
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

- [ ] **Step 2: Verify with an inline smoke test**

Run this one-liner to confirm the function works with a synthetic obs:

```bash
python -c "
import importlib.util, sys, types
spec = importlib.util.spec_from_file_location('m', '135-Polars_mini_max.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

obs = types.SimpleNamespace()
# planets: [id, owner, x, y, radius, ships, production]
obs.planets = [
    [0, 0, 10.0, 10.0, 2.0, 50, 3],   # mine
    [1, 1, 90.0, 90.0, 2.0, 30, 2],   # opponent
]
obs.fleets = [
    [0, 0, 15.0, 15.0, 0.0, 0, 10],   # my fleet
    [1, 1, 85.0, 85.0, 0.0, 1, 5],    # opponent fleet
]
score = m.evaluate(obs, player_id=0)
print(score)
assert score == (3 - 2, (50 + 10) - (30 + 5)), f'got {score}'
print('evaluate OK')
"
```

Expected output: `(1, 25)` then `evaluate OK`

- [ ] **Step 3: Commit**

```bash
git add 135-Polars_mini_max.py
git commit -m "feat: 135 — add evaluate() function"
```

---

### Task 3: `_simulate()` module-level helper

**Files:**
- Modify: `135-Polars_mini_max.py` — add `_simulate()` after `evaluate()`

**Interfaces:**
- Consumes: `obs` (game object with `.planets`, `.fleets`, etc.), `move: list | None` (e.g. `[id_src, angle, ships]`), `n_steps: int`, `start_step: int`, `num_agents: int`, `player_id: int`
- Produces: `_simulate(...) -> obs_sim` — mutated deep copy after `n_steps` interpreter steps

- [ ] **Step 1: Add `_simulate()` in `135-Polars_mini_max.py`**

Insert this block immediately after `evaluate()`:

```python
def _simulate(obs, move, n_steps: int, start_step: int,
              num_agents: int, player_id: int):
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

- [ ] **Step 2: Verify with a smoke test**

```bash
python -c "
import importlib.util, sys, types, math
spec = importlib.util.spec_from_file_location('m', '135-Polars_mini_max.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

obs = types.SimpleNamespace()
obs.planets = [[0, 0, 10.0, 10.0, 2.0, 100, 3], [1, 1, 90.0, 90.0, 2.0, 50, 2]]
obs.initial_planets = list(obs.planets)
obs.fleets = []
obs.next_fleet_id = 0
obs.comets = []
obs.comet_planet_ids = []
obs.angular_velocity = 0.0

sim = m._simulate(obs, None, 5, 0, 2, 0)
# After 5 steps, my planet (owner=0) should have gained 3*5=15 ships
my_planet = next(p for p in sim.planets if p[1] == 0)
assert my_planet[5] == 115, f'expected 115 ships, got {my_planet[5]}'
print('_simulate OK')
"
```

Expected output: `_simulate OK`

- [ ] **Step 3: Commit**

```bash
git add 135-Polars_mini_max.py
git commit -m "feat: 135 — add _simulate() helper"
```

---

### Task 4: `build_df_s_n()` variable-length simulation wrapper

**Files:**
- Modify: `135-Polars_mini_max.py` — add `build_df_s_n()` after `_simulate()`

**Interfaces:**
- Consumes: `cache: GameCache` (for `_planet_pos`, `_planet_meta`, `_fleet_arrival`, `_compute_fleet_arrival`), `obs` (step-N game object), `current_step: int`, `nb_steps: int`
- Produces: `build_df_s_n(cache, obs, current_step, nb_steps) -> tuple[pl.DataFrame, pl.DataFrame]` — `(df_s, planet_disp)` exactly as `GameCache.build_df_s` but running for `nb_steps` instead of `NB_STEPS_SIM`

- [ ] **Step 1: Add `build_df_s_n()` in `135-Polars_mini_max.py`**

Insert this block after `_simulate()`. It is a copy of `GameCache.build_df_s` body with `GameConfig.NB_STEPS_SIM` replaced by `nb_steps` and made into a module-level function:

```python
def build_df_s_n(cache, obs, current_step: int, nb_steps: int):
    planets = obs.planets if hasattr(obs, "planets") else obs["planets"]
    fleets  = obs.fleets  if hasattr(obs, "fleets")  else obs["fleets"]

    ship_state  = {p[0]: p[5] for p in planets}
    owner_state = {p[0]: p[1] for p in planets}
    production  = {p[0]: p[6] for p in planets}
    radius_map  = {p[0]: p[4] for p in planets}

    arrivals_by_step = {}
    for fleet in fleets:
        fid     = fleet[0]
        arrival = cache._fleet_arrival.get(fid)
        if arrival is None:
            arrival = cache._compute_fleet_arrival(fleet, current_step)
        if arrival is not None:
            planet_id, arrival_step = arrival
            arrivals_by_step.setdefault(arrival_step, []).append((fleet, planet_id))

    rows = []
    for k in range(nb_steps + 1):
        game_step = current_step + k
        for pid in list(ship_state.keys()):
            pos = cache._planet_pos(pid, game_step)
            if pos is None:
                continue
            x, y = pos
            meta = cache._planet_meta.get(pid)
            if meta is None:
                continue
            rows.append({
                "step": game_step,
                "id": pid,
                "x": x,
                "y": y,
                "radius": radius_map[pid],
                "ships": ship_state[pid],
                "production": production[pid],
                "owner": owner_state[pid],
                "nature": meta["nature"],
            })

        if k == nb_steps:
            break

        new_ships = dict(ship_state)
        new_owner = dict(owner_state)
        for pid, owner in owner_state.items():
            if owner != -1:
                new_ships[pid] += production[pid]

        planet_arrivals = {}
        for fleet, planet_id in arrivals_by_step.get(game_step, []):
            if planet_id in ship_state:
                planet_arrivals.setdefault(planet_id, []).append(fleet)

        for planet_id, fleet_list in planet_arrivals.items():
            player_ships = {}
            for fleet in fleet_list:
                fowner = fleet[1]
                player_ships[fowner] = player_ships.get(fowner, 0) + fleet[6]
            sorted_players = sorted(player_ships.items(), key=lambda x: x[1], reverse=True)
            top_player, top_ships = sorted_players[0]
            if len(sorted_players) > 1:
                second_ships = sorted_players[1][1]
                survivor_ships = 0 if sorted_players[0][1] == sorted_players[1][1] else top_ships - second_ships
                survivor_owner = top_player if survivor_ships > 0 else -1
            else:
                survivor_owner = top_player
                survivor_ships = top_ships
            if survivor_ships > 0:
                if new_owner[planet_id] == survivor_owner:
                    new_ships[planet_id] += survivor_ships
                else:
                    new_ships[planet_id] -= survivor_ships
                    if new_ships[planet_id] < 0:
                        new_owner[planet_id] = survivor_owner
                        new_ships[planet_id] = abs(new_ships[planet_id])

        ship_state  = new_ships
        owner_state = new_owner

    df_s = pl.DataFrame(rows).sort("step")
    prev_pos = (
        df_s.lazy()
        .select(["id", "step", "x", "y"])
        .rename({"x": "x_prev", "y": "y_prev"})
        .with_columns((pl.col("step") + 1).alias("step"))
    )
    planet_disp = (
        df_s.lazy()
        .select(["id", "step", "x", "y"])
        .join(prev_pos, on=["id", "step"], how="left")
        .with_columns(
            ((pl.col("x") - pl.col("x_prev").fill_null(pl.col("x"))).pow(2) +
             (pl.col("y") - pl.col("y_prev").fill_null(pl.col("y"))).pow(2)
            ).sqrt().alias("planet_disp")
        )
        .select(["id", "step", "planet_disp"])
        .collect()
    )
    return df_s, planet_disp
```

- [ ] **Step 2: Verify the file still imports cleanly**

```bash
python -c "
import importlib.util
spec = importlib.util.spec_from_file_location('m', '135-Polars_mini_max.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
print(hasattr(m, 'build_df_s_n'))
"
```

Expected output: `True`

- [ ] **Step 3: Commit**

```bash
git add 135-Polars_mini_max.py
git commit -m "feat: 135 — add build_df_s_n() variable-length df builder"
```

---

### Task 5: Implement `_04_minimax_search()`

**Files:**
- Modify: `135-Polars_mini_max.py` — replace the stub `_04_minimax_search` with the full implementation

**Interfaces:**
- Consumes: `safe_lf: pl.LazyFrame` (output of `_03`), `obs` (current game object), `cache: GameCache`, `player_id: int`
- Produces: `list` — zero or one move `[[id_src, angle, ships]]` (plus possible comet-evasion prepended moves)

- [ ] **Step 1: Replace the stub `_04_minimax_search` with the full implementation**

Find and replace the stub body (the `return []` stub added in Task 1) with:

```python
    @staticmethod
    def _04_minimax_search(safe_lf: pl.LazyFrame, obs, cache, player_id: int) -> list:
        attacks_with_angle = safe_lf.collect()
        moves_out = []

        # ── Comet evasion (preserved from original _04) ───────────────────
        if not attacks_with_angle.is_empty():
            awa_comets = attacks_with_angle.filter(pl.col("nature_src") == "comet")
            if not awa_comets.is_empty():
                x_off = (awa_comets["x_src"] - GameConfig.CENTER).abs().max() or 0
                y_off = (awa_comets["y_src"] - GameConfig.CENTER).abs().max() or 0
                if max(x_off, y_off) > 45:
                    moves_out += [list(r) for r in (
                        awa_comets
                        .sort(["ships_sent", "step"], descending=[True, False])
                        .group_by("id_src", maintain_order=True)
                        .first()
                        .select(["id_src", "final_angle", "ships_sent"])
                        .rows()
                    )]
                    id_to_avoid = awa_comets["id_src"].unique().to_list()
                    attacks_with_angle = attacks_with_angle.filter(
                        ~pl.col("id_src").is_in(id_to_avoid)
                    )

        # ── Step-0 candidates: top-5 per source (plus "do nothing") ──────
        NB_STEPS_5 = 5
        num_agents   = cache.num_agents
        current_step = cache.step

        step0_candidates = [None]  # None means "do nothing"
        if not attacks_with_angle.is_empty():
            top5_df = (
                attacks_with_angle
                .sort(["step", "ships_sent"])
                .group_by(["id_src", "id"], maintain_order=True)
                .first()
                .sort(["step", "ships_sent"])
                .group_by("id_src", maintain_order=True)
                .head(5)
            )
            for row in top5_df.select(["id_src", "final_angle", "ships_sent"]).iter_rows():
                step0_candidates.append(list(row))

        # ── Base cache: "do nothing" → step-5 candidate table ────────────
        obs_base5 = _simulate(obs, None, NB_STEPS_5, current_step, num_agents, player_id)
        df_s5_base, pd5_base = build_df_s_n(
            cache, obs_base5, current_step + NB_STEPS_5, NB_STEPS_5
        )
        pa5_base   = StrategyPipeline._02_get_all_opportunities(df_s5_base, pd5_base, player_id)
        safe5_base = StrategyPipeline._03_filter_collision(pa5_base).collect()

        base_planets5 = {
            p[0]: (p[1], p[5])
            for p in (obs_base5.planets if hasattr(obs_base5, "planets") else obs_base5["planets"])
        }

        best_score: tuple | None = None
        best_c0 = None

        for c0 in step0_candidates:
            # Simulate this step-0 candidate to step 5
            obs_c5 = _simulate(obs, c0, NB_STEPS_5, current_step, num_agents, player_id)
            c5_planets = obs_c5.planets if hasattr(obs_c5, "planets") else obs_c5["planets"]

            # Detect which of MY planets changed (ownership or ship count)
            changed_ids: set = set()
            for p in c5_planets:
                pid, owner = p[0], p[1]
                if owner != player_id:
                    continue
                base = base_planets5.get(pid)
                if base is None or base[0] != owner or base[1] != p[5]:
                    changed_ids.add(pid)

            # Build merged step-5 candidate table
            if changed_ids:
                df_s5_c, pd5_c = build_df_s_n(
                    cache, obs_c5, current_step + NB_STEPS_5, NB_STEPS_5
                )
                pa5_c   = StrategyPipeline._02_get_all_opportunities(df_s5_c, pd5_c, player_id)
                safe5_c = StrategyPipeline._03_filter_collision(pa5_c).collect()

                changed_list = list(changed_ids)
                new_rows  = safe5_c.filter(pl.col("id_src").is_in(changed_list)) if not safe5_c.is_empty() else safe5_c
                keep_rows = safe5_base.filter(~pl.col("id_src").is_in(changed_list)) if not safe5_base.is_empty() else safe5_base
                parts = [df for df in [keep_rows, new_rows] if not df.is_empty()]
                merged5 = pl.concat(parts) if parts else pl.DataFrame()
            else:
                merged5 = safe5_base

            # Enumerate step-5 candidates (plus "do nothing")
            step5_candidates = [None]
            if not merged5.is_empty():
                for row in merged5.select(["id_src", "final_angle", "ships_sent"]).iter_rows():
                    step5_candidates.append(list(row))

            # Evaluate each leaf
            best_score_5: tuple | None = None
            for c5 in step5_candidates:
                obs_leaf = _simulate(
                    obs_c5, c5, NB_STEPS_5, current_step + NB_STEPS_5, num_agents, player_id
                )
                score = evaluate(obs_leaf, player_id)
                if best_score_5 is None or score > best_score_5:
                    best_score_5 = score

            if best_score is None or best_score_5 > best_score:
                best_score  = best_score_5
                best_c0     = c0

        if best_c0 is not None:
            moves_out.append(best_c0)

        if moves_out:
            print(f"Minimax best move: {moves_out[-1]}  score={best_score}")
        return moves_out
```

- [ ] **Step 2: Verify the file still imports cleanly**

```bash
python -c "
import importlib.util
spec = importlib.util.spec_from_file_location('m', '135-Polars_mini_max.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
print('import OK')
"
```

Expected output: `import OK`

- [ ] **Step 3: Commit**

```bash
git add 135-Polars_mini_max.py
git commit -m "feat: 135 — implement _04_minimax_search 2-level lookahead"
```

---

### Task 6: Smoke test and lab submission

**Files:**
- Read: `135-Polars_mini_max.py` (no changes unless bugs found)

**Interfaces:**
- Consumes: completed `135-Polars_mini_max.py`
- Produces: verified agent + lab submission

- [ ] **Step 1: Run a full synthetic agent call**

```bash
python -c "
import importlib.util, types, math
spec = importlib.util.spec_from_file_location('m', '135-Polars_mini_max.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

obs = types.SimpleNamespace()
obs.player = 0
obs.planets = [
    [0, 0, 20.0, 50.0, 3.0, 80, 4],   # mine, fixed
    [1, 1, 80.0, 50.0, 3.0, 40, 3],   # opponent, fixed
    [2, -1, 50.0, 20.0, 2.0, 10, 2],  # neutral, fixed
]
obs.initial_planets = [list(p) for p in obs.planets]
obs.fleets = []
obs.next_fleet_id = 0
obs.comets = []
obs.comet_planet_ids = []
obs.angular_velocity = 0.0

moves = m.agent(obs)
print('moves:', moves)
print('smoke test OK')
"
```

Expected output: `moves:` followed by a list (may be empty or contain one move), then `smoke test OK`. Any exception is a bug — fix before proceeding.

- [ ] **Step 2: Deploy to lab**

```
/lab 135-Polars_mini_max.py
```

- [ ] **Step 3: Final commit**

```bash
git add 135-Polars_mini_max.py
git commit -m "feat: 135 — minimax search agent complete"
```
