# 142 — First Minimax Then Scoring Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create `142-First_Minimax_Then_Scoring.py` by copying `140-BoardCache_optimised.py` and replacing `_04_minimax_search` with a lighter sequential 2-ply search that scores step-0 moves by the net-production delta they unlock at step+5.

**Architecture:** Copy `140` verbatim, remove the two batched `_02`/`_03` methods and `_04_minimax_search`, add a `_net_prod` helper, implement `_04_first_minimax_then_scoring`, and update `agent()` to call the new method. The `Board` class, `_02_get_all_opportunities`, `_03_filter_collision`, and all helper classes are unchanged.

**Tech Stack:** Python 3, Polars, math, types.SimpleNamespace (test only)

## Global Constraints

- File must be self-contained and runnable as `python 142-First_Minimax_Then_Scoring.py`
- `def agent(obs)` must be the callable entry point
- All physics constants identical to `140` (CENTER=50.0, SUN_RADIUS=10.0, MAX_SPEED=6.0, NB_STEPS_SIM=10, etc.)
- Do not modify `140-BoardCache_optimised.py`

---

### Task 1: Scaffold `142` from `140` — remove dead methods, stub new `_04`

**Files:**
- Create: `142-First_Minimax_Then_Scoring.py`

**Interfaces:**
- Produces: file with `agent(obs)` callable, `StrategyPipeline._04_first_minimax_then_scoring` stub returning `[]`

- [ ] **Step 1: Copy 140 to 142**

```powershell
Copy-Item "140-BoardCache_optimised.py" "142-First_Minimax_Then_Scoring.py"
```

- [ ] **Step 2: Remove the three methods that will be replaced**

Open `142-First_Minimax_Then_Scoring.py` and delete these three static methods from `StrategyPipeline`:
- `_02_get_all_opportunities_batched` (lines ~305–483 in 140)
- `_03_filter_collision_batched` (lines ~486–517 in 140)
- `_04_minimax_search` (lines ~520–656 in 140)

- [ ] **Step 3: Add `_net_prod` stub + `_04_first_minimax_then_scoring` stub inside `StrategyPipeline`**

Add these two static methods after `_03_filter_collision`:

```python
    @staticmethod
    def _net_prod(horizon: dict, player_id: int) -> int:
        opponents = {v[1] for v in horizon.values() if v[1] not in (-1, player_id)}
        my_prod = sum(v[2] for v in horizon.values() if v[1] == player_id)
        if not opponents:
            return my_prod
        opp_prod = max(
            sum(v[2] for v in horizon.values() if v[1] == opp) for opp in opponents
        )
        return my_prod - opp_prod

    @staticmethod
    def _04_first_minimax_then_scoring(safe_lf: pl.LazyFrame, obs, board: "Board") -> list:
        return []
```

- [ ] **Step 4: Update `agent()` to call the new method**

Replace the last three lines of `agent()` from:

```python
    pa_lf   = StrategyPipeline._02_get_all_opportunities(df_s, planet_disp, BOARD.player_id)
    safe_lf = StrategyPipeline._03_filter_collision(pa_lf)
    moves   = StrategyPipeline._04_minimax_search(safe_lf, obs, BOARD)
```

To:

```python
    pa_lf   = StrategyPipeline._02_get_all_opportunities(df_s, planet_disp, BOARD.player_id)
    safe_lf = StrategyPipeline._03_filter_collision(pa_lf)
    moves   = StrategyPipeline._04_first_minimax_then_scoring(safe_lf, obs, BOARD)
```

- [ ] **Step 5: Replace the `__main__` smoke test with a minimal version**

Replace the entire `if __name__ == "__main__":` block with:

```python
if __name__ == "__main__":
    import time
    from types import SimpleNamespace

    def _make_obs():
        planets = [
            [0, 0, 30.0, 50.0, 31.0, 100, 2],
            [1, 1, 70.0, 50.0, 5.0,   20, 2],
            [2, -1, 50.0, 20.0, 4.0,   0, 1],
        ]
        return SimpleNamespace(
            planets=[list(p) for p in planets],
            initial_planets=[list(p) for p in planets],
            fleets=[],
            comets=[],
            comet_planet_ids=[],
            angular_velocity=0.01,
            next_fleet_id=0,
            player=0,
            step=0,
        )

    BOARD = None
    obs0 = _make_obs()
    t0 = time.time()
    result = agent(obs0)
    elapsed = time.time() - t0

    print(f"agent() -> {result}")
    print(f"elapsed: {elapsed:.3f}s")

    assert isinstance(result, list), "agent must return a list"
    for move in result:
        assert len(move) == 3, f"move must be [id_src, angle, ships]: {move}"
    assert elapsed < 5.0, f"too slow: {elapsed:.3f}s"

    print("Stub smoke test PASSED.")
```

- [ ] **Step 6: Run the stub to verify it imports and runs**

```powershell
cd "C:\Users\trant\Documents\Programmation\Orbit Wars"
python 142-First_Minimax_Then_Scoring.py
```

Expected output ends with: `Stub smoke test PASSED.`

- [ ] **Step 7: Commit**

```powershell
git add 142-First_Minimax_Then_Scoring.py
git commit -m "feat: 142 — scaffold from 140, stub _04_first_minimax_then_scoring"
```

---

### Task 2: Implement `_04_first_minimax_then_scoring` and final smoke test

**Files:**
- Modify: `142-First_Minimax_Then_Scoring.py` — replace stub body, expand smoke test

**Interfaces:**
- Consumes: `StrategyPipeline._net_prod(horizon: dict, player_id: int) -> int` (Task 1)
- Consumes: `Board._apply_sim_fleet(horizon: dict, sim_row) -> dict` (unchanged from 140)
- Consumes: `board._recompute_from_sim(df_ships_base, move, id_tgt, step_tgt) -> pl.DataFrame`
- Consumes: `board.build_df_s_slice(df_ships, step_from) -> (pl.DataFrame, pl.DataFrame)`
- Consumes: `StrategyPipeline._02_get_all_opportunities(df_s, planet_disp, player_id) -> pl.LazyFrame`
- Consumes: `StrategyPipeline._03_filter_collision(pa_lf) -> pl.LazyFrame`
- Consumes: `board.extract_horizon_dict(df_ships) -> dict`
- Produces: `_04_first_minimax_then_scoring(safe_lf, obs, board) -> list[list]`

- [ ] **Step 1: Replace the stub body of `_04_first_minimax_then_scoring`**

Replace the stub `return []` body with the full implementation:

```python
    @staticmethod
    def _04_first_minimax_then_scoring(safe_lf: pl.LazyFrame, obs, board: "Board") -> list:
        NB_STEPS_5 = 5
        player_id = board.player_id

        attacks_with_angle = safe_lf.collect()
        moves_out: list = []

        # ── Comet evasion ────────────────────────────────────────────────
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

        # ── Step-0 candidates: None = "do nothing" ───────────────────────
        step0_candidates: list = [None]
        if not attacks_with_angle.is_empty():
            for row in attacks_with_angle.select(
                ["id_src", "id", "step", "final_angle", "ships_sent"]
            ).iter_rows():
                src_id, tgt_id, step_tgt, angle, ships = row
                step0_candidates.append((src_id, tgt_id, step_tgt, angle, ships))

        df_ships_full = board.df_planete_ships
        step5_from = board.step + NB_STEPS_5

        best_score: int = -1
        best_c0 = None

        for c0 in step0_candidates:
            # Build ship state after this step-0 action
            if c0 is None:
                df_ships_c0 = df_ships_full
            else:
                src_id, tgt_id, step_tgt, angle, ships = c0
                df_ships_c0 = board._recompute_from_sim(
                    df_ships_full, [src_id, angle, ships], tgt_id, step_tgt
                )

            # Step-5 opportunity search
            df_s5, pd5 = board.build_df_s_slice(df_ships_c0, step_from=step5_from)
            pa5_lf = StrategyPipeline._02_get_all_opportunities(df_s5, pd5, player_id)
            safe5 = StrategyPipeline._03_filter_collision(pa5_lf).collect()

            # Score: max net-production delta across all valid step-5 attacks
            horizon_c0 = board.extract_horizon_dict(df_ships_c0)
            baseline_np = StrategyPipeline._net_prod(horizon_c0, player_id)

            score_c0 = 0
            if not safe5.is_empty():
                for row5 in safe5.select(["id_src", "id", "step", "ships_sent"]).iter_rows():
                    h_after = Board._apply_sim_fleet(horizon_c0, row5)
                    delta = StrategyPipeline._net_prod(h_after, player_id) - baseline_np
                    if delta > score_c0:
                        score_c0 = delta

            if score_c0 > best_score:
                best_score = score_c0
                best_c0 = c0

        # ── Return ───────────────────────────────────────────────────────
        if best_c0 is not None:
            src_id, tgt_id, step_tgt, angle, ships = best_c0
            moves_out.append([src_id, angle, ships])
            print(f"142 best move: src={src_id} tgt={tgt_id} step={step_tgt} ships={ships} score={best_score}")
        return moves_out
```

- [ ] **Step 2: Expand the `__main__` smoke test to cover the full search**

Replace the existing `__main__` block with:

```python
if __name__ == "__main__":
    import time
    from types import SimpleNamespace

    def _make_obs(step=0):
        planets = [
            [0, 0, 30.0, 50.0, 31.0, 100, 2],
            [1, 1, 70.0, 50.0, 5.0,   20, 2],
            [2, -1, 50.0, 20.0, 4.0,   0, 1],
        ]
        return SimpleNamespace(
            planets=[list(p) for p in planets],
            initial_planets=[list(p) for p in planets],
            fleets=[],
            comets=[],
            comet_planet_ids=[],
            angular_velocity=0.01,
            next_fleet_id=0,
            player=0,
            step=step,
        )

    # ── _net_prod unit test ───────────────────────────────────────────
    horizon_test = {
        0: (100, 0, 2),   # player 0 owns, prod=2
        1: (20,  1, 2),   # player 1 owns, prod=2
        2: (0,  -1, 1),   # neutral
    }
    np0 = StrategyPipeline._net_prod(horizon_test, player_id=0)
    assert np0 == 0, f"_net_prod: expected 0, got {np0}"  # my_prod=2, opp_prod=2

    # capture planet 2 (neutral, prod=1): my_prod becomes 3, opp stays 2 → delta=+1
    h_after = Board._apply_sim_fleet(horizon_test, (0, 2, 3, 50))
    np_after = StrategyPipeline._net_prod(h_after, player_id=0)
    assert np_after - np0 == 1, f"capture neutral delta: expected +1, got {np_after - np0}"

    # capture planet 1 (opponent, prod=2): my_prod=4, opp_prod=0 → net=4, delta=+4
    h_after2 = Board._apply_sim_fleet(horizon_test, (0, 1, 5, 50))
    np_after2 = StrategyPipeline._net_prod(h_after2, player_id=0)
    assert np_after2 - np0 == 4, f"capture opp delta: expected +4, got {np_after2 - np0}"

    print("_net_prod unit tests PASSED")

    # ── agent() smoke test ────────────────────────────────────────────
    global BOARD
    BOARD = None
    obs0 = _make_obs(step=0)
    t0 = time.time()
    result = agent(obs0)
    elapsed = time.time() - t0

    print(f"agent() step=0 -> {result}")
    print(f"elapsed: {elapsed:.3f}s")

    assert isinstance(result, list), "agent must return a list"
    for move in result:
        assert len(move) == 3, f"move must be [id_src, angle, ships]: {move}"
    assert elapsed < 5.0, f"too slow: {elapsed:.3f}s"

    # ── agent() step=1 (tests advance path) ──────────────────────────
    obs1 = _make_obs(step=1)
    t1 = time.time()
    result1 = agent(obs1)
    elapsed1 = time.time() - t1
    print(f"agent() step=1 -> {result1}, elapsed: {elapsed1:.3f}s")
    assert isinstance(result1, list)
    assert elapsed1 < 5.0

    print("Smoke test PASSED.")
```

- [ ] **Step 3: Run the full smoke test**

```powershell
cd "C:\Users\trant\Documents\Programmation\Orbit Wars"
python 142-First_Minimax_Then_Scoring.py
```

Expected output:
```
_net_prod unit tests PASSED
Agent called step=0 ...
142 best move: ...   (or no line if no move found)
agent() step=0 -> [...]
elapsed: <5.0s
Agent called step=1 ...
agent() step=1 -> [...]
Smoke test PASSED.
```

- [ ] **Step 4: Commit**

```powershell
git add 142-First_Minimax_Then_Scoring.py
git commit -m "feat: 142 — _04_first_minimax_then_scoring with net-prod delta scoring"
```
