# Minimax Restricted Inner Loop Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace `_04_minimax_search` in a new file `136-Polars_mini_max_fast.py` so that the inner step-5 evaluation loop runs O(K) per non-None c0 instead of O(M), bringing per-turn time under 1 second.

**Architecture:** Copy `135-Polars_mini_max.py` verbatim, then replace only `_04_minimax_search`. The new version runs a full O(M) baseline pass once (c0=None), records `best_c5_base`, then for each non-None c0 evaluates a restricted set: {None, all src_id attacks, all tgt_id attacks, best_c5_base}. All other functions are untouched.

**Tech Stack:** Python 3.10+, Polars, `copy.deepcopy`, `types.SimpleNamespace` (test only)

## Global Constraints

- Output file: `136-Polars_mini_max_fast.py` (root of repo)
- Entry point stays `def agent(obs)` — signature unchanged
- No new dependencies; no changes outside `_04_minimax_search`
- `step0_candidates` element shape changes from `move | None` → `(move | None, tgt_id | None)` — internal to `_04_minimax_search` only, not visible to callers
- Smoke test lives in `if __name__ == "__main__":` block at bottom of `136-Polars_mini_max_fast.py`

---

### Task 1: Create `136-Polars_mini_max_fast.py` with restricted minimax

**Files:**
- Create: `136-Polars_mini_max_fast.py`

**Interfaces:**
- Consumes: `_simulate`, `build_df_s_n`, `StrategyPipeline._02_get_all_opportunities`, `StrategyPipeline._03_filter_collision`, `evaluate`, `GameCache`, `GameConfig` — all copied verbatim from 135
- Produces: `def agent(obs) -> list` — same as 135

---

- [ ] **Step 1: Copy 135 to 136**

```bash
cp "135-Polars_mini_max.py" "136-Polars_mini_max_fast.py"
```

Expected: `136-Polars_mini_max_fast.py` exists and is identical to 135.

---

- [ ] **Step 2: Replace `_04_minimax_search` in `136-Polars_mini_max_fast.py`**

Find the method starting at the line `def _04_minimax_search(safe_lf:` inside `class StrategyPipeline` and replace the entire method body (lines 680–795 in 135) with the following. Leave everything else in the file untouched.

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

        NB_STEPS_5   = 5
        num_agents   = cache.num_agents
        current_step = cache.step

        # ── Step-0 candidates: (move, tgt_id) tuples ──────────────────────
        step0_candidates = [(None, None)]
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
            for row in top5_df.select(["id_src", "id", "final_angle", "ships_sent"]).iter_rows():
                src_id, tgt_id, angle, ships = row
                step0_candidates.append(([src_id, angle, ships], tgt_id))

        # ── Base state: simulate "do nothing" 5 steps ─────────────────────
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

        # ── Baseline pass: full step-5 evaluation → record best_c5_base ───
        base_step5_candidates = [None]
        if not safe5_base.is_empty():
            for row in safe5_base.select(["id_src", "final_angle", "ships_sent"]).iter_rows():
                base_step5_candidates.append(list(row))

        best_c5_base: list | None = None
        best_score_base: tuple | None = None
        for c5 in base_step5_candidates:
            obs_leaf = _simulate(
                obs_base5, c5, NB_STEPS_5, current_step + NB_STEPS_5, num_agents, player_id
            )
            score = evaluate(obs_leaf, player_id)
            if best_score_base is None or score > best_score_base:
                best_score_base, best_c5_base = score, c5

        best_score: tuple | None = best_score_base
        best_c0 = None

        # ── Non-None c0 loop: restricted step-5 candidates ────────────────
        for c0_move, c0_tgt_id in step0_candidates[1:]:
            obs_c5 = _simulate(obs, c0_move, NB_STEPS_5, current_step, num_agents, player_id)
            c5_planets = obs_c5.planets if hasattr(obs_c5, "planets") else obs_c5["planets"]

            changed_ids: set = set()
            for p in c5_planets:
                pid, owner = p[0], p[1]
                if owner != player_id:
                    continue
                base = base_planets5.get(pid)
                if base is None or base[0] != owner or base[1] != p[5]:
                    changed_ids.add(pid)

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

            src_id = c0_move[0]
            covered_srcs = {src_id}
            if c0_tgt_id is not None:
                covered_srcs.add(c0_tgt_id)

            restricted: list = [None]
            if not merged5.is_empty():
                filtered = merged5.filter(pl.col("id_src").is_in(list(covered_srcs)))
                for row in filtered.select(["id_src", "final_angle", "ships_sent"]).iter_rows():
                    restricted.append(list(row))

            if best_c5_base is not None and best_c5_base[0] not in covered_srcs:
                restricted.append(best_c5_base)

            best_score_5: tuple | None = None
            for c5 in restricted:
                obs_leaf = _simulate(
                    obs_c5, c5, NB_STEPS_5, current_step + NB_STEPS_5, num_agents, player_id
                )
                score = evaluate(obs_leaf, player_id)
                if best_score_5 is None or score > best_score_5:
                    best_score_5 = score

            if best_score is None or best_score_5 > best_score:
                best_score  = best_score_5
                best_c0     = c0_move

        if best_c0 is not None:
            moves_out.append(best_c0)

        if moves_out:
            print(f"Minimax best move: {moves_out[-1]}  score={best_score}")
        return moves_out
```

---

- [ ] **Step 3: Add smoke test block at the bottom of `136-Polars_mini_max_fast.py`**

Append after the existing `def agent(obs):` function (after line 1183 in 135's layout, i.e. after the closing of `agent`):

```python


if __name__ == "__main__":
    import time
    from types import SimpleNamespace

    def _make_obs():
        # planet format: [id, owner, x, y, radius, ships, production]
        planets = [
            [0, 0, 30.0, 50.0, 5.0, 100, 2],  # player 0
            [1, 1, 70.0, 50.0, 5.0,  20, 2],  # player 1
            [2, -1, 50.0, 20.0, 4.0,   0, 1], # neutral
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
        )

    # First call initialises CACHE
    obs0 = _make_obs()
    t0 = time.time()
    result = agent(obs0)
    elapsed = time.time() - t0

    print(f"agent() → {result}")
    print(f"elapsed: {elapsed:.3f}s")

    assert isinstance(result, list), "agent must return a list"
    for move in result:
        assert len(move) == 3, f"move must be [id_src, angle, ships_sent]: {move}"
    assert elapsed < 5.0, f"too slow: {elapsed:.3f}s (budget 1s in competition)"

    print("Smoke test PASSED.")
```

---

- [ ] **Step 4: Run the smoke test to verify correctness and timing**

```bash
cd "C:\Users\trant\Documents\Programmation\Orbit Wars"
python "136-Polars_mini_max_fast.py"
```

Expected output (exact values will differ):
```
Agent called step: 0 remainingOverageTime: 0
Minimax best move: [0, 1.5707963..., 100]  score=(...)
agent() → [[0, 1.5707963..., 100]]
elapsed: 0.XXXs
Smoke test PASSED.
```

If `elapsed` is above 5 seconds there is a logic error (likely an accidental fallback to the full M loop). Inspect the `restricted` list size printed by adding a temporary `print(f"restricted size: {len(restricted)}")` inside the loop.

---

- [ ] **Step 5: Commit**

```bash
git add "136-Polars_mini_max_fast.py"
git commit -m "feat: 136 — restricted minimax inner loop (O(M) + O(N*K) vs O(N*M))"
```

---

### Self-review checklist (for implementer)

- [ ] `step0_candidates` is `list[tuple]` not `list` — confirm loop unpacks `c0_move, c0_tgt_id`
- [ ] `c0_move` passed to `_simulate` is a 3-element list `[id_src, angle, ships_sent]`, not the 4-field tuple
- [ ] `best_c5_base` is `None` when `safe5_base` is empty and `base_step5_candidates = [None]` only — `best_c5_base` stays `None`, the `if best_c5_base is not None` guard fires correctly
- [ ] When `merged5` is empty, `restricted = [None]` only — agent returns `[]` (same as 135 when no moves found)
- [ ] `covered_srcs` check uses `best_c5_base[0]` which is `id_src` (int) — confirmed `best_c5_base` is `[id_src, angle, ships_sent]` or `None`
