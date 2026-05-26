# Design: 76-RL Tournament Runner

**Date:** 2026-05-26  
**File to create:** `76-RL_tournament.py` (project root)

---

## Overview

A standalone script that scans `76-RL_tournament/` for tournament folders, runs all
unresolved ones, and writes `results.json` into each folder. Each folder contains
4 agent config JSON files; the script runs a full 24-game tournament (18 × 2p +
6 × 4p) and prints a leaderboard summary at the end.

---

## Folder Scanning

- Scan every immediate subfolder of `76-RL_tournament/` (e.g. `000-test/`)
- **Skip** any folder that already contains `results.json`
- For each unresolved folder: find all `*.json` files (sorted by filename), ignore
  keys starting with `_` as metadata
- Expect exactly 4 configs per folder; raise a clear error if not

---

## Seed Strategy

1. Pick one `root_seed` = `random.randint(0, 100)`
2. Derive all 24 game seeds deterministically:
   ```python
   rng = random.Random(root_seed)
   game_seeds = [rng.randint(0, 2**31) for _ in range(24)]
   ```
3. Assign seeds in match order (2p games first, then 4p games)
4. Store `root_seed` in `results.json` — replaying any game is possible by
   reconstructing the sequence from the same root

---

## Agent Instantiation — `make_agent(cfg)`

Import `76-RL_tournament/main.py` **once** as a shared module. `make_agent(cfg)`
returns a closure that:

1. Before each call, sets all 14 module-level scoring constants from `cfg`
   (e.g. `mod.PROD_MULT = cfg["PROD_MULT"]`). Safe because games run sequentially.
2. Maintains its own `state = {"step": 0, "num_agents": None}` — no shared globals.
3. Calls `mod._simulate()` and `mod.take_action()` directly.

This avoids filesystem copies, subprocesses, or any modification to `main.py`.

---

## Match Schedule

### 2-player (18 games)

All C(4,2) = 6 pairs × 3 games each. Fixed player order per pair (agent with
lower filename index is always player-0). Seeds are `game_seeds[0..17]`.

| Pair | Games |
|------|-------|
| 001 v 002 | g0, g1, g2 |
| 001 v 003 | g3, g4, g5 |
| 001 v 004 | g6, g7, g8 |
| 002 v 003 | g9, g10, g11 |
| 002 v 004 | g12, g13, g14 |
| 003 v 004 | g15, g16, g17 |

### 4-player (6 games)

Six fixed orderings (Agent1=001, Agent2=002, Agent3=003, Agent4=004).
Seeds are `game_seeds[18..23]`.

| Game | Player order |
|------|-------------|
| 4p_0 | [001, 002, 003, 004] |
| 4p_1 | [001, 002, 004, 003] |
| 4p_2 | [001, 003, 002, 004] |
| 4p_3 | [001, 003, 004, 002] |
| 4p_4 | [001, 004, 002, 003] |
| 4p_5 | [001, 004, 003, 002] |

---

## Game Loop

Drive the step loop manually using `env.step()` instead of `env.run()`:

```python
env = make("orbit_wars", configuration={"seed": game_seed}, debug=False)
env.reset(num_agents=N)
states = env.step([[] for _ in range(N)])

while any status is ACTIVE:
    actions = [agent(extract_obs(states[i])) for i, agent in enumerate(agents)]
    prev_planets = current planet snapshot
    prev_fleet_ids = current fleet id set
    states = env.step(actions)
    collect_step_stats(states, prev_planets, prev_fleet_ids)
```

---

## Stats Collected Per Game Per Player

All derived from the observation at each step — no stdout parsing needed.

| Stat | How computed |
|------|-------------|
| `planets_owned_avg` | mean of per-step planet count |
| `peak_planets` | max of per-step planet count |
| `ships_sent_total` | sum of `fleet.ships` for new fleet IDs appearing each step |
| `neutral_captures` | count of neutral→own ownership transitions |
| `enemy_flips` | count of enemy→own ownership transitions |
| `planets_lost` | count of own→enemy ownership transitions |
| `first_action_step` | first step where agent dispatched ≥1 ship |
| `final_score` | ships on own planets + own ships in flight at termination |
| `win_turn` | step count at game end (game-level, not per-player) |

---

## `results.json` Schema

```json
{
  "root_seed": 42,
  "generated_at": "2026-05-26T14:00:00Z",
  "agents": {
    "001": { "_name": "Berserker", "PROD_MULT": 8.0, "...": "..." }
  },
  "matches": [
    {
      "match_id": "2p_001v002_g0",
      "format": "2p",
      "players": ["001", "002"],
      "seed": 918273645,
      "status": "ok",
      "winner": "001",
      "stats": {
        "win_turn": 381,
        "per_player": {
          "001": {
            "planets_owned_avg": 5.2,
            "peak_planets": 8,
            "ships_sent_total": 1420,
            "neutral_captures": 4,
            "enemy_flips": 3,
            "planets_lost": 1,
            "first_action_step": 3,
            "final_score": 4149
          }
        }
      }
    }
  ],
  "summary": {
    "2p": {
      "001": { "wins": 5, "losses": 1, "draws": 0 }
    },
    "4p": {
      "001": { "wins": 2, "top2": 4, "losses": 4 }
    }
  }
}
```

---

## Console Output

Progress line per game:
```
[001 v 002 g0] seed=918273645  001 wins  turn=381  scores: 001=4149 002=0
```

Leaderboard printed after all games:
```
══════════════════════════════════════════
  LEADERBOARD — 000-test  (seed=42)
══════════════════════════════════════════
  2-PLAYER
  Rank  Agent            W   L   D  Win%
   1    001-Berserker    5   1   0  83%
   2    003-Coordinator  4   2   0  67%
   3    002-Economist    2   4   0  33%
   4    004-OrbitalDom.  1   5   0  17%

  4-PLAYER
  Rank  Agent            Wins  Top2  Losses
   1    001-Berserker      3     5      1
   ...
══════════════════════════════════════════
```

---

## Error Handling

- Crashed game (`status="crashed"`): record `winner=null`, stats as zeros, continue
- Wrong number of configs (≠ 4): print error and skip that folder
- Missing `main.py`: raise immediately with a clear message

---

## Files Changed

| File | Change |
|------|--------|
| `76-RL_tournament.py` | **Create** — tournament runner |
| `76-RL_tournament/main.py` | No changes needed |
