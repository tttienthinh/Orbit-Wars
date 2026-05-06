# Orbit Wars Agent

## What This Is

A competitive Python AI agent for the Kaggle Orbit Wars competition — a 2/4-player real-time
strategy game where agents conquer planets orbiting a central sun by sending fleets of ships.
The project evolves from a working heuristic baseline (nearest-planet sniper with orbital
intercept) toward a self-play RL agent, using a local backtesting lab to benchmark every
iteration against known opponents.

## Core Value

Beat yuriygreben's "architect" agent in local backtests — that is the quality gate before
touching the Kaggle leaderboard.

## Requirements

### Validated

- ✓ Basic fleet launching (nearest-planet sniper) — existing in `01`–`06` files
- ✓ Orbital intercept prediction (predict target position N turns ahead) — existing in `05`
- ✓ Fleet speed formula (`1 + (maxSpeed-1) * (log(ships)/log(1000))^1.5`) — existing in `04`–`06`
- ✓ Local backtesting lab with TrueSkill ratings and replay viewer — `orbit-wars-lab/`
- ✓ Per-turn print visualizer (readable in Kaggle episode logs) — existing

### Active

- [ ] Accurate fleet aiming with spawn-offset correction (fleet spawns at planet surface, not center)
- [ ] Sun collision avoidance — reject angles whose straight-line path crosses the sun disc
- [ ] Production-value targeting — score targets by `production / cost` not just distance
- [ ] Threat detection — identify incoming enemy fleets and their arrival turn
- [ ] Defensive response — reinforce or counter-attack when a planet will fall
- [ ] N-turn forward simulation — predict board state T turns ahead for scoring
- [ ] Game phase awareness — distinct behaviour for early (expand), mid (contest), late (hold)
- [ ] Self-play RL training loop — PettingZoo multi-agent env + StableBaselines3 PPO
- [ ] Ship-advantage shaped reward — continuous reward signal, not sparse win/loss

### Out of Scope

- 4-player dedicated strategy — focus on 2-player until top-10 range; 4-player is a later extension
- Custom game engine rewrite — use `kaggle-environments` as the simulator throughout

## Context

**Competition:** Kaggle Orbit Wars — currently in top half of leaderboard, weeks remaining.

**Existing codebase structure:**
- `main.py` (root) — active Kaggle submission, currently the nearest-planet sniper
- `01`–`06` scripts — progressive experiments showing the learning path
- `orbit-wars-lab/` — full FastAPI + TypeScript lab cloned from GitHub; provides match runner
  (fast/faithful modes), TrueSkill leaderboard, replay visualiser, and agent zoo at
  `agents/{baselines,external,mine}/`
- `orbit-wars-lab/agents/external/` — holds yuriygreben's "architect" agent (06) as the
  primary benchmark

**Physics gotchas already discovered:**
- Fleet spawns at planet surface (`planet_radius` offset from center), not at planet center —
  affects intercept angle for large origin planets hitting small fast targets
- Orbiting planets: `orbital_radius + planet_radius < 50` (i.e. center-distance < 50)
- All planets share a single `angular_velocity` (per game, not per planet)

**The yuriygreben advantage:** ~120-constant rule-based agent with 10+ action types
(attack, snipe, reinforce, recapture, crash-exploit, swarm), 110-turn sim horizon, and
game-phase detection. The gap from current agent to that level is ~2 weeks of heuristic work,
or less if a RL layer is trained on top of a solid foundation.

**RL plan:** CPU training initially; willing to rent GPU. Use existing PettingZoo +
StableBaselines3 PPO pipeline rather than building from scratch. Shape reward as ship
advantage over time (not sparse win/loss). Warm-start from the heuristic policy.

## Constraints

- **Turn time:** 1-second `actTimeout` — agent must return moves in < 1 second per turn
- **Submission format:** single `main.py` at repo root with `def agent(obs)` function
- **Simulator speed:** `kaggle-environments` Python engine; ~10–50 games/sec on CPU
- **Timeline:** Weeks remaining in competition; prioritise quality at each phase over completeness
- **RL infrastructure:** CPU-first; GPU rental available if RL training is too slow on CPU

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Heuristic foundation before RL | Pure RL with slow CPU simulator burns timeline; better to have a strong prior | — Pending |
| PettingZoo + SB3 as RL framework | Existing ecosystem, avoids building training infrastructure from scratch | — Pending |
| Ship-advantage reward (not win/loss) | Sparse reward makes RL impractical over 500 turns | — Pending |
| yuriygreben as local quality gate | Concrete, reachable benchmark before Kaggle evaluation | — Pending |

## Evolution

This document evolves at phase transitions and milestone boundaries.

**After each phase transition** (via `/gsd-transition`):
1. Requirements invalidated? → Move to Out of Scope with reason
2. Requirements validated? → Move to Validated with phase reference
3. New requirements emerged? → Add to Active
4. Decisions to log? → Add to Key Decisions
5. "What This Is" still accurate? → Update if drifted

**After each milestone** (via `/gsd-complete-milestone`):
1. Full review of all sections
2. Core Value check — still the right priority?
3. Audit Out of Scope — reasons still valid?
4. Update Context with current state

---
*Last updated: 2026-05-06 after initialization*
