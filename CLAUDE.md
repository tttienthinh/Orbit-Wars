# Orbit Wars Competitive Agent

## Project

A competitive Python AI agent for the Kaggle Orbit Wars competition. Evolves from a heuristic
baseline (nearest-planet sniper with orbital intercept) toward a production-value forward-sim
agent. RL (PettingZoo + SB3 PPO) is planned for v2.

**Core value:** Beat yuriygreben's "architect" agent in local backtests before touching the
Kaggle leaderboard.

**Current submission:** `main.py` (root) — nearest-planet sniper with orbital intercept (`05` level)

## Key Files

- `main.py` — Kaggle submission entry point (`def agent(obs)`)
- `01`–`06-*.py` — Progressive experiment scripts (history, not submissions)
- `agent/` — Agent package (created in Phase 1; not yet in codebase)
- `orbit-wars-lab/` — Local backtesting lab (FastAPI + TrueSkill + replay viewer)
- `.planning/` — GSD planning artifacts

## Physics Facts (critical for implementation)

- Board: 100×100, sun at (50, 50) radius 10
- Fleet spawns at **planet surface** (`center + unit(angle) * planet_radius`), not center
- Fleet speed: `1.0 + (6.0-1.0) * (log(ships)/log(1000))^1.5`
- Orbiting condition: `dist(planet_center, sun_center) + planet_radius < 50`
- All orbiting planets share one `angular_velocity` (per game)
- Orbital position: `θ(t) = θ₀ + ω × t` (use cumulative formula, not iterative steps)
- Turn budget: 1 second (`actTimeout = 1`)

## GSD Workflow

This project uses GSD (Get Shit Done) for planning and execution.

**Current state:** Roadmap created, Phase 1 not yet started.

**To start working:**
```
/gsd-discuss-phase 1   ← gather context and clarify approach
/gsd-plan-phase 1      ← create execution plan
/gsd-execute-phase 1   ← execute the plan
```

**Workflow gates:**
- Always run `/gsd-plan-phase N` before executing a phase
- Commit planning docs alongside code (already configured)
- Verify each phase with `/gsd-verify-work` before advancing

**Planning artifacts:**
- `.planning/PROJECT.md` — project context and requirements
- `.planning/REQUIREMENTS.md` — 13 v1 requirements with REQ-IDs
- `.planning/ROADMAP.md` — 4-phase roadmap
- `.planning/research/` — domain research (stack, features, architecture, pitfalls)
- `.planning/codebase/` — codebase map (architecture, stack, conventions, concerns)

## Benchmark

Local benchmark: yuriygreben's "architect" agent at
`orbit-wars-lab/agents/external/` — run 50-game series via orbit-wars-lab to measure progress.
