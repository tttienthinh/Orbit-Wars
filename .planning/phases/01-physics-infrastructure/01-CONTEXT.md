# Phase 1: Physics & Infrastructure - Context

**Gathered:** 2026-05-11
**Status:** Ready for planning

<domain>
## Phase Boundary

Restructure the existing 05-level agent (`09-Perfect_Aiming/main.py`) into an importable
`agent/` package (`state.py`, `physics.py`, `scorer.py`, `planner.py`, `defense.py`), fixing
four confirmed physics bugs (PHYS-01 spawn-offset, PHYS-02 sun collision, PHYS-03 cumulative
formula, PHYS-04 garrison rule) and adding a per-turn time-budget guard (INFRA-03).

The agent's strategy stays minimal for this phase — nearest unowned planet, orbital intercept,
garrison guard. Advanced scoring and defensive responses belong in Phases 2 and 3.

</domain>

<decisions>
## Implementation Decisions

### Starting Point
- **D-01:** Build the `agent/` package fresh from `orbit-wars-lab/agents/mine/09-Perfect_Aiming/main.py`
  (the 05-level baseline), NOT by refactoring `07-claude_code.py`. The user explicitly chose the
  cleaner-slate approach.
- **D-02:** `07-claude_code.py` is a REFERENCE ONLY — use it to understand what correct
  physics algorithms look like, but do not copy its strategic heuristics (garrison tiers,
  scoring weights, multi-fleet coordination). Those belong in Phases 2-3.

### Physics Corrections
- **D-03:** Orbital position MUST use the cumulative formula: `θ(t) = θ₀ + ω × t` applied
  to the ORIGINAL planet position — not iterative `next_position` steps. Concretely:
  `r = dist(planet, center)`, `θ₀ = atan2(py - cy, px - cx)`, then
  `px(t) = cx + r·cos(θ₀ + ω·t)`, `py(t) = cy + r·sin(θ₀ + ω·t)`.
- **D-04:** Intercept horizon raised from `t_max=20` to `t_max=200` turns.
  The 05 agent's t_max=20 misses all medium-range orbiting targets.

### Claude's Discretion
- **Garrison module placement:** The user said "you decide" — researcher and planner should
  determine whether `max(production × 2, 5)` garrison enforcement lives in `planner.py`
  (strategic decision) or `physics.py` (hard constraint). Either is acceptable provided
  PHYS-04 acceptance criteria are met.
- **Spawn-offset fix depth (PHYS-01):** One-pass or two-pass iterative refinement — researcher
  should evaluate what's needed for correctness vs. simplicity. The `07-claude_code.py`
  reference uses a two-pass approach (compute shot, then refine angle with actual fleet size).
- **Time-budget guard placement (INFRA-03):** Researcher/planner decides whether the 0.8s
  check is per-candidate during the scoring loop, or a single gate after expansion. Must
  truncate evaluation and return best moves found so far.
- **Package internal boundaries:** Exact function→module assignment for the 5 modules is
  at Claude's discretion. The module names (state.py, physics.py, scorer.py, planner.py,
  defense.py) are locked by INFRA-01; the internal split is not.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Requirements & Roadmap
- `.planning/REQUIREMENTS.md` — All Phase 1 requirements: PHYS-01, PHYS-02, PHYS-03,
  PHYS-04, INFRA-01, INFRA-03. Read acceptance criteria verbatim — they are the
  verification bar.
- `.planning/ROADMAP.md` §Phase 1 — Goal statement and 5 success criteria with exact
  measurable conditions.

### Starting Point Code
- `orbit-wars-lab/agents/mine/09-Perfect_Aiming/main.py` — The 05-level baseline that
  Phase 1 packages into `agent/`. Contains `fleet_speed`, `is_planet_moving`,
  `next_position` (buggy iterative version to be replaced), `nearest_planet_sniper`.

### Reference Implementation (physics algorithms, NOT strategy)
- `07-claude_code.py` — Working implementations of: cumulative orbital formula
  (`orbital_position`), sun-collision rejection (`path_hits_sun`, `_pt_seg_dist`),
  spawn-surface launch offset (`intercept_time`), two-pass angle refinement
  (`compute_shot`). Study these algorithms; do not copy the garrison tiers or
  multi-fleet coordination logic.

### Game Engine
- `.planning/codebase/STACK.md` §Game Environment — `kaggle-environments` VCS pin,
  `Planet`/`Fleet` namedtuple source.
- `.planning/codebase/ARCHITECTURE.md` §Agent Code Layer — Agent function contract:
  `def agent(obs) -> list[list]`.
- `CLAUDE.md` §Physics Facts — Board dimensions, fleet speed formula, orbiting condition,
  angular velocity rule, turn budget (1 second).

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets from 09-Perfect_Aiming
- `fleet_speed(ships)` — Correct speed formula; reuse directly in `physics.py`
- `is_planet_moving(planet)` — Orbiting condition check; reuse in `physics.py`
- `nearest_planet_sniper` loop structure — Reuse the outer loop pattern in `planner.py`
  (iterate owned planets, pick nearest target, compute angle, append move)

### Reference Algorithms from 07-claude_code.py
- `orbital_position(x, y, angular_velocity, t)` — Correct cumulative formula; copy into `physics.py`
- `path_hits_sun(mine, aim_x, aim_y, target_radius)` — Sun collision rejection; copy into `physics.py`
- `_pt_seg_dist(px, py, x1, y1, x2, y2)` — Helper for segment distance; copy into `physics.py`
- `intercept_time(ox, oy, tx, ty, angular_velocity, ships, target_radius)` — Spawn-surface aware intercept; copy into `physics.py`

### Integration Points
- `main.py` (root) — Kaggle submission entry point; must do `from agent.planner import agent`
  or equivalent. Must remain self-contained for Kaggle multi-file submission format.
- `orbit-wars-lab/agents/mine/` — New `09-Perfect_Aiming` is where the packaged version
  lives for local backtesting via `simulate.py`.

### Established Patterns
- Agent function contract: `def agent(obs)` accepting dict or Namespace, returning
  `[[from_planet_id, angle_radians, num_ships], ...]`
- Physics constants: `CENTER_X = CENTER_Y = 50.0`, `SUN_R = 10.0`, `MAX_SPEED = 6.0`

</code_context>

<specifics>
## Specific Ideas

- The package lives at `agent/` (repo root), importable as `from agent.physics import ...`
- `main.py` (root) stays minimal — just imports from `agent/` and exposes `agent(obs)`
- The 09-Perfect_Aiming `main.py` should be updated to use the new package once built

</specifics>

<deferred>
## Deferred Ideas

- Advanced garrison tiers (step-based, planet-ratio-based, threat-reactive) — these are in
  `07-claude_code.py` but belong in Phase 2 once the physics foundation is solid
- Multi-fleet coordination — Phase 2/3 scope
- Counter-attack and reinforcement logic — Phase 3 scope

</deferred>

---

*Phase: 1-physics-infrastructure*
*Context gathered: 2026-05-11*
