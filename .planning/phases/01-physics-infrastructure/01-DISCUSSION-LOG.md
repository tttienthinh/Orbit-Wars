# Phase 1 Discussion Log

**Date:** 2026-05-11
**Phase:** 1 — Physics & Infrastructure
**Outcome:** CONTEXT.md written, ready for planning

## Gray Areas Identified

1. **Starting point** — Build fresh from `09-Perfect_Aiming/main.py` OR refactor `07-claude_code.py`
2. **Garrison module placement** — `planner.py` (strategic) vs `physics.py` (hard constraint)
3. **Spawn-offset fix depth** — one-pass or two-pass iterative refinement (PHYS-01)
4. **Time-budget guard placement** — per-candidate during scoring loop vs single gate after expansion (INFRA-03)

## Discussed

### Starting point (user selected for discussion)

**Question:** Should Phase 1 start by refactoring `07-claude_code.py` into the `agent/` package, or build fresh from `09-Perfect_Aiming/main.py`?

**Discussion summary:**
- `07-claude_code.py` is a 520-line monolith mixing strategic heuristics (garrison tiers, multi-fleet coordination, EXPOSED_BONUS weights) with correct physics algorithms
- Refactoring it risks carrying over Phase 2/3 concerns that are explicitly deferred
- `09-Perfect_Aiming/main.py` is the clean 05-level baseline: correct `fleet_speed`, correct `is_planet_moving`, minimal structure — good starting point for adding the 4 physics fixes
- The physics algorithms in `07-claude_code.py` (`orbital_position`, `path_hits_sun`, `intercept_time`) are still the reference implementations to copy into `physics.py`

**Decision (D-01/D-02):** Build the `agent/` package fresh from `09-Perfect_Aiming/main.py`. Use `07-claude_code.py` as physics reference only — do NOT copy its strategic heuristics.

## Left at Claude's Discretion

- **Garrison module placement** — researcher/planner to decide; either `planner.py` or `physics.py` acceptable provided PHYS-04 acceptance criteria met
- **Spawn-offset fix depth** — one-pass vs two-pass; evaluate correctness vs simplicity
- **Time-budget guard placement** — must truncate and return best moves so far at 0.8s; exact insertion point is researcher/planner's call
- **Package internal boundaries** — module names locked (INFRA-01), internal function→module split is not
