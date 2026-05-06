# Requirements — Orbit Wars Competitive Agent

**Milestone:** v1 — Beat yuriygreben's "architect" agent in local backtests
**Approach:** Enhance `05-Getting_started_moving_precision.py`, then study and surpass `06-yuriygreben-architect_moving_precision.py`

---

## v1 Requirements

### Physics

- [ ] **PHYS-01**: Fleet launch angle accounts for spawn position at planet surface (not center) — iterative solve corrects for large origin planets targeting small fast targets
- [ ] **PHYS-02**: Sun collision detection rejects any fleet path whose straight-line segment intersects the sun disc (center 50,50, radius 10 + 1.5 safety margin)
- [ ] **PHYS-03**: Orbital position uses cumulative formula `θ(t) = θ₀ + ω × t` (not iterative `next_position` steps) to eliminate drift over long prediction horizons
- [ ] **PHYS-04**: Agent never sends ships that reduce a planet's garrison below `max(production × 2, 5)` ships

### Targeting & Offense

- [ ] **TARG-01**: Before committing ships to a target, agent checks whether any enemy fleet will arrive at that target earlier than the player's fleet would, and adds those incoming ships to the effective garrison in `ships_needed`
- [ ] **TARG-02**: Target scoring uses an N-turn forward simulation (horizon ≥ 60 turns) to evaluate the future production value of a capture, not just current garrison size — integrates `production × (turns_remaining - arrival_turn)` minus cost
- [ ] **TARG-03**: Agent prioritises high-production planets when scores are comparable; production value multiplier applied per target

### Defense

- [ ] **DEF-01**: Each turn, agent identifies all enemy fleets heading toward owned planets and computes their expected arrival turn and ship count
- [ ] **DEF-02**: When an owned planet will be lost to an incoming threat, agent sends reinforcement ships from the nearest safe owned planet if reinforcement arrives before the threat
- [ ] **DEF-03**: When a planet is lost (ownership changed to enemy), agent immediately evaluates a counter-attack and executes it if the planet can be recaptured profitably within the next 10 turns

### Infrastructure

- [ ] **INFRA-01**: Agent logic is structured as an `agent/` package (`state.py`, `physics.py`, `scorer.py`, `planner.py`, `defense.py`) importable from `main.py`
- [ ] **INFRA-02**: Submission can be bundled as `submission.tar.gz` containing `main.py` + `agent/` package for Kaggle multi-file submission
- [ ] **INFRA-03**: Per-turn computation includes a time-budget guard: if elapsed time exceeds 0.8 seconds, truncate candidate evaluation and return best moves found so far

---

## v2 Requirements (deferred)

- Game phase awareness — distinct behaviour for early (expand), mid (contest), late (hold+snipe) turns
- Multi-fleet coordination — multiple source planets target same destination simultaneously
- Comet timing strategy — opportunistically capture comets at turns 50/150/250/350/450
- Crash-exploit detection — identify mutual-destruction opportunities between enemy fleets
- PettingZoo multi-agent environment wrapper for RL training
- StableBaselines3 PPO self-play training loop (production-advantage reward)
- 4-player mode tuning — adjusted aggression for 4-player maps

---

## Out of Scope

- Custom game engine rewrite — `kaggle-environments` is the simulator throughout
- Real-time pathfinding (A*) — fleets travel in straight lines only; no pathfinding needed
- ML-based target selection — rule-based scoring is sufficient for v1; RL comes in v2
- Fleet splitting into scouts — 1-ship fleets move at speed 1.0, not useful strategically

---

## Traceability

*Filled by roadmapper*

| REQ-ID | Phase |
|--------|-------|
| PHYS-01 | — |
| PHYS-02 | — |
| PHYS-03 | — |
| PHYS-04 | — |
| TARG-01 | — |
| TARG-02 | — |
| TARG-03 | — |
| DEF-01 | — |
| DEF-02 | — |
| DEF-03 | — |
| INFRA-01 | — |
| INFRA-02 | — |
| INFRA-03 | — |
