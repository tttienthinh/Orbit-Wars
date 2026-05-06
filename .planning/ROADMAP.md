# Roadmap: Orbit Wars Competitive Agent

## Overview

Starting from a working nearest-planet sniper baseline, the agent evolves through four
phases: first fixing foundational physics bugs and establishing the multi-file package
structure, then layering in production-value targeting with forward simulation, then adding
threat detection and defensive responses, and finally packaging the submission and verifying
the agent beats yuriygreben's "architect" in local backtests. Every phase delivers a
measurable, independently verifiable capability that compounds into the v1 milestone.

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

Decimal phases appear between their surrounding integers in numeric order.

- [ ] **Phase 1: Physics & Infrastructure** - Fix foundational physics bugs and establish the agent package structure
- [ ] **Phase 2: Targeting & Offense** - Replace proximity scoring with production-value forward simulation
- [ ] **Phase 3: Defense & Counter-Attack** - Detect threats, reinforce, and counter-attack lost planets
- [ ] **Phase 4: Submission & Benchmark** - Package for Kaggle and verify the agent beats "architect" locally

## Phase Details

### Phase 1: Physics & Infrastructure
**Goal**: The agent's physics engine is correct and its code is organized as a proper package
**Depends on**: Nothing (first phase)
**Requirements**: PHYS-01, PHYS-02, PHYS-03, PHYS-04, INFRA-01, INFRA-03
**Success Criteria** (what must be TRUE):
  1. A fleet sent from a large planet to a small fast-moving planet arrives within 1 turn of predicted — spawn-offset correction is verified in the backtester replay
  2. No fleet path in any backtested game crosses the sun disc — sun-collision rejection is confirmed via replay inspection
  3. Orbital position prediction for any planet matches the game engine's position at turn 60+ with zero visible drift (cumulative formula, not iterative steps)
  4. The agent never sends a fleet that leaves a garrison below `max(production × 2, 5)` ships — confirmed across 10+ backtested games
  5. `agent/` package (`state.py`, `physics.py`, `scorer.py`, `planner.py`, `defense.py`) exists and `main.py` imports cleanly; a time-budget guard truncates evaluation above 0.8 s
**Plans**: TBD

### Phase 2: Targeting & Offense
**Goal**: The agent selects targets by future production value, accounts for in-flight enemy fleets, and prioritises high-production planets
**Depends on**: Phase 1
**Requirements**: TARG-01, TARG-02, TARG-03
**Success Criteria** (what must be TRUE):
  1. The agent does not send a fleet to a target that an enemy fleet will reach first — fleet-race awareness visible in turn-by-turn print logs
  2. Target scoring integrates `production × (turns_remaining - arrival_turn) - cost` over a ≥ 60-turn horizon — verifiable by inspecting scorer output in logs
  3. When two targets have comparable scores, the agent chooses the higher-production one — production multiplier confirmed in at least 5 log comparisons
  4. Win rate vs "architect" measurably improves over Phase 1 baseline (TrueSkill rating rises) after 20+ backtested games
**Plans**: TBD

### Phase 3: Defense & Counter-Attack
**Goal**: The agent identifies threats to owned planets, reinforces before they fall, and immediately evaluates counter-attacks on lost planets
**Depends on**: Phase 2
**Requirements**: DEF-01, DEF-02, DEF-03
**Success Criteria** (what must be TRUE):
  1. Every turn, the agent prints or logs all incoming enemy fleets with expected arrival turn and ship count — visible in episode logs across backtested games
  2. When an owned planet is under threat and a safe neighbour can arrive in time, a reinforcement fleet is dispatched — confirmed in at least 3 replay scenarios
  3. Within 1 turn of losing a planet, the agent evaluates and (if profitable) launches a counter-attack fleet targeting recapture within 10 turns — visible in replay
  4. Win rate vs "architect" further improves over Phase 2 (TrueSkill continues rising) after 20+ backtested games
**Plans**: TBD

### Phase 4: Submission & Benchmark
**Goal**: The agent is packaged for Kaggle multi-file submission and beats yuriygreben's "architect" agent in local backtests
**Depends on**: Phase 3
**Requirements**: INFRA-02
**Success Criteria** (what must be TRUE):
  1. `submission.tar.gz` containing `main.py` + `agent/` package extracts cleanly and the agent function runs without import errors in a fresh Python 3.12 environment
  2. The agent achieves a positive TrueSkill rating delta vs "architect" across a 50-game backtesting series in orbit-wars-lab (win rate > 50%)
  3. At least one full game replay shows the agent surviving to late game (turn 400+) against "architect" without timing out
**Plans**: TBD
**UI hint**: yes

## Progress

**Execution Order:**
Phases execute in numeric order: 1 → 2 → 3 → 4

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Physics & Infrastructure | 0/TBD | Not started | - |
| 2. Targeting & Offense | 0/TBD | Not started | - |
| 3. Defense & Counter-Attack | 0/TBD | Not started | - |
| 4. Submission & Benchmark | 0/TBD | Not started | - |
