---
phase: 1
slug: physics-infrastructure
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-05-11
---

# Phase 1 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 9.0.2 |
| **Config file** | none — Wave 0 creates `pytest.ini` |
| **Quick run command** | `python -m pytest tests/ -x -q` |
| **Full suite command** | `python -m pytest tests/ -v` |
| **Estimated runtime** | ~5 seconds |

---

## Sampling Rate

- **After every task commit:** Run `python -m pytest tests/ -x -q`
- **After every plan wave:** Run `python -m pytest tests/ -v`
- **Before `/gsd-verify-work`:** Full suite must be green
- **Max feedback latency:** 5 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| PHYS-01 | 01 | 1 | PHYS-01 | — | N/A | unit | `python -m pytest tests/test_physics.py::test_spawn_offset_correction -x` | ❌ W0 | ⬜ pending |
| PHYS-02 | 01 | 1 | PHYS-02 | — | N/A | unit | `python -m pytest tests/test_physics.py::test_sun_collision_rejection -x` | ❌ W0 | ⬜ pending |
| PHYS-03 | 01 | 1 | PHYS-03 | — | N/A | unit | `python -m pytest tests/test_physics.py::test_orbital_position_no_drift -x` | ❌ W0 | ⬜ pending |
| PHYS-04 | 02 | 1 | PHYS-04 | — | N/A | unit | `python -m pytest tests/test_planner.py::test_garrison_floor -x` | ❌ W0 | ⬜ pending |
| INFRA-01 | 02 | 1 | INFRA-01 | — | N/A | smoke | `python -c "from agent.planner import select_moves; print('ok')"` | ❌ W0 | ⬜ pending |
| INFRA-03 | 02 | 1 | INFRA-03 | — | N/A | unit | `python -m pytest tests/test_planner.py::test_time_budget_guard -x` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/__init__.py` — package init
- [ ] `tests/test_physics.py` — stubs for PHYS-01, PHYS-02, PHYS-03
- [ ] `tests/test_planner.py` — stubs for PHYS-04, INFRA-03
- [ ] `pytest.ini` — project root config

*pytest 9.0.2 already installed — no install task needed.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Fleet arrives within 1 turn on moving planet | PHYS-01 | End-to-end requires backtester | Run simulate.py, inspect replay, compare predicted vs actual arrival |
| No fleet path crosses sun in 10+ games | PHYS-02 | Requires live game replay | Run 10 games vs yuriygreben, inspect each fleet trajectory in replay viewer |
| Garrison floor never violated | PHYS-04 | Requires full game scan | Run 10 games, grep logs for garrison below floor |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 5s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
