# Explorer Score Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the flat ETA-order Explorer logic in `26-Board_Explorer.py` with a two-priority system: intercept planets being claimed by enemies first, then pick the highest-scoring neutral planet using the Exploring score formula.

**Architecture:** Two changes to the `Board` class only: (1) replace the `elif planet.nature == "Explorer"` branch in `_create_actions`, (2) add `Board._explorer_score_action(planet)` helper method. Everything else in `26-Board_Explorer.py` is unchanged.

**Tech Stack:** Python 3, kaggle_environments, pytest

---

## File Map

| File | Action | Responsibility |
|------|--------|---------------|
| `26-Board_Explorer.py` | Modify | Replace Explorer branch + add `_explorer_score_action` |
| `tests/test_explorer_score.py` | Create | Formula unit test + valid-moves integration test |

---

### Task 1: Write failing tests

**Files:**
- Create: `tests/test_explorer_score.py`

- [ ] **Step 1: Write the test file**

```python
# tests/test_explorer_score.py
import os
import sys
import importlib.util

import kaggle_environments as ke

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_module(filename, module_name):
    path = os.path.join(ROOT, filename)
    spec = importlib.util.spec_from_file_location(module_name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


class MockTarget:
    def __init__(self, production, ships, owner=-1):
        self.production = production
        self.ships = ships
        self.owner = owner


def test_explorer_score_formula_picks_best():
    """Score = production * (T - ships/planet_prod). Best target has max score."""
    planet_production = 4
    targets = [
        MockTarget(production=1, ships=10),  # T-cost=2.5, score=1*2.0=2.0
        MockTarget(production=5, ships=2),   # T-cost=0.5, score=5*4.0=20.0 ← wins
        MockTarget(production=3, ships=6),   # T-cost=1.5, score=3*3.0=9.0
    ]
    T = sum(t.ships for t in targets) / planet_production  # (10+2+6)/4 = 4.5
    best = max(targets, key=lambda t: t.production * (T - t.ships / planet_production))
    assert best is targets[1], f"Expected targets[1] (prod=5,ships=2) but got prod={best.production}"


def test_explorer_score_formula_prefers_high_prod_low_cost():
    """Given equal ships, prefer higher production."""
    planet_production = 2
    targets = [
        MockTarget(production=3, ships=4),  # score=3*(6-2)=12
        MockTarget(production=6, ships=4),  # score=6*(6-2)=24 ← wins
    ]
    T = sum(t.ships for t in targets) / planet_production  # (4+4)/2 = 4.0... wait
    # T = 8/2 = 4.0, time_cost each = 4/2 = 2.0, T - 2.0 = 2.0
    # score[0] = 3 * 2.0 = 6, score[1] = 6 * 2.0 = 12
    best = max(targets, key=lambda t: t.production * (T - t.ships / planet_production))
    assert best is targets[1]


def test_agent_26_produces_valid_moves():
    """Agent returns a list of [planet_id, angle, ships] triples (or empty list)."""
    mod = load_module("26-Board_Explorer.py", "agent26")
    env = ke.make("orbit_wars", debug=True)
    env.reset(2)
    obs = env.state[0].observation
    moves = mod.agent(obs)
    assert isinstance(moves, list)
    for move in moves:
        assert len(move) == 3, f"Move must be [id, angle, ships], got {move}"
        assert isinstance(move[2], (int, float)) and move[2] > 0, f"Ships must be positive: {move}"


def test_explorer_score_empty_neutral_returns_no_action():
    """_explorer_score_action returns [] when no neutral neighbours exist."""
    mod = load_module("26-Board_Explorer.py", "agent26b")

    class FakePlanet:
        NB_NEARBY_OUT_OF = 5
        owner = 0
        production = 3
        nearby_planets = [
            MockTarget(production=2, ships=5, owner=1),  # enemy, not neutral
        ]

    class FakeBoard:
        _explorer_score_action = mod.Board._explorer_score_action
        def _create_one_action(self, planet, target, ships_needed=None):
            return [[planet.owner, 0.0, 1]]

    result = FakeBoard()._explorer_score_action(FakePlanet())
    assert result == [], f"Expected [] when no neutral planets, got {result}"


def test_explorer_score_zero_production_returns_no_action():
    """_explorer_score_action returns [] when planet.production == 0."""
    mod = load_module("26-Board_Explorer.py", "agent26c")

    class FakePlanet:
        NB_NEARBY_OUT_OF = 5
        owner = 0
        production = 0  # can't compute score
        nearby_planets = [MockTarget(production=2, ships=5, owner=-1)]

    class FakeBoard:
        _explorer_score_action = mod.Board._explorer_score_action
        def _create_one_action(self, planet, target, ships_needed=None):
            return [[0, 0.0, 1]]

    result = FakeBoard()._explorer_score_action(FakePlanet())
    assert result == [], f"Expected [] when production==0, got {result}"
```

- [ ] **Step 2: Run tests to verify they fail**

```
pytest tests/test_explorer_score.py -v
```

Expected: `test_explorer_score_formula_picks_best` and `test_explorer_score_formula_prefers_high_prod_low_cost` may PASS (pure formula, no Board needed). `test_agent_26_produces_valid_moves` should PASS if agent runs. `test_explorer_score_empty_neutral_returns_no_action` and `test_explorer_score_zero_production_returns_no_action` will FAIL because `_explorer_score_action` does not exist yet on Board.

---

### Task 2: Implement Explorer logic in `26-Board_Explorer.py`

**Files:**
- Modify: `26-Board_Explorer.py`

Two edits: replace the Explorer branch in `_create_actions`, and add `_explorer_score_action` after it.

- [ ] **Step 1: Replace the Explorer branch in `_create_actions`**

Find this exact block in `26-Board_Explorer.py`:

```python
        elif planet.nature == "Explorer":
            for target in planet.nearby_planets[:planet.NB_NEARBY_OUT_OF]:
                if target.nexts[NB_FORECAST_STEPS - 1]['owner'] != planet.owner:
                    action = self._create_one_action(planet, target)
                    if action != []:
                        actions += action
                        return actions
```

Replace it with:

```python
        elif planet.nature == "Explorer":
            # Priority 1: intercept planets being claimed by an enemy
            for target in planet.nearby_planets[:NEIGHBOURHOOD]:
                if target.nexts[NB_FORECAST_STEPS - 1]['owner'] not in (-1, planet.owner):
                    action = self._create_one_action(planet, target)
                    if action != []:
                        return action
            # Priority 2: highest Exploring score among neutral neighbours
            return self._explorer_score_action(planet)
```

- [ ] **Step 2: Add `_explorer_score_action` after `_create_actions`**

Immediately after the closing `return actions` line of `_create_actions`, add this new method:

```python
    def _explorer_score_action(self, planet) -> list:
        neutral = [t for t in planet.nearby_planets[:NEIGHBOURHOOD] if t.owner == -1]
        if not neutral or planet.production == 0:
            return []
        T = sum(t.ships for t in neutral) / planet.production
        best = max(neutral, key=lambda t: t.production * (T - t.ships / planet.production))
        return self._create_one_action(planet, best)
```

- [ ] **Step 3: Verify the file is importable**

```
python -c "import importlib.util, sys; spec = importlib.util.spec_from_file_location('t', '26-Board_Explorer.py'); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); print(hasattr(mod.Board, '_explorer_score_action'))"
```

Expected output (after INFO lines): `True`

---

### Task 3: Run all tests and commit

**Files:**
- Run: `tests/test_explorer_score.py`

- [ ] **Step 1: Run the full test suite**

```
pytest tests/test_explorer_score.py -v -s
```

Expected: `5 passed` (the `-s` lets the `print(obs.remainingOverageTime)` from the integration test through).

If `test_explorer_score_empty_neutral_returns_no_action` or `test_explorer_score_zero_production_returns_no_action` fail, verify that `_explorer_score_action` correctly checks `if not neutral or planet.production == 0: return []` before the `max()` call.

- [ ] **Step 2: Commit**

```bash
git add 26-Board_Explorer.py tests/test_explorer_score.py
git commit -m "feat: Explorer two-priority system with Exploring score (26-Board_Explorer.py)"
```
