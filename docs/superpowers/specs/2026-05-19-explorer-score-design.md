# Explorer Score Design

**Date:** 2026-05-19  
**File:** `26-Board_Explorer.py` (based on `25-Board_refactor.py`)  
**Scope:** Explorer planet decision logic only — no other changes.

---

## Goal

Replace the Explorer planet's flat ETA-order attack with a two-priority system:

1. **Intercept** neutral planets being claimed by an enemy before we can capture them.
2. **Score-based selection** — when no interception is needed, pick the neutral planet with the highest Exploring score.

---

## Constants (unchanged from 25)

```python
NEIGHBOURHOOD = 5       # nearest planets considered (by ETA)
NB_FORECAST_STEPS = 20  # forward simulation horizon
```

---

## Explorer Decision Logic

### Context

A planet has `nature == "Explorer"` only when `nb_nearby_enemy == 0` and `nb_nearby_neutral >= 1`. This guarantees that all planets in `nearby_planets[:NEIGHBOURHOOD]` are either neutral (`owner == -1`) or owned by the player. There are no enemies in the neighbourhood.

### Priority 1 — Intercept

Iterate `nearby_planets[:NEIGHBOURHOOD]` in ETA order. For each target:

```
if target.nexts[NB_FORECAST_STEPS - 1]['owner'] not in (-1, planet.owner):
```

→ An enemy is about to claim this neutral planet. Attack it immediately with `_create_one_action(planet, target)` (default ships: enough to capture). Return on first successful action.

### Priority 2 — Exploring Score

If no interception target was actionable, score all currently neutral neighbours and attack the best one.

**Score formula:**

```
neutral = [t for t in nearby_planets[:NEIGHBOURHOOD] if t.owner == -1]
T       = sum(t.ships for t in neutral) / planet.production
score(t) = t.production × (T − t.ships / planet.production)
```

Where:
- `t.ships / planet.production` = turns of own production needed to pay for capturing `t`
- `T` = total time cost of all neutral neighbours
- `T − time_cost(t)` = opportunity cost of NOT attacking `t` (time spent on everything else)
- `score(t)` = production gained × opportunity cost → rewards high-production, low-cost targets

Pick `best = argmax(score)`, call `_create_one_action(planet, best)`.

**Edge cases:**
- If `neutral` is empty: return `[]`
- If `planet.production == 0`: return `[]` (avoids division by zero; planet can't fund an attack)

---

## Code Changes

### `Board._create_actions(planet)` — Explorer branch (replace existing)

```python
elif planet.nature == "Explorer":
    for target in planet.nearby_planets[:NEIGHBOURHOOD]:
        if target.nexts[NB_FORECAST_STEPS - 1]['owner'] not in (-1, planet.owner):
            action = self._create_one_action(planet, target)
            if action != []:
                return action
    return self._explorer_score_action(planet)
```

### New method `Board._explorer_score_action(planet)`

```python
def _explorer_score_action(self, planet) -> list:
    neutral = [t for t in planet.nearby_planets[:NEIGHBOURHOOD] if t.owner == -1]
    if not neutral or planet.production == 0:
        return []
    T = sum(t.ships for t in neutral) / planet.production
    best = max(neutral, key=lambda t: t.production * (T - t.ships / planet.production))
    return self._create_one_action(planet, best)
```

---

## What Does NOT Change

- All other planet types (Comet, Supplier, Conqueror) — unchanged.
- `Board.__init__`, `_run_simulation`, `get_moves`, `_create_one_action` — unchanged.
- `Planet` class — unchanged.
- `NEIGHBOURHOOD` and `NB_FORECAST_STEPS` constants — unchanged.
- `agent()` entry point — unchanged.
