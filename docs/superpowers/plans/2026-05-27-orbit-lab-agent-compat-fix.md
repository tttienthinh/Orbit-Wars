# Orbit-Lab Agent Compatibility Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix orbit-wars-lab so polars-based agents (72, 74) can run correctly in both fast and faithful modes without modifying any agent files.

**Architecture:** Two independent bugs are confirmed via live testing. Bug 1 (fast mode): `pyarrow` is not installed in the lab venv, but pandas 3.0 creates `StringDtype` for string columns, which `pl.from_pandas()` requires pyarrow to convert — add it to requirements. Bug 2 (faithful mode): `agent_serve.py` passes a plain `dict` as `obs`, but agents use attribute access (`obs.planets`) — wrap with `structify()`.

**Tech Stack:** Python 3.12, polars 1.41, pandas 3.0, pyarrow, kaggle-environments (Struct/structify), FastAPI, httpx, pytest.

---

## Confirmed Bug Evidence

Running `run_match_fast(["mine/74-Polars_scoring_upgrade", "baselines/random"])` produces:
- Steps 1–499: `status=ERROR` for agent 0 (polars agent crashes silently)
- Traceback (in debug=True mode): `ImportError: pyarrow is required for converting a pandas dataframe to Polars, unless each of its columns is a simple numpy-backed one`
- Root cause: `take_action()` calls `pl.from_pandas(df)` where `df` has a `nature` column with `StringDtype(storage='python')` — pandas 3.0 default for strings, which polars 1.x cannot convert without pyarrow.

For faithful mode: `obs = payload["state"]["observation"]` in `_act_handler` (agent_serve.py:77) returns a plain Python dict. Agents that call `obs.planets` raise `AttributeError: 'dict' object has no attribute 'planets'`.

---

## File Map

| File | Change |
|------|--------|
| `orbit-wars-lab/requirements.txt` | Add `polars>=1.0`, `pandas>=2.0`, `pyarrow>=14.0` |
| `orbit-wars-lab/pyproject.toml` | Same in `[project.dependencies]` |
| `orbit-wars-lab/orbit_wars_app/agent_serve.py` | Line 77: wrap obs with `structify()` |
| `orbit-wars-lab/tests/fixtures/agent_attr_access/main.py` | New: uses `obs.planets` attribute access |
| `orbit-wars-lab/tests/fixtures/agent_polars_basic/main.py` | New: calls `pl.from_pandas(df_with_string_col)` |
| `orbit-wars-lab/tests/unit/test_agent_subprocess.py` | Add: structify test via real subprocess HTTP |
| `orbit-wars-lab/tests/integration/test_match_fast.py` | Add: polars agent fast-mode smoke test |

---

### Task 1: Create test fixtures

**Files:**
- Create: `orbit-wars-lab/tests/fixtures/agent_attr_access/main.py`
- Create: `orbit-wars-lab/tests/fixtures/agent_polars_basic/main.py`

- [ ] **Step 1: Create `agent_attr_access` fixture**

  `orbit-wars-lab/tests/fixtures/agent_attr_access/main.py`:
  ```python
  def agent(obs):
      _ = obs.planets  # attribute access — fails with plain dict, works with Struct
      return []
  ```

- [ ] **Step 2: Create `agent_polars_basic` fixture**

  `orbit-wars-lab/tests/fixtures/agent_polars_basic/main.py`:
  ```python
  import polars as pl
  import pandas as pd

  def agent(obs):
      # pandas 3.0 uses StringDtype for string columns; pl.from_pandas needs pyarrow for that
      df = pd.DataFrame({"nature": ["moving", "fix"], "ships": [10, 20]})
      pl.from_pandas(df)
      return []
  ```

- [ ] **Step 3: Verify fixtures load cleanly**

  Run from `orbit-wars-lab/`:
  ```
  .venv\Scripts\python.exe -c "from orbit_wars_app.agent_serve import load_agent; print(load_agent('tests/fixtures/agent_attr_access'))"
  ```
  Expected: prints a callable (not None, no exception)

- [ ] **Step 4: Commit fixtures**

  ```bash
  git add orbit-wars-lab/tests/fixtures/agent_attr_access orbit-wars-lab/tests/fixtures/agent_polars_basic
  git commit -m "test: add agent_attr_access and agent_polars_basic fixtures for compat tests"
  ```

---

### Task 2: Fix faithful mode — structify obs in `agent_serve.py`

**Files:**
- Modify: `orbit-wars-lab/orbit_wars_app/agent_serve.py:12-20` (imports)
- Modify: `orbit-wars-lab/orbit_wars_app/agent_serve.py:77` (`_act_handler`)
- Test: `orbit-wars-lab/tests/unit/test_agent_subprocess.py`

- [ ] **Step 1: Write failing test**

  Add to `orbit-wars-lab/tests/unit/test_agent_subprocess.py`:
  ```python
  def test_act_handler_structifies_observation():
      """Agent in faithful mode must receive Struct obs so attribute access works."""
      handle = spawn_agent(FIXTURES / "agent_attr_access", agent_id="test/attr_access")
      try:
          r = httpx.post(
              f"{handle.url}/act",
              json={
                  "action": "act",
                  "configuration": {},
                  "state": {
                      "observation": {
                          "planets": [[1, 0, 25.0, 50.0, 5.0, 10, 2]],
                          "fleets": [],
                          "player": 0,
                          "remainingOverageTime": 60,
                      }
                  },
              },
              timeout=5,
          )
          data = r.json()
          assert data["action"] == [], f"Expected [] but got: {data['action']}"
      finally:
          shutdown(handle)
  ```

- [ ] **Step 2: Run test — verify it fails**

  Run from `orbit-wars-lab/`:
  ```
  .venv\Scripts\python.exe -m pytest tests/unit/test_agent_subprocess.py::test_act_handler_structifies_observation -v
  ```
  Expected: FAIL — `AssertionError: Expected [] but got: "BaseException::AttributeError: 'dict' object has no attribute 'planets'"`

- [ ] **Step 3: Apply fix — add `structify` import to `agent_serve.py`**

  In `orbit-wars-lab/orbit_wars_app/agent_serve.py`, change the imports block. Currently the file starts with:
  ```python
  from __future__ import annotations

  import argparse
  import importlib.util
  import json
  import socket
  import sys
  from pathlib import Path
  from typing import Callable, Optional
  ```

  Change to:
  ```python
  from __future__ import annotations

  import argparse
  import importlib.util
  import json
  import socket
  import sys
  from pathlib import Path
  from typing import Callable, Optional

  from kaggle_environments.utils import structify
  ```

- [ ] **Step 4: Apply fix — structify obs in `_act_handler`**

  In `orbit-wars-lab/orbit_wars_app/agent_serve.py`, the `_act_handler` function currently has:
  ```python
  async def _act_handler(request: Request) -> JSONResponse:
      try:
          payload = await request.json()
          obs = payload["state"]["observation"]
          cfg = payload.get("configuration", {})
  ```

  Change to:
  ```python
  async def _act_handler(request: Request) -> JSONResponse:
      try:
          payload = await request.json()
          obs = structify(payload["state"]["observation"])
          cfg = payload.get("configuration", {})
  ```

- [ ] **Step 5: Run test — verify it passes**

  Run from `orbit-wars-lab/`:
  ```
  .venv\Scripts\python.exe -m pytest tests/unit/test_agent_subprocess.py::test_act_handler_structifies_observation -v
  ```
  Expected: PASS

- [ ] **Step 6: Run the full unit test suite to check no regressions**

  ```
  .venv\Scripts\python.exe -m pytest tests/unit/ -v
  ```
  Expected: all pass (existing agent_ok tests still work — Struct is a dict subclass, `.get()` and `[]` access still work)

- [ ] **Step 7: Commit**

  ```bash
  git add orbit-wars-lab/orbit_wars_app/agent_serve.py orbit-wars-lab/tests/unit/test_agent_subprocess.py
  git commit -m "fix: structify obs in agent_serve._act_handler so faithful-mode agents get Struct"
  ```

---

### Task 3: Fix fast mode — add pyarrow (and polars/pandas) to lab requirements

**Files:**
- Modify: `orbit-wars-lab/requirements.txt`
- Modify: `orbit-wars-lab/pyproject.toml`
- Test: `orbit-wars-lab/tests/integration/test_match_fast.py`

- [ ] **Step 1: Write failing test**

  Add to `orbit-wars-lab/tests/integration/test_match_fast.py` (after the existing imports):
  ```python
  FIXTURES = PROJECT_ROOT / "tests" / "fixtures"
  ```
  Then add the test:
  ```python
  def test_fast_match_polars_agent_no_error_status():
      """Polars+pandas agent must not get ERROR status in fast mode (requires pyarrow)."""
      outcome = run_match_fast(
          agent_ids=["test/agent_polars_basic", "baselines/random"],
          agent_paths=[FIXTURES / "agent_polars_basic", _agent_path("baselines/random")],
          seed=42,
      )
      steps = outcome.replay.get("steps", [])
      assert len(steps) > 1, "Match must have more than 1 step"
      step1_status = steps[1][0].get("status")
      assert step1_status != "ERROR", (
          f"Polars agent errored on step 1 — likely missing pyarrow: status={step1_status}"
      )
  ```

- [ ] **Step 2: Run test — verify it fails**

  Run from `orbit-wars-lab/`:
  ```
  .venv\Scripts\python.exe -m pytest tests/integration/test_match_fast.py::test_fast_match_polars_agent_no_error_status -v
  ```
  Expected: FAIL — `AssertionError: Polars agent errored on step 1 — likely missing pyarrow: status=ERROR`

- [ ] **Step 3: Add dependencies to `requirements.txt`**

  Current `orbit-wars-lab/requirements.txt` ends with:
  ```
  # Dev / test
  pytest>=8.0
  pytest-asyncio>=0.23
  ```

  Add before the Dev/test block:
  ```
  # Agent runtime dependencies — polars-based agents (72, 74, 76) need these.
  # pandas 3.0 uses StringDtype for string columns; polars 1.x needs pyarrow to convert them.
  polars>=1.0
  pandas>=2.0
  pyarrow>=14.0
  ```

- [ ] **Step 4: Add dependencies to `pyproject.toml`**

  In `orbit-wars-lab/pyproject.toml`, the `[project.dependencies]` list currently ends with:
  ```toml
  "psutil>=5.9",
  ```

  Add after `"psutil>=5.9",`:
  ```toml
  "polars>=1.0",
  "pandas>=2.0",
  "pyarrow>=14.0",
  ```

- [ ] **Step 5: Install pyarrow in the current venv**

  Run from `orbit-wars-lab/`:
  ```
  .venv\Scripts\pip.exe install "pyarrow>=14.0"
  ```
  Expected output ends with: `Successfully installed pyarrow-<version>`

- [ ] **Step 6: Run test — verify it passes**

  Run from `orbit-wars-lab/`:
  ```
  .venv\Scripts\python.exe -m pytest tests/integration/test_match_fast.py::test_fast_match_polars_agent_no_error_status -v
  ```
  Expected: PASS

- [ ] **Step 7: Run the full fast-mode integration test suite**

  ```
  .venv\Scripts\python.exe -m pytest tests/integration/test_match_fast.py -v
  ```
  Expected: all 3 existing tests + new test = 4 PASS

- [ ] **Step 8: Commit**

  ```bash
  git add orbit-wars-lab/requirements.txt orbit-wars-lab/pyproject.toml orbit-wars-lab/tests/integration/test_match_fast.py
  git commit -m "fix: add polars/pandas/pyarrow to lab requirements for polars-based agents"
  ```

---

## Verification Smoke Test

After both tasks are done, run this end-to-end check from `orbit-wars-lab/`:

```
.venv\Scripts\python.exe -c "
from orbit_wars_app.match import run_match_fast
from pathlib import Path

outcome = run_match_fast(
    agent_ids=['mine/74-Polars_scoring_upgrade', 'baselines/random'],
    agent_paths=[Path('agents/mine/74-Polars_scoring_upgrade'), Path('agents/baselines/random')],
    seed=42,
)
steps = outcome.replay.get('steps', [])
step1_status = steps[1][0].get('status') if len(steps) > 1 else 'NO_STEP'
print('Status:', outcome.status)
print('Winner:', outcome.winner)
print('Step1 agent0 status:', step1_status)
assert step1_status != 'ERROR', f'Agent still crashing: {step1_status}'
print('PASS: 74-Polars_scoring_upgrade runs correctly in fast mode')
"
```

Expected output:
```
Status: ok
Winner: baselines/random  (or mine/74-Polars_scoring_upgrade — depends on seed)
Step1 agent0 status: ACTIVE
PASS: 74-Polars_scoring_upgrade runs correctly in fast mode
```
