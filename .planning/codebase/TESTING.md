# Testing Patterns

**Analysis Date:** 2026-05-06

## Test Framework

**Runner:**
- `pytest` >= 8.0
- Config: `orbit-wars-lab/pyproject.toml` (`[tool.pytest.ini_options]`)

**Async support:**
- `pytest-asyncio` >= 0.23
- `asyncio_mode = "auto"` — all `async def test_*` functions run automatically without explicit `@pytest.mark.asyncio` on the test (mark is used for clarity but not required)

**HTTP client for API tests:**
- `httpx.AsyncClient` with `ASGITransport` — tests hit the FastAPI app in-process, no real server needed

**Assertion Library:**
- Built-in `pytest` assertions + `pytest.approx` for float comparisons
- `pytest.raises` for exception assertions

**Run Commands:**
```bash
make test                      # Run all tests (verbose)
# expands to:
.venv/bin/pytest tests/ -v

.venv/bin/pytest tests/unit/   # Unit tests only
.venv/bin/pytest tests/integration/  # Integration tests only

# Coverage: not configured; no --cov flag in default setup
```

**pytest options (from pyproject.toml):**
```
testpaths = ["tests"]
asyncio_mode = "auto"
addopts = "-ra --strict-markers"
```
- `-ra` — show short summary for all non-passing tests
- `--strict-markers` — unknown markers cause errors (prevents typos in `@pytest.mark.asyncio`)

## Test File Organization

**Location:** Separate `tests/` tree, not co-located with source

**Structure:**
```
orbit-wars-lab/tests/
├── conftest.py              # Shared fixtures: tmp_zoo, tmp_runs, copy_fixture_agent()
├── __init__.py
├── fixtures/                # Static test data (agent directories, replay JSON)
│   ├── agent_ok/            # Valid agent with agent.yaml
│   ├── agent_broken_yaml/   # Agent with malformed YAML
│   ├── agent_crashing/      # Agent that raises at runtime
│   ├── agent_disabled/      # Agent with disabled: true
│   ├── agent_no_yaml/       # Agent without agent.yaml
│   ├── agent_raises/        # Agent that raises during execution
│   └── minimal_replay.json  # Minimal valid replay for load/save tests
├── unit/                    # Pure logic tests, no real matches or HTTP
│   ├── test_agent_serve_loader.py
│   ├── test_agent_subprocess.py
│   ├── test_discovery.py
│   ├── test_external.py
│   ├── test_kaggle_auth.py
│   ├── test_kaggle_submissions.py
│   ├── test_replay_store.py
│   ├── test_schemas.py
│   ├── test_tournament_callback.py
│   ├── test_tournament_cli.py
│   ├── test_tournament_quickmatch.py
│   └── test_trueskill_store.py
└── integration/             # Run real matches, real FastAPI app, real filesystem
    ├── test_api_agents.py
    ├── test_api_health.py
    ├── test_api_kaggle_submissions.py
    ├── test_api_ratings.py
    ├── test_api_replays.py
    ├── test_api_runs.py
    ├── test_api_tournaments.py
    ├── test_cli.py
    ├── test_match_faithful.py
    ├── test_match_fast.py
    ├── test_tournament.py
    └── test_zoo_real.py
```

**Naming:**
- Test files: `test_<module_name>.py`
- Test functions: `test_<what>_<scenario>` — `test_scan_empty_zoo`, `test_scan_finds_agent_with_yaml`, `test_fast_match_random_vs_random_terminates`

## Test Structure

**Suite Organization:**

Unit tests use flat `def test_*()` functions (no classes). Functions are grouped by feature within the file, sometimes separated by `# ---------- section name ----------` comments:

```python
"""Tests for orbit_wars_app.discovery."""
from __future__ import annotations
from pathlib import Path
import pytest
from orbit_wars_app.discovery import scan_zoo
from tests.conftest import copy_fixture_agent

def test_scan_empty_zoo(tmp_zoo: Path):
    agents = scan_zoo(tmp_zoo)
    assert agents == []

def test_scan_finds_agent_with_yaml(tmp_zoo: Path):
    copy_fixture_agent("agent_ok", tmp_zoo / "mine")
    agents = scan_zoo(tmp_zoo)
    assert len(agents) == 1
    a = agents[0]
    assert a.id == "mine/agent_ok"
```

**Patterns:**
- Arrange-Act-Assert with blank line separation between phases
- `tmp_path` (pytest built-in) used extensively for filesystem isolation
- Each test creates only what it needs — no shared mutable state between tests
- `caplog` fixture used for log-level assertions (e.g., deprecated YAML fields emit `WARNING`)

## Mocking

**Primary tool:** `pytest`'s `monkeypatch` fixture — preferred over `unittest.mock`

**Patterns:**

Environment variable injection:
```python
monkeypatch.setenv("ORBIT_WARS_RUNS_DIR", str(tmp_path))
monkeypatch.setenv("ORBIT_WARS_ZOO_DIR", str(PROJECT_ROOT / "agents"))
monkeypatch.delenv("KAGGLE_USERNAME", raising=False)
```

Function replacement via `monkeypatch.setattr`:
```python
monkeypatch.setattr(ks, "list_my_submissions", boom)
monkeypatch.setattr(ks, "fetch_agent_logs", fake_fetch)
```

**Hand-written fakes (no `MagicMock`):**
For external dependencies (Kaggle SDK), tests build explicit fake classes:
```python
class _FakeApi:
    """Minimal stand-in for kaggle.api.kaggle_api_extended.KaggleApi."""
    def __init__(self, items=None, auth_raises=False, ...):
        ...
    def authenticate(self): ...
    def competition_submissions(self, competition): ...
```

`SimpleNamespace` is used to create duck-typed test objects:
```python
def _fake_submission(**kw) -> SimpleNamespace:
    defaults = dict(ref=0, description="", date="...", status="COMPLETE", public_score="")
    defaults.update(kw)
    return SimpleNamespace(**defaults)
```

**What to Mock:**
- External CLI calls (Kaggle CLI)
- Kaggle SDK API client
- Environment variables pointing to filesystem paths
- Functions that call network endpoints

**What NOT to Mock:**
- The kaggle-environments engine — integration tests run it for real (`run_match_fast`)
- The FastAPI app itself — `ASGITransport` + `AsyncClient` exercises the full ASGI stack
- Filesystem I/O — `tmp_path` isolates it instead

## Fixtures and Factories

**Shared fixtures (`tests/conftest.py`):**

```python
@pytest.fixture
def tmp_zoo(tmp_path: Path) -> Path:
    """Tmp directory with agents/ skeleton. Tests populate it per-case."""
    zoo = tmp_path / "agents"
    (zoo / "baselines").mkdir(parents=True)
    (zoo / "external").mkdir(parents=True)
    (zoo / "mine").mkdir(parents=True)
    return zoo

@pytest.fixture
def tmp_runs(tmp_path: Path) -> Path:
    runs = tmp_path / "runs"
    runs.mkdir()
    return runs
```

**Fixture copy helper:**
```python
def copy_fixture_agent(fixture_name: str, dest: Path) -> Path:
    """Copy tests/fixtures/<fixture_name>/ into dest/<fixture_name>/."""
    src = FIXTURES_DIR / fixture_name
    target = dest / fixture_name
    shutil.copytree(src, target)
    return target
```

**Local builder helpers (per test file):**
```python
def _ainfo(aid: str, bucket: str, tags: list[str], disabled: bool = False) -> AgentInfo:
    return AgentInfo(id=aid, name=aid.split("/")[-1], bucket=bucket, ...)

def _write_agent(zoo: Path, bucket: str, name: str, yaml_data: dict | None, has_main: bool = True):
    """Helper: create agent folder with agent.yaml + main.py."""
```

**Static fixtures location:** `tests/fixtures/`
- `agent_ok/` — canonical valid agent (has `agent.yaml` + `main.py`)
- `agent_broken_yaml/` — malformed YAML for error path testing
- `agent_crashing/` — agent that crashes at runtime
- `agent_disabled/` — agent with `disabled: true`
- `agent_no_yaml/` — agent without metadata file
- `agent_raises/` — agent that raises an exception
- `minimal_replay.json` — smallest valid replay dict for roundtrip tests

**Integration fixture pattern (local to test file):**
```python
@pytest.fixture
def isolated_runs_dir(tmp_path: Path):
    runs = tmp_path / "runs"
    runs.mkdir()
    return runs
```

## Coverage

**Requirements:** Not enforced — no `--cov` flag in `addopts`, no coverage threshold configured

**View Coverage:**
```bash
# Not configured; run manually if needed:
.venv/bin/pytest tests/ --cov=orbit_wars_app --cov-report=term-missing
```

**Total test count:** 201 `def test_*` functions across all test files

## Test Types

**Unit Tests (`tests/unit/`):**
- Scope: Single module in isolation, no real matches or HTTP requests
- Covers: `discovery`, `schemas`, `trueskill_store`, `replay_store`, `kaggle_auth`, `kaggle_submissions`, `tournament` (CLI/callback/quickmatch logic), `agent_subprocess`, `external`
- Uses: `tmp_path`, `tmp_zoo`, `copy_fixture_agent`, hand-written fakes, `monkeypatch`

**Integration Tests (`tests/integration/`):**
- Scope: Multiple real components working together
- API tests: FastAPI app via `httpx.AsyncClient` + `ASGITransport(app=app)` — no server process
- Match tests: Real `kaggle-environments` engine runs actual game logic
- Tournament tests: Full `Tournament.run()` cycle writing to `tmp_path`
- Uses: `monkeypatch.setenv` to redirect filesystem roots, `asyncio.sleep` polling for background tasks

**E2E / Zoo Tests:**
- `tests/integration/test_zoo_real.py` — exercises real agents in `agents/` directory

## Common Patterns

**Async API Testing:**
```python
@pytest.mark.asyncio
async def test_api_agents_lists_baselines():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        r = await ac.get("/api/agents")
    assert r.status_code == 200
    data = r.json()
    assert isinstance(data, list)
```

**Polling for background completion:**
```python
for _ in range(60):
    p = await ac.get(f"/api/runs/{run_id}/progress")
    if p.status_code == 200 and p.json()["status"] == "completed":
        break
    await asyncio.sleep(0.5)
else:
    pytest.fail("Tournament never completed within 30 s")
```

**Error Testing:**
```python
def test_parse_token_rejects_empty():
    with pytest.raises(KaggleAuthError) as exc:
        ka.parse_token("")
    assert exc.value.status_code == 400
    assert "empty" in exc.value.message.lower()
```

**Float Approximation:**
```python
assert ra.mu == pytest.approx(rb.mu, abs=0.5)
```

**Log Assertion:**
```python
with caplog.at_level(logging.WARNING):
    agents = scan_zoo(zoo)
assert any("deprecated" in r.message.lower() for r in caplog.records)
```

**Filesystem Isolation for Integration Tests:**
```python
async def test_runs_list_after_one_tournament(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("ORBIT_WARS_RUNS_DIR", str(tmp_path))
    run_id = _run_one(tmp_path)
    # ... rest of test uses real files in tmp_path
```

---

*Testing analysis: 2026-05-06*
