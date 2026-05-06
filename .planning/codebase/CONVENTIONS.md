# Coding Conventions

**Analysis Date:** 2026-05-06

## Naming Patterns

**Files:**
- Snake_case for Python modules: `discovery.py`, `trueskill_store.py`, `replay_store.py`, `agent_subprocess.py`
- Test files prefixed with `test_`: `test_discovery.py`, `test_schemas.py`
- Agent entry points are always named `main.py`

**Functions:**
- Public functions: `snake_case` — `scan_zoo`, `run_match_fast`, `save_replay`, `make_match_filename`
- Private/internal helpers: prefixed with `_` — `_build_agent_info`, `_extract_outcome`, `_zoo_root`, `_safe_subpath`, `_filter_agents_by_tags`
- Worker functions (for `ProcessPoolExecutor`): prefixed with `_run_*` — `_run_match_in_worker`, `_run_and_clear`

**Variables:**
- `snake_case` throughout
- Module-level loggers: `_log = logging.getLogger(__name__)`
- Module-level constants: `UPPER_SNAKE_CASE` — `TS_MU_0`, `TS_SIGMA_0`, `VALID_BUCKETS`, `FIXTURES_DIR`, `KAGGLE_API_ROOT`

**Types / Classes:**
- `PascalCase` for classes: `TrueSkillStore`, `Tournament`, `MatchOutcome`, `KaggleAuthError`, `KaggleCliError`
- Pydantic models: `PascalCase` with `BaseModel` — `AgentInfo`, `Rating`, `MatchResult`, `RunSummary`, `TournamentConfig`, `KaggleSubmission`
- Dataclasses: `PascalCase` — `_WorkerResult`, `MatchOutcome`
- Type aliases: `PascalCase` — `Bucket`, `Format`, `Mode`, `TournamentShape`, `MatchStatus`

## Code Style

**Formatting:**
- Tool: `ruff` (formatter + linter, configured in `orbit-wars-lab/pyproject.toml`)
- Line length: 100 characters (`line-length = 100`)
- Target version: `py312`

**Linting:**
- Ruff rule sets: `E` (pycodestyle), `F` (pyflakes), `W` (warnings), `I` (isort), `UP` (pyupgrade)
- `E501` (line too long) is ignored — formatter handles wrapping
- Tests exempt from `F401` (unused imports) and `F811` (redefinition) — normal in fixtures

**Run linting:**
```bash
make lint
# expands to:
.venv/bin/ruff check orbit_wars_app/ tests/
```

## Import Organization

**Order (enforced by ruff `I` ruleset):**
1. `from __future__ import annotations` — always first when present
2. Standard library imports (alphabetical within group): `import json`, `import os`, `from pathlib import Path`
3. Third-party imports: `from fastapi import ...`, `from pydantic import ...`
4. Local package imports (relative): `from .discovery import scan_zoo`, `from . import kaggle_auth`

**Pattern:**
- All `orbit_wars_app` source files begin with `from __future__ import annotations`
- All test files begin with `from __future__ import annotations`
- Relative imports used inside `orbit_wars_app`: `from .schemas import AgentInfo`
- In tests, package is imported by full name: `from orbit_wars_app.discovery import scan_zoo`

**Path Aliases:**
- None — no path aliases; imports use full dotted names

## Error Handling

**Domain error types:**
- `KaggleAuthError(status_code: int, message: str)` in `orbit_wars_app/kaggle_auth.py` — raised on token parse/IO failures, carries HTTP status for API translation
- `KaggleCliError(status_code: int, message: str)` in `orbit_wars_app/kaggle_submissions.py` — similar pattern for CLI call failures
- Custom exceptions always carry `status_code` + `message` attributes so API handlers can re-raise as `HTTPException`

**API error translation pattern:**
```python
try:
    return kaggle_submissions.list_my_submissions()
except KaggleCliError as e:
    raise HTTPException(status_code=e.status_code, detail=e.message)
```

**Non-fatal failures:**
- Agent zoo scan: errors in `agent.yaml` parsing are stored in `AgentInfo.last_error`; the agent is still included (graceful degradation)
- Match crashes: return a `MatchOutcome` with `status="crashed"` and a `_crashed_replay_skeleton` rather than raising
- Path traversal: rejected early with `HTTPException(400)` via `_safe_subpath()` in `orbit_wars_app/api.py`

**No bare `except`:** Specific exception types are always caught (`yaml.YAMLError`, `json.JSONDecodeError`, `ValueError`, `RuntimeError`).

## Logging

**Framework:** Standard library `logging`

**Logger creation:**
```python
_log = logging.getLogger(__name__)
```
Used in `orbit_wars_app/discovery.py` and `orbit_wars_app/tournament.py`. Module-level, prefixed with `_`.

**Log levels:**
- `_log.warning(...)` for deprecated YAML fields, agent scan anomalies
- Worker process: `logging.getLogger("kaggle_environments").setLevel(logging.WARNING)` to suppress noisy third-party output
- No `_log.debug` or `_log.info` calls visible in app code — lean logging policy

## Comments

**Module-level docstrings:**
- Every module has a one-line or short paragraph `"""..."""` docstring explaining purpose
- Examples: `"""Scan agents/ tree and return AgentInfo list."""`, `"""Persistent TrueSkill ratings per agent_id per format ('2p' / '4p')."""`

**Inline comments:**
- Used liberally to explain non-obvious decisions: algorithm choices, edge-case handling, Kaggle API quirks
- Bilingual: English for code logic; Polish comments appear in test descriptions for business-level assertions (`# Agent w nieznanym buckecie — pomijany`)

**Field-level docstrings (Pydantic):**
- `Field(default=None, description="...")` used on `AgentInfo` fields with non-obvious semantics

**Type: ignore comments:**
- `# type: ignore[arg-type]` used sparingly where Literal type narrowing is impractical at call sites (`orbit_wars_app/match.py`, `orbit_wars_app/tournament.py`, `orbit_wars_app/api.py`)

## Function Design

**Size:** Functions are kept focused; private helpers extract logic from public functions (e.g., `_build_agent_info` from `scan_zoo`, `_extract_outcome` from `run_match_fast`)

**Parameters:**
- Keyword-only enforcement via `*` for optional params that could be confused positionally: `run_match_fast(agent_ids, agent_paths, *, seed=0)`
- Named tuple / dataclass for multi-field return values (`MatchOutcome` dataclass, `_WorkerResult` dataclass)

**Return Values:**
- Functions return domain objects (`AgentInfo`, `MatchOutcome`, `Rating`), not raw dicts, from the `orbit_wars_app` layer
- API route handlers return Pydantic models or plain `dict` (FastAPI serializes both)

## Module Design

**Exports:**
- No `__all__` defined; imports are explicit at call sites
- Internal helpers are name-prefixed with `_` to signal non-public status

**Barrel Files:**
- None — `orbit_wars_app/__init__.py` exposes only `__version__`; consumers import from specific modules

## Pydantic Model Conventions

- All models inherit from `pydantic.BaseModel`
- Optional fields use `Optional[T] = None` (not `T | None` with default) for clarity
- `Field(default_factory=list)` for mutable defaults
- Deprecated fields retained with `Field(default=None, description="DEPRECATED — ...")` pattern
- `Literal` type aliases defined at module level for reuse: `Bucket`, `Format`, `Mode`

---

*Convention analysis: 2026-05-06*
