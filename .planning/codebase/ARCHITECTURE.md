<!-- refreshed: 2026-05-06 -->
# Architecture

**Analysis Date:** 2026-05-06

## System Overview

```text
┌──────────────────────────────────────────────────────────────────────┐
│                     Browser (Hash-based SPA)                         │
│  viewer/src/main.ts  →  router.ts  →  views/*.ts  →  components/*.ts │
└────────────────────────────┬─────────────────────────────────────────┘
                             │  HTTP /api/*  (proxy in dev, same-origin in prod)
                             ▼
┌──────────────────────────────────────────────────────────────────────┐
│                   FastAPI Backend  (Python 3.12)                      │
│   orbit_wars_app/main.py  →  api.py  (APIRouter /api)                │
│                                                                       │
│  ┌──────────────┐  ┌─────────────┐  ┌──────────────┐  ┌──────────┐  │
│  │ discovery.py │  │tournament.py│  │  match.py    │  │ external │  │
│  │ (agent zoo   │  │ (run mgmt + │  │  (run_match  │  │   .py    │  │
│  │  scan)       │  │ TrueSkill)  │  │  fast/faith) │  │ (fetch)  │  │
│  └──────────────┘  └─────────────┘  └──────┬───────┘  └──────────┘  │
│                                             │                         │
│                             ┌───────────────┴────────────────────┐   │
│                             │  fast mode          faithful mode   │   │
│                             │  (in-process)   (subprocess+HTTP)  │   │
│                             └──────────┬──────────────┬──────────┘   │
└────────────────────────────────────────┼──────────────┼──────────────┘
                                         │              │
                             ┌───────────┘     ┌────────▼──────────────┐
                             ▼                 │  agent_subprocess.py  │
┌───────────────────────┐   ┌──────────────┐  │  + agent_serve.py     │
│  kaggle-environments  │◄──│ kaggle-envs  │  │  (per-agent FastAPI   │
│  (orbit_wars engine)  │   │  make()/run()│  │   HTTP server)        │
└───────────────────────┘   └──────────────┘  └───────────────────────┘
                                         │
                                         ▼
┌──────────────────────────────────────────────────────────────────────┐
│                       Filesystem (persistent)                        │
│  agents/{baselines,external,mine}/<name>/main.py + agent.yaml        │
│  runs/<date-id>/{run.json, config.json, results.json, replays/*.json}│
│  runs/trueskill.json  (global TrueSkill state)                       │
│  replays/             (Kaggle-scraped episodes)                       │
└──────────────────────────────────────────────────────────────────────┘
```

## Component Responsibilities

| Component | Responsibility | File |
|-----------|----------------|------|
| FastAPI app | HTTP server, lifespan management, static mount | `orbit-wars-lab/orbit_wars_app/main.py` |
| API router | All `/api/*` endpoint handlers | `orbit-wars-lab/orbit_wars_app/api.py` |
| Discovery | Scan `agents/` tree, build `AgentInfo` list | `orbit-wars-lab/orbit_wars_app/discovery.py` |
| Tournament | Run orchestration, pairing, parallel workers, run.json writes | `orbit-wars-lab/orbit_wars_app/tournament.py` |
| Match runner | Execute single game (fast/faithful dispatch) | `orbit-wars-lab/orbit_wars_app/match.py` |
| Agent subprocess | Spawn per-agent HTTP server subprocess, handshake, teardown | `orbit-wars-lab/orbit_wars_app/agent_subprocess.py` |
| Agent serve | Micro FastAPI per agent, loaded via importlib, exposed over localhost | `orbit-wars-lab/orbit_wars_app/agent_serve.py` |
| TrueSkill store | Load/update/save TrueSkill ratings JSON atomically | `orbit-wars-lab/orbit_wars_app/trueskill_store.py` |
| Replay store | Save/load kaggle-environments replay JSON to disk | `orbit-wars-lab/orbit_wars_app/replay_store.py` |
| Schemas | Pydantic models shared by API and tournament | `orbit-wars-lab/orbit_wars_app/schemas.py` |
| External utils | Fetch/update external agents from Kaggle notebooks | `orbit-wars-lab/orbit_wars_app/external.py` |
| Kaggle auth | Read/write/validate `~/.kaggle/kaggle.json` | `orbit-wars-lab/orbit_wars_app/kaggle_auth.py` |
| Kaggle scraper | Download replay JSON from Kaggle episode API | `orbit-wars-lab/orbit_wars_app/kaggle_scraper.py` |
| Kaggle submissions | List/upload competition submissions via Kaggle CLI | `orbit-wars-lab/orbit_wars_app/kaggle_submissions.py` |
| SPA router | Hash-based client-side routing (`#/` → view dispatch) | `orbit-wars-lab/viewer/src/router.ts` |
| SPA main | View dispatch, theme init, hashchange listener | `orbit-wars-lab/viewer/src/main.ts` |
| API client (TS) | Typed fetch wrappers for all backend endpoints | `orbit-wars-lab/viewer/src/api.ts` |
| Views | Per-route render functions (quick-match, leaderboard, tournaments…) | `orbit-wars-lab/viewer/src/views/` |
| Components | Reusable DOM builders (agent-picker, embedded-replay, sidebar…) | `orbit-wars-lab/viewer/src/components/` |
| Standalone agent | Competition submission entry point (no lab framework) | `main.py` (repo root) |

## Pattern Overview

**Overall:** Layered monolith — single Python process serves both REST API and pre-built static frontend. The frontend is a framework-free TypeScript SPA using vanilla DOM manipulation.

**Key Characteristics:**
- No React/Vue/Angular — viewer uses imperative DOM rendering (`document.createElement`, innerHTML)
- Tournament parallelism via `ProcessPoolExecutor` (Python `multiprocessing`), not async workers
- Agent isolation: in-process (fast mode) or subprocess+HTTP (faithful mode)
- Filesystem is the primary data store — no database, no cache layer
- TrueSkill ratings mutate a single JSON file with atomic write (tmp → rename)

## Layers

**HTTP API Layer:**
- Purpose: Translate HTTP requests to domain calls, serialize responses
- Location: `orbit-wars-lab/orbit_wars_app/api.py`
- Contains: `APIRouter`, endpoint functions, `ProcessPoolExecutor` (`_executor`) for async tournament launch
- Depends on: `tournament.py`, `discovery.py`, `trueskill_store.py`, `kaggle_auth.py`, `kaggle_scraper.py`, `kaggle_submissions.py`
- Used by: Vite SPA frontend (via `/api/*`)

**Domain Layer:**
- Purpose: Core business logic — running tournaments, matches, rating updates
- Location: `orbit-wars-lab/orbit_wars_app/tournament.py`, `match.py`, `trueskill_store.py`
- Contains: `Tournament` class, `run_match_fast`, `run_match_faithful`, `TrueSkillStore`
- Depends on: `kaggle-environments`, `replay_store.py`, `discovery.py`, `schemas.py`
- Used by: `api.py`, CLI entry point `orbit-wars-tournament`

**Agent Execution Layer:**
- Purpose: Load and run agent code in isolation
- Location: `orbit-wars-lab/orbit_wars_app/agent_subprocess.py`, `orbit_wars_app/agent_serve.py`
- Contains: `spawn_agent`, `shutdown`, `load_agent`, per-agent FastAPI micro-server
- Depends on: `kaggle-environments` (UrlAgent protocol), `uvicorn`, `importlib`
- Used by: `match.py` (faithful mode)

**Infrastructure / Persistence Layer:**
- Purpose: Filesystem read/write abstractions
- Location: `orbit-wars-lab/orbit_wars_app/replay_store.py`, `trueskill_store.py`, `discovery.py`
- Contains: `save_replay`, `load_replay`, `scan_zoo`, `TrueSkillStore._load`/`.save`
- Depends on: standard library (`json`, `pathlib`, `shutil`)
- Used by: domain layer

**Frontend Layer:**
- Purpose: Browser SPA — routing, views, reusable components
- Location: `orbit-wars-lab/viewer/src/`
- Contains: `router.ts`, `main.ts`, `api.ts`, `views/*.ts`, `components/*.ts`
- Depends on: `@kaggle-environments/core` (replay player), backend `/api/*`
- Used by: end users via browser

**Agent Code Layer:**
- Purpose: Standalone game agents submitted to Kaggle competition
- Location: `agents/{baselines,external,mine}/<name>/main.py`, root `main.py`
- Contains: `def agent(obs)` functions, optional `agent.yaml` metadata
- Depends on: `kaggle_environments.envs.orbit_wars.orbit_wars` (Planet, Fleet namedtuples)
- Used by: `match.py` (loaded as files by `kaggle-environments`)

## Data Flow

### Quick Match (Browser → Replay)

1. User picks agents in browser → clicks Play (`viewer/src/views/quick-match.ts`)
2. `POST /api/tournaments` with `TournamentConfig` (`orbit-wars-lab/orbit_wars_app/api.py`)
3. `Tournament.run()` called in `ProcessPoolExecutor` thread (`tournament.py`)
4. `_run_match_in_worker()` dispatched per match pair (`tournament.py:66`)
5. `run_match_fast()` calls `kaggle_environments.make("orbit_wars").run([agent_paths])` (`match.py:32`)
6. `_extract_outcome(replay, agent_ids)` derives winner, scores, status (`match.py:93`)
7. `save_replay()` writes `runs/<id>/replays/<NNN>-<a>__vs__<b>.json` (`replay_store.py`)
8. `TrueSkillStore.update_match()` + `.save()` updates `runs/trueskill.json` (`trueskill_store.py`)
9. `run.json` / `results.json` written to `runs/<id>/`
10. Browser polls `GET /api/runs/{run_id}/progress` until `status == "completed"`
11. Browser fetches replay JSON and passes it to `@kaggle-environments/core` replay player

### Faithful Mode (subprocess path)

1. `run_match_faithful()` invoked instead of fast (`match.py:162`)
2. `spawn_agent(agent_dir)` launches `python -m orbit_wars_app.agent_serve --agent-dir <path>` per agent (`agent_subprocess.py:57`)
3. `agent_serve.py` emits `{"status":"ready","url":"http://127.0.0.1:<port>"}` on stdout
4. `kaggle_environments` uses UrlAgent, POSTs observations to each agent's HTTP server
5. `shutdown(handle)` terminates subprocesses after match concludes

### Agent Discovery

1. `scan_zoo(zoo_dir)` walks `agents/{baselines,external,mine}/*/main.py` (`discovery.py:20`)
2. Each directory's `agent.yaml` is parsed for metadata (name, tags, kernel_slug, etc.)
3. `AgentInfo` list returned — drives `/api/agents` endpoint and picker UI

**State Management (backend):**
- No in-memory session state — all state is on disk (`runs/`, `agents/`)
- `api._executor` is a module-level `ProcessPoolExecutor` singleton managed by FastAPI lifespan
- `TrueSkillStore` is instantiated per-request from `runs/trueskill.json`; writes are atomic (tmp rename)

**State Management (frontend):**
- No global store — each view manages its own local state via closure
- Route state is encoded in `location.hash`; views re-render on `hashchange`

## Key Abstractions

**`AgentInfo` (Pydantic model):**
- Purpose: Metadata for one agent, scanned from `agent.yaml` + filesystem
- Examples: `orbit-wars-lab/orbit_wars_app/schemas.py:19`
- Pattern: Pydantic v2 `BaseModel` with optional deprecated backward-compat fields

**`MatchOutcome` (dataclass):**
- Purpose: Structured result of a single game (winner, scores, turns, status, raw replay dict)
- Examples: `orbit-wars-lab/orbit_wars_app/match.py:13`
- Pattern: Python `@dataclass`, returned by both fast and faithful match runners

**`TournamentConfig` (Pydantic model):**
- Purpose: Validated config for a tournament run (agents, games_per_pair, mode, format, shape)
- Examples: `orbit-wars-lab/orbit_wars_app/schemas.py:93`
- Pattern: Pydantic v2 model, serves as POST body and stored as `config.json`

**`TrueSkillStore` (class):**
- Purpose: Load, update, and persist TrueSkill ratings per `(agent_id, format)` pair
- Examples: `orbit-wars-lab/orbit_wars_app/trueskill_store.py:30`
- Pattern: File-backed singleton per path; atomic write via `.json.tmp` → rename

**`Agent` function contract:**
- Purpose: The universal agent interface for the Orbit Wars game
- Pattern: `def agent(obs) -> list[list]` where obs has `.planets`, `.fleets`, `.player`; returns `[[from_planet_id, angle, num_ships], ...]`
- Examples: root `main.py`, `orbit-wars-lab/agents/baselines/*/main.py`

**`Route` (TypeScript discriminated union):**
- Purpose: Type-safe representation of all SPA routes
- Examples: `orbit-wars-lab/viewer/src/router.ts:17`
- Pattern: TypeScript union of `{ view: string; ...params }` objects; `parseHash` / `navigate` functions

## Entry Points

**FastAPI backend:**
- Location: `orbit-wars-lab/orbit_wars_app/main.py`
- Triggers: `uvicorn orbit_wars_app.main:app --host 127.0.0.1 --port 8000`
- Responsibilities: Mounts API router, manages executor lifespan, serves `viewer/dist` as static files in production

**Tournament CLI:**
- Location: `orbit-wars-lab/orbit_wars_app/tournament.py` (via `if __name__ == "__main__"` + console_scripts)
- Triggers: `orbit-wars-tournament gauntlet <challenger> --games-per-pair 10` or `python -m orbit_wars_app.tournament ...`
- Responsibilities: Headless tournament run without web UI

**Agent HTTP server:**
- Location: `orbit-wars-lab/orbit_wars_app/agent_serve.py`
- Triggers: `python -m orbit_wars_app.agent_serve --agent-dir <path>` (spawned by `agent_subprocess.py`)
- Responsibilities: Load `main.py` via `importlib`, serve `/` and `/act` POST endpoints (UrlAgent protocol)

**Viewer SPA:**
- Location: `orbit-wars-lab/viewer/src/main.ts`
- Triggers: Browser loads `index.html`, `main.ts` bootstraps hash routing
- Responsibilities: Route dispatch, theme init, view rendering

**Competition submission:**
- Location: root `main.py`
- Triggers: Kaggle environment calls `agent(obs)` each turn
- Responsibilities: Implement the nearest-planet sniper strategy; must be self-contained

## Architectural Constraints

- **Threading model:** FastAPI handles async HTTP on the main thread; tournament execution offloaded to `ProcessPoolExecutor`. Worker processes re-import `kaggle-environments` (~150MB RSS each) — executor is capped at 16 workers.
- **Global state:** `api._executor` (module-level `ProcessPoolExecutor`) in `orbit-wars-lab/orbit_wars_app/api.py`. `trueskill._env` (module-level `TrueSkill` instance) in `trueskill_store.py:21`.
- **Circular imports:** None detected. `api.py` imports from `tournament.py`, `discovery.py`, `trueskill_store.py`, etc. — all one-directional.
- **Agent sandboxing:** Faithful mode strips `KAGGLE_*` env vars from subprocess environment (`agent_subprocess.py:29`). Fast mode runs agents in-process — no isolation.
- **Filesystem as database:** No SQL or NoSQL store. All persistence is JSON files under `runs/` and `agents/`. Concurrent writes to `trueskill.json` are safe only because the `ProcessPoolExecutor` workers write replay files, and the main process updates TrueSkill serially after receiving `_WorkerResult`.
- **No hot-reload for agents:** Fast mode loads agent `main.py` via `kaggle-environments` file path at match start — code changes between matches are picked up automatically.

## Anti-Patterns

### In-process agent execution (fast mode)

**What happens:** `run_match_fast()` calls `kaggle_environments.make().run([path_to_main_py])` — all agent code runs in the same Python process as the FastAPI server (`match.py:56`).
**Why it's wrong:** A misbehaving or malicious agent can crash the server, leak memory, or access the server's environment variables (including Kaggle tokens).
**Do this instead:** Use faithful mode (`mode="faithful"`) for untrusted agents — it isolates each agent in its own subprocess with credentials stripped (`agent_subprocess.py`).

### Replay replay dict returned across process boundary

**What happens:** In early design the full replay dict (~5-10 MB) was returned from worker to main process through the `ProcessPoolExecutor` pipe.
**Why it's wrong:** Saturates IPC bandwidth; slow and serialization-heavy for round-robin tournaments.
**Do this instead:** Workers write the replay file directly to `runs/<id>/replays/` and return only the path string — as implemented in `_WorkerResult.replay_path` (`tournament.py:48`).

## Error Handling

**Strategy:** Defensive — match runners catch all exceptions and return a typed `MatchOutcome` with `status="crashed"` rather than propagating. API layer converts domain errors to HTTP exceptions.

**Patterns:**
- Match crashes: `try/except Exception` in `run_match_fast` and `run_match_faithful` return a `MatchOutcome(status="crashed", replay=_crashed_replay_skeleton(str(e)))` (`match.py:57`)
- Agent spawn failures: `RuntimeError`/`TimeoutError` caught in `run_match_faithful`, returned as `status="agent_failed_to_start"` (`match.py:186`)
- API errors: `HTTPException(status_code=404/400)` raised directly in route handlers (`api.py`)
- Path traversal: `_safe_subpath()` in `api.py:40` validates all path parameters stay inside parent dir

## Cross-Cutting Concerns

**Logging:** Python stdlib `logging` under `orbit_wars_app.*` namespace. `kaggle_environments` logger silenced to `WARNING` in worker processes (`tournament.py:51`). Frontend has no structured logging — browser `console.*` only.
**Validation:** Pydantic v2 models for all API request/response bodies. Agent YAML validated at scan time with graceful fallback and `last_error` propagation.
**Authentication:** Kaggle API token stored at `~/.kaggle/kaggle.json` (chmod 600). Read via `kaggle_auth.py`; never echoed to frontend. Agent subprocesses receive no Kaggle env vars.

---

*Architecture analysis: 2026-05-06*
