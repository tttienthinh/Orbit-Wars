# Codebase Structure

**Analysis Date:** 2026-05-06

## Directory Layout

```
Orbit Wars/                        # Repo root — Kaggle competition workspace
├── main.py                        # Competition submission entry point (agent function)
├── agents.md                      # Getting-started guide for building agents
├── README.md                      # Game rules and observation reference
├── 00-replay.html                 # Standalone replay viewer (HTML, no lab required)
├── 00-setup.ipynb                 # Kaggle setup notebook
├── 01-Getting_started.py          # Sample agent scripts (numbered progression)
├── 02-Getting_started_only_non_moving.py
├── 03-Getting_started_one_fleet.py
├── 04-Getting_started_non_moving_rules.py
├── 05-Getting_started_moving_precision.py
├── 06-yuriygreben-architect_moving_precision.py  # Advanced reference agent
└── orbit-wars-lab/                # Subproject: local tournament runner + visualizer
    ├── orbit_wars_app/            # Python FastAPI backend
    │   ├── main.py                # App factory + static mount
    │   ├── api.py                 # All /api/* route handlers (APIRouter)
    │   ├── schemas.py             # Pydantic models (shared across layers)
    │   ├── discovery.py           # Agent zoo scanner (scan_zoo)
    │   ├── match.py               # Match runner (fast + faithful modes)
    │   ├── tournament.py          # Tournament orchestration + TrueSkill update
    │   ├── trueskill_store.py     # Persistent TrueSkill ratings (JSON)
    │   ├── replay_store.py        # Replay JSON save/load
    │   ├── agent_subprocess.py    # Spawn/teardown per-agent subprocess
    │   ├── agent_serve.py         # Per-agent FastAPI HTTP server (UrlAgent)
    │   ├── external.py            # Fetch/update external agents from Kaggle
    │   ├── kaggle_auth.py         # Kaggle token read/write/validate
    │   ├── kaggle_scraper.py      # Download replays from Kaggle episode API
    │   ├── kaggle_submissions.py  # List/upload competition submissions
    │   └── __init__.py            # Package version (__version__)
    ├── viewer/                    # Vite + TypeScript SPA frontend
    │   ├── src/
    │   │   ├── main.ts            # App bootstrap + hash routing dispatch
    │   │   ├── router.ts          # Route types + parseHash/navigate
    │   │   ├── api.ts             # Typed fetch client for all /api/* endpoints
    │   │   ├── theme.ts           # Light/dark mode toggle
    │   │   ├── timing.ts          # ETA / timing utilities
    │   │   ├── renderer.ts        # Replay renderer helpers
    │   │   ├── renderer/          # Renderer sub-modules
    │   │   ├── views/             # Per-route render functions
    │   │   │   ├── quick-match.ts
    │   │   │   ├── leaderboard.ts
    │   │   │   ├── tournaments.ts
    │   │   │   ├── tournament-detail.ts
    │   │   │   ├── replays.ts
    │   │   │   ├── replay.ts
    │   │   │   ├── kaggle-replay.ts
    │   │   │   ├── agents.ts
    │   │   │   ├── agent.ts
    │   │   │   ├── submissions.ts
    │   │   │   └── settings.ts
    │   │   ├── components/        # Reusable DOM component builders
    │   │   │   ├── agent-picker.ts
    │   │   │   ├── agent-card.ts
    │   │   │   ├── embedded-replay.ts
    │   │   │   ├── header-nav.ts
    │   │   │   ├── match-config-bar.ts
    │   │   │   ├── ratings-table.ts
    │   │   │   ├── runs-list.ts
    │   │   │   ├── sidebar-cards.ts
    │   │   │   └── theme-toggle.ts
    │   │   ├── utils/             # Shared utilities (escape, etc.)
    │   │   ├── styles/            # Scoped CSS per feature
    │   │   └── style.css          # Global CSS
    │   ├── index.html             # SPA shell
    │   ├── package.json           # Viewer package manifest
    │   ├── tsconfig.json          # TypeScript config (extends base)
    │   └── vite.config.ts         # Vite config + dev proxy to :8000
    ├── web/                       # Vendored npm packages
    │   └── core/                  # @kaggle-environments/core (Kaggle replay player)
    │       └── src/               # TypeScript source (compiled at build time)
    ├── agents/                    # Agent zoo (live-mounted in Docker)
    │   ├── baselines/             # Kaggle reference agents
    │   │   ├── random/
    │   │   ├── starter/
    │   │   └── nearest-sniper/
    │   ├── external/              # Curated public notebook agents
    │   │   ├── kashiwaba-rl/      # PPO neural-net agent
    │   │   ├── pilkwang-structured/
    │   │   ├── tamrazov-starwars/
    │   │   ├── sigmaborov-reinforce/
    │   │   ├── sigmaborov-starter/
    │   │   ├── yuriygreben-architect/
    │   │   ├── ykhnkf-distance-prioritized/
    │   │   └── pascal-orbitwork-v14/
    │   └── mine/                  # User's own agents (gitignored contents)
    │       ├── 04-Getting_started_non_moving_rules/
    │       ├── 05-Getting_started_moving_precision/
    │       └── xx-Do_nothing/
    ├── runs/                      # Tournament results (live-mounted in Docker)
    │   ├── trueskill.json         # Global TrueSkill leaderboard state
    │   └── <date-id>/             # Per-tournament run directory
    │       ├── run.json           # Run summary (status, progress)
    │       ├── config.json        # Tournament config snapshot
    │       ├── results.json       # All match results
    │       ├── trueskill.json     # TrueSkill snapshot at end of run
    │       └── replays/           # Match replay JSON files
    │           └── <NNN>-<a>__vs__<b>.json
    ├── replays/                   # Kaggle-scraped episode replays
    ├── tests/                     # Pytest test suite
    │   ├── unit/                  # Unit tests (isolated, no kaggle-envs)
    │   ├── integration/           # Integration tests (real HTTP, real files)
    │   ├── fixtures/              # Shared pytest fixtures
    │   └── conftest.py
    ├── docs/                      # Project documentation + screenshots
    ├── scripts/                   # Shell helper scripts
    │   ├── dev.sh                 # Start backend + viewer in native dev mode
    │   └── fetch-kashiwaba-weights.sh
    ├── pyproject.toml             # Python package metadata + tool config
    ├── requirements.txt           # Pinned dependencies for Docker build
    ├── package.json               # pnpm workspace root
    ├── pnpm-workspace.yaml        # Workspace packages: viewer, web/core
    ├── Dockerfile                 # Multi-stage: Node builder + Python runtime
    ├── docker-compose.yml         # Single-service compose (app on :6001)
    └── Makefile                   # Dev shortcuts
```

## Directory Purposes

**`orbit-wars-lab/orbit_wars_app/`:**
- Purpose: FastAPI application package — the entire Python backend
- Contains: Route handlers, domain logic, persistence utilities, external integrations
- Key files: `main.py` (app factory), `api.py` (all routes), `tournament.py` (core domain)

**`orbit-wars-lab/viewer/src/views/`:**
- Purpose: One TypeScript module per SPA route; each exports a `render*(container, ...params)` function
- Contains: Imperative DOM rendering code, local state management per view
- Key files: `quick-match.ts` (primary gameplay view), `tournament-detail.ts`, `leaderboard.ts`

**`orbit-wars-lab/viewer/src/components/`:**
- Purpose: Reusable DOM component builders shared across views
- Contains: Functions that accept a container element and return handles or nothing
- Key files: `agent-picker.ts` (agent selection UI), `embedded-replay.ts` (replay player wrapper)

**`orbit-wars-lab/agents/`:**
- Purpose: Agent zoo — all playable agents organized by bucket
- Contains: `main.py` (agent function) + optional `agent.yaml` (metadata) per agent
- Key files: Any `main.py` under `baselines/`, `external/`, or `mine/`

**`orbit-wars-lab/runs/`:**
- Purpose: Persistent tournament history and TrueSkill state
- Contains: `trueskill.json` (live leaderboard), per-run subdirectories with results + replays
- Generated: Yes (by the app at runtime)
- Committed: Partially — `trueskill.json` pre-seeded snapshot is committed; run subdirs are gitignored

**`orbit-wars-lab/web/core/`:**
- Purpose: Vendored `@kaggle-environments/core` (the official Kaggle replay player React component)
- Contains: TypeScript source; built at Docker image build time
- Generated: Build output `dist/` is gitignored; source is committed
- Committed: Yes (source only)

**`orbit-wars-lab/tests/`:**
- Purpose: Full test suite organized by scope
- Contains: `unit/` (isolated, mock-heavy), `integration/` (real HTTP with TestClient), `fixtures/`

## Key File Locations

**Entry Points:**
- `orbit-wars-lab/orbit_wars_app/main.py`: FastAPI app factory; `uvicorn orbit_wars_app.main:app`
- `orbit-wars-lab/orbit_wars_app/tournament.py`: CLI entry point `orbit-wars-tournament` (console_scripts)
- `orbit-wars-lab/viewer/src/main.ts`: SPA bootstrap; loaded by `viewer/index.html`
- `main.py`: Kaggle competition submission (root-level standalone agent)

**Configuration:**
- `orbit-wars-lab/pyproject.toml`: Python deps, tool config (ruff, pytest), console_scripts
- `orbit-wars-lab/requirements.txt`: Pinned deps used by Docker build
- `orbit-wars-lab/viewer/vite.config.ts`: Vite build + dev proxy config
- `orbit-wars-lab/docker-compose.yml`: Container orchestration (single service, live-mounted volumes)
- `orbit-wars-lab/pnpm-workspace.yaml`: Declares `viewer` and `web/core` as workspace packages

**Core Logic:**
- `orbit-wars-lab/orbit_wars_app/api.py`: All REST endpoints (~650 lines)
- `orbit-wars-lab/orbit_wars_app/tournament.py`: Tournament orchestration + parallel execution (~400 lines)
- `orbit-wars-lab/orbit_wars_app/match.py`: Match runner (fast + faithful modes)
- `orbit-wars-lab/orbit_wars_app/trueskill_store.py`: Rating persistence

**Schemas / Types:**
- `orbit-wars-lab/orbit_wars_app/schemas.py`: All Pydantic models (Python backend)
- `orbit-wars-lab/viewer/src/api.ts`: Matching TypeScript interfaces (frontend)

**Testing:**
- `orbit-wars-lab/tests/unit/`: 13 unit test modules
- `orbit-wars-lab/tests/integration/`: 10 integration test modules
- `orbit-wars-lab/tests/conftest.py`: Shared fixtures

## Naming Conventions

**Files:**
- Python modules: `snake_case.py` (e.g., `agent_subprocess.py`, `trueskill_store.py`)
- TypeScript modules: `kebab-case.ts` (e.g., `quick-match.ts`, `agent-picker.ts`)
- CSS: `kebab-case.css` (e.g., `quick-match.css`, `tournament-detail.css`)
- Test files: `test_<module>.py` (unit) and `test_<feature>.py` (integration)
- Replay files: `<NNN>-<agent_a>__vs__<agent_b>.json` (double underscore separators)
- Run directories: `<YYYY-MM-DDTHH-MM-SS>-<short-id>` (ISO date prefix)

**Directories:**
- Agent buckets: `baselines/`, `external/`, `mine/` (fixed; scanned by `discovery.py`)
- Agent names: `kebab-case` directory names (e.g., `nearest-sniper`, `kashiwaba-rl`)
- Views: match SPA route names kebab-case (e.g., `tournament-detail/`, `quick-match/`)

**Python:**
- Classes: `PascalCase` (`TrueSkillStore`, `Tournament`, `MatchOutcome`)
- Functions: `snake_case` (`run_match_fast`, `scan_zoo`, `save_replay`)
- Private helpers: `_leading_underscore` (`_extract_outcome`, `_zoo_root`, `_agent_safe_env`)
- Constants: `UPPER_SNAKE_CASE` (`TS_MU_0`, `TS_BETA`, `KAGGLE_API_ROOT`)

**TypeScript:**
- Interfaces/types: `PascalCase` (`AgentInfo`, `Route`, `MatchStateRunning`)
- Functions/variables: `camelCase` (`parseHash`, `renderQuickMatch`, `mountAgentPicker`)
- Files: `kebab-case.ts`

## Where to Add New Code

**New API endpoint:**
- Add route handler to `orbit-wars-lab/orbit_wars_app/api.py` (existing `router = APIRouter(prefix="/api")`)
- Add Pydantic request/response models to `orbit-wars-lab/orbit_wars_app/schemas.py`
- Add corresponding TypeScript interface + fetch function to `orbit-wars-lab/viewer/src/api.ts`

**New SPA view:**
- Create `orbit-wars-lab/viewer/src/views/<kebab-name>.ts` exporting `render<Name>(container: HTMLElement, ...params)`
- Add route type to the `Route` union in `orbit-wars-lab/viewer/src/router.ts`
- Add `parseHash` branch and `navigate` branch in `orbit-wars-lab/viewer/src/router.ts`
- Add dispatch branch in `orbit-wars-lab/viewer/src/main.ts`
- Add CSS to `orbit-wars-lab/viewer/src/styles/<kebab-name>.css` and import it in `main.ts`

**New reusable UI component:**
- Create `orbit-wars-lab/viewer/src/components/<kebab-name>.ts`
- Export a `mount<Name>(container, ...)` or `render<Name>(...)` function following existing component patterns

**New agent for competition:**
- Add directory `orbit-wars-lab/agents/mine/<name>/`
- Create `main.py` with `def agent(obs)` function
- Optionally add `agent.yaml` with `name`, `description`, `tags` fields

**New domain service (Python):**
- Create `orbit-wars-lab/orbit_wars_app/<service_name>.py`
- Import and call from `api.py` or `tournament.py`
- Add unit tests in `orbit-wars-lab/tests/unit/test_<service_name>.py`

**New backend utility / helper:**
- Add to the most relevant existing module or create a new one in `orbit-wars-lab/orbit_wars_app/`
- Mirror in `orbit-wars-lab/viewer/src/utils/` if a TS equivalent is needed

## Special Directories

**`orbit-wars-lab/web/core/`:**
- Purpose: Vendored `@kaggle-environments/core` — React-based Kaggle replay player
- Generated: `dist/` is generated at build time (`pnpm -r build`)
- Committed: Source only (dist is gitignored); must be built before `viewer` can compile

**`orbit-wars-lab/.venv/`:**
- Purpose: Python virtual environment
- Generated: Yes (by `scripts/dev.sh` or manual `python -m venv .venv`)
- Committed: No (gitignored)

**`orbit-wars-lab/node_modules/`:**
- Purpose: pnpm workspace node_modules (hoisted)
- Generated: Yes (by `pnpm install`)
- Committed: No (gitignored)

**`orbit-wars-lab/runs/`:**
- Purpose: Tournament history + TrueSkill state
- Generated: Yes (by the app at runtime)
- Committed: Partially — `runs/trueskill.json` initial seed is committed; run subdirs are gitignored

**`orbit-wars-lab/orbit_wars_lab.egg-info/`:**
- Purpose: Python editable install metadata
- Generated: Yes (by `pip install -e .`)
- Committed: No (gitignored)

---

*Structure analysis: 2026-05-06*
