# Technology Stack

**Analysis Date:** 2026-05-06

## Languages

**Primary:**
- Python 3.12 - Agent logic (`main.py`, all `orbit_wars_app/` backend), tournament runner, test suite
- TypeScript 5.x - Frontend viewer (`orbit-wars-lab/viewer/src/`) and shared core library (`orbit-wars-lab/web/core/src/`)

**Secondary:**
- JavaScript - Vite config files, minor glue scripts

## Runtime

**Environment:**
- Python >=3.12 (required by `pyproject.toml` `requires-python = ">=3.12"`)
- Node.js >=20 (required by `orbit-wars-lab/package.json` `engines.node`)

**Package Manager:**
- Python: `pip` / `setuptools>=68` (build backend in `pyproject.toml`)
- Node: `pnpm@9.15.3` (pinned in `orbit-wars-lab/package.json` `packageManager` field)
- Lockfile: `orbit-wars-lab/pnpm-lock.yaml` (present and committed)

## Frameworks

**Backend API:**
- FastAPI >=0.110 - REST API server (`orbit_wars_app/main.py`, `orbit_wars_app/api.py`)
- Uvicorn >=0.28 (standard extras) - ASGI server; Docker CMD is `uvicorn orbit_wars_app.main:app`
- Pydantic >=2.0 - Request/response schemas (`orbit_wars_app/schemas.py`)

**Frontend:**
- React ^18 or ^19 (peer dep) - UI framework for `@kaggle-environments/core` library
- Vite ^5.0 - Build tool and dev server for both `viewer/` and `web/core/`
- MUI (Material UI) ^5 or ^6 - Component library (`orbit-wars-lab/web/core/package.json`)
- Emotion (`@emotion/react`, `@emotion/styled`) ^11 - CSS-in-JS for MUI

**Game Environment:**
- `kaggle-environments` (VCS pin: `git+https://github.com/Kaggle/kaggle-environments.git@27660ecca553b96b1e3d6b282ca855fd1f274847`) - Provides `orbit_wars` game engine, `make()`, `env.run()`, `Planet`/`Fleet` named tuples
- This is pinned to a specific commit on branch `orbit-wars-rotational-symmetry` (not yet on PyPI)

**Testing:**
- Python: `pytest >=8.0` + `pytest-asyncio >=0.23` (async test support, `asyncio_mode = "auto"`)
- TypeScript: `vitest ^2.1.9` + `@testing-library/react ^16.3.2` + `jsdom ~25.0.1`

**Reinforcement Learning (optional):**
- PyTorch >=2.0 (CPU-only wheel) - Optional `[rl]` extra for the `kashiwaba-rl` PPO agent

## Key Dependencies

**Critical:**
- `kaggle~=2.0.2` - Kaggle Python SDK (pinned compatible-release); used for submission API, auth, log downloads in `orbit_wars_app/kaggle_submissions.py` and `orbit_wars_app/kaggle_auth.py`
- `kaggle-environments` (VCS pin) - Core game engine; the entire project depends on this for running matches
- `trueskill >=0.4.5` - TrueSkill rating system for the tournament leaderboard (`orbit_wars_app/trueskill_store.py`)
- `numpy >=1.26.0` - Numerical computation for agent strategies
- `httpx >=0.27` - Async HTTP client (used internally by FastAPI test client and scraper utilities)

**Infrastructure:**
- `requests` (transitive via `kaggle`) - Used directly in `orbit_wars_app/kaggle_auth.py` and `orbit_wars_app/kaggle_scraper.py` for HTTP calls to Kaggle API
- `pyyaml >=6.0` - Agent metadata `agent.yaml` parsing in `orbit_wars_app/external.py` and `orbit_wars_app/discovery.py`
- `psutil >=5.9` - Process monitoring for tournament match subprocesses
- `react-virtuoso ^4.12.3` - Virtualized list rendering in the viewer
- `react-markdown ^9.0.0` - Markdown rendering in the viewer UI
- `@kaggle-environments/core` (workspace:* local) - Shared visualizer/player library built from `orbit-wars-lab/web/core/`

**Build tooling:**
- `vite-plugin-dts ^4.5.4` - TypeScript declaration file generation for `@kaggle-environments/core`
- `@vitejs/plugin-react ^4.2.0` - React fast-refresh and JSX transform
- `cross-env ^10.1.0` - Cross-platform env var injection for dev scripts
- `ruff` (dev tool, configured in `pyproject.toml`) - Python linter/formatter (lint rules: E, F, W, I, UP; line length 100)

## Configuration

**Environment:**
- Python backend reads these env vars:
  - `KAGGLE_CONFIG_DIR` - Override for Kaggle token directory (default `~/.kaggle/`)
  - `KAGGLE_USERNAME` / `KAGGLE_KEY` - Direct SDK auth env vars (take priority over config file)
  - `KAGGLE_API_TOKEN` - Bearer token for new-style `KGAT_` access tokens
  - `ORBIT_WARS_ZOO_DIR` - Agent zoo directory (default `agents/`)
  - `ORBIT_WARS_RUNS_DIR` - Tournament runs directory (default `runs/`)
  - `ORBIT_WARS_REPLAYS_DIR` - Kaggle replay storage (default `replays/`)
  - `HOME` - Used in Docker to point to non-root user home
  - `KAGGLE_CLI` - Override path to the `kaggle` CLI binary (default `"kaggle"`)
- TypeScript/Vite env vars:
  - `VITE_REPLAY_FILE` - Dev mode: load a specific replay JSON for testing
  - `VITE_PORT` - Override dev server port (default 5173)
  - `VITE_CUSTOM_HEADER_NAME` / `VITE_CUSTOM_HEADER_PATH` - Dev server header display

**Build:**
- `orbit-wars-lab/pyproject.toml` - Python package config, ruff lint config, pytest config
- `orbit-wars-lab/web/tsconfig.base.json` - Base TypeScript compiler options (target ESNext, strict mode, `noUnusedLocals`, `noUnusedParameters`, `noImplicitReturns`)
- `orbit-wars-lab/web/core/vite.config.ts` - Library build config for `@kaggle-environments/core`
- `orbit-wars-lab/viewer/vite.config.ts` - App build config for the viewer SPA (proxies `/api` to `:8000` in dev)
- `orbit-wars-lab/pnpm-workspace.yaml` - Workspace packages: `web/*` and `viewer`

## Platform Requirements

**Development:**
- Python 3.12+
- Node.js 20+
- pnpm 9.15.3 (via corepack)
- Git (required at Docker runtime to resolve VCS-pinned `kaggle-environments` dependency)
- Optional: `torch>=2.0` (CPU wheel) for the kashiwaba PPO RL agent

**Production:**
- Docker multi-stage build (`orbit-wars-lab/Dockerfile`):
  - Stage 1: `node:20-alpine` - Builds Vite bundle
  - Stage 2: `python:3.12-slim` + `build-essential` + `git` - Builds Python wheels
  - Stage 3: `python:3.12-slim` + `git` only - Final runtime image
- Exposes port 8000; Docker Compose maps to host port 6001 by default
- Runs as non-root user `app` (UID 1000)
- Volumes: `./agents` and `./runs` mounted for persistence; `.kaggle/` optional

---

*Stack analysis: 2026-05-06*
