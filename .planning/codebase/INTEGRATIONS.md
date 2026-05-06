# External Integrations

**Analysis Date:** 2026-05-06

## APIs & External Services

**Kaggle Competition Platform:**
- Kaggle REST API v1 - Competition submissions, leaderboard, episode logs
  - SDK/Client: `kaggle~=2.0.2` (`kaggle.api.kaggle_api_extended.KaggleApi`)
  - Direct HTTP: `requests` library (bypasses SDK for auth validation)
  - Auth: `~/.kaggle/kaggle.json` (fields: `username`, `key`) or env vars `KAGGLE_USERNAME`/`KAGGLE_KEY`/`KAGGLE_API_TOKEN`
  - Base URL: `https://www.kaggle.com/api/v1`
  - Used in: `orbit-wars-lab/orbit_wars_app/kaggle_auth.py`, `orbit-wars-lab/orbit_wars_app/kaggle_submissions.py`

- Kaggle Internal Episode Service API - Scraping episode replays and metadata (no auth required)
  - SDK/Client: `requests` (anonymous HTTP POST)
  - Base URL: `https://www.kaggle.com/api/i`
  - Endpoints used:
    - `POST /competitions.EpisodeService/ListEpisodes` - list episodes for a submission
    - `POST /competitions.EpisodeService/GetEpisodeReplay` - fetch full replay JSON
  - Used in: `orbit-wars-lab/orbit_wars_app/kaggle_scraper.py`

- Kaggle CLI (`kaggle` binary) - Kernel/notebook management and agent submission
  - Called via `subprocess.run` in `orbit-wars-lab/orbit_wars_app/external.py`
  - Commands: `kaggle kernels status`, `kaggle kernels pull`, `kaggle competitions submit`, `kaggle competitions logs`, `kaggle competitions submissions`
  - Binary path override: `KAGGLE_CLI` env var

- Kaggle SDK (kagglesdk) - Optional; used for KGAT_ access token introspection only
  - Package: `kagglesdk` (imported lazily in `orbit_wars_app/kaggle_auth.py`)
  - Used only when user pastes a bare `KGAT_` bearer token without a username

**PyTorch Download Index (optional):**
- CPU-only torch wheels from `https://download.pytorch.org/whl/cpu`
  - Required only when using the `kashiwaba-rl` PPO reinforcement learning agent
  - Configured via `--extra-index-url` in `orbit-wars-lab/requirements.txt`

## Data Storage

**Databases:**
- None - No external database server

**File-based persistence (local filesystem):**
- `runs/` - Tournament run directories; each contains `run.json`, `results.json`, `config.json`, `trueskill.json`, `replays/`
- `runs/trueskill.json` - Global TrueSkill ratings JSON file (schema: `{schema_version, last_updated, ratings: {agent_id: {"2p": {...}, "4p": {...}}}}`)
- `replays/kaggle/<submission_id>/` - Scraped Kaggle episode replays and metadata
- `agents/` - Agent zoo: `baselines/`, `external/`, `mine/` buckets; each agent has `main.py` + optional `agent.yaml`

**File Storage:**
- Local filesystem only (no cloud object storage)

**Caching:**
- In-memory Python dict: `orbit_wars_app/kaggle_submissions._submissions_cache` (TTL 60 seconds; keyed by call parameters)
- No Redis or distributed cache

## Authentication & Identity

**Auth Provider:**
- Kaggle (external) - All auth flows are against Kaggle's API
  - Two token types supported:
    1. Legacy API key: `{"username": "...", "key": "<32-hex>"}` in `~/.kaggle/kaggle.json` — validated via HTTP Basic auth
    2. New access token: bare `KGAT_<...>` string — validated via HTTP Bearer auth; username resolved via kagglesdk introspection endpoint
  - Token persistence: `~/.kaggle/kaggle.json` written with mode `0o600`
  - No application-level user accounts; the app is a single-user local tool

**Settings UI Auth Flow:**
- User pastes token into browser Settings tab → POST `/api/kaggle-auth` → `orbit_wars_app/kaggle_auth.save_token()` validates against Kaggle API and persists to disk
- Env vars `KAGGLE_USERNAME`/`KAGGLE_KEY` always shadow the config file (reported as `source: "env"`)

## Monitoring & Observability

**Error Tracking:**
- None (no Sentry or similar)

**Logs:**
- Python standard `logging` module (`logger = logging.getLogger(__name__)` per module)
- `orbit_wars_app/kaggle_scraper.py` uses module-level logger
- Kaggle-environments import-time logging suppressed in tournament worker processes (bumped above INFO level)
- No structured log shipping; logs go to stdout/stderr via uvicorn

**Health Check:**
- `GET /api/health` endpoint → `{"status": "ok", "version": "..."}` used as Docker `HEALTHCHECK`

## CI/CD & Deployment

**Hosting:**
- Local Docker container (primary deployment target)
- Docker Compose: `orbit-wars-lab/docker-compose.yml` — single `app` service
- Image tag: `orbit-wars-lab:latest`
- Port mapping: `${PORT:-6001}:8000`
- Restart policy: `unless-stopped`

**CI Pipeline:**
- Not detected (no `.github/workflows/`, no CI config files found)

**Build pipeline:**
1. `pnpm -r build` in Node stage (builds `@kaggle-environments/core` then `viewer/`)
2. `pip wheel` in Python builder stage (pre-compiles all wheels)
3. Final image copies wheels and installs with `--no-index`

## Environment Configuration

**Required env vars (for full functionality):**
- `KAGGLE_USERNAME` + `KAGGLE_KEY` OR `KAGGLE_API_TOKEN` - Kaggle API auth (or configure via Settings UI → `~/.kaggle/kaggle.json`)

**Optional env vars:**
- `ORBIT_WARS_ZOO_DIR` - Override agents directory (default `agents/`)
- `ORBIT_WARS_RUNS_DIR` - Override runs directory (default `runs/`)
- `ORBIT_WARS_REPLAYS_DIR` - Override replays directory (default `replays/`)
- `KAGGLE_CONFIG_DIR` - Override Kaggle config directory (default `~/.kaggle/`)
- `KAGGLE_CLI` - Override path to the `kaggle` CLI binary
- `HOME` - Set in Docker to non-root home `/home/app`
- `UID` / `GID` - Docker Compose user mapping (default 1000)
- `PORT` - Docker Compose host port (default 6001)
- `VITE_REPLAY_FILE` - Dev only: load a specific replay JSON for viewer testing
- `VITE_PORT` - Dev only: override viewer dev server port (default 5173)

**Secrets location:**
- `~/.kaggle/kaggle.json` on the host (or container home) — written with `chmod 0o600`
- Docker Compose volume `~/.kaggle:/home/app/.kaggle` (optional, commented out by default)
- Never committed to source control

## Webhooks & Callbacks

**Incoming:**
- None

**Outgoing:**
- None (all external calls are request-response, not webhook-based)

## External Package Sources

**Python (non-PyPI):**
- `kaggle-environments` pinned to GitHub VCS ref:
  `git+https://github.com/Kaggle/kaggle-environments.git@27660ecca553b96b1e3d6b282ca855fd1f274847`
  - Branch: `orbit-wars-rotational-symmetry` (4-fold rotational symmetry fix)
  - Requires `git` at install time and Docker runtime

**Node.js (workspace-local):**
- `@kaggle-environments/core` - Resolved as `workspace:*` from `orbit-wars-lab/web/core/`; not fetched from npm

---

*Integration audit: 2026-05-06*
