# Codebase Concerns

**Analysis Date:** 2026-05-06

## Tech Debt

**Seed parameter ignored by the engine:**
- Issue: `run_match_fast` and `run_match_faithful` accept a `seed` parameter that is logged and stored in `MatchResult`, but the `kaggle-environments` engine ignores it internally. Matches are not reproducible by replaying with the same seed.
- Files: `orbit-wars-lab/orbit_wars_app/match.py:20`, `orbit-wars-lab/orbit_wars_app/match.py:44`
- Impact: Tournament results cannot be replayed deterministically for debugging; "seed for audit" is misleading to users and future developers.
- Fix approach: Either remove the seed parameter and audit trail documentation, or upstream a patch to `kaggle-environments` to honour it.

**Deprecated `source_url` and `version` fields still in schema:**
- Issue: `AgentInfo` in `schemas.py` carries `source_url` and `version` as deprecated fields retained "for backward compat." `discovery.py` logs a warning when they appear in `agent.yaml`, but they are never removed.
- Files: `orbit-wars-lab/orbit_wars_app/schemas.py:49-57`, `orbit-wars-lab/orbit_wars_app/discovery.py:15`, `orbit-wars-lab/orbit_wars_app/tournament.py:557-561`
- Impact: Deprecated fields surface in the API (`GET /api/agents`) and viewer (`api.ts` still types `source_url` and `version`). Old `agent.yaml` files that still use them silently work while generating log noise.
- Fix approach: Set a removal milestone. Add migration script to rewrite old `agent.yaml` files to `kernel_slug` + `kernel_version`. Then drop the fields from `AgentInfo` and `api.ts`.

**`_new_run_id` uses directory count, not max suffix:**
- Issue: `Tournament._new_run_id` counts directories with today's date prefix and uses `len(existing) + 1` as the sequence number. If any earlier run directory is deleted, the next run reuses a previously-used ID (e.g., if `2026-05-06-003` is deleted, the next run produces `2026-05-06-003` again, overwriting results if the directory is recreated).
- Files: `orbit-wars-lab/orbit_wars_app/tournament.py:427-436`
- Impact: Collision risk when users delete runs through the UI's `DELETE /runs/{run_id}` endpoint. Rare but silent data corruption.
- Fix approach: Parse the integer suffix from existing directory names (`max(suffix) + 1`) rather than counting directory count.

**`todo` placeholder in `external.py` fetch output:**
- Issue: `fetch_notebook()` stubs `description` with `"TODO: fill after subagent analysis"` when the existing `agent.yaml` has no description. This placeholder leaks into `GET /api/agents` and the viewer's agent cards.
- Files: `orbit-wars-lab/orbit_wars_app/external.py:428`
- Impact: UI shows "TODO: fill after subagent analysis" as the description for any freshly-fetched agent.
- Fix approach: Replace with `None` or empty string; let the UI render a dash for missing descriptions.

**`GET /runs/{run_id}` and `GET /replays/{run_id}/{match_id}` lack path-traversal guard:**
- Issue: `get_run` and `get_replay` construct their paths via `_runs_root() / run_id` without calling `_safe_subpath`. The DELETE variants for the same paths do use `_safe_subpath`. This inconsistency means a read-only traversal is not blocked (e.g., `GET /runs/../../etc/passwd` would resolve to a non-directory and return 404, but the guard is not principled).
- Files: `orbit-wars-lab/orbit_wars_app/api.py:104-119`, `orbit-wars-lab/orbit_wars_app/api.py:444-455`
- Impact: Currently low severity because the path must resolve to a directory (for `get_run`) and then have known filenames (config.json, results.json). A future endpoint change could widen the attack surface.
- Fix approach: Wrap both `get_run` and `get_replay` with `_safe_subpath` for consistency with the DELETE counterparts.

**`kaggle_environments` dependency pinned to a specific git commit:**
- Issue: `pyproject.toml` pins `kaggle-environments` to a specific upstream commit hash (`27660ecca553b96b1e3d6b282ca855fd1f274847`). There is no mechanism to check if that commit is ever rebased or removed.
- Files: `orbit-wars-lab/pyproject.toml:26`
- Impact: Build breaks if the upstream repository removes or force-pushes past that commit. The `requirements.txt` that the Docker stage reads may diverge from `pyproject.toml`.
- Fix approach: Pin to a stable tag or release instead of a commit hash. Alternatively vendor the relevant files inside the repo.

## Known Bugs

**`list_local_kaggle_replays` defines `_has_names` inside a loop:**
- Symptoms: A new closure object is created on every loop iteration. In Python this is a minor performance/style issue, not a functional bug. However, if any future change captures loop variables it will silently access stale values.
- Files: `orbit-wars-lab/orbit_wars_app/kaggle_scraper.py:340-341`
- Trigger: Called on every `GET /api/replays?source=kaggle` or `source=all` request.
- Workaround: No functional impact currently.

**Symlink creation silently falls back on Windows:**
- Symptoms: `Tournament.run()` tries to create a `latest` symlink to the newest run directory. On Windows (and some Docker bind mounts), `Path.symlink_to()` raises `OSError` or `NotImplementedError`, so it falls back to writing a `latest.txt` text file. The UI and CLI do not read `latest.txt`; they always use the most-recent `run.json`. The fallback is therefore a dead code path.
- Files: `orbit-wars-lab/orbit_wars_app/tournament.py:330-337`
- Trigger: Running on Windows (the current dev environment) or on Docker with restricted privileges.
- Workaround: The UI does not rely on `latest`; only the CLI `--runs latest` shorthand would be affected.

## Security Considerations

**Local API has no authentication:**
- Risk: `POST /api/tournaments`, `DELETE /api/runs/{id}`, `DELETE /api/agents/{id}`, `POST /api/ratings/reset`, and `POST /api/kaggle-auth` are all unauthenticated. Any process that can reach the server port can start tournaments, delete all data, or overwrite the Kaggle token.
- Files: `orbit-wars-lab/orbit_wars_app/main.py`, `orbit-wars-lab/orbit_wars_app/api.py`
- Current mitigation: The Makefile binds to `127.0.0.1` only in local dev (`make backend`). Docker's compose file publishes to `localhost:6001`, not `0.0.0.0`.
- Recommendations: Add a note in the README that the app must not be exposed on a shared network without an auth layer (e.g., a reverse-proxy with Basic Auth). For multi-user or remote setups, add a simple token check middleware.

**`scrape_url` endpoint accepts arbitrary user-supplied URLs:**
- Risk: `POST /api/replays/scrape-url` parses a URL from the request body and makes an outbound HTTP request to that URL via `kaggle_scraper.scrape_single_episode`. Although the URL is validated to contain `/episodes/<id>` or `?episodeId=<id>`, the host is not restricted — a crafted URL could reach internal network services.
- Files: `orbit-wars-lab/orbit_wars_app/api.py:204-256`, `orbit-wars-lab/orbit_wars_app/kaggle_scraper.py`
- Current mitigation: Regex requires a numeric episode ID to appear in the URL, limiting what makes it past parsing. The `requests` call still uses whatever host was provided.
- Recommendations: Validate that the parsed host is `www.kaggle.com` before making the outbound request.

**Agent subprocess does not sandbox filesystem access:**
- Risk: External agent code in `agents/external/*/main.py` runs as the full server process user. A malicious or buggy external agent can read or write any file accessible to the server, including `runs/trueskill.json`, `agents/`, and `~/.kaggle/kaggle.json` (if the `.kaggle` volume mount is active in Docker).
- Files: `orbit-wars-lab/orbit_wars_app/agent_subprocess.py:28-39`
- Current mitigation: `_agent_safe_env()` strips `KAGGLE_*` environment variables from spawned subprocesses. The `safety_audit()` regex in `external.py` flags suspicious import patterns at fetch time and sets `disabled: true`.
- Recommendations: Run agents inside a container or with filesystem namespace isolation (e.g., `unshare`, `firejail`, or a Docker-in-Docker pattern) for untrusted third-party code.

**`runs-list.ts` injects API data into innerHTML without escaping:**
- Risk: In the inline match expansion of `mountRunsList`, `m.agent_ids`, `m.match_id`, and `m.winner` are interpolated directly into an `innerHTML` string without calling `escapeHtml`. Agent IDs that contain `<`, `>`, or `"` (possible if a rogue `agent.yaml` name field contains HTML) would be rendered as markup.
- Files: `orbit-wars-lab/viewer/src/components/runs-list.ts:36`
- Current mitigation: `agent_id` values are derived from filesystem directory names and the `name` field of `agent.yaml`. Exploitation requires a maliciously-named agent directory or YAML. All other components (`embedded-replay.ts`, `agent-card.ts`, `replays.ts`) do call `escapeHtml` consistently.
- Recommendations: Apply `escapeHtml` from `utils/escape.ts` to `m.agent_ids`, `m.match_id`, and `m.winner` in `runs-list.ts:36`.

## Performance Bottlenecks

**`GET /api/replays` scans all run directories on every request:**
- Problem: `list_replays` calls `sorted(runs.iterdir(), reverse=True)` and for each run directory reads `results.json`, then for each match globs the `replays/` subdirectory to get the file mtime. With many tournaments and many matches per tournament, this becomes O(runs × matches) filesystem reads on every request.
- Files: `orbit-wars-lab/orbit_wars_app/api.py:140-191`
- Cause: No in-memory cache or index; every poll from the UI (Quick Match polls every 500ms during a running tournament) triggers a full scan.
- Improvement path: Build a lightweight index file (`runs/replay-index.json`) updated on each tournament completion. Fall back to full scan only when the index is missing.

**`list_local_kaggle_replays` may fully parse 2MB replay files as last resort:**
- Problem: When a replay file has no per-episode `.meta.json` or has a stale (pre-schema-2) meta, `list_local_kaggle_replays` reads and JSON-parses the full 2MB replay file to extract agent names. This "last resort" path runs on every `GET /api/replays` call until the meta file is written.
- Files: `orbit-wars-lab/orbit_wars_app/kaggle_scraper.py:337-358`
- Cause: Schema-version guard bypasses the cache for older scraped replays.
- Improvement path: A one-time migration command that regenerates all pre-schema-2 meta files would eliminate the fallback entirely for existing replays.

**`GET /api/agents` re-scans the zoo on every request:**
- Problem: `scan_zoo` walks the `agents/` directory tree and reads every `agent.yaml` on every `GET /api/agents` call. With ~10 agents this is fast. With 50+ agents (heavy users of the external fetch workflow), it adds noticeable latency.
- Files: `orbit-wars-lab/orbit_wars_app/api.py:62-63`, `orbit-wars-lab/orbit_wars_app/discovery.py:20-45`
- Cause: No caching between requests.
- Improvement path: Cache `scan_zoo` result with filesystem mtime-based invalidation on `agents/` directory.

**Parallel tournament uses `ProcessPoolExecutor` with up to 16 workers, each importing kaggle-environments (~150MB):**
- Problem: With `parallel=16`, up to 16 worker processes each import the `kaggle-environments` package. On machines with limited RAM this causes OOM; the `BaseException` catch in the parallel path converts OOM workers to "crashed" matches silently.
- Files: `orbit-wars-lab/orbit_wars_app/tournament.py:243-290`, `orbit-wars-lab/orbit_wars_app/schemas.py:102`
- Cause: `parallel` is capped at 16 in the schema but the in-memory cost scales linearly.
- Improvement path: Document RAM requirements per worker (≈150MB × parallel). Reduce default cap or add a warning when parallel > available_memory / 200MB.

## Fragile Areas

**`agent_serve.py` `load_agent` heuristic — returns last callable in namespace:**
- Files: `orbit-wars-lab/orbit_wars_app/agent_serve.py:23-50`
- Why fragile: The agent function is detected by taking the last callable from `vars(mod)`. If an agent defines utility functions or imports callable objects after the main agent function, the wrong callable is used. This matches `kaggle-environments` agent.py's own behavior, but diverges from Kaggle production (which uses a different resolution path).
- Safe modification: Only modify if agent loading breaks for a specific agent. Adding a `__agent__` convention (exported sentinel) would be safer long-term.
- Test coverage: Covered by `tests/unit/test_agent_serve_loader.py` but only with simple fixtures.

**`_safe_subpath` does not guard `GET /runs/{run_id}` and `GET /replays/{run_id}/{match_id}`:**
- Files: `orbit-wars-lab/orbit_wars_app/api.py:104-119`, `orbit-wars-lab/orbit_wars_app/api.py:444-455`
- Why fragile: The path-safety guard is applied to DELETE but not GET on the same resource. Any refactor that makes GET endpoints more permissive (e.g., serving arbitrary files from within a run directory) would open path traversal without the guard in place.
- Safe modification: Add `_safe_subpath` calls to `get_run` and `get_replay` matching the pattern in `delete_run` and `delete_local_replay`.

**`start_tournament` endpoint relies on polling `runs_root` to detect run creation:**
- Files: `orbit-wars-lab/orbit_wars_app/api.py:622-684`
- Why fragile: After submitting the tournament to the thread pool executor, the async handler polls `runs_root.iterdir()` every 50ms for up to 5 seconds waiting for `run.json` to appear with `status == "running"`. If the filesystem is slow or the tournament crashes before writing `run.json`, the endpoint returns 500 and releases the `_tournament_lock`, but the background thread may still be running in the executor — the lock state and the actual execution state diverge.
- Safe modification: Replace the polling approach with an asyncio `Event` or `Queue` signalled by the background thread when `run.json` is written.

**`_submissions_cache` is not thread-safe for concurrent writes:**
- Files: `orbit-wars-lab/orbit_wars_app/kaggle_submissions.py:24-103`
- Why fragile: `_submissions_cache` is a module-level `dict` mutated by `list_my_submissions` and `submit_agent` without a lock. In CPython with the GIL this is practically safe for dict read/write, but concurrent `POST /api/kaggle-submissions` + `GET /api/kaggle-submissions` can produce a stale-after-invalidation read window.
- Safe modification: Wrap cache access with a `threading.Lock` consistent with `_jobs_lock` in `kaggle_scraper.py`.

## Scaling Limits

**Replay storage grows unbounded:**
- Current capacity: Each tournament match replay is 5–10 MB of JSON. The `runs/` directory accumulates all replays indefinitely. The `replays/kaggle/` directory also grows indefinitely as Kaggle episodes are scraped.
- Limit: Local disk space; no automatic pruning exists.
- Scaling path: Add a `max_replays` configuration option and a background cleanup job that removes the oldest replays beyond the configured limit. The UI's per-replay `DELETE` is the only current mechanism.

**`_jobs` dict in `kaggle_scraper` never evicts completed jobs:**
- Current capacity: The module-level `_jobs` dict accumulates every `ScrapeJob` ever started for the lifetime of the process. Each job holds a `list[int]` of downloaded episode IDs.
- Limit: After thousands of scrape jobs the dict grows unboundedly; no TTL or eviction.
- Scaling path: Add TTL-based eviction (e.g., remove completed/failed jobs older than 1 hour) triggered on `get_job` or `start_scrape`.

## Dependencies at Risk

**`kaggle~=2.0.2` pinned to a narrow minor range:**
- Risk: Kaggle frequently releases SDK updates that change authentication paths (KGAT_ token handling in particular). The `~=2.0.2` pin blocks Kaggle from pushing breaking changes but also blocks security fixes.
- Impact: If Kaggle deprecates the `/api/v1` endpoint used for token validation or changes the submission API, the auth and submission features will break silently.
- Migration plan: Monitor Kaggle SDK changelogs; test against the next minor version in a branch before widening the pin.

**`trueskill>=0.4.5` — unmaintained package:**
- Risk: The `trueskill` PyPI package has had no new release since 2016. It works with Python 3.12 but is not actively maintained.
- Impact: A future Python version may break internal assumptions in the library without any upstream fix.
- Migration plan: Vendor the relevant `trueskill` source (~300 lines) inside `orbit_wars_app/` or switch to `openskill` which is actively maintained.

## Missing Critical Features

**No frontend test suite:**
- Problem: The viewer (`viewer/src/`) has zero test files. `viewer/package.json` has no `test` script, no `vitest` or `jest` dependency. UI logic (routing, polling state machine in `quick-match.ts`, HTML generation in all views) is completely untested.
- Blocks: Confident refactoring of the frontend; regression detection on viewer changes.

**No rate-limiting on Kaggle scraper outbound requests:**
- Problem: `scrape_submission` downloads episodes one at a time with no delay or backoff between `fetch_replay` calls. Kaggle's internal `EpisodeService` endpoint (`api/i/competitions.EpisodeService/GetEpisodeReplay`) is undocumented and unauthenticated; aggressive scraping could result in IP-level rate limiting or blocking.
- Blocks: Reliable bulk scraping for large submission counts.

## Test Coverage Gaps

**Faithful mode match runner (`run_match_faithful`) has limited integration coverage:**
- What's not tested: Full subprocess lifecycle — particularly the agent-stderr drain path in `shutdown()`, and the case where `_wait_for_port` times out after `ready` is emitted. The `test_match_faithful.py` integration test exercises the happy path only.
- Files: `orbit-wars-lab/orbit_wars_app/agent_subprocess.py`, `orbit-wars-lab/orbit_wars_app/match.py:162-233`, `orbit-wars-lab/tests/integration/test_match_faithful.py`
- Risk: Subprocess lifecycle bugs (zombie processes, port-binding races) go undetected until production use.
- Priority: Medium

**Viewer frontend has no tests at all:**
- What's not tested: All TypeScript in `viewer/src/` — routing, state machines, API client, renderer, component templates.
- Files: `orbit-wars-lab/viewer/src/` (all files)
- Risk: HTML-injection bugs (see `runs-list.ts`), broken polling logic, or broken replay rendering ship without detection.
- Priority: High

**`list_local_kaggle_replays` fallback path (parse full replay) not directly tested:**
- What's not tested: The branch in `list_local_kaggle_replays` that opens and parses the full 2MB replay JSON when meta is missing or pre-schema-2. Tests mock the filesystem with small JSON fixtures.
- Files: `orbit-wars-lab/orbit_wars_app/kaggle_scraper.py:337-358`
- Risk: Silent metadata extraction failures (wrong agent names, incorrect winner) on large real replays.
- Priority: Low

---

*Concerns audit: 2026-05-06*
