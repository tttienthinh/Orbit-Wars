"""Download winning episode replays for a Kaggle submission.

Usage:
    python 110-kaggle_scraper.py [--submission-id ID] [--count N]

Defaults to submission 53402535. Saves to 110-replays/<submission_id>/.

Uses Kaggle API v1 with Bearer auth from ~/.kaggle/kaggle.json.
Winner is pre-filtered from episode list metadata (reward field) — no
replay is downloaded for episodes the submission lost.
"""
import argparse
import json
import logging
import sys
import time
from pathlib import Path

import requests
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

KAGGLE_BASE = "https://www.kaggle.com/api/v1"
LIST_EPISODES_URL = KAGGLE_BASE + "/competitions/submissions/{submission_id}/episodes"
GET_REPLAY_URL = KAGGLE_BASE + "/competitions/episodes/{episode_id}/replay"

OUT_ROOT = Path(__file__).parent / "110-replays"


def _session() -> requests.Session:
    cfg_path = Path.home() / ".kaggle" / "kaggle.json"
    cfg = json.loads(cfg_path.read_text())
    token = cfg.get("token") or cfg.get("key", "")
    s = requests.Session()
    s.headers.update({
        "Authorization": f"Bearer {token}",
        "User-Agent": "orbit-wars-scraper/1.0",
    })
    return s


def list_episodes(session: requests.Session, submission_id: int) -> list[dict]:
    url = LIST_EPISODES_URL.format(submission_id=submission_id)
    r = session.get(url, timeout=30)
    r.raise_for_status()
    data = r.json()
    # Response may be a list directly or wrapped in a field
    if isinstance(data, list):
        return data
    return data.get("episodes", data.get("items", []))


def fetch_replay(session: requests.Session, episode_id: int) -> dict:
    url = GET_REPLAY_URL.format(episode_id=episode_id)
    r = session.get(url, timeout=60)
    r.raise_for_status()
    return r.json()


def _agent_idx(ep: dict, submission_id: int) -> int | None:
    for i, agent in enumerate(ep.get("agents", []) or []):
        if isinstance(agent, dict) and agent.get("submissionId") == submission_id:
            return i
    return None


def _won_from_meta(ep: dict, agent_idx: int) -> bool:
    """Check winner using reward already present in episode list metadata."""
    agents = ep.get("agents", []) or []
    rewards = [a.get("reward") for a in agents if isinstance(a, dict)]
    clean = [r for r in rewards if isinstance(r, (int, float))]
    if not clean:
        return False
    max_r = max(clean)
    if max_r <= 0:
        return False
    top = [i for i, a in enumerate(agents) if isinstance(a, dict) and a.get("reward") == max_r]
    return len(top) == 1 and top[0] == agent_idx


def _already_done(out_dir: Path, ep_id: int) -> bool:
    return (out_dir / f"episode_{ep_id}.json").exists() or \
           (out_dir / f"episode_{ep_id}.loss").exists()


def scrape(submission_id: int, count: int) -> None:
    out_dir = OUT_ROOT / str(submission_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    session = _session()

    log.info("Listing episodes for submission %d …", submission_id)
    episodes = list_episodes(session, submission_id)
    log.info("Found %d episodes total", len(episodes))

    (out_dir / "_metadata.json").write_text(json.dumps(episodes, indent=2), encoding="utf-8")

    # Pre-filter: only episodes where this submission participated AND won
    winning = []
    for ep in episodes:
        idx = _agent_idx(ep, submission_id)
        if idx is None:
            continue
        if _won_from_meta(ep, idx):
            winning.append(ep)

    log.info("%d winning episodes found in metadata", len(winning))

    missing = [ep for ep in winning if not _already_done(out_dir, int(ep["id"]))]
    to_download = missing[:count] if count > 0 else missing
    log.info("Downloading %d replays (skipping %d already on disk)", len(to_download), len(winning) - len(missing))

    ok = errors = 0
    bar = tqdm(to_download, unit="ep", desc="Downloading", dynamic_ncols=True)
    for ep in bar:
        ep_id = int(ep["id"])
        path = out_dir / f"episode_{ep_id}.json"
        if path.exists():
            ok += 1
            bar.set_postfix(saved=ok, errors=errors)
            continue
        try:
            payload = fetch_replay(session, ep_id)
            path.write_text(json.dumps(payload), encoding="utf-8")
            ok += 1
        except Exception as e:
            errors += 1
            log.warning("episode %d failed: %s", ep_id, e)
        bar.set_postfix(saved=ok, errors=errors)
        time.sleep(1)

    log.info("Done. %d saved, %d errors → %s", ok, errors, out_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download winning Kaggle episode replays")
    parser.add_argument("--submission-id", type=int, default=53402535)
    parser.add_argument("--count", type=int, default=0, help="Max replays to download (0 = all)")
    args = parser.parse_args()
    try:
        scrape(args.submission_id, args.count)
    except Exception as e:
        log.error("Fatal: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
