"""
Parse the most recent tournament results, output METRIC lines, commit results.json.

Run from project root: python experiments/parse_results.py
"""
import json
import re
import subprocess
import sys
from pathlib import Path

TOURNAMENT_DIR = Path(__file__).parent.parent / "76-RL_tournament"
PROJECT_ROOT = Path(__file__).parent.parent


def fitness(summary, pid):
    s2 = summary["2p"][pid]
    s4 = summary["4p"][pid]
    return s2["wins"] * 3 + s4["wins"] * 2 + s4["top2"]


def main():
    completed = []
    for d in TOURNAMENT_DIR.iterdir():
        if not d.is_dir():
            continue
        m = re.match(r'^(\d{3})', d.name)
        if not m:
            continue
        rpath = d / "results.json"
        if rpath.exists():
            completed.append((int(m.group(1)), d, rpath))

    if not completed:
        print("METRIC best_fitness=0")
        print("METRIC best_2p_wins=0")
        return

    completed.sort(key=lambda x: x[0])
    num, folder, rpath = completed[-1]

    with open(rpath, encoding="utf-8") as f:
        data = json.load(f)

    summary = data["summary"]
    agents = data["agents"]
    pids = sorted(agents.keys())

    fits = {pid: fitness(summary, pid) for pid in pids}
    best_pid = max(fits, key=fits.get)
    best_fitness = fits[best_pid]
    best_2p_wins = summary["2p"][best_pid]["wins"]
    best_name = agents[best_pid].get("_name", best_pid)

    # Print full standings
    ranked = sorted(pids, key=lambda p: fits[p], reverse=True)
    print(f"Gen {num:03d} standings:")
    for pid in ranked:
        name = agents[pid].get("_name", pid)
        s2 = summary["2p"][pid]
        s4 = summary["4p"][pid]
        print(f"  {pid} {name}: fitness={fits[pid]}  "
              f"2p={s2['wins']}W/{s2['losses']}L  "
              f"4p={s4['wins']}W top2={s4['top2']}")
    print(f"Best: {best_pid} ({best_name})  fitness={best_fitness}  2p_wins={best_2p_wins}")

    print(f"METRIC best_fitness={best_fitness}")
    print(f"METRIC best_2p_wins={best_2p_wins}")

    # Commit results.json if not already committed
    subprocess.run(["git", "add", str(rpath)], cwd=PROJECT_ROOT, check=True)
    diff = subprocess.run(
        ["git", "diff", "--cached", "--quiet"], cwd=PROJECT_ROOT
    )
    if diff.returncode != 0:
        subprocess.run(
            ["git", "commit", "-m",
             f"data: gen {num:03d} results  best_fitness={best_fitness}  "
             f"2p_wins={best_2p_wins}"],
            cwd=PROJECT_ROOT, check=True,
        )
        print(f"Committed gen {num:03d} results.json")


if __name__ == "__main__":
    main()
