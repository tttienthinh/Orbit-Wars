"""
Genetic algorithm: reads all previous tournament results, ranks agents by fitness,
creates the next-generation folder with 4 evolved configs.

Run from project root: python experiments/evolve.py
"""
import json
import math
import random
import re
import subprocess
import sys
from pathlib import Path

TOURNAMENT_DIR = Path(__file__).parent.parent / "76-RL_tournament"
PROJECT_ROOT = Path(__file__).parent.parent

CONSTANTS = [
    "PROD_MULT", "TIME_PROD_MULT", "ENEMY_MULT", "COMPOUND_MULT",
    "MINE_NEAR_TGT_MULT", "ENEMY_NEAR_TGT_MULT", "PROD_SRC_MULT",
    "ORBIT_BONUS", "PROXIMITY_MULT", "DIST_MULT", "SHIPS_MULT",
    "ETA_MULT", "OVEREXTEND_MULT", "PROXIMITY_DIST",
]

BOUNDS = {
    "PROD_MULT":           (0.5,  40.0),
    "TIME_PROD_MULT":      (0.1,   6.0),
    "ENEMY_MULT":          (0.5,  30.0),
    "COMPOUND_MULT":       (0.5,  25.0),
    "MINE_NEAR_TGT_MULT":  (0.5,  25.0),
    "ENEMY_NEAR_TGT_MULT": (0.5,  20.0),
    "PROD_SRC_MULT":       (0.5,  12.0),
    "ORBIT_BONUS":         (0.5,  35.0),
    "PROXIMITY_MULT":      (0.5,  25.0),
    "DIST_MULT":           (0.01,  5.0),
    "SHIPS_MULT":          (0.01,  3.0),
    "ETA_MULT":            (0.01,  5.0),
    "OVEREXTEND_MULT":     (0.01,  4.0),
    "PROXIMITY_DIST":      (5.0,  50.0),
}


def fitness(summary, pid):
    s2 = summary["2p"][pid]
    s4 = summary["4p"][pid]
    return s2["wins"] * 3 + s4["wins"] * 2 + s4["top2"]


def clamp(val, key):
    lo, hi = BOUNDS[key]
    return max(lo, min(hi, val))


def mutate(cfg, sigma=0.12):
    """Apply lognormal mutation to each constant independently."""
    result = {}
    for key in CONSTANTS:
        val = float(cfg.get(key, 1.0))
        val *= math.exp(random.gauss(0, sigma))
        result[key] = round(clamp(val, key), 4)
    return result


def crossover(a, b, sigma=0.18):
    """Uniform crossover (50/50 per gene) + lognormal mutation."""
    result = {}
    for key in CONSTANTS:
        parent = a if random.random() < 0.5 else b
        val = float(parent.get(key, 1.0))
        val *= math.exp(random.gauss(0, sigma))
        result[key] = round(clamp(val, key), 4)
    return result


def get_completed_tournaments():
    """Return [(numeric_prefix, folder_path, results_dict), ...] sorted by prefix."""
    completed = []
    for d in TOURNAMENT_DIR.iterdir():
        if not d.is_dir():
            continue
        m = re.match(r'^(\d{3})', d.name)
        if not m:
            continue
        rpath = d / "results.json"
        if not rpath.exists():
            continue
        with open(rpath, encoding="utf-8") as f:
            data = json.load(f)
        completed.append((int(m.group(1)), d, data))
    return sorted(completed, key=lambda x: x[0])


def extract_constants(agent_dict):
    """Pull only the 14 scoring constants from an agent record."""
    return {k: float(v) for k, v in agent_dict.items()
            if k in CONSTANTS}


def find_hall_of_fame(completed):
    """
    Scan all completed tournaments and return (best_fitness, best_cfg).
    Ties broken by most recent tournament.
    """
    best_fitness = -1
    best_cfg = None
    for _, _, data in completed:
        for pid in data["agents"]:
            fit = fitness(data["summary"], pid)
            if fit > best_fitness:
                best_fitness = fit
                best_cfg = extract_constants(data["agents"][pid])
    return best_fitness, best_cfg


def highest_folder_number():
    """Return the highest 3-digit numeric prefix seen, regardless of results.json."""
    highest = -1
    for d in TOURNAMENT_DIR.iterdir():
        if not d.is_dir():
            continue
        m = re.match(r'^(\d{3})$', d.name)
        if m:
            highest = max(highest, int(m.group(1)))
    return highest


def main():
    completed = get_completed_tournaments()
    if not completed:
        print("ERROR: no completed tournaments found in 76-RL_tournament/")
        sys.exit(1)

    # Determine next folder number
    last_completed_num = completed[-1][0]
    next_num = last_completed_num + 1
    next_folder = TOURNAMENT_DIR / f"{next_num:03d}"

    if next_folder.exists():
        print(f"Folder {next_folder.name} already exists (tournament in progress). Nothing to do.")
        return

    # Latest tournament for ranking parents
    latest_num, latest_folder, latest_data = completed[-1]
    agents = latest_data["agents"]
    summary = latest_data["summary"]
    pids = sorted(agents.keys())

    # Rank by fitness
    ranked = sorted(pids, key=lambda p: fitness(summary, p), reverse=True)
    print(f"Latest: {latest_folder.name}  root_seed={latest_data['root_seed']}")
    for i, pid in enumerate(ranked):
        name = agents[pid].get("_name", pid)
        fit = fitness(summary, pid)
        s2 = summary["2p"][pid]
        s4 = summary["4p"][pid]
        print(f"  {i+1}. {pid} {name}: fitness={fit}  "
              f"2p={s2['wins']}W/{s2['losses']}L  "
              f"4p={s4['wins']}W top2={s4['top2']}")

    best_cfg    = extract_constants(agents[ranked[0]])
    second_cfg  = extract_constants(agents[ranked[1]])
    third_cfg   = extract_constants(agents[ranked[2]])

    # Hall of fame check with stagnation detection
    hof_fitness, hof_cfg = find_hall_of_fame(completed)
    current_best_fitness = fitness(summary, ranked[0])

    # Count consecutive recent gens where current-gen best did NOT beat HoF
    # (i.e. HoF kept being used as Elite but the population didn't improve past it)
    stagnation_count = 0
    for _, _, past_data in reversed(completed[1:]):  # skip gen 000-test seed
        past_pids = sorted(past_data["agents"].keys())
        past_best = max(fitness(past_data["summary"], p) for p in past_pids)
        if past_best < hof_fitness:
            stagnation_count += 1
        else:
            break
    HOF_STAGNATION_LIMIT = 3
    HOF_DEEP_STAGNATION = 6  # after this many gens, use HoF in crossovers directly

    if hof_fitness > current_best_fitness and stagnation_count < HOF_STAGNATION_LIMIT:
        print(f"\nHall-of-fame (fitness={hof_fitness}) beats current best ({current_best_fitness}) — using as Elite")
        elite_cfg = hof_cfg
        elite_desc = f"hall-of-fame (fitness={hof_fitness})"
        crossover_a, crossover_b, crossover_c = elite_cfg, second_cfg, third_cfg
        x12_desc = f"Crossover(HoF,2nd) from gen {latest_num:03d} + mutation (sigma=0.18)"
        x23_desc = f"Crossover(2nd,3rd) from gen {latest_num:03d} + mutation (sigma=0.18)"
    elif hof_fitness > current_best_fitness and stagnation_count >= HOF_DEEP_STAGNATION:
        # Deep stagnation: wide-explore Elite + HoF near-clone + direct 1st×2nd crossover
        # Clamp PROXIMITY_DIST >= 40 on wide-explore to prevent short-range regressions
        # X23 now crosses 1st×2nd to fuse complementary 2p/4p archetypes
        print(f"\nHoF deep stagnation ({stagnation_count} gens) — wide-explore Elite + HoF injection + 1st×2nd")
        candidate = mutate(best_cfg, sigma=0.30)
        candidate["PROXIMITY_DIST"] = max(40.0, candidate["PROXIMITY_DIST"])
        elite_cfg = candidate
        elite_desc = f"wide-explore mutation of gen {latest_num:03d} best (stagnation={stagnation_count})"
        crossover_a, crossover_b, crossover_c = hof_cfg, best_cfg, second_cfg
        x12_desc = f"HoF near-clone (sigma=0.05, stagnation={stagnation_count})"
        x23_desc = f"Crossover(1st,2nd) from gen {latest_num:03d} + mutation (sigma=0.18) — fuse archetypes"
    elif hof_fitness > current_best_fitness and stagnation_count >= HOF_STAGNATION_LIMIT:
        # HoF has been Elite for too long without the population catching up — explore instead
        print(f"\nHoF stagnation ({stagnation_count} gens) — replacing Elite with wide-mutation of current best")
        elite_cfg = mutate(best_cfg, sigma=0.30)
        elite_desc = f"wide-explore mutation of gen {latest_num:03d} best (stagnation={stagnation_count})"
        crossover_a, crossover_b, crossover_c = elite_cfg, second_cfg, third_cfg
        x12_desc = f"Crossover(1st,2nd) from gen {latest_num:03d} + mutation (sigma=0.18)"
        x23_desc = f"Crossover(2nd,3rd) from gen {latest_num:03d} + mutation (sigma=0.18)"
    else:
        elite_cfg = best_cfg
        elite_desc = f"best from gen {latest_num:03d} (fitness={current_best_fitness})"
        crossover_a, crossover_b, crossover_c = elite_cfg, second_cfg, third_cfg
        x12_desc = f"Crossover(1st,2nd) from gen {latest_num:03d} + mutation (sigma=0.18)"
        x23_desc = f"Crossover(2nd,3rd) from gen {latest_num:03d} + mutation (sigma=0.18)"

    # Build new generation
    gen_tag = f"g{next_num:03d}"
    new_configs = {
        "001": {
            "_name": f"Elite-{gen_tag}",
            "_description": f"Elite: {elite_desc}",
            **elite_cfg,
        },
        "002": {
            "_name": f"Mut2-{gen_tag}",
            "_description": f"Mutant of 2nd place from gen {latest_num:03d} (sigma=0.12)",
            **mutate(second_cfg, sigma=0.12),
        },
        "003": {
            "_name": f"X12-{gen_tag}",
            "_description": x12_desc,
            **(mutate(crossover_a, sigma=0.05)
               if x12_desc.startswith("HoF near-clone")
               else crossover(crossover_a, crossover_b, sigma=0.18)),
        },
        "004": {
            "_name": f"X23-{gen_tag}",
            "_description": x23_desc,
            **crossover(crossover_b, crossover_c, sigma=0.18),
        },
    }

    # In deep stagnation, clamp PROXIMITY_DIST >= 40 for all configs to prevent short-range regressions
    if stagnation_count >= HOF_DEEP_STAGNATION and hof_fitness > current_best_fitness:
        for cfg in new_configs.values():
            if cfg.get("PROXIMITY_DIST", 50.0) < 40.0:
                cfg["PROXIMITY_DIST"] = 40.0

    # Write configs
    next_folder.mkdir(parents=True)
    for pid, cfg in new_configs.items():
        path = next_folder / f"{pid}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)
    print(f"\nCreated {next_folder.name}: {[c['_name'] for c in new_configs.values()]}")

    # Commit new configs so git clean is safe later
    subprocess.run(["git", "add", str(next_folder)], cwd=PROJECT_ROOT, check=True)
    subprocess.run(
        ["git", "commit", "-m",
         f"chore: gen {next_num:03d} — genetic evolution from gen {latest_num:03d}"],
        cwd=PROJECT_ROOT, check=True,
    )
    print(f"Committed gen {next_num:03d}.")


if __name__ == "__main__":
    main()
