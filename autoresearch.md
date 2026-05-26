# Autoresearch: Orbit Wars Agent Evolution

## Objective
Maximize agent fitness (2p_wins × 3 + 4p_wins × 2 + 4p_top2) through genetic algorithm
evolution of 14 scoring constants in `76-RL_tournament/main.py`. Each iteration runs a
24-game tournament (18×2p + 6×4p) with 4 evolved agents and records best fitness.

## Metrics
- **Primary**: best_fitness (higher is better; theoretical max = 9×3 + 6×2 + 6 = 45)
- **Secondary**: best_2p_wins (max 9 — each agent plays 3 opponents × 3 games)

## How to Run
```
./autoresearch.sh
```
Outputs `METRIC name=number` lines. Takes ~10-15 min per run (24 games).

## Files in Scope
- `76-RL_tournament/[0-9][0-9][0-9]*/` — one folder per generation; each has 001–004.json + results.json
- `experiments/evolve.py` — genetic algorithm: reads all previous results, creates next-gen folder
- `experiments/parse_results.py` — reads latest results.json, outputs METRIC lines, commits data

## Off Limits
- `76-RL_tournament/main.py` — agent implementation (DO NOT MODIFY)
- `76-RL_tournament.py` — tournament runner (DO NOT MODIFY)
- `76-RL_tournament/000-test/` — seed population (DO NOT MODIFY)

## Constraints
- Exactly 4 agents per folder; tournament runner enforces this
- Generation folders named `001`, `002`, ... (3-digit zero-padded)
- All new folders + results.json are committed immediately (so git clean is safe)
- Mutation uses lognormal noise: val *= exp(N(0, σ)); σ=0.12 for small, 0.18 for crossover

## Scoring Constants (genes)
PROD_MULT, TIME_PROD_MULT, ENEMY_MULT, COMPOUND_MULT, MINE_NEAR_TGT_MULT,
ENEMY_NEAR_TGT_MULT, PROD_SRC_MULT, ORBIT_BONUS, PROXIMITY_MULT,
DIST_MULT, SHIPS_MULT, ETA_MULT, OVEREXTEND_MULT, PROXIMITY_DIST

## Fitness Formula
fitness(pid) = summary["2p"][pid]["wins"] * 3
             + summary["4p"][pid]["wins"] * 2
             + summary["4p"][pid]["top2"]

## Genetic Algorithm (per iteration)
From the most recent completed tournament:
1. **001.json** (Elite)     — exact copy of best agent (hall-of-fame if better)
2. **002.json** (Mutant2)   — 2nd best + small mutation (σ=0.12)
3. **003.json** (Cross12)   — uniform crossover(1st, 2nd) + mutation (σ=0.18)
4. **004.json** (Cross23)   — uniform crossover(2nd, 3rd) + mutation (σ=0.18)

Hall of fame: the single best-ever agent config across all completed tournaments.
If hall-of-fame fitness > current-best fitness, use hall-of-fame as Elite.

## What's Been Tried
- **Gen 000-test** (seed population):
  - Coordinator (003): 7W/2L in 2p — high COMPOUND_MULT=15, MINE_NEAR_TGT_MULT=12
  - OrbitalDominance (004): 7W/2L in 2p — high ORBIT_BONUS=20, low DIST_MULT=0.3
  - Berserker (001): 4W/5L in 2p but 3W in 4p — very high ENEMY_MULT=15, PROXIMITY_MULT=12
  - Economist (002): 0W/9L — high DIST_MULT=1.0 + OVEREXTEND_MULT=1.5 seems harmful
  - Key insight: low DIST_MULT and low OVEREXTEND_MULT seem beneficial
  - Key insight: high COMPOUND_MULT or high ORBIT_BONUS are competitive strategies
