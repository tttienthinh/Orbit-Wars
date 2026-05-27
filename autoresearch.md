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
- **NEW HoF** (gen 029 Elite, fitness=36): PROD=15.97, SHIPS=0.087, PROX=50.0, ENEMY=6.81, ORBIT=27.75, COMPOUND=17.73, OVER=0.476
  - PERFECT 9W/0L 2p + 2W 4p + top2=5 — best result ever, stagnation resets to 0
  - Key: PROX=50 (max board reach) + LOW ENEMY=6.81 + HIGH ORBIT=27.75 + HIGH COMPOUND=17.73

- **Old HoF** (gen 015, fitness=34): PROD=14.82, SHIPS=0.103, PROX=44.49, ENEMY=6.72, OVER=0.256, ORBIT=~12
  - Superseded by gen 029

- **Vice-HoF** (gen 024 Elite, fitness=33): PROD=11.83, SHIPS=0.053, PROX=50, ENEMY=13.94, ORBIT=18.24
  - Still useful as a diverse archetype (HIGH ENEMY vs LOW ENEMY)

- **Critical constraints** (hard rules from 44 experiments):
  - PROXIMITY_DIST MUST be ≥ 44; PROX<40 = catastrophic 0-win collapse in 2p
  - PROXIMITY_DIST = 50 optimal — used in both HoF (gen 029) and vice-HoF (gen 024)
  - ENEMY_MULT ≤ 7 strongly preferred for 2p; higher ENEMY only works with PROX=50 AND abundant neutrals
  - ORBIT_BONUS 27+ appears highly beneficial — gen 029 HoF has ORBIT=27.75 (highest yet in a winner)
  - COMPOUND_MULT 17+ may help coordinate multi-fleet captures
  - SHIPS_MULT 0.04-0.11 is the working range; 0.087 optimal for combined 2p+4p
  - PROD_MULT 15-16 works well; combined with low ENEMY and high ORBIT

- **Failed strategies** (dead ends):
  - HoF near-clone (σ=0.05): consistently fails to reproduce HoF=34/36 on different seeds
  - High ENEMY (10-16) with PROX<45: 0W/9L 2p pattern
  - Ultra-wide exploration (σ=0.30) without PROX clamp: generates PROX=25-35 → catastrophe
  - Crossover of 2p-specialist × 4p-specialist: if ENEMY blends to 8-10+, 2p collapses

- **Promising directions**:
  - Wide-explore (σ=0.30) of recent winners with PROX=50: produced new HoF=36!
  - High ORBIT_BONUS (27+) + low ENEMY (≤7) + PROX=50: the new champion archetype
  - Explore COMPOUND_MULT further (17.73 in HoF — is higher better?)
  - Push fitness above 36: theoretical max = 45; currently at 80% of max

- **Stagnation tracking**: HoF=36 (gen 029), stagnation=0 (just reset)
