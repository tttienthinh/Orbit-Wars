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
- **Baseline HoF** (gen 015, fitness=34): PROD=14.82, SHIPS=0.103, PROX=44.49, ENEMY=6.72, OVER=0.256, ORBIT=~12

- **Best non-HoF** (gen 024 Elite, fitness=33): PROD=11.83, SHIPS=0.053, PROX=50, ENEMY=13.94, ORBIT=18.24
  - Achieved 7W/2L 2p + 3W 4p + top2=6 simultaneously — proving both-format dominance is possible

- **Critical constraints** (hard rules from 41 experiments):
  - PROXIMITY_DIST MUST be ≥ 40; PROX<35 = catastrophic 0-win collapse in 2p
  - PROXIMITY_DIST = 50 is safe with HIGH ENEMY (13+) because abundant neutrals absorb aggression
  - ENEMY_MULT 8-16 with PROX=50 seems fine for 2p — contradicts earlier assumption of ≤7
  - ENEMY_MULT ≤ 7 favors 2p when PROX < 48 (agent runs out of neutrals, avoids enemy planets)
  - SHIPS_MULT 0.04-0.11 is the working range; lower = faster expansion (more attacks)
  - PROD_MULT 10-16 works; PROD~12 may be optimal for balanced 2p+4p

- **Failed strategies** (dead ends):
  - HoF near-clone (σ=0.05 of gen 015): consistently fails in 2p (0-2W/9L typical), fitness 6-28
  - High ENEMY (10-16) with PROX<45: 0W/9L 2p pattern (gen 037 X23 ENEMY=10.15, PROX=40.7)
  - Ultra-wide exploration (σ=0.30) without PROX clamp: generates PROX=25-35 → catastrophe
  - Crossover of 2p-specialist × 4p-specialist: if ENEMY blends to 8-10, can work; if to 10+, 2p collapses

- **Promising directions**:
  - Wide-explore (σ=0.30) of recent winners with PROX clamped ≥ 40: produced best results (33)
  - PROX=50 + HIGH ENEMY (8-16): the gen 024 Elite archetype — most reliable high-fitness pattern
  - Crossover of gen 015 HoF × gen 024 Elite configs: unexplored, might find fitness>34 sweet spot

- **Stagnation tracking**: HoF=34 (gen 015), current stagnation=11 gens (gens 016-026 all below 34)
