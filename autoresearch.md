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

### HoF (gen 029, fitness=36) — UNREPRODUCIBLE
ORBIT=27.75, ENEMY=6.81, SHIPS=0.087, PROX=50, PROD=15.97, COMPOUND=17.73
- PERFECT 9W/0L 2p + 2W 4p + top2=5 — seed-dependent, cannot be reproduced on different seeds
- 34 consecutive gens (030-063) below this — all near-HoF clones fail 4p on different seeds

### Vice-HoF (gen 049 Elite, fitness=34)
ORBIT=28.96, ENEMY=5.67, SHIPS=0.125, PROX=50
- 9W/0L 2p + 1W 4p top2=5 — best 2p record since HoF but 4p weaker

### Confirmed Archetypes (3 distinct behaviors)
1. **2p-specialist** (ORBIT=27-32, ENEMY=5-8, PROX=48-50): 6-8W 2p, 0-2W 4p → 15-28pts
2. **4p-ALL-TOP2** (ORBIT=20-32, ENEMY=8-16, SHIPS<0.10): 3-6W 2p, 0W but top2=ALL 6 4p → 17-28pts
   - Occurred 10+ consecutive times — these configs finish 2nd in every 4p game but never win
3. **Balanced** (ORBIT=25-30, ENEMY=6-9): 4-6W 2p, 1-3W 4p → 18-25pts

### Hard Rules (confirmed by 78 experiments)
- PROXIMITY_DIST ≥ 44 always (clamped in evolve.py); PROX=50 optimal
- SHIPS_MULT ≤ 0.20 always (clamped); HoF used 0.087
- ORBIT_BONUS max = 32 (capped; ORBIT>32 = extreme seed variance, often kills 2p)
- ENEMY_MULT < 5 with ORBIT>30 = 2p collapse (confirmed gen 062 X23)
- ORBIT=35: extreme variance — 8W/1L one seed, 2W/7L another seed

### Failed Directions
- Near-HoF clones (ORBIT~28, ENEMY~6-7): 6W 2p but 0-1W 4p on every non-HoF seed
- HIGH PROD (>19): no benefit (confirmed gen 060 X23)
- HIGH SHIPS (0.20 MAX): no benefit (confirmed gen 063 Elite)
- ORBIT>33: consistently bad in 2p (~2-3W/7-6L)
- Very low ENEMY (<5) + high ORBIT: 2p collapse
- Vice-HoF direction (ORBIT~28-30, ENEMY~5-6): 5-7W 2p but 0W 4p wins

### Seed Variance Problem
The fitness metric is dominated by seed luck. Same ORBIT=35 config:
- Gen 054: 2W/7L 2p (bad seed)
- Gen 061: 8W/1L 2p (good seed)
The HoF=36 required an exceptional seed alignment that gives 9W 2p + 2W 4p simultaneously.

### Current Stagnation: 34 gens (gens 030-063)
Best reachable without HoF seed: ~28pts (gen 027/028/055/059/062)
The algorithm generates good configs (22-28pts typical) but cannot reproduce HoF=36 seed luck.

### Next Directions to Try
- Focus ORBIT=27-30 + ENEMY=7-9 zone more tightly (the balanced archetype)
- Try COMPOUND_MULT exploration (HoF has 17.73 — much higher than typical ~5-10)
- Try ETA_MULT and OVEREXTEND_MULT exploration (rarely explored)
- Possible that a fundamentally different gene combination unlocks the 4p wins needed
