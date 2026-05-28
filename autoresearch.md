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

### New Vice-HoF (gen 066 Mut2, fitness=34)
ORBIT=32.00, ENEMY=4.14, SHIPS=0.097, PROX=48.3, COMPOUND=11.26
- 8W/1L 2p + 3W 4p top2=4 — exceptional combined 2p+4p performance
- Ties gen 049 Elite (old vice-HoF) at 34pts but with different params (higher ORBIT, lower ENEMY)
- Emerged from ultra-deep stagnation (37 gens) via Mut2 of X23-g065

### Current Stagnation: 37 gens (gens 030-066)
Best reachable without HoF seed: ~34pts (gen 066 Mut2) — new ceiling found!
Previous ceiling was 28-30pts; gen 066 shows the algorithm CAN approach HoF territory.

### Hard Rules (updated through gen 081, 81 experiments)
- PROXIMITY_DIST ≥ 44 always (clamped); PROX=48-50 optimal
- SHIPS_MULT ≤ 0.20 always (clamped); 0.087-0.13 optimal range
- ORBIT_BONUS max = 32 (capped; ORBIT>32 = extreme seed variance)
- ENEMY_MULT < 3: CATASTROPHIC (gen 066 Elite ENEMY=2.33 → 1W/8L 2p, fitness=8)
- ENEMY_MULT 4-5 with ORBIT=32 works well (gen 066 Mut2 ENEMY=4.14 → 8W/1L)
- COMPOUND_MULT > 20: consistently underperforms despite being explored 5+ times
- ORBIT=32 + ENEMY~4 + SHIPS~0.10 is an emerging sweet spot (fitness=34)

### Failed Directions
- Near-HoF clones (ORBIT~28, ENEMY~6-7): 6W 2p but 0-1W 4p on every non-HoF seed
- HIGH PROD (>19): no benefit
- HIGH SHIPS (0.20 MAX): no benefit
- ORBIT>32: consistently bad in 2p
- ENEMY < 3: catastrophic (1W/8L 2p confirmed gen 066)
- HIGH COMPOUND (>20): underperforms consistently despite many attempts
- Vice-HoF clones (ORBIT~28-30, ENEMY~5-6): 5-7W 2p but 0W 4p wins

### Seed Variance Problem
The fitness metric is dominated by seed luck. Examples:
- Gen 064 Elite (ORBIT=21.5, ENEMY=3.28): 6W/3L 2p on seed=84
- Gen 065 Elite (same params): 4W/5L 2p on seed=9
The HoF=36 required an exceptional seed alignment. Gen 066 Mut2=34 also benefited from seed=22.

### Next Directions to Try (if resuming)
- Focus ORBIT=30-32 + ENEMY=4-6 + SHIPS=0.08-0.12 (new sweet spot from gen 066)
- Keep exploring gen 067+ which evolves from gen 066's vice-HoF Mut2
- ETA_MULT and OVEREXTEND_MULT rarely explored — unknown territory
- Gen 067 configs already created, tournament was in progress when stopped
