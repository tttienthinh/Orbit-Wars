# Autoresearch Dashboard: orbit-wars-agent-evolution

**Runs:** 15 | **Kept:** 3 | **Discarded:** 12 | **Crashed:** 0
**Baseline:** best_fitness: 27pts (gen 000 OrbDom, seed=16)
**Best:** best_fitness: **34pts** (#15, +25.9% vs baseline) — X23-g015

| # | commit | best_fitness | 2p_wins | status | description |
|---|--------|-------------|---------|--------|-------------|
| 1 | 37d1db6 | 25 (-7.4%) | 6 | discard | gen 001: X12(OrbDom×Berserker) best; Elite worst in 2p |
| 2 | 2ec994e | 23 (-14.8%) | 4 | discard | gen 002: X23 dominates 2p (7W) but 0 4p wins |
| 3 | 2303269 | 22 (-18.5%) | 5 | discard | gen 003: plateau; X12 crossover collapsed in 4p |
| 4 | c8285be | 22 (-18.5%) | 5 | discard | gen 004: OrbDom Elite got 0 4p wins — HoF seed-specific |
| 5 | 5faa2ad | 22 (-18.5%) | 5 | discard | gen 005: 3rd straight at 22; stagnation detector primed |
| 6 | e5966bd | **31 (+14.8%)** | 7 | **keep** | gen 006: stagnation-break (σ=0.30); Mut2(mutated X23-g005)=31 |
| 7 | 9f14628 | 31 (+14.8%) | 9 | discard | gen 007: X12 ties HoF (9W/0L 2p!) but only 1 4p win |
| 8 | 8f933e5 | 26 (-3.7%) | 7 | discard | gen 008: Elite(X12-g007 clone)=26; HoF remains Mut2-g006 |
| 9 | f987cef | **33 (+22.2%)** | 8 | **keep** | gen 009: Mut2-g009 new HoF=33; 8W/1L 2p + 2W 4p + top2=5 |
| 10 | 2ccc711 | 23 (-14.8%) | 5 | discard | gen 010: tightly clustered 18-23; HoF clone variance |
| 11 | 4906162 | 27 (+0%) | 6 | discard | gen 011: Mut2 best=27; stagnation_count=2 |
| 12 | a4a6ed4 | 25 (-7.4%) | 6 | discard | gen 012: Mut2 best=25; stagnation_count=3, wide-explore primed |
| 13 | edeb0b5 | 28 (+3.7%) | 7 | discard | gen 013: wide-explore Elite failed; X23=28 — high COMPOUND_MULT |
| 14 | a228bb2 | 33 (+22.2%) | 7 | discard | gen 014: X12 ties HoF=33 via PROXIMITY_DIST=44.9, low ENEMY_MULT |
| 15 | f87d874 | **34 (+25.9%)** | 7 | **keep** | gen 015: X23-g015 new HoF=34; PROD_MULT=14.82, SHIPS_MULT=0.103 |

## Gen Standings

### Gen 009 (new HoF!)
| Rank | Agent | Fitness | 2p W/L | 4p W | top2 |
|------|-------|---------|--------|------|------|
| 1 | **Mut2-g009** (mutated Mut2-g008) | **33** | 8/1 | 2 | 5 |
| 2 | Elite-g009 (HoF Mut2-g006 clone) | 24 | 6/3 | 2 | 2 |
| 3 | X12-g009 | 13 | 3/6 | 1 | 2 |
| 4 | X23-g009 | 8 | 1/8 | 1 | 3 |

### Gen 007
| Rank | Agent | Fitness | 2p W/L | 4p W | top2 |
|------|-------|---------|--------|------|------|
| 1 | X12-g007 (crossover HoF×Mut2-g006-2nd) | 31 | 9/0 | 1 | 2 |
| 2 | Mut2-g007 | 19 | 6/3 | 0 | 1 |
| 2 | Elite-g007 (Mut2-g006 clone) | 19 | 3/6 | 2 | 6 |
| 4 | X23-g007 | 9 | 0/9 | 3 | 3 |

### Gen 006 (first breakthrough)
| Rank | Agent | Fitness | 2p W/L | 4p W | top2 |
|------|-------|---------|--------|------|------|
| 1 | **Mut2-g006** (mutated X23-g005) | **31** | 7/2 | 3 | 4 |
| 2 | Elite-g006 (wide-explore σ=0.30) | 19 | 4/5 | 1 | 5 |
| 3 | X12-g006 | 18 | 4/5 | 2 | 2 |
| 4 | X23-g006 | 10 | 3/6 | 0 | 1 |

## New Hall-of-Fame Config (X23-g015, fitness=34)
```json
PROD_MULT=14.82, TIME_PROD_MULT=0.65, ENEMY_MULT=6.72, COMPOUND_MULT=4.49,
MINE_NEAR_TGT_MULT=2.24, ENEMY_NEAR_TGT_MULT=1.34, PROD_SRC_MULT=6.72,
ORBIT_BONUS=13.25, PROXIMITY_MULT=5.34, DIST_MULT=0.37, SHIPS_MULT=0.103,
ETA_MULT=0.85, OVEREXTEND_MULT=0.26, PROXIMITY_DIST=44.49
```
Key insight: PROD_MULT=14.82 (highest ever), SHIPS_MULT=0.103 (lowest ever), PROXIMITY_DIST=44.49 (near full-board evaluation). Wide-proximity archetype from X12-g014 parent. ENEMY_MULT=6.72 lower than previous HoF.

## Previous Hall-of-Fame Config (Mut2-g009, fitness=33)
```json
PROD_MULT=11.76, TIME_PROD_MULT=0.47, ENEMY_MULT=7.89, COMPOUND_MULT=4.56,
MINE_NEAR_TGT_MULT=3.03, ENEMY_NEAR_TGT_MULT=1.82, PROD_SRC_MULT=6.32,
ORBIT_BONUS=14.66, PROXIMITY_MULT=5.89, DIST_MULT=0.56, SHIPS_MULT=0.15,
ETA_MULT=0.63, OVEREXTEND_MULT=0.22, PROXIMITY_DIST=28.07
```
Key delta from previous HoF: higher PROD_MULT (+0.8), COMPOUND_MULT (+0.8), PROD_SRC_MULT (+1.2), lower OVEREXTEND_MULT (0.22 vs 0.28). Very low OVEREXTEND_MULT is the key differentiator — agent commits ships aggressively with minimal penalty for stretching resources.

## Key Insights
1. Low DIST_MULT (<0.6) consistently outperforms high DIST_MULT
2. HoF mechanism caused 5-gen stagnation (seed=16 OrbDom was seed-lucky)
3. Stagnation-break via σ=0.30 wide-explore was the right intervention
4. Very low SHIPS_MULT + low ETA_MULT = core optimum region
5. **Very low OVEREXTEND_MULT** (<0.25) = new critical insight from gen 009
6. HoF clones consistently underperform original (19-24 vs 31+) — high seed variance
7. Balanced 2p+4p optimum found: Mut2-g009 achieves both without extreme specialization

## Key Insights
1. Low DIST_MULT (<0.5) consistently outperforms high DIST_MULT
2. Very low SHIPS_MULT + ETA_MULT = core optimum (don't care about source size or timing)
3. **PROXIMITY_DIST=44-45** (near full-board) is new breakthrough — evaluate all planets
4. **High PROD_MULT (14+)** is new direction — production dominance strategy
5. Two distinct archetypes at fitness=33-34: high-SHIPS aggression (gen 009) vs wide-proximity production (gen 014-015)
6. Stagnation-break (σ=0.30) works reliably when plateau detected (3 gens)
7. HoF clones always underperform original — high tournament seed variance

## Trajectory
Gen: 000(27) → 006(**31**) → 009(**33**) → 015(**34**)
