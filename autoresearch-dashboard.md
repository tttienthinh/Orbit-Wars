# Autoresearch Dashboard: orbit-wars-agent-evolution

**Runs:** 36 | **Kept:** 3 | **Discarded:** 33 | **Crashed:** 0
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
| 16 | 9da2f41 | 31 (+14.8%) | 7 | discard | gen 016: HoF clone 0W/9L collapse; X23=31 |
| 17 | af8fcb8 | 22 (-18.5%) | 6 | discard | gen 017: sharp regression; all 16-22 |
| 18 | 67d8634 | 24 (-11.1%) | 7 | discard | gen 018: Mut2=24; stagnation_count=3 fires wide-explore |
| 19 | 5d6a2aa | 31 (+14.8%) | 8 | discard | gen 019: X23=31 (8W/1L 2p); wide-explore |
| 20 | ffd1638 | 22 (-18.5%) | 6 | discard | gen 020: Elite 5W/6 4p but 2W 2p; tradeoff maximally visible |
| 21 | 355076d | 30 (+11.1%) | 7 | discard | gen 021: Elite=30 balanced (ORBIT_BONUS=22.67 archetype) |
| 22 | 776c296 | 27 (+0%) | 6 | discard | gen 022: X12=27; stagnation_count=7 |
| 23 | 9d53978 | 30 (+11.1%) | 6 | discard | gen 023: X23=30 (6W/3L 2p, 4W 4p, top2=4); stagnation_count=8 |
| 24 | 97e2496 | 24 (-11.1%) | 6 | discard | gen 024: X23=24 (6W/3L 2p, 2W 4p); Elite survival top2=6 but low wins; stagnation_count=9 |
| 25 | ec2ba5b | 24 (-11.1%) | 7 | discard | gen 025: X12=24 (7W/2L 2p, 1W 4p); Elite ORBIT=34 survives 4p; 10 gens stagnation → deep strategy |
| 26 | af40e6a | 30 (+11.1%) | 6 | discard | gen 026: X23(HoF×2nd)=30 (6W/3L 2p, 4W 4p, top2=4); HoF crossover works; X12(HoF×1st)=9 failed |
| 27 | ac0e0ea | 28 (+3.7%) | 7 | discard | gen 027: Mut2=28 (7W/2L 2p, 2W 4p); HoF crossovers fail (SHIPS contamination); strategy → HoF injection |
| 28 | e0642da | 30 (+11.1%) | 8 | discard | gen 028: HoF near-clone=30 (8W/1L 2p!); Mut2=30; tied but 4p limited by seed variance |
| 29 | 464571b | 28 (+3.7%) | 6 | discard | gen 029: 3 low-SHIPS agents split wins; Elite(high-SHIPS)=28 won by differentiation; PROD=18 collapsed |
| 30 | 09b0a25 | 28 (+3.7%) | 8 | discard | gen 030: X23(PROD=15.5,SHIPS=0.417)=28 (8W/1L 2p); Mut2 4p-specialist=15; 2p/4p split persists |
| 31 | f46b769 | 24 (-11.1%) | 6 | discard | gen 031: X23(PROD=17.1,SHIPS=0.14,PROX=44,ENEMY=6.73)=24; near-HoF params still 2p/4p split |
| 32 | 359aff2 | 28 (+3.7%) | 7 | discard | gen 032: Elite(PROD=20.69)=28 (7W 2p); HoF near-clone crushed at 9; high PROD beats HoF in head-to-head |
| 33 | dbd9093 | 25 (-7.4%) | 7 | discard | gen 033: HoF near-clone=25 (7W 2p) when PROD=26.88+PROX=27.6 collapsed; PROX as critical as PROD |
| 34 | 96d909c | 22 (-18.5%) | 6 | discard | gen 034: Mut2=X23=22; PROX<35 again crippled Elite; PROX≥40 clamp added to wide-explore |
| 35 | 1cb84be | 28 (+3.7%) | 9 | discard | gen 035: X12 PERFECT 9W/0L 2p! but 0W 4p — ultimate 2p specialist; PROX clamp working |
| 36 | f2ef72b | 26 (-3.7%) | 7 | discard | gen 036: Elite(SHIPS=0.057)=26 2p; Mut2(ENEMY=16.21)=24 5W 4p RECORD! fusion=36 if combined |

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
