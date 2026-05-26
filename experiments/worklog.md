# Autoresearch Worklog: Orbit Wars Agent Evolution

**Session start:** 2026-05-26
**Branch:** autoresearch/agent-evolution-2026-05-26
**Goal:** Maximize agent fitness = 2p_wins × 3 + 4p_wins × 2 + 4p_top2

## Seed Population (Gen 000-test)

| Agent | Strategy | 2p W/L | 4p W | top2 | Fitness |
|-------|----------|--------|------|------|---------|
| 001 Berserker | High aggression (ENEMY=15, PROX=12) | 4/5 | 3 | 5 | 23 |
| 002 Economist | High dist/overextend cost | 0/9 | 1 | 3 | 7 |
| 003 Coordinator | Cluster attacks (COMP=15) | 7/2 | 0 | 2 | 23 |
| 004 OrbitalDominance | Orbit bonus (ORBIT=20, DIST=0.3) | 7/2 | 2 | 2 | 27 |

**Seed = 16** | **Hall-of-fame: OrbitalDominance (fitness=27)**

Key observations:
- 004 OrbitalDominance wins with fitness=27 — low DIST_MULT=0.3, high ORBIT_BONUS=20
- 002 Economist is worst — high DIST_MULT=1.0 and OVEREXTEND_MULT=1.5 appear very harmful
- Low DIST_MULT seems crucial: 001(0.2), 003(0.8), 004(0.3) > 002(1.0)

---

### Run 1: Gen 001 tournament — best_fitness=25 (DISCARD — regression from 27)
- Timestamp: 2026-05-26
- What changed: Elite=OrbitalDominance clone, Mut2=mutated Berserker, X12=crossover(OrbDom,Berserker), X23=crossover(Berserker,Coordinator)
- Result: X12-g001 best at fitness=25 (6W/3L 2p, 2W 4p, top2=3); Elite last in 2p (3W/6L) but strong in 4p (3W top2=6)
- Insight: Pure clone of OrbitalDominance struggles in 2p head-to-head but excels in 4p survival. Crossover with Berserker improves 2p aggression. X23(Berserker+Coordinator) worst — fitness=12, no 4p wins at all.
- Next: Gen 002 — hall-of-fame (OrbDom) as Elite + mutations of X12 and Elite; new crossovers from top-2 of gen 001.

---

## Key Insights

1. **Low DIST_MULT wins**: agents with DIST_MULT < 0.5 consistently outperform those with DIST_MULT ≥ 0.8
2. **High ORBIT_BONUS is competitive**: inner-ring planets as launch pads seem valuable
3. **Economist archetype loses badly**: balanced production + high penalties = 0 wins
4. **2p vs 4p tradeoff**: OrbitalDominance archetype wins 4p survival but loses 2p head-to-head; X12 crossover with Berserker balances both modes better
5. **HoF stagnation is real**: seed=16 OrbDom was lucky; hard-code HoF stagnation limit (3 gens) saved the run
6. **New optimum region (gen 006-009)**: SHIPS_MULT~0.13-0.15, ETA_MULT~0.62-0.75, DIST_MULT~0.45-0.56, OVEREXTEND_MULT~0.22-0.28 — very low source-ship/timing sensitivity + very low overextend penalty. Grab planets aggressively early; production advantage compounds.
7. **2p specialist pattern**: extreme 2p wins (9W/0L) comes with 4p collapse — true combined optimum not yet found
8. **Balanced optimum emerging (gen 009)**: Mut2-g009 fitness=33 with 8W/1L 2p AND 2W 4p + top2=5 — best combined performance yet. Very low OVEREXTEND_MULT=0.22 key differentiator.
9. **HoF clone underperforms**: exact clones of HoF agent consistently score lower (19-24) than original run — variance is real, agent configs are at optimum only for specific seed conditions

## Next Ideas

- Try even lower SHIPS_MULT (0.05-0.10) and ETA_MULT (0.3-0.5)
- Try PROXIMITY_DIST scan: 10-15 vs 20-25 — does tighter proximity window improve 4p?
- Investigate what drives 4p wins vs 2p wins — need an agent that balances both modes

## Next Ideas

- Try much lower OVEREXTEND_MULT (<0.1)
- Try even higher ORBIT_BONUS (25-30)
- Try very aggressive crossover between OrbitalDominance and Coordinator
- Investigate why X23(Berserker+Coordinator) had 0 4p wins — check X23 config for extreme parameter values

### Run 2: Gen 002 tournament — best_fitness=23 (DISCARD — still below HoF 27)
- Timestamp: 2026-05-26
- What changed: Elite=OrbDom (hall-of-fame), Mut2=mutated X12-g001, X12=crossover(OrbDom,X12-g001), X23=crossover(X12-g001,Elite-g001)
- Result: Elite-g002=23 (4W/5L 2p, 3W 4p top2=5), X23-g002=22 (7W/2L 2p, 0W 4p) — strong 2p vs 4p split
- Insight: X23-g002 hit 7W/2L in 2p (best 2p result so far) but 0 4p wins. This 2p specialist / 4p failure pattern recurring — the genes that win 2p duels (high aggression, push resources) cause overextension / early elimination in 4p. OrbDom Elite continues to be 4p-dominant but mediocre in 2p head-to-head. True combined optimum hasn't been found yet.
- Next: Gen 003 — hall-of-fame Elite + Mut2=mutated Elite-g002, X12=crossover(OrbDom HoF, X23-g002 2p-specialist), X23=crossover(X23-g002, X12-g002)

### Run 3: Gen 003 tournament — best_fitness=22 (DISCARD — declining trend)
- Timestamp: 2026-05-26
- What changed: Elite=OrbDom HoF, Mut2=mutated Elite-g002, X12=crossover(OrbDom,Elite-g002), X23=crossover(Elite-g002, X23-g002)
- Result: Mut2-g003=22, Elite-g003=21, X23-g003=21, X12-g003=14 — all agents tightly clustered except X12 collapsed (0 4p wins)
- Insight: Declining fitness trend (27→25→23→22). OrbDom HoF likely had lucky seed=16. High variance per tournament (only 24 games). X12 crossover pattern consistently produces 4p failures — might be unlucky gene combo that over-commits ships early. The clustering of agents around 21-22 suggests local convergence but not yet above the HoF baseline.

### Run 4: Gen 004 tournament — best_fitness=22 (DISCARD — plateau at 22)
- Timestamp: 2026-05-26
- What changed: Elite=OrbDom HoF, Mut2=mutated Mut2-g003, X12=crossover(OrbDom,Mut2-g003), X23=crossover(Mut2-g003,Elite-g003)
- Result: X12-g004=22, Mut2-g004=21, Elite-g004=18 (0 4p wins!), X23-g004=17
- Insight: Confirmed — OrbDom Elite gets 0 4p wins in gen 004 tournament despite being HoF. The HoF fitness=27 was almost certainly seed=16 specific. Algorithm is plateauing at 22. Hall-of-fame mechanism is counter-productive here: keeps injecting an agent that can't reproduce its seed-lucky performance. Trend: 27→25→23→22→22.

### Run 5: Gen 005 tournament — best_fitness=22 (DISCARD — 3rd straight at 22)
- Timestamp: 2026-05-26
- What changed: Elite=OrbDom HoF, Mut2=mutated X12-g004, X12=crossover(OrbDom,X12-g004), X23=crossover(X12-g004,Mut2-g004)
- Result: X12-g005=22, X23-g005=20, Elite=18, Mut2=18 — same plateau
- Insight: 3rd consecutive gen at 22. stagnation_count=3 threshold reached — gen 006 will fire wide-explore mutation.

### Run 6: Gen 006 tournament — best_fitness=31 (KEEP — new hall-of-fame!)
- Timestamp: 2026-05-26
- What changed: Elite=wide-mutation(X12-g005, sigma=0.30) due to stagnation; Mut2=mutated X23-g005
- Result: Mut2-g006=31 (7W/2L 2p, 3W 4p, top2=4), Elite=19, X12=18, X23=10
- Insight: BREAKTHROUGH! stagnation-break via σ=0.30 wide-explore worked. Mut2-g006 (small mutation of X23-g005) jumped to 31. Key config differences from OrbDom: SHIPS_MULT=0.15 (vs 1.5), ETA_MULT=0.75 (vs 2.0), OVEREXTEND_MULT=0.28 — doesn't value source strength or timing, focuses on production/orbit.
- Next: gen 007 — Mut2-g006 as Elite, evolve around this new optimum

### Run 7: Gen 007 tournament — best_fitness=31 (DISCARD — ties HoF, no improvement)
- Timestamp: 2026-05-26
- What changed: Elite=Mut2-g006 clone, Mut2=mutated Elite-g006(2nd), X12=crossover(Mut2-g006,Elite-g006), X23=crossover(Elite-g006,X12-g006)
- Result: X12-g007=31 (9W/0L 2p, 1W 4p, top2=2), Elite=19 (3W/6L 2p but 2W 4p top2=6), Mut2=19, X23=9
- Insight: X12-g007 achieves PERFECT 9W/0L in 2p but only 1 4p win — extreme 2p specialist. Elite clone (Mut2-g006) excels in 4p (top2=6) but struggles in 2p. Same extreme specialization split as before. Fitness=31 ties HoF but doesn't beat it → DISCARD. Key: DIST_MULT drifted lower (0.26) in X12.
- Next: Gen 008 — use X12-g007 as Elite (best 2p), try to find balance

### Run 8: Gen 008 tournament — best_fitness=26 (DISCARD — regression from HoF 31)
- Timestamp: 2026-05-26
- What changed: Elite=X12-g007 clone (2p specialist), Mut2=mutated Elite-g007(2nd), X12=crossover(X12-g007,Mut2-g007), X23=crossover(Mut2-g007,X12-g007)
- Result: Elite-g008=26 (7W/2L 2p, 1W 4p top2=3), Mut2-g008=25, X12-g008=17, X23-g008=10
- Insight: X12-g007 clone as Elite reaches only 26 — lower than HoF=31. Stagnation_count reset to 0 since gen007 tied HoF. Algorithm falls back to HoF Mut2-g006 as Elite for gen009. X23 again fails badly (10) — 3rd/4th in gen consistently underperform.
- Next: Gen 009 — HoF (Mut2-g006) as Elite, evolve from Mut2-g008 and X12-g008

### Run 9: Gen 009 tournament — best_fitness=33 (KEEP — new hall-of-fame!)
- Timestamp: 2026-05-26
- What changed: Elite=HoF Mut2-g006 clone, Mut2=mutated Mut2-g008 (2nd from gen008), X12=crossover(HoF,Mut2-g008), X23=crossover(Mut2-g008,X12-g008)
- Result: Mut2-g009=33 (8W/1L 2p, 2W 4p, top2=5), Elite=24, X12=13, X23=8
- Insight: NEW RECORD fitness=33! Mut2-g009 (mutant of gen008 2nd place) achieves best-ever combined 2p+4p performance. Key deltas from HoF: DIST_MULT=0.56 (vs 0.45), OVEREXTEND_MULT=0.22 (vs 0.28), PROXIMITY_DIST=28.1 (vs 24.7). Very low OVEREXTEND_MULT continues to be critical. Elite clone only reached 24 — HoF fitness not reproducible even with exact clone.
- Next: Gen 010 — Mut2-g009 as new Elite, evolve around fitness=33 optimum
