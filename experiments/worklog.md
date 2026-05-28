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
10. **Wide-proximity archetype (gen 014-015)**: PROXIMITY_DIST~44-45 (near full-board) + low ENEMY_MULT + very low SHIPS_MULT = new optimum region. Evaluates almost all planets, ignores source ship count, focuses purely on production value. Leads to 4p dominance (4 4p wins).
11. **New HoF=34 (gen 015)**: X23-g015 — PROD_MULT=14.82, SHIPS_MULT=0.103, PROXIMITY_DIST=44.49, ENEMY_MULT=6.72. Trajectory: 000(27)→006(31)→009(33)→015(34)

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

### Run 10: Gen 010 tournament — best_fitness=23 (DISCARD — regression from HoF 33)
- Timestamp: 2026-05-26
- What changed: Elite=Mut2-g009 clone (HoF), Mut2=mutated Elite-g009(2nd), X12=crossover(Mut2-g009,Elite-g009), X23=crossover(Elite-g009,X12-g009)
- Result: X12-g010=23 (5W/4L 2p, 2W 4p top2=4), Mut2-g010=19, Elite=18, X23=18 — all tightly clustered
- Insight: High variance between runs (re-run with different seed gave Mut2-g010=27 vs X12=23 originally). On re-run: Mut2-g010=27, Elite=22, X23=18, X12=11 — Mut2 emerged as best. Still < HoF=33 → DISCARD. stagnation_count=1. Note: discard protocol now keeps results.json untracked (not deleted) to allow evolve.py to progress.
- Next: Gen 011 — HoF (Mut2-g009) as Elite; evolve from Mut2-g010 (2nd re-run) and Elite-g010

### Run 11: Gen 011 tournament — best_fitness=27 (DISCARD — below HoF 33)
- Timestamp: 2026-05-27
- What changed: Elite=HoF Mut2-g009 clone, Mut2=mutated Elite-g010(2nd), X12=crossover(HoF,Elite-g010), X23=crossover(Elite-g010,Mut2-g010)
- Result: Mut2-g011=27 (6W/3L 2p, 3W 4p, top2=3), Elite=25 (5W/4L 2p, 2W 4p top2=6), X23=14, X12=12
- Insight: stagnation_count=2. Mut2 operator consistently outperforms X12/X23 crossovers in recent gens. Elite clone best 4p survival (top2=6) but loses 2p. Plateau at 25-27 is persistent under HoF guidance. Need either lucky seed or different mutation strategy.
- Next: Gen 012 — HoF as Elite; stagnation_count=2, one more before wide-explore fires

### Run 12: Gen 012 tournament — best_fitness=25 (DISCARD — stagnation_count=3, wide-explore fires next!)
- Timestamp: 2026-05-27
- What changed: Elite=HoF Mut2-g009 clone, Mut2=mutated Mut2-g011(1st), X12=crossover(HoF,Mut2-g011), X23=crossover(Mut2-g011,Elite-g011)
- Result: Mut2-g012=25, X12-g012=23, Elite=20 (0 4p wins!), X23=10
- Insight: stagnation_count=3 reached. HoF clone again gets 0 4p wins — extreme seed variance. Mut2 operator consistently best (25-27) but can't reach HoF=33. Pattern: gens 010-012 plateau at 25-27 under HoF guidance. Wide-explore (σ=0.30) fires for gen 013.
- Next: Gen 013 — wide-explore mutation (σ=0.30) of current best replaces Elite

### Run 13: Gen 013 tournament — best_fitness=28 (DISCARD — wide-explore Elite failed)
- Timestamp: 2026-05-27
- What changed: Elite=wide-explore(σ=0.30) of Mut2-g012; Mut2=mutated X12-g012, X12/X23=crossovers of gen012 top agents
- Result: X23-g013=28 (7W/2L 2p, 2W 4p, top2=3), X12-g013=21, Mut2=18, Elite=11 (2W/7L — ETA_MULT jumped to 1.40)
- Insight: Wide-explore Elite catastrophically bad (ETA_MULT=1.40 vs optimum ~0.63). But X23-g013 (crossover of X12×Elite from gen012) hit 28 — best in recent gens. X23 has COMPOUND_MULT=5.81 (high), PROXIMITY_DIST=30.75 (high). stagnation_count=4, wide-explore fires again for gen 014.
- Next: Gen 014 — wide-explore(σ=0.30) of X23-g013 (best this gen); explore high COMPOUND_MULT direction

### Run 14: Gen 014 tournament — best_fitness=33 (DISCARD — ties HoF, no improvement)
- Timestamp: 2026-05-27
- What changed: Elite=wide-explore(σ=0.30) of X23-g013; Mut2=mutated X12-g013(2nd), X12/X23=crossovers
- Result: X12-g014=33 (7W/2L 2p, 4W 4p, top2=4), Elite=24, X23=18, Mut2=3 (complete collapse)
- Insight: X12-g014 ties HoF=33 but via DIFFERENT archetype — ENEMY_MULT=4.50 (low), COMPOUND_MULT=6.48 (high), PROXIMITY_DIST=44.92 (nearly full board). This is a 4p-dominant config (4 4p wins vs HoF's 2). Two distinct strategies now tie at 33. Stagnation resets. Gen 015 = X12-g014 as Elite (HoF no longer strictly > current).
- Next: Gen 015 — X12-g014 as Elite; evolve around low-ENEMY/high-COMPOUND archetype

### Run 15: Gen 015 tournament — best_fitness=34 (KEEP — new hall-of-fame!)
- Timestamp: 2026-05-27
- What changed: Elite=X12-g014 clone, Mut2=mutated Elite-g014(2nd), X12/X23=crossovers of gen014 top agents
- Result: X23-g015=34 (7W/2L 2p, 4W 4p, top2=5), Elite=23, Mut2=12, X12=6
- Insight: BREAKTHROUGH! X23-g015 (crossover of Elite-g014×X23-g014) achieves fitness=34 — new HoF. Key config: PROD_MULT=14.82 (highest ever), SHIPS_MULT=0.103 (lowest ever), PROXIMITY_DIST=44.49 (near full-board). Inherited X12-g014's wide-proximity archetype. Two gens of wide-PROXIMITY_DIST convergence → new optimum.
- Next: Gen 016 — X23-g015 as Elite; explore PROD_MULT>14, SHIPS_MULT<0.10 direction

### Run 16: Gen 016 tournament — best_fitness=31 (DISCARD — HoF clone collapses again)
- Timestamp: 2026-05-27
- What changed: Elite=X23-g015 clone (HoF), Mut2=mutated Elite-g015(2nd), X12/X23=crossovers
- Result: X23-g016=31 (7W/2L 2p, 3W 4p, top2=4), Mut2=22, X12=18, Elite=7 (0W/9L 2p!)
- Insight: HoF clone gets 0W/9L in 2p — extreme seed variance at play. X23 operator again best (31), X12 reasonable (18). Pattern: X23 crossover(2nd,3rd) consistently produces competitive agents even when Elite collapses. stagnation_count=1.
- Next: Gen 017 — HoF (X23-g015) as Elite; stagnation_count=1

### Run 17: Gen 017 tournament — best_fitness=22 (DISCARD — sharp regression)
- Timestamp: 2026-05-27
- What changed: Elite=HoF X23-g015 clone, Mut2=mutated X23-g016(1st), X12/X23=crossovers of gen016 top agents
- Result: X12-g017=X23-g017=22 (tied), Elite=18, Mut2=16
- Insight: Sharp regression to 22 — worst since gens 003-005. HoF clone mediocre again (18). All agents compressed in 16-22 range. Tournament variance likely — different seed than gen 015's 34. stagnation_count=2, one more discard before wide-explore fires.
- Next: Gen 018 — HoF as Elite; stagnation_count=2

### Run 18: Gen 018 tournament — best_fitness=24 (DISCARD — stagnation_count=3, wide-explore fires!)
- Timestamp: 2026-05-27
- What changed: Elite=HoF X23-g015 clone, Mut2=mutated X12-g017(1st), X12/X23=crossovers
- Result: Mut2-g018=24 (5W/4L 2p, 2W 4p, top2=5), X12-g018=21 (7W/2L but 0 4p), X23=18, Elite=15
- Insight: Persistent plateau at 15-24. HoF clone collapses in 2p (3W/6L). X12 shows classic 2p-only specialization again (7W/2L but 0 4p). stagnation_count=3 triggers wide-explore for gen 019 — last time (gen 013) it produced X23-g013=28 which led to the gen 014-015 breakthrough chain.
- Next: Gen 019 — wide-explore (σ=0.30) of Mut2-g018 (best this gen)

### Run 19: Gen 019 tournament — best_fitness=31 (DISCARD — wide-explore plateau)
- Timestamp: 2026-05-27
- What changed: Elite=wide-explore(σ=0.30) of Mut2-g018; Mut2/X12/X23=crossovers of gen018 agents
- Result: X23-g019=31 (8W/1L 2p, 2W 4p, top2=3), Elite=21, Mut2=16, X12=10
- Insight: X23 continues to be the best operator — 8W/1L 2p but only 2 4p wins. Wide-explore Elite (21) is mediocre. Persistent plateau: gens 016-019 all 22-31, can't reach 34. stagnation_count=4, wide-explore fires again for gen 020 off X23-g019.
- Next: Gen 020 — wide-explore(σ=0.30) of X23-g019 (8W/1L 2p agent)

### Run 20: Gen 020 tournament — best_fitness=22 (DISCARD — 2p/4p tradeoff maximally visible)
- Timestamp: 2026-05-27
- What changed: Elite=wide-explore(σ=0.30) of X23-g019; Mut2/X12/X23=crossovers
- Result: Elite=22 (2W/7L 2p but 5W 4p — best 4p record ever!), Mut2=22 (6W/3L 2p, 1W 4p), X12=X23=17
- Insight: Perfect illustration of 2p/4p tradeoff — Elite has 5W/6 in 4p (best ever) but only 2W/9 in 2p. Mut2 is the opposite: 6W in 2p but 1W in 4p. Neither is the combined champion. The 2p formula awards 3pts/win vs 2pts/4p-win — to beat HoF=34, we need ≥6W 2p + ≥4W 4p + high top2. stagnation_count=5 — wide-explore fires again.
- Next: Gen 021 — wide-explore(σ=0.30) of Elite-g020 (4p specialist); try to combine 4p strength with better 2p

### Run 21: Gen 021 tournament — best_fitness=30 (DISCARD — closest to HoF since gen 016)
- Timestamp: 2026-05-27
- What changed: Elite=wide-explore(σ=0.30) of Elite-g020 (4p specialist); Mut2/X12/X23=crossovers of gen020 agents
- Result: Elite-g021=30 (6W/3L 2p, 3W 4p, top2=6), X12-g021=25, Mut2=13, X23=10
- Insight: Elite-g021 is best in gens 016-021. Config: ORBIT_BONUS=22.67 (very high), TIME_PROD_MULT=0.18 (ultra-low), SHIPS_MULT=0.094 (lowest ever), DIST_MULT=0.25. Different from PROXIMITY_DIST=44 archetype — this is an orbit-bonus focused grabber. stagnation_count=6, wide-explore fires again for gen 022.
- Next: Gen 022 — wide-explore(σ=0.30) of Elite-g021 (orbit+low-ships archetype, fitness=30)

### Run 22: Gen 022 tournament — best_fitness=27 (DISCARD — below HoF 34)
- Timestamp: 2026-05-27
- What changed: Elite=wide-explore(σ=0.30) of Elite-g021; Mut2/X12/X23=crossovers; stagnation_count=7
- Result: X12-g022=27 (6W/3L 2p, 2W 4p, top2=3); Elite=24; Mut2=22; X23=18
- Insight: Wide-explore keeps producing 27-31 range but cannot break 34. X12 crossover consistently finds decent combined configs but lacks the breakthrough. HoF=34 remains distant. stagnation_count now=7.
- Next: Gen 023 — stagnation_count=8 fires wide-explore again; use X12-g022 top seed.

### Run 23: Gen 023 tournament — best_fitness=30 (DISCARD — below HoF 34)
- Timestamp: 2026-05-27
- What changed: Elite=wide-explore(σ=0.30) of X12-g022; Mut2/X12/X23=crossovers; stagnation_count=8
- Result: X23-g023=30 (6W/3L 2p, 4W 4p, top2=4); X12=25; Mut2=12; Elite=11
- Insight: X23 crossover continues to be most consistent performer (30+) but Elite with wide-explore sigma now collapses (11 fitness). The wide-explore mutation is generating Elite configs that lose everything. X23 picks up good combination from parent mix. Two archetypes diverging: X12 finds 2p specialists, X23 finds 4p specialists. stagnation_count=8.
- Next: Gen 024 — continue wide-explore of X23-g023; explore if any config can break 34.

### Run 24: Gen 024 tournament — best_fitness=24 (DISCARD — below HoF 34)
- Timestamp: 2026-05-27
- What changed: Elite=wide-explore(σ=0.30) of X23-g023; Mut2=mutated X12-g023; X12/X23=crossovers; stagnation_count=9
- Result: X23-g024=24 (6W/3L 2p, 2W 4p, top2=2); Elite=20 (4W/5L 2p, 1W 4p, top2=6!); X12=19; Mut2=15
- Insight: Elite-g024 has notable survival (top2=6/6 games, losses=0 in 4p) but only 1 win. PROXIMITY_MULT=15.38, PROXIMITY_DIST=50.0, ORBIT_BONUS=19.1 — extreme proximity-first config survives well but doesn't close games. X23 retains 2p dominance but weak 4p this gen. 9 gens of stagnation below HoF=34.
- Next: Gen 025 — wide-explore of X23-g024; if this continues to stagnate consider changing stagnation strategy.

### Run 25: Gen 025 tournament — best_fitness=24 (DISCARD — 10 gens stagnation)
- Timestamp: 2026-05-27
- What changed: Elite=wide-explore(σ=0.30) of X23-g024 (ORBIT_BONUS=34.43, extreme orbit config); X12/X23=crossovers; stagnation_count=10
- Result: X12-g025=24 (7W/2L 2p, 1W 4p, top2=1); Elite=18 (3W/6L 2p, 2W 4p, top2=5); Mut2=18; X23=18
- Insight: Elite with ORBIT_BONUS=34.43 again survives 4p (2W,top2=5) but weak 2p. X12 wins most 2p but loses 4p. No agent integrates both. Modified evolve.py: stagnation≥6 now uses HoF directly in crossovers (X12=HoF×1st, X23=HoF×2nd) to reintroduce winning genome.
- Next: Gen 026 — deep stagnation strategy: X12=Crossover(HoF,best), X23=Crossover(HoF,2nd). X12-g026 has PROD=13.81 (near HoF's 14.82).

### Run 26: Gen 026 tournament — best_fitness=30 (DISCARD — below HoF 34, but best in 5 gens!)
- Timestamp: 2026-05-27
- What changed: Elite=wide-explore(σ=0.30) X12-g025; X12=Crossover(HoF,X12-g025); X23=Crossover(HoF,Elite-g025 survival specialist); stagnation_count=11
- Result: X23-g026=30 (6W/3L 2p, 4W 4p, top2=4); Elite=22; Mut2=17; X12=9
- Insight: HoF×2nd crossover (X23=30) worked well — combining HoF's production strategy with Elite-g025's orbit survival produced a balanced 30. HoF×1st crossover (X12=9) failed — mixing HoF with 2p-specialist X12-g025 destroyed both strengths. X23-g026 config: PROD=11.72, ENEMY=13.17 (high), SHIPS=0.434 (too high vs HoF 0.103), PROX_DIST=50. SHIPS still drifting high from population contamination.
- Next: Gen 027 — deep stagnation continues; X12=Crossover(HoF, X23-g026), X23=Crossover(HoF, Elite-g026). X12 has PROD=13.84. SHIPS still high (0.515-0.565 range).

### Run 27: Gen 027 tournament — best_fitness=28 (DISCARD — HoF crossovers failed again)
- Timestamp: 2026-05-27
- What changed: Elite=wide-explore of X23-g026; X12=HoF×X23-g026; X23=HoF×Elite-g026; stagnation_count=12
- Result: Mut2-g027=28 (7W/2L 2p, 2W 4p, top2=3); Elite=22; X23=16; X12=12
- Insight: HoF crossovers AGAIN failed (X12=12, X23=16). SHIPS_MULT contamination from population (0.515-0.565) makes crossovers inherit wrong values. Mut2 (pure pop mutant) scored 28 — pure population evolution outperforms HoF crossovers. Changed strategy: deep stagnation now injects HoF near-clone (σ=0.05) as X12 to preserve SHIPS/PROD correlation, plus Crossover(HoF,1st) as X23.
- Next: Gen 028 — X12=HoF near-clone (PROD=15.38, SHIPS=0.104, PROX=43.8); X23=Crossover(HoF,Mut2-g027). Tests if HoF reproduces near-34 on new seed.

### Run 28: Gen 028 tournament — best_fitness=30 (DISCARD — seed variance holds HoF back)
- Timestamp: 2026-05-27
- What changed: X12=HoF near-clone (PROD=15.38, SHIPS=0.104, PROX=43.8); X23=HoF×Mut2-g027; stagnation_count=13
- Result: X12-g028=30 (8W/1L 2p!, 2W 4p, top2=2); Mut2-g028=30 tied; Elite=9 (ENEMY=19.88 failed); X23=9
- Insight: HoF near-clone dominates 2p (8/9!) confirming PROD+low-SHIPS is genuinely strong. But 4p limited to 2W — seed variance is the barrier (original HoF got 3-4W 4p). Mut2-g028 (mutated Elite-g027) also 30 with 7W 2p + 3W 4p — more balanced. Gen 029: Mut2 of HoF near-clone got PROD=18.41, SHIPS=0.098 — highest PROD ever! Three agents with SHIPS<0.105.
- Next: Gen 029 — PROD=18.41+SHIPS=0.098 (Mut2), HoF near-clone (X12), HoF×Mut2-g028 (X23). Best gen setup yet for low-SHIPS territory.

### Run 29: Gen 029 tournament — best_fitness=28 (DISCARD — convergent population hurts)
- Timestamp: 2026-05-27
- What changed: 3 low-SHIPS agents (Mut2=0.098, X12=0.102, X23=0.100) + Elite high-SHIPS=0.496; stagnation_count=14
- Result: Elite-g029=28 (6W/3L 2p, 2W 4p, top2=6); X12=21; X23=17; Mut2(PROD=18.41)=12
- Insight: KEY INSIGHT — when 3 agents share the same low-SHIPS archetype, they split wins among themselves and none scores high enough. Diversity is REQUIRED for any single agent to dominate. Elite (SHIPS=0.496) won precisely because it was differentiated. PROD=18.41 with SHIPS=0.098 failed at 12 — too similar to X12 and X23 in strategy. Gen 030 has better diversity: 2 low-SHIPS (Mut2=0.094, X12=0.101) vs 2 high-SHIPS (Elite=0.413, X23=0.417).
- Next: Gen 030 — Mut2(PROD=14.59,SHIPS=0.094,PROX=36.4) vs X12(SHIPS=0.101,PROX=43.1) vs 2 diverse agents.

### Run 30: Gen 030 tournament — best_fitness=28 (DISCARD — 2p/4p split continues)
- Timestamp: 2026-05-27
- What changed: Elite=wide-explore(X23-g029,σ=0.30); Mut2=mutated X12-g029; X12=HoF near-clone; X23=Crossover(HoF,X23-g029); stagnation_count=15
- Result: X23-g030=28 (8W/1L 2p!, 1W 4p, top2=2, PROD=15.50,SHIPS=0.417); X12-g030=21; Mut2=15 (3W 4p!); Elite=14
- Insight: X23(PROD=15.5,SHIPS=0.417) again dominated 2p (8/9!) but only 1 4p win. Mut2(SHIPS=0.094) is 4p specialist (3W 4p!) but only 2W 2p. Pattern: LOW SHIPS → 4p specialist, HIGH SHIPS → 2p specialist. The magic of HoF=34 was achieving BOTH — likely required the specific tournament seed. Gen 031 has X12(PROD=15.91,SHIPS=0.098,PROX=46.9) and X23(PROD=17.10,SHIPS=0.140,PROX=44.0,ENEMY=6.73). X23-g031 nearly matches HoF's PROX/ENEMY with higher PROD.
- Next: Gen 031 — most promising setup in gens: X23 has PROD=17.10, SHIPS=0.140, PROX=44.0, ENEMY=6.73 (HoF had 6.72).

### Run 31: Gen 031 tournament — best_fitness=24 (DISCARD — near-HoF params still fail)
- Timestamp: 2026-05-27
- What changed: Elite=wide-explore(X23-g030,σ=0.30); X23(PROD=17.10,SHIPS=0.140,PROX=44.0,ENEMY=6.73); stagnation_count=16
- Result: X23-g031=24 (6W/3L 2p, 2W 4p); Elite=20; X12=18; Mut2(SHIPS=0.102)=16 (3W 4p!)
- Insight: X23-g031 matched HoF's ENEMY=6.72 and PROX=44.49 almost exactly (6.73/44.0) with higher PROD but still only 24. The 2p/4p split is NOT about ENEMY or PROX — it's about SHIPS_MULT. Low SHIPS → 4p specialist (3W 4p, 2W 2p). Moderate SHIPS (0.14) → still 2p specialist (6W 2p, 2W 4p). Need SHIPS≈0.103 + specific tournament seed to achieve both. Gen 032: X12 near-perfect HoF clone (PROD=14.89,SHIPS=0.104,PROX=44.1,OVER=0.256). Elite PROD=20.69 (record!) but OVER=0.798 (kills aggression).
- Next: Gen 032 — X12 is most faithful HoF clone yet; Elite PROD=20.69 is interesting outlier.

### Run 32: Gen 032 tournament — best_fitness=28 (DISCARD — PROD=20.69 dominates HoF near-clone)
- Timestamp: 2026-05-27
- What changed: Elite=wide-explore(X23-g031,σ=0.30)→PROD=20.69,SHIPS=0.157,OVER=0.798; X12=HoF near-clone(PROD=14.89,SHIPS=0.104,OVER=0.256); stagnation_count=17
- Result: Elite-g032=28 (7W/2L 2p, 1W 4p, top2=5); Mut2=22 (3W 4p); X23=19; X12=9 (crushed!)
- Insight: CRITICAL — PROD=20.69 agent CRUSHED HoF near-clone (14.89) from 30 → 9. When PROD=20.69 enters, it outcompetes HoF's PROD=14.82. So the HoF=34 relied on NOT facing such a high-PROD opponent. The landscape: PROD=20+ beats PROD=14-15; but PROD=20+ with OVER=0.798 still only scores 28 due to 4p conservatism. Gen 033: Elite escalates to PROD=26.88 (record!), OVER=0.967 (hyper-conservative), PROX=27.6 (short-range only). X23=PROD=21.98+ENEMY=13.73.
- Next: Gen 033 — exploring very-high-PROD territory (26.88, 21.98) with mixed OVER values; X12 again HoF near-clone.

### Run 33: Gen 033 tournament — best_fitness=25 (DISCARD — PROX killed high-PROD agents)
- Timestamp: 2026-05-27
- What changed: Elite=wide-explore(Elite-g032)→PROD=26.88,PROX=27.6,OVER=0.967; X12=HoF near-clone(PROD=14.69,PROX=43.5); stagnation_count=18
- Result: X12-g033=25 (7W/2L 2p, 1W 4p); X23=21; Elite=16 (PROX=27.6 crippled it); Mut2=16
- Insight: CRITICAL — PROX_DIST is a key constraint: Elite(PROD=26.88) failed because PROX=27.6 (too short-range). HoF near-clone won despite PROD=14.69 < 21.98 because PROX=43.5 vs 41.1. On a 100x100 board, agents that can "see" far (PROX=44-50) evaluate more planets and find better targets. PROX<35 is seriously limiting. The HoF's PROX=44.49 is a core requirement, not just PROD. Gen 034: X12=PROD=15.62,SHIPS=0.100,PROX=45.2; Elite=PROD=23.36,SHIPS=0.116,PROX=34.3(lower).
- Next: Gen 034 — test if Elite(PROD=23.36,PROX=34.3) works better than PROX=27.6 case.

### Run 34: Gen 034 tournament — best_fitness=22 (DISCARD — PROX<35 pattern repeats)
- Timestamp: 2026-05-27
- What changed: X12=HoF near-clone(PROD=15.62,SHIPS=0.100,PROX=45.2); Elite=PROD=23.36,PROX=34.3; stagnation_count=19
- Result: Mut2=X23=22 tied; X12=18; Elite=16 (PROX=34.3 again hurt)
- Insight: Confirmed: PROX<35 consistently underperforms. Wide-explore keeps generating PROX=27-35 because lognormal mutation of PROX=43-45 with σ=0.30 can hit PROX≈32 (exp(-0.30)≈0.74 → 45*0.74=33). Fixed evolve.py to clamp PROX≥40 in wide-explore. Gen 035: Elite clamped to PROX=40.0; X12=near-perfect HoF match (PROD=14.76,SHIPS=0.095,ENEMY=6.72,OVER=0.262 — exact HoF values!).
- Next: Gen 035 — PROX clamp active; X12 best HoF replica yet with exact ENEMY+OVER match.

### Run 35: Gen 035 tournament — best_fitness=28 (DISCARD — perfect 2p, zero 4p)
- Timestamp: 2026-05-27
- What changed: Elite=PROX clamped to 40.0; X12=near-perfect HoF clone(PROD=14.76,SHIPS=0.095,ENEMY=6.72,OVER=0.262); stagnation_count=20
- Result: X12-g035=28 (9W/0L 2p PERFECT, 0W 4p top2=1); Mut2=24 (3W 4p); Elite=11 (ENEMY=13 killed 2p); X23=15
- Insight: X12 went 9/9 in 2p — the most dominant 2p result ever. ENEMY=6.72 (HoF-exact) + SHIPS=0.095 (lower than HoF 0.103) = pure 2p dominance. But 0/6 4p wins — extreme 2p specialist pathology. With multiple enemies in 4p, ENEMY=6.72 (low) means the agent doesn't prioritize taking enemy planets fast enough. The 4p dynamic requires some ENEMY aggression to eliminate opponents. Gen 036: Elite SHIPS=0.057 (lowest ever!); X12=PROD=16.65,SHIPS=0.101,PROX=43.6.
- Next: Gen 036 — all agents PROX≥39.8; Elite SHIPS=0.057 explores sub-0.1 territory.

### Run 36: Gen 036 tournament — best_fitness=26 (DISCARD — complementary split discovered)
- Timestamp: 2026-05-27
- What changed: Elite=SHIPS=0.057 (ultra-low); X12=HoF near-clone; stagnation_count=21
- Result: Elite=26 (7W/2L 2p, 0W 4p, SHIPS=0.057); Mut2=24 (3W 2p, 5W 4p RECORD!, ENEMY=16.21); X12=16; X23=12
- Insight: MAJOR DISCOVERY — if you combine Elite's 2p dominance + Mut2's 4p dominance: 7×3 + 5×2 + 5 = 36 > HoF=34! The two complementary archetypes TOGETHER would beat HoF. Modified evolve.py: X23 now = Crossover(1st,2nd) directly to fuse the archetypes. Gen 037: X23-g037 fusion has SHIPS=0.062 (from 0.057) + ENEMY=10.15 (between 8.52 and 16.21) + PROX=40.7. Elite SHIPS=0.044 (lowest ever).
- Next: Gen 037 — archetype fusion: X23(SHIPS=0.062,ENEMY=10.15) directly tries to combine 2p+4p strengths.

### Run 37: Gen 037 tournament — best_fitness=29 (DISCARD — fusion collapsed, ENEMY must stay low)
- Timestamp: 2026-05-27
- What changed: Elite=wide-explore(Elite-g036,σ=0.30,PROX≥40)→SHIPS=0.044,PROD=18.08; X12=HoF near-clone(σ=0.05); X23=Crossover(Elite-g036,Mut2-g036)→SHIPS=0.062,ENEMY=10.15,PROX=40.7; stagnation=22
- Result: Elite-g037=29 (7W/2L 2p, 1W 4p, top2=6); X12-g037=24 (7W/2L 2p, 1W 4p, top2=1); Mut2=19; X23=6 (0W/9L 2p, 2W 4p)
- Insight: FUSION FAILED — X23(SHIPS=0.062,ENEMY=10.15) collapsed 0W/9L 2p. ENEMY=10.15 too high: in 1v1 it attacks enemy planets instead of expanding neutrals, losing production race. Critical constraint: ENEMY must stay ≤7 for 2p viability. Two 7W/2L agents (Elite+X12) coexisted by sharing wins from weak opponents. Gen 038: X23 fuses Elite-g037(7W/2L) × X12-g037(7W/2L) — both ENEMY-safe (8.58 vs 6.28 blend ≈7).
- Next: Gen 038 — more aligned fusion parents; ENEMY will blend to ~7 (safe zone).

### Run 38: Gen 038 tournament — best_fitness=28 (DISCARD — PROX<40 kills X23 again)
- Timestamp: 2026-05-27
- What changed: Elite=wide-explore(Elite-g037,σ=0.30)→ORBIT=31.93,SHIPS=0.042; X12=HoF near-clone(σ=0.05,PROX=49.61); Mut2=mutate(X12-g037,σ=0.12); X23=Crossover(Elite-g037,X12-g037)→SHIPS=0.034,ENEMY=4.01,PROX=37.24; stagnation=23
- Result: Elite-g038=28 (6W/3L 2p, 2W 4p, top2=6 PERFECT 4p survival); X12-g038=22 (6W/3L 2p, 1W 4p); Mut2-g038=21 (6W/3L 2p, 1W 4p); X23-g038=7 (0W/9L 2p, 2W 4p)
- Insight: X23 PROX=37.24 (below 40) → 0W/9L 2p again. The PROX≥40 clamp on wide-explore didn't help X23 from crossover drift below 40. Need to clamp PROX≥40 in crossover too when deep stagnation. Elite's ORBIT_BONUS=31.93 (record!) gave 2W 4p + top2=6 — maximum 4p survival. X12(PROX=49.61) showed PROX can go above HoF's 44.49 safely. Gen 039: X23 will inherit from Elite(ORBIT=31.93,PROX=40) × X12(PROX=49.61) — need to clamp PROX≥40 in crossover too.
- Next: Gen 039 — consider clamping PROX≥40 in crossover for deep stagnation to prevent PROX drift.











### Run 41: Gen 026 — best_fitness=27 (DISCARD — HoF near-clone collapsed, Elite dipped to 6W/3L)
- Timestamp: 2026-05-27
- What changed: Elite=wide-explore(Elite-g025,σ=0.30,PROX=50,ENEMY=8.35,PROD=15.27,ORBIT=23.37); X12=HoF near-clone(σ=0.05,PROX=44.57,ENEMY=7.14,PROD=15.85,SHIPS=0.100); Mut2=mutate(X23-g025,σ=0.12,PROX=46.47,ENEMY=9.36); X23=crossover(Elite-g025,X23-g025); stagnation=11
- Result: Elite-g026=27 (6W/3L 2p, 2W 4p, top2=5); Mut2-g026=19 (5W/4L 2p, 1W 4p); X23-g026=18; X12-g026=14 (2W/7L 2p, 2W 4p)
- Insight: Elite dropped from 7W/2L (gen 025) to 6W/3L. The difference vs gen 024 Elite (fitness=33): ENEMY went from 13.94→8.35 and PROD went from 11.83→15.27. One of these changes hurt 2p. HoF near-clone (X12) again collapsed in 2p despite close ENEMY=7.14 to HoF's 6.72 — X12's SHIPS=0.100 matched HoF but PROD=15.85 slightly higher. INSIGHT: Elite-g024's key feature might be LOW PROD (11.83) not high — agents with PROD≥15 seem to over-invest in distant planets. NEW INSIGHT: High ENEMY (13.94) with PROX=50 may HELP 2p by prioritizing aggressive expansion (enemies count for urgency) while PROX=50 ensures neutrals are always available.
- Next: Gen 027 — deep stagnation=11, wide-explore Elite-g026 (PROX clamped at 40); X12=HoF near-clone; X23=crossover(Elite-g026,Mut2-g026). Watch if lower PROD in Elite helps.

### Run 40: Gen 025 — best_fitness=26 (DISCARD — Elite lost 4p edge vs gen024)
- Timestamp: 2026-05-27
- What changed: Elite=wide-explore(Elite-g024,σ=0.30,PROX=45.41,ORBIT=20.83,PROD=9.14); Mut2=mutate(X23-g024,σ=0.12,PROX=48.80); X12=HoF near-clone(σ=0.05,PROX=45.90,PROD=14.24); X23=crossover(Elite-g024,X23-g024,σ=0.18,PROX=48.81); deep stagnation=10
- Result: Elite-g025=26 (7W/2L 2p, 0W 4p, top2=5); X23-g025=19 (5W/4L 2p, 1W 4p); X12-g025=18 (3W/6L 2p, 3W 4p); Mut2-g025=15 (3W/6L 2p, 2W 4p)
- Insight: Elite's 2p performance held (7W/2L) but lost all 4p wins (vs gen024 Elite: 3W 4p). Random seed variance likely — different 4p opponents. X12-g025 (HoF near-clone) got 3W 4p but only 3W/6L 2p, confirming low-SHIPS-MULT archetype is 4p-only. Wide-explore of pop-1 → pop-1 mutation chain is losing variance — each gen just clones Elite with noise.
- Next: Gen 026 — evolve.py creating fresh wide-explore of Elite-g025 + Mut2 from X23-g025 + HoF near-clone + crossover(Elite,X23).

### Run 39: Gen 024 (new session re-run) — best_fitness=33 (DISCARD — 1 below HoF=34!)
- Timestamp: 2026-05-27
- What changed: Elite=wide-explore(σ=0.30) of Mut2-g023 (PROX=50.0,ORBIT=18.24,PROD=11.83); Mut2=mutated X23-g023; X12=HoF near-clone(σ=0.05); X23=Crossover(Mut2-g023,X23-g023); deep stagnation=9
- Result: Elite-g024=33 (7W/2L 2p, 3W 4p, top2=6); X23-g024=25; Mut2-g024=14; X12-g024=6 (HoF near-clone COLLAPSED again)
- Insight: CLOSEST TO HOF SINCE GEN 015! Elite achieved 33 via wide-explore with PROX=50.0 — full-board proximity. The 3W 4p + top2=6 shows excellent survival. HoF near-clone (X12) again failed (6) — PROD=14.5,SHIPS=0.1 archetype cannot reproduce HoF=34 reliably on different seeds. Wide-explore of pop best continues to be the most productive operator.
- Next: Gen 025 — deep stagnation continues; Elite=wide-explore of Elite-g024 (PROX=50.0 strategy), HoF near-clone attempted again, crossovers from g024 top-2.

### Run 42: Gen 027 — best_fitness=25 (DISCARD — Elite PROX=40 crippled, X23 best)
- Timestamp: 2026-05-27
- What changed: Elite-g027=wide-explore(Elite-g026,σ=0.30,PROX clamped to 40.0,ENEMY=8.52,SHIPS=0.043,PROD=9.89,ORBIT=28.25); Mut2=mutate(Mut2-g026,σ=0.12,PROX=50); X12=HoF near-clone(σ=0.05,PROX=45.09,ENEMY=6.89); X23=Crossover(Elite-g026,Mut2-g026,PROX=44.59,ENEMY=7.99); stagnation=12
- Result: X23-g027=25 (6W/3L 2p, 2W 4p, top2=3); Elite-g027=21 (3W/6L 2p, 3W 4p); Mut2-g027=19 (6W/3L 2p, 0W 4p); X12-g027=13 (3W/6L 2p, 1W 4p)
- Insight: Elite-g027 PROX=40.0 (exactly at old clamp minimum) suffered severe 2p penalty (3W/6L) — confirms PROX≥44 needed, PROX=40 is a danger zone. X23 won with PROX=44.59 + ENEMY=7.99 (moderate). evolve.py already updated to clamp PROX≥44 for future gens. HoF near-clone (X12) again fails in 2p (3W/6L) despite ENEMY=6.89 matching HoF archetype — confirms that HoF config doesn't reproduce on different seeds. Ultra-stagnation strategy (HoF×vice-HoF crossover) will activate at stagnation=15 (gen 030).
- Next: Gen 028 — stagnation=12, deep stagnation strategy continues; all configs PROX≥44; X23 winner becomes Mut2 parent; vice-HoF crossover 3 gens away.

### Run 43: Gen 028 — best_fitness=26 (DISCARD — Elite PROX=44 baseline, X23 ENEMY=9 weak)
- Timestamp: 2026-05-27
- What changed: Elite-g028=wide-explore(X23-g027,σ=0.30,PROX=44.0,ENEMY=7.59,SHIPS=0.065,PROD=14.95); Mut2=mutate(Elite-g027,σ=0.12,PROX=44.0,SHIPS=0.044); X12=HoF near-clone(σ=0.05,PROX=44.0,ENEMY=6.55,SHIPS=0.100); X23=Crossover(X23-g027,Elite-g027,PROX=44.0,ENEMY=9.23); stagnation=13
- Result: Elite-g028=26 (6W/3L 2p, 1W 4p, top2=6); X12-g028=19 (4W/5L 2p, 2W 4p); Mut2-g028=18 (4W/5L 2p); X23-g028=15 (4W/5L 2p, ENEMY=9.23)
- Insight: PROX=44.0 (minimum clamp) is less robust than PROX=48-50 — agents with PROX just at 44 tend to have mediocre 2p results. X23 at ENEMY=9.23+PROX=44 confirmed weak (4W/5L 2p) — consistent with ENEMY>8 needing PROX≥48. HoF near-clone X12 again 4W/5L — cannot reproduce HoF=34. Elite's top2=6 (perfect 4p survival) with only 1W 4p shows high ORBIT_BONUS (13.1) = good survival but not aggressive enough. Ultra-stagnation strategy activates next gen (stagnation=14, threshold=15 → gen 030 if gen 029 also fails).
- Next: Gen 029 — stagnation=13, deep stagnation strategy; Ultra-stagnation at stagnation=15 triggers gen 030 if needed.

### Run 44: Gen 029 — best_fitness=36 *** NEW HOF! *** (KEEP — beats old HoF=34!)
- Timestamp: 2026-05-27
- What changed: Elite-g029=wide-explore(Elite-g028,σ=0.30,PROX=50.0,ENEMY=6.81,SHIPS=0.087,PROD=15.97,ORBIT=27.75,COMPOUND=17.73,PROX_MULT=25.0(capped)); stagnation=13 (deep stagnation → wide-explore + HoF injection)
- Result: Elite-g029=36 PERFECT 9W/0L 2p + 2W 4p top2=5; X23-g029=19 (3W/6L 2p, 3W 4p); Mut2-g029=12; X12-g029=11
- Insight: NEW HOF (36 > old 34)! PERFECT 2p record — never seen before in this session. Key config: PROX=50.0 (max board reach), ENEMY=6.81 (low — no wasted attacks on enemy planets), ORBIT=27.75 (HIGH — prizes orbiting planets), SHIPS=0.087 (moderate expansion rate), PROD=15.97 (moderate-high). COMPOUND_MULT=17.73 (HIGH) may help coordinate multi-fleet attacks. Wide-explore of Elite-g028 with σ=0.30 jumped PROX from 44→50, ORBIT from 13.1→27.75. The ORBIT jump is significant — orbiting planets generate continuous production, and high ORBIT_BONUS incentivizes capturing them. Stagnation counter resets to 0. Vice-HoF archetype (gen 024, ENEMY=13.94) still held for comparison.
- Next: Gen 030 — stagnation=0 (new HoF!); Elite=Elite-g029 (fitness=36); standard evolution: Mut2+crossovers from gen 029 top agents. Goal: push past 36.

### Run 45: Gen 030 — best_fitness=32 (DISCARD — HoF clone dropped, seed variance)
- Timestamp: 2026-05-27
- What changed: Elite-g030=exact clone of Elite-g029 (HoF: PROX=50,ENEMY=6.81,ORBIT=27.75,SHIPS=0.087,PROD=15.97); Mut2=mutate(X23-g029,σ=0.12,PROX=44.0); X12=Crossover(Elite-g029,X23-g029,σ=0.18,PROX=44.0 clamped,PROD=23.7); X23=Crossover(X23-g029,Mut2-g029,σ=0.18,PROX=50); stagnation=1
- Result: Elite-g030=32 (8W/1L 2p, 1W 4p, top2=6); X23-g030=21 (6W/3L 2p, 1W 4p); Mut2-g030=18 (4W/5L 2p, 2W 4p); X12-g030=7 (0W/9L 2p, 2W 4p, PROD=23.7)
- Insight: SEED VARIANCE CONFIRMED — exact HoF clone went 8W/1L (32) instead of 9W/0L (36). The fitness=36 in gen 029 may partly be seed luck. X12 with PROD=23.7 went 0W/9L in 2p — confirms PROD>20 is harmful (over-invests in distant planets). Clamped PROX fix worked: X12 at PROX=44 didn't collapse from PROX alone, just PROD issue. X23 (PROX=50) second-best at 21.
- Next: Gen 031 — HoF still elite (36>32, stagnation=1); standard evolution from gen 030 top-2; try to find configs that replicate or exceed HoF=36 on different seeds.

### Run 46: Gen 031 — best_fitness=29 (DISCARD — HoF clone variance, stagnation=2)
- Timestamp: 2026-05-27
- What changed: Elite-g031=exact HoF clone (PROX=50,ENEMY=6.81,ORBIT=27.75); Mut2=mutate(X23-g030,σ=0.12,PROX=45.9,SHIPS=0.110); X12=Crossover(HoF,X23-g030,PROX=44.27,ORBIT=21.67); X23=Crossover(X23-g030,Mut2-g030,PROX=50,ENEMY=5.10)
- Result: Elite-g031=29 (7W/2L 2p, 1W 4p, top2=6); X12-g031=21 (6W/3L 2p, 1W 4p); Mut2-g031=19 (3W/6L 2p, 3W 4p); X23-g031=9 (2W/7L 2p)
- Insight: HoF config variance across seeds: 36 (gen029), 32 (gen030), 29 (gen031). Average ~32, peak=36. Seed variance is ~±4pts. X23 with ENEMY=5.10 (very low) + PROX=50 went 2W/7L 2p — confirms ENEMY can't be TOO low either; range 6-8 seems optimal. Mut2 got 3W 4p (strong 4p) with SHIPS=0.110 + PROX=45.9. Stagnation=2, one more gen before wide-explore kicks in. Key: need to find configs that reliably outperform HoF=36 on average, not just lucky seeds.
- Next: Gen 032 — stagnation=2, at limit; if gen 032 fails, stagnation=3 triggers wide-explore of current best instead of HoF clone.

### Run 47: Gen 032 — best_fitness=27 (DISCARD — HoF clone declining, ORBIT=35 failed)
- Timestamp: 2026-05-27
- What changed: Elite-g032=exact HoF clone (4th test); Mut2=mutate(X12-g031,σ=0.12,PROX=50,ORBIT=25.73); X12=Crossover(HoF,X12-g031,ORBIT=35.0 MAX,PROX=50); X23=Crossover(X12-g031,Mut2-g031,PROX=44,ENEMY=5.37)
- Result: Elite-g032=27 (7W/2L 2p, 0W 4p, top2=6); X23-g032=24 (5W/4L 2p, 3W 4p); Mut2-g032=18; X12-g032=9 (2W/7L 2p, ORBIT=35)
- Insight: ORBIT=35.0 (max bound) confirmed HARMFUL — 2W/7L 2p. Agent over-values orbiting planets, ignores closer expansion targets. Optimal ORBIT range appears to be 20-30 (HoF=27.75). HoF clone variance: 36/32/29/27 across 4 seeds — mean ~31, so true fitness is ~31, peak of 36 was partly luck. stagnation=3 hits HOF_STAGNATION_LIMIT → gen 033 uses wide-explore (σ=0.30) of current best instead of bare HoF clone. X23 (ENEMY=5.37+PROX=44) got decent 3W 4p — very low ENEMY works for 4p.
- Next: Gen 033 — stagnation=3, HOF_STAGNATION_LIMIT reached; wide-explore σ=0.30 of HoF config; crossovers from gen 032 top-2 (Elite+X23).

### Run 48: Gen 033 — best_fitness=22 (DISCARD — wide-explore drifted, ORBIT 20-27 sweet spot)
- Timestamp: 2026-05-27
- What changed: Elite-g033=wide-explore(Elite-g032,σ=0.30,ORBIT=34.28,PROD=22.94); Mut2=mutate(X23-g032,PROX=44,ENEMY=5.15,PROD=11.85,ORBIT=22.44); X12=Crossover(Elite-g032,X23-g032,PROX=50,ORBIT=24.27); X23=Crossover(X23-g032,Mut2-g032,PROX=44.85,ENEMY=7.74); stagnation=4
- Result: X12-g033=22 (7W/2L 2p, PROX=50,ORBIT=24); Elite-g033=21 (4W/5L 2p, ORBIT=34,PROD=23); X23-g033=19 (6W/3L 2p); Mut2-g033=16 (1W/8L 2p, 4W 4p RECORD tie!)
- Insight: ORBIT sweet spot confirmed at 20-27 — ORBIT=24 (X12) beat ORBIT=34 (Elite) in 2p. PROD=22.94 too high (over-invests in far planets). Mut2 (PROD=11.85,ENEMY=5.15) got 4W 4p — tied the 4p record! Low ENEMY+PROD in 4p means less aggression = better survival. Wide-explore from σ=0.30 drifted to extremes (ORBIT>30, PROD>20) that hurt performance. Gen 034 will wide-explore X12-g033 (the best recent config, ORBIT=24, closer to HoF=27).
- Next: Gen 034 — stagnation=4; wide-explore (σ=0.30) of X12-g033 (PROX=50, ORBIT=24, ENEMY=6.47); crossovers from Elite+X23 of gen 033.

### Run 49: Gen 034 — best_fitness=24 (DISCARD — SHIPS=0.216 toxic, X23 winner at 24)
- Timestamp: 2026-05-27
- What changed: Elite-g034=wide-explore(X12-g033,σ=0.30,SHIPS=0.216,PROD=25.45,ORBIT=30.87); Mut2=mutate(Elite-g033,PROX=44,ORBIT=33.77); X12=Crossover(wide,Elite-g033,ORBIT=35,ENEMY=5.0); X23=Crossover(Elite-g033,X23-g033,PROX=50,ENEMY=6.49,PROD=11.73,ORBIT=16.12); stagnation=5
- Result: X23-g034=24 (6W/3L 2p, 2W 4p); X12-g034=21 (6W/3L 2p, 1W 4p); Mut2-g034=18 (3W 4p); Elite-g034=15 (3W/6L 2p — worst despite being wide-explore!)
- Insight: SHIPS>0.15 is CONFIRMED HARMFUL — Elite with SHIPS=0.216 went 3W/6L 2p. The agent waits too long before attacking (needs too many ships). SHIPS range 0.04-0.11 is the only working zone; 0.087 (HoF) is optimal. X23 won with low PROD=11.73 + moderate ORBIT=16 — gen 034 confirms that PROD and ORBIT don't need to be high; the key is PROX=50 + ENEMY<7. Wide-explore consistently drifts to SHIPS>0.15 and PROD>20 — both are dangerous zones. Deep stagnation triggers at stagnation=6 (gen 035 if it fails), bringing HoF near-clone injection.
- Next: Gen 035 — stagnation=5; wide-explore (σ=0.30) of X23-g034; crossovers from X23+X12 of gen 034. Deep stagnation strategy at stagnation=6 (gen 036).

### Run 50: Gen 035 — best_fitness=28 (DISCARD — deep stagnation at 6; ORBIT=35 worked?!)
- Timestamp: 2026-05-27
- What changed: Elite=wide-explore(X23-g034,σ=0.30,PROX=44,ENEMY=8.33,PROD=8.47,ORBIT=13.71); Mut2=mutate(X12-g034,PROX=50,ENEMY=4.67,SHIPS=0.173,PROD=22.82,ORBIT=35.0); X12=Crossover(wide,X12-g034,PROX=50,ENEMY=11.69,SHIPS=0.141,ORBIT=35.0); X23=Crossover(X12-g034,Mut2-g034,PROX=44,ENEMY=5.47,PROD=24.08); stagnation=6
- Result: Mut2-g035=28 (7W/2L 2p, 2W 4p, top2=3); X12-g035=25 (6W/3L 2p, 2W 4p); Elite-g035=16 (2W/7L 2p); X23-g035=9 (PROD=24 catastrophic)
- Insight: ORBIT=35 REVISITED — both Mut2 and X12 had ORBIT=35 and performed well! Previous failures with ORBIT=35 may have been due to other bad params (PROX<45, PROD>20). With PROX=50: ORBIT=35 seems survivable or even good. SHIPS=0.173 (Mut2) worked with low ENEMY=4.67 — challenging the SHIPS<0.15 hypothesis. KEY INSIGHT: ENEMY is the dominant factor; when ENEMY is very low (4.67), more ships can be sent (higher SHIPS) without the agent collapsing. X12 (ENEMY=11.69) replicated vice-HoF archetype success. Elite (PROX=44, ENEMY=8.33) failed — confirms ENEMY>7 needs PROX≥48. Deep stagnation gen 036: Crossover(Mut2-g035 × X12-g035) blends ENEMY 4.67 × 11.69 → ~7-8, ORBIT=35 preserved.
- Next: Gen 036 — deep stagnation; wide-explore Elite + HoF near-clone (σ=0.05) + Crossover(Mut2-g035,X12-g035) — fusing low-ENEMY×high-ENEMY archetypes with ORBIT=35.

### Run 51: Gen 036 — best_fitness=25 (DISCARD — deep stagnation continues at 7; HoF near-clone best)
- Timestamp: 2026-05-27
- What changed: Elite-g036=wide-explore(Mut2-g035,σ=0.30,PROD=25.82,SHIPS=0.238,ORBIT=35,ENEMY=6.57,PROX=50); Mut2-g036=mutate(X12-g035,σ=0.12,ENEMY=11.70,ORBIT=33.08,PROX=49.4); X12-g036=HoF near-clone(σ=0.05,ENEMY=7.46,ORBIT=29.43,SHIPS=0.083,PROX=49.1); X23-g036=Crossover(Mut2-g035,X12-g035,ENEMY=4.39,ORBIT=35,SHIPS=0.145,PROX=50)
- Result: X12-g036=25 (8W/1L 2p, 0W 4p, top2=1); Elite-g036=19 (3W/6L 2p, 2W 4p); X23-g036=18 (4W/5L 2p, 2W 4p); Mut2-g036=16 (3W/6L 2p, 2W 4p)
- Insight: HoF near-clone (X12-g036) won 2p with 8W/1L but scored 0 4p wins → pure 2p specialist, not good enough overall. SHIPS=0.238 (Elite) confirmed toxic: 3W/6L 2p. ENEMY=4.39 (X23) — too aggressive in 2p? 4W/5L suggests ENEMY too low is bad (doesn't attack enough?). Wide-explore keeps generating SHIPS>0.15 and PROD>20 — danger zone. 7 consecutive gens below HoF=36; stagnation stays deep.
- Next: Gen 037 — stagnation=7 (still deep); wide-explore Elite + HoF near-clone (X12) + Crossover(1st,2nd) archetype fusion. Consider tightening SHIPS upper bound in evolve.py.

### Run 52: Gen 037 — best_fitness=30 (DISCARD — stagnation=8; X23 surprise winner, SHIPS=0.186 worked)
- Timestamp: 2026-05-27
- What changed: Elite-g037=wide-explore(X12-g036,sigma=0.30,PROD=6.01,ENEMY=9.07,ORBIT=27.96,SHIPS=0.063,PROX=50); Mut2-g037=mutate(Elite-g036,PROD=25.34,SHIPS=0.241,ORBIT=35,ENEMY=6.15,PROX=50); X12-g037=HoF near-clone(sigma=0.05,ENEMY=6.85,ORBIT=27.65,SHIPS=0.087,PROX=49.89); X23-g037=Crossover(X12-g036,Elite-g036,PROD=19.78,SHIPS=0.186,ORBIT=33.77,ENEMY=7.18,PROX=44)
- Result: X23-g037=30 (7W/2L 2p, 3W 4p top2=3); Elite-g037=20 (5W/4L 2p, 0W 4p); X12-g037=18 (5W/4L 2p, 1W 4p); Mut2-g037=10 (1W/8L — SHIPS=0.24 toxic)
- Insight: X23 (SHIPS=0.186, ORBIT=33.77, ENEMY=7.18, PROX=44) won — surprising given SHIPS>0.15 and PROX=44 are usually danger signals. BUT: 3W 4p + 7W 2p = solid combined performance. SHIPS can be higher when ENEMY is moderate (7.18) and ORBIT is very high (33.77). HoF near-clone (X12) only scored 18 despite ideal params — seed variance again. Wide-explore Elite (PROD=6.01, ENEMY=9.07) scored 20 with PROD very low — interesting that low PROD + moderate ENEMY still worked OK. Stagnation=8, deep stagnation continues.
- Next: Gen 038 — stagnation=8; deep stagnation strategy; wide-explore of X23-g037 + HoF near-clone + Crossover(1st,2nd) fusion.

### Run 53: Gen 038 — best_fitness=33 (DISCARD — stagnation=9; ORBIT=35+SHIPS=0.175 works!)
- Timestamp: 2026-05-28
- What changed: Elite-g038=wide-explore(X23-g037,sigma=0.30,PROD=15.47,ENEMY=7.18,ORBIT=35.0,SHIPS=0.175,PROX=44); Mut2-g038=mutate(Elite-g037,PROD=6.36,ENEMY=9.18,ORBIT=30.9,SHIPS=0.063,PROX=50); X12-g038=HoF near-clone(sigma=0.05,PROD=16.38,ENEMY=6.34,ORBIT=28.0,SHIPS=0.079,PROX=50); X23-g038=Crossover(X23-g037,Elite-g037,PROD=14.93,ENEMY=6.52,ORBIT=28.34,SHIPS=0.050,PROX=50)
- Result: Elite-g038=33 (7W/2L 2p, 3W 4p top2=6); X12-g038=26 (6W/3L 2p, 2W 4p); X23-g038=15 (4W/5L 2p); Mut2-g038=4 (1W/8L — ENEMY=9.18+PROD=6.36 catastrophic)
- Insight: ORBIT=35 + SHIPS=0.175 + ENEMY=7.18 = fitness=33! Revised understanding: SHIPS<0.15 is NOT an absolute rule — when ORBIT is HIGH (35.0), agent earns more points from orbital positions and can afford to be patient (higher SHIPS threshold). SHIPS=0.050 (X23) too conservative — 4W/5L 2p. ENEMY=9.18 with PROD=6.36 confirmed death combination (0 growth + over-attacks). The winner X23-g037 archetype (SHIPS=0.175, ORBIT=33-35, ENEMY=7-8) is emerging as a viable alternative to the HoF (SHIPS=0.087, ORBIT=27.75, ENEMY=6.81). High ORBIT enables higher SHIPS tolerance. Stagnation=9.
- Next: Gen 039 — stagnation=9 (deep); wide-explore Elite-g038 + HoF near-clone (X12) + Crossover(Elite-g038, X12-g038). Try pushing ORBIT→35 with SHIPS ~0.12-0.18 range.

### Run 54: Gen 039 — best_fitness=22 (DISCARD — stagnation=10; SHIPS=0.239 toxic, ENEMY<5 weak)
- Timestamp: 2026-05-28
- What changed: Elite-g039=wide-explore(Elite-g038,sigma=0.30,PROD=15.73,ENEMY=4.85,ORBIT=35,SHIPS=0.239,PROX=50); Mut2-g039=mutate(X12-g038,PROD=13.76,ENEMY=4.66,ORBIT=30.57,SHIPS=0.091,PROX=44); X12-g039=HoF near-clone(sigma=0.05,PROD=14.98,ENEMY=6.36,ORBIT=24.90,SHIPS=0.089,PROX=50); X23-g039=Crossover(Elite-g038,X12-g038,PROD=18.10,ENEMY=5.68,ORBIT=35,SHIPS=0.135,PROX=44)
- Result: Elite-g039=22 (4W/5L 2p, SHIPS=0.239 toxic); Mut2-g039=22 tied (5W/4L 2p, ENEMY=4.66); X23-g039=19; X12-g039=15 (HoF near-clone underperformed!)
- Insight: SHIPS=0.239 confirmed toxic — same failure mode as gen 036 Elite (0.238). SHIPS ceiling for high-ORBIT archetype is ~0.175-0.18, not 0.24. ENEMY<5 (4.66-4.85) also hurt: very passive agents lose territory to moderate-aggression opponents. HoF near-clone scored only 15 (4W/5L 2p) — seed variance or the gen 039 seed is unfavorable to ORBIT=24.9. Gen 038 Elite (SHIPS=0.175, ORBIT=35, ENEMY=7.18, PROX=44) remains the best high-ORBIT config found. Confirmed refined SHIPS ceiling: 0.12-0.18 range for high-ORBIT archetype, 0.08-0.10 for HoF archetype.
- Next: Gen 040 — stagnation=10; deep strategy; wide-explore of Elite-g038 (best recent non-HoF config); focus on ENEMY 6-8 range, SHIPS 0.10-0.18.

### Run 55: Gen 040 — best_fitness=22 (DISCARD — stagnation=11; 3-way tie, SHIPS clamp validated)
- Timestamp: 2026-05-28
- What changed: Elite-g040=wide-explore(Elite-g039,SHIPS=0.479→CLAMPED=0.20,ORBIT=35,ENEMY=6.07,PROX=50); Mut2-g040=mutate(Mut2-g039,SHIPS=0.098,ORBIT=27.16,ENEMY=5.53,PROX=44); X12-g040=HoF near-clone(sigma=0.05,SHIPS=0.083,ENEMY=7.04,ORBIT=29.91,PROX=47); X23-g040=Crossover(Elite,Mut2,SHIPS=0.074,ENEMY=4.50,ORBIT=29.30,PROX=44.7)
- Result: 3-way tie at 22 — Elite(5W/4L 2p, 1W 4p); Mut2(6W/3L 2p — best 2p, 1W 4p); X12(5W/4L 2p, 2W 4p); X23=12 (2W/7L — ENEMY=4.50 too passive)
- Insight: SHIPS clamp to 0.20 prevented Elite catastrophe (0.479 would have been disaster). 3-way tie at 22 suggests seed is limiting all agents equally — the tournament seed constrains maximum achievable fitness. Mut2 (ORBIT=27, SHIPS=0.098, ENEMY=5.53) got best 2p at 6W/3L — close to HoF archetype but ENEMY=5.53 slightly low. HoF near-clone (X12) tied at 22 with 2W 4p — good 4p but mediocre 2p on this seed. X23 confirmed: ENEMY<5 = 2W/7L disaster. stagnation=11, next deep-stagnation gen. Ultra-stagnation (15) approaching.
- Next: Gen 041 — stagnation=11; deep strategy continues. Ultra-stagnation at 15 will bring HoF×vice-HoF crossover.

### Run 56: Gen 041 — best_fitness=24 (DISCARD — stagnation=12; PROD=26 high but 6W/3L 2p)
- Timestamp: 2026-05-28
- What changed: Elite-g041=wide-explore(Elite-g040,PROD=26.24,ENEMY=5.68,ORBIT=32.25,SHIPS=0.20↓clamped,PROX=44.4); Mut2-g041=mutate(Mut2-g040,PROD=14.55,ENEMY=4.82,ORBIT=34.19,SHIPS=0.104,PROX=50); X12-g041=HoF near-clone(PROD=16.07,ENEMY=6.86,ORBIT=28.91,SHIPS=0.088,PROX=49.4); X23-g041=Crossover(Elite,Mut2,PROD=11.47,ENEMY=6.24,ORBIT=35,SHIPS=0.113,PROX=44)
- Result: Elite-g041=24 (6W/3L 2p, 0W 4p top2=6); X12-g041=21 (4W/5L 2p, 3W 4p!); X23-g041=18 (3W/6L 2p, 3W 4p!); Mut2-g041=15 (5W/4L 2p, 0W 4p)
- Insight: X12 and X23 both got 3W 4p — strong 4p performance from HoF-archetype configs but weak 2p on this seed. PROD=26.24 (Elite) didn't kill 2p (6W/3L) — high PROD tolerable when ORBIT is also high (32.25) and PROX=44+. Mut2 (ENEMY=4.82, PROX=50) got 0 4p wins — very low ENEMY doesn't help 4p either. Ultra-stagnation (15) approaching in 3 more gens.
- Next: Gen 042 — stagnation=12; deep strategy. Ultra-stagnation (HOF×vice-HoF crossover) triggers at stagnation=15.

### Run 57: Gen 042 — best_fitness=24 (DISCARD — stagnation=13; X12 near-clone consistent at 24)
- Timestamp: 2026-05-28
- What changed: Elite-g042=wide-explore(Elite-g041,PROD=28.35,ENEMY=3.82,ORBIT=25.32,SHIPS=0.173,PROX=44); Mut2-g042=mutate(Mut2-g041,PROD=16.82,ENEMY=6.28,ORBIT=34.14,SHIPS=0.083,PROX=50); X12-g042=HoF near-clone(PROD=16.61,ENEMY=7.64,ORBIT=26.93,SHIPS=0.091,PROX=50); X23-g042=Crossover(X12-g041,X23-g041,PROD=13.40,ENEMY=5.98,ORBIT=32.88,SHIPS=0.192,PROX=50)
- Result: X12-g042=24 (6W/3L 2p, 2W 4p); Elite-g042=20 (4W/5L 2p, 1W 4p); Mut2-g042=18 (3W/6L 2p, 3W 4p); X23-g042=16 (5W/4L 2p, 0W 4p)
- Insight: HoF near-clone (X12) wins again at 24 — consistent but capped. ENEMY=7.64 slightly higher than HoF=6.81 but still works. Mut2 (ORBIT=34) got 3W 4p but only 3W/6L 2p — high ORBIT hurts 2p head-to-head. Elite ENEMY=3.82 (very passive) = 4W/5L 2p + only 1W 4p — too passive even for 4p. SHIPS=0.192 (X23) worked for 2p (5W/4L) but 0W 4p. Stagnation=13 — ultra-stagnation (15) triggers in 2 more gens if stuck.
- Next: Gen 043 — stagnation=13; deep strategy. Ultra-stagnation (HoF×vice-HoF crossover) at stagnation=15.

### Run 58: Gen 043 — best_fitness=31 (DISCARD — stagnation=14; X23 surprise 8W/1L 2p!)
- Timestamp: 2026-05-28
- What changed: Elite-g043=wide-explore(X12-g042,PROD=9.11,ENEMY=8.28,ORBIT=33.91,SHIPS=0.105,PROX=50); Mut2-g043=mutate(Elite-g042,PROD=27.96,ENEMY=4.10,ORBIT=21.93,SHIPS=0.20↓,PROX=44); X12-g043=HoF near-clone(PROD=15.56,ENEMY=6.43,ORBIT=25.50,SHIPS=0.085,PROX=47.6); X23-g043=Crossover(X12-g042,Mut2-g042,PROD=12.89,ENEMY=4.02,ORBIT=24.27,SHIPS=0.189,PROX=46.5)
- Result: X23-g043=31 (8W/1L 2p, 2W 4p top2=3); X12-g043=21 (5W/4L 2p, 2W 4p); Elite-g043=19 (3W/6L 2p, ENEMY=8.28 hurts); Mut2-g043=7 (PROD=27.96+SHIPS=0.20 = disaster)
- Insight: ENEMY=4.02 + SHIPS=0.189 + ORBIT=24.27 + PROX=46.5 = 31 (8W/1L 2p!)! REVISES earlier "ENEMY<5 always bad" rule — with SHIPS~0.19 + moderate ORBIT=24 + PROX=46+, very low ENEMY can dominate 2p. Hypothesis: low ENEMY means agent doesn't overvalue defending, attacks faster (lower fear of retaliation), captures territory quickly. This archetype differs from vice-HoF (high ENEMY=13.94) and HoF (ENEMY=6.81) — a third archetype. PROD=27.96+SHIPS=0.20 confirmed catastrophic (Mut2). Ultra-stagnation (HOF×vice-HoF crossover) triggers next gen if this fails... but 31 is promising!
- Next: Gen 044 — stagnation=14; THIS IS ULTRA-STAGNATION THRESHOLD. evolve.py will trigger ultra-stagnation strategy if stagnation reaches 15. But X23-g043 archetype (ENEMY~4, SHIPS~0.18, ORBIT~24, PROX~47) is exciting — explore it further.

### Run 59: Gen 044 — best_fitness=23 (DISCARD — stagnation=15 ULTRA-STAGNATION REACHED!)
- Timestamp: 2026-05-28
- What changed: Elite-g044=wide-explore(X23-g043,PROD=12.53,ENEMY=2.18,ORBIT=26.69,SHIPS=0.20↓,PROX=50); Mut2-g044=mutate(X12-g043,PROD=16.80,ENEMY=5.90,ORBIT=32.56,SHIPS=0.079,PROX=50); X12-g044=HoF near-clone(PROD=14.82,ENEMY=6.20,ORBIT=28.06,SHIPS=0.082,PROX=50); X23-g044=Crossover(X23-g043,X12-g043,PROD=17.54,ENEMY=7.66,ORBIT=21.59,SHIPS=0.082,PROX=50)
- Result: Elite-g044=23 (6W/3L 2p, 0W 4p top2=5); X23-g044=21 (4W/5L 2p, 3W 4p); X12-g044=19 (4W/5L 2p, 2W 4p); Mut2-g044=15
- Insight: ENEMY=2.18 (Elite) got 6W/3L 2p with 0W 4p — very low ENEMY is 2p-only specialist, no 4p wins. X12 (HoF near-clone: ENEMY=6.20, ORBIT=28.06) only 4W/5L 2p on this seed — consistent variance. HoF near-clone consistently scores 19-24 across seeds, never reaching 36 peak. Ultra-stagnation (15 gens) triggers gen 045: vice-HoF near-clone + HoF×vice-HoF crossover strategy to escape local optimum. vice-HoF (gen 024 Elite: ENEMY=13.94, PROX=50, SHIPS=0.053, PROD=11.83, ORBIT=18.24) — completely different archetype.
- Next: Gen 045 — ULTRA-STAGNATION! evolve.py triggers: Elite=ultra-explore(current best,PROX≥44), X12=vice-HoF near-clone(sigma=0.05), X23=Crossover(HoF×vice-HoF).

### Run 60: Gen 045 — best_fitness=27 (DISCARD — stagnation=16; 4W 4p RECORD from Mut2!)
- Timestamp: 2026-05-28
- What changed: Elite-g045=ultra-explore(Elite-g044,ENEMY=1.75,ORBIT=35,SHIPS=0.20↓,PROX=49.4); Mut2-g045=mutate(X23-g044,ENEMY=10.01,ORBIT=23.03,SHIPS=0.085,PROX=50); X12-g045=vice-HoF near-clone(ENEMY=6.94,ORBIT=14.99,SHIPS=0.103,PROX=44); X23-g045=Crossover(HoF,vice-HoF,ENEMY=6.71,ORBIT=20.81,SHIPS=0.096,PROX=50)
- Result: Elite-g045=27 (7W/2L 2p, 0W 4p top2=6 — 2p specialist); Mut2-g045=27 tied (5W/4L 2p, 4W 4p RECORD! top2=4); X12-g045=15 (ORBIT=15 too low); X23-g045=9 (bad)
- Insight: TWO NEW ARCHETYPES compete: (1) Ultra-low ENEMY=1.75+ORBIT=35+SHIPS=0.20 → 7W/2L 2p but 0W 4p — pure 2p specialist. (2) High ENEMY=10.01+ORBIT=23+SHIPS=0.085 → 5W/4L 2p + 4W 4p RECORD (8pts from 4p alone!) — best 4p performance ever seen. ORBIT=15 (X12) confirmed too low. The HoF×vice-HoF crossover (X23) failed completely (ORBIT=20.81+ENEMY=6.71=9pts). High-ENEMY (10.01) archetype produces exceptional 4p: 4 wins out of 6 games! Combining high-ENEMY for 4p defense + low-ENEMY for 2p aggression is the key tension. Stagnation=16.
- Next: Gen 046 — stagnation=16; ultra-stagnation continues; explore high-ENEMY archetype (ENEMY~10, ORBIT~23, SHIPS~0.085) for 4p dominance.

### Run 61: Gen 046 — best_fitness=29 (DISCARD — stagnation=17; ultra-low ENEMY 8W/1L 2p again)
- Timestamp: 2026-05-28
- What changed: Elite-g046=ultra-explore(Elite-g045,ENEMY=2.25,ORBIT=24.87,SHIPS=0.20↓,PROX=50); Mut2-g046=mutate(Mut2-g045,ENEMY=11.09,ORBIT=22.31,SHIPS=0.088,PROX=50); X12-g046=vice-HoF near-clone(ENEMY=6.86,ORBIT=13.19,SHIPS=0.112,PROX=44); X23-g046=Crossover(HoF,vice-HoF,ENEMY=6.68,ORBIT=25.49,SHIPS=0.069,PROX=44)
- Result: Elite-g046=29 (8W/1L 2p, 0W 4p top2=5); X12-g046=21 (3W/6L 2p, 4W 4p!); X23-g046=16 (3W/6L 2p, 2W 4p); Mut2-g046=12 (4W/5L 2p, 0W 4p — high ENEMY archetype failed!)
- Insight: Ultra-low ENEMY (2.25) consistently gets 7-8W/1-2L 2p but ALWAYS 0W 4p — it's a 2p-only specialist, useless for combined score. ORBIT=13 (X12 vice-HoF clone) still got 4W 4p — confirms ORBIT matters less for 4p than ENEMY does. High-ENEMY (11.09) archetype got 0W 4p on this seed — Mut2's 4W 4p in gen 045 may have been seed-lucky. Need an agent that combines 2p dominance (≥8W) WITH 4p wins — HoF achieved this with ENEMY=6.81, ORBIT=27.75 on its lucky seed. The 2p-specialist archetypes (ultra-low ENEMY) structurally can't win 4p. Stagnation=17.
- Next: Gen 047 — stagnation=17; ultra-stagnation continues; need to find archetype that wins both 2p AND 4p simultaneously.

### Run 62: Gen 047 — best_fitness=26 (DISCARD — stagnation=18; vice-HoF low-ORBIT pulls crossovers weak)
- Timestamp: 2026-05-28
- What changed: Elite-g047=ultra-explore(Elite-g046,ENEMY=4.43,ORBIT=31.22,SHIPS=0.20↓,PROX=50); Mut2-g047=mutate(X12-g046,ENEMY=6.75,ORBIT=14.53,SHIPS=0.088,PROX=44); X12-g047=vice-HoF near-clone(ENEMY=6.33,ORBIT=13.08,SHIPS=0.108,PROX=44); X23-g047=Crossover(HoF,vice-HoF,PROD=25.05,ENEMY=6.18,ORBIT=12.32,SHIPS=0.095,PROX=44)
- Result: Elite-g047=26 (6W/3L 2p, 1W 4p top2=6); X12-g047=19 (3W/6L 2p, 3W 4p); Mut2-g047=18; X23-g047=15
- Insight: Vice-HoF near-clone (ORBIT=13) still gets 3W 4p — confirms low ORBIT doesn't kill 4p. But X23 crossover (HoF×vice-HoF) consistently pulls ORBIT toward ~12-20 due to vice-HoF's low ORBIT=13.25, while HoF has ORBIT=27.75 — midpoint lands at ~20 which is suboptimal. Ultra-explore Elite (ENEMY=4.43) got 6W/3L 2p, 1W 4p — moderate-low ENEMY slightly better than 1.75-2.25 range, but still 4p-weak. The X23 crossover archetype concept is flawed: vice-HoF's ORBIT=13 drags the crossover into weak 2p territory (3W/6L 2p). Stagnation=18.
- Next: Gen 048 — stagnation=18; ultra-stagnation continues. Consider exploring new approach: HoF-anchored mutation with guided perturbation rather than vice-HoF crossover. Key insight: we need ENEMY~6-8 + ORBIT~25-30 + SHIPS~0.085-0.12 + PROX~48-50 — essentially near-HoF territory but with fresh σ exploration.

### Run 63: Gen 048 — best_fitness=28 (DISCARD — stagnation=19; Elite dominates 4p but 2p ceiling at 6W)
- Timestamp: 2026-05-28
- What changed: Elite-g048=ultra-explore(Elite-g047,ENEMY=4.26,ORBIT=35,SHIPS=0.196,PROX=50); Mut2-g048=mutate(X12-g047,ENEMY=5.55,ORBIT=12.61,SHIPS=0.099,PROX=44); X12-g048=vice-HoF near-clone(ENEMY=7.23,ORBIT=14.48,SHIPS=0.116,PROX=44.8); X23-g048=Crossover(HoF,vice-HoF,ENEMY=5.92,ORBIT=8.24,SHIPS=0.124,PROX=44)
- Result: Elite-g048=28 (6W/3L 2p, 2W 4p, top2=6 ALL TOP2!); X23-g048=19 (4W/5L 2p, 2W 4p top2=3); X12-g048=16 (4W/5L 2p, 1W 4p top2=2); Mut2-g048=15 (4W/5L 2p, 1W 4p top2=1)
- Insight: MAJOR DISCOVERY! Elite-g048 (ENEMY=4.26, ORBIT=35, SHIPS=0.196, PROX=50) placed TOP2 in ALL 6 4p games (2W + 4 runner-up)! This contradicts previous pattern of "ultra-low ENEMY = 0W 4p". ORBIT=35 (max) appears to be the key differentiator — earlier "0W 4p" agents had ORBIT=24-31. High ORBIT gives territorial advantage in 4p multi-player scenarios. However 6W/3L 2p still 3 short of HoF's 9W/0L. X23 (ORBIT=8.24 — vice-HoF drag) still managed 2W 4p. ORBIT=8.24 is the lowest we've tried and still viable. The crossover's COMPOUND=19.24 is much higher than HoF's (need to investigate).
- Next: Gen 049 — stagnation=19; ultra-stagnation continues. ORBIT=35 + ultra-low ENEMY looks very promising for 4p. Need to combine with better 2p to break HoF=36. Target: 7-8W 2p + 2-3W 4p + top2≥5 → fitness ≥ 33+.

### Run 64: Gen 049 — best_fitness=34 (DISCARD — stagnation=20; NEAR-HOF! 9W/0L 2p PERFECT but only 1W 4p)
- Timestamp: 2026-05-28
- What changed: Elite-g049=ultra-explore(Elite-g048,ORBIT=28.96,ENEMY=5.67,SHIPS=0.125,PROX=50); Mut2-g049=mutate(X23-g048,ORBIT=8.92,ENEMY=6.90,SHIPS=0.135,PROX=45.9); X12-g049=vice-HoF near-clone(ORBIT=12.81,ENEMY=6.29,SHIPS=0.104,PROX=45.3); X23-g049=Crossover(HoF,vice-HoF,ORBIT=29.82,ENEMY=5.40,SHIPS=0.081,PROX=47.5)
- Result: Elite-g049=34 (9W/0L 2p PERFECT, 1W 4p top2=5); Mut2-g049=22 (5W/4L 2p, 2W 4p top2=3); X12-g049=15 (3W/6L 2p, 2W 4p top2=2); X23-g049=7 (1W/8L 2p — catastrophically bad)
- Insight: BREAKTHROUGH NEAR-MISS! Elite-g049 (ORBIT=28.96, ENEMY=5.67, SHIPS=0.125, PROX=50) achieved PERFECT 9W/0L 2p — same as HoF! Only 1W 4p vs HoF's 2W kept it at 34 vs 36. This is the best score since HoF itself. The ultra-explore from Elite-g048's ORBIT=35 landed at ORBIT=28.96, converging toward HoF's ORBIT=27.75 range! PARADOX: X23 (ORBIT=29.82, ENEMY=5.40, SHIPS=0.081) looks even more like HoF on paper but scored only 7pts (1W/8L 2p!) — different seeds expose different weaknesses. X23's TIME_PROD_MULT=0.1253 (very low) and PROXIMITY_MULT=4.04 (very low) may be the culprits. Vice-HoF drag: X12 near-clone (ORBIT=12.8) still weak in 2p. Stagnation=20.
- Next: Gen 050 — stagnation=20; ultra-stagnation continues. Elite-g049 is tantalizingly close to HoF (ORBIT~29, ENEMY~5.7, SHIPS~0.125 vs HoF ORBIT~27.75, ENEMY~6.81, SHIPS~0.087). The gap is mostly 4p variance (1W vs 2W). Need to explore Elite-g049 neighborhood more carefully. Key concern: vice-HoF (ORBIT=13.25) drags X12/X23 to weak territory; consider patching evolve.py to update vice-HoF when ties occur.

### Run 65: Gen 050 — best_fitness=27 (DISCARD — stagnation=21; brutal seed variance destroys near-HoF clone)
- Timestamp: 2026-05-28
- What changed: Elite-g050=ultra-explore(Elite-g049,ORBIT=35,ENEMY=5.24,SHIPS=0.118,PROX=50); Mut2-g050=mutate(Mut2-g049,ORBIT=8.93,ENEMY=7.74,SHIPS=0.179,PROX=44); X12-g050=vice-HoF near-clone(ORBIT=28.12,ENEMY=5.48,SHIPS=0.133,PROX=50); X23-g050=Crossover(HoF,vice-HoF,ORBIT=22.06,ENEMY=5.82,SHIPS=0.113,PROX=44)
- Result: Mut2-g050=27 (6W/3L 2p, 3W 4p top2=3); X23-g050=20 (4W/5L 2p, 2W 4p top2=4); Elite-g050=19 (4W/5L 2p, 1W 4p top2=5); X12-g050=12 (4W/5L 2p, 0W 4p — near-Elite-g049 clone destroyed by seed)
- Insight: BRUTAL SEED VARIANCE. X12-g050 (almost identical to Elite-g049 which scored 34 last gen) only got 12pts on this seed — 0W 4p despite ORBIT=28.12. Confirms that the ORBIT~28-29 archetype is highly seed-sensitive: 34 on gen049 seed, 12 on gen050 seed. The TRUE quality is somewhere between. Mut2 (ORBIT=8.93, ENEMY=7.74, SHIPS=0.179) unexpectedly best at 27pts — interesting that low-ORBIT + moderate-high-ENEMY did better than near-HoF clones on this seed. X23 (ORBIT=22.06) did okay with 20pts. Vice-HoF fix confirmed working — X12 now clones Elite-g049 instead of old gen015. Stagnation=21.
- Next: Gen 051 — stagnation=21; ultra-stagnation continues. Best from gen050 is Mut2 (ORBIT=8.93, ENEMY=7.74) — evolve.py will build from this. Vice-HoF is still Elite-g049 (34pts, ORBIT=28.96). Need to find archetype that's robust across seeds.

### Run 66: Gen 051 — best_fitness=25 (DISCARD — stagnation=22; near-HoF X23 failed, ORBIT=10+high-ENEMY top2 all 4p again)
- Timestamp: 2026-05-28
- What changed: Elite-g051=ultra-explore(Mut2-g050,ORBIT=10.67,ENEMY=12.93,SHIPS=0.20↓,PROX=44); Mut2-g051=mutate(X23-g050,ORBIT=8.37,ENEMY=8.56,SHIPS=0.143,PROX=46); X12-g051=vice-HoF near-clone(Elite-g049,ORBIT=30.26,ENEMY=5.23,SHIPS=0.128,PROX=50); X23-g051=Crossover(HoF,Elite-g049,ORBIT=32.79,ENEMY=5.23,SHIPS=0.085,PROX=50)
- Result: Elite-g051=25 (5W/4L 2p, 2W 4p top2=6 ALL TOP2!); Mut2-g051=19 (3W/6L 2p, 3W 4p top2=4); X12-g051=19 tied (6W/3L 2p, 0W 4p top2=1); X23-g051=15 (4W/5L 2p, 1W 4p top2=1)
- Insight: RECURRING PATTERN confirmed across gen 048 and 051: low-ORBIT (10-11) + high-ENEMY (12-13) = consistently top2 in ALL 6 4p games but mediocre 2p (5W/4L). X23-g051 (ORBIT=32.79, ENEMY=5.23, SHIPS=0.085) — nearly identical to HoF on paper — scored only 15pts (1W 4p, 4W/5L 2p). X12-g051 (ORBIT=30.26, ENEMY=5.23) got 6W/3L 2p but catastrophic 0W 4p. The ORBIT~28-32 configs show extreme seed dependency: 34 in gen049, 0 4p wins in gen051. HIGH-ENEMY+LOW-ORBIT archetype (10-11, 12-13) is consistently the strongest 4p performer but cannot achieve 9W 2p. Stagnation=22.
- Next: Gen 052 — stagnation=22; ultra-stagnation. Best is Elite-g051 (ORBIT=10.67, ENEMY=12.93). Vice-HoF remains Elite-g049 (34pts). Evolve will ultra-explore from this low-ORBIT high-ENEMY base. The fundamental tension: need 9W/0L 2p (requires ORBIT~27-29, ENEMY~5-7) + 2W 4p (requires either high-ENEMY or high-ORBIT). HoF achieved both by luck on a favorable seed.

### Run 67: Gen 052 — best_fitness=25 (DISCARD — stagnation=23; X23 best combined but still short; Elite 3W 4p ALL TOP2 confirmed)
- Timestamp: 2026-05-28
- What changed: Elite-g052=ultra-explore(Elite-g051,ORBIT=6.78,ENEMY=17.82,SHIPS=0.159,PROX=44); Mut2-g052=mutate(Mut2-g051,ORBIT=8.61,ENEMY=9.60,SHIPS=0.131,PROX=47); X12-g052=vice-HoF near-clone(Elite-g049,ORBIT=28.85,ENEMY=5.58,SHIPS=0.122,PROX=50); X23-g052=Crossover(HoF,Elite-g049,ORBIT=34.01,ENEMY=6.57,SHIPS=0.085,PROX=44.3)
- Result: X23-g052=25 (5W/4L 2p, 3W 4p top2=4); Elite-g052=24 (4W/5L 2p, 3W 4p top2=6 ALL TOP2!); X12-g052=18 (6W/3L 2p, 0W 4p); Mut2-g052=11
- Insight: THREE CONFIRMED PATTERNS now: (1) Low-ORBIT(6-11)+High-ENEMY(12-18): consistently top2 ALL 6 4p games, 3-5W 4p — but caps at ~5W/4L 2p. (2) Near-HoF(ORBIT~28-30,ENEMY~5-6): consistently 6W/3L 2p but consistently 0W 4p on bad seeds. (3) X23 crossover(HoF×Elite-g049): ORBIT=32-34, ENEMY=5-7, SHIPS=0.085 — emerging as best compromise: 5W 2p + 3W 4p + top2=4 = 25pts. X23-g052 is the closest to "balanced" we've seen. The missing link is finding a config that combines 8-9W 2p WITH 2-3W 4p. HoF (ORBIT=27.75) happened to be in the sweet spot where it won 9W 2p AND 2W 4p on seed 029. Stagnation=23.
- Next: Gen 053 — stagnation=23; ultra-stagnation. Best is X23-g052 (ORBIT=34.01, ENEMY=6.57). Evolve will ultra-explore from this balanced archetype. X12 will continue as vice-HoF (Elite-g049) clone. X23 will be another HoF×Elite-g049 crossover — keep trying this promising direction.

### Run 68: Gen 053 — best_fitness=26 (DISCARD — stagnation=24; Elite ORBIT=25.94 best balanced yet, 3W 4p)
- Timestamp: 2026-05-28
- What changed: Elite-g053=ultra-explore(X23-g052,ORBIT=25.94,ENEMY=5.88,SHIPS=0.111,PROX=50); Mut2-g053=mutate(Elite-g052,ORBIT=8.91,ENEMY=19.60,SHIPS=0.169,PROX=44); X12-g053=vice-HoF near-clone(Elite-g049,ORBIT=28.67,ENEMY=5.93,SHIPS=0.124,PROX=49.3); X23-g053=Crossover(HoF,Elite-g049,ORBIT=28.96,ENEMY=7.29,SHIPS=0.074,PROX=50)
- Result: Elite-g053=26 (5W/4L 2p, 3W 4p top2=5); X23-g053=20 (5W/4L 2p, 1W 4p top2=3); Mut2-g053=18 (4W/5L 2p, 2W 4p top2=2); X12-g053=14 (4W/5L 2p, 0W 4p top2=2)
- Insight: Elite-g053 (ORBIT=25.94, ENEMY=5.88) achieved 5W 2p + 3W 4p = best combined performance in recent gens. This is an ultra-explore from X23-g052 (ORBIT=34.01) that converged toward HoF territory (ORBIT=27.75). The convergence pattern: X23-g052 ORBIT=34→ultra-explore→Elite-g053 ORBIT=25.94. 3W 4p is solid but still 1 win short of HoF's 4p record. X23-g053 (ORBIT=28.96, ENEMY=7.29) only 1W 4p despite similar ORBIT to X12. X12 again 0W 4p. Near-HoF clone (ORBIT~28-29, ENEMY~5-6) consistently fails 4p across many seeds. Stagnation=24.
- Next: Gen 054 — stagnation=24; ultra-stagnation. Best is Elite-g053 (ORBIT=25.94). Ultra-explore will push from here — could land at ORBIT~27-28 which is HoF's exact sweet spot. The ultra-explore strategy is slowly converging toward the HoF ORBIT zone.

### Run 69: Gen 054 — best_fitness=25 (DISCARD — stagnation=25; ORBIT=35 MAX kills 2p; Mut2 7W/2L 2p but weak 4p)
- Timestamp: 2026-05-28
- What changed: Elite-g054=ultra-explore(Elite-g053,ORBIT=20.27,ENEMY=8.70,SHIPS=0.143,PROX=50); Mut2-g054=mutate(X23-g053,ORBIT=?,ENEMY=?,SHIPS=?,PROX=?); X12-g054=vice-HoF near-clone(Elite-g049,ORBIT=30.84,ENEMY=5.92,SHIPS=0.116,PROX=46.2); X23-g054=Crossover(HoF,Elite-g049,ORBIT=35.0 MAX,ENEMY=6.19,SHIPS=0.101,PROX=48.4)
- Result: Mut2-g054=25 (7W/2L 2p, 1W 4p top2=2); Elite-g054=24 (5W/4L 2p, 2W 4p top2=5); X12-g054=17 (4W 2p, 1W 4p top2=3); X23-g054=12 (2W/7L 2p — ORBIT=35 catastrophic in 2p)
- Insight: CONFIRMED: ORBIT=35 MAX is lethal to 2p performance (2W/7L). The 2p-optimal ORBIT range is ~25-30, not at the 35 cap. Elite (ORBIT=20.27, ENEMY=8.70) balanced again with 5W 2p + 2W 4p top2=5. Mut2 (mutation of X23-g053 which had ORBIT=28.96, ENEMY=7.29) got 7W/2L 2p but only 1W 4p — this archetype trades 4p for 2p dominance. Key lesson: ORBIT must be ~25-30 for balanced play; ORBIT=35 and ORBIT<15 both compromise 2p. Stagnation=25.
- Next: Gen 055 — stagnation=25; ultra-stagnation. Best is Mut2-g054 (mutation of X23-g053, strong 2p). Evolve will ultra-explore from this. Vice-HoF remains Elite-g049 (ORBIT=28.96). If the HoF is truly at ORBIT=27.75, and we've identified the 2p-optimal zone is ORBIT~25-30, we need to find a config that reliably hits both 2p and 4p on the same seed.

### Run 70: gen 055 — Mut2(7W/2L 2p, 2W 4p top2=3)=28 (DISCARD)
- Timestamp: 2026-05-28 18:16
- What changed: Elite-g055=ultra-explore(Mut2-g054,ORBIT=25.97,ENEMY=11.98,PROX=44); Mut2-g055=mutate(Mut2-g054); X12-g055=vice-HoF near-clone(Elite-g049,ORBIT=30.06,ENEMY=5.34,SHIPS=0.116); X23-g055=Crossover(HoF,Elite-g049,ORBIT=25.16,ENEMY=8.84)
- Result: Mut2-g055=28 (7W/2L 2p, 2W 4p top2=3); Elite-g055=19 (3W/6L 2p, 2W 4p top2=6 ALL TOP2!); X12-g055=16; X23-g055=15 — best=28 < HoF=36, stagnation=26
- Insight: Elite-g055 (ORBIT=25.97, ENEMY=11.98, PROX=44) = 4th time a LOW-ORBIT+HIGH-ENEMY config placed top2 in ALL 6 4p games! But still lost 2p (3W/6L). This archetype is consistent at 4p dominance but 2p-weak. Mut2 continues to be the reliable ~28pt scorer (7W 2p, moderate 4p). Vice-HoF correctly reflects Elite-g049 (ORBIT=28.96) after tie-breaking fix. Stagnation=26.
- Next: Gen 056 — ultra-stagnation. evolve.py will try bold exploration. The ALL-TOP2 4p pattern deserves direct attention: consider a config that intentionally combines ORBIT~25 + ENEMY~8-12 + stronger 2p focus (increase PROD or SHIPS slightly?). The HoF (ORBIT=27.75, ENEMY=6.81) is somewhere between current archetypes.

### Run 71: gen 056 — Elite(6W/3L 2p, ALL TOP2! 4p)=28 (DISCARD)
- Timestamp: 2026-05-28 18:37
- What changed: Elite-g056=ultra-explore(Mut2-g055,ORBIT=20.12,ENEMY=13.00,PROX=50); Mut2-g056=mutate(ORBIT=24.58,ENEMY=12.95,PROX=44.7); X12-g056=vice-HoF near-clone(ORBIT=29.32,ENEMY=5.68,PROX=45.6); X23-g056=Crossover(HoF,vice-HoF,ORBIT=22.83,ENEMY=7.04,PROX=50)
- Result: Elite-g056=28 (6W/3L 2p, 2W 4p top2=6 ALL TOP2!); X23-g056=19; Mut2-g056=16 (3W/6L 2p — high ENEMY hurts 2p); X12-g056=15 (0W 0top2 4p). Best=28 < HoF=36, stagnation=27
- Insight: 5th occurrence of LOW-ORBIT+HIGH-ENEMY config placing top2 in ALL 6 4p games (Elite ORBIT=20.12, ENEMY=13.00). But 2p is limited to 6W/3L (not the 9W/0L needed to match HoF). X12 (vice-HoF ORBIT=29.32, ENEMY=5.68) again completely failed 4p (0W, 0top2). The pattern is consistent: ORBIT<25 + ENEMY>10 = 4p dominance; ORBIT~28-30 + ENEMY~6 = 2p dominance. HoF (ORBIT=27.75, ENEMY=6.81) landed in the magic zone where both aligned on a lucky seed. We can't reproduce that seed alignment.
- Next: Gen 057 — stagnation=27 ultra-deep. Consider whether to try deliberately targeting ORBIT~24-26 + ENEMY~8-10 (midzone between the two archetypes) to see if it can achieve better 2p while keeping some 4p. Also: SHIPS_MULT exploration — HoF had SHIPS=0.087, most configs are 0.10-0.13. Maybe very low SHIPS is key to 2p dominance.

### Run 72: gen 057 — X23(5W 2p, 3W 4p)=24 (DISCARD)
- Timestamp: 2026-05-28 18:54
- What changed: Elite-g057=ultra-explore(Elite-g056,ORBIT=24.92,ENEMY=14.59,SHIPS=0.078,PROX=49); Mut2-g057=mutate(ORBIT=21.12,ENEMY=7.48,SHIPS=0.069,PROX=50); X12-g057=vice-HoF near-clone(ORBIT=27.86,ENEMY=5.82,SHIPS=0.118,PROX=50); X23-g057=Crossover(HoF,vice-HoF,ORBIT=33.08,ENEMY=7.82,SHIPS=0.132,PROX=44)
- Result: X23-g057=24 (5W/4L 2p, 3W 4p top2=3); Elite-g057=21 (5W/4L 2p, 0W 4p top2=6 ALL 2nd!); Mut2-g057=21; X12-g057=12 (3W/6L 2p — vice-HoF clone flopped). Best=24 < HoF=36, stagnation=28
- Insight: 6th time a LOW-ORBIT+HIGH-ENEMY config placed top2 in ALL 6 4p games (Elite-g057 ORBIT=24.92, ENEMY=14.59) — but it's always 2nd, never 1st! The high-ENEMY archetype consistently finishes 2nd in 4p, not 1st. X23-g057 with ORBIT=33.08 got 5W/4L 2p (much better than 35 MAX). X12 (vice-HoF ORBIT=27.86) lost 2p badly — seed variance hurting this direction again. Very low SHIPS (0.069-0.078) didn't help. Stagnation=28, ultra-deep.
- Next: Gen 058 — stagnation=28. Consider: the 2nd-place-always pattern suggests high-ENEMY agents expand moderately and survive but can't deliver the killing blow in 4p. Maybe the 4p winner needs a different combination. Also: X23-g057 (ORBIT=33, ENEMY=7.82) performed decently — moderate ORBIT + moderate ENEMY might be the right zone for balanced play.

### Run 73: gen 058 — Mut2(6W/3L 2p, 2W 4p)=25 (DISCARD)
- Timestamp: 2026-05-28 19:12
- What changed: Elite-g058=ultra-explore(X23-g057,ORBIT=30.81,ENEMY=9.50,SHIPS=0.138,PROX=44,PROD=21.51); Mut2-g058=mutate(ORBIT=26.57,ENEMY=16.45,SHIPS=0.079,PROX=49.7); X12-g058=vice-HoF near-clone(ORBIT=29.48,ENEMY=6.45,SHIPS=0.129); X23-g058=Crossover(HoF,vice-HoF,ORBIT=27.30,ENEMY=5.82,SHIPS=0.081)
- Result: Mut2-g058=25 (6W/3L 2p, 2W 4p top2=3); X23-g058=21 (6W/3L 2p, 1W 4p — near-HoF params fail 4p on new seed); Elite-g058=19 (3W/6L 2p, 2W 4p top2=6 ALL TOP2 — 7th occurrence!); X12-g058=13. Best=25 < 36, stagnation=29
- Insight: CONFIRMED: HoF params (ORBIT=27.30, ENEMY=5.82, SHIPS=0.081) get good 2p (6W/3L) but only 1W 4p — the HoF 4p score was seed-dependent. Mut2 with ENEMY=16.45 (highest tested) still got 6W/3L 2p — very high ENEMY doesn't necessarily hurt 2p at this ORBIT level. Elite (ORBIT=30.81, ENEMY=9.50, PROD=21.51) shows high PROD doesn't help much. 7th ALL-TOP2 4p occurrence confirms this archetype is systematic (ORBIT~20-31, ENEMY~9-14 → always top2 in 4p but rarely wins). Stagnation=29.
- Next: Gen 059 — stagnation=29. The Mut2 (ORBIT=26.57, ENEMY=16.45) result is interesting — maybe high ENEMY + mid-ORBIT is worth exploring deeper. The ALL-TOP2 4p pattern needs investigation: why do these configs always finish 2nd? Is it an aggression issue in late-game 4p?

### Run 74: gen 059 — X12(7W/2L 2p, 2W 4p)=28 (DISCARD)
- Timestamp: 2026-05-28 19:30
- What changed: Elite-g059=ultra-explore(Mut2-g058,ORBIT=31.99,ENEMY=15.53,SHIPS=0.079,PROX=50); Mut2-g059=mutate(ORBIT=31.86,ENEMY=6.41,SHIPS=0.085,PROX=48.3); X12-g059=vice-HoF near-clone(ORBIT=29.98,ENEMY=6.17,SHIPS=0.137,PROX=47.4); X23-g059=Crossover(HoF,vice-HoF,ORBIT=23.58,ENEMY=4.52,SHIPS=0.053,PROX=44)
- Result: X12-g059=28 (7W/2L 2p, 2W 4p top2=3 — best 2p performance in many gens!); Elite-g059=24 (4W/5L 2p, 3W 4p top2=6 ALL TOP2 8th time!); Mut2-g059=16; X23-g059=10. Best=28 < 36, stagnation=30
- Insight: X12-g059 (ORBIT=29.98, ENEMY=6.17, SHIPS=0.137) got 7W/2L 2p — excellent 2p but only 2W 4p top2=3. ORBIT~30 + moderate ENEMY is strong for 2p. Elite (ORBIT=31.99, ENEMY=15.53, SHIPS=0.079) placed top2 in ALL 6 4p games — 8th consecutive occurrence of this archetype! But X23 with very low SHIPS=0.053 and ENEMY=4.52 totally bombed (10pts). Mut2 with ORBIT=31.86 + SHIPS=0.085 (near-HoF SHIPS) only got 16 — ORBIT=31 is too high for 2p dominance. The vice-HoF direction (ORBIT~30, ENEMY~6) can deliver 7W 2p but fails 4p systematically. Stagnation=30.
- Next: Gen 060 — stagnation=30. The 8-consecutive ALL-TOP2 4p pattern for low-ORBIT+high-ENEMY needs investigation. X12 (ORBIT=29.98, ENEMY=6.17) delivered 7W/2L 2p — maybe even lower ENEMY (~5-6) with ORBIT=27-28 zone would be worth exploring more carefully. HoF (ORBIT=27.75, ENEMY=6.81) seems to be the 2p sweet spot.

### Run 75: gen 060 — X12(6W/3L 2p, 2W 4p)=25 (DISCARD)
- Timestamp: 2026-05-28 19:47
- What changed: Elite-g060=ultra-explore(X12-g059,ORBIT=29.93,ENEMY=8.14,SHIPS=0.103,PROX=44); Mut2-g060=mutate(ORBIT=30.04,ENEMY=12.35,SHIPS=0.082,PROX=44.3); X12-g060=vice-HoF near-clone(ORBIT=28.51,ENEMY=5.49,SHIPS=0.131,PROX=48); X23-g060=Crossover(HoF,vice-HoF,ORBIT=25.71,ENEMY=5.69,SHIPS=0.142,PROX=46.9,PROD=19.25)
- Result: X12-g060=25 (6W/3L 2p, 2W 4p top2=3); Mut2-g060=24 (5W 2p, 3W 4p top2=3); Elite-g060=17 (3W/6L 2p, 1W 4p top2=6 ALL TOP2 9th time!); X23-g060=12 (0W 4p — high PROD confirmed useless). Best=25 < 36, stagnation=31
- Insight: 9th consecutive occurrence of mid-ORBIT+mid-ENEMY config placing top2 in ALL 6 4p games. This is a completely systematic behavior of this archetype. High PROD (X23=19.25) didn't help at all. X12 (ORBIT=28.51, ENEMY=5.49) continues the vice-HoF direction's strong 2p (6W/3L) but moderate 4p. The 2p-optimal configs cluster around ORBIT=28-30, ENEMY=5-6. The 4p-winner configs need ORBIT<25 + ENEMY>10 but never reach top overall fitness. Stagnation=31.
- Next: Gen 061 — stagnation=31, ultra-deep. The algorithm has been stuck for 31 gens. Need to consider whether evolve.py's ultra-stagnation logic is actually diversifying enough, or whether we're trapped in a local maximum of the search space. Maybe it's time to evaluate whether the current evolve.py strategies need structural changes.

### Run 76: gen 061 — Mut2(8W/1L 2p, ORBIT=35)=27 (DISCARD)
- Timestamp: 2026-05-28 20:06
- What changed: Elite-g061=ultra-explore(X12-g060,ORBIT=33.31,ENEMY=5.38,SHIPS=0.155,PROX=46.8); Mut2-g061=mutate(ORBIT=35.0 MAX,ENEMY=11.03,SHIPS=0.069,PROX=50); X12-g061=vice-HoF near-clone(ORBIT=28.39,ENEMY=6.16,SHIPS=0.122); X23-g061=Crossover(HoF,vice-HoF,ORBIT=30.25,ENEMY=7.98,SHIPS=0.123)
- Result: Mut2-g061=27 (8W/1L 2p — incredible! but 1W 4p only); X12-g061=20 (5W 2p, 1W 4p top2=3); X23-g061=15; Elite-g061=16 (2W/7L 2p — ORBIT=33.31 is bad). Best=27 < 36, stagnation=32
- Insight: EXTREME SEED VARIANCE CONFIRMED. Mut2-g061 ORBIT=35 got 8W/1L 2p — the opposite of gen 054 X23 ORBIT=35 which got 2W/7L 2p. Same ORBIT value, radically different seed → radically different result. This means ORBIT=35 is not inherently bad — it's just high variance. Elite-g061 ORBIT=33.31 went 2W/7L 2p reinforcing that higher ORBIT is unstable. Applied fix: capped ORBIT_BONUS max at 32 to reduce search variance (ORBIT>32 has shown extreme seed sensitivity). Mut2's 8W/1L 2p came from a lucky seed combo. Stagnation=32.
- Next: Gen 062 — stagnation=32. With ORBIT max capped at 32, future configs will stay in the more stable range. The search should focus on ORBIT=26-32, ENEMY=5-12 zone. The systematic ALL-TOP2 4p pattern (10+ occurrences now) suggests ORBIT~20-32 + ENEMY~8-15 consistently produces 2nd-place 4p finishers. What makes a 4p winner? Need a config that can close out 4p games when down to 2-3 players.

### Run 77: gen 062 — Elite(7W/2L 2p, top2=5)=28 (DISCARD)
- Timestamp: 2026-05-28 20:24
- What changed: Elite-g062=ultra-explore(Mut2-g061,ORBIT=32.00,ENEMY=7.05,SHIPS=0.147,PROX=44); Mut2-g062=mutate(ORBIT=25.70,ENEMY=5.65,SHIPS=0.128,PROX=50); X12-g062=vice-HoF near-clone(ORBIT=31.97,ENEMY=5.53,SHIPS=0.121); X23-g062=Crossover(HoF,vice-HoF,ORBIT=32.00,ENEMY=4.63,SHIPS=0.106)
- Result: Elite-g062=28 (7W/2L 2p, 1W 4p top2=5); Mut2-g062=20; X12-g062=15 (4W 2p, low 4p); X23-g062=15 (2W/7L 2p — ORBIT=32 + ENEMY=4.63 is bad combo). Best=28, stagnation=33
- Insight: Elite-g062 (ORBIT=32, ENEMY=7.05) showed ORBIT=32 is OK for 2p when ENEMY is ~7 (not ~5). But X23 with same ORBIT=32 but ENEMY=4.63 went 2W/7L. This confirms ENEMY matters a lot at high ORBIT: ENEMY~7 is needed to maintain 2p competitiveness when ORBIT~32. The ORBIT cap at 32 is working — no more runaway ORBIT=35 configs. Stagnation=33.
- Next: Gen 063 — stagnation=33. Elite-g062 (ORBIT=32, ENEMY=7) is the new best of gen. Need configs that can combine 7-8W 2p with 3+W 4p. The persistent 4p weakness (only 1-2W per gen) remains the main bottleneck. ORBIT=27-30, ENEMY=7-10 might be the best zone to explore deeper.

### Run 78: gen 063 — Mut2(6W/3L 2p)=22 (DISCARD)
- Timestamp: 2026-05-28 20:38
- What changed: Elite-g063=ultra-explore(Elite-g062,ORBIT=32,ENEMY=9.75,SHIPS=0.200 MAX,PROX=45.8); Mut2-g063=mutate(ORBIT=21.92,ENEMY=5.68,SHIPS=0.137); X12-g063=vice-HoF near-clone(ORBIT=30.17,ENEMY=6.29,SHIPS=0.116); X23-g063=Crossover(HoF,vice-HoF,ORBIT=27.63,ENEMY=6.01,SHIPS=0.122)
- Result: Mut2-g063=22 (6W/3L 2p, 1W 4p); Elite-g063=20 (ALL TOP2 4p 10th time! SHIPS=0.200 MAX confirmed useless); X12/X23=18 each. Near-HoF X23 (ORBIT=27.63, ENEMY=6.01) again failed (4W/5L 2p). Best=22, stagnation=34
- Insight: 10th consecutive occurrence of the ALL-TOP2 4p pattern (ORBIT~21-32, ENEMY~8-15 → always 2nd in 4p). SHIPS=0.200 MAX (Elite) didn't help at all. Near-HoF clones continue to underperform — the HoF=36 was truly a lucky seed alignment. At stagnation=34, updating autoresearch.md with major insights. The algorithm is stuck in a 22-28pt range and cannot break through 36. Needs structural insight or lucky seed.
- Next: Gen 064 — stagnation=34. Should update autoresearch.md "What's Been Tried" with all learnings. Core pattern: 2p-optimal = ORBIT=27-32, ENEMY=5-10; 4p-specialist = ORBIT<25, ENEMY>10 (always 2nd not 1st!). HoF combination was seed-dependent.
