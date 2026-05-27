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
