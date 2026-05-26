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
| 004 OrbitalDominance | Orbit bonus (ORBIT=20, DIST=0.3) | 7/2 | 2 | 2 | 29 |

**Seed = 16**

Key observations:
- 004 OrbitalDominance wins with fitness=29 — low DIST_MULT=0.3, high ORBIT_BONUS=20
- 002 Economist is worst — high DIST_MULT=1.0 and OVEREXTEND_MULT=1.5 appear very harmful
- Low DIST_MULT seems crucial: 001(0.2), 003(0.8), 004(0.3) > 002(1.0)

---

## Key Insights

1. **Low DIST_MULT wins**: agents with DIST_MULT < 0.5 consistently outperform those with DIST_MULT ≥ 0.8
2. **High ORBIT_BONUS is competitive**: inner-ring planets as launch pads seem valuable
3. **Economist archetype loses badly**: balanced production + high penalties = 0 wins

## Next Ideas

- Try much lower OVEREXTEND_MULT (<0.1)
- Try even higher ORBIT_BONUS (25-30)
- Try very aggressive crossover between OrbitalDominance and Coordinator
