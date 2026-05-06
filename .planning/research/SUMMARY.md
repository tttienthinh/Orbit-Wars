# Research Summary — Orbit Wars Competitive Agent

**Analysis Date:** 2026-05-06

## Stack

**Submission:** Pure Python 3.12 + NumPy. No new dependencies (main.py must be self-contained or tarball).

**RL training:** `pettingzoo` 1.24 + `stable-baselines3` 2.3 (PPO) + `supersuit` 3.9. Train against a snapshot pool of past checkpoints for stable self-play. GPU optional — start on CPU, rent if sim speed is bottleneck.

**Backtesting:** orbit-wars-lab already provides everything needed (TrueSkill, replay viewer, agent zoo).

## Table Stakes (must fix first)

1. **Spawn-offset bug** — fleet angle must account for fleet spawning at planet surface, not center. Iterative solve required.
2. **Sun collision avoidance** — parametric ray vs circle test; reject angles that cross sun disc.
3. **Minimum garrison** — never send more than `ships - max(production*2, 5)`.

All three are pre-conditions. Without them, everything else is built on a broken foundation.

## Key Differentiators (vs yuriygreben's agent)

yuriygreben has: 110-turn sim horizon, 10+ action types, game-phase detection, weight-tuned scoring.

To match and surpass:
1. **Production-value targeting** — score targets by `production / ships_needed` not proximity
2. **Threat detection + defense** — detect incoming enemy fleets, compute arrival turn, reinforce
3. **Fleet race detection** — account for enemy fleets already en route to target
4. **N-turn sim horizon** — score targets by *future* production value, not current garrison
5. **Game phase awareness** — early=expand, mid=contest, late=hold+snipe

## Watch Out For

- **Orbital prediction**: use `θ₀ + ω × t` formula (not iterative position steps) to avoid drift
- **Double-spending**: track committed ships per planet before scoring next candidate  
- **1-second timeout**: vectorize trajectory computation with NumPy; pre-compute all planet positions for all T at turn start
- **RL reward hacking**: use production-advantage reward per turn, not ship accumulation
- **Submission isolation**: multi-file agent requires `.tar.gz` bundle, not just `main.py`
- **4-player divergence**: tune 2-player first; 4-player needs different aggression level

## Recommended Build Sequence

```
Phase 1: Fix physics + min garrison          → immediate Elo gain
Phase 2: Production targeting + defense      → approach yuriygreben level
Phase 3: N-turn sim + game phase awareness   → match yuriygreben, exceed with timing
Phase 4: PettingZoo + PPO self-play          → potentially exceed rule-based ceiling
```

---
*Summary synthesized: 2026-05-06 (inline)*
