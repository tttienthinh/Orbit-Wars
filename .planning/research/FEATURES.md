# Features Research — Orbit Wars Competitive Agent

**Analysis Date:** 2026-05-06

## Table Stakes (must have or lose consistently)

These are the baseline features that every agent above the bottom quartile implements.
Missing any one of these causes systematic losses.

| Feature | Why it's required | Current status |
|---------|-------------------|----------------|
| **Orbital intercept prediction** | Targeting current position of orbiting planets wastes fleets | ✓ In `05` (approximate) |
| **Spawn-offset correction** | Fleet starts at planet surface; aiming at planet center causes misses on large+fast combos | ✗ Bug present |
| **Sun collision avoidance** | Fleets on certain angles cross the sun disc and are destroyed | ✗ Not implemented |
| **Production-value targeting** | Nearest planet ≠ best planet; high-production planets compound advantage | ✗ Not implemented |
| **Minimum garrison** | Never strip a planet to 0; always keep N ships for production safety | ✗ Not implemented |
| **Incoming threat detection** | Identify enemy fleets heading toward owned planets, compute arrival turn | ✗ Not implemented |
| **Defensive response** | Reinforce threatened planet or counter-attack source | ✗ Not implemented |

## Differentiators (top 25% → top 10%)

These features separate competitive agents from good ones. yuriygreben's agent has most of them.

| Feature | What it does | Complexity |
|---------|--------------|------------|
| **N-turn forward simulation** | Predict board state T turns ahead; score targets by future value not current value | High |
| **Game phase awareness** | Different strategy for early (expand aggressively), mid (contest), late (hold + snipe) | Medium |
| **Fleet race detection** | Know if enemy fleet will arrive before yours; adjust send count accordingly | Medium |
| **Multi-fleet coordination** | Multiple planets attack same target simultaneously; avoid over-sending | Medium |
| **Crash-exploit** | When two enemies' fleets arrive at same time, they cancel; swoop in after | High |
| **Reinforce routing** | Send ships from rear/safe planets to frontline planets efficiently | Medium |
| **Comet timing** | Comets spawn at turns 50/150/250/350/450; opportunistically capture if profitable | Low |
| **Snipe detection** | Detect when enemy will capture a neutral just before you; adjust timing | Medium |
| **Weight-tuned scoring** | All the above factors weighted and combined into a single score per candidate move | Medium |

## Anti-Features (don't build these)

| Anti-feature | Why not |
|--------------|---------|
| **Pure RL before fixing heuristics** | RL learns on top of the reward signal, not bugs; a buggy physics layer means RL learns wrong strategies |
| **Real-time pathfinding (A*)** | Fleets travel in straight lines with fixed angle; there is no pathfinding; the only "path" is the angle |
| **Comet priority in early game** | Comets appear at turn 50+; optimizing for them early delays planet expansion |
| **4-player-first strategy** | 4-player games have different dynamics (3 enemies); solve 2-player first, then adapt |
| **Fleet splitting into 1-ship scouts** | `03` showed this — 1 ship is speed 1.0 units/turn; too slow to be useful as scout |
| **Complex ML for targeting** | A well-tuned scoring function outperforms ML targeting in short competition windows |

## Feature Priority Order (given current state)

```
Phase 1 — Fix bugs (immediate Elo gain):
  1. Spawn-offset correction          ← known bug, easy fix
  2. Sun collision avoidance          ← silent ship destroyer
  3. Minimum garrison (don't strip)   ← stops self-destruction

Phase 2 — Add intelligence:
  4. Production-value targeting       ← biggest strategic leverage
  5. Threat detection + defense       ← stops easy conquests
  6. Fleet race detection             ← avoids wasteful sends

Phase 3 — Forward simulation:
  7. N-turn sim horizon               ← matches yuriygreben's level
  8. Game phase awareness             ← opens/closes early/late strategies
  9. Multi-fleet coordination         ← efficient resource use

Phase 4 — RL layer:
  10. PettingZoo env wrapper
  11. PPO self-play training
  12. Fine-tune weights
```

## Feature Dependencies

```
Spawn-offset fix ─────────────────────→ all aiming features
Sun avoidance ────────────────────────→ route planning
Threat detection ─────────────────────→ defensive response
                                     ↘ fleet race detection
N-turn simulation ────────────────────→ game phase awareness
                                     ↘ crash-exploit detection
Heuristic foundation (phases 1-3) ───→ RL warm-start policy
```

---
*Features research: 2026-05-06 (inline, agents rate-limited)*
