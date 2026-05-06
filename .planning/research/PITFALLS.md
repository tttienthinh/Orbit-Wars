# Pitfalls Research — Orbit Wars Competitive Agent

**Analysis Date:** 2026-05-06

## Physics Bugs

### Spawn-offset error (KNOWN, high priority)
- **What:** Fleet is launched from planet center in angle calculation, but actually spawns at the planet's surface edge. For a large origin planet (radius ~2.6 for production=5) targeting a small fast orbiting planet, the angle error compounds with travel time to produce a miss.
- **Warning sign:** Fleets aimed at orbiting planets arrive behind the planet's expected position at large source radii.
- **Fix:** Iterative angle solve: `spawn_pos = center + unit(angle) * src_radius`; recompute `angle = atan2(tgt_future_y - spawn_y, tgt_future_x - spawn_x)`, iterate 3–5 times.
- **Phase:** Phase 1 (first fix)

### Sun collision ignored
- **What:** Fleets launched at angles that pass near the center (sun at 50,50, radius 10) are silently destroyed. Current agents don't check this.
- **Warning sign:** Ships disappear mid-flight with no combat reported; happens on attacks across the center of the map.
- **Fix:** `path_crosses_sun()` parametric ray vs circle test; reject or find alternate angle.
- **Phase:** Phase 1

### Orbital prediction drift
- **What:** `next_position` called with `t` total turns assumes constant angular velocity from current position. This is correct per game rules but can accumulate float errors over 100+ turns.
- **Warning sign:** Intercept slightly off for very long-travel attacks (T > 80 turns).
- **Fix:** Use `initial_planets` + cumulative angle formula: `θ(t) = θ₀ + ω × t` rather than iterating `next_position` in a loop.
- **Phase:** Phase 1

### Wrong orbital center reference in `05`
- **What:** `angle_to_position` in `05` adds `CENTER_X/CENTER_Y` but `position_to_angle` subtracts them. If the coordinate convention is inconsistent between files, predictions break silently.
- **Warning sign:** Fleets targeting orbiting planets arrive consistently to the left/right of target.
- **Fix:** Standardize all physics math in a single `physics.py` module with explicit coordinate conventions.
- **Phase:** Phase 1

## Strategy Bugs

### Stripping planets to zero ships
- **What:** Agent sends all ships, leaving owned planet with 0 garrison. Next turn, production adds N ships, but if an enemy fleet arrives that turn the planet is lost.
- **Warning sign:** Lose planets you just captured because they were left empty.
- **Fix:** `MIN_GARRISON = max(production * 2, 5)` — never send more than `ships - MIN_GARRISON`.
- **Phase:** Phase 1

### Double-spending planets
- **What:** Two targets score highest for the same source planet. Agent sends to both but can only afford one. The second fleet never launches (not enough ships), or worse, both launch with insufficient ships to capture.
- **Warning sign:** Attempted captures repeatedly fail by a few ships.
- **Fix:** Track committed ships per source planet in the planner; deduct before scoring next candidate.
- **Phase:** Phase 2

### No fleet race awareness
- **What:** Agent sends X ships to capture a planet but enemy has already sent a fleet that arrives 3 turns earlier. Your fleet arrives to fight a now-enemy planet with reinforced garrison.
- **Warning sign:** "Wasteful" attacks on planets that change hands just before your fleet arrives.
- **Fix:** In intercept solve, check if any enemy fleet arrives before your fleet at T. Add enemy incoming ships to target garrison before computing `ships_needed`.
- **Phase:** Phase 2

### Attacking into mutual destruction
- **What:** Two enemy fleets collide at a planet and cancel. Current agent doesn't detect this opportunity to swoop in with minimal ships.
- **Phase:** Phase 3 (crash-exploit, after basics are solid)

## RL-Specific Pitfalls

### Reward hacking on ship accumulation
- **What:** Agent learns to park all ships on safe planets and never attack (maximises ship count without risking losses). Terminal win reward is too sparse to overcome.
- **Warning sign:** Training reward increases but agent becomes passive; loses to aggressive baselines.
- **Fix:** Shape reward as `Δ(my_production - enemy_production)` per turn (production advantage reflects territory control, not just accumulated ships).
- **Phase:** Phase 4 (RL)

### Training instability with self-play
- **What:** Agent overfits to its current self, then collapses when the opponent updates. Leads to oscillating Elo rather than monotonic improvement.
- **Warning sign:** Training reward is noisy with large swings; checkpoint performance non-monotonic.
- **Fix:** Maintain a pool of N past checkpoints (e.g., N=5); sample opponent uniformly. Use league play only if self-play is unstable after 2–3 attempts.
- **Phase:** Phase 4

### Observation encoding instability
- **What:** If planets are sorted by ID (not by distance from player), the same strategic situation looks different every game due to map randomness. RL fails to generalize.
- **Warning sign:** Agent plays well on seed=42 but badly on different seeds.
- **Fix:** Sort planets by distance from player centroid; this gives a stable, strategy-relevant ordering.
- **Phase:** Phase 4

### Action masking neglect
- **What:** RL agent tries to send from planets it doesn't own, or sends to non-existent planets in padded slots. Wasted action capacity + training noise.
- **Warning sign:** Many no-op actions in the replay; agent sends from enemy planets.
- **Fix:** Apply action masking: set logit = -∞ for invalid actions (not owned, or padded slot). SB3 supports action masking via `MaskablePPO` from `sb3-contrib`.
- **Phase:** Phase 4

## Competition-Specific Pitfalls

### 1-second timeout
- **What:** `actTimeout = 1 second`. Forward simulation loops (N-turn intercept search, T=120) must complete in time. With 40 planets × 10 sources = 400 candidates × 120-turn search = 48,000 iterations per turn.
- **Warning sign:** Kaggle reports agent crashes/timeouts; submission gets 0 score.
- **Fix:** Vectorize with NumPy; pre-compute planet trajectories for all T at turn start (not per-candidate). Profile with `time.time()` guard: if > 0.8 seconds, truncate candidate list.
- **Phase:** Phase 2 (when forward sim is added)

### Submission isolation
- **What:** `main.py` at submission root must be self-contained. It cannot import from subpackages unless bundled as `.tar.gz`. Missing import = crash on Kaggle eval.
- **Warning sign:** Works locally, crashes on Kaggle with `ModuleNotFoundError`.
- **Fix:** Either inline all code in `main.py` (ugly but safe) OR use `tar -czf submission.tar.gz main.py agent/` as documented in `agents.md`.
- **Phase:** Phase 2 (when multi-file structure is introduced)

### 4-player vs 2-player divergence
- **What:** Optimal strategy changes significantly in 4-player mode. An agent tuned for 2-player (attack aggressively) gets flanked by third/fourth players in 4-player.
- **Warning sign:** Strong in 2v2 matches on Kaggle but poor overall Elo.
- **Fix:** Add `if len(active_players) == 4: use_cautious_expansion()` logic. Tune separately.
- **Phase:** Phase 3

---
*Pitfalls research: 2026-05-06 (inline, agents rate-limited)*
