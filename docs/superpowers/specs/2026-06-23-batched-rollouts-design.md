# Batched Action Rollout Search — Design Spec
**Date:** 2026-06-23  
**File:** `148-H_turn_one_file_batch.py`  
**Status:** Approved, ready for implementation

---

## Problem Statement

The current agent runs its full planning pipeline in ~0.1 s against a 1.0 s time budget, leaving ~0.9 s unused. This spec upgrades `ProducerLiteRuntime.tensor_action` from a single-step greedy planner into a **Batched Action Rollout** search that uses the spare time to evaluate B parallel futures and return the first move from the best-scoring universe.

---

## Chosen Approach

**Approach B — True H-step batched rollout with pre-scored candidate table reuse.**

- B = 30 parallel universes (tunable; CPU-safe starting point)
- H = 20 simulation steps
- Enemy assumption: **frozen** — no new enemy launches during rollout. Existing enemy in-flight fleets still advance and resolve normally.
- Per-step planner: **pre-scored candidate table reuse** — score all (source, target) candidates once at step 0, compute a fixed `style_score[B, C]` matrix, re-mask it each step as ship counts change, pick argmax per universe.
- Fallback: if no valid rollout first-action exists, return the greedy `LaunchEntries` unchanged.

---

## Constants

```python
ROLLOUT_B:    int   = 30     # parallel universes
ROLLOUT_H:    int   = 20     # simulation horizon (steps)
ARRIVALS_H:   int   = ROLLOUT_H + 20   # = 40; arrival buffer depth
                             # 20 = max config.horizon (candidate ETA cap)
                             # guarantees no overflow: max k + eta = 19 + 20 < 40
PROD_WEIGHT:  float = 10.0   # terminal score: ships_own + prod_own * PROD_WEIGHT
NOISE_SCALE:  float = 0.15   # style_score noise = NOISE_SCALE * base_score.std()
```

---

## New Data Structures

### `CandidateTable` (read-only, built once at step 0)

```python
@dataclass(frozen=True)
class CandidateTable:
    source_slots:   Tensor   # [C] long  — source planet slot
    target_slots:   Tensor   # [C] long  — target planet slot
    angle:          Tensor   # [C] float — launch angle (radians, step-0 aim)
    eta_ceil:       Tensor   # [C] long  — ceil(eta); direct arrivals index offset
    required_ships: Tensor   # [C] float — capture floor at step-0 ETA
    drain_ships:    Tensor   # [C] float — ships to send (safe_drain at step 0)
    target_prod:    Tensor   # [C] float — target planet production (for rescore)
    base_score:     Tensor   # [C] float — competitive_score value at step 0
    valid:          Tensor   # [C] bool  — viable at step 0
    planet_ids:     Tensor   # [P] long  — planet_id per slot (for payload build)
    C: int
    P: int
```

ETAs and angles are fixed from step 0. Orbital drift over 20 steps is ≤ 5° at typical angular velocities — acceptable approximation.

### `RolloutState` (mutable, lives inside `rollout_search`)

```python
@dataclass
class RolloutState:
    ships:    Tensor   # [B, P] float — ships per planet (mutated each step)
    owner:    Tensor   # [B, P] long  — owner (-1=neutral, 0..A-1=players)
    prod:     Tensor   # [B, P] float — production rate (static)
    alive:    Tensor   # [B, P] bool  — alive mask (static)
    arrivals: Tensor   # [B, P, ARRIVALS_H, A] float — player arrival buckets
                       # Neutral garrison is NOT stored here; it is derived
                       # dynamically from state.ships during resolve_arrivals.
    player_id: int
    A: int             # player count (2 or 4)
    B: int
    P: int
```

`arrivals[b, p, k, a]` = ships from player `a` arriving at planet slot `p` at absolute simulation step `k`.

---

## Output Contract

`rollout_search` returns the **exact dict** format that `entries_to_sparse_payload` produces:

```python
{
    "from_planet_id": Tensor,  # [1] long
    "angle":          Tensor,  # [1] float
    "num_ships":      Tensor,  # [1] float — rounded, clamped ≥ 1
    "counts":         Tensor,  # [] long scalar = 1
}
```

No new types are introduced. `sparse_action_row_to_moves` in `agent()` receives either this dict or the greedy fallback dict — both are the same format.

---

## Architecture

```
tensor_action(obs_tensors)
│
├── [existing, unchanged]
│   parse_obs → PlanetMovement → build_distance_cache → garrison_status
│
├── [modified] plan_lite_waves_with_table(...)
│       Exposes pre-greedy candidate tensors as CandidateTable second return value.
│       LaunchEntries (greedy first action) kept as the zero-cost fallback.
│
└── [new] rollout_search(obs_tensors, movement, candidate_table, ...) → dict
    │
    ├── init_rollout_state(...)       → RolloutState[B, P]
    ├── build_style_score(...)        → style_score[B, C]  (computed ONCE)
    ├── pick_first_actions(...)       → first_cand_idx[B], first_ships[B]
    │
    └── for k in 1..H:
        ├── credit_production(state)
        ├── resolve_arrivals(state, k)
        └── apply_best_launch(state, candidate_table, style_score, k)
    │
    ├── terminal_score(state)         → scores[B]
    └── return payload for first_actions[argmax(scores)]
        or greedy_fallback if no valid rollout action exists
```

---

## Initialization (`init_rollout_state`)

```
planets = obs_tensors["planets"]          # [P, 7]
ships   = planets[:, 5]                   # [P]
owner   = planets[:, 1].long()            # [P]
prod    = planets[:, 6]                   # [P]
alive   = planets[:, 0] >= 0             # [P]

state.ships = ships.unsqueeze(0).expand(B, -1).clone()   # [B, P]
state.owner = owner.unsqueeze(0).expand(B, -1).clone()   # [B, P]
state.prod  = prod.unsqueeze(0).expand(B, -1)            # static view
state.alive = alive.unsqueeze(0).expand(B, -1)           # static view

# Arrivals buffer: A player slots (neutral garrison held in state.ships, not here)
state.arrivals = zeros(B, P, ARRIVALS_H, A)

# Pre-bucket in-flight fleets from PlanetMovement cache (same across all B)
# movement.fleet_buckets contains own prior-turn + enemy in-flight fleets
# (does NOT include this turn's new greedy planned launch, which runs after)
if movement.fleet_buckets is not None:          # [P, H_mov, A]
    H_copy = min(H_mov, ARRIVALS_H)
    state.arrivals[:, :, :H_copy, :A] = (
        movement.fleet_buckets[:, :H_copy, :].unsqueeze(0)
    )
```

Planet ships in the observation already reflect all prior-turn deductions (in-flight fleets have left their sources), so no double-counting occurs.

---

## Step-0 Diversity (`build_style_score` + `pick_first_actions`)

**Style score matrix — computed once, reused every step:**

```
w_prod = Uniform(0.5, 2.0)[B, 1]    # production emphasis per universe
noise  = randn(B, C) * NOISE_SCALE * base_score.std().clamp(min=1.0)

style_score[B, C] = (
    candidate_table.base_score            # [C] step-0 competitive_score
  + w_prod * candidate_table.target_prod  # [B, C] production bias
  + noise                                 # [B, C] exploration
)
```

**First-action selection:**

```
# Validity at step 0
valid0 = (
    candidate_table.valid                                    # [C]
  & (state.owner[:, source_slots] == player_id)             # [B, C]
  & (state.ships[:, source_slots] >= required_ships)        # [B, C]
)
masked         = where(valid0, style_score, -inf)
first_cand_idx = masked.argmax(dim=-1)    # [B]
has_valid_b    = valid0.any(dim=-1)       # [B]

# Ships to send (capped at available)
src         = source_slots[first_cand_idx]               # [B]
first_ships = where(has_valid_b,
                    min(drain_ships[first_cand_idx],
                        state.ships[arange(B), src]).floor(),
                    0.0)                                 # [B]

# Deduct from source
state.ships.scatter_add_(1, src.unsqueeze(1), (-first_ships).unsqueeze(1))

# Write to arrivals (slot = player_id, step = eta_ceil[c])
arr_step    = eta_ceil[first_cand_idx]            # [B]
valid_write = has_valid_b & (arr_step < ARRIVALS_H)
# index_put_ over sparse (b, tgt[b], arr_step[b], player_id) tuples
```

---

## Rollout Loop Body (steps 1 → H)

**Tick order: production → combat → launch** (confirmed from `PlanetGarrisonStatus` docstring: *"pre_combat_ships … after production has been credited but before same-step arrivals are applied"*).

### `credit_production`

```
own_any = (state.owner >= 0) & state.alive     # [B, P] all owned planets
state.ships += state.prod * own_any.float()
```

Enemy planets grow too — required for accurate capture-cost modelling.

### `resolve_arrivals(state, k)`

Unified `[A+1]` combatant axis (slot `A` = neutral garrison) handles owned and neutral planets with a single sort:

```
arriving = state.arrivals[:, :, k, :A]     # [B, P, A] — player arrivals only

# Fold owned garrison into the owner's player slot
owned_mask  = (state.owner >= 0) & state.alive
owner_safe  = state.owner.clamp(min=0)
garrison    = state.ships * owned_mask.float()
combatants  = arriving.clone()
combatants.scatter_add_(2, owner_safe.unsqueeze(-1), garrison.unsqueeze(-1))
combatants[:, :, 0] -= garrison * (~owned_mask).float()   # undo neutral contamination

# Append neutral garrison as slot A
neutral_garrison = state.ships * (state.owner == -1).float()
combatants = cat([combatants, neutral_garrison.unsqueeze(-1)], dim=-1)  # [B, P, A+1]

# Sort, top1−top2
sorted_ships, sorted_owners = combatants.sort(dim=-1, descending=True)
winner_ships = (sorted_ships[:,:,0] - sorted_ships[:,:,1]).clamp(min=0)
winner_owner = where(sorted_owners[:,:,0] == A,
                     full(-1), sorted_owners[:,:,0]).long()  # remap A→-1

# Apply where player fleets arrived
has_activity = arriving.sum(-1) > 0        # [B, P]
update       = has_activity & state.alive

state.ships = where(update, winner_ships, state.ships)
state.owner = where(update & (winner_ships > 0), winner_owner, state.owner)
state.owner = where(update & (winner_ships == 0), full(-1), state.owner)
```

### `apply_best_launch(state, candidate_table, style_score, k)`

```
# Re-mask the pre-computed style_score
src_ships = state.ships[:, source_slots]       # [B, C]
src_owner = state.owner[:, source_slots]       # [B, C]

valid_now = (
    candidate_table.valid
  & (src_owner == player_id)                   # [B, C]
  & (src_ships >= required_ships)              # [B, C]
)
masked    = where(valid_now, style_score, -inf)
best_c    = masked.argmax(dim=-1)              # [B]
has_valid = valid_now.any(dim=-1)              # [B]

src_b    = source_slots[best_c]                # [B]
avail    = state.ships[arange(B), src_b]       # [B]
drain    = drain_ships[best_c]                 # [B]
send     = where(has_valid, min(drain, avail).floor(), 0.0)

# Overflow guard
arr_step    = k + eta_ceil[best_c]             # [B]
valid_write = has_valid & (arr_step < ARRIVALS_H) & (send >= 1)

# Deduct
state.ships.scatter_add_(1, src_b.unsqueeze(1),
    (-where(valid_write, send, 0)).unsqueeze(1))

# Write arrivals (index_put_ over sparse valid subset)
if valid_write.any():
    vb  = arange(B)[valid_write]
    vt  = target_slots[best_c[valid_write]]
    vk  = arr_step[valid_write]
    pid = full_like(vb, player_id)
    state.arrivals.index_put_((vb, vt, vk, pid), send[valid_write], accumulate=True)
```

---

## Terminal Scoring + Action Selection

```
def terminal_score(state) → [B]:
    own = (state.owner == player_id) & state.alive    # [B, P]
    return (state.ships * own).sum(-1) \
         + (state.prod  * own).sum(-1) * PROD_WEIGHT

scores = terminal_score(state)
best_b = scores.argmax()
c      = first_cand_idx[best_b]
src    = candidate_table.source_slots[c]

# Fallback: no universe produced a valid first action
if first_ships.max() < 1:
    return greedy_fallback_payload

return {
    "from_planet_id": candidate_table.planet_ids[src].unsqueeze(0),
    "angle":          candidate_table.angle[c].unsqueeze(0),
    "num_ships":      first_ships[best_b].round().clamp(min=1).unsqueeze(0),
    "counts":         tensor(1, device=device),
}
```

---

## Required Change to Existing Code

`plan_lite_waves` must expose its pre-greedy local tensors as a `CandidateTable` second return value. The tensors are already computed before `_greedy_select` is called — this is a zero-cost extraction:

```python
# Inside plan_lite_waves, before _greedy_select:
candidate_table = CandidateTable(
    source_slots   = cand_src.squeeze(-1),          # [C]
    target_slots   = cand_tgt_slot,                 # [C]
    angle          = cand_angle.squeeze(-1),         # [C]
    eta_ceil       = cand_eta.squeeze(-1).ceil().long(),
    required_ships = floor_at_arr.reshape(C),        # [C]
    drain_ships    = cand_send.squeeze(-1),          # [C]
    target_prod    = obs.prod[target_idx][cand_tgt_short],  # [C]
    base_score     = score,                         # [C]
    valid          = cand_valid,                    # [C]
    planet_ids     = obs_tensors["planets"][:,0].long(),
    C=C, P=obs.P,
)
```

`run_turn` is renamed `run_turn_with_candidates` and returns `(payload, candidate_table, movement)`. `tensor_action` uses the candidate table for the rollout and falls back to `payload` otherwise.

---

## Masking Invariants

Three invariants must hold at every `apply_best_launch` call to prevent negative ship counts or out-of-bounds writes:

1. `src_owner[b, c] == player_id` — only launch from owned planets
2. `src_ships[b, c] >= required_ships[c]` — sufficient ships to send
3. `k + eta_ceil[c] < ARRIVALS_H` — arrival lands within the buffer (always true given `ARRIVALS_H = H + 20` and `eta_ceil ≤ 20`)

All three are enforced in `valid_write` before `scatter_add_` and `index_put_`.

---

## Out of Scope

- Multi-wave per rollout step (step 0 sends one launch per universe; steps 1–H also send one)
- Combining rollout winner with greedy secondary waves (future enhancement)
- GPU: design is device-agnostic; `B` should be tuned per device
- Adaptive depth based on wall-clock (`time.perf_counter`) — add after baseline works
