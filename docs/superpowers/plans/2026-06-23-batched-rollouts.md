# Batched Action Rollout Search Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Upgrade `ProducerLiteRuntime.tensor_action` in `148-H_turn_one_file_batch.py` from a single-step greedy planner to a B=30, H=20 batched rollout search that returns the first move from the highest-scoring parallel universe.

**Architecture:** At step 0, `plan_lite_waves` is extended to return a read-only `CandidateTable` (pre-greedy candidate tensors). `rollout_search` initialises B copies of the game state, samples B random strategy weight vectors to build a fixed `style_score[B,C]` matrix, picks B potentially different first actions, then runs H ticks of production → combat → launch, and scores terminal states. The winning universe's step-0 action replaces the greedy payload; the greedy payload is kept as a zero-cost fallback.

**Tech Stack:** Python 3.11+, PyTorch (CPU). No new dependencies.

## Global Constraints

- All new code lives in `148-H_turn_one_file_batch.py` (single-file Kaggle submission format)
- All rollout tensor ops must be vectorised — no Python `for` loops over `B` or `P`
- `ROLLOUT_B = 30`, `ROLLOUT_H = 20`, `ARRIVALS_H = 40`, `PROD_WEIGHT = 10.0`, `NOISE_SCALE = 0.15`
- Greedy fallback (`run_turn` payload) must always be used when `rollout_search` returns `None`
- Tick order inside the rollout loop: **production → arrivals/combat → launch** (engine-accurate)
- Three masking invariants at every launch: source owned by player, ships ≥ required, arrival step < `ARRIVALS_H`

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `148-H_turn_one_file_batch.py` | Modify | All new dataclasses and rollout logic added inline |
| `tests/test_rollout_148.py` | Create | Unit + integration tests for all new code |

---

### Task 1: Add `CandidateTable`, expose it from `plan_lite_waves`, update `run_turn`

**Files:**
- Modify: `148-H_turn_one_file_batch.py:3536` (insert `CandidateTable` after `LaunchEntries`)
- Modify: `148-H_turn_one_file_batch.py:4485` (`plan_lite_waves` — return tuple)
- Modify: `148-H_turn_one_file_batch.py:4695` (`run_turn` — return triple)
- Modify: `148-H_turn_one_file_batch.py:4792` (`tensor_action` — unpack triple)
- Create: `tests/test_rollout_148.py`

**Interfaces:**
- Produces: `CandidateTable` (frozen dataclass), `plan_lite_waves` returns `tuple[LaunchEntries, CandidateTable | None]`, `run_turn` returns `tuple[dict, CandidateTable | None, PlanetMovement]`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_rollout_148.py`:

```python
import importlib.util, pathlib, pytest, torch

ROOT = pathlib.Path(__file__).parent.parent

def _load():
    spec = importlib.util.spec_from_file_location(
        "agent148", ROOT / "148-H_turn_one_file_batch.py"
    )
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m

mod = _load()


def _raw_obs(player_id=0, step=0):
    return {
        "planets": [
            [0, player_id,       20.0, 50.0, 5.0, 100.0, 3.0],
            [1, player_id,       30.0, 50.0, 5.0,  80.0, 2.0],
            [2, -1,              60.0, 50.0, 4.0,  20.0, 2.0],
            [3, 1 - player_id,   80.0, 50.0, 4.0,  60.0, 3.0],
            [4, -1,              50.0, 70.0, 3.0,  10.0, 1.0],
        ],
        "fleets": [],
        "step": step,
        "angular_velocity": 0.0,
        "episode_steps": 500,
        "remainingOverageTime": 2.0,
        "next_fleet_id": 0,
    }


def _obs_tensors(player_id=0, step=0):
    raw = _raw_obs(player_id, step)
    return mod.single_obs_to_tensor(raw, player_id=player_id)


def test_candidate_table_is_dataclass():
    # CandidateTable must exist and be frozen
    ct = mod.CandidateTable
    assert hasattr(ct, "__dataclass_fields__")


def test_plan_lite_waves_returns_tuple():
    obs_t = _obs_tensors()
    obs = mod.parse_obs(obs_t)
    player_count = 2
    config = mod._config_for(player_count)
    mem = mod.ProducerLiteMemory()
    movement = mod.ensure_planet_movement(
        obs_tensors=obs_t,
        expected_cfg=mod._movement_config(config, player_count=player_count),
        cached_movement=None,
    )
    mem.movement = movement
    cache = mod.build_distance_cache(movement, max_k=config.horizon)
    H = config.horizon
    status = movement.garrison_status(max_horizon=H)
    alive = movement.alive_by_step[: H + 1]
    result = mod.plan_lite_waves(
        movement=movement, obs=obs, obs_tensors=obs_t, cache=cache,
        garrison_status=status, prod=movement.planet_prod,
        alive_by_step=alive, config=config, player_count=player_count,
    )
    assert isinstance(result, tuple) and len(result) == 2
    entries, ct = result
    assert hasattr(entries, "source_slots")
    # ct may be None if no candidates, but on this obs should have some
    assert ct is not None
    assert isinstance(ct, mod.CandidateTable)


def test_candidate_table_shapes():
    obs_t = _obs_tensors()
    obs = mod.parse_obs(obs_t)
    player_count = 2
    config = mod._config_for(player_count)
    mem = mod.ProducerLiteMemory()
    movement = mod.ensure_planet_movement(
        obs_tensors=obs_t,
        expected_cfg=mod._movement_config(config, player_count=player_count),
        cached_movement=None,
    )
    cache = mod.build_distance_cache(movement, max_k=config.horizon)
    H = config.horizon
    status = movement.garrison_status(max_horizon=H)
    alive = movement.alive_by_step[: H + 1]
    _, ct = mod.plan_lite_waves(
        movement=movement, obs=obs, obs_tensors=obs_t, cache=cache,
        garrison_status=status, prod=movement.planet_prod,
        alive_by_step=alive, config=config, player_count=player_count,
    )
    C, P = ct.C, ct.P
    assert ct.source_slots.shape == (C,)
    assert ct.target_slots.shape == (C,)
    assert ct.angle.shape == (C,)
    assert ct.eta_ceil.shape == (C,)
    assert ct.required_ships.shape == (C,)
    assert ct.drain_ships.shape == (C,)
    assert ct.target_prod.shape == (C,)
    assert ct.base_score.shape == (C,)
    assert ct.valid.shape == (C,)
    assert ct.planet_ids.shape == (P,)
    assert ct.eta_ceil.dtype == torch.long


def test_run_turn_returns_triple():
    obs_t = _obs_tensors()
    mem = mod.ProducerLiteMemory()
    config = mod._config_for(2)
    result = mod.run_turn(obs_t, config=config, player_count=2, memory=mem)
    assert isinstance(result, tuple) and len(result) == 3
    payload, ct, movement = result
    assert "from_planet_id" in payload
    assert "counts" in payload
    # movement must be a PlanetMovement
    assert hasattr(movement, "fleet_buckets")
```

- [ ] **Step 2: Run tests — expect failures**

```
cd "C:\Users\trant\Documents\Programmation\Orbit Wars"
python -m pytest tests/test_rollout_148.py -v 2>&1 | head -40
```

Expected: `FAILED` on all four tests (CandidateTable not defined, plan_lite_waves returns LaunchEntries not tuple, run_turn returns dict not triple).

- [ ] **Step 3: Add `CandidateTable` dataclass after line 3536 (after `concat_launch_entries`)**

Insert immediately after the `concat_launch_entries` function (after line ~3557):

```python
@dataclass(frozen=True)
class CandidateTable:
    """Read-only candidate set built once at step 0 inside plan_lite_waves.

    Reused for all H rollout steps — only the validity mask changes.
    """
    source_slots:   Tensor   # [C] long  — source planet slot
    target_slots:   Tensor   # [C] long  — target planet slot
    angle:          Tensor   # [C] float — launch angle (radians, step-0 aim)
    eta_ceil:       Tensor   # [C] long  — ceil(eta); direct arrivals-buffer offset
    required_ships: Tensor   # [C] float — capture floor at step-0 ETA
    drain_ships:    Tensor   # [C] float — ships to send (safe_drain at step 0)
    target_prod:    Tensor   # [C] float — target planet production rate
    base_score:     Tensor   # [C] float — competitive_score at step 0
    valid:          Tensor   # [C] bool  — viable at step 0
    planet_ids:     Tensor   # [P] long  — planet_id per slot (for payload build)
    C: int
    P: int
```

- [ ] **Step 4: Modify `plan_lite_waves` early-return paths and add table build**

Change the two early returns (before any candidates exist) to return `(entries, None)`:

```python
    if not bool(source_mask.any()):
        return _empty_entries(device, dtype), None
```

```python
    if not bool(target_exists.any()):
        return _empty_entries(device, dtype), None
```

Then insert the `CandidateTable` build just before the `_greedy_select` call (after line 4609, where `score` is computed):

```python
    candidate_table = CandidateTable(
        source_slots=cand_src.squeeze(-1),
        target_slots=cand_tgt_slot,
        angle=cand_angle.squeeze(-1),
        eta_ceil=cand_eta.squeeze(-1).ceil().long(),
        required_ships=floor_at_arr.reshape(C),
        drain_ships=cand_send.squeeze(-1),
        target_prod=obs.prod[cand_tgt_slot.clamp(0, max(P - 1, 0))],
        base_score=score,
        valid=cand_valid,
        planet_ids=obs_tensors["planets"][:, 0].long(),
        C=C,
        P=P,
    )
```

Change the two `return` statements at the end of `plan_lite_waves`:

```python
    if not bool(config.enable_regroup):
        return wave_entries, candidate_table
    enemy_mass = cheap_enemy_pressure(obs, cache, horizon=float(K_eta), player_id=pid)
    regroup_entries = _plan_regroup(
        movement=movement, obs=obs, obs_tensors=obs_tensors, garrison_status=garrison_status,
        leftover=leftover, original_ships=obs.ships.to(dtype), pressure=enemy_mass,
        config=config, H=H,
    )
    return concat_launch_entries([wave_entries, regroup_entries]), candidate_table
```

- [ ] **Step 5: Modify `run_turn` to return triple**

Change the function signature docstring and the single `return` at the bottom of `run_turn` (line ~4755):

```python
def run_turn(obs_tensors: dict, *, config: ProducerLiteConfig, player_count: int, memory) -> tuple[dict, "CandidateTable | None", "PlanetMovement"]:
```

Change the `plan_lite_waves` call inside `run_turn` to unpack the tuple:

```python
    entries, candidate_table = plan_lite_waves(
        movement=movement, obs=obs, obs_tensors=obs_tensors, cache=cache,
        garrison_status=status, prod=movement.planet_prod,
        alive_by_step=alive_by_step, config=config, player_count=int(player_count),
    )
```

Change the final `return`:

```python
    planet_ids = obs_tensors["planets"][..., 0].long()
    return entries_to_sparse_payload(entries, planet_ids=planet_ids), candidate_table, movement
```

- [ ] **Step 6: Modify `tensor_action` to unpack the triple**

Replace the body of `ProducerLiteRuntime.tensor_action` (lines 4792–4804):

```python
    def tensor_action(self, obs_tensors: dict):
        mem = self.memory
        if bool((obs_tensors["step"] == 0).all()):
            mem.cached_player_count = None
        if mem.cached_player_count is None:
            mem.cached_player_count = largest_initial_player_count(obs_tensors)
        config = _config_for(mem.cached_player_count)
        row, _candidate_table, _movement = run_turn(
            obs_tensors, config=config,
            player_count=int(mem.cached_player_count), memory=mem,
        )
        mem.last_sparse_action_row = row
        return row
```

(The rollout call is wired in Task 6; for now we keep `_candidate_table` and `_movement` as unused locals so the greedy path stays intact.)

- [ ] **Step 7: Run tests — expect all to pass**

```
python -m pytest tests/test_rollout_148.py -v
```

Expected: all 4 tests `PASSED`.

- [ ] **Step 8: Smoke-test the agent end-to-end**

```python
# Run from project root in Python
import importlib.util, pathlib
spec = importlib.util.spec_from_file_location("a", pathlib.Path("148-H_turn_one_file_batch.py"))
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
obs = {"planets": [[0,0,20,50,5,100,3],[1,-1,80,50,4,20,2]], "fleets": [],
       "step": 0, "angular_velocity": 0.0, "episode_steps": 500,
       "remainingOverageTime": 2.0, "next_fleet_id": 0, "player": 0}
print(m.agent(obs))  # must print a list (possibly empty)
```

Expected: no exception, prints `[[from_pid, angle, ships], ...]` or `[]`.

- [ ] **Step 9: Commit**

```bash
git add 148-H_turn_one_file_batch.py tests/test_rollout_148.py
git commit -m "feat: expose CandidateTable from plan_lite_waves; run_turn returns triple"
```

---

### Task 2: Add `RolloutState` dataclass + `init_rollout_state`

**Files:**
- Modify: `148-H_turn_one_file_batch.py` (insert before `ProducerLiteMemory` at line ~4773)
- Modify: `tests/test_rollout_148.py`

**Interfaces:**
- Consumes: `CandidateTable` (Task 1), `PlanetMovement.fleet_buckets`
- Produces: `RolloutState`, `init_rollout_state(obs_tensors, movement, B, H, A, player_id) → RolloutState`, constants `ROLLOUT_B`, `ROLLOUT_H`, `ARRIVALS_H`, `PROD_WEIGHT`, `NOISE_SCALE`

- [ ] **Step 1: Write failing tests** — add to `tests/test_rollout_148.py`:

```python
def test_rollout_constants():
    assert mod.ROLLOUT_B == 30
    assert mod.ROLLOUT_H == 20
    assert mod.ARRIVALS_H == 40
    assert mod.PROD_WEIGHT == 10.0
    assert mod.NOISE_SCALE == 0.15


def test_init_rollout_state_shapes():
    obs_t = _obs_tensors()
    mem = mod.ProducerLiteMemory()
    config = mod._config_for(2)
    _, _, movement = mod.run_turn(obs_t, config=config, player_count=2, memory=mem)
    B, H, A = 4, mod.ROLLOUT_H, 2
    state = mod.init_rollout_state(obs_t, movement, B=B, H=H, A=A, player_id=0)
    P = state.P
    assert state.ships.shape  == (B, P)
    assert state.owner.shape  == (B, P)
    assert state.prod.shape   == (B, P)
    assert state.alive.shape  == (B, P)
    assert state.arrivals.shape == (B, P, mod.ARRIVALS_H, A)
    assert state.A == A
    assert state.B == B


def test_init_rollout_state_ships_match_obs():
    obs_t = _obs_tensors()
    mem = mod.ProducerLiteMemory()
    _, _, movement = mod.run_turn(obs_t, config=mod._config_for(2), player_count=2, memory=mem)
    state = mod.init_rollout_state(obs_t, movement, B=3, H=mod.ROLLOUT_H, A=2, player_id=0)
    # All B copies start with identical ship counts from the observation
    assert (state.ships[0] == state.ships[1]).all()
    assert (state.ships[0] == state.ships[2]).all()
    # Ships for planet 0 should match obs (planet 0 has 100 ships)
    planet_ships = obs_t["planets"][:, 5]
    assert (state.ships[0] - planet_ships).abs().max() < 1e-3


def test_init_rollout_state_arrivals_nonnegative():
    obs_t = _obs_tensors()
    mem = mod.ProducerLiteMemory()
    _, _, movement = mod.run_turn(obs_t, config=mod._config_for(2), player_count=2, memory=mem)
    state = mod.init_rollout_state(obs_t, movement, B=2, H=mod.ROLLOUT_H, A=2, player_id=0)
    assert (state.arrivals >= 0).all()
```

- [ ] **Step 2: Run tests — expect failures**

```
python -m pytest tests/test_rollout_148.py::test_rollout_constants tests/test_rollout_148.py::test_init_rollout_state_shapes -v
```

Expected: `FAILED` — `ROLLOUT_B` and `init_rollout_state` not defined.

- [ ] **Step 3: Add constants and `RolloutState` before `ProducerLiteMemory`**

Insert before the `class ProducerLiteMemory:` line:

```python
# ---------------------------------------------------------------------------
# Rollout search — constants and state
# ---------------------------------------------------------------------------

ROLLOUT_B:  int   = 30    # parallel universes
ROLLOUT_H:  int   = 20    # simulation horizon (steps)
ARRIVALS_H: int   = 40    # ROLLOUT_H + max candidate ETA cap (20); no overflow
PROD_WEIGHT: float = 10.0
NOISE_SCALE: float = 0.15


@dataclass
class RolloutState:
    """Mutable batched game state for B parallel universes."""
    ships:    Tensor   # [B, P] float — ships per planet
    owner:    Tensor   # [B, P] long  — owner (-1=neutral, 0..A-1=players)
    prod:     Tensor   # [B, P] float — production rate (static after init)
    alive:    Tensor   # [B, P] bool  — alive mask (static after init)
    arrivals: Tensor   # [B, P, ARRIVALS_H, A] float — player arrival buckets
    player_id: int
    A: int
    B: int
    P: int
```

- [ ] **Step 4: Add `init_rollout_state` after the `RolloutState` dataclass**

```python
def init_rollout_state(
    obs_tensors: dict,
    movement: PlanetMovement,
    B: int,
    H: int,
    A: int,
    player_id: int,
) -> RolloutState:
    planets = obs_tensors["planets"]          # [P, 7]
    P = int(planets.shape[0])
    device = planets.device
    dtype = torch.float32

    ships = planets[:, 5].to(dtype)
    owner = planets[:, 1].long()
    prod  = planets[:, 6].to(dtype)
    alive = (planets[:, 0] >= 0)

    state_ships = ships.unsqueeze(0).expand(B, P).clone()
    state_owner = owner.unsqueeze(0).expand(B, P).clone()
    state_prod  = prod.unsqueeze(0).expand(B, P)
    state_alive = alive.unsqueeze(0).expand(B, P)

    arrivals = torch.zeros(B, P, ARRIVALS_H, A, dtype=dtype, device=device)

    if movement.fleet_buckets is not None:
        fb = movement.fleet_buckets                # [P, H_mov, A_mov]
        H_mov  = int(fb.shape[1])
        A_mov  = int(fb.shape[2])
        H_copy = min(H_mov, ARRIVALS_H)
        A_copy = min(A_mov, A)
        arrivals[:, :, :H_copy, :A_copy] = (
            fb[:, :H_copy, :A_copy].to(dtype=dtype, device=device).unsqueeze(0)
        )

    return RolloutState(
        ships=state_ships,
        owner=state_owner,
        prod=state_prod,
        alive=state_alive,
        arrivals=arrivals,
        player_id=player_id,
        A=A,
        B=B,
        P=P,
    )
```

- [ ] **Step 5: Run tests — expect all to pass**

```
python -m pytest tests/test_rollout_148.py -v
```

Expected: all tests including 4 new ones `PASSED`.

- [ ] **Step 6: Commit**

```bash
git add 148-H_turn_one_file_batch.py tests/test_rollout_148.py
git commit -m "feat: add RolloutState, init_rollout_state, rollout constants"
```

---

### Task 3: Add `build_style_score` + `pick_first_actions`

**Files:**
- Modify: `148-H_turn_one_file_batch.py` (insert after `init_rollout_state`)
- Modify: `tests/test_rollout_148.py`

**Interfaces:**
- Consumes: `CandidateTable` (Task 1), `RolloutState` (Task 2)
- Produces:
  - `build_style_score(candidate_table: CandidateTable, B: int, device: torch.device) -> Tensor`  `[B, C]`
  - `pick_first_actions(state: RolloutState, candidate_table: CandidateTable, style_score: Tensor) -> tuple[Tensor, Tensor]`  `([B], [B])`

- [ ] **Step 1: Write failing tests** — add to `tests/test_rollout_148.py`:

```python
def _make_state_and_table(B=4):
    obs_t = _obs_tensors()
    mem = mod.ProducerLiteMemory()
    config = mod._config_for(2)
    _, ct, movement = mod.run_turn(obs_t, config=config, player_count=2, memory=mem)
    state = mod.init_rollout_state(obs_t, movement, B=B, H=mod.ROLLOUT_H, A=2, player_id=0)
    return state, ct, obs_t


def test_style_score_shape():
    state, ct, _ = _make_state_and_table(B=5)
    if ct is None:
        pytest.skip("no candidates on this obs")
    ss = mod.build_style_score(ct, B=5, device=state.ships.device)
    assert ss.shape == (5, ct.C)


def test_style_score_varies_across_universes():
    state, ct, _ = _make_state_and_table(B=20)
    if ct is None:
        pytest.skip("no candidates on this obs")
    ss = mod.build_style_score(ct, B=20, device=state.ships.device)
    # Different universes should not be identical (noise + w_prod variation)
    assert not (ss[0] == ss[1]).all()


def test_pick_first_actions_returns_shapes():
    state, ct, _ = _make_state_and_table(B=4)
    if ct is None:
        pytest.skip("no candidates on this obs")
    ss = mod.build_style_score(ct, B=4, device=state.ships.device)
    idx, ships = mod.pick_first_actions(state, ct, ss)
    assert idx.shape == (4,)
    assert ships.shape == (4,)


def test_pick_first_actions_ships_nonnegative():
    state, ct, _ = _make_state_and_table(B=8)
    if ct is None:
        pytest.skip("no candidates on this obs")
    ss = mod.build_style_score(ct, B=8, device=state.ships.device)
    _, ships = mod.pick_first_actions(state, ct, ss)
    assert (ships >= 0).all()


def test_pick_first_actions_deducts_source():
    state, ct, obs_t = _make_state_and_table(B=4)
    if ct is None:
        pytest.skip("no candidates on this obs")
    ships_before = state.ships.clone()
    ss = mod.build_style_score(ct, B=4, device=state.ships.device)
    _, sent = mod.pick_first_actions(state, ct, ss)
    # For universes that launched, source ships must decrease
    diff = (ships_before - state.ships).clamp(min=0)
    # At least one universe should have deducted something
    assert diff.sum() > 0


def test_pick_first_actions_only_own_source():
    """No launches from enemy or neutral planets."""
    state, ct, _ = _make_state_and_table(B=4)
    if ct is None:
        pytest.skip("no candidates on this obs")
    ss = mod.build_style_score(ct, B=4, device=state.ships.device)
    idx, ships = mod.pick_first_actions(state, ct, ss)
    for b in range(4):
        if float(ships[b]) > 0:
            src = int(ct.source_slots[int(idx[b])].item())
            assert int(state.owner[b, src].item()) == state.player_id
```

- [ ] **Step 2: Run tests — expect failures**

```
python -m pytest tests/test_rollout_148.py::test_style_score_shape tests/test_rollout_148.py::test_pick_first_actions_returns_shapes -v
```

Expected: `FAILED` — `build_style_score` and `pick_first_actions` not defined.

- [ ] **Step 3: Add `build_style_score` after `init_rollout_state`**

```python
def build_style_score(
    candidate_table: CandidateTable,
    B: int,
    device: torch.device,
) -> Tensor:
    """Fixed [B, C] score matrix: sampled once, reused for all H rollout steps."""
    base  = candidate_table.base_score.to(device=device)    # [C]
    tprod = candidate_table.target_prod.to(device=device)   # [C]
    w_prod = torch.empty(B, 1, device=device).uniform_(0.5, 2.0)
    noise_scale = float(base.std().clamp(min=1.0).item()) * NOISE_SCALE
    noise = torch.randn(B, candidate_table.C, device=device) * noise_scale
    return base.unsqueeze(0) + w_prod * tprod.unsqueeze(0) + noise   # [B, C]
```

- [ ] **Step 4: Add `pick_first_actions` after `build_style_score`**

```python
def pick_first_actions(
    state: RolloutState,
    candidate_table: CandidateTable,
    style_score: Tensor,
) -> tuple[Tensor, Tensor]:
    """Pick one first action per universe, deduct ships, write to arrivals.

    Returns (first_cand_idx [B], first_ships [B]).
    """
    B   = state.B
    pid = state.player_id
    dev = state.ships.device
    P_safe = max(state.P - 1, 0)
    src_slots = candidate_table.source_slots   # [C]

    src_ships = state.ships[:, src_slots.clamp(0, P_safe)]   # [B, C]
    src_owner = state.owner[:, src_slots.clamp(0, P_safe)]   # [B, C]

    valid0 = (
        candidate_table.valid.unsqueeze(0)
        & (src_owner == pid)
        & (src_ships >= candidate_table.required_ships.unsqueeze(0))
    )   # [B, C]

    masked = torch.where(valid0, style_score,
                         torch.full_like(style_score, float("-inf")))
    first_cand_idx = masked.argmax(dim=-1)   # [B]
    has_valid = valid0.any(dim=-1)           # [B]

    src_b  = src_slots[first_cand_idx].clamp(0, P_safe)          # [B]
    drain  = candidate_table.drain_ships[first_cand_idx]          # [B]
    avail  = state.ships[torch.arange(B, device=dev), src_b]      # [B]
    first_ships = torch.where(
        has_valid,
        torch.min(drain, avail).floor(),
        torch.zeros(B, device=dev, dtype=state.ships.dtype),
    )

    # Deduct from source (scatter_add_ is per-row; no cross-universe collision)
    state.ships.scatter_add_(
        1, src_b.unsqueeze(1),
        (-first_ships).unsqueeze(1),
    )

    # Write to arrivals
    arr_step = candidate_table.eta_ceil[first_cand_idx]   # [B]
    valid_write = has_valid & (arr_step < ARRIVALS_H) & (first_ships >= 1.0)
    if bool(valid_write.any()):
        vb  = torch.where(valid_write)[0]
        vt  = candidate_table.target_slots[first_cand_idx[valid_write]].clamp(0, state.P - 1)
        vk  = arr_step[valid_write]
        pid_t = torch.full_like(vb, pid)
        state.arrivals.index_put_((vb, vt, vk, pid_t), first_ships[valid_write], accumulate=True)

    return first_cand_idx, first_ships
```

- [ ] **Step 5: Run tests — expect all to pass**

```
python -m pytest tests/test_rollout_148.py -v
```

Expected: all tests `PASSED`.

- [ ] **Step 6: Commit**

```bash
git add 148-H_turn_one_file_batch.py tests/test_rollout_148.py
git commit -m "feat: add build_style_score and pick_first_actions"
```

---

### Task 4: Add `credit_production` + `resolve_arrivals`

**Files:**
- Modify: `148-H_turn_one_file_batch.py` (insert after `pick_first_actions`)
- Modify: `tests/test_rollout_148.py`

**Interfaces:**
- Consumes: `RolloutState` (Task 2)
- Produces:
  - `credit_production(state: RolloutState) -> None`
  - `resolve_arrivals(state: RolloutState, k: int) -> None`

- [ ] **Step 1: Write failing tests** — add to `tests/test_rollout_148.py`:

```python
def _minimal_state(B=2, P=3, A=2, player_id=0, ships=None, owner=None, prod=None):
    """Build a RolloutState directly without going through init_rollout_state."""
    dtype = torch.float32
    s = ships if ships is not None else torch.zeros(B, P, dtype=dtype)
    o = owner if owner is not None else torch.full((B, P), -1, dtype=torch.long)
    pr = prod if prod is not None else torch.zeros(B, P, dtype=dtype)
    alive = torch.ones(B, P, dtype=torch.bool)
    arrivals = torch.zeros(B, P, mod.ARRIVALS_H, A, dtype=dtype)
    return mod.RolloutState(
        ships=s, owner=o, prod=pr, alive=alive,
        arrivals=arrivals, player_id=player_id, A=A, B=B, P=P,
    )


def test_credit_production_adds_to_owned():
    state = _minimal_state(
        ships=torch.tensor([[10.0, 5.0, 20.0], [10.0, 5.0, 20.0]]),
        owner=torch.tensor([[0, -1, 1], [0, -1, 1]], dtype=torch.long),
        prod=torch.tensor([[3.0, 2.0, 4.0], [3.0, 2.0, 4.0]]),
    )
    mod.credit_production(state)
    # planet 0 (owned by player 0): +3
    assert float(state.ships[0, 0]) == pytest.approx(13.0)
    # planet 1 (neutral): unchanged
    assert float(state.ships[0, 1]) == pytest.approx(5.0)
    # planet 2 (enemy, owner=1): +4 (enemy planets also grow — accurate simulation)
    assert float(state.ships[0, 2]) == pytest.approx(24.0)


def test_credit_production_all_universes():
    state = _minimal_state(
        ships=torch.zeros(3, 3),
        owner=torch.zeros(3, 3, dtype=torch.long),
        prod=torch.ones(3, 3),
    )
    mod.credit_production(state)
    assert (state.ships == 1.0).all()


def test_resolve_arrivals_player_beats_neutral():
    """Player 0 fleet of 25 arrives at neutral planet with 20 ships."""
    B, P, A = 1, 2, 2
    state = _minimal_state(B=B, P=P, A=A,
        ships=torch.tensor([[30.0, 20.0]]),
        owner=torch.tensor([[0, -1]], dtype=torch.long),
    )
    state.arrivals[0, 1, 3, 0] = 25.0   # 25 ships of player 0 arrive at planet 1, step 3
    mod.resolve_arrivals(state, 3)
    assert int(state.owner[0, 1].item()) == 0     # player 0 now owns it
    assert float(state.ships[0, 1]) == pytest.approx(5.0)   # 25 - 20 = 5


def test_resolve_arrivals_player_loses_to_garrison():
    """Player 0 fleet of 15 arrives at enemy planet with 30 ships."""
    B, P, A = 1, 2, 2
    state = _minimal_state(B=B, P=P, A=A,
        ships=torch.tensor([[10.0, 30.0]]),
        owner=torch.tensor([[0, 1]], dtype=torch.long),
    )
    state.arrivals[0, 1, 2, 0] = 15.0   # player 0 attacks with 15, step 2
    mod.resolve_arrivals(state, 2)
    assert int(state.owner[0, 1].item()) == 1          # player 1 still owns it
    assert float(state.ships[0, 1]) == pytest.approx(15.0)  # 30 - 15 = 15


def test_resolve_arrivals_no_activity_no_change():
    state = _minimal_state(
        ships=torch.tensor([[50.0, 30.0]]),
        owner=torch.tensor([[0, 1]], dtype=torch.long),
    )
    before = state.ships.clone()
    mod.resolve_arrivals(state, 5)
    assert (state.ships == before).all()
```

- [ ] **Step 2: Run tests — expect failures**

```
python -m pytest tests/test_rollout_148.py::test_credit_production_adds_to_owned tests/test_rollout_148.py::test_resolve_arrivals_player_beats_neutral -v
```

Expected: `FAILED` — `credit_production` and `resolve_arrivals` not defined.

- [ ] **Step 3: Add `credit_production`**

Insert after `pick_first_actions`:

```python
def credit_production(state: RolloutState) -> None:
    """Credit production to all owned planets (all players, for simulation accuracy)."""
    own_any = (state.owner >= 0) & state.alive   # [B, P]
    state.ships = state.ships + state.prod * own_any.to(state.prod.dtype)
```

- [ ] **Step 4: Add `resolve_arrivals`**

```python
def resolve_arrivals(state: RolloutState, k: int) -> None:
    """Apply top1−top2 combat for all fleets arriving at absolute step k."""
    A   = state.A
    arriving = state.arrivals[:, :, k, :]         # [B, P, A]
    has_activity = arriving.sum(dim=-1) > 0        # [B, P]
    if not bool(has_activity.any()):
        return

    # Fold owned garrison into the owner's player slot
    owned_mask = (state.owner >= 0) & state.alive  # [B, P]
    owner_safe = state.owner.clamp(min=0)           # [B, P] — neutral clamped to 0 temporarily
    garrison   = state.ships * owned_mask.to(state.ships.dtype)

    combatants = arriving.clone()
    combatants.scatter_add_(2, owner_safe.unsqueeze(-1), garrison.unsqueeze(-1))
    # Undo the false neutral-as-slot-0 contribution
    combatants[:, :, 0] = combatants[:, :, 0] - garrison * (~owned_mask).to(garrison.dtype)

    # Append neutral garrison as slot A (not stored in arrivals; derived from state.ships)
    neutral_garrison = state.ships * (state.owner == -1).to(state.ships.dtype)
    combatants = torch.cat([combatants, neutral_garrison.unsqueeze(-1)], dim=-1)  # [B, P, A+1]

    sorted_ships, sorted_owners = combatants.sort(dim=-1, descending=True)
    winner_ships = (sorted_ships[:, :, 0] - sorted_ships[:, :, 1]).clamp(min=0)
    winner_owner_raw = sorted_owners[:, :, 0]
    winner_owner = torch.where(
        winner_owner_raw == A,
        torch.full_like(winner_owner_raw, -1),
        winner_owner_raw,
    ).long()   # remap neutral slot A back to -1

    update = has_activity & state.alive
    state.ships = torch.where(update, winner_ships, state.ships)
    state.owner = torch.where(update & (winner_ships > 0), winner_owner, state.owner)
    state.owner = torch.where(
        update & (winner_ships == 0),
        torch.full_like(state.owner, -1), state.owner,
    )
```

- [ ] **Step 5: Run tests — expect all to pass**

```
python -m pytest tests/test_rollout_148.py -v
```

Expected: all tests `PASSED`.

- [ ] **Step 6: Commit**

```bash
git add 148-H_turn_one_file_batch.py tests/test_rollout_148.py
git commit -m "feat: add credit_production and resolve_arrivals for rollout loop"
```

---

### Task 5: Add `apply_best_launch` + `terminal_score` + `rollout_search`

**Files:**
- Modify: `148-H_turn_one_file_batch.py` (insert after `resolve_arrivals`)
- Modify: `tests/test_rollout_148.py`

**Interfaces:**
- Consumes: All prior rollout primitives
- Produces:
  - `apply_best_launch(state, candidate_table, style_score, k) -> None`
  - `terminal_score(state, prod_weight=PROD_WEIGHT) -> Tensor`  `[B]`
  - `rollout_search(obs_tensors, movement, candidate_table, player_id, A, B, H) -> dict | None`

- [ ] **Step 1: Write failing tests** — add to `tests/test_rollout_148.py`:

```python
def test_apply_best_launch_no_launch_from_enemy():
    """When player 0 owns no planets, apply_best_launch must not deduct any ships."""
    state, ct, _ = _make_state_and_table(B=4)
    if ct is None:
        pytest.skip("no candidates")
    # Override: player 0 owns nothing
    state.owner[:] = 1
    ships_before = state.ships.clone()
    ss = mod.build_style_score(ct, B=4, device=state.ships.device)
    mod.apply_best_launch(state, ct, ss, k=1)
    # Ships must be unchanged (or only decrease — never negative)
    assert (state.ships >= 0).all()
    assert (state.ships == ships_before).all()


def test_apply_best_launch_no_overflow():
    """Arrival step must never exceed ARRIVALS_H - 1."""
    state, ct, _ = _make_state_and_table(B=4)
    if ct is None:
        pytest.skip("no candidates")
    ss = mod.build_style_score(ct, B=4, device=state.ships.device)
    # Use k = ARRIVALS_H - 1 (would overflow if unchecked)
    k = mod.ARRIVALS_H - 1
    mod.apply_best_launch(state, ct, ss, k=k)
    # If it didn't crash, the overflow guard worked
    assert (state.ships >= 0).all()


def test_terminal_score_shape():
    state, ct, _ = _make_state_and_table(B=6)
    scores = mod.terminal_score(state)
    assert scores.shape == (6,)


def test_terminal_score_only_counts_own_planets():
    state = _minimal_state(
        B=1, P=3, A=2,
        ships=torch.tensor([[100.0, 50.0, 80.0]]),
        owner=torch.tensor([[0, 1, -1]], dtype=torch.long),
        prod=torch.tensor([[3.0, 2.0, 1.0]]),
    )
    score = mod.terminal_score(state, prod_weight=10.0)
    # Only planet 0 (owned by player 0): ships=100 + prod=3*10 = 130
    assert float(score[0]) == pytest.approx(130.0)


def test_rollout_search_returns_valid_payload_or_none():
    obs_t = _obs_tensors()
    mem = mod.ProducerLiteMemory()
    config = mod._config_for(2)
    greedy, ct, movement = mod.run_turn(obs_t, config=config, player_count=2, memory=mem)
    result = mod.rollout_search(
        obs_tensors=obs_t, movement=movement, candidate_table=ct,
        player_id=0, A=2, B=mod.ROLLOUT_B, H=mod.ROLLOUT_H,
    )
    if result is None:
        # None is valid (no valid first action found)
        return
    assert "from_planet_id" in result
    assert "angle" in result
    assert "num_ships" in result
    assert "counts" in result
    # counts must be 1
    assert int(result["counts"].item()) == 1
    # num_ships must be >= 1
    assert float(result["num_ships"][0].item()) >= 1.0


def test_rollout_search_returns_none_for_none_table():
    obs_t = _obs_tensors()
    mem = mod.ProducerLiteMemory()
    _, _, movement = mod.run_turn(obs_t, config=mod._config_for(2), player_count=2, memory=mem)
    result = mod.rollout_search(
        obs_tensors=obs_t, movement=movement, candidate_table=None,
        player_id=0, A=2, B=10, H=5,
    )
    assert result is None
```

- [ ] **Step 2: Run tests — expect failures**

```
python -m pytest tests/test_rollout_148.py::test_terminal_score_shape tests/test_rollout_148.py::test_rollout_search_returns_valid_payload_or_none -v
```

Expected: `FAILED` — functions not defined.

- [ ] **Step 3: Add `apply_best_launch`**

```python
def apply_best_launch(
    state: RolloutState,
    candidate_table: CandidateTable,
    style_score: Tensor,
    k: int,
) -> None:
    """Pick one launch per universe using pre-computed style_score, apply it."""
    B   = state.B
    pid = state.player_id
    dev = state.ships.device
    P_safe = max(state.P - 1, 0)
    src_slots = candidate_table.source_slots   # [C]

    src_ships = state.ships[:, src_slots.clamp(0, P_safe)]   # [B, C]
    src_owner = state.owner[:, src_slots.clamp(0, P_safe)]   # [B, C]

    valid_now = (
        candidate_table.valid.unsqueeze(0)
        & (src_owner == pid)
        & (src_ships >= candidate_table.required_ships.unsqueeze(0))
    )   # [B, C]

    masked = torch.where(valid_now, style_score,
                         torch.full_like(style_score, float("-inf")))
    best_c    = masked.argmax(dim=-1)         # [B]
    has_valid = valid_now.any(dim=-1)          # [B]

    src_b  = src_slots[best_c].clamp(0, P_safe)              # [B]
    avail  = state.ships[torch.arange(B, device=dev), src_b]  # [B]
    drain  = candidate_table.drain_ships[best_c]              # [B]
    send = torch.where(
        has_valid,
        torch.min(drain, avail).floor(),
        torch.zeros(B, device=dev, dtype=state.ships.dtype),
    )

    arr_step    = k + candidate_table.eta_ceil[best_c]        # [B]
    valid_write = has_valid & (arr_step < ARRIVALS_H) & (send >= 1.0)

    send_deduct = torch.where(valid_write, send, torch.zeros_like(send))
    state.ships.scatter_add_(1, src_b.unsqueeze(1), (-send_deduct).unsqueeze(1))

    if bool(valid_write.any()):
        vb  = torch.where(valid_write)[0]
        vt  = candidate_table.target_slots[best_c[valid_write]].clamp(0, state.P - 1)
        vk  = arr_step[valid_write]
        pid_t = torch.full_like(vb, pid)
        state.arrivals.index_put_((vb, vt, vk, pid_t), send[valid_write], accumulate=True)
```

- [ ] **Step 4: Add `terminal_score`**

```python
def terminal_score(state: RolloutState, prod_weight: float = PROD_WEIGHT) -> Tensor:
    """Economic score for each universe at the end of the rollout. [B]"""
    own   = (state.owner == state.player_id) & state.alive   # [B, P]
    own_f = own.to(state.ships.dtype)
    return (state.ships * own_f).sum(-1) + (state.prod * own_f).sum(-1) * prod_weight
```

- [ ] **Step 5: Add `rollout_search`**

```python
def rollout_search(
    obs_tensors: dict,
    movement: PlanetMovement,
    candidate_table: "CandidateTable | None",
    player_id: int,
    A: int,
    B: int = ROLLOUT_B,
    H: int = ROLLOUT_H,
) -> "dict | None":
    """Run B-universe H-step rollout; return step-0 action of best universe.

    Returns None when no valid first action exists (caller uses greedy fallback).
    """
    if candidate_table is None or candidate_table.C == 0:
        return None

    device = obs_tensors["planets"].device
    state  = init_rollout_state(obs_tensors, movement, B=B, H=H, A=A, player_id=player_id)
    style_score = build_style_score(candidate_table, B=B, device=device)

    first_cand_idx, first_ships = pick_first_actions(state, candidate_table, style_score)

    for k in range(1, H + 1):
        credit_production(state)
        resolve_arrivals(state, k)
        apply_best_launch(state, candidate_table, style_score, k)

    scores = terminal_score(state)   # [B]

    # Only select universes where a valid first action was taken
    valid_b = first_ships >= 1.0     # [B]
    if not bool(valid_b.any()):
        return None
    scores_masked = torch.where(valid_b, scores,
                                torch.full_like(scores, float("-inf")))
    best_b = int(scores_masked.argmax().item())

    c   = int(first_cand_idx[best_b].item())
    src = int(candidate_table.source_slots[c].item())
    ships_val = float(first_ships[best_b].item())

    return {
        "from_planet_id": candidate_table.planet_ids[src].unsqueeze(0).to(torch.int32),
        "angle":          candidate_table.angle[c].unsqueeze(0).to(torch.float32),
        "num_ships":      torch.tensor([round(ships_val)], dtype=torch.float32, device=device),
        "counts":         torch.tensor(1, dtype=torch.int32, device=device),
    }
```

- [ ] **Step 6: Run tests — expect all to pass**

```
python -m pytest tests/test_rollout_148.py -v
```

Expected: all tests `PASSED`.

- [ ] **Step 7: Commit**

```bash
git add 148-H_turn_one_file_batch.py tests/test_rollout_148.py
git commit -m "feat: add apply_best_launch, terminal_score, rollout_search"
```

---

### Task 6: Wire `rollout_search` into `tensor_action` + integration tests

**Files:**
- Modify: `148-H_turn_one_file_batch.py:4792` (`tensor_action`)
- Modify: `tests/test_rollout_148.py`

**Interfaces:**
- Consumes: `run_turn` triple (Task 1), `rollout_search` (Task 5)
- Produces: `tensor_action` returns rollout payload when possible, greedy payload otherwise; `agent()` entry point unchanged

- [ ] **Step 1: Write failing tests** — add to `tests/test_rollout_148.py`:

```python
def test_tensor_action_output_format():
    """tensor_action must return the dict format sparse_action_row_to_moves expects."""
    obs_t = _obs_tensors()
    runtime = mod.ProducerLiteRuntime()
    with torch.no_grad():
        result = runtime.tensor_action(obs_t)
    assert "from_planet_id" in result
    assert "angle" in result
    assert "num_ships" in result
    assert "counts" in result


def test_agent_end_to_end_no_crash():
    raw = _raw_obs()
    raw["player"] = 0
    moves = mod.agent(raw)
    assert isinstance(moves, list)
    for move in moves:
        assert len(move) == 3   # [from_planet_id, angle, num_ships]


def test_agent_returns_valid_planet():
    """from_planet_id in the move list must be an owned planet."""
    raw = _raw_obs()
    raw["player"] = 0
    moves = mod.agent(raw)
    owned_ids = {p[0] for p in raw["planets"] if p[1] == 0}
    for move in moves:
        assert int(move[0]) in owned_ids, f"Planet {move[0]} not owned by player 0"


def test_tensor_action_fallback_on_step0():
    """On step 0 with no prior movement cache, must not crash."""
    obs_t = _obs_tensors(step=0)
    runtime = mod.ProducerLiteRuntime()
    with torch.no_grad():
        result = runtime.tensor_action(obs_t)
    assert int(result["counts"].item()) >= 0


def test_greedy_fallback_used_when_rollout_none(monkeypatch):
    """If rollout_search returns None, the greedy payload is used."""
    original = mod.rollout_search
    mod.rollout_search = lambda **kwargs: None  # force fallback
    try:
        obs_t = _obs_tensors()
        runtime = mod.ProducerLiteRuntime()
        with torch.no_grad():
            result = runtime.tensor_action(obs_t)
        assert "counts" in result
    finally:
        mod.rollout_search = original
```

- [ ] **Step 2: Run tests — expect `test_greedy_fallback_used_when_rollout_none` and `test_tensor_action_fallback_on_step0` to fail**

```
python -m pytest tests/test_rollout_148.py::test_greedy_fallback_used_when_rollout_none tests/test_rollout_148.py::test_tensor_action_fallback_on_step0 -v
```

Expected: `FAILED` — `tensor_action` still uses old signature (doesn't call `rollout_search`).

- [ ] **Step 3: Replace `tensor_action` body**

Replace lines 4792–4804 in `148-H_turn_one_file_batch.py`:

```python
    def tensor_action(self, obs_tensors: dict):
        mem = self.memory
        if bool((obs_tensors["step"] == 0).all()):
            mem.cached_player_count = None
        if mem.cached_player_count is None:
            mem.cached_player_count = largest_initial_player_count(obs_tensors)
        config = _config_for(mem.cached_player_count)
        greedy_payload, candidate_table, movement = run_turn(
            obs_tensors, config=config,
            player_count=int(mem.cached_player_count), memory=mem,
        )
        player_id = int(obs_tensors["player"].flatten()[0].item())
        rollout_payload = rollout_search(
            obs_tensors=obs_tensors,
            movement=movement,
            candidate_table=candidate_table,
            player_id=player_id,
            A=int(mem.cached_player_count),
            B=ROLLOUT_B,
            H=ROLLOUT_H,
        )
        row = rollout_payload if rollout_payload is not None else greedy_payload
        mem.last_sparse_action_row = row
        return row
```

- [ ] **Step 4: Run all tests**

```
python -m pytest tests/test_rollout_148.py -v
```

Expected: all tests `PASSED`.

- [ ] **Step 5: Timing smoke test — verify rollout completes well within budget**

```python
# Run from project root
import importlib.util, pathlib, time, torch
spec = importlib.util.spec_from_file_location("a", pathlib.Path("148-H_turn_one_file_batch.py"))
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
raw = {
    "planets": [
        [0, 0, 20.0, 50.0, 5.0, 100.0, 3.0],
        [1, 0, 30.0, 50.0, 5.0,  80.0, 2.0],
        [2,-1, 60.0, 50.0, 4.0,  20.0, 2.0],
        [3, 1, 80.0, 50.0, 4.0,  60.0, 3.0],
        [4,-1, 50.0, 70.0, 3.0,  10.0, 1.0],
    ],
    "fleets": [], "step": 5, "angular_velocity": 0.03,
    "episode_steps": 500, "remainingOverageTime": 2.0,
    "next_fleet_id": 0, "player": 0,
}
obs_t = m.single_obs_to_tensor(raw, player_id=0)
runtime = m.ProducerLiteRuntime()
# warm-up
with torch.no_grad(): runtime.tensor_action(obs_t)
# time 5 calls
t0 = time.perf_counter()
for _ in range(5):
    with torch.no_grad(): runtime.tensor_action(obs_t)
elapsed = (time.perf_counter() - t0) / 5
print(f"avg per turn: {elapsed:.3f}s")
assert elapsed < 0.95, f"Too slow: {elapsed:.3f}s (budget 1.0s)"
```

Expected: prints `avg per turn: X.XXXs` with `X < 0.95`. If too slow, reduce `ROLLOUT_B` (try 15 or 10).

- [ ] **Step 6: Final commit**

```bash
git add 148-H_turn_one_file_batch.py tests/test_rollout_148.py
git commit -m "feat: wire rollout_search into tensor_action with greedy fallback"
```
