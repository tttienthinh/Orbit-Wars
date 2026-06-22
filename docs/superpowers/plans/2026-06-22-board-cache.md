# Board Cache Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create `138-BoardCache.py` — a drop-in replacement for `137-Polars_minimax_fast2.py` that eliminates all `copy.deepcopy` calls from the minimax inner loop by using a `Board` class with incremental Polars DataFrame recomputation and a pure-Python dict evaluation path.

**Architecture:** `Board` maintains five Polars DataFrames (planet nature/pos/ships, real fleets, simulated fleets). Each turn: slide the position window forward, sync in-flight fleets, and fully recompute the base ship state. During minimax: per c0 candidate, recompute only the ~10 dirty planet-steps using `_recompute_from_sim`; per c5 candidate (inner loop), apply the fleet effect to a Python dict copy and evaluate with `_evaluate_dict` — zero Polars allocation in the hot path.

**Tech Stack:** Python 3.11+, Polars ≥ 0.20, math, copy (no deepcopy used in hot path)

## Global Constraints

- Output file is `138-BoardCache.py` in project root — self-contained, no imports from other numbered files
- `agent(obs)` must return a `list` of `[id_src, angle, ships]` 3-element lists
- Turn budget: < 1 second for typical game (P≈15, N≈6, K≈5)
- Combat resolution must be byte-for-byte equivalent to `1_29_3.py` interpreter
- `_02_get_all_opportunities` and `_03_filter_collision` are copied verbatim from `137` — do not modify their signatures or logic
- `df_planete_nature` has columns `id, radius, production, nature` (4 cols; spec listed 3 but radius is required by `_02`)
- Candidate tuples for c5 have 5 elements: `[id_src, id_tgt, step_tgt, angle, ships_sent]`
- `ships_horizon` dict values are 3-tuples `(ships, owner, production)`

---

### Task 1: Scaffold — copy unchanged components

**Files:**
- Create: `138-BoardCache.py`

**Interfaces:**
- Produces: `GameConfig`, `PhysicsEngine`, `CENTER`, `SUN_RADIUS`, `ROTATION_RADIUS_LIMIT`, `BOARD_SIZE`, `MAX_NB_STEP`, `evaluate()`, `_02_get_all_opportunities()`, `_03_filter_collision()` — all identical to `137`

- [ ] **Step 1: Create the file with shared constants and physics**

```python
import math
import polars as pl


# ── Configuration ─────────────────────────────────────────────────────────────
class GameConfig:
    CENTER = 50.0
    SUN_RADIUS = 10.0
    ROTATION_RADIUS_LIMIT = 50.0
    MAX_SPEED = 6.0
    NB_STEPS_SIM = 10
    PLANET_MARGIN = 0.1
    PLANET_MOVEMENT_SLACK = 3.0


# ── Physics helpers ───────────────────────────────────────────────────────────
class PhysicsEngine:
    @staticmethod
    def distance(p1, p2):
        return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

    @staticmethod
    def point_to_segment_distance(p, v, w):
        l2 = (v[0] - w[0]) ** 2 + (v[1] - w[1]) ** 2
        if l2 == 0.0:
            return PhysicsEngine.distance(p, v)
        t = max(
            0, min(1, ((p[0] - v[0]) * (w[0] - v[0]) + (p[1] - v[1]) * (w[1] - v[1])) / l2)
        )
        projection = (v[0] + t * (w[0] - v[0]), v[1] + t * (w[1] - v[1]))
        return PhysicsEngine.distance(p, projection)

    @staticmethod
    def swept_pair_hit(A, B, P0, P1, r):
        d0x, d0y = A[0] - P0[0], A[1] - P0[1]
        dvx = (B[0] - A[0]) - (P1[0] - P0[0])
        dvy = (B[1] - A[1]) - (P1[1] - P0[1])
        a = dvx * dvx + dvy * dvy
        b = 2.0 * (d0x * dvx + d0y * dvy)
        c = d0x * d0x + d0y * d0y - r * r
        if a < 1e-12:
            return c <= 0.0
        disc = b * b - 4.0 * a * c
        if disc < 0.0:
            return False
        sq = math.sqrt(disc)
        t1 = (-b - sq) / (2.0 * a)
        t2 = (-b + sq) / (2.0 * a)
        return t2 >= 0.0 and t1 <= 1.0

    @staticmethod
    def fleet_speed(ships):
        if ships <= 1:
            return 1.0
        ratio = math.log(ships) / math.log(1000.0)
        return 1.0 + (GameConfig.MAX_SPEED - 1.0) * max(0.0, min(1.0, ratio)) ** 1.5


CENTER = GameConfig.CENTER
SUN_RADIUS = GameConfig.SUN_RADIUS
ROTATION_RADIUS_LIMIT = GameConfig.ROTATION_RADIUS_LIMIT
BOARD_SIZE = 100.0
MAX_NB_STEP = 500
```

- [ ] **Step 2: Copy `evaluate()` verbatim from `137` (lines 271–291)**

```python
def evaluate(obs, player_id: int) -> tuple:
    planets = obs.planets if hasattr(obs, "planets") else obs["planets"]
    fleets  = obs.fleets  if hasattr(obs, "fleets")  else obs["fleets"]

    opponents = {p[1] for p in planets if p[1] not in (-1, player_id)}
    opponents |= {f[1] for f in fleets  if f[1] not in (-1, player_id)}

    my_prod  = sum(p[6] for p in planets if p[1] == player_id)
    my_ships = (sum(p[5] for p in planets if p[1] == player_id)
              + sum(f[6] for f in fleets  if f[1] == player_id))

    if not opponents:
        return (my_prod, my_ships)

    opp_prod  = max(sum(p[6] for p in planets if p[1] == opp) for opp in opponents)
    opp_ships = max(
        sum(p[5] for p in planets if p[1] == opp)
      + sum(f[6] for f in fleets  if f[1] == opp)
        for opp in opponents
    )
    return (my_prod - opp_prod, my_ships - opp_ships)
```

- [ ] **Step 3: Copy `_02_get_all_opportunities` and `_03_filter_collision` verbatim from `137` (lines 466–677)**

Paste the two `@staticmethod` methods under a `class StrategyPipeline:` header. Do not alter any logic.

- [ ] **Step 4: Add stubs for Board and agent**

```python
class Board:
    pass  # implemented in later tasks


BOARD: "Board | None" = None


def agent(obs):
    raise NotImplementedError
```

- [ ] **Step 5: Add minimal smoke-test scaffold**

```python
if __name__ == "__main__":
    import time
    from types import SimpleNamespace

    def _make_obs():
        planets = [
            [0, 0, 30.0, 50.0, 5.0, 100, 2],
            [1, 1, 70.0, 50.0, 5.0,  20, 2],
            [2, -1, 50.0, 20.0, 4.0,   0, 1],
        ]
        return SimpleNamespace(
            planets=[list(p) for p in planets],
            initial_planets=[list(p) for p in planets],
            fleets=[],
            comets=[],
            comet_planet_ids=[],
            angular_velocity=0.01,
            next_fleet_id=0,
            player=0,
        )

    obs0 = _make_obs()
    print("Scaffold OK — Board and agent are stubs.")
```

- [ ] **Step 6: Run the scaffold**

```
python 138-BoardCache.py
```

Expected output: `Scaffold OK — Board and agent are stubs.`

- [ ] **Step 7: Commit**

```bash
git add 138-BoardCache.py
git commit -m "feat: 138 — scaffold, copy unchanged components from 137"
```

---

### Task 2: Board.__init__ — planet nature, positions, and comet tracking

**Files:**
- Modify: `138-BoardCache.py` — replace `Board: pass` stub

**Interfaces:**
- Consumes: `obs` (SimpleNamespace or dict with `.planets`, `.initial_planets`, `.comets`, `.comet_planet_ids`, `.angular_velocity`)
- Produces:
  - `board.df_planete_nature : pl.DataFrame` — columns `id(i64), radius(f64), production(i64), nature(str)` — shape `(P, 4)`
  - `board.df_planete_pos    : pl.DataFrame` — columns `id(i64), step(i64), x(f64), y(f64)` — shape `(P×11, 4)` for steps `current_step..current_step+10`
  - `board.step : int`
  - `board._planet_pos_analytical(pid, game_step) -> tuple[float,float] | None`

- [ ] **Step 1: Write a test for Board.__init__**

Add to `if __name__ == "__main__":` block, replacing the stub print:

```python
    obs0 = _make_obs()
    board = Board(obs0, step=0, num_agents=2, player_id=0)

    # df_planete_nature
    assert board.df_planete_nature.shape == (3, 4), \
        f"Expected (3,4), got {board.df_planete_nature.shape}"
    assert set(board.df_planete_nature.columns) == {"id", "radius", "production", "nature"}
    natures = dict(zip(
        board.df_planete_nature["id"].to_list(),
        board.df_planete_nature["nature"].to_list(),
    ))
    assert natures[0] == "fix", f"Planet 0 should be fix, got {natures[0]}"

    # df_planete_pos: 3 planets × 11 steps = 33 rows
    assert board.df_planete_pos.shape[0] == 33, \
        f"Expected 33 pos rows, got {board.df_planete_pos.shape[0]}"
    assert board.df_planete_pos["step"].min() == 0
    assert board.df_planete_pos["step"].max() == 10

    # _planet_pos_analytical for a static planet returns constant position
    pos0 = board._planet_pos_analytical(0, 5)
    assert pos0 == (30.0, 50.0), f"Expected (30.0, 50.0), got {pos0}"

    print("Task 2 PASSED")
```

- [ ] **Step 2: Run test to verify it fails**

```
python 138-BoardCache.py
```

Expected: error — `Board` is a stub.

- [ ] **Step 3: Implement Board.__init__ and helpers**

Replace `class Board: pass` with:

```python
class Board:
    def __init__(self, obs, step: int, num_agents: int, player_id: int):
        self.step = step
        self.num_agents = num_agents
        self.player_id = player_id
        self.angular_velocity = obs.angular_velocity if hasattr(obs, "angular_velocity") else obs["angular_velocity"]

        self._init_nature_and_pos(obs)
        self.df_fleet = pl.DataFrame(schema={
            "id": pl.Int64, "owner": pl.Int64, "ships": pl.Int64,
            "id_tgt": pl.Int64, "step_tgt": pl.Int64,
        })
        self.df_fleet_sim = pl.DataFrame(schema={
            "id_src": pl.Int64, "step_src": pl.Int64, "ships_sent": pl.Int64,
            "owner": pl.Int64, "id_tgt": pl.Int64, "step_tgt": pl.Int64,
        })
        self.df_planete_ships = pl.DataFrame(schema={
            "id": pl.Int64, "step": pl.Int64, "ships": pl.Int64,
            "owner": pl.Int64, "recompute": pl.Boolean,
        })

    def _init_nature_and_pos(self, obs):
        planets  = obs.planets         if hasattr(obs, "planets")         else obs["planets"]
        initial  = obs.initial_planets if hasattr(obs, "initial_planets") else obs["initial_planets"]
        comets   = obs.comets          if hasattr(obs, "comets")          else obs["comets"]
        cpids    = obs.comet_planet_ids if hasattr(obs, "comet_planet_ids") else obs["comet_planet_ids"]

        self._comet_pid_set = set(cpids)
        self._comet_path_by_pid: dict = {}
        self._comet_idx_by_pid:  dict = {}
        for group in comets:
            for i, pid in enumerate(group["planet_ids"]):
                self._comet_path_by_pid[pid] = group["paths"][i]
                self._comet_idx_by_pid[pid]  = group["path_index"]

        initial_by_id = {p[0]: p for p in initial}
        nature_rows = []
        for p in planets:
            pid, owner, x, y, radius, ships, production = p[0], p[1], p[2], p[3], p[4], p[5], p[6]
            r = math.hypot(x - CENTER, y - CENTER)
            if pid in self._comet_pid_set:
                nature = "comet"
            elif r + radius < ROTATION_RADIUS_LIMIT:
                nature = "moving"
            else:
                nature = "fix"
            nature_rows.append({"id": pid, "radius": radius, "production": production, "nature": nature})

        self._planet_meta: dict = {}
        for row in nature_rows:
            pid = row["id"]
            meta: dict = {"nature": row["nature"]}
            if row["nature"] == "moving":
                ip = initial_by_id.get(pid)
                if ip is not None:
                    dx, dy = ip[2] - CENTER, ip[3] - CENTER
                    meta["r"]      = math.sqrt(dx * dx + dy * dy)
                    meta["theta0"] = math.atan2(dy, dx)
            elif row["nature"] == "fix":
                p_obj = next(pp for pp in planets if pp[0] == pid)
                meta["pos"] = (p_obj[2], p_obj[3])
            self._planet_meta[pid] = meta

        self.df_planete_nature = pl.DataFrame(nature_rows)

        # Build initial position window: steps self.step .. self.step+NB_STEPS_SIM
        pos_rows = []
        for pid in self._planet_meta:
            for k in range(GameConfig.NB_STEPS_SIM + 1):
                game_step = self.step + k
                pos = self._planet_pos_analytical(pid, game_step)
                if pos is not None:
                    pos_rows.append({"id": pid, "step": game_step, "x": pos[0], "y": pos[1]})
        self.df_planete_pos = pl.DataFrame(pos_rows)

    def _planet_pos_analytical(self, pid: int, game_step: int):
        meta = self._planet_meta.get(pid)
        if meta is None:
            return None
        nature = meta["nature"]
        if nature == "moving":
            theta = meta["theta0"] + self.angular_velocity * (game_step - 1)
            return (CENTER + meta["r"] * math.cos(theta), CENTER + meta["r"] * math.sin(theta))
        if nature == "fix":
            return meta["pos"]
        path = self._comet_path_by_pid.get(pid)
        idx  = self._comet_idx_by_pid.get(pid)
        if path is None or idx is None:
            return None
        comet_idx = idx + (game_step - self.step)
        if 0 <= comet_idx < len(path):
            return (path[comet_idx][0], path[comet_idx][1])
        return None
```

- [ ] **Step 4: Run tests**

```
python 138-BoardCache.py
```

Expected: `Task 2 PASSED`

- [ ] **Step 5: Commit**

```bash
git add 138-BoardCache.py
git commit -m "feat: 138 — Board.__init__, df_planete_nature, df_planete_pos"
```

---

### Task 3: Board.advance — slide position window + sync df_fleet

**Files:**
- Modify: `138-BoardCache.py` — add `advance()` and `_compute_fleet_arrival()` methods to `Board`

**Interfaces:**
- Consumes: `board` from Task 2; `obs` updated game state; `step` new step number
- Produces:
  - `board.df_planete_pos` updated (old step evicted, new far-end step added)
  - `board.df_fleet : pl.DataFrame` — columns `id, owner, ships, id_tgt, step_tgt` synced to `obs.fleets`
  - `board._compute_fleet_arrival(fleet, current_step) -> tuple[int,int] | None`

- [ ] **Step 1: Write tests**

Add after `print("Task 2 PASSED")`:

```python
    # Task 3: advance slides the window
    from types import SimpleNamespace as NS
    obs1 = _make_obs()
    board2 = Board(obs1, step=0, num_agents=2, player_id=0)
    obs1b = _make_obs()  # same state, step 1
    board2.advance(obs1b, step=1)

    assert board2.step == 1
    assert board2.df_planete_pos["step"].min() == 1, \
        f"Min step should be 1, got {board2.df_planete_pos['step'].min()}"
    assert board2.df_planete_pos["step"].max() == 11, \
        f"Max step should be 11, got {board2.df_planete_pos['step'].max()}"

    # df_fleet should be empty (no fleets in obs)
    assert board2.df_fleet.shape[0] == 0, f"Expected 0 fleets, got {board2.df_fleet.shape[0]}"

    print("Task 3 PASSED")
```

- [ ] **Step 2: Run test to verify it fails**

```
python 138-BoardCache.py
```

Expected: error — `Board` has no `advance` method.

- [ ] **Step 3: Implement `_compute_fleet_arrival`**

Add inside `class Board:`:

```python
    def _compute_fleet_arrival(self, fleet, current_step: int):
        _, owner, fx, fy, angle, src_id, ships = fleet
        speed = PhysicsEngine.fleet_speed(ships)
        dx_f = math.cos(angle) * speed
        dy_f = math.sin(angle) * speed

        non_comet_pids = [pid for pid in self._planet_meta if pid not in self._comet_pid_set]
        comet_pids     = list(self._comet_pid_set)

        for k in range(GameConfig.NB_STEPS_SIM):
            f_old = (fx + k * dx_f, fy + k * dy_f)
            f_new = (fx + (k + 1) * dx_f, fy + (k + 1) * dy_f)
            game_step = current_step + k

            for pid in non_comet_pids:
                p_old = self._planet_pos_analytical(pid, game_step)
                p_new = self._planet_pos_analytical(pid, game_step + 1)
                if p_old is None or p_new is None:
                    continue
                radius = self._planet_meta[pid].get("radius",
                    self.df_planete_nature.filter(pl.col("id") == pid)["radius"][0])
                if PhysicsEngine.swept_pair_hit(f_old, f_new, p_old, p_new, radius):
                    return (pid, game_step)

            if not (0 <= f_new[0] <= BOARD_SIZE and 0 <= f_new[1] <= BOARD_SIZE):
                return None
            if PhysicsEngine.point_to_segment_distance((CENTER, CENTER), f_old, f_new) < SUN_RADIUS:
                return None

            for pid in comet_pids:
                c_old = self._planet_pos_analytical(pid, game_step)
                c_new = self._planet_pos_analytical(pid, game_step + 1)
                if c_old is None or c_new is None:
                    continue
                radius = self._planet_meta[pid].get("radius",
                    self.df_planete_nature.filter(pl.col("id") == pid)["radius"][0])
                if PhysicsEngine.point_to_segment_distance(f_new, c_old, c_new) < radius:
                    return (pid, game_step)

        return None
```

Note: cache radius in `_planet_meta` to avoid per-call DataFrame filter. Update `_init_nature_and_pos` to store radius in `_planet_meta`:

```python
            # add to meta dict after existing logic:
            meta["radius"] = row["radius"]
```

- [ ] **Step 4: Implement `advance`**

Add inside `class Board:`:

```python
    def advance(self, obs, step: int):
        old_step = self.step
        self.step = step

        # ── Step 1: slide df_planete_pos ─────────────────────────────────
        # evict old step
        self.df_planete_pos = self.df_planete_pos.filter(pl.col("step") >= step)

        # sync expired comets
        cpids = obs.comet_planet_ids if hasattr(obs, "comet_planet_ids") else obs["comet_planet_ids"]
        current_comet_pids = set(cpids)
        for pid in list(self._comet_pid_set - current_comet_pids):
            self._planet_meta.pop(pid, None)
            self._comet_path_by_pid.pop(pid, None)
            self._comet_idx_by_pid.pop(pid, None)
            self.df_planete_pos = self.df_planete_pos.filter(pl.col("id") != pid)
        self._comet_pid_set = current_comet_pids

        # add new far-end step
        new_step = step + GameConfig.NB_STEPS_SIM
        new_rows = []
        for pid in self._planet_meta:
            pos = self._planet_pos_analytical(pid, new_step)
            if pos is not None:
                new_rows.append({"id": pid, "step": new_step, "x": pos[0], "y": pos[1]})
        if new_rows:
            self.df_planete_pos = pl.concat([self.df_planete_pos, pl.DataFrame(new_rows)])

        # ── Step 2: sync df_fleet ─────────────────────────────────────────
        fleets = obs.fleets if hasattr(obs, "fleets") else obs["fleets"]
        current_fids = {f[0] for f in fleets}

        # drop departed fleets
        if self.df_fleet.shape[0] > 0:
            self.df_fleet = self.df_fleet.filter(pl.col("id").is_in(list(current_fids)))

        # add newly seen fleets
        tracked_fids = set(self.df_fleet["id"].to_list()) if self.df_fleet.shape[0] > 0 else set()
        new_fleet_rows = []
        for fleet in fleets:
            fid = fleet[0]
            if fid not in tracked_fids:
                arrival = self._compute_fleet_arrival(fleet, step)
                new_fleet_rows.append({
                    "id":       fid,
                    "owner":    fleet[1],
                    "ships":    fleet[6],
                    "id_tgt":   arrival[0] if arrival else None,
                    "step_tgt": arrival[1] if arrival else None,
                })
        if new_fleet_rows:
            self.df_fleet = pl.concat([
                self.df_fleet,
                pl.DataFrame(new_fleet_rows, schema={
                    "id": pl.Int64, "owner": pl.Int64, "ships": pl.Int64,
                    "id_tgt": pl.Int64, "step_tgt": pl.Int64,
                })
            ])
```

- [ ] **Step 5: Run tests**

```
python 138-BoardCache.py
```

Expected: `Task 2 PASSED` then `Task 3 PASSED`

- [ ] **Step 6: Commit**

```bash
git add 138-BoardCache.py
git commit -m "feat: 138 — Board.advance, _compute_fleet_arrival, df_fleet sync"
```

---

### Task 4: Board.build_base_ships — full df_planete_ships recompute

**Files:**
- Modify: `138-BoardCache.py` — add `_combat_step()` free function and `build_base_ships()` method

**Interfaces:**
- Consumes: `board.df_fleet`, `board.df_planete_nature`, `board.step`, `obs.planets`
- Produces:
  - `board.df_planete_ships : pl.DataFrame` — columns `id(i64), step(i64), ships(i64), owner(i64), recompute(bool)` — shape `(P×11, 5)`, all `recompute=False`
  - `_combat_step(arrivals, ship_state, owner_state)` — free function, modifies dicts in place

- [ ] **Step 1: Write tests**

Add after `print("Task 3 PASSED")`:

```python
    # Task 4: build_base_ships
    obs_t4 = _make_obs()
    board4 = Board(obs_t4, step=0, num_agents=2, player_id=0)
    board4.advance(obs_t4, step=0)   # init df_fleet (no-op for step 0)
    board4.build_base_ships(obs_t4)

    assert board4.df_planete_ships.shape == (33, 5), \
        f"Expected (33,5), got {board4.df_planete_ships.shape}"

    # At step 0, ships match obs
    step0 = board4.df_planete_ships.filter(pl.col("step") == 0).sort("id")
    assert step0["ships"].to_list() == [100, 20, 0], \
        f"Step-0 ships wrong: {step0['ships'].to_list()}"
    assert step0["owner"].to_list() == [0, 1, -1], \
        f"Step-0 owners wrong: {step0['owner'].to_list()}"

    # At step 1, planets with owner produce ships
    step1 = board4.df_planete_ships.filter(pl.col("step") == 1).sort("id")
    assert step1["ships"].to_list() == [102, 22, 0], \
        f"Step-1 ships wrong: {step1['ships'].to_list()}"

    # No recompute flags set
    assert not board4.df_planete_ships["recompute"].any(), "recompute should all be False"

    print("Task 4 PASSED")
```

- [ ] **Step 2: Run test to verify it fails**

```
python 138-BoardCache.py
```

Expected: error — `Board` has no `build_base_ships` method.

- [ ] **Step 3: Implement `_combat_step` free function**

Add before `class Board:`:

```python
def _combat_step(planet_arrivals: dict, ship_state: dict, owner_state: dict):
    """Resolve combat for all planets with arriving fleets this step.

    planet_arrivals: {pid: [(owner, ships), ...]}
    Modifies ship_state and owner_state in place. Same formula as 1_29_3.py.
    """
    for planet_id, fleet_list in planet_arrivals.items():
        player_ships: dict = {}
        for fowner, fships in fleet_list:
            player_ships[fowner] = player_ships.get(fowner, 0) + fships
        if not player_ships:
            continue
        sorted_players = sorted(player_ships.items(), key=lambda x: x[1], reverse=True)
        top_player, top_ships = sorted_players[0]
        if len(sorted_players) > 1:
            second_ships = sorted_players[1][1]
            survivor_ships = 0 if sorted_players[0][1] == sorted_players[1][1] else top_ships - second_ships
            survivor_owner = top_player if survivor_ships > 0 else -1
        else:
            survivor_owner = top_player
            survivor_ships = top_ships
        if survivor_ships > 0:
            if owner_state[planet_id] == survivor_owner:
                ship_state[planet_id] += survivor_ships
            else:
                ship_state[planet_id] -= survivor_ships
                if ship_state[planet_id] < 0:
                    owner_state[planet_id] = survivor_owner
                    ship_state[planet_id] = abs(ship_state[planet_id])
```

- [ ] **Step 4: Implement `build_base_ships`**

Add inside `class Board:`:

```python
    def build_base_ships(self, obs):
        planets    = obs.planets if hasattr(obs, "planets") else obs["planets"]
        production = dict(zip(
            self.df_planete_nature["id"].to_list(),
            self.df_planete_nature["production"].to_list(),
        ))

        ship_state  = {p[0]: p[5] for p in planets}
        owner_state = {p[0]: p[1] for p in planets}

        # Build arrivals lookup from df_fleet: step_tgt -> {pid: [(owner, ships)]}
        arrivals_by_step: dict = {}
        if self.df_fleet.shape[0] > 0:
            for row in self.df_fleet.filter(pl.col("step_tgt").is_not_null()).iter_rows(named=True):
                arrivals_by_step.setdefault(row["step_tgt"], {}).setdefault(row["id_tgt"], []).append(
                    (row["owner"], row["ships"])
                )

        rows = []
        for k in range(GameConfig.NB_STEPS_SIM + 1):
            game_step = self.step + k
            for pid in ship_state:
                rows.append({
                    "id": pid, "step": game_step,
                    "ships": ship_state[pid], "owner": owner_state[pid],
                    "recompute": False,
                })
            if k == GameConfig.NB_STEPS_SIM:
                break
            # production
            for pid, owner in owner_state.items():
                if owner != -1:
                    ship_state[pid] += production[pid]
            # combat
            if game_step in arrivals_by_step:
                _combat_step(arrivals_by_step[game_step], ship_state, owner_state)

        self.df_planete_ships = pl.DataFrame(rows)
```

- [ ] **Step 5: Run tests**

```
python 138-BoardCache.py
```

Expected: `Task 2 PASSED`, `Task 3 PASSED`, `Task 4 PASSED`

- [ ] **Step 6: Commit**

```bash
git add 138-BoardCache.py
git commit -m "feat: 138 — Board.build_base_ships, _combat_step helper"
```

---

### Task 5: Evaluation helpers — build_df_s_slice, extract_horizon_dict, _apply_sim_fleet, _evaluate_dict

**Files:**
- Modify: `138-BoardCache.py` — add four methods/functions

**Interfaces:**
- Consumes: `board.df_planete_ships`, `board.df_planete_pos`, `board.df_planete_nature`
- Produces:
  - `board.build_df_s_slice(df_ships, step_from) -> tuple[pl.DataFrame, pl.DataFrame]` — returns `(df_s, planet_disp)` matching signature expected by `_02_get_all_opportunities`
  - `board.extract_horizon_dict(df_ships) -> dict[int, tuple[int,int,int]]` — `{pid: (ships, owner, production)}`
  - `board._apply_sim_fleet(horizon, sim_row) -> dict` — `sim_row = (id_src, id_tgt, step_tgt, ships_sent)` or `None`
  - `board._evaluate_dict(horizon, player_id) -> tuple[int,int]`

- [ ] **Step 1: Write tests**

Add after `print("Task 4 PASSED")`:

```python
    # Task 5: evaluation helpers
    obs_t5 = _make_obs()
    board5 = Board(obs_t5, step=0, num_agents=2, player_id=0)
    board5.advance(obs_t5, step=0)
    board5.build_base_ships(obs_t5)

    # build_df_s_slice returns df_s with correct columns
    df_s5, pd5 = board5.build_df_s_slice(board5.df_planete_ships, step_from=5)
    required_cols = {"step", "id", "x", "y", "radius", "ships", "production", "owner", "nature"}
    assert required_cols.issubset(set(df_s5.columns)), \
        f"Missing columns: {required_cols - set(df_s5.columns)}"
    assert df_s5["step"].min() == 5, f"df_s5 should start at step 5"
    assert "planet_disp" in pd5.columns

    # extract_horizon_dict
    horizon = board5.extract_horizon_dict(board5.df_planete_ships)
    assert len(horizon) == 3
    assert isinstance(horizon[0], tuple) and len(horizon[0]) == 3  # (ships, owner, production)
    assert horizon[0][1] == 0   # planet 0 owned by player 0
    assert horizon[1][1] == 1   # planet 1 owned by player 1

    # _evaluate_dict: player 0 has more production and ships
    score = board5._evaluate_dict(horizon, player_id=0)
    assert isinstance(score, tuple) and len(score) == 2
    assert score[0] > 0, f"Player 0 should have prod advantage, got {score}"

    # _apply_sim_fleet None → unchanged
    h2 = board5._apply_sim_fleet(horizon, None)
    assert h2 is horizon  # identity for None

    # _apply_sim_fleet with a fleet that captures planet 1
    # send 50 ships from planet 0 → planet 1; planet 1 has 20 ships at horizon
    ships_at_10 = horizon[0][0]
    h3 = board5._apply_sim_fleet(horizon, (0, 1, 7, 50))
    assert h3[0][0] == ships_at_10- 50, f"Src should lose 50 ships, got {h3[0][0]}"
    # 50 vs 20+at_horizon: attacker wins, planet flips to owner 0
    assert h3[1][1] == 0, f"Planet 1 should be captured by player 0, got {h3[1][1]}"

    print("Task 5 PASSED")
```

- [ ] **Step 2: Run test to verify it fails**

```
python 138-BoardCache.py
```

Expected: error — method not found.

- [ ] **Step 3: Implement `build_df_s_slice`**

Add inside `class Board:`:

```python
    def build_df_s_slice(self, df_ships: pl.DataFrame, step_from: int):
        """Build df_s and planet_disp for _02_get_all_opportunities."""
        ships_slice = df_ships.filter(pl.col("step") >= step_from).drop("recompute")
        pos_slice   = self.df_planete_pos.filter(pl.col("step") >= step_from)

        df_s = (
            ships_slice
            .join(pos_slice, on=["id", "step"], how="left")
            .join(self.df_planete_nature.select(["id", "radius", "production", "nature"]), on="id", how="left")
            .sort("step")
        )

        # planet_disp: distance from previous step position
        prev_pos = (
            self.df_planete_pos
            .filter(pl.col("step") >= step_from - 1)
            .select(["id", "step", "x", "y"])
            .rename({"x": "x_prev", "y": "y_prev"})
            .with_columns((pl.col("step") + 1).alias("step"))
        )
        planet_disp = (
            df_s.lazy()
            .select(["id", "step", "x", "y"])
            .join(prev_pos.lazy(), on=["id", "step"], how="left")
            .with_columns(
                ((pl.col("x") - pl.col("x_prev").fill_null(pl.col("x"))).pow(2) +
                 (pl.col("y") - pl.col("y_prev").fill_null(pl.col("y"))).pow(2)
                ).sqrt().alias("planet_disp")
            )
            .select(["id", "step", "planet_disp"])
            .collect()
        )
        return df_s, planet_disp
```

- [ ] **Step 4: Implement `extract_horizon_dict`**

Add inside `class Board:`:

```python
    def extract_horizon_dict(self, df_ships: pl.DataFrame) -> dict:
        """Return {pid: (ships, owner, production)} at the last step in df_ships."""
        horizon_step = df_ships["step"].max()
        prod_map = dict(zip(
            self.df_planete_nature["id"].to_list(),
            self.df_planete_nature["production"].to_list(),
        ))
        result = {}
        for row in df_ships.filter(pl.col("step") == horizon_step).iter_rows(named=True):
            pid = row["id"]
            result[pid] = (row["ships"], row["owner"], prod_map.get(pid, 0))
        return result
```

- [ ] **Step 5: Implement `_apply_sim_fleet`**

Add inside `class Board:`:

```python
    @staticmethod
    def _apply_sim_fleet(horizon: dict, sim_row) -> dict:
        """Apply one simulated fleet to horizon dict. sim_row=(id_src,id_tgt,step_tgt,ships_sent) or None."""
        if sim_row is None:
            return horizon
        id_src, id_tgt, step_tgt, ships_sent = sim_row
        d = dict(horizon)  # shallow copy

        # ships leave source
        if id_src in d:
            s, o, p = d[id_src]
            d[id_src] = (s - ships_sent, o, p)

        # combat at target (only if within horizon window)
        if step_tgt is not None and id_tgt in d:
            tgt_ships, tgt_owner, tgt_prod = d[id_tgt]
            fleet_owner = horizon[id_src][1]  # owner unchanged
            if tgt_owner == fleet_owner:
                d[id_tgt] = (tgt_ships + ships_sent, tgt_owner, tgt_prod)
            else:
                surviving = tgt_ships - ships_sent
                if surviving < 0:
                    d[id_tgt] = (-surviving, fleet_owner, tgt_prod)
                elif surviving == 0:
                    d[id_tgt] = (0, -1, tgt_prod)
                # surviving > 0: defender wins, no change needed beyond src deduction
        return d
```

- [ ] **Step 6: Implement `_evaluate_dict`**

Add inside `class Board:`:

```python
    @staticmethod
    def _evaluate_dict(horizon: dict, player_id: int) -> tuple:
        """Pure-Python evaluate() equivalent operating on horizon dict."""
        opponents = {v[1] for v in horizon.values() if v[1] not in (-1, player_id)}

        my_prod  = sum(v[2] for v in horizon.values() if v[1] == player_id)
        my_ships = sum(v[0] for v in horizon.values() if v[1] == player_id)

        if not opponents:
            return (my_prod, my_ships)

        opp_prod  = max(sum(v[2] for v in horizon.values() if v[1] == opp) for opp in opponents)
        opp_ships = max(sum(v[0] for v in horizon.values() if v[1] == opp) for opp in opponents)
        return (my_prod - opp_prod, my_ships - opp_ships)
```

- [ ] **Step 7: Run tests**

```
python 138-BoardCache.py
```

Expected: `Task 2 PASSED`, `Task 3 PASSED`, `Task 4 PASSED`, `Task 5 PASSED`

- [ ] **Step 8: Commit**

```bash
git add 138-BoardCache.py
git commit -m "feat: 138 — build_df_s_slice, extract_horizon_dict, _apply_sim_fleet, _evaluate_dict"
```

---

### Task 6: Board._recompute_from_sim — dirty-row incremental recompute

**Files:**
- Modify: `138-BoardCache.py` — add `_recompute_from_sim()` method to `Board`

**Interfaces:**
- Consumes: `df_ships_base: pl.DataFrame` (P×11, from `build_base_ships`); `move: list` (`[id_src, angle, ships_sent]`); `id_tgt: int | None`; `step_tgt: int | None`; `board.df_fleet`, `board.df_planete_nature`, `board.step`
- Produces: `pl.DataFrame` — same schema as `df_planete_ships`, with dirty rows replaced by recomputed values

- [ ] **Step 1: Write tests**

Add after `print("Task 5 PASSED")`:

```python
    # Task 6: _recompute_from_sim
    obs_t6 = _make_obs()
    board6 = Board(obs_t6, step=0, num_agents=2, player_id=0)
    board6.advance(obs_t6, step=0)
    board6.build_base_ships(obs_t6)

    # Send 50 ships from planet 0 (step_src=0), targeting planet 1 (step_tgt=5)
    df_c0 = board6._recompute_from_sim(
        board6.df_planete_ships,
        move=[0, math.atan2(50.0 - 50.0, 70.0 - 30.0), 50],
        id_tgt=1, step_tgt=5,
    )

    assert df_c0.shape == board6.df_planete_ships.shape, "Shape must match base"

    # Source planet at step 1 should have 50 fewer ships than base
    base_src_1 = board6.df_planete_ships.filter(
        (pl.col("id") == 0) & (pl.col("step") == 1)
    )["ships"][0]
    c0_src_1 = df_c0.filter(
        (pl.col("id") == 0) & (pl.col("step") == 1)
    )["ships"][0]
    assert c0_src_1 == base_src_1 - 50, \
        f"Src step-1 ships: expected {base_src_1-50}, got {c0_src_1}"

    # Planet 1 at step_tgt=5 should reflect combat (fleet captures since 50 > 20+production)
    c0_tgt_5_owner = df_c0.filter(
        (pl.col("id") == 1) & (pl.col("step") == 5)
    )["owner"][0]
    assert c0_tgt_5_owner == 0, \
        f"Planet 1 at step 5 should be captured by player 0, got {c0_tgt_5_owner}"

    # Base df unchanged
    base_tgt_5_owner = board6.df_planete_ships.filter(
        (pl.col("id") == 1) & (pl.col("step") == 5)
    )["owner"][0]
    assert base_tgt_5_owner == 1, "Base df must be immutable"

    print("Task 6 PASSED")
```

- [ ] **Step 2: Run test to verify it fails**

```
python 138-BoardCache.py
```

Expected: error — method not found.

- [ ] **Step 3: Implement `_recompute_from_sim`**

Add inside `class Board:`:

```python
    def _recompute_from_sim(
        self,
        df_ships_base: pl.DataFrame,
        move: list,
        id_tgt: "int | None",
        step_tgt: "int | None",
    ) -> pl.DataFrame:
        """Return new df_planete_ships with the effect of `move` applied.

        move = [id_src, angle, ships_sent].  Recomputes only dirty rows.
        """
        id_src, angle, ships_sent = move[0], move[1], move[2]
        step_src = self.step  # fleet launches at current step

        # Identify dirty planets and the minimum step to recompute from
        dirty_pids = {id_src}
        min_dirty  = step_src + 1
        if id_tgt is not None and step_tgt is not None:
            dirty_pids.add(id_tgt)
            min_dirty = min(min_dirty, step_tgt)

        dirty_list = list(dirty_pids)

        # Rows to keep: not-dirty OR before min_dirty
        keep = df_ships_base.filter(
            ~pl.col("id").is_in(dirty_list) | (pl.col("step") < min_dirty)
        )

        # Seed state from the last kept step for each dirty planet
        production = dict(zip(
            self.df_planete_nature["id"].to_list(),
            self.df_planete_nature["production"].to_list(),
        ))
        ship_state:  dict = {}
        owner_state: dict = {}
        for pid in dirty_pids:
            seed_step = min_dirty - 1
            row = df_ships_base.filter(
                (pl.col("id") == pid) & (pl.col("step") == seed_step)
            )
            if row.shape[0] > 0:
                ship_state[pid]  = row["ships"][0]
                owner_state[pid] = row["owner"][0]
            else:
                # fallback: step 0 from base
                row0 = df_ships_base.filter(
                    (pl.col("id") == pid) & (pl.col("step") == self.step)
                )
                ship_state[pid]  = row0["ships"][0]
                owner_state[pid] = row0["owner"][0]

        # Apply the sim fleet departure at step_src: src loses ships_sent at step_src+1
        if id_src in ship_state and min_dirty == step_src + 1:
            ship_state[id_src] = max(0, ship_state[id_src] - ships_sent)

        # Build arrivals from df_fleet (real fleets) PLUS the sim fleet for dirty planets
        arrivals_by_step: dict = {}
        if self.df_fleet.shape[0] > 0:
            for row in self.df_fleet.filter(
                pl.col("id_tgt").is_in(dirty_list) &
                pl.col("step_tgt").is_not_null() &
                (pl.col("step_tgt") >= min_dirty)
            ).iter_rows(named=True):
                arrivals_by_step.setdefault(row["step_tgt"], {}).setdefault(row["id_tgt"], []).append(
                    (row["owner"], row["ships"])
                )
        # sim fleet arrives at id_tgt at step_tgt
        if id_tgt is not None and step_tgt is not None:
            fleet_owner = self.player_id
            arrivals_by_step.setdefault(step_tgt, {}).setdefault(id_tgt, []).append(
                (fleet_owner, ships_sent)
            )

        # Recompute rows for dirty planets from min_dirty to step+NB_STEPS_SIM
        new_rows = []
        max_step = self.step + GameConfig.NB_STEPS_SIM
        for k_abs in range(min_dirty, max_step + 1):
            for pid in dirty_pids:
                if pid not in ship_state:
                    continue
                new_rows.append({
                    "id": pid, "step": k_abs,
                    "ships": ship_state[pid], "owner": owner_state[pid],
                    "recompute": False,
                })
            if k_abs == max_step:
                break
            # production for this step
            for pid in dirty_pids:
                if pid in owner_state and owner_state[pid] != -1:
                    ship_state[pid] += production.get(pid, 0)
            # combat
            if k_abs in arrivals_by_step:
                dirty_arrivals = {
                    pid: lst for pid, lst in arrivals_by_step[k_abs].items()
                    if pid in dirty_pids
                }
                if dirty_arrivals:
                    _combat_step(dirty_arrivals, ship_state, owner_state)

        new_df = pl.DataFrame(new_rows, schema={
            "id": pl.Int64, "step": pl.Int64, "ships": pl.Int64,
            "owner": pl.Int64, "recompute": pl.Boolean,
        }) if new_rows else pl.DataFrame(schema={
            "id": pl.Int64, "step": pl.Int64, "ships": pl.Int64,
            "owner": pl.Int64, "recompute": pl.Boolean,
        })

        return pl.concat([keep, new_df]).sort(["step", "id"])
```

- [ ] **Step 4: Run tests**

```
python 138-BoardCache.py
```

Expected: `Task 2 PASSED` through `Task 6 PASSED`

- [ ] **Step 5: Commit**

```bash
git add 138-BoardCache.py
git commit -m "feat: 138 — Board._recompute_from_sim, incremental dirty-row recompute"
```

---

### Task 7: _04_minimax_search rewrite + agent() + smoke test

**Files:**
- Modify: `138-BoardCache.py` — add `StrategyPipeline._04_minimax_search`, implement `agent()`, and complete `__main__` smoke test

**Interfaces:**
- Consumes: all Board methods from Tasks 2–6; `_02_get_all_opportunities`, `_03_filter_collision` from Task 1
- Produces: `agent(obs) -> list[list]` — list of `[id_src, angle, ships]` moves

- [ ] **Step 1: Write smoke test (will fail until agent() implemented)**

Replace the existing `__main__` block with:

```python
if __name__ == "__main__":
    import time
    from types import SimpleNamespace

    def _make_obs():
        planets = [
            [0, 0, 30.0, 50.0, 5.0, 100, 2],
            [1, 1, 70.0, 50.0, 5.0,  20, 2],
            [2, -1, 50.0, 20.0, 4.0,   0, 1],
        ]
        return SimpleNamespace(
            planets=[list(p) for p in planets],
            initial_planets=[list(p) for p in planets],
            fleets=[],
            comets=[],
            comet_planet_ids=[],
            angular_velocity=0.01,
            next_fleet_id=0,
            player=0,
        )

    # ── Unit tests (tasks 2–6) ────────────────────────────────────────────
    obs0 = _make_obs()
    board = Board(obs0, step=0, num_agents=2, player_id=0)

    assert board.df_planete_nature.shape == (3, 4)
    assert board.df_planete_pos.shape[0] == 33
    pos0 = board._planet_pos_analytical(0, 5)
    assert pos0 == (30.0, 50.0)
    print("T2 OK")

    board.advance(obs0, step=0)
    assert board.df_planete_pos["step"].min() == 0
    assert board.df_planete_pos["step"].max() == 10
    assert board.df_fleet.shape[0] == 0
    print("T3 OK")

    board.build_base_ships(obs0)
    assert board.df_planete_ships.shape == (33, 5)
    step0r = board.df_planete_ships.filter(pl.col("step") == 0).sort("id")
    assert step0r["ships"].to_list() == [100, 20, 0]
    assert not board.df_planete_ships["recompute"].any()
    print("T4 OK")

    df_s5, pd5 = board.build_df_s_slice(board.df_planete_ships, step_from=5)
    assert {"step","id","x","y","radius","ships","production","owner","nature"}.issubset(set(df_s5.columns))
    horizon = board.extract_horizon_dict(board.df_planete_ships)
    assert len(horizon) == 3 and isinstance(horizon[0], tuple) and len(horizon[0]) == 3
    assert board._evaluate_dict(horizon, 0)[0] > 0
    h_none = board._apply_sim_fleet(horizon, None)
    assert h_none is horizon
    ships_src_10 = horizon[0][0]
    h3 = board._apply_sim_fleet(horizon, (0, 1, 7, 50))
    assert h3[0][0] == ships_src_10 - 50
    assert h3[1][1] == 0
    print("T5 OK")

    df_c0 = board._recompute_from_sim(
        board.df_planete_ships,
        move=[0, math.atan2(0.0, 40.0), 50],
        id_tgt=1, step_tgt=5,
    )
    assert df_c0.shape == board.df_planete_ships.shape
    base_s1 = board.df_planete_ships.filter((pl.col("id")==0)&(pl.col("step")==1))["ships"][0]
    c0_s1   = df_c0.filter((pl.col("id")==0)&(pl.col("step")==1))["ships"][0]
    assert c0_s1 == base_s1 - 50
    assert board.df_planete_ships.filter((pl.col("id")==1)&(pl.col("step")==5))["owner"][0] == 1
    print("T6 OK")

    # ── Full agent smoke test ─────────────────────────────────────────────
    BOARD = None
    obs_a = _make_obs()
    t0 = time.time()
    result = agent(obs_a)
    elapsed = time.time() - t0

    print(f"agent() -> {result}")
    print(f"elapsed: {elapsed:.3f}s")

    assert isinstance(result, list), "agent must return a list"
    for move in result:
        assert len(move) == 3, f"move must be [id_src, angle, ships]: {move}"
    assert elapsed < 5.0, f"too slow: {elapsed:.3f}s"

    print("Smoke test PASSED.")
```

- [ ] **Step 2: Implement `StrategyPipeline._04_minimax_search`**

Add inside `class StrategyPipeline:` after `_03_filter_collision`:

```python
    @staticmethod
    def _04_minimax_search(safe_lf: pl.LazyFrame, obs, board: "Board") -> list:
        NB_STEPS_5   = 5
        player_id    = board.player_id
        current_step = board.step

        attacks_with_angle = safe_lf.collect()
        moves_out: list = []

        # ── Comet evasion (unchanged from 137) ───────────────────────────
        if not attacks_with_angle.is_empty():
            awa_comets = attacks_with_angle.filter(pl.col("nature_src") == "comet")
            if not awa_comets.is_empty():
                x_off = (awa_comets["x_src"] - GameConfig.CENTER).abs().max() or 0
                y_off = (awa_comets["y_src"] - GameConfig.CENTER).abs().max() or 0
                if max(x_off, y_off) > 45:
                    moves_out += [list(r) for r in (
                        awa_comets
                        .sort(["ships_sent", "step"], descending=[True, False])
                        .group_by("id_src", maintain_order=True)
                        .first()
                        .select(["id_src", "final_angle", "ships_sent"])
                        .rows()
                    )]
                    id_to_avoid = awa_comets["id_src"].unique().to_list()
                    attacks_with_angle = attacks_with_angle.filter(
                        ~pl.col("id_src").is_in(id_to_avoid)
                    )

        # ── Step-0 candidates (include id_tgt and step_tgt) ──────────────
        step0_candidates: list = [(None, None, None)]  # (move, tgt_id, step_tgt)
        if not attacks_with_angle.is_empty():
            top5_df = (
                attacks_with_angle
                .sort(["step", "ships_sent"])
                .group_by(["id_src", "id"], maintain_order=True)
                .first()
                .sort(["step", "ships_sent"])
                .group_by("id_src", maintain_order=True)
                .head(5)
            )
            for row in top5_df.select(["id_src", "id", "step", "final_angle", "ships_sent"]).iter_rows():
                src_id, tgt_id, step_tgt, angle, ships = row
                step0_candidates.append(([src_id, angle, ships], tgt_id, step_tgt))

        # ── Baseline: df_ships at step NB_STEPS_5 (do nothing at step 0) ─
        df_ships_base5 = board.df_planete_ships  # covers step 0..10
        step5_from = current_step + NB_STEPS_5
        df_s5_base, pd5_base = board.build_df_s_slice(df_ships_base5, step_from=step5_from)
        pa5_base   = StrategyPipeline._02_get_all_opportunities(df_s5_base, pd5_base, player_id)
        safe5_base = StrategyPipeline._03_filter_collision(pa5_base).collect()

        # c5 candidates: [id_src, id_tgt, step_tgt, angle, ships_sent]
        c5_candidates_base: list = [None]
        if not safe5_base.is_empty():
            for row in safe5_base.select(["id_src", "id", "step", "final_angle", "ships_sent"]).iter_rows():
                src, tgt, st, ang, sh = row
                c5_candidates_base.append([src, tgt, st, ang, sh])

        horizon_base = board.extract_horizon_dict(df_ships_base5)

        top3_scored: list = []
        for c5 in c5_candidates_base:
            sim = None if c5 is None else (c5[0], c5[1], c5[2], c5[4])
            score = board._evaluate_dict(board._apply_sim_fleet(horizon_base, sim), player_id)
            top3_scored.append((score, c5))
        top3_scored.sort(key=lambda x: x[0], reverse=True)
        top3_c5_base = [c5 for _, c5 in top3_scored[:3]]
        best_score   = top3_scored[0][0] if top3_scored else None
        best_c0      = None

        # ── Per c0 candidate ──────────────────────────────────────────────
        for c0_move, c0_tgt_id, c0_step_tgt in step0_candidates[1:]:

            df_ships_c0 = board._recompute_from_sim(
                df_ships_base5, c0_move, c0_tgt_id, c0_step_tgt
            )

            # changed-ids shortcut (same as 137)
            base_at5 = {
                row["id"]: (row["owner"], row["ships"])
                for row in df_ships_base5.filter(
                    pl.col("step") == current_step + NB_STEPS_5
                ).iter_rows(named=True)
            }
            c0_at5 = {
                row["id"]: (row["owner"], row["ships"])
                for row in df_ships_c0.filter(
                    pl.col("step") == current_step + NB_STEPS_5
                ).iter_rows(named=True)
            }
            changed_ids = {
                pid for pid, (owner, ships) in c0_at5.items()
                if base_at5.get(pid) != (owner, ships)
            }

            if changed_ids:
                df_s5_c, pd5_c = board.build_df_s_slice(df_ships_c0, step_from=step5_from)
                pa5_c   = StrategyPipeline._02_get_all_opportunities(df_s5_c, pd5_c, player_id)
                safe5_c = StrategyPipeline._03_filter_collision(pa5_c).collect()

                changed_list = list(changed_ids)
                new_rows  = safe5_c.filter(pl.col("id_src").is_in(changed_list)) if not safe5_c.is_empty() else safe5_c
                keep_rows = safe5_base.filter(~pl.col("id_src").is_in(changed_list)) if not safe5_base.is_empty() else safe5_base
                parts = [df for df in [keep_rows, new_rows] if not df.is_empty()]
                merged5 = pl.concat(parts) if parts else pl.DataFrame()
            else:
                merged5 = safe5_base

            src_id = c0_move[0]
            covered_srcs = {src_id}
            if c0_tgt_id is not None:
                covered_srcs.add(c0_tgt_id)

            restricted: list = [None]
            if not merged5.is_empty():
                for filt, col in [("id_src", "id_src"), ("id", "id_src")]:
                    sub = merged5.filter(pl.col(filt).is_in(list(covered_srcs)))
                    for row in sub.select(["id_src", "id", "step", "final_angle", "ships_sent"]).iter_rows():
                        src, tgt, st, ang, sh = row
                        restricted.append([src, tgt, st, ang, sh])

            for c5 in top3_c5_base:
                if c5 is not None and c5[0] not in covered_srcs:
                    restricted.append(c5)
                    break

            horizon_c0  = board.extract_horizon_dict(df_ships_c0)
            best_score5 = None
            for c5 in restricted:
                sim = None if c5 is None else (c5[0], c5[1], c5[2], c5[4])
                score = board._evaluate_dict(board._apply_sim_fleet(horizon_c0, sim), player_id)
                if best_score5 is None or score > best_score5:
                    best_score5 = score

            if best_score is None or (best_score5 is not None and best_score5 > best_score):
                best_score = best_score5
                best_c0    = c0_move

        if best_c0 is not None:
            moves_out.append(best_c0)
            print(f"Minimax best move: {best_c0}  score={best_score}")
        return moves_out
```

- [ ] **Step 3: Implement `agent()`**

Replace the stub:

```python
BOARD: "Board | None" = None


def agent(obs):
    global BOARD
    step = obs.get("step", 0) if isinstance(obs, dict) else getattr(obs, "step", 0)
    print(f"Agent called step={step} "
          f"remainingOverageTime={obs.get('remainingOverageTime', 0) if isinstance(obs, dict) else 0}")

    if BOARD is None:
        initial = obs.initial_planets if hasattr(obs, "initial_planets") else obs["initial_planets"]
        owners  = {p[1] for p in initial if p[1] != -1}
        num_agents = 4 if len(owners) > 2 else 2
        player_id  = obs.get("player", 0) if isinstance(obs, dict) else obs.player
        BOARD = Board(obs, step=step, num_agents=num_agents, player_id=player_id)
    else:
        BOARD.advance(obs, step=step)

    BOARD.build_base_ships(obs)

    df_s, planet_disp = BOARD.build_df_s_slice(BOARD.df_planete_ships, step_from=step)
    pa_lf   = StrategyPipeline._02_get_all_opportunities(df_s, planet_disp, BOARD.player_id)
    safe_lf = StrategyPipeline._03_filter_collision(pa_lf)
    moves   = StrategyPipeline._04_minimax_search(safe_lf, obs, BOARD)
    return moves
```

- [ ] **Step 4: Run full smoke test**

```
python 138-BoardCache.py
```

Expected output ends with `Smoke test PASSED.` and `elapsed: X.XXXs` where X.XXX < 5.0.

- [ ] **Step 5: Verify timing and move validity**

Check that:
- `agent()` returns `[[0, <angle>, 100]]` or similar (planet 0 attacking planet 1)
- Elapsed < 1.0 s ideally

- [ ] **Step 6: Commit**

```bash
git add 138-BoardCache.py
git commit -m "feat: 138 — _04_minimax_search rewrite, agent() wiring, smoke test"
```

---

## Self-Review

**Spec coverage:**

| Spec requirement | Task |
|---|---|
| `df_planete_nature (id, radius, production, nature)` | Task 2 |
| `df_planete_pos` sliding window, step 1 | Task 3 |
| `df_fleet` sync, `_compute_fleet_arrival`, step 2 | Task 3 |
| `build_base_ships` full recompute, step 3 | Task 4 |
| Combat formula identical to `1_29_3.py` | Task 4 (`_combat_step`) |
| `build_df_s_slice` for `_02` input | Task 5 |
| `extract_horizon_dict` | Task 5 |
| `_apply_sim_fleet` (handles None) | Task 5 |
| `_evaluate_dict` | Task 5 |
| `_recompute_from_sim` dirty-row incremental | Task 6 |
| Immutable base (base df never mutated) | Task 6 test asserts this |
| `_02`, `_03` untouched | Task 1 |
| Comet evasion block untouched | Task 7 |
| Changed-ids shortcut preserved | Task 7 |
| Restricted c5 set from covered_srcs | Task 7 |
| top3_c5_base tie-breaker | Task 7 |
| `agent()` entry point | Task 7 |
| Smoke test < 5s | Task 7 |
| `df_fleet_sim` struct defined in Board | Task 2 (`__init__` stub) |

**Placeholder scan:** No TBD/TODO/similar found.

**Type consistency:** `sim_row` passed to `_apply_sim_fleet` is always `(id_src, id_tgt, step_tgt, ships_sent)` — a 4-tuple with indices 0,1,2,3. This is constructed consistently in Task 7's `_04_minimax_search` as `(c5[0], c5[1], c5[2], c5[4])` where c5 = `[id_src, id_tgt, step_tgt, angle, ships_sent]`. ✓

**One gap found and fixed:** `_recompute_from_sim`'s `arrivals_by_step` also needs real fleets that hit non-dirty planets adjacent to dirty planets — but since we only recompute dirty planet rows, non-dirty arrivals don't affect our output. ✓
