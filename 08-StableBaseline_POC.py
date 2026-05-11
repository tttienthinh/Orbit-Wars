"""
08-StableBaseline_POC.py — Gymnasium + MaskablePPO for Kaggle Orbit Wars

Sections:
  1. Constants & normalization ceilings
  2. build_observation()  — planets → flat float32 vector (OBS_SIZE = 320)
  3. compute_masks()      — valid source/target pairs → action mask
  4. decode_action()      — MultiDiscrete action → kaggle move list
  5. random_opponent()    — initial self-play adversary
  6. OrbitWarsEnv         — gym.Env wrapping kaggle orbit_wars
  7. train()              — MaskablePPO training loop
  8. make_kaggle_agent()  — load model, return agent(obs) callable

Install deps (if not already present):
  pip install stable-baselines3 sb3-contrib gymnasium

Run:
  python 08-StableBaseline_POC.py
"""

import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces

import kaggle_environments as ke
from kaggle_environments.envs.orbit_wars.orbit_wars import Planet

from stable_baselines3.common.callbacks import BaseCallback
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker


# =============================================================================
# 1) Constants & normalisation ceilings
# =============================================================================
MAX_PLANETS         = 40          # fixed observation width (pad with zeros)
MAX_OWNED           = 10          # owned-planet slots in the action space
CENTER_X, CENTER_Y  = 50.0, 50.0  # sun position
LOG_MAX_SHIPS       = math.log1p(1_000.0)   # log-normalisation ceiling for ships
MAX_PROD            = 10.0                   # production normalisation ceiling

FEATURES_PER_PLANET = 8           # see build_observation() below
OBS_SIZE            = MAX_PLANETS * FEATURES_PER_PLANET   # 320-dim observation


# =============================================================================
# Utility helpers
# =============================================================================

def _dist_center(x: float, y: float) -> float:
    return math.sqrt((x - CENTER_X) ** 2 + (y - CENTER_Y) ** 2)


def _raw_obs(state, player: int) -> dict:
    """Extract a plain dict observation for `player` from a kaggle state list."""
    try:
        o = state[player].observation
        return dict(o) if not isinstance(o, dict) else o
    except Exception:
        return {}


def _parse_planets(raw: dict) -> list:
    """Convert raw obs dict → list of Planet namedtuples."""
    return [Planet(*p) for p in raw.get("planets", [])]


# =============================================================================
# 2) Observation Builder
# =============================================================================

def build_observation(planets: list, player: int) -> np.ndarray:
    """
    Encode up to MAX_PLANETS planets into a flat float32 vector of length OBS_SIZE.

    Planets are sorted by distance from the sun so slot ordering is stable
    across steps even as ownership changes.  Padding slots remain all-zero
    (is_valid == 0 tells the network this slot is empty).

    Per-planet feature layout (FEATURES_PER_PLANET = 8):
        [0]  x / 100                          → [0, 1]
        [1]  y / 100                          → [0, 1]
        [2]  is_mine      (owner == player)   → {0, 1}
        [3]  is_enemy     (owner != -1,player)→ {0, 1}
        [4]  is_neutral   (owner == -1)       → {0, 1}
        [5]  log1p(ships) / log1p(1000)       → [0, 1]
        [6]  production / MAX_PROD            → [0, 1]
        [7]  is_valid     (1 = real slot)     → {0, 1}
    """
    sorted_p = sorted(planets, key=lambda p: _dist_center(p.x, p.y))
    obs = np.zeros(OBS_SIZE, dtype=np.float32)

    for i, p in enumerate(sorted_p[:MAX_PLANETS]):
        b = i * FEATURES_PER_PLANET
        obs[b + 0] = p.x / 100.0
        obs[b + 1] = p.y / 100.0
        obs[b + 2] = float(p.owner == player)
        obs[b + 3] = float(p.owner not in (-1, player))
        obs[b + 4] = float(p.owner == -1)
        obs[b + 5] = math.log1p(max(0, p.ships)) / LOG_MAX_SHIPS
        obs[b + 6] = min(p.production, MAX_PROD) / MAX_PROD
        obs[b + 7] = 1.0   # is_valid: this slot has a real planet

    return obs


# =============================================================================
# 3) Action Mask
# =============================================================================

def compute_masks(planets: list, player: int) -> np.ndarray:
    """
    Build a boolean mask of shape (MAX_OWNED * MAX_PLANETS,).

    Layout: mask[slot * MAX_PLANETS : (slot+1) * MAX_PLANETS] controls
    which target planets are valid for owned-planet slot `slot`.

    A target j within slot s is True iff:
      • `player` actually owns a planet in slot s (sorted by dist-from-center)
      • that source planet has >= 2 ships (can't send from empty garrison)
      • j is not the source planet's own sorted index (no self-launch)

    Safety guarantee: if nothing is sendable, mask[0] is forced True so
    MaskablePPO never sees an all-zero logit vector (which causes NaN loss).
    """
    sorted_p = sorted(planets, key=lambda p: _dist_center(p.x, p.y))
    owned    = [p for p in sorted_p if p.owner == player]
    # Map planet id → position in the sorted list (for self-exclusion)
    idx_of   = {p.id: i for i, p in enumerate(sorted_p)}
    n_real   = min(len(sorted_p), MAX_PLANETS)

    mask = np.zeros(MAX_OWNED * MAX_PLANETS, dtype=bool)

    for slot in range(min(len(owned), MAX_OWNED)):
        src = owned[slot]
        if src.ships < 2:
            continue                     # can't send ships from this planet
        src_idx = idx_of[src.id]
        base    = slot * MAX_PLANETS
        for j in range(n_real):
            if j != src_idx:             # exclude self-targeting
                mask[base + j] = True

    if not mask.any():
        mask[0] = True                   # safety valve
    return mask


# =============================================================================
# 4) Action Decoder
# =============================================================================

def decode_action(action: np.ndarray, planets: list, player: int) -> list:
    """
    Convert a MultiDiscrete action (shape MAX_OWNED,) into kaggle move triples.

    action[slot] = sorted-planet-index of the desired target.

    For each owned-planet slot (sorted by dist-from-center, up to MAX_OWNED):
      • Skip if source has < 2 ships or action points at itself.
      • Send 50% of ships; compute launch angle from src-centre → tgt-centre.

    Returns list of [from_planet_id, angle_radians, num_ships].
    """
    sorted_p = sorted(planets, key=lambda p: _dist_center(p.x, p.y))
    owned    = [p for p in sorted_p if p.owner == player]
    idx_map  = dict(enumerate(sorted_p[:MAX_PLANETS]))   # sorted-idx → Planet

    moves = []
    for slot, tgt_idx in enumerate(action):
        if slot >= len(owned):
            break
        src = owned[slot]
        if src.ships < 2:
            continue
        tgt = idx_map.get(int(tgt_idx))
        if tgt is None or tgt.id == src.id:
            continue
        send  = max(1, src.ships // 2)
        angle = math.atan2(tgt.y - src.y, tgt.x - src.x)
        moves.append([src.id, angle, send])

    return moves


# =============================================================================
# 5) Random Opponent (initial self-play adversary)
# =============================================================================

def random_opponent(raw: dict) -> list:
    """
    For each owned planet with ≥ 2 ships: pick a uniformly random target,
    send 50% of ships.  Serves as the first opponent during training.
    Replace with a rule-based or older PPO checkpoint for curriculum self-play.
    """
    player  = raw.get("player", 1)
    planets = _parse_planets(raw)
    owned   = [p for p in planets if p.owner == player and p.ships >= 2]

    moves = []
    for src in owned:
        cands = [p for p in planets if p.id != src.id]
        if not cands:
            continue
        tgt   = cands[np.random.randint(len(cands))]
        angle = math.atan2(tgt.y - src.y, tgt.x - src.x)
        moves.append([src.id, angle, max(1, src.ships // 2)])
    return moves


# =============================================================================
# 6) Gymnasium Environment
# =============================================================================

class OrbitWarsEnv(gym.Env):
    """
    Single-agent gymnasium wrapper around kaggle's orbit_wars environment.

    Design choices:
      • Player 0 = RL agent; Player 1 = self.opponent_fn.
      • A fresh kaggle env is created every episode (cleanest reset strategy;
        avoids internal state leaks from the kaggle engine).
      • Observation: flat Box(OBS_SIZE,) — planets sorted by dist from sun.
      • Action:      MultiDiscrete([MAX_PLANETS] * MAX_OWNED) — for each of
                     the up to MAX_OWNED owned planets, choose a target planet.
      • Masking:     action_masks() is consumed by ActionMasker + MaskablePPO.

    Reward shaping:
      +planets_owned   dense, given every step (≈0–20 per step)
      +10              one-time win bonus  (only on terminal step)
      −10              one-time loss penalty (only on terminal step)
    """

    metadata = {"render_modes": []}

    def __init__(self, opponent_fn=None, episode_steps: int = 300):
        super().__init__()
        self.opponent_fn   = opponent_fn or random_opponent
        self.episode_steps = episode_steps

        # Observation space: all features normalised to [0, 1]
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(OBS_SIZE,), dtype=np.float32
        )

        # Action space: MAX_OWNED independent discrete choices over MAX_PLANETS targets
        self.action_space = spaces.MultiDiscrete([MAX_PLANETS] * MAX_OWNED)

        self._ke_env   = None    # kaggle environment instance
        self._planets  = []      # current parsed Planet list (player-0 view)
        self._step_num = 0

    # ── Reset ──────────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Create a brand-new kaggle episode
        self._ke_env = ke.make(
            "orbit_wars",
            debug=False,
            configuration={"episodeSteps": self.episode_steps},
        )
        self._ke_env.reset()
        self._step_num = 0

        raw           = _raw_obs(self._ke_env.state, player=0)
        self._planets = _parse_planets(raw)
        obs           = build_observation(self._planets, player=0)
        return obs, {}

    # ── Step ───────────────────────────────────────────────────────────────

    def step(self, action):
        # --- Opponent moves (uses player-1 observation) ---
        opp_raw  = _raw_obs(self._ke_env.state, player=1)
        opp_acts = self.opponent_fn(opp_raw)

        # --- Decode RL action → kaggle move list for player 0 ---
        my_acts = decode_action(action, self._planets, player=0)

        # --- Advance the kaggle game by one step ---
        self._ke_env.step([my_acts, opp_acts])
        self._step_num += 1
        state = self._ke_env.state

        # --- Parse new observation for player 0 ---
        raw           = _raw_obs(state, player=0)
        self._planets = _parse_planets(raw)
        obs           = build_observation(self._planets, player=0)

        # --- Dense reward: number of planets I own this step ---
        my_count = sum(1 for p in self._planets if p.owner == 0)
        reward   = float(my_count)

        # --- Terminal detection ---
        terminated = False
        if state:
            terminal_statuses = {"DONE", "INVALID", "ERROR"}
            if any(s.status in terminal_statuses for s in state):
                terminated = True
                # Sparse win/loss bonus on top of the dense reward
                r0 = state[0].reward if state[0].reward is not None else 0
                r1 = state[1].reward if state[1].reward is not None else 0
                if r0 > r1:
                    reward += 10.0    # win
                elif r0 < r1:
                    reward -= 10.0    # loss

        truncated = self._step_num >= self.episode_steps
        return obs, reward, terminated, truncated, {}

    # ── Action Masking ─────────────────────────────────────────────────────

    def action_masks(self) -> np.ndarray:
        """
        Called by ActionMasker / MaskablePPO before every forward pass.
        Returns a bool array of shape (MAX_OWNED * MAX_PLANETS,).
        """
        return compute_masks(self._planets, player=0)


# =============================================================================
# 7) Training
# =============================================================================

class RewardLogger(BaseCallback):
    """Print mean episode reward every `freq` environment steps."""

    def __init__(self, freq: int = 10_000):
        super().__init__()
        self.freq        = freq
        self._ep_buf     = []    # finished episode rewards
        self._cur_ep_rew = 0.0

    def _on_step(self) -> bool:
        # Accumulate reward for the current episode
        self._cur_ep_rew += float(self.locals["rewards"][0])
        if self.locals["dones"][0]:
            self._ep_buf.append(self._cur_ep_rew)
            self._cur_ep_rew = 0.0

        # Print every `freq` steps
        if self.num_timesteps % self.freq < self.training_env.num_envs:
            n_ep  = len(self._ep_buf)
            mean  = np.mean(self._ep_buf[-50:]) if self._ep_buf else float("nan")
            print(
                f"[step {self.num_timesteps:>8,}]  "
                f"mean_ep_reward={mean:>8.2f}  "
                f"episodes={n_ep}"
            )
        return True


def train(total_timesteps: int = 500_000, save_path: str = "orbit_wars_ppo") -> MaskablePPO:
    """
    Train a MaskablePPO agent vs a random opponent.

    Curriculum suggestion (after this POC):
      1. Swap random_opponent for the rule-based agent (07-claude_code.py).
      2. Periodically save checkpoints and use the previous checkpoint as opponent.
    """
    print("=== Orbit Wars — MaskablePPO POC ===")
    print(f"  obs_size   : {OBS_SIZE}  ({MAX_PLANETS} planets × {FEATURES_PER_PLANET} features)")
    print(f"  action     : MultiDiscrete([{MAX_PLANETS}] × {MAX_OWNED})")
    print(f"  opponent   : random agent")
    print(f"  timesteps  : {total_timesteps:,}\n")

    # Wrap with ActionMasker so MaskablePPO fetches the mask before every forward pass
    env = OrbitWarsEnv(opponent_fn=random_opponent, episode_steps=300)
    env = ActionMasker(env, lambda e: e.action_masks())

    model = MaskablePPO(
        "MlpPolicy",
        env,
        verbose=0,               # suppress SB3's own logger; we use RewardLogger
        n_steps=2048,            # steps collected per policy update
        batch_size=64,
        n_epochs=4,              # gradient passes per collected rollout
        learning_rate=3e-4,
        gamma=0.99,              # discount for future planet counts
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,           # entropy bonus — keeps exploration alive
    )

    model.learn(
        total_timesteps=total_timesteps,
        callback=RewardLogger(freq=10_000),
    )

    model.save(save_path)
    print(f"\nModel saved -> {save_path}.zip")
    return model


# =============================================================================
# 8) Kaggle-compatible agent from trained model
# =============================================================================

def make_kaggle_agent(model_path: str = "orbit_wars_ppo"):
    """
    Load a saved MaskablePPO model and return an agent(obs) callable
    compatible with kaggle_environments:

        ke.make("orbit_wars").run([make_kaggle_agent(), random_opponent])

    The returned function handles both dict and Configuration/Namespace obs
    objects so it works both locally (simulate.py) and on the Kaggle server.
    """
    model = MaskablePPO.load(model_path)

    def _agent(raw):
        # Parse obs regardless of whether kaggle passes a dict or Namespace
        if isinstance(raw, dict):
            player  = raw.get("player", 0)
            planets = _parse_planets(raw)
        else:
            player  = getattr(raw, "player", 0)
            planets = [Planet(*p) for p in getattr(raw, "planets", [])]

        obs    = build_observation(planets, player)
        mask   = compute_masks(planets, player)
        action, _ = model.predict(obs, action_masks=mask, deterministic=True)
        return decode_action(action, planets, player)

    return _agent


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    # --- Train ---
    model = train(total_timesteps=500_000, save_path="orbit_wars_ppo")

    # --- Sanity check: 5 episodes vs random opponent ---
    print("\n--- Sanity check: 5 games vs random opponent ---")
    agent = make_kaggle_agent("orbit_wars_ppo")
    wins = losses = draws = 0

    for g in range(5):
        ke_env = ke.make(
            "orbit_wars", debug=False,
            configuration={"episodeSteps": 300},
        )
        ke_env.run([agent, random_opponent])

        # ke_env.steps[-1] is a list of AgentState objects for the final step
        final = ke_env.steps[-1]
        r0 = final[0].reward if (final[0].reward is not None) else 0
        r1 = final[1].reward if (final[1].reward is not None) else 0

        if r0 > r1:
            wins   += 1; tag = "WIN"
        elif r0 < r1:
            losses += 1; tag = "LOSS"
        else:
            draws  += 1; tag = "DRAW"

        print(f"  Game {g + 1}: {tag}  (r0={r0:.1f}  r1={r1:.1f})")

    print(f"\n  Result: {wins}W  {losses}L  {draws}D  "
          f"(win-rate vs random = {wins / 5 * 100:.0f}%)")
