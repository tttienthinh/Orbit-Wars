"""
17-Play_XGBoost.py — XGBoost inference agent for Orbit Wars.

Uses three trained models (from, to, ships) to predict actions turn-by-turn.
Label remapping arrays (from_classes.npy, to_classes.npy) are loaded at init
to convert XGBoost contiguous predictions back to original planet slots.
"""

import math
import numpy as np
import xgboost as xgb
from pathlib import Path

from pipeline.features import (
    NB_FUTURE_STEP,
    build_slot_map,
    extract_features,
    slot_to_planet_id,
)
# NOTE: pipeline.env_sim is imported lazily (inside predict_actions) to avoid
# a Windows + OpenSpiel crash when the module is loaded at script-init time
# via certain Python launch modes (PowerShell, direct script run).
# The function works fine when called at runtime.

MODEL_DIR       = Path("models/xgboost")
FEATURE_VERSION = 1   # must match training
MAX_ACTIONS     = 5
STOP_SLOT       = 44  # class 44 in the from-model means "stop"

CENTER_X = 50.0
CENTER_Y = 50.0
MAX_SPEED = 6.0
PLANET_MARGIN = 0.1


# ── Aiming helpers (orbital-intercept) ───────────────────────────────────────

def _fleet_speed(ships):
    if ships <= 1:
        return 1.0
    ratio = math.log(ships) / math.log(1000.0)
    ratio = max(0.0, min(1.0, ratio))
    return 1.0 + (MAX_SPEED - 1.0) * (ratio ** 1.5)


def _get_distance(x0, y0, x1=CENTER_X, y1=CENTER_Y):
    return math.sqrt((x0 - x1) ** 2 + (y0 - y1) ** 2)


def _is_moving(planet_x, planet_y, planet_radius):
    return _get_distance(planet_x, planet_y) + planet_radius < 50


def _next_position(x, y, angular_velocity, t):
    r = _get_distance(x, y)
    angle = math.atan2(y - CENTER_Y, x - CENTER_X)
    new_angle = angle + angular_velocity * t
    return (math.cos(new_angle) * r + CENTER_X,
            math.sin(new_angle) * r + CENTER_Y)


def _get_intercept_t(x0, y0, x1, y1, angular_velocity, speed, r0=0, r1=0, t_max=100):
    for t in range(t_max):
        nx, ny = _next_position(x1, y1, angular_velocity, t)
        if _get_distance(x0, y0, nx, ny) - t * speed < r0 + r1 + PLANET_MARGIN:
            return True, t
    return False, t_max


def default_aiming_fn(from_pid, to_pid, raw_planets, angular_velocity=0.0):
    """
    Returns the launch angle from from_pid toward to_pid,
    accounting for orbital motion when applicable.
    raw_planets: list of [id, owner, x, y, radius, ships, production]
    """
    pmap = {p[0]: p for p in raw_planets}
    src = pmap.get(from_pid)
    tgt = pmap.get(to_pid)
    if src is None or tgt is None:
        return 0.0

    x0, y0, r0 = src[2], src[3], src[4]
    x1, y1, r1 = tgt[2], tgt[3], tgt[4]
    ships = src[5]
    speed = _fleet_speed(max(1, int(ships)))

    if _is_moving(x1, y1, r1):
        ok, t = _get_intercept_t(x0, y0, x1, y1, angular_velocity, speed, r0=r0, r1=r1)
        if ok:
            nx, ny = _next_position(x1, y1, angular_velocity, t)
            return math.atan2(ny - y0, nx - x0)

    return math.atan2(y1 - y0, x1 - x0)


# ── Agent class ───────────────────────────────────────────────────────────────

class XGBoostAgent:
    """
    Inference agent that wraps three XGBoost models trained in 17-Train_XGBoost.py.

    Parameters
    ----------
    model_dir : str or Path
        Directory containing from.ubj, to.ubj, ships.ubj,
        from_classes.npy, and to_classes.npy.
    feature_version : int
        Feature version used during training (default 1).
    """

    def __init__(self, model_dir=MODEL_DIR, feature_version=FEATURE_VERSION):
        model_dir = Path(model_dir)

        self.m_from = xgb.XGBClassifier()
        self.m_from.load_model(model_dir / "from.ubj")

        self.m_to = xgb.XGBClassifier()
        self.m_to.load_model(model_dir / "to.ubj")

        self.m_ships = xgb.XGBRegressor()
        self.m_ships.load_model(model_dir / "ships.ubj")

        # from_classes[i] = original planet slot for remapped index i
        self.from_classes = np.load(model_dir / "from_classes.npy")
        # to_classes[i]   = original planet slot for remapped index i
        self.to_classes   = np.load(model_dir / "to_classes.npy")

        self.feature_version = feature_version

    # ── core inference ────────────────────────────────────────────────────────

    def predict_actions(self, obs, aiming_fn=None):
        """
        Predict a list of actions for one game turn.

        Parameters
        ----------
        obs : dict
            Raw observation dict from the Kaggle Orbit Wars environment.
        aiming_fn : callable or None
            Signature: aiming_fn(from_pid, to_pid, raw_planets) -> angle (float).
            Defaults to default_aiming_fn (orbital-intercept).

        Returns
        -------
        list of [from_planet_id, angle, nb_ships, to_planet_id]
        """
        if aiming_fn is None:
            angular_velocity = obs.get("angular_velocity", 0.0)
            aiming_fn = lambda fp, tp, planets: default_aiming_fn(
                fp, tp, planets, angular_velocity=angular_velocity
            )

        non_comet_ids, comet_ids = build_slot_map(obs)
        raw_planets = obs.get("planets", [])

        done_set = []
        actions  = []

        for _ in range(MAX_ACTIONS):
            # Step 1 — simulate future planet states given already-decided actions
            from pipeline.env_sim import simulate_futures  # lazy import (see top-of-file note)
            future_planets = simulate_futures(obs, done_set, NB_FUTURE_STEP)

            # Step 2 — extract features
            feat = extract_features(
                future_planets, non_comet_ids, comet_ids,
                version=self.feature_version
            )
            feat_2d = feat.reshape(1, -1)  # (1, n_features)

            # Step 3 — predict from-slot
            from_idx  = int(self.m_from.predict(feat_2d)[0])
            from_slot = int(self.from_classes[from_idx])   # original slot

            # Stop condition: from-slot 44 means "do nothing more"
            if from_slot == STOP_SLOT:
                break

            # Step 4 — predict to-slot
            to_idx  = int(self.m_to.predict(feat_2d)[0])
            to_slot = int(self.to_classes[to_idx])         # original slot

            # Step 5 — predict ship count (conditioned on from + to)
            from_oh = np.eye(44, dtype=np.float32)[np.clip(from_slot, 0, 43)]
            to_oh   = np.eye(44, dtype=np.float32)[np.clip(to_slot,   0, 43)]
            feat_ships = np.concatenate([feat, from_oh, to_oh]).reshape(1, -1)
            nb_ships = int(max(1, round(float(self.m_ships.predict(feat_ships)[0]))))

            # Step 6 — convert slots to planet IDs
            try:
                from_pid = slot_to_planet_id(from_slot, non_comet_ids, comet_ids)
                to_pid   = slot_to_planet_id(to_slot,   non_comet_ids, comet_ids)
            except IndexError:
                # Slot out of range for this game's planet list — stop safely
                break

            # Step 7 — compute launch angle
            angle = aiming_fn(from_pid, to_pid, raw_planets)

            # Step 8 — record action
            action = [from_pid, angle, nb_ships, to_pid]
            actions.append(action)
            done_set.append(action)

        return actions


# ── Kaggle-compatible entry point ─────────────────────────────────────────────

_agent_instance = None


def agent(obs):
    """Top-level function expected by Kaggle Orbit Wars."""
    global _agent_instance
    if _agent_instance is None:
        _agent_instance = XGBoostAgent()
    obs_dict = obs if isinstance(obs, dict) else dict(obs)
    raw_actions = _agent_instance.predict_actions(obs_dict)
    # Return 3-element moves [from_pid, angle, ships] for the environment
    return [[a[0], a[1], a[2]] for a in raw_actions]


# ── Smoke-test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Loading XGBoostAgent from", MODEL_DIR)
    ag = XGBoostAgent()

    print(f"  from  model: {len(ag.from_classes)} classes")
    print(f"    from_classes (remapped->original slot): {ag.from_classes.tolist()}")

    print(f"  to    model: {len(ag.to_classes)} classes")
    print(f"    to_classes  (remapped->original slot): {ag.to_classes.tolist()}")

    print(f"  ships model: loaded (regressor)")
    print("Agent loaded successfully.")
