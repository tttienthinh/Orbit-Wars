"""
18-Play_ML.py — MLP inference agent for Orbit Wars.

Uses the trained OrbitMLP (multi-task MLP) to predict actions turn-by-turn.
No label remapping needed — the MLP uses slots 0–43 directly for from and to,
with slot 44 as the stop signal for the from-head.
"""

import math
import numpy as np
import torch
from pathlib import Path

from pipeline.features import (
    NB_FUTURE_STEP,
    build_slot_map,
    extract_features,
    slot_to_planet_id,
)
from pipeline.model import OrbitMLP

# NOTE: pipeline.env_sim is imported lazily (inside predict_actions) to avoid
# a Windows + OpenSpiel crash when the module is loaded at script-init time
# via certain Python launch modes (PowerShell, direct script run).
# The function works fine when called at runtime.

MODEL_PATH      = Path("models/mlp/orbit_mlp.pt")
FEATURE_VERSION = 1   # must match training
MAX_ACTIONS     = 5
STOP_SLOT       = 44  # class 44 in the from-head means "stop"

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

class MLPAgent:
    """
    Inference agent that wraps the trained OrbitMLP model.

    Parameters
    ----------
    model_path : str or Path
        Path to the checkpoint file (orbit_mlp.pt) containing
        'state_dict' and 'input_dim'.
    feature_version : int
        Feature version used during training (default 1).
    """

    def __init__(self, model_path=MODEL_PATH, feature_version=FEATURE_VERSION):
        model_path = Path(model_path)

        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        self.input_dim = int(checkpoint["input_dim"])

        self.model = OrbitMLP(self.input_dim)
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.eval()

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
            feat_tensor = torch.tensor(feat, dtype=torch.float32).unsqueeze(0)  # (1, input_dim)

            # Step 3 — forward pass
            with torch.no_grad():
                logits_from, logits_to, ships_pred = self.model(feat_tensor)

            # Step 4 — argmax to get from-slot; stop if STOP_SLOT
            from_slot = int(logits_from.argmax(dim=1).item())
            if from_slot == STOP_SLOT:
                break

            # Step 5 — argmax to get to-slot
            to_slot = int(logits_to.argmax(dim=1).item())

            # Step 6 — compute ship count from regression head
            nb_ships = int(max(1, round(float(ships_pred.item()))))

            # Step 7 — convert slots to planet IDs
            try:
                from_pid = slot_to_planet_id(from_slot, non_comet_ids, comet_ids)
                to_pid   = slot_to_planet_id(to_slot,   non_comet_ids, comet_ids)
            except IndexError:
                # Slot out of range for this game's planet list — stop safely
                break

            # Step 8 — compute launch angle
            angle = aiming_fn(from_pid, to_pid, raw_planets)

            # Step 9 — record action
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
        _agent_instance = MLPAgent()
    obs_dict = obs if isinstance(obs, dict) else dict(obs)
    raw_actions = _agent_instance.predict_actions(obs_dict)
    # Return 3-element moves [from_pid, angle, ships] for the environment
    return [[a[0], a[1], a[2]] for a in raw_actions]


# ── Smoke-test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Loading MLPAgent from", MODEL_PATH)
    ag = MLPAgent()
    print("MLPAgent loaded OK")
    print(f"  input_dim : {ag.input_dim}")
