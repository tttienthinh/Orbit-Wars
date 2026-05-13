import json
import numpy as np
from pathlib import Path

NB_PLANETS     = 44   # 40 non-comets + 4 comets
NB_FUTURE_STEP = 10


def build_slot_map(obs):
    """
    Returns (non_comet_ids sorted asc, comet_ids sorted asc) from obs dict.
    Uses obs['comet_planet_ids'] directly if available; falls back to empty list.
    Slot mapping: non_comet[i] -> slot i,  comet[j] -> slot 40+j.
    """
    comet_ids = sorted(obs.get("comet_planet_ids", []))
    comet_set = set(comet_ids)
    all_ids = [p[0] for p in obs["planets"]]
    non_comet_ids = sorted(pid for pid in all_ids if pid not in comet_set)
    return non_comet_ids, comet_ids


def planet_id_to_slot(planet_id, non_comet_ids, comet_ids):
    """Returns slot 0-43, or -1 if unknown."""
    if planet_id in comet_ids:
        return 40 + comet_ids.index(planet_id)
    if planet_id in non_comet_ids:
        return non_comet_ids.index(planet_id)
    return -1


def slot_to_planet_id(slot, non_comet_ids, comet_ids):
    """Inverse of planet_id_to_slot."""
    if slot >= 40:
        return comet_ids[slot - 40]
    return non_comet_ids[slot]


def extract_features(future_planets, non_comet_ids, comet_ids, version=1):
    """
    version=1 -> 2200 features: 44 * NB_FUTURE_STEP * {owner,ships,x,y,production}
    version=2 -> 1540 features: 44 * NB_FUTURE_STEP * {owner,ships,production}
                              + 11 * NB_FUTURE_STEP * {x,y}  (slots 0-10)
    """
    if version == 1:
        feat = np.zeros((NB_FUTURE_STEP, NB_PLANETS, 5), dtype=np.float32)
        for step_i, planets in enumerate(future_planets):
            for p in planets:
                pid, owner, x, y, radius, ships, prod = p[0], p[1], p[2], p[3], p[4], p[5], p[6]
                slot = planet_id_to_slot(pid, non_comet_ids, comet_ids)
                if slot < 0:
                    continue
                feat[step_i, slot] = [owner, ships, x, y, prod]
        return feat.flatten()
    else:
        state = np.zeros((NB_FUTURE_STEP, NB_PLANETS, 3), dtype=np.float32)
        pos   = np.zeros((NB_FUTURE_STEP, 11, 2), dtype=np.float32)
        for step_i, planets in enumerate(future_planets):
            for p in planets:
                pid, owner, x, y, radius, ships, prod = p[0], p[1], p[2], p[3], p[4], p[5], p[6]
                slot = planet_id_to_slot(pid, non_comet_ids, comet_ids)
                if slot < 0:
                    continue
                state[step_i, slot] = [owner, ships, prod]
                if 0 <= slot <= 10:
                    pos[step_i, slot] = [x, y]
        return np.concatenate([state.flatten(), pos.flatten()])


def build_feature_matrix(simulate_dir, version=1):
    """
    Reads all JSON files in simulate_dir (step 4 output).
    Returns (X, y_from, y_to, y_ships):
      X       -- float32 array, shape (N, n_features)
      y_from  -- float32 array, shape (N,), values 0-43=slot or 44=stop
      y_to    -- float32 array, shape (N,), values 0-43=slot or nan when stop
      y_ships -- float32 array, shape (N,), nan when stop

    Rows with malformed actions (missing dest_planet_id) are skipped.
    """
    simulate_dir = Path(simulate_dir)
    X_list, y_from_list, y_to_list, y_ships_list = [], [], [], []

    for json_file in sorted(simulate_dir.glob("*.json")):
        with open(json_file) as f:
            rows = json.load(f)
        if not rows:
            continue
        non_comet_ids, comet_ids = build_slot_map(rows[0]["obs"])

        for row in rows:
            feat = extract_features(
                row["future_planets"], non_comet_ids, comet_ids, version=version
            )

            action = row["action_to_do"]
            if action is None:
                X_list.append(feat)
                y_from_list.append(44)
                y_to_list.append(np.nan)
                y_ships_list.append(np.nan)
            elif len(action) == 4:
                from_pid, _angle, ships, to_pid = action
                from_slot = planet_id_to_slot(from_pid, non_comet_ids, comet_ids)
                to_slot   = planet_id_to_slot(to_pid,   non_comet_ids, comet_ids)
                X_list.append(feat)
                y_from_list.append(float(from_slot))
                y_to_list.append(float(to_slot))
                y_ships_list.append(float(ships))
            # else: malformed action (e.g. 3-element intercept without dest) — skip row

    X       = np.array(X_list,       dtype=np.float32)
    y_from  = np.array(y_from_list,  dtype=np.float32)
    y_to    = np.array(y_to_list,    dtype=np.float32)
    y_ships = np.array(y_ships_list, dtype=np.float32)
    return X, y_from, y_to, y_ships
