"""
Local sanity check for 19-Pipeline_Kaggle.ipynb inlined functions.
Run from project root: python tests/test_notebook19_sanity.py
Verifies inlined get_winner_data + augment_data + generate_combinations
produce results consistent with the existing processed data.
"""
import json, copy, math, warnings, sys
from pathlib import Path
import numpy as np
import pandas as pd
from itertools import combinations as _combinations

warnings.filterwarnings("ignore")

# Constants (same as notebook Cell 1)
MAX_SPEED = 6.0
PLANET_MARGIN = 0.1
NB_FUTURE_STEP = 10
NB_PLANETS = 44

# ── Paste Cell 2 functions verbatim ──────────────────────────────────────────
def get_winner_data(raw):
    winner_i = int(np.argmax(raw["rewards"]))
    new_data = []
    for step_list in raw["steps"]:
        obs = dict(step_list[winner_i]["observation"])
        obs["action"] = step_list[winner_i]["action"]
        new_data.append(obs)
    return new_data

def _fleet_speed(nb_ships):
    if nb_ships <= 1:
        return 1.0
    ratio = math.log(nb_ships) / math.log(1000.0)
    return 1.0 + (MAX_SPEED - 1.0) * max(0.0, min(1.0, ratio)) ** 1.5

def augment_data(data):
    fleet_rows, planet_rows = [], []
    for obs in data:
        s = obs["step"]
        for f in obs["fleets"]:
            fleet_rows.append({"step": s, "id": f[0], "owner": f[1],
                                "x": f[2], "y": f[3], "angle": f[4],
                                "from_planet_id": f[5], "ships": f[6]})
        for p in obs["planets"]:
            planet_rows.append({"step": s, "id": p[0], "owner": p[1],
                                 "x": p[2], "y": p[3], "radius": p[4],
                                 "ships": p[5], "production": p[6]})
    if not fleet_rows or not planet_rows:
        return data
    df_fleet   = pd.DataFrame(fleet_rows).drop_duplicates(["step", "id"]).reset_index(drop=True)
    df_planets = pd.DataFrame(planet_rows).drop_duplicates(["step", "id"]).reset_index(drop=True)
    df_fleet_last = (
        df_fleet
        .assign(first_step=lambda df: df.groupby("id")["step"].transform("min"))
        .groupby("id").last().reset_index()
        .assign(
            speed=lambda df: df["ships"].apply(_fleet_speed),
            dx=lambda df: df["speed"] * np.cos(df["angle"]),
            dy=lambda df: df["speed"] * np.sin(df["angle"]),
            next_step=lambda df: df["step"] + 1,
        )
        .merge(df_planets, left_on="next_step", right_on="step",
               suffixes=("_fleet", "_planet"), how="inner")
        .assign(
            t=lambda df: (
                (df["dx"] * (df["x_planet"] - df["x_fleet"]) +
                 df["dy"] * (df["y_planet"] - df["y_fleet"])) / df["speed"] ** 2
            ).clip(lower=0, upper=1),
            distance=lambda df: np.sqrt(
                (df["x_fleet"] + df["dx"] * df["t"] - df["x_planet"]) ** 2 +
                (df["y_fleet"] + df["dy"] * df["t"] - df["y_planet"]) ** 2),
            on_target=lambda df: df["distance"] < df["radius"] + PLANET_MARGIN,
            angle_3=lambda df: df["angle"].round(3),
        )
        .query("on_target")
        .groupby("id_fleet").first()
        .set_index(["first_step", "from_planet_id", "angle_3", "ships_fleet"])
    )
    fleet_arrival = df_fleet_last["id_planet"].to_dict()
    for step_i, obs in enumerate(data):
        if step_i == 0:
            continue
        new_actions = []
        for from_pid, angle, ships in obs["action"]:
            key = (step_i, from_pid, round(angle, 3), ships)
            if key in fleet_arrival:
                new_actions.append([from_pid, angle, ships, fleet_arrival[key]])
        data[step_i - 1]["action"] = new_actions
    return data

def generate_combinations(obs):
    actions = obs["action"]
    nb = len(actions)
    rows = []
    for nb_done in range(nb):
        for done_idx in _combinations(range(nb), nb_done):
            done = [actions[i] for i in done_idx]
            remaining = [actions[i] for i in range(nb) if i not in done_idx]
            for action_to_do in remaining:
                rows.append({"obs": obs, "done_set": done, "action_to_do": action_to_do})
    rows.append({"obs": obs, "done_set": actions, "action_to_do": None})
    return rows

# ── Tests ─────────────────────────────────────────────────────────────────────
root = Path("11-download_logs")

# Test 1: augment_data produces correct dest planet
with open(root / "00-raw/episode-76319029.json") as f:
    raw = json.load(f)
winner = get_winner_data(raw)
augmented = augment_data(winner)
step_41 = next(o for o in augmented if o["step"] == 41)
assert step_41["action"] == [[0, 5.199833393096924, 70, 8]], \
    f"augment_data broken: {step_41['action']}"
print("[PASS] augment_data: dest planet 8 confirmed for step 41")

# Test 2: generate_combinations count matches existing 03-combinations
rows = []
for obs in augmented:
    if obs.get("action"):
        rows.extend(copy.deepcopy(generate_combinations(obs)))
with open(root / "03-combinations/episode-76319029.json") as f:
    expected_rows = json.load(f)
assert len(rows) == len(expected_rows), \
    f"generate_combinations count mismatch: {len(rows)} vs {len(expected_rows)}"
print(f"[PASS] generate_combinations: {len(rows)} rows matches existing 03-combinations")

print("\nAll sanity checks passed.")
