import copy
import kaggle_environments as ke

ENV_NAME = "orbit_wars"
NB_FUTURE_STEP = 10


def _to_game_action(action):
    """
    The step 2 augmentation adds dest_planet_id → 4-element actions.
    The env only accepts 3-element actions: [from_planet_id, angle, ships].
    """
    return action[:3]


def simulate_futures(obs, done_set, nb_future_step=NB_FUTURE_STEP):
    """
    Given an obs dict and a list of already-decided actions (done_set),
    injects obs into a fresh env, applies done_set (opponent = empty),
    then steps nb_future_step times with empty actions.

    done_set entries may be 3-element [from, angle, ships] or 4-element
    [from, angle, ships, dest] — dest is stripped automatically.

    Returns list of nb_future_step planet snapshots, each a list of
    [id, owner, x, y, radius, ships, production].
    """
    env = ke.make(ENV_NAME, debug=False)
    env.reset()

    for key, val in obs.items():
        env.state[0].observation[key] = val
        env.state[1].observation[key] = val

    game_done_set = [_to_game_action(a) for a in done_set]
    env.step([game_done_set, []])

    snapshots = []
    last_planets = copy.deepcopy(list(env.state[0].observation.get("planets", [])))
    for _ in range(nb_future_step):
        if env.done:
            # Game ended early — repeat last known planet state to pad to nb_future_step
            snapshots.append(last_planets)
            continue
        env.step([[], []])
        last_planets = copy.deepcopy(list(env.state[0].observation.get("planets", [])))
        snapshots.append(last_planets)
    return snapshots
