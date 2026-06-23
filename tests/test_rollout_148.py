import importlib.util, pathlib, pytest, torch

ROOT = pathlib.Path(__file__).parent.parent

def _load():
    import sys
    spec = importlib.util.spec_from_file_location(
        "agent148", ROOT / "148-H_turn_one_file_batch.py"
    )
    m = importlib.util.module_from_spec(spec)
    sys.modules["agent148"] = m
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


def test_run_turn_movement_not_none():
    obs_t = _obs_tensors()
    mem = mod.ProducerLiteMemory()
    _, _, movement = mod.run_turn(obs_t, config=mod._config_for(2), player_count=2, memory=mem)
    assert movement is not None
    assert hasattr(movement, "fleet_buckets")


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
