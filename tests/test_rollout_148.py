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
