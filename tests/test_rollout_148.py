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


def _minimal_state(B=2, P=3, A=2, player_id=0, ships=None, owner=None, prod=None):
    """Build a RolloutState directly without going through init_rollout_state."""
    dtype = torch.float32
    s = ships if ships is not None else torch.zeros(B, P, dtype=dtype)
    o = owner if owner is not None else torch.full((B, P), -1, dtype=torch.long)
    pr = prod if prod is not None else torch.zeros(B, P, dtype=dtype)
    # Infer B and P from the actual tensors in case caller passed different shapes
    actual_B, actual_P = s.shape
    alive = torch.ones(actual_B, actual_P, dtype=torch.bool)
    arrivals = torch.zeros(actual_B, actual_P, mod.ARRIVALS_H, A, dtype=dtype)
    return mod.RolloutState(
        ships=s, owner=o, prod=pr, alive=alive,
        arrivals=arrivals, player_id=player_id, A=A, B=actual_B, P=actual_P,
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
