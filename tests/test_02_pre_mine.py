import pytest
from tests.conftest import StrategyPipeline


def test_returns_only_mine_planet_as_source(simple_obs):
    df_s, _ = StrategyPipeline._01_get_obs_dataframe(simple_obs, step=0, num_agents=2)
    coarse = StrategyPipeline._02_pre_mine(df_s, player_id=0)
    assert not coarse.empty
    assert set(coarse["id_src"].unique()) == {0}


def test_ships_sent_is_list_starting_at_1(simple_obs):
    df_s, _ = StrategyPipeline._01_get_obs_dataframe(simple_obs, step=0, num_agents=2)
    coarse = StrategyPipeline._02_pre_mine(df_s, player_id=0)
    assert not coarse.empty
    first_list = coarse.iloc[0]["ships_sent"]
    assert isinstance(first_list, list)
    assert first_list[0] == 1


def test_enemy_player_id_returns_enemy_source(simple_obs):
    df_s, _ = StrategyPipeline._01_get_obs_dataframe(simple_obs, step=0, num_agents=2)
    coarse = StrategyPipeline._02_pre_mine(df_s, player_id=1)
    assert not coarse.empty
    assert set(coarse["id_src"].unique()) == {1}


def test_no_self_attacks(simple_obs):
    df_s, _ = StrategyPipeline._01_get_obs_dataframe(simple_obs, step=0, num_agents=2)
    coarse = StrategyPipeline._02_pre_mine(df_s, player_id=0)
    assert not coarse.empty
    assert (coarse["id_src"] != coarse["id"]).all()
