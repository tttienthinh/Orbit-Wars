import pytest
from tests.conftest import StrategyPipeline


def test_includes_all_planets_as_sources(simple_obs):
    df_s, _ = StrategyPipeline._01_get_obs_dataframe(simple_obs, step=0, num_agents=2)
    coarse = StrategyPipeline._02_pre_all(df_s, ships_list=[4, 16, 64, 256])
    assert not coarse.empty
    sources = set(coarse["id_src"].unique())
    assert 0 in sources
    assert 1 in sources


def test_ships_sent_is_the_fixed_list(simple_obs):
    df_s, _ = StrategyPipeline._01_get_obs_dataframe(simple_obs, step=0, num_agents=2)
    coarse = StrategyPipeline._02_pre_all(df_s, ships_list=[4, 16, 64, 256])
    assert not coarse.empty
    for row_list in coarse["ships_sent"]:
        assert row_list == [4, 16, 64, 256]


def test_no_self_attacks(simple_obs):
    df_s, _ = StrategyPipeline._01_get_obs_dataframe(simple_obs, step=0, num_agents=2)
    coarse = StrategyPipeline._02_pre_all(df_s, ships_list=[4, 16, 64, 256])
    assert not coarse.empty
    assert (coarse["id_src"] != coarse["id"]).all()


def test_ships_sent_column_is_not_exploded(simple_obs):
    df_s, _ = StrategyPipeline._01_get_obs_dataframe(simple_obs, step=0, num_agents=2)
    coarse = StrategyPipeline._02_pre_all(df_s, ships_list=[4, 16, 64, 256])
    assert not coarse.empty
    assert coarse["ships_sent"].apply(lambda x: isinstance(x, list)).all()
