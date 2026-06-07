import pytest
import pandas as pd
from tests.conftest import StrategyPipeline


def test_new_api_returns_dataframe_for_mine_path(simple_obs):
    df_s, planet_disp = StrategyPipeline._01_get_obs_dataframe(simple_obs, step=0, num_agents=2)
    coarse = StrategyPipeline._02_pre_mine(df_s, player_id=0)
    result = StrategyPipeline._02_get_all_opportunities(coarse, df_s, planet_disp)
    assert isinstance(result, pd.DataFrame)
    assert not result.empty
    assert "angle" in result.columns
    assert "ships_sent" in result.columns
    assert "id_src" in result.columns


def test_reach_path_ships_only_from_fixed_list(simple_obs):
    df_s, planet_disp = StrategyPipeline._01_get_obs_dataframe(simple_obs, step=0, num_agents=2)
    coarse = StrategyPipeline._02_pre_all(df_s, ships_list=[4, 16, 64, 256])
    result = StrategyPipeline._02_get_all_opportunities(coarse, df_s, planet_disp)
    assert isinstance(result, pd.DataFrame)
    assert not result.empty
    assert set(result["ships_sent"].unique()).issubset({4, 16, 64, 256})


def test_empty_coarse_returns_empty_dataframe(simple_obs):
    df_s, planet_disp = StrategyPipeline._01_get_obs_dataframe(simple_obs, step=0, num_agents=2)
    result = StrategyPipeline._02_get_all_opportunities(pd.DataFrame(), df_s, planet_disp)
    assert isinstance(result, pd.DataFrame)
    assert result.empty
