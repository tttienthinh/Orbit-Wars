import pandas as pd
from tests.conftest import StrategyPipeline


def test_accepts_reach_matrix_as_second_arg():
    moves = StrategyPipeline._04_score_and_decide(
        pd.DataFrame(), pd.DataFrame(), player_id=0
    )
    assert moves == []


def test_existing_scoring_unaffected(simple_obs):
    df_s, planet_disp = StrategyPipeline._01_get_obs_dataframe(simple_obs, step=0, num_agents=2)
    coarse_mine = StrategyPipeline._02_pre_mine(df_s, 0)
    pa          = StrategyPipeline._02_get_all_opportunities(coarse_mine, df_s, planet_disp)
    safe        = StrategyPipeline._03_filter_collision(pa)
    reach       = pd.DataFrame()
    moves       = StrategyPipeline._04_score_and_decide(safe, reach, player_id=0)
    assert isinstance(moves, list)
    assert len(moves) > 0, "Expected at least one attack move for simple_obs"
    move = moves[0]
    assert len(move) == 3, "Move should be [id_src, angle, ships_sent]"
    assert move[0] == 0, "Source should be planet 0 (mine)"
