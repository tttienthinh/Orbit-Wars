import pandas as pd
from tests.conftest import StrategyPipeline


def test_removes_path_crossing_sun():
    # (20,50) → (80,50) passes through sun center (50,50), dist=0
    coarse = pd.DataFrame({'x_src': [20.0], 'y_src': [50.0], 'x': [80.0], 'y': [50.0]})
    result = StrategyPipeline._sun_crossing_filter(coarse)
    assert len(result) == 0


def test_keeps_path_not_crossing_sun():
    # (1,10) → (99,10): closest point to (50,50) is (50,10), dist=40 > 10.1
    coarse = pd.DataFrame({'x_src': [1.0], 'y_src': [10.0], 'x': [99.0], 'y': [10.0]})
    result = StrategyPipeline._sun_crossing_filter(coarse)
    assert len(result) == 1


def test_filters_only_crossing_rows():
    coarse = pd.DataFrame({
        'x_src': [20.0,  1.0],
        'y_src': [50.0, 10.0],
        'x':     [80.0, 99.0],
        'y':     [50.0, 10.0],
    })
    result = StrategyPipeline._sun_crossing_filter(coarse)
    assert len(result) == 1
    assert float(result.iloc[0]['x_src']) == 1.0
