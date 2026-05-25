import math, importlib.util, os

def load_module():
    path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "66-One_angle_polars_updated.py"))
    spec = importlib.util.spec_from_file_location("agent66", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def test_swept_pair_static_inside():
    mod = load_module()
    assert mod.swept_pair_hit((0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0), 1.0)

def test_swept_pair_static_miss():
    mod = load_module()
    assert not mod.swept_pair_hit((0.0, 0.0), (0.0, 0.0), (5.0, 0.0), (5.0, 0.0), 1.0)

def test_swept_pair_fleet_through_static_planet():
    mod = load_module()
    assert mod.swept_pair_hit((0.0, 0.0), (10.0, 0.0), (5.0, 0.0), (5.0, 0.0), 0.5)

def test_swept_pair_fleet_misses_static_planet():
    mod = load_module()
    assert not mod.swept_pair_hit((0.0, 0.0), (10.0, 0.0), (5.0, 2.0), (5.0, 2.0), 0.5)

def test_swept_pair_tunneling_detected():
    """Fleet and planet cross paths — old point_to_segment_distance misses, swept detects."""
    mod = load_module()
    A, B   = (0.0, 0.5), (2.0, 0.5)   # fleet moves right at y=0.5
    P0, P1 = (1.0, 1.5), (1.0, -0.5)  # planet moves down through y=0.5
    r = 0.6
    assert mod.swept_pair_hit(A, B, P0, P1, r)
    # Confirm old static check would miss
    assert mod.point_to_segment_distance(P0, A, B) > r
    assert mod.point_to_segment_distance(P1, A, B) > r
