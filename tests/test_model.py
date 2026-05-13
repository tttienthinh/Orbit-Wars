import torch
from pipeline.model import OrbitMLP


def test_output_shapes():
    model = OrbitMLP(input_dim=2200)
    model.eval()
    x = torch.randn(4, 2200)
    lf, lt, ps = model(x)
    assert lf.shape == (4, 45), f"Expected (4,45), got {lf.shape}"
    assert lt.shape == (4, 44), f"Expected (4,44), got {lt.shape}"
    assert ps.shape == (4,),    f"Expected (4,), got {ps.shape}"


def test_has_dropout():
    model = OrbitMLP(input_dim=2200)
    assert any(isinstance(m, torch.nn.Dropout) for m in model.modules()), \
        "Expected Dropout layers in encoder"


def test_param_count_larger():
    model = OrbitMLP(input_dim=2200)
    n = sum(p.numel() for p in model.parameters())
    assert n > 2_000_000, f"Expected >2M params for wider net, got {n:,}"
