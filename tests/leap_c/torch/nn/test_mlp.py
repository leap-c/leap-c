import torch

from leap_c.torch.nn.mlp import Mlp, MlpConfig


def test_const_param_mlp():
    cfg = MlpConfig(hidden_dims=None)
    mlp = Mlp(input_sizes=3, output_sizes=2, mlp_cfg=cfg)

    assert mlp.mlp is None
    assert mlp.param is not None

    x = [torch.randn(4, 3)]
    y = mlp(*x)
    assert y.shape == (4, 2)
    assert torch.allclose(y, y[0].unsqueeze(0).expand_as(y))

    x = [torch.randn(4, 3)]
    y2 = mlp(*x)
    assert torch.allclose(y, y2)
