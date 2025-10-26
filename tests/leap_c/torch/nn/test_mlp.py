import numpy as np
import pytest
import torch

from leap_c.torch.nn.mlp import Mlp, MlpConfig


@pytest.mark.parametrize("single_output_dim", (True, False))
@pytest.mark.parametrize("as_tuple", (True, False))
def test_nn_mlp(single_output_dim: bool, as_tuple: bool):
    rng = np.random.default_rng()
    in_dim, out_dim, hidden_dims, batch = rng.integers(2, 10, size=4)
    if single_output_dim:
        out_dim = 1
    input_sizes = rng.integers(1, 10, size=in_dim)
    output_sizes = rng.integers(1, 10, size=out_dim)
    hidden_dims = rng.integers(1, 10, size=hidden_dims)

    cfg = MlpConfig(hidden_dims=hidden_dims)
    mlp = Mlp(input_sizes, output_sizes, cfg)

    assert mlp.mlp is not None
    assert mlp.param is None

    x = [torch.randn(batch, sz) for sz in input_sizes]
    if as_tuple:
        y = mlp(*x)
    else:
        y = mlp(torch.cat(x, dim=-1))

    if single_output_dim:
        expected_shape = (batch, sum(output_sizes))
        assert y.shape == expected_shape
    else:
        expected_shapes = ((batch, sz) for sz in output_sizes)
        assert all(yi.shape == shape for yi, shape in zip(y, expected_shapes))


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
