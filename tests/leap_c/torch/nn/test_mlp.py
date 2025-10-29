import numpy as np
import pytest
import torch

from leap_c.torch.nn.mlp import Mlp, MlpConfig


@pytest.mark.parametrize("arg_type", (tuple, list, int, np.int16, np.int32, np.int64))
@pytest.mark.parametrize("hidden_type", ("empty", "nonempty", "none"))
def test_nn_mlp__init__with_different_args_combination(arg_type: type, hidden_type: str):
    rng = np.random.default_rng()

    if arg_type in {tuple, list}:
        in_dim, out_dim = rng.integers(1, 10, size=2)
        input_sizes = arg_type(rng.integers(1, 10, size=in_dim))
        output_sizes = arg_type(rng.integers(1, 10, size=out_dim))
    else:
        input_sizes = arg_type(rng.integers(1, 10).item())
        output_sizes = arg_type(rng.integers(1, 10).item())

    match hidden_type:
        case "nonempty":
            ndim = rng.integers(1, 10).item()
            hidden_dims = rng.integers(1, 10, size=ndim)
        case "empty":
            hidden_dims = []
        case _:  # "none"
            hidden_dims = None

    cfg = MlpConfig(hidden_dims=hidden_dims)
    mlp = Mlp(input_sizes, output_sizes, cfg)

    assert (hidden_type == "nonempty" and mlp.mlp is not None and mlp.param is None) or (
        mlp.mlp is None and mlp.param is not None
    )


@pytest.mark.parametrize("single_output_dim", (True, False))
@pytest.mark.parametrize("as_tuple", (True, False))
def test_nn_mlp__forward__handles_shapes_as_expected(single_output_dim: bool, as_tuple: bool):
    rng = np.random.default_rng()
    in_dim, out_dim, hidden_dims, batch = rng.integers(2, 10, size=4)
    if single_output_dim:
        out_dim = 1
    input_sizes = rng.integers(1, 10, size=in_dim)
    output_sizes = rng.integers(1, 10, size=out_dim)
    hidden_dims = rng.integers(1, 10, size=hidden_dims)

    cfg = MlpConfig(hidden_dims=hidden_dims)
    mlp = Mlp(input_sizes, output_sizes, cfg)

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
