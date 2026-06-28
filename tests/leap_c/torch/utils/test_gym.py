import numpy as np
import torch
from gymnasium import spaces

from leap_c.torch.utils import gym as torch_gym


def test_flatten_unflatten_box_preserves_gradients():
    space = spaces.Box(-1.0, 1.0, shape=(2, 2))
    base = torch.arange(12.0, requires_grad=True)
    x = base.reshape(3, 2, 2)

    flat = torch_gym.flatten(space, x)
    out = torch_gym.unflatten(space, flat)

    assert flat.shape == (3, 4)
    torch.testing.assert_close(out, x)
    out.sum().backward()
    assert base.grad is not None


def test_flatten_unflatten_dict_preserves_structure_and_gradients():
    space = spaces.Dict(
        {
            "a": spaces.Box(-1.0, 1.0, shape=(2,)),
            "b": spaces.Box(-1.0, 1.0, shape=(1, 2)),
        }
    )
    x = {
        "a": torch.tensor([[1.0, 2.0]], requires_grad=True),
        "b": torch.tensor([[[3.0, 4.0]]], requires_grad=True),
    }

    flat = torch_gym.flatten(space, x)
    out = torch_gym.unflatten(space, flat)

    loss = out["a"].sum() + out["b"].sum()

    assert flat.shape == (1, 4)
    torch.testing.assert_close(out["a"], x["a"])
    torch.testing.assert_close(out["b"], x["b"])
    loss.backward()
    assert x["a"].grad is not None
    assert x["b"].grad is not None


def test_flatten_unflatten_tuple():
    space = spaces.Tuple(
        (
            spaces.Box(np.array([-1.0]), np.array([1.0])),
            spaces.Box(-1.0, 1.0, shape=(2,)),
        )
    )
    x = (torch.tensor([[1.0]]), torch.tensor([[2.0, 3.0]]))

    flat = torch_gym.flatten(space, x)
    out = torch_gym.unflatten(space, flat)

    torch.testing.assert_close(flat, torch.tensor([[1.0, 2.0, 3.0]]))
    torch.testing.assert_close(out[0], x[0])
    torch.testing.assert_close(out[1], x[1])


def test_flatten_space_returns_box():
    space = spaces.Dict({"a": spaces.Box(-1.0, 1.0, shape=(2,))})

    flat_space = torch_gym.flatten_space(space)

    assert isinstance(flat_space, spaces.Box)
    assert flat_space.shape == (2,)
