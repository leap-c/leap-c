from gymnasium.spaces import Box
import numpy as np
import pytest
import torch


from leap_c.torch.nn.scale import min_max_scaling


def test_min_max_scaling():
    """Test the min_max_scaling function."""
    # Create a Box space with low and high bounds
    space = Box(low=0, high=3, shape=(3,))

    # Create a tensor to scale
    x = torch.tensor([[1.5, 1.5, 1.5], [3.0, 3.0, 3.0]])

    # Scale the tensor
    scaled_x = min_max_scaling(x, space)

    # Check the scaled values
    assert (scaled_x[0] == 0.5).all()
    assert (scaled_x[1] == 1.0).all()

    # Check for error if low >= high or there is infinity
    low_invalid = np.array([0, 0, 1], dtype=np.float32)
    high_invalid = np.array([1, 1, 1], dtype=np.float32)
    space_invalid = Box(low=low_invalid, high=high_invalid, shape=(3,))

    x_invalid = torch.tensor([[1.5, 1.5, 1.5], [3.0, 3.0, 3.0]])

    with pytest.raises(ValueError):
        min_max_scaling(x_invalid, space_invalid)

    low_invalid_inf = np.array([0, 0, -np.inf], dtype=np.float32)
    high_invalid_inf = np.array([1, 1, 1], dtype=np.float32)
    space_invalid_inf = Box(low=low_invalid_inf, high=high_invalid_inf, shape=(3,))

    with pytest.raises(ValueError):
        min_max_scaling(x_invalid, space_invalid_inf)


def test_min_max_scaling_straight_through():
    """Test the min_max_scaling function with straight through."""
    # Create a Box space with low and high bounds
    space = Box(low=0, high=3, shape=(3,))

    # Create a tensor to scale
    x = torch.tensor([[1.5, 1.5, 1.5], [3.0, 3.0, 3.0]], requires_grad=True)

    # Scale the tensor
    scaled_x = min_max_scaling(x, space, straight_through=True)

    # Check the scaled values
    assert (scaled_x[0] == 0.5).all()
    assert (scaled_x[1] == 1.0).all()

    # Check the gradient
    scaled_x.sum().backward()
    assert (x.grad == 1).all()
