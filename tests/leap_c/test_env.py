import gymnasium as gym
import numpy as np
import pytest


def test_env_reset(all_env: list[gym.Env]):
    """Test the reset functionality of our environments.

    This test function verifies that the reset method of the environments produces
    consistent initial states when called with the same seed.

    Args:
        all_env (list[gym.Env]): List of environment instances to test.

    The test:
    1. Calls reset twice with same seed (0)
    2. Verifies state is not None
    3. Asserts both calls return same initial state vectors
    """
    for env in all_env:
        x, _ = env.reset(seed=0)
        assert x is not None
        x_, _ = env.reset(seed=0)

        assert np.allclose(x, x_)


def test_env_step(all_env: list[gym.Env]):
    """Test if the environment step function is deterministic.

    Tests if the environment's step function produces the same output when called with the same initial state and action.

    Args:
        all_env (list[OCPEnv]): List of optimal control problem environments to test.

    The test performs the following steps for each environment:
    1. Resets the environment with a fixed seed
    2. Performs a step with zero action
    3. Repeats the process
    4. Compares the state outputs to ensure they are identical
    """

    for env in all_env:
        env.reset(seed=0)
        action = np.zeros(env.action_space.shape)  # type:ignore
        x_next, _, _, _, _ = env.step(action)

        env.reset(seed=0)
        action = np.zeros(env.action_space.shape)  # type:ignore
        x_next_, _, _, _, _ = env.step(action)

        assert np.allclose(x_next, x_next_)


if __name__ == "__main__":
    pytest.main([__file__])
