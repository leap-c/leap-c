import numpy as np
import pytest

from leap_c.ocp_env import OCPEnv


def test_env_reset(all_ocp_env: list[OCPEnv]):
    """Test the reset functionality of OCP environments.

    This test function verifies that the reset method of OCP environments produces consistent initial states when called with
    the same seed.

    Args:
        all_ocp_env (list[OCPEnv]): List of OCP environment instances to test.

    The test:
    1. Calls reset twice with same seed (0)
    2. Verifies state is not None
    3. Asserts both calls return same initial state vectors
    """
    for env in all_ocp_env:
        (x, p), _ = env.reset(seed=0)
        assert x is not None
        (x_, p_), _ = env.reset(seed=0)

        assert np.allclose(x, x_)


def test_env_step(all_ocp_env: list[OCPEnv]):
    """Test if the environment step function is deterministic.

    Tests if the environment's step function produces the same output when called with the same initial state and action.

    Args:
        all_ocp_env (list[OCPEnv]): List of optimal control problem environments to test.

    The test performs the following steps for each environment:
    1. Resets the environment with a fixed seed
    2. Performs a step with zero action
    3. Repeats the process
    4. Compares the state outputs to ensure they are identical
    """

    for env in all_ocp_env:
        env.reset(seed=0)
        action = np.zeros(env.action_space.shape)
        (x_next, p_next), _, _, _, _ = env.step(action)

        env.reset(seed=0)
        action = np.zeros(env.action_space.shape)
        (x_next_, p_next_), _, _, _, _ = env.step(action)

        assert np.allclose(x_next, x_next_)


def test_env_terminates(all_ocp_env: list[OCPEnv]):
    """Test if all ocp environments terminate correctly when applying minimum and maximum control inputs.

    This test ensures that the pendulum on cart environment terminates properly when applying either minimum or maximum control
    inputs continuously. It checks both termination conditions and verifies that the episode ends with a natural termination
    (term=True) rather than a truncation (trunc=False).

    Args:
        all_ocp_env (list[OCPEnv]): List of optimal control problem environments to test.
    """

    for env in all_ocp_env:
        env.reset(seed=0)

        for action in [
            env.mpc.ocp.constraints.lbu,
            env.mpc.ocp.constraints.ubu,
        ]:
            for _ in range(1000):
                _, _, term, trunc, _ = env.step(action)
                if term:
                    break
            assert term
            assert not trunc


if __name__ == "__main__":
    pytest.main([__file__])
