import numpy as np

import leap_c.examples
import leap_c.rl  # noqa: F401
from leap_c.examples.pointmass.env import PointMassEnv
from leap_c.examples.pointmass.mpc import PointMassMPC

MAX_FINAL_DIST = 1.0
MAX_FINAL_VEL = 0.1


def run_closed_loop(
    mpc: PointMassMPC,
    env: PointMassEnv,
    n_iter: int = int(2e2),
) -> np.ndarray:
    """Run a closed-loop simulation of a point mass system using a given model predictive controller (MPC) and environment.

    Args:
        mpc (PointMassMPC): The model predictive controller to use for generating actions.
        env (PointMassEnv): The environment representing the point mass system.
        n_iter (int, optional): The number of iterations to run the simulation. Defaults to 200.

    Returns:
        np.ndarray: A numpy array containing the states and actions over the simulation. The array has shape (n_iter, 6),
                    where the first 4 columns are the states and the last 2 columns are the actions.

    """
    s, _ = env.reset()

    states = np.zeros((n_iter, 4))
    states[0, :] = s
    actions = np.zeros((n_iter, 2))
    for i in range(n_iter - 1):
        actions[i, :] = mpc.policy(state=states[i, :], p_global=None)[0]
        states[i + 1, :], _, _, _, _ = env.step(actions[i, :])

    return np.hstack([states, actions])


def test_run_closed_loop(
    learnable_point_mass_mpc_m: PointMassMPC, point_mass_env: PointMassEnv, n_iter: int = int(2e2)
) -> None:
    """Test the closed-loop performance of a learnable point mass MPC (Model Predictive Control) in a point mass environment.

    Args:
    learnable_point_mass_mpc_m (PointMassMPC): The learnable point mass MPC to be tested.
    point_mass_env (PointMassEnv): The point mass environment in which the MPC will be tested.
    n_iter (int, optional): The number of iterations for the closed-loop simulation. Default is 200.

    Asserts:
    - The final position of the point mass is close to the origin (within MAX_FINAL_DIST).
    - The final velocity of the point mass is close to zero (within MAX_FINAL_VEL).

    """
    sim_data = run_closed_loop(mpc=learnable_point_mass_mpc_m, env=point_mass_env, n_iter=n_iter)
    assert np.linalg.norm(sim_data[-1, :2]) < MAX_FINAL_DIST  # Check that the final position is close to the origin
    assert np.linalg.norm(sim_data[-1, 2:4]) < MAX_FINAL_VEL  # Check that the final velocity is close to zero
