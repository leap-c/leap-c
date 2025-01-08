import numpy as np
from numpy.testing import assert_almost_equal

from leap_c.dynamics import create_dynamics_from_mpc
from leap_c.mpc import MPC


def test_casadi_dynamics(linear_mpc: MPC):
    dynamics = create_dynamics_from_mpc(linear_mpc)

    # test forward single sample
    x = np.array([0.5, 0.5])
    u = np.array([0.5])
    p = None

    x_next = dynamics(x, u, None)

    # test forward multiple samples
    x = np.array([[0.5, 0.5], [0.3, 0.2]])
    u = np.array([[0.5], [0.2]])
    p = None

    x_next_batch = dynamics(x, u, None)

    assert_almost_equal(x_next, x_next_batch[0])

    # test jacobian
    x = np.array([0.5, 0.5])
    u = np.array([0.5])
    # p = linear_system_default_param_creator.current_param

    x_next, Sx, Su = dynamics(x, u, p, with_sens=True)

    # test jacobian multiple samples
    x = np.array([[0.5, 0.5], [0.3, 0.2]])
    u = np.array([[0.5], [0.2]])
    # p = np.stack([linear_system_default_param_creator.current_param, linear_system_default_param_creator.current_param])

    x_next_batch, Sx_batch, Su_batch = dynamics(x, u, p, with_sens=True)

    assert_almost_equal(x_next, x_next_batch[0])
    assert_almost_equal(Sx, Sx_batch[0])
    assert_almost_equal(Su, Su_batch[0])
