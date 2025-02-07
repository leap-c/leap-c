import numpy as np
import pytest
from leap_c.examples.linear_system import LinearSystemMPC, LinearSystemOcpEnv
from leap_c.examples.pendulum_on_a_cart.env import PendulumOnCartSwingupEnv
from leap_c.examples.pendulum_on_cart import PendulumOnCartMPC
from leap_c.examples.point_mass import PointMassMPC, PointMassOcpEnv


def generate_batch_variation(
    nominal: np.ndarray, n_batch: int, delta: float = 0.05
) -> np.ndarray:
    """Set up test parameters for the linear system MPC.

    Args:
        learnable_linear_mpc (MPC): Linear system MPC with learnable parameters.
        n_batch (int): Number of test parameters to generate.
        delta (float, optional): Relative distance from nominal value.

    Returns:
    --------
    np.ndarray (n_batch, n_p_global, n_p_global): Test parameters for the linear system MPC where n_p_global is the number of
    global parameters. The last axis determines the parameter that is varied.
    """

    width = np.array([delta * val if np.abs(val) > 1e-6 else delta for val in nominal])

    # repeat mean into an array with shape (n_batch, n_param)
    batch_val = np.tile(nominal, (n_batch, 1))

    # repeat mean along a third axis
    batch_val = np.repeat(batch_val[:, :, np.newaxis], nominal.shape[0], axis=2)

    for i, idx in enumerate(np.arange(len(nominal))):
        batch_val[:, idx, i] = np.linspace(
            nominal[idx] - width[idx], nominal[idx] + width[idx], n_batch
        )

    return batch_val


def generate_batch_constant(val: np.ndarray, shape) -> np.ndarray:
    """Create a batched array by repeating a constant value.

    This function generates a batch of arrays by repeating a constant value array
    across specified dimensions.

    Args:
        val (np.ndarray): The constant value array to be repeated.
        shape (tuple): Target shape for the output batch. Should be a 3D shape
            (batch_size, feature_dim, sequence_length).

    Returns:
        np.ndarray: A 3D array of shape `shape` where the input `val` is repeated
            across batch and sequence dimensions.
    """
    batch_val = np.repeat(
        np.tile(val, (shape[0], 1))[:, :, np.newaxis],
        shape[2],
        axis=2,
    )

    return batch_val


@pytest.fixture(scope="session")
def linear_mpc():
    """Fixture for the linear system MPC."""
    return LinearSystemMPC()


@pytest.fixture(scope="session")
def point_mass_mpc():
    """Fixture for the point mass MPC."""
    return PointMassMPC()


@pytest.fixture(scope="session")
def pendulum_on_cart_mpc() -> PendulumOnCartMPC:
    """Fixture for the pendulum on cart MPC."""
    return PendulumOnCartMPC()


@pytest.fixture(scope="session")
def n_batch() -> int:
    return 4


@pytest.fixture(scope="session")
def learnable_linear_mpc(n_batch: int) -> LinearSystemMPC:
    """Fixture for the linear system MPC with learnable parameters."""
    return LinearSystemMPC(
        learnable_params=["A", "B", "Q", "R", "b", "f", "V_0"], n_batch=n_batch
    )


@pytest.fixture(scope="session")
def learnable_point_mass_mpc(n_batch: int) -> PointMassMPC:
    """Fixture for the linear system MPC with learnable parameters."""
    return PointMassMPC(learnable_params=["m", "c"], n_batch=n_batch)


@pytest.fixture(scope="session")
def linear_system_ocp_env(learnable_linear_mpc: LinearSystemMPC) -> LinearSystemOcpEnv:
    return LinearSystemOcpEnv(learnable_linear_mpc, render_mode="rgb_array")


@pytest.fixture(scope="session")
def point_mass_ocp_env(learnable_point_mass_mpc: PointMassMPC) -> PointMassOcpEnv:
    return PointMassOcpEnv(learnable_point_mass_mpc)


@pytest.fixture(scope="session")
def linear_mpc_p_global(
    learnable_linear_mpc: LinearSystemMPC, n_batch: int
) -> np.ndarray:
    """Fixture for the global parameters of the linear system MPC."""
    return generate_batch_variation(
        learnable_linear_mpc.ocp_solver.acados_ocp.p_global_values, n_batch
    )


@pytest.fixture(scope="session")
def learnable_pendulum_on_cart_mpc_ext_cost(n_batch: int) -> PendulumOnCartMPC:
    """Fixture for the pendulum on cart MPC with learnable parameters, using a general quadratic cost."""
    return PendulumOnCartMPC(
        learnable_params=["M", "m", "g", "L11", "c1"],
        n_batch=n_batch,
        least_squares_cost=False,
    )


@pytest.fixture(scope="session")
def learnable_pendulum_on_cart_mpc_lls_cost(n_batch: int) -> PendulumOnCartMPC:
    """Fixture for the pendulum on cart MPC with learnable parameters, using Linear Least Squares cost."""
    return PendulumOnCartMPC(
        learnable_params=["M", "m", "g", "L11", "xref1"],
        n_batch=n_batch,
        least_squares_cost=True,
    )


@pytest.fixture(scope="session")
def pendulum_on_cart_ocp_swingup_env() -> PendulumOnCartSwingupEnv:
    return PendulumOnCartSwingupEnv(render_mode="rgb_array")


@pytest.fixture(scope="session")
def all_env(
    pendulum_on_cart_ocp_swingup_env: PendulumOnCartSwingupEnv,
):
    return [
        pendulum_on_cart_ocp_swingup_env,
    ]


@pytest.fixture(scope="session")
def pendulum_on_cart_p_global(
    learnable_pendulum_on_cart_mpc_lls_cost: PendulumOnCartMPC, n_batch: int
) -> np.ndarray:
    """Fixture for the global parameters of the pendulum on cart MPC."""
    return generate_batch_variation(
        learnable_pendulum_on_cart_mpc_lls_cost.ocp_solver.acados_ocp.p_global_values,
        n_batch,
    )
