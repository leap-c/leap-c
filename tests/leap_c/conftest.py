from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pytest
from leap_c.examples.chain.env import ChainEnv
from leap_c.examples.chain.mpc import ChainMpc
from leap_c.examples.chain.utils import Ellipsoid
from leap_c.examples.pendulum_on_a_cart.env import PendulumOnCartSwingupEnv
from leap_c.examples.pendulum_on_a_cart.mpc import PendulumOnCartMPC
from leap_c.examples.pointmass.env import PointMassEnv
from leap_c.examples.pointmass.mpc import PointMassMPC
from leap_c.registry import (
    TRAINER_REGISTRY,
    create_default_cfg,
    create_task,
    create_trainer,
)
from leap_c.trainer import Trainer  # noqa: F401


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
def point_mass_mpc():
    """Fixture for the point mass MPC."""
    return PointMassMPC()


@pytest.fixture(scope="session")
def pendulum_on_cart_mpc() -> PendulumOnCartMPC:
    """Fixture for the pendulum on cart MPC."""
    return PendulumOnCartMPC()


@pytest.fixture(scope="session")
def n_batch() -> int:
    return 2


@pytest.fixture(scope="session")
def learnable_point_mass_mpc_different_params(n_batch: int) -> PointMassMPC:
    return PointMassMPC(
        learnable_params=[
            "m",
            "cx",
            "r_diag",
            "q_diag_e",
            "uref",
            "xref_e",
        ],
        n_batch=n_batch,
    )


@pytest.fixture(scope="session")
def learnable_point_mass_mpc_m(n_batch: int) -> PointMassMPC:
    return PointMassMPC(learnable_params=["m"], n_batch=n_batch)


@pytest.fixture(scope="session")
def point_mass_env() -> PointMassEnv:
    return PointMassEnv()


@pytest.fixture(scope="session")
def point_mass_mpc_p_global(
    learnable_point_mass_mpc_m: PointMassMPC, n_batch: int
) -> np.ndarray:
    """Fixture for the global parameters of the point mass MPC."""
    return generate_batch_variation(
        learnable_point_mass_mpc_m.ocp_solver.acados_ocp.p_global_values, n_batch
    )


PENDULUM_ON_CART_LEARNABLE_PARAMS = ["M", "m", "g", "L11", "xref1"]
PENDULUM_ON_CART_EXT_COST_LEARNABLE_PARAMS = PENDULUM_ON_CART_LEARNABLE_PARAMS + ["c1"]


@pytest.fixture(scope="session")
def learnable_pendulum_on_cart_mpc_ext_cost(n_batch: int) -> PendulumOnCartMPC:
    """Fixture for the pendulum on cart MPC with learnable parameters, using a general quadratic cost."""
    return PendulumOnCartMPC(
        learnable_params=PENDULUM_ON_CART_EXT_COST_LEARNABLE_PARAMS,
        n_batch=n_batch,
        cost_type="EXTERNAL",
    )


@pytest.fixture(scope="session")
def learnable_pendulum_on_cart_mpc(
    n_batch: int,
) -> PendulumOnCartMPC:
    """Fixture for the pendulum on cart MPC with learnable parameters."""
    return PendulumOnCartMPC(
        learnable_params=PENDULUM_ON_CART_LEARNABLE_PARAMS,
        n_batch=n_batch,
        cost_type="NONLINEAR_LS",
    )


@pytest.fixture(scope="session")
def learnable_pendulum_on_cart_mpc_only_cost_params(
    n_batch: int,
) -> PendulumOnCartMPC:
    """Fixture for the pendulum on cart MPC with learnable parameters where only cost params are learnable."""
    return PendulumOnCartMPC(
        learnable_params=["L11", "xref1"],
        n_batch=n_batch,
        cost_type="NONLINEAR_LS",
    )


@pytest.fixture(scope="session")
def pendulum_on_cart_ocp_swingup_env() -> PendulumOnCartSwingupEnv:
    return PendulumOnCartSwingupEnv(render_mode="rgb_array")


@pytest.fixture(scope="session")
def learnable_chain_cost_mpc(
    n_batch: int,
) -> ChainMpc:
    """Fixture for the hanging chain MPC with learnable cost parameters."""
    n_mass = 3
    params = {}
    # rest length of spring
    params["L"] = np.repeat([0.033, 0.033, 0.033], n_mass - 1)

    # spring constant
    params["D"] = np.repeat([1.0, 1.0, 1.0], n_mass - 1)

    # damping constant
    params["C"] = np.repeat([0.1, 0.1, 0.1], n_mass - 1)

    # mass of the balls
    params["m"] = np.repeat([0.033], n_mass - 1)

    # disturbance on intermediate balls
    params["w"] = np.repeat([0.0, 0.0, 0.0], n_mass - 2)

    # Weight on state
    params["q_sqrt_diag"] = np.ones(3 * (n_mass - 1) + 3 * (n_mass - 2))

    # Weight on control inputs
    params["r_sqrt_diag"] = 1e-1 * np.ones(3)

    fix_point = np.zeros(3)
    ellipsoid = Ellipsoid(
        center=fix_point, radii=10 * 0.033 * (n_mass - 1) * np.ones(3)
    )

    pos_last_mass_ref = ellipsoid.spherical_to_cartesian(
        phi=0.75 * np.pi, theta=np.pi / 2
    )

    return ChainMpc(
        params=params,
        learnable_params=[
            "q_sqrt_diag",
            "r_sqrt_diag",
        ],
        fix_point=fix_point,
        pos_last_mass_ref=pos_last_mass_ref,
        n_mass=n_mass,
        n_batch=n_batch,
    )


@pytest.fixture(scope="session")
def chain_cost_p_global(
    learnable_chain_cost_mpc: ChainMpc,
    n_batch: int,
) -> np.ndarray:
    """Fixture for the global parameters of the pendulum on cart MPC."""
    return generate_batch_variation(
        learnable_chain_cost_mpc.ocp_solver.acados_ocp.p_global_values,
        n_batch,
    )


@pytest.fixture(scope="session")
def chain_mass_cost_env() -> ChainEnv:
    n_mass = 3
    phi_range = (0.5 * np.pi, 1.5 * np.pi)
    theta_range = (-0.1 * np.pi, 0.1 * np.pi)
    fix_point = np.zeros(3)
    ellipsoid = Ellipsoid(
        center=fix_point, radii=10 * 0.033 * (n_mass - 1) * np.ones(3)
    )

    pos_last_mass_ref = ellipsoid.spherical_to_cartesian(
        phi=0.75 * np.pi, theta=np.pi / 2
    )

    return ChainEnv(
        max_time=10.0,
        phi_range=phi_range,
        theta_range=theta_range,
        fix_point=fix_point,
        n_mass=n_mass,
        pos_last_ref=pos_last_mass_ref,
    )


@pytest.fixture(scope="session")
def all_env(
    pendulum_on_cart_ocp_swingup_env: PendulumOnCartSwingupEnv,
    chain_mass_cost_env: ChainEnv,
    point_mass_env: PointMassEnv,
):
    return [pendulum_on_cart_ocp_swingup_env, chain_mass_cost_env, point_mass_env]


@pytest.fixture(scope="session")
def pendulum_on_cart_p_global(
    learnable_pendulum_on_cart_mpc: PendulumOnCartMPC,
    n_batch: int,
) -> np.ndarray:
    """Fixture for the global parameters of the pendulum on cart MPC."""
    return generate_batch_variation(
        learnable_pendulum_on_cart_mpc.ocp_solver.acados_ocp.p_global_values,
        n_batch,
    )


@pytest.fixture(scope="session")
def pendulum_on_cart_ext_cost_p_global(
    learnable_pendulum_on_cart_mpc_ext_cost: PendulumOnCartMPC,
    n_batch: int,
) -> np.ndarray:
    """Fixture for the global parameters of the pendulum on cart MPC."""
    return generate_batch_variation(
        learnable_pendulum_on_cart_mpc_ext_cost.ocp_solver.acados_ocp.p_global_values,
        n_batch,
    )


@pytest.fixture(scope="session", params=["point_mass"])
def task(request):
    """Fixture for the task."""
    return create_task(request.param)


@pytest.fixture(scope="function", params=list(TRAINER_REGISTRY.keys()))
def trainer(request, task):
    with TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        cfg = create_default_cfg(request.param)
        trainer = create_trainer(request.param, task, tmpdir, "cpu", cfg)

        yield trainer
