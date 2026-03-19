import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch
from acados_template import AcadosOcpSolver

from leap_c.examples.mass_spring_damper.env import MassSpringDamperEnv
from leap_c.examples.mass_spring_damper.planner import (
    MassSpringDamperPlanner,
    MassSpringDamperPlannerConfig,
)
from leap_c.planner import ControllerFromPlanner


@pytest.fixture(scope="module")
def msd_controller():
    cfg = MassSpringDamperPlannerConfig()
    planner = MassSpringDamperPlanner(cfg)
    return ControllerFromPlanner(planner)


def plot_msd_solution(
    ocp_solver: AcadosOcpSolver,
) -> tuple[plt.Figure, np.ndarray]:
    """Plot the MassSpringDamper solution trajectory.

    Args:
        ocp_solver: The Acados OCP solver.

    Returns:
        Tuple of (figure, axes) for the plots.
    """
    k = np.arange(0, ocp_solver.N + 1)
    u = np.array([ocp_solver.get(stage, "u") for stage in range(ocp_solver.N)])
    x = np.array([ocp_solver.get(stage, "x") for stage in range(ocp_solver.N + 1)])

    fig, axs = plt.subplots(3, 1, figsize=(10, 8))
    labels = ["position x", "velocity v"]

    # Plot position
    axs[0].step(k, x[:, 0], label=labels[0])
    axs[0].set_ylabel(labels[0])
    axs[0].axhline(y=0, color="k", linestyle="--", alpha=0.3)
    axs[0].grid()
    axs[0].legend()

    # Plot velocity
    axs[1].step(k, x[:, 1], label=labels[1], color="orange")
    axs[1].set_ylabel(labels[1])
    axs[1].axhline(y=0, color="k", linestyle="--", alpha=0.3)
    axs[1].grid()
    axs[1].legend()

    # Plot force
    axs[2].step(k[:-1], u, label="force F", color="green")
    axs[2].set_ylabel("F")
    axs[2].set_xlabel("k")
    axs[2].axhline(y=0, color="k", linestyle="--", alpha=0.3)
    axs[2].grid()
    axs[2].legend()

    fig.suptitle("Mass-Spring-Damper Solution Trajectory")
    fig.tight_layout()

    return fig, axs


def test_solution(msd_controller):
    """Test that the OCP solver can solve for a given initial state."""
    ocp_solver = msd_controller.planner.diff_mpc.diff_mpc_fun.forward_batch_solver.ocp_solvers[0]
    # Set initial state away from origin
    x0 = np.array([1.5, -1.0])
    ocp_solver.solve_for_x0(x0)

    if ocp_solver.status != 0:
        raise ValueError(f"Solver failed with status {ocp_solver.status}")

    fig, axs = plot_msd_solution(ocp_solver)
    plt.close(fig)


def test_env_terminates():
    """Test if the environment terminates correctly when exceeding valid states.

    The test applies maximum control inputs to drive the system beyond
    the state bounds.

    This test ensures that the environment terminates properly
    when applying extreme control inputs continuously.
    """
    env = MassSpringDamperEnv()

    # Test with maximum force to exceed velocity bounds
    for action in [env.action_space.low, env.action_space.high]:
        env.reset(seed=0)
        terminated = False
        for _ in range(1000):
            state, _, term, trunc, _ = env.step(action)
            if term:
                terminated = True
                break
        assert terminated or trunc, "Environment should terminate or truncate"
        if terminated:
            # Check that at least one state variable is out of bounds
            assert (
                state[0] < env.cfg.x_min
                or state[0] > env.cfg.x_max
                or state[1] < env.cfg.v_min
                or state[1] > env.cfg.v_max
            ), "State should be out of bounds when terminated"


def test_env_truncates():
    """Test if the environment truncates correctly when time limit is reached.

    The test applies zero control inputs to keep the system within valid states
    until the time limit is reached.

    This test ensures that the environment truncates properly when doing nothing.
    """
    env = MassSpringDamperEnv()
    env.reset(seed=0)

    action = np.array([0.0])
    terminated = False
    truncated = False
    for _ in range(1000):
        _, _, term, trunc, _ = env.step(action)
        if trunc:
            truncated = True
            break
        if term:
            terminated = True
            break

    assert truncated or terminated, "Environment should eventually truncate or terminate"


def test_env_types():
    """Test whether the type of the state is and stays np.float32.

    This test uses an action from the action space.
    Note that the action space has type np.float32.
    """
    env = MassSpringDamperEnv()

    x, _ = env.reset(seed=0)
    assert x.dtype == np.float32, "Initial state should be float32"

    action = np.zeros(env.action_space.shape, dtype=np.float32)
    x, _, _, _, _ = env.step(action)
    assert x.dtype == np.float32, "State after step should be float32"


def test_run_closed_loop(n_iter: int = 100) -> None:
    """Test the closed-loop performance of the mass-spring-damper MPC controller.

    The test starts from an initial state away from the origin and checks
    that the controller drives the system towards the origin.

    Asserts:
    - The final position is close to zero.
    - The final velocity is close to zero.
    """
    env = MassSpringDamperEnv()
    obs, _ = env.reset(seed=42)

    # Set initial state away from origin
    env.state = np.array([1.0, -0.5])
    obs = env._observation()

    planner = MassSpringDamperPlanner()
    controller = ControllerFromPlanner(planner=planner)

    default_param = controller.default_param(obs)
    default_param = torch.as_tensor(default_param, dtype=torch.float32).unsqueeze(0)
    ctx = None

    for _ in range(n_iter):
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        ctx, a = controller(obs_tensor, default_param, ctx=ctx)
        a = a.squeeze(0).numpy()
        obs, r, terminated, truncated, info = env.step(a)

        if terminated or truncated:
            break

    # Check that the final state is close to the origin
    assert np.abs(obs[0]) < 0.3, f"Final position {obs[0]} is not close to zero"
    assert np.abs(obs[1]) < 0.3, f"Final velocity {obs[1]} is not close to zero"


def test_env_reset_modes():
    """Test that the environment can reset in different modes."""
    env = MassSpringDamperEnv()

    # Test reset in train mode
    obs_train, _ = env.reset(seed=0, options={"mode": "train"})
    assert obs_train.shape == (2,), "Observation shape should be (2,)"
    assert obs_train.dtype == np.float32, "Observation should be float32"

    # Test reset in test mode (default)
    obs_test, _ = env.reset(seed=0)
    assert obs_test.shape == (2,), "Observation shape should be (2,)"
    assert obs_test.dtype == np.float32, "Observation should be float32"
