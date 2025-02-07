import os
import shutil
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pytest
from acados_template import AcadosOcpSolver
from gymnasium.utils.save_video import save_video
from leap_c.examples.pendulum_on_a_cart.env import PendulumOnCartSwingupEnv
from leap_c.examples.pendulum_on_cart import PendulumOnCartMPC
from leap_c.util import create_dir_if_not_exists


def plot_cart_pole_solution(
    ocp_solver: AcadosOcpSolver,
) -> Tuple[plt.Figure, np.ndarray[plt.Axes]]:
    k = np.arange(0, ocp_solver.N + 1)
    u = np.array([ocp_solver.get(stage, "u") for stage in range(ocp_solver.N)])
    x = np.array([ocp_solver.get(stage, "x") for stage in range(ocp_solver.N + 1)])

    fig, axs = plt.subplots(5, 1)
    labels = ["x", "theta", "dx", "dtheta"]

    for i in range(4):
        axs[i].step(k, x[:, i])
        axs[i].set_ylabel(labels[i])
        axs[i].grid()

    axs[4].step(k[:-1], u)
    axs[4].set_ylabel("F")
    axs[4].set_xlabel("k")
    axs[4].grid()

    return fig, axs


def test_solution(
    mpc: PendulumOnCartMPC = PendulumOnCartMPC(
        learnable_params=[
            "M",
            "m",
            "g",
            "l",
            "L11",
            "L22",
            "L33",
            "L44",
            "L55",
            "Lloweroffdiag",
            "c1",
            "c2",
            "c3",
            "c4",
            "c5",
        ],
        exact_hess_dyn=False,
        least_squares_cost=False,
    ),
):
    ocp_solver = mpc.ocp_solver
    ocp_solver.solve_for_x0(np.array([0.0, np.pi, 0.0, 0.0]))

    if ocp_solver.status != 0:
        raise ValueError(f"Solver failed with status {ocp_solver.status}")

    fig, axs = plot_cart_pole_solution(ocp_solver)


def test_env_terminates(pendulum_on_cart_ocp_swingup_env: PendulumOnCartSwingupEnv):
    """Test if the environment terminates correctly when applying minimum and maximum control inputs.

    This test ensures that the environment terminates properly when applying either minimum or maximum control
    inputs continuously. It checks both termination conditions and verifies that the episode ends with a termination
    rather than a truncation.
    """

    env = pendulum_on_cart_ocp_swingup_env

    for action in [env.action_space.low, env.action_space.high]:  # type:ignore
        env.reset(seed=0)
        for _ in range(1000):
            state, _, term, trunc, _ = env.step(action)
            if term:
                break
        assert term
        assert not trunc
        assert (
            state[0] < -pendulum_on_cart_ocp_swingup_env.x_threshold
            or state[0] > pendulum_on_cart_ocp_swingup_env.x_threshold
        )


def test_env_truncates(pendulum_on_cart_ocp_swingup_env: PendulumOnCartSwingupEnv):
    """Test if the environment truncates correctly when applying minimum and maximum control inputs.

    This test ensures that the environment truncates properly when doing nothing (i.e. it cannot come from termination).
    It checks both termination conditions and verifies that the episode ends with a truncation
    rather than a truncation.
    """

    env = pendulum_on_cart_ocp_swingup_env
    env.reset(seed=0)

    action = np.array([0])
    for _ in range(1000):
        _, _, term, trunc, _ = env.step(action)
        if trunc:
            break
    assert not term
    assert trunc


def test_closed_loop_rendering(
    learnable_pendulum_on_cart_mpc_lls_cost: PendulumOnCartMPC,
    pendulum_on_cart_ocp_swingup_env: PendulumOnCartSwingupEnv,
):
    obs, _ = pendulum_on_cart_ocp_swingup_env.reset(seed=1337)

    count = 0
    terminated = False
    truncated = False
    frames = []
    cwd = os.getcwd()
    savefile_dir_path = os.path.join(cwd, "test_closed_loop_pendulum_on_cart")
    create_dir_if_not_exists(savefile_dir_path)
    while count < 300 and not terminated and not truncated:
        a = learnable_pendulum_on_cart_mpc_lls_cost.policy(
            obs[0], learnable_pendulum_on_cart_mpc_lls_cost.default_p_global
        )[0]
        obs_prime, r, terminated, truncated, info = (
            pendulum_on_cart_ocp_swingup_env.step(a)
        )
        frames.append(pendulum_on_cart_ocp_swingup_env.render())
        obs = obs_prime
        count += 1
    assert (
        count <= 200
    ), "max_time and dt dictate that no more than 200 steps should be possible until termination."
    save_video(
        frames,  # type:ignore
        video_folder=savefile_dir_path,
        name_prefix="pendulum_on_cart",
        fps=pendulum_on_cart_ocp_swingup_env.metadata["render_fps"],
    )

    shutil.rmtree(savefile_dir_path)


def main():
    test_solution()
    plt.show()


if __name__ == "__main__":
    pytest.main([__file__])
