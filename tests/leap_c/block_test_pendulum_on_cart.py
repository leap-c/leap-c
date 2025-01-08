import os
import shutil
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pytest
from acados_template import AcadosOcpSolver
from gymnasium.utils.save_video import save_video
from leap_c.examples.pendulum_on_cart import PendulumOnCartMPC, PendulumOnCartOcpEnv
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
        learnable_params=["M", "m", "g", "l", "Q", "R"], exact_hess_dyn=False
    ),
):
    ocp_solver = mpc.ocp_solver
    ocp_solver.solve_for_x0(np.array([0.0, np.pi, 0.0, 0.0]))

    if ocp_solver.status != 0:
        raise ValueError(f"Solver failed with status {ocp_solver.status}")

    fig, axs = plot_cart_pole_solution(ocp_solver)


def test_closed_loop_rendering(
    learnable_pendulum_on_cart_mpc: PendulumOnCartMPC,
    pendulum_on_cart_ocp_env: PendulumOnCartOcpEnv,
):
    obs, _ = pendulum_on_cart_ocp_env.reset(seed=1337)

    count = 0
    terminated = False
    truncated = False
    frames = []
    cwd = os.getcwd()
    savefile_dir_path = os.path.join(cwd, "test_closed_loop_pendulum_on_cart")
    create_dir_if_not_exists(savefile_dir_path)
    while count < 200 and not terminated and not truncated:
        a = learnable_pendulum_on_cart_mpc.policy(
            obs[0], learnable_pendulum_on_cart_mpc.default_p_global
        )[0]
        obs_prime, r, terminated, truncated, info = pendulum_on_cart_ocp_env.step(a)
        frames.append(info["frame"])
        obs = obs_prime
        count += 1
    assert (
        count < 200
    ), "max_time and dt dictate that no more than 100 steps should be possible until termination."
    save_video(
        frames,  # type:ignore
        video_folder=savefile_dir_path,
        name_prefix="pendulum_on_cart",
        fps=pendulum_on_cart_ocp_env.metadata["render_fps"],
    )

    shutil.rmtree(savefile_dir_path)


def main():
    test_solution()
    plt.show()


if __name__ == "__main__":
    pytest.main([__file__])
