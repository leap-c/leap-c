import numpy as np
from leap_c.examples.pointmass.mpc import PointMassMPC
from leap_c.examples.pointmass.env import PointMassEnv, PointMassParam
from leap_c.examples.pointmass.task import PointMassTask

from leap_c.nn.modules import MPCSolutionModule
import datetime

# from leap_c.linear_mpc import LinearMPC
from leap_c.mpc import MPC

from leap_c.registry import create_task, create_default_cfg, create_trainer


from leap_c.mpc import MPCInput, MPCParameter

import matplotlib.pyplot as plt
from pathlib import Path

from argparse import ArgumentParser
import datetime
from pathlib import Path

import leap_c.examples  # noqa: F401
import leap_c.rl  # noqa: F401
from leap_c.registry import create_task, create_default_cfg, create_trainer
from leap_c.trainer import BaseConfig

from leap_c.rollout import episode_rollout


def run_test_pointmass_functions(mpc: PointMassMPC):
    s = np.array([1.0, 0.0, 0.0, 0.0])
    a = np.array([0.0, 0.0])

    _ = mpc.policy(state=s, p_global=None)[0]
    _ = mpc.state_value(state=s, p_global=None)[0]
    _ = mpc.state_action_value(state=s, action=a, p_global=None)[0]


def run_closed_loop_test(mpc: PointMassMPC, env: PointMassEnv, n_iter: int = int(2e2)):
    s, _ = env.reset(seed=0)
    for _ in range(n_iter):
        a = mpc.policy(state=s, p_global=None)[0]
        s, _, _, _, _ = env.step(a)

    assert np.linalg.norm(s) < 1e-1


# def test_closed_loop(
#     learnable_point_mass_mpc: PointMassMPC, point_mass_env: PointMassEnv
# ):
#     run_closed_loop_test(learnable_point_mass_mpc, point_mass_env)


def simple_test_dudx0(
    mpc: PointMassMPC,
    x0: np.ndarray,
    u0: np.ndarray,
    n_batch: int,
):
    # x0_torch = torch.tensor(x0, dtype=torch.float64)
    x0 = np.tile(x0, (n_batch, 1))
    x0[:, 0] = np.linspace(0.9, 1.1, n_batch)
    p_global = np.tile(mpc.ocp.p_global_values, (n_batch, 1))

    mpc_input = MPCInput(x0=x0, parameters=MPCParameter(p_global=p_global))
    mpc_output, _ = mpc(mpc_input=mpc_input, dudx=True)

    u0 = mpc_output.u0
    du0_dx0 = mpc_output.du0_dx0[:, :, 0]
    du0_dx0_fd = np.gradient(u0, x0[:, 0].flatten(), axis=0)

    print(du0_dx0 - du0_dx0_fd)

    plt.figure()
    plt.plot(x0[:, 0], du0_dx0, label="ad")
    plt.plot(x0[:, 0], du0_dx0_fd, label="fd")
    plt.legend()
    plt.grid()
    plt.ylabel("du0_dx0")
    plt.show()


def prototyping():
    n_batch = 100
    mpc = PointMassMPC(
        learnable_params=["m", "c"],
        n_batch=100,
        export_directory=Path("c_generated_code"),
        export_directory_sensitivity=Path("c_generated_code_sens"),
    )

    x0 = np.array([1.0, 1.0, 0.0, 0.0])
    u0 = np.array([0.5, 0.5])
    p_global = np.linspace(
        0.9 * mpc.default_p_global, 1.1 * mpc.default_p_global, n_batch
    )

    # Tile x0 to match the batch size
    x0 = np.tile(x0, (n_batch, 1))
    u0 = np.tile(u0, (n_batch, 1))

    # mpc_parameter = MPCParameter(p_global=p_global)
    mpc_input = MPCInput(x0=x0, parameters=MPCParameter(p_global=p_global))

    print("mpc_input.is_batched()", mpc_input.is_batched())

    mpc_output, _ = mpc(mpc_input=mpc_input, dudp=True, use_adj_sens=True)

    u0 = mpc_output.u0
    Q = mpc_output.Q
    V = mpc_output.V
    dvalue_dx0 = mpc_output.dvalue_dx0
    dvalue_du0 = mpc_output.dvalue_du0
    dvalue_dp_global = mpc_output.dvalue_dp_global
    du0_dp_global = mpc_output.du0_dp_global
    du0_dx0 = mpc_output.du0_dx0

    du0_dp_global_fd = np.gradient(u0, p_global.flatten(), axis=0)

    plt.figure()
    for i in range(2):
        plt.subplot(2, 1, i + 1)
        plt.plot(p_global, du0_dp_global[:, i], label=f"ad {i}")
        plt.plot(p_global, du0_dp_global_fd[:, i], label=f"fd {i}")
        plt.grid()
    plt.legend()
    plt.show()

    exit(0)

    diff = np.abs(du0_dp_global.squeeze() - du0_dp_global_fd)

    print("x0.shape", x0.shape)
    print(p_global.shape)
    out = [mpc.policy(state=x0, p_global=p, sens=True) for p in p_global]

    mpc_input = MPCInput(x0=x0, u0=u0, parameters=MPCParameter(p_global=p_global))

    # Stack all first elements of the tuple
    policy = np.stack([o[0] for o in out])
    policy_gradient = np.stack([o[1] for o in out]).squeeze()

    # Compute the np.gradient of policy with respect to p_test
    policy_gradient_fd = np.gradient(policy, p_global.flatten(), axis=0)

    plt.figure()
    plt.plot(p_global, policy_gradient, label="ad")
    plt.plot(p_global, policy_gradient_fd, label="fd")
    plt.legend()
    plt.grid()
    plt.ylabel("Policy gradient")
    plt.show()

    print("Done")


def run_closed_loop(
    mpc: PointMassMPC,
    env: PointMassEnv,
    dt: float | None = None,
    n_iter: int = int(2e2),
):
    s, _ = env.reset()

    S = np.zeros((n_iter, 4))
    S[0, :] = s
    A = np.zeros((n_iter, 2))
    for i in range(n_iter - 1):
        A[i, :] = mpc.policy(state=S[i, :], p_global=None)[0]
        S[i + 1, :], _, _, _, _ = env.step(A[i, :])

    plot_data = np.hstack([S, A])
    return plot_data


def test_episode_rollout(mpc: PointMassMPC, env: PointMassEnv):
    pass


def default_output_path() -> Path:
    # derive output path from date and time
    now = datetime.datetime.now()
    return Path(f"output/{now.strftime('%Y_%m_%d/%H_%M_%S')}")


if __name__ == "__main__":
    trainer = create_trainer(
        name="sac_fou",
        task=create_task("point_mass"),
        output_path="output/videos",
        device="cpu",
        cfg=create_default_cfg("sac_fou"),
    )

    trainer.validate()
