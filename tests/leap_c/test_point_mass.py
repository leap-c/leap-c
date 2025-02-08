import numpy as np
from leap_c.examples.pointmass.mpc import PointMassMPC
from leap_c.examples.pointmass.env import PointMassEnv, PointMassParam
from leap_c.examples.pointmass.task import PointMassTask

from leap_c.nn.modules import MPCSolutionModule

# from leap_c.linear_mpc import LinearMPC
from leap_c.mpc import MPC
from test_mpc_p_global import (
    run_test_mpc_solve_and_batch_solve_on_batch_p_global,
)
from conftest import generate_batch_variation

import torch

from leap_c.mpc import MPCInput, MPCParameter

import matplotlib.pyplot as plt
from pathlib import Path


def run_test_parametric_sensitivities(
    learnable_point_mass_mpc: MPC, point_mass_mpc_p_global: np.ndarray
):
    run_test_mpc_solve_and_batch_solve_on_batch_p_global(
        learnable_point_mass_mpc, point_mass_mpc_p_global, plot=False
    )


def run_test_pointmass_functions(mpc: PointMassMPC):
    s = np.array([1.0, 0.0, 0.0, 0.0])
    a = np.array([0.0, 0.0])

    _ = mpc.policy(state=s, p_global=None)[0]
    _ = mpc.state_value(state=s, p_global=None)[0]
    _ = mpc.state_action_value(state=s, action=a, p_global=None)[0]


def run_closed_loop_test(mpc: PointMassMPC, env: PointMassEnv, n_iter: int = int(5e2)):
    s = env.reset(seed=0)
    for _ in range(n_iter):
        a = mpc.policy(state=s, p_global=None)[0]
        s, _, _, _, _ = env.step(a)

    assert np.allclose(s, np.array([0.0, 0.0, 0.0, 0.0]), atol=1e-6)


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


def test_solution_module():
    batch_size = 2
    mpc = PointMassMPC(
        learnable_params=["m", "c"],
        n_batch=batch_size,
        export_directory=Path("c_generated_code"),
        export_directory_sensitivity=Path("c_generated_code_sens"),
    )

    x0 = np.array([0.5, 0.5, 0.0, 0.0])
    u0 = np.array([0.5, 0.5])

    batch_variation = generate_batch_variation(
        mpc.ocp_solver.acados_ocp.p_global_values,
        batch_size,
    )
    print(batch_variation.shape)

    varying_params_to_test = [0, 1]  # A_0, Q_0, b_1, f_1
    chosen_samples = []
    for i in range(batch_size):
        vary_idx = varying_params_to_test[i % len(varying_params_to_test)]
        chosen_samples.append(batch_variation[i, :, vary_idx].squeeze())
    test_param = np.stack(chosen_samples, axis=0)
    assert test_param.shape == (batch_size, 2)  # Sanity check

    p_rests = MPCParameter(None, None, None)

    mpc_module = MPCSolutionModule(mpc)
    x0_torch = torch.tensor(x0, dtype=torch.float64)
    x0_torch = torch.tile(x0_torch, (batch_size, 1))
    p = torch.tensor(test_param, dtype=torch.float64)
    u0 = torch.tensor([u0], dtype=torch.float64)
    u0 = torch.tile(u0, (batch_size, 1))

    u0.requires_grad = True
    x0_torch.requires_grad = True
    p.requires_grad = True

    def only_du0dx0(x0: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        u_star, val, status, stats = mpc_module.forward(
            x0, u0=None, p_global=p, p_stagewise=p_rests, initializations=None
        )
        return u_star, status

    torch.autograd.gradcheck(
        only_du0dx0, x0_torch, atol=1e-2, eps=1e-4, raise_exception=True
    )

    def only_dVdx0(x0: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        u_star, val, status, stats = mpc_module.forward(
            x0, u0=None, p_global=p, p_stagewise=p_rests, initializations=None
        )
        return val, status

    torch.autograd.gradcheck(
        only_dVdx0, x0_torch, atol=1e-2, eps=1e-4, raise_exception=True
    )

    def only_dQdx0(x0: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:  #
        u_star, Q, status, stats = mpc_module.forward(
            x0, u0=u0, p_global=p, p_stagewise=p_rests, initializations=None
        )
        assert torch.all(
            torch.isnan(u_star)
        ), "u_star should be nan, since u0 is given."
        return Q, status

    torch.autograd.gradcheck(
        only_dQdx0, x0_torch, atol=1e-2, eps=1e-4, raise_exception=True
    )

    def only_du0dp_global(p_global: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        u_star, V, status, stats = mpc_module.forward(
            x0=x0_torch,
            u0=None,
            p_global=p_global,
            p_stagewise=p_rests,
            initializations=None,
        )
        return u_star, status

    torch.autograd.gradcheck(
        only_du0dp_global, p, atol=1e-2, eps=1e-4, raise_exception=True
    )

    def only_dVdp_global(p_global: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        u_star, V, status, stats = mpc_module.forward(
            x0=x0_torch,
            u0=None,
            p_global=p_global,
            p_stagewise=p_rests,
            initializations=None,
        )
        return V, status

    # NOTE: A higher tolerance than in the other checks is used here.
    torch.autograd.gradcheck(
        only_dVdp_global, p, atol=5 * 1e-2, eps=1e-4, raise_exception=True
    )

    def only_dQdp_global(p_global: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        u_star, Q, status, stats = mpc_module.forward(
            x0=x0_torch,
            u0=u0,
            p_global=p_global,
            p_stagewise=p_rests,
            initializations=None,
        )
        assert torch.all(
            torch.isnan(u_star)
        ), "u_star should be nan, since u0 is given."
        return Q, status

    torch.autograd.gradcheck(
        only_dQdp_global, p, atol=1e-2, eps=1e-6, raise_exception=True
    )

    def only_dQdu0(u0: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        u_star, Q, status, stats = mpc_module.forward(
            x0=x0_torch, u0=u0, p_global=p, p_stagewise=p_rests, initializations=None
        )
        assert torch.all(
            torch.isnan(u_star)
        ), "u_star should be nan, since u0 is given."
        return Q, status

    torch.autograd.gradcheck(only_dQdu0, u0, atol=1e-2, eps=1e-4, raise_exception=True)


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


if __name__ == "__main__":
    test_solution_module()
    print("All autograd tests passed")

    n_batch = 100
    mpc = PointMassMPC(
        learnable_params=["m", "c"],
        n_batch=n_batch,
        export_directory=Path("c_generated_code"),
        export_directory_sensitivity=Path("c_generated_code_sens"),
    )

    x0 = np.array([1.0, 1.0, 0.0, 0.0])
    u0 = np.array([0.5, 0.5])

    simple_test_dudx0(mpc, x0=x0, u0=u0, n_batch=n_batch)
