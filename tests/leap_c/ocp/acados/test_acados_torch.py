from pathlib import Path

import numpy as np

from leap_c.ocp.acados.implicit import (
    AcadosImplicitCtx,
    AcadosImplicitFunction,
)
import torch

from leap_c.ocp.acados.torch import AcadosImplicitLayer

from leap_c.ocp.acados.initializer import ZeroInitializer

import conftest


def test_initialization(implicit_layer):
    assert True


def test_file_management(implicit_layer, export_dir):
    pass


def test_ctx_loading(implicit_layer: AcadosImplicitLayer, export_dir):
    pass


def test_forward_backward_sensitivity(
    implicit_layer: AcadosImplicitLayer,
    export_dir: str,
    rng: np.random.Generator,
):
    """Test the forward method of AcadosImplicitFunction."""

    print(export_dir)

    acados_ocp = implicit_layer.implicit_fun.ocp
    n_batch = implicit_layer.implicit_fun.forward_batch_solver.N_batch_max

    p_global = torch.tensor(acados_ocp.p_global_values).unsqueeze(0).repeat(n_batch, 1)

    print("p_global shape:", p_global.shape)

    # x0 = np.vstack(
    #     [
    #         rng.normal(
    #             loc=acados_ocp.constraints.x0, scale=0.1, size=acados_ocp.dims.nx
    #         )
    #         for _ in range(n_batch)
    #     ]
    # )

    # Create the mean and std tensors
    loc = torch.tensor(acados_ocp.constraints.x0).unsqueeze(0).repeat(n_batch, 1)
    scale = torch.full((n_batch, acados_ocp.dims.nx), 0.1)

    x0 = torch.normal(mean=loc, std=scale)

    print("x0 shape:", x0.shape)

    out = implicit_layer.forward(x0=x0)

    print(out)

    # implicit_function = AcadosImplicitFunction(ocp=acados_ocp)

    # ctx, sol_u0, sol_x, sol_u, sol_value = implicit_function.forward(x0=x0)

    # assert isinstance(ctx, AcadosImplicitCtx)
    # assert sol_u0 is not None
    # assert sol_x is not None
    # assert sol_u is not None
    # assert sol_value is not None

    # gradients = implicit_function.backward(
    #     ctx=ctx, u0_grad=None, x_grad=None, u_grad=None, value_grad=None
    # )

    # grad_u0 = gradients[2]
    # grad_p_global = gradients[3]

    # assert grad_u0 is not None
    # assert grad_p_global is not None

    # du0_dp_global = implicit_function.sensitivity(ctx, "du0_dp_global")

    # assert du0_dp_global is not None

    assert True


def test_translations(test_ocp, export_dir):
    # Test the translations of the cost between [external, linear_ls, nonlinear_ls] to external
    pass


def test_statelessness(test_ocp):
    # See current MPC implementation. Needs rewrite.
    pass


def test_backup_functionality(test_ocp):
    # See current MPC implementation. Needs rewrite.
    pass


def test_closed_loop(acados_test_implicit_function):
    # Test the acados_example_ocp in closed loop. Do we need a reference fixture?
    # x0 = np.array([0.5, 0.5, 0.5, 0.5])
    # x = [x0]
    # u = []

    # p_global = acados_test_impl_fun.ocp.p_global_values

    # for step in range(100):
    #     u_star, _, status = learnable_linear_mpc.policy(x[-1], p_global=p_global)
    #     assert status == 0, f"Did not converge to a solution in step {step}"
    #     u.append(u_star)
    #     x.append(learnable_linear_mpc.ocp_batch_solver.ocp_solvers[0].get(1, "x"))
    #     assert status == 0

    # x = np.array(x)
    # u = np.array(u)

    # assert (
    #     np.median(x[-10:, 0]) <= 1e-1
    #     and np.median(x[-10:, 1]) <= 1e-1
    #     and np.median(u[-10:]) <= 1e-1
    # )
    pass


def test_sensitivites_are_correct(
    implicit_layer: AcadosImplicitLayer,
):
    n_batch = 4
    ocp = implicit_layer.implicit_fun.ocp
    loc = torch.tensor(ocp.p_global_values).unsqueeze(0).repeat(n_batch, 1)
    scale = 0.1 * loc
    p_global = torch.normal(mean=loc, std=scale)

    x0 = torch.normal(mean=loc, std=scale)

    batch_size = p_global.shape[0]
    assert batch_size <= 10, "Using batch_sizes too large will make the test very slow."

    varying_params_to_test = [0]
    chosen_samples = []
    for i in range(batch_size):
        vary_idx = varying_params_to_test[i % len(varying_params_to_test)]
        chosen_samples.append(p_global[i, :, vary_idx].squeeze())
    test_param = np.stack(chosen_samples, axis=0)

    if len(varying_params_to_test) == 1:
        test_param = test_param.reshape(-1, 1)
    assert test_param.shape == (batch_size, 1)  # Sanity check

    p_rests = None

    # mpc_module = MpcSolutionModule(learnable_point_mass_mpc_m)
    x0_torch = torch.tensor(x0, dtype=torch.float64)
    x0_torch = torch.tile(x0_torch, (batch_size, 1))
    p = torch.tensor(test_param, dtype=torch.float64)
    # u0 = torch.tensor(u0, dtype=torch.float64)
    # u0 = torch.tile(u0, (batch_size, 1))

    u0.requires_grad = True
    x0_torch.requires_grad = True
    p.requires_grad = True

    def only_du0dx0(x0: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mpc_input = MpcInput(
            x0=x0,
            u0=None,
            parameters=MpcParameter(p_global=p, p_stagewise=p_rests),
        )
        mpc_output, _, _ = mpc_module.forward(
            mpc_input=mpc_input,
            mpc_state=None,
        )
        return mpc_output.u0, mpc_output.status

    torch.autograd.gradcheck(
        only_du0dx0, x0_torch, atol=1e-2, eps=1e-4, raise_exception=True
    )

    def only_dVdx0(x0: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mpc_input = MpcInput(
            x0=x0,
            u0=None,
            parameters=MpcParameter(p_global=p, p_stagewise=p_rests),
        )
        mpc_output, _, _ = mpc_module.forward(
            mpc_input=mpc_input,
            mpc_state=None,
        )
        return mpc_output.V, mpc_output.status

    torch.autograd.gradcheck(
        only_dVdx0, x0_torch, atol=1e-2, eps=1e-4, raise_exception=True
    )

    def only_dQdx0(x0: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:  #
        mpc_input = MpcInput(
            x0=x0,
            u0=u0,
            parameters=MpcParameter(p_global=p, p_stagewise=p_rests),
        )
        mpc_output, _, _ = mpc_module.forward(
            mpc_input=mpc_input,
            mpc_state=None,
        )
        assert torch.all(torch.isnan(mpc_output.u0)), (
            "u_star should be nan, since u0 is given."
        )
        return mpc_output.Q, mpc_output.status

    torch.autograd.gradcheck(
        only_dQdx0, x0_torch, atol=1e-2, eps=1e-4, raise_exception=True
    )

    def only_du0dp_global(p_global: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mpc_input = MpcInput(
            x0=x0_torch,
            parameters=MpcParameter(p_global=p_global, p_stagewise=p_rests),
        )
        mpc_output, _, _ = mpc_module.forward(
            mpc_input=mpc_input,
            mpc_state=None,
        )
        return mpc_output.u0, mpc_output.status

    torch.autograd.gradcheck(
        only_du0dp_global, p, atol=1e-2, eps=1e-4, raise_exception=True
    )

    def only_dVdp_global(p_global: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mpc_input = MpcInput(
            x0=x0_torch,
            parameters=MpcParameter(p_global=p_global, p_stagewise=p_rests),
        )
        mpc_output, _, _ = mpc_module.forward(
            mpc_input=mpc_input,
            mpc_state=None,
        )
        return mpc_output.V, mpc_output.status

    # NOTE: A higher tolerance than in the other checks is used here.
    torch.autograd.gradcheck(
        only_dVdp_global, p, atol=5 * 1e-2, eps=1e-4, raise_exception=True
    )

    def only_dQdp_global(p_global: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mpc_input = MpcInput(
            x0=x0_torch,
            u0=u0,
            parameters=MpcParameter(p_global=p_global, p_stagewise=p_rests),
        )
        mpc_output, _, _ = mpc_module.forward(
            mpc_input=mpc_input,
            mpc_state=None,
        )
        assert torch.all(torch.isnan(mpc_output.u0)), (
            "u_star should be nan, since u0 is given."
        )
        return mpc_output.Q, mpc_output.status

    torch.autograd.gradcheck(
        only_dQdp_global, p, atol=1e-2, eps=1e-6, raise_exception=True
    )

    def only_dQdu0(u0: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mpc_input = MpcInput(
            x0=x0_torch,
            u0=u0,
            parameters=MpcParameter(p_global=p, p_stagewise=p_rests),
        )
        mpc_output, _, _ = mpc_module.forward(
            mpc_input=mpc_input,
            mpc_state=None,
        )
        assert torch.all(torch.isnan(mpc_output.u0)), (
            "u_star should be nan, since u0 is given."
        )
        return mpc_output.Q, mpc_output.status

    torch.autograd.gradcheck(only_dQdu0, u0, atol=1e-2, eps=1e-4, raise_exception=True)
