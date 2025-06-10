from pathlib import Path

import numpy as np

from leap_c.ocp.acados.implicit import (
    AcadosImplicitCtx,
    AcadosImplicitFunction,
)
from leap_c.ocp.acados.initializer import ZeroInitializer


def test_file_management(acados_test_ocp, export_dir):
    pass


def test_ctx_loading(acados_test_ocp, export_dir):
    pass


def test_forward_backward_sensitivity(acados_test_ocp, export_dir):
    """Test the forward method of AcadosImplicitFunction."""
    # x0 = np.vstack(
    #     [
    #         rng.normal(
    #             loc=acados_ocp.constraints.x0, scale=0.1, size=acados_ocp.dims.nx
    #         )
    #         for _ in range(n_batch)
    #     ]
    # )

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
    pass


def test_translations(acados_test_ocp, export_dir):
    pass


def test_statelessness(acados_test_ocp):
    pass


def test_backup_functionality(acados_test_ocp):
    pass


def test_closed_loop(acados_test_impl_fun):
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

