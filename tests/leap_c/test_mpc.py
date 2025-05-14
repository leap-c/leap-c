from dataclasses import fields

import numpy as np
from acados_template.acados_ocp_iterate import (
    AcadosOcpFlattenedBatchIterate,
)

# from leap_c.examples.linear_system import LinearSystemMPC
from leap_c.mpc import Mpc, MpcInput, MpcOutput, MpcParameter, create_zero_init_state_fn
from leap_c.utils import find_idx_for_labels


def mpc_outputs_assert_allclose(
    mpc_output: MpcOutput, mpc_output2: MpcOutput, test_u_star: bool
):
    allclose = True
    for fld in mpc_output._fields:
        val1 = getattr(mpc_output, fld)
        val2 = getattr(mpc_output2, fld)
        if fld == "u0" and not test_u_star:
            continue  # Testing u_star when different u0 were given makes no sense
        if isinstance(val1, np.ndarray):
            tolerance = (
                1e-3 if not fld.startswith("d") else 1e-3
            )  # 1e-3 is probably close enough for gradients, at least thats also what we do in pytorch.gradcheck wrt the numerical gradient
            assert np.allclose(val1, val2, atol=tolerance), (
                f"Field {fld} not close, maximal difference is {np.abs(val1 - val2).max()}"
            )
        elif isinstance(val1, type(None)):
            assert val1 == val2
        else:
            raise NotImplementedError(
                "Only np.ndarray fields known. Did new fields get added to MPCOutput?"
            )
    return allclose


# def test_raising_exception_if_u0_outside_bounds(learnable_linear_mpc: MPC):
#     x0 = np.array([0.5, 0.5])
#     u0 = np.array([1000.0])
#     learnable_linear_mpc.throw_error_if_u0_is_outside_ocp_bounds = True
#     try:
#         learnable_linear_mpc(MPCInput(x0=x0, u0=u0))
#         assert False
#     except ValueError:
#         pass
#     learnable_linear_mpc.throw_error_if_u0_is_outside_ocp_bounds = False
#     learnable_linear_mpc(MPCInput(x0=x0, u0=u0))
#     learnable_linear_mpc.throw_error_if_u0_is_outside_ocp_bounds = True


def test_statelessness(
    learnable_point_mass_mpc_different_params: Mpc,
):
    x0 = np.array([0.5, 0.5, 0.5, 0.5])
    u0 = np.array([0.5, 0.5])
    # Create MPC with some stateless and some global parameters
    lin_mpc = learnable_point_mass_mpc_different_params
    n_batch = lin_mpc.n_batch_max
    x0 = np.tile(x0, (lin_mpc.n_batch_max, 1))
    u0 = np.tile(u0, (lin_mpc.n_batch_max, 1))
    mpc_input_standard = MpcInput(x0=x0, u0=u0)
    solution_standard, _ = lin_mpc(
        mpc_input=mpc_input_standard, dudp=True, dvdp=True, dudx=True
    )
    p_global = lin_mpc.default_p_global
    assert p_global is not None
    p_global = p_global + np.ones(p_global.shape[0]) * 0.01
    p_global = np.tile(p_global, (n_batch, 1))
    p_stagewise = lin_mpc.default_p_stagewise
    assert p_stagewise is not None
    p_stagewise = p_stagewise + np.ones(p_stagewise.shape[1]) * 0.01
    assert len(p_stagewise.shape) == 2, (
        f"I assumed this would be of shape ({lin_mpc.N + 1}, #p_stagewise) but shape is {p_stagewise.shape}"
    )
    p_stagewise = np.tile(p_stagewise, (n_batch, 1, 1))
    params = MpcParameter(p_global, p_stagewise)
    x0_different = x0 - 0.01
    u0_different = u0 - 0.01
    mpc_input_different = MpcInput(x0=x0_different, u0=u0_different, parameters=params)
    solution_different, _ = lin_mpc(
        mpc_input=mpc_input_different, dudp=True, dvdp=True, dudx=True
    )
    # Use this as proxy to verify the different solution is different enough
    assert not np.allclose(
        solution_standard.Q,  # type:ignore
        solution_different.Q,  # type:ignore
    )
    solution_supposedly_standard, _ = lin_mpc(
        mpc_input=mpc_input_standard, dudp=True, dvdp=True, dudx=True
    )
    mpc_outputs_assert_allclose(
        solution_standard, solution_supposedly_standard, test_u_star=True
    )


def test_statelessness_pendulum_on_cart(
    n_batch: int, learnable_pendulum_on_cart_mpc: Mpc
):
    # Create MPC with some stateless and some global parameters
    x0 = np.array([0, -np.pi, 0, 0])
    x0 = np.tile(x0, (n_batch, 1))
    mpc_input_standard = MpcInput(x0=x0)
    solution_standard, _ = learnable_pendulum_on_cart_mpc(
        mpc_input=mpc_input_standard, dudp=True, dvdp=True, dudx=True
    )

    assert learnable_pendulum_on_cart_mpc.default_p_global is not None
    p_global_def = learnable_pendulum_on_cart_mpc.default_p_global
    p_global_def = np.tile(p_global_def, (n_batch, 1))

    idx = find_idx_for_labels(
        learnable_pendulum_on_cart_mpc.ocp_solver.acados_ocp.model.p_global, "xref1"
    )

    p_global_def[:, idx] = 1  # Set reference position to 1
    params = MpcParameter(p_global=p_global_def)
    mpc_input_different = MpcInput(x0=x0, parameters=params)
    solution_different, _ = learnable_pendulum_on_cart_mpc(
        mpc_input=mpc_input_different, dudp=True, dvdp=True, dudx=True
    )
    # Use this as proxy to verify the different solution is different enough
    assert not np.allclose(
        solution_standard.V,  # type:ignore
        solution_different.V,  # type:ignore
    )
    solution_supposedly_standard, _ = learnable_pendulum_on_cart_mpc(
        mpc_input=mpc_input_standard, dudp=True, dvdp=True, dudx=True
    )
    mpc_outputs_assert_allclose(
        solution_standard, solution_supposedly_standard, test_u_star=True
    )


def test_using_mpc_state(
    learnable_point_mass_mpc_different_params: Mpc,
    n_batch: int,
):
    x0 = np.array([0.5, 0.5, 0.5, 0.5])
    u0 = np.array([0.5, 0.5])
    learnable_linear_mpc = learnable_point_mass_mpc_different_params
    x0 = np.tile(x0, (n_batch, 1))
    u0 = np.tile(u0, (n_batch, 1))
    inp = MpcInput(x0=x0, u0=u0)
    sol, state = learnable_linear_mpc(inp)
    for solver in learnable_linear_mpc.ocp_batch_solver.ocp_solvers:
        qp_iters = solver.get_stats("qp_iter").sum()  # type:ignore
        assert qp_iters > 1
    same_sol, state = learnable_linear_mpc(inp, mpc_state=state)
    for solver in learnable_linear_mpc.ocp_batch_solver.ocp_solvers:
        qp_iters = solver.get_stats("qp_iter").sum()  # type:ignore
        assert qp_iters == 0
    mpc_outputs_assert_allclose(sol, same_sol, test_u_star=True)


def test_backup_fn(learnable_point_mass_mpc_different_params: Mpc, n_batch: int):
    learnable_linear_mpc = learnable_point_mass_mpc_different_params
    x0 = np.array([0.5, 0.5, 0.5, 0.5])
    u0 = np.array([0.5, 0.5])
    x0 = np.tile(x0, (n_batch, 1))
    u0 = np.tile(u0, (n_batch, 1))
    increment = np.ones(4) * 1 / 2 * n_batch
    # The exact increment is not important, just that it is different for each sample
    for i in range(n_batch):
        x0[i] = x0[i] + i * increment
    inp = MpcInput(x0=x0, u0=u0)
    sol, template_state = learnable_linear_mpc(inp)
    default_init = learnable_linear_mpc.init_state_fn  # For restoring fixture
    learnable_linear_mpc.init_state_fn = None  # Make sure no backup is used
    assert np.all(sol.status == 0)
    assert isinstance(template_state, AcadosOcpFlattenedBatchIterate), (
        f"This test assumed state would be of type AcadosOcpFlattenedBatchIterate, but got {type(template_state)}"
    )
    nan_state = AcadosOcpFlattenedBatchIterate(
        x=np.ones_like(template_state.x) * np.nan,
        u=np.ones_like(template_state.u) * np.nan,
        z=np.ones_like(template_state.z) * np.nan,
        sl=np.ones_like(template_state.sl) * np.nan,
        su=np.ones_like(template_state.su) * np.nan,
        pi=np.ones_like(template_state.pi) * np.nan,
        lam=np.ones_like(template_state.lam) * np.nan,
        N_batch=template_state.N_batch,
    )
    no_sol, _ = learnable_linear_mpc(inp, mpc_state=nan_state)
    assert np.all(no_sol.status != 0)

    def backup_fn_batched(input: MpcInput):
        """Simplified backup function for the resolve."""
        if not input.is_batched():
            index = -1
            for ind in range(n_batch):
                if np.allclose(input.x0, inp.x0[ind]):
                    index = ind
                    break
            assert ind != -1, "There has to be a corresponding x0 in the batch."
            return AcadosOcpFlattenedBatchIterate(
                x=template_state.x[[index]],
                u=template_state.u[[index]],
                z=template_state.z[[index]],
                sl=template_state.sl[[index]],
                su=template_state.su[[index]],
                pi=template_state.pi[[index]],
                lam=template_state.lam[[index]],
                N_batch=1,
            )
        else:
            raise ValueError("Is assumed to not happen here.")

    learnable_linear_mpc.init_state_fn = backup_fn_batched

    sol_again, _ = learnable_linear_mpc(
        inp,
        mpc_state=nan_state,
    )
    learnable_linear_mpc.init_state_fn = default_init  # Restore fixture
    mpc_outputs_assert_allclose(sol, sol_again, test_u_star=True)


def test_closed_loop(
    learnable_point_mass_mpc_different_params: Mpc,
):
    x0 = np.array([0.5, 0.5, 0.5, 0.5])
    learnable_linear_mpc = learnable_point_mass_mpc_different_params
    x = [x0]
    u = []

    p_global = learnable_linear_mpc.ocp.p_global_values

    for step in range(100):
        u_star, _, status = learnable_linear_mpc.policy(x[-1], p_global=p_global)
        assert status == 0, f"Did not converge to a solution in step {step}"
        u.append(u_star)
        x.append(learnable_linear_mpc.ocp_batch_solver.ocp_solvers[0].get(1, "x"))
        assert status == 0

    x = np.array(x)
    u = np.array(u)

    assert (
        np.median(x[-10:, 0]) <= 1e-1
        and np.median(x[-10:, 1]) <= 1e-1
        and np.median(u[-10:]) <= 1e-1
    )
