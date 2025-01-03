import numpy as np
import numpy.testing as npt
import pytest
from leap_c.examples.linear_system import LinearSystemMPC
from leap_c.mpc import MPC, MPCInput, MPCOutput, MPCParameter


def mpc_outputs_assert_allclose(
    mpc_output: MPCOutput, mpc_output2: MPCOutput, test_u_star: bool
):
    allclose = True
    for fld in mpc_output._fields:
        val1 = getattr(mpc_output, fld)
        val2 = getattr(mpc_output2, fld)
        if fld == "u0" and not test_u_star:
            continue  # Testing u_star when different u0 were given makes no sense
        if isinstance(val1, np.ndarray):
            tolerance = (
                1e-5 if not fld.startswith("d") else 1e-3
            )  # 1e-3 is probably close enough for gradients, at least thats also what we do in pytorch.gradcheck wrt the numerical gradient
            assert np.allclose(
                val1, val2, atol=tolerance
            ), f"Field {fld} not close, maximal difference is {np.abs(val1 - val2).max()}"
        elif isinstance(val1, type(None)):
            assert val1 == val2
        else:
            raise NotImplementedError(
                "Only np.ndarray fields known. Did new fields get added to MPCOutput?"
            )
    return allclose


def test_stage_cost(linear_mpc: MPC):
    x = np.array([0.0, 0.0])
    u = np.array([0.0])

    stage_cost = linear_mpc.stage_cost(x, u)

    assert stage_cost == 0.0


def test_stage_cons(linear_mpc: MPC):
    x = np.array([2.0, 1.0])
    u = np.array([0.0])

    stage_cons = linear_mpc.stage_cons(x, u)

    npt.assert_array_equal(stage_cons["ubx"], np.array([1.0, 0.0]))


def test_statelessness(
    x0: np.ndarray = np.array([0.5, 0.5]), u0: np.ndarray = np.array([0.5])
):
    # Create MPC with some stateless and some global parameters
    lin_mpc = LinearSystemMPC(learnable_params=["A", "B", "Q", "R", "f"])
    mpc_input_standard = MPCInput(x0=x0, u0=u0)
    solution_standard, _ = lin_mpc(
        mpc_input=mpc_input_standard, dudp=True, dvdp=True, dudx=True
    )
    p_global = lin_mpc.default_p_global
    assert p_global is not None
    p_global = p_global + np.ones(p_global.shape[0]) * 0.01
    p_stagewise = lin_mpc.default_p_stagewise
    assert p_stagewise is not None
    p_stagewise = p_stagewise + np.ones(p_stagewise.shape[1]) * 0.01
    assert (
        len(p_stagewise.shape) == 2
    ), f"I assumed this would be of shape ({lin_mpc.N+1}, #p_stagewise) but shape is {p_stagewise.shape}"
    params = MPCParameter(p_global, p_stagewise)
    x0_different = x0 - 0.01
    u0_different = u0 - 0.01
    mpc_input_different = MPCInput(x0=x0_different, u0=u0_different, parameters=params)
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


def test_using_mpc_state(
    linear_mpc: MPC,
    x0: np.ndarray = np.array([0.5, 0.5]),
    u0: np.ndarray = np.array([0.5]),
):
    inp = MPCInput(x0=x0, u0=u0)
    sol, state = linear_mpc(inp)
    first_iters = linear_mpc.ocp_solver.get_stats("qp_iter").sum()  # type:ignore
    assert first_iters > 1
    same_sol, _ = linear_mpc(inp, mpc_state=state)
    mpc_outputs_assert_allclose(sol, same_sol, test_u_star=True)
    second_iters = linear_mpc.ocp_solver.get_stats("qp_iter").sum()  # type:ignore
    assert second_iters == 0


def test_using_mpc_state_batched(
    learnable_linear_mpc: MPC,
    n_batch: int,
    x0: np.ndarray = np.array([0.5, 0.5]),
    u0: np.ndarray = np.array([0.5]),
):
    x0 = np.tile(x0, (n_batch, 1))
    u0 = np.tile(u0, (n_batch, 1))
    inp = MPCInput(x0=x0, u0=u0)
    sol, state = learnable_linear_mpc(inp)
    for solver in learnable_linear_mpc.ocp_batch_solver.ocp_solvers:
        qp_iters = solver.get_stats("qp_iter").sum()  # type:ignore
        assert qp_iters > 1
    same_sol, _ = learnable_linear_mpc(inp, mpc_state=state)
    for solver in learnable_linear_mpc.ocp_batch_solver.ocp_solvers:
        qp_iters = solver.get_stats("qp_iter").sum()  # type:ignore
        assert qp_iters == 0
    mpc_outputs_assert_allclose(sol, same_sol, test_u_star=True)


def test_closed_loop(
    learnable_linear_mpc: MPC,
    x0: np.ndarray = np.array([0.5, 0.5]),
):
    x = [x0]
    u = []

    p_global = learnable_linear_mpc.ocp.p_global_values

    for step in range(100):
        u_star, _, status = learnable_linear_mpc.policy(x[-1], p_global=p_global)
        assert status == 0, f"Did not converge to a solution in step {step}"
        u.append(u_star)
        x.append(learnable_linear_mpc.ocp_solver.get(1, "x"))
        assert learnable_linear_mpc.ocp_solver.get_status() == 0

    x = np.array(x)
    u = np.array(u)

    assert (
        np.median(x[-10:, 0]) <= 1e-1
        and np.median(x[-10:, 1]) <= 1e-1
        and np.median(u[-10:]) <= 1e-1
    )


if __name__ == "__main__":
    pytest.main([__file__])
