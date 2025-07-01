import numpy as np
import torch
from acados_template import AcadosOcp, AcadosOcpSolver

from leap_c.ocp.acados.parameters import AcadosParamManager, Parameter
from leap_c.ocp.acados.torch import AcadosDiffMpc


def test_param_manager_combine_parameter_values(
    acados_test_ocp_with_stagewise_varying_params: AcadosOcp,
    nominal_stage_wise_params: tuple[Parameter, ...],
    rng: np.random.Generator,
) -> None:
    """
    Test the addition of parameters to the AcadosParamManager and verify correct
    retrieval and mapping of dense parameter values.

    Args:
        acados_test_ocp_with_stagewise_varying_params (AcadosOcp): AcadosOcp instance with
         stagewise varying parameters.
        nominal_varying_params_for_param_manager_tests (tuple[Parameter, ...]): Tuple of
         test parameters to overwrite.
        rng (np.random.Generator): Random number generator for reproducible noise.

    Raises:
        AssertionError: If the mapped and retrieved dense values do not match within
        the specified tolerance.
    """
    acados_param_manager = AcadosParamManager(
        params=nominal_stage_wise_params,
        ocp=acados_test_ocp_with_stagewise_varying_params,
    )

    keys = [
        key
        for key in list(acados_param_manager.parameter_values.keys())
        if not key.startswith("indicator")
    ]

    # Get a random batch_size
    batch_size = rng.integers(low=5, high=10)

    N_horizon = acados_test_ocp_with_stagewise_varying_params.solver_options.N_horizon
    # Pick random keys

    # Build random overwrites
    overwrite = {}
    for key in keys:
        overwrite[key] = rng.random(
            size=(
                batch_size,
                N_horizon + 1,
                acados_param_manager.parameter_values[key].shape[0],
            )
        )

    res = acados_param_manager.combine_parameter_values(**overwrite)

    assert res.shape == (
        batch_size,
        N_horizon + 1,
        acados_param_manager.parameter_values.cat.shape[0],
    ), "The shape of the combined parameter values does not match the expected shape."


def test_diff_mpc_with_stage_wise_varying_params_equivalent_to_diff_mpc(
    diff_mpc: AcadosDiffMpc,
    diff_mpc_with_stagewise_varying_params: AcadosDiffMpc,
    nominal_stage_wise_params: tuple[Parameter, ...],
) -> None:
    mpc = {
        "stagewise": diff_mpc_with_stagewise_varying_params,
        "global": diff_mpc,
    }

    N_horizon = (
        mpc["global"]
        .diff_mpc_fun.forward_batch_solver.ocp_solvers[0]
        .acados_ocp.solver_options.N_horizon
    )

    # Create a parameter manager for the stagewise varying parameters.
    parameter_manager = AcadosParamManager(
        params=nominal_stage_wise_params,
        N_horizon=N_horizon,
    )
    p_stagewise = parameter_manager.combine_parameter_values()

    x0 = np.array([1.0, 1.0, 0.0, 0.0])

    # ctx, u0, x, u, value
    sol_forward = {}
    sol_forward["global"] = mpc["global"].forward(
        x0=torch.tensor(x0, dtype=torch.float32).reshape(1, -1)
    )
    sol_forward["stagewise"] = mpc["stagewise"].forward(
        x0=torch.tensor(x0, dtype=torch.float32).reshape(1, -1),
        p_stagewise=p_stagewise,
    )

    for key, val in sol_forward.items():
        print(f"sol_forward_{key} u:", val[3])

    out = ["ctx", "u0", "x", "u", "value"]
    for idx, label in enumerate(out[1:]):
        assert np.allclose(
            sol_forward["global"][idx + 1].detach().numpy(),
            sol_forward["stagewise"][idx + 1].detach().numpy(),
            atol=1e-3,
            rtol=1e-3,
        ), f"The {label} does not match between global and stagewise varying diff MPC."


def test_stagewise_varying_params_equivalent(
    # nominal_stage_wise_params: tuple[Parameter, ...],
    nominal_params: tuple[Parameter, ...],
    acados_test_ocp_no_p_global: AcadosOcp,
    # diff_mpc_with_stagewise_varying_params: AcadosDiffMpc,
    diff_mpc: AcadosDiffMpc,
    rng: np.random.Generator,
) -> None:
    global_solver = AcadosOcpSolver(acados_test_ocp_no_p_global)

    # ocp = diff_mpc_with_stagewise_varying_params.diff_mpc_fun.ocp
    # pm = AcadosParamManager(params=nominal_stage_wise_params, ocp=ocp)
    ocp = diff_mpc.diff_mpc_fun.ocp
    pm = AcadosParamManager(params=nominal_params, ocp=ocp)

    p_global_values = pm.p_global_values

    # xref_0 = rng.random(size=4)
    # uref_0 = rng.random(size=2)
    # yref_0 = np.concatenate((xref_0, uref_0))

    # p_global_values["xref", 0] = xref_0
    # p_global_values["uref", 0] = uref_0

    # global_solver.cost_set(stage_=0, field_="yref", value_=yref_0)

    x0 = ocp.constraints.x0

    u0_global = global_solver.solve_for_x0(x0_bar=x0)

    p_global = p_global_values.cat.full().flatten().reshape(1, ocp.dims.np_global)
    x0 = torch.tensor(x0, dtype=torch.float32).reshape(1, -1)

    # ctx, u0, x, u, value = diff_mpc_with_stagewise_varying_params.forward(
    ctx, u0, x, u, value = diff_mpc.forward(x0=x0, p_global=p_global)

    print("u0_global:", u0_global)
    print("u0:", u0)
