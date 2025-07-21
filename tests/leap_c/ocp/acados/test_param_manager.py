import numpy as np
import torch
from acados_template import AcadosOcp, AcadosOcpSolver

from leap_c.ocp.acados.parameters import AcadosParamManager, Parameter
from leap_c.ocp.acados.torch import AcadosDiffMpc


def test_param_manager_combine_parameter_values(
    acados_test_ocp_with_stagewise_varying_params: AcadosOcp,
    nominal_stagewise_params: tuple[Parameter, ...],
    rng: np.random.Generator,
) -> None:
    """
    Test the addition of parameters to the AcadosParamManager and verify correct
    retrieval and mapping of dense parameter values.

    Args:
        acados_test_ocp_with_stagewise_varying_params: AcadosOcp instance
        with stagewise varying parameters.
        nominal_stagewise_params: Tuple of
         test parameters to overwrite.
        rng: Random number generator for reproducible noise.

    Raises:
        AssertionError: If the mapped and retrieved dense values do not match within
        the specified tolerance.
    """
    N_horizon = acados_test_ocp_with_stagewise_varying_params.solver_options.N_horizon

    acados_param_manager = AcadosParamManager(
        params=nominal_stagewise_params,
        N_horizon=N_horizon,
    )

    keys = [
        key
        for key in list(acados_param_manager.parameter_values.keys())
        if not key.startswith("indicator")
    ]

    # Get a random batch_size
    batch_size = rng.integers(low=5, high=10)

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


def test_diff_mpc_with_stagewise_params_equivalent_to_diff_mpc(
    diff_mpc: AcadosDiffMpc,
    diff_mpc_with_stagewise_varying_params: AcadosDiffMpc,
    nominal_stagewise_params: tuple[Parameter, ...],
) -> None:
    """
    Test that the diff_mpc with stagewise varying parameters is equivalent to the
    diff_mpc with global parameters by comparing the forward pass results under
    the condition that the stagewise varying parameters are set to the nominal values.
    """
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
        params=nominal_stagewise_params,
        N_horizon=N_horizon,
    )
    p_stagewise = parameter_manager.combine_parameter_values()

    x0 = np.array([1.0, 1.0, 0.0, 0.0])

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


def test_stagewise_solution_matches_global_solver_for_initial_reference_change(
    nominal_stagewise_params: tuple[Parameter, ...],
    acados_test_ocp_no_p_global: AcadosOcp,
    diff_mpc_with_stagewise_varying_params: AcadosDiffMpc,
    diff_mpc: AcadosDiffMpc,
    rng: np.random.Generator,
) -> None:
    """
    Test that setting parameters stagewise has the expected effect by comparing it to
    an ocp_solver with global parameters and nonlinear_ls cost.
    """
    global_solver = AcadosOcpSolver(acados_test_ocp_no_p_global)

    ocp = diff_mpc_with_stagewise_varying_params.diff_mpc_fun.ocp
    pm = AcadosParamManager(
        params=nominal_stagewise_params,
        N_horizon=ocp.solver_options.N_horizon,
    )

    p_global_values = pm.p_global_values
    p_stagewise = pm.combine_parameter_values()

    xref_0 = rng.random(size=4)
    uref_0 = rng.random(size=2)
    yref_0 = np.concatenate((xref_0, uref_0))

    p_global_values["xref", 0] = xref_0
    p_global_values["uref", 0] = uref_0

    global_solver.cost_set(stage_=0, field_="yref", value_=yref_0)

    x0 = ocp.constraints.x0

    _ = global_solver.solve_for_x0(x0_bar=x0)

    u_global = np.vstack(
        [
            global_solver.get(stage_=stage, field_="u")
            for stage in range(ocp.solver_options.N_horizon)
        ]
    )

    x_global = np.vstack(
        [
            global_solver.get(stage_=stage, field_="x")
            for stage in range(ocp.solver_options.N_horizon + 1)
        ]
    )

    p_global = p_global_values.cat.full().flatten().reshape(1, ocp.dims.np_global)
    x0 = torch.tensor(x0, dtype=torch.float32).reshape(1, -1)

    sol_pert = diff_mpc_with_stagewise_varying_params.forward(
        x0=x0, p_global=p_global, p_stagewise=p_stagewise
    )

    u_stagewise = sol_pert[3].detach().numpy().reshape(-1, ocp.dims.nu)
    x_stagewise = sol_pert[2].detach().numpy().reshape(-1, ocp.dims.nx)

    # TODO: Use flattened_iterate.allclose() when available for batch iterates.
    # NOTE: Use flattened_iterate = global_solver.store_iterate_to_flat_obj()
    # NOTE: and sol_flattened_batch_iterate = sol_pert[0].iterate

    assert np.allclose(
        u_global,
        u_stagewise,
        atol=1e-3,
        rtol=1e-3,
    ), "The control trajectory does not match between global and stagewise diff MPC."

    assert np.allclose(
        x_global,
        x_stagewise,
        atol=1e-3,
        rtol=1e-3,
    ), "The state trajectory does not match between global and stagewise diff MPC."

    sol_nom = diff_mpc.forward(x0=x0)

    u_stagewise_nom = sol_nom[3].detach().numpy().reshape(-1, ocp.dims.nu)
    x_stagewise_nom = sol_nom[2].detach().numpy().reshape(-1, ocp.dims.nx)

    assert not np.allclose(
        u_stagewise_nom,
        u_stagewise,
        atol=1e-3,
        rtol=1e-3,
    ), (
        "The control trajectory matches between nominal and stagewise diff MPC \
            despite different initial reference."
    )

    assert not np.allclose(
        x_stagewise_nom,
        x_stagewise,
        atol=1e-3,
        rtol=1e-3,
    ), (
        "The state trajectory matches between nominal and stagewise diff MPC \
            despite different initial reference."
    )
