from dataclasses import asdict

import casadi as ca
import numpy as np
import pytest
import torch
from acados_template import AcadosOcp, AcadosOcpOptions

from leap_c.jax import AcadosDiffMpcJax
from leap_c.parameters import AcadosParameterManager
from leap_c.parameters.data import _AcadosParameter
from leap_c.torch import AcadosDiffMpcTorch


@pytest.fixture(scope="session")
def nominal_params() -> tuple[_AcadosParameter, ...]:
    return (
        _AcadosParameter(
            name="m",
            default=np.array([1.0]),
            interface="non-differentiable",
        ),
        _AcadosParameter(
            name="cx",
            default=np.array([0.1]),
            interface="non-differentiable",
        ),
        _AcadosParameter(
            name="cy",
            default=np.array([0.1]),
            interface="non-differentiable",
        ),
        _AcadosParameter(
            name="q_diag",
            default=np.array([1.0, 1.0, 1.0, 1.0]),
            interface="differentiable",
        ),
        _AcadosParameter(
            name="r_diag",
            default=np.array([0.1, 0.1]),
            interface="differentiable",
        ),
        _AcadosParameter(
            name="q_diag_e",
            default=np.array([1.0, 1.0, 1.0, 1.0]),
            interface="differentiable",
        ),
        _AcadosParameter(
            name="xref",
            default=np.array([0.0, 0.0, 0.0, 0.0]),
            interface="differentiable",
        ),
        _AcadosParameter(
            name="uref",
            default=np.array([0.0, 0.0]),
            interface="differentiable",
        ),
        _AcadosParameter(
            name="xref_e",
            default=np.array([0.0, 0.0, 0.0, 0.0]),
            interface="differentiable",
        ),
    )


def get_A_disc(
    m: float | ca.SX,
    cx: float | ca.SX,
    cy: float | ca.SX,
    dt: float | ca.SX,
) -> np.ndarray | ca.SX:
    if any(isinstance(i, ca.SX) for i in [m, cx, cy, dt]):
        return ca.vertcat(
            ca.horzcat(1, 0, dt, 0),
            ca.horzcat(0, 1, 0, dt),
            ca.horzcat(0, 0, ca.exp(-cx * dt / m), 0),
            ca.horzcat(0, 0, 0, ca.exp(-cy * dt / m)),
        )

    return np.array(
        [
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, np.exp(-cx * dt / m), 0],
            [0, 0, 0, np.exp(-cy * dt / m)],
        ]
    )


def get_B_disc(
    m: float | ca.SX,
    cx: float | ca.SX,
    cy: float | ca.SX,
    dt: float | ca.SX,
) -> np.ndarray | ca.SX:
    if any(isinstance(i, ca.SX) for i in [m, cx, cy, dt]):
        return ca.vertcat(
            ca.horzcat(0, 0),
            ca.horzcat(0, 0),
            ca.horzcat((m / cx) * (1 - ca.exp(-cx * dt / m)), 0),
            ca.horzcat(0, (m / cy) * (1 - ca.exp(-cy * dt / m))),
        )

    return np.array(
        [
            [0, 0],
            [0, 0],
            [(m / cx) * (1 - np.exp(-cx * dt / m)), 0],
            [0, (m / cy) * (1 - np.exp(-cy * dt / m))],
        ]
    )


@pytest.fixture(scope="session", params=["exact", "gn"])
def ocp_options(request: pytest.FixtureRequest) -> AcadosOcpOptions:
    """Configure the OCP options."""
    ocp_options = AcadosOcpOptions()
    ocp_options.integrator_type = "DISCRETE"
    ocp_options.nlp_solver_type = "SQP"
    ocp_options.hessian_approx = "EXACT" if request.param == "exact" else "GAUSS_NEWTON"
    ocp_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
    ocp_options.qp_solver_ric_alg = 1
    ocp_options.with_value_sens_wrt_params = True
    ocp_options.with_solution_sens_wrt_params = True
    ocp_options.with_batch_functionality = True

    ocp_options.tf = 2.0
    ocp_options.N_horizon = 10

    return ocp_options


@pytest.fixture(scope="session")
def nominal_stagewise_params(
    nominal_params: tuple[_AcadosParameter, ...],
    ocp_options: AcadosOcpOptions,
) -> tuple[_AcadosParameter, ...]:
    """Copy nominal_params and modify specific parameters to be stagewise."""
    N_horizon = ocp_options.N_horizon
    # Override specific fields for stage-wise parameters
    # q_diag_e and xref_e are their own parameters, only adding fields up to N_horizon - 1.
    stagewise_overrides = {
        "q_diag": {"splits": list(range(N_horizon))},
        "xref": {"splits": list(range(N_horizon))},
        "uref": {"splits": list(range(N_horizon))},
    }

    modified_params = []
    for param in nominal_params:
        if param.name in stagewise_overrides:
            # Create new parameter with overridden fields
            kwargs = asdict(param)
            kwargs.update(stagewise_overrides[param.name])
            modified_params.append(_AcadosParameter(**kwargs))
        else:
            modified_params.append(param)

    return tuple(modified_params)


@pytest.fixture(scope="session")
def acados_test_ocp_no_p_global(
    ocp_options: AcadosOcpOptions,
    nominal_params: tuple[_AcadosParameter, ...],
) -> AcadosOcp:
    """Define a simple AcadosOcp for testing purposes."""
    name = "test_ocp"

    ocp = AcadosOcp()

    ocp.solver_options.integrator_type = ocp_options.integrator_type
    ocp.solver_options.nlp_solver_type = ocp_options.nlp_solver_type
    ocp.solver_options.hessian_approx = ocp_options.hessian_approx
    ocp.solver_options.qp_solver = ocp_options.qp_solver
    ocp.solver_options.qp_solver_ric_alg = ocp_options.qp_solver_ric_alg
    ocp.solver_options.tf = ocp_options.tf
    ocp.solver_options.N_horizon = ocp_options.N_horizon

    param_defaults = {p.name: p.default for p in nominal_params}

    ocp.model.name = name

    ocp.model.x = ca.SX.sym("x", 4)
    ocp.model.u = ca.SX.sym("u", 2)

    ocp.dims.nx = ocp.model.x.shape[0]
    ocp.dims.nu = ocp.model.u.shape[0]

    kwargs = {
        "m": param_defaults["m"].item(),
        "cx": param_defaults["cx"].item(),
        "cy": param_defaults["cy"].item(),
        "dt": ocp.solver_options.tf / ocp.solver_options.N_horizon,
    }

    ocp.model.disc_dyn_expr = (
        get_A_disc(**kwargs) @ ocp.model.x + get_B_disc(**kwargs) @ ocp.model.u
    )

    # Initial stage cost
    ocp.cost.cost_type_0 = "NONLINEAR_LS"
    ocp.model.cost_y_expr_0 = ca.vertcat(
        ocp.model.x,
        ocp.model.u,
    )
    ocp.cost.yref_0 = np.concatenate(
        [
            param_defaults["xref"],
            param_defaults["uref"],
        ]
    )

    ocp.cost.W_0 = np.diag(
        np.concatenate(
            [
                param_defaults["q_diag"],
                param_defaults["r_diag"],
            ]
        )
    )
    ocp.cost.W_0 = ocp.cost.W_0 @ ocp.cost.W_0.T

    # Intermediate stage costs
    ocp.cost.cost_type = "NONLINEAR_LS"
    ocp.model.cost_y_expr = ca.vertcat(
        ocp.model.x,
        ocp.model.u,
    )
    ocp.cost.yref = ocp.cost.yref_0
    ocp.cost.W = ocp.cost.W_0

    # Terminal cost
    ocp.cost.cost_type_e = "NONLINEAR_LS"
    ocp.model.cost_y_expr_e = ocp.model.x
    ocp.cost.yref_e = param_defaults["xref_e"]
    ocp.cost.W_e = np.diag(param_defaults["q_diag_e"])
    ocp.cost.W_e = ocp.cost.W_e @ ocp.cost.W_e.T

    ocp.constraints.x0 = np.array([1.0, 1.0, 0.0, 0.0])

    Fmax = 10.0
    # Box constraints on u
    ocp.constraints.lbu = np.array([-Fmax, -Fmax])
    ocp.constraints.ubu = np.array([Fmax, Fmax])
    ocp.constraints.idxbu = np.array([0, 1])

    ocp.constraints.lbx = np.array([0.05, 0.05, -20.0, -20.0])
    ocp.constraints.ubx = np.array([3.95, 0.95, 20.0, 20.0])
    ocp.constraints.idxbx = np.array([0, 1, 2, 3])

    ocp.constraints.idxsbx = np.array([0, 1, 2, 3])

    ns = ocp.constraints.idxsbx.size
    ocp.cost.zl = 10000 * np.ones((ns,))
    ocp.cost.Zl = 10 * np.ones((ns,))
    ocp.cost.zu = 10000 * np.ones((ns,))
    ocp.cost.Zu = 10 * np.ones((ns,))

    return ocp


def define_external_cost(ocp: AcadosOcp, param_manager: AcadosParameterManager):
    y = ca.vertcat(
        ocp.model.x,
        ocp.model.u,
    )
    yref = ca.vertcat(
        param_manager.get(name="xref"),
        param_manager.get(name="uref"),
    )
    W_sqrt = ca.diag(
        ca.vertcat(
            param_manager.get(name="q_diag"),
            param_manager.get(name="r_diag"),
        )
    )
    xref_e = param_manager.get(name="xref_e")
    Q_sqrt_e = ca.diag(param_manager.get(name="q_diag_e"))

    stage_cost = 0.5 * (
        ca.mtimes(
            [
                ca.transpose(y - yref),
                W_sqrt,
                ca.transpose(W_sqrt),
                y - yref,
            ]
        )
    )

    # # Initial stage costs
    ocp.cost.cost_type_0 = "EXTERNAL"
    ocp.model.cost_expr_ext_cost_0 = stage_cost
    # # Intermediate stage costs
    ocp.cost.cost_type = "EXTERNAL"
    ocp.model.cost_expr_ext_cost = stage_cost
    # Terminal cost
    ocp.cost.cost_type_e = "EXTERNAL"
    ocp.model.cost_expr_ext_cost_e = 0.5 * (
        ca.mtimes(
            [
                ca.transpose(ocp.model.x - xref_e),
                Q_sqrt_e,
                ca.transpose(Q_sqrt_e),
                ocp.model.x - xref_e,
            ]
        )
    )


def define_discrete_dynamics(ocp: AcadosOcp, param_manager: AcadosParameterManager) -> None:
    kwargs = {
        "m": param_manager.get(name="m"),
        "cx": param_manager.get(name="cx"),
        "cy": param_manager.get(name="cy"),
        "dt": ocp.solver_options.tf / ocp.solver_options.N_horizon,
    }
    ocp.model.disc_dyn_expr = (
        get_A_disc(**kwargs) @ ocp.model.x + get_B_disc(**kwargs) @ ocp.model.u
    )


def define_nonlinear_discrete_dynamics(
    ocp: AcadosOcp, param_manager: AcadosParameterManager
) -> None:
    """Discrete dynamics with a nonlinear term.

    Identical to :func:`define_discrete_dynamics` but adds a ``sin`` term to the
    velocity states. The ``sin`` has a nonzero second derivative, so under an
    EXACT-hessian SQP the multiplier-weighted dynamics curvature enters the
    Hessian of the Lagrangian (the purely linear model contributes none and is
    effectively a convex QP). ``k_nl`` is chosen large enough that the assembled
    exact Hessian is genuinely indefinite at the nominal operating point (acados
    ``get_hessian_block`` reports a min eigenvalue of about -0.1, versus +0.002
    for a small term). Paired with the wide, inactive control bounds of
    :func:`define_loose_constraints`, the optimal solution is interior and the
    solution map stays smooth, so the finite-difference backward gradient checks
    remain valid despite the indefinite full Hessian.
    """
    kwargs = {
        "m": param_manager.get(name="m"),
        "cx": param_manager.get(name="cx"),
        "cy": param_manager.get(name="cy"),
        "dt": ocp.solver_options.tf / ocp.solver_options.N_horizon,
    }
    dt = kwargs["dt"]
    k_nl = 12.0  # strength of the nonlinear term (tuned for indefiniteness)
    nonlinear = ca.vertcat(
        0,
        0,
        k_nl * ca.sin(ocp.model.x[0]),
        k_nl * ca.sin(ocp.model.x[1]),
    )
    ocp.model.disc_dyn_expr = (
        get_A_disc(**kwargs) @ ocp.model.x + get_B_disc(**kwargs) @ ocp.model.u + dt * nonlinear
    )


def define_constraints(ocp: AcadosOcp, param_manager: AcadosParameterManager) -> None:
    """Define constraints for the OCP."""
    ocp.constraints.x0 = np.array([1.0, 0.5, 0.0, 0.0])

    Fmax = 10.0
    # Box constraints on u
    ocp.constraints.lbu = np.array([-Fmax, -Fmax])
    ocp.constraints.ubu = np.array([Fmax, Fmax])
    ocp.constraints.idxbu = np.array([0, 1])

    ocp.constraints.lbx = np.array([0.05, 0.05, -20.0, -20.0])
    ocp.constraints.ubx = np.array([3.95, 0.95, 20.0, 20.0])
    ocp.constraints.idxbx = np.array([0, 1, 2, 3])

    ocp.constraints.idxsbx = np.array([0, 1, 2, 3])

    ns = ocp.constraints.idxsbx.size
    ocp.cost.zl = 10000 * np.ones((ns,))
    ocp.cost.Zl = 10 * np.ones((ns,))
    ocp.cost.zu = 10000 * np.ones((ns,))
    ocp.cost.Zu = 10 * np.ones((ns,))


def define_loose_constraints(ocp: AcadosOcp) -> None:
    """Initial state plus wide (inactive) control bounds, no state constraints.

    Used by the indefinite-Hessian fixtures: with the bounds inactive the
    optimal solution is interior and the solution map is smooth, so the
    finite-difference gradient checks stay valid even though the full Hessian is
    indefinite. (The linear fixtures use the tighter :func:`define_constraints`
    with active bounds and slacks.)
    """
    ocp.constraints.x0 = np.array([1.0, 0.5, 0.0, 0.0])
    u_bound = 100.0
    ocp.constraints.lbu = np.array([-u_bound, -u_bound])
    ocp.constraints.ubu = np.array([u_bound, u_bound])
    ocp.constraints.idxbu = np.array([0, 1])


@pytest.fixture(scope="session", params=["external", "nonlinear_ls"])
def acados_test_ocp(
    ocp_options: AcadosOcpOptions,
    nominal_params: tuple[_AcadosParameter, ...],
    request: pytest.FixtureRequest,
) -> AcadosOcp:
    """Define a simple AcadosOcp for testing purposes."""
    name = "test_ocp"

    ocp = AcadosOcp()
    ocp.solver_options = ocp_options

    param_manager = AcadosParameterManager(N_horizon=ocp.solver_options.N_horizon)
    for param in nominal_params:
        param_manager.register_parameter(
            name=param.name,
            default=param.default,
            differentiable=(param.interface == "differentiable"),
            splits=param.splits,
        )

    ocp.model.name = name

    ocp.model.x = ca.SX.sym("x", 4)
    ocp.model.u = ca.SX.sym("u", 2)

    define_discrete_dynamics(ocp, param_manager)
    define_constraints(ocp, param_manager)
    # Define cost
    if request.param == "external":
        define_external_cost(ocp, param_manager)
    elif request.param == "nonlinear_ls":
        # Initial stage cost
        ocp.cost.cost_type_0 = "NONLINEAR_LS"
        ocp.model.cost_y_expr_0 = ca.vertcat(
            ocp.model.x,
            ocp.model.u,
        )
        ocp.cost.yref_0 = ca.vertcat(
            param_manager.get("xref"),
            param_manager.get("uref"),
        )
        ocp.cost.W_0 = ca.diag(
            ca.vertcat(
                param_manager.get("q_diag"),
                param_manager.get("r_diag"),
            )
        )
        ocp.cost.W_0 = ca.mtimes(ocp.cost.W_0, ca.transpose(ocp.cost.W_0))

        # Intermediate stage costs
        ocp.cost.cost_type = "NONLINEAR_LS"
        ocp.model.cost_y_expr = ca.vertcat(
            ocp.model.x,
            ocp.model.u,
        )
        ocp.cost.yref = ca.vertcat(
            param_manager.get("xref"),
            param_manager.get("uref"),
        )
        ocp.cost.W = ca.diag(
            ca.vertcat(
                param_manager.get("q_diag"),
                param_manager.get("r_diag"),
            )
        )
        ocp.cost.W = ca.mtimes(ocp.cost.W, ca.transpose(ocp.cost.W))

        # Terminal cost
        ocp.cost.cost_type_e = "NONLINEAR_LS"
        ocp.model.cost_y_expr_e = ocp.model.x
        ocp.cost.yref_e = param_manager.get("xref_e")
        ocp.cost.W_e = ca.diag(
            param_manager.get("q_diag_e"),
        )
        ocp.cost.W_e = ca.mtimes(ocp.cost.W_e, ca.transpose(ocp.cost.W_e))

    ocp._test_pm = param_manager
    return ocp


# TODO: Remove this fixture once the nominal_stagewise_params can be used in acados_test_ocp
@pytest.fixture(scope="session")
def acados_test_ocp_with_stagewise_varying_params(
    ocp_options: AcadosOcpOptions,
    nominal_stagewise_params: tuple[_AcadosParameter, ...],
) -> AcadosOcp:
    """Define a simple AcadosOcp for testing purposes."""
    name = "test_ocp_with_stagewise_varying_params"

    ocp = AcadosOcp()
    ocp.solver_options = ocp_options

    param_manager = AcadosParameterManager(N_horizon=ocp.solver_options.N_horizon)
    for param in nominal_stagewise_params:
        param_manager.register_parameter(
            name=param.name,
            default=param.default,
            differentiable=(param.interface == "differentiable"),
            splits=param.splits,
        )

    ocp.model.name = name

    ocp.model.x = ca.SX.sym("x", 4)
    ocp.model.u = ca.SX.sym("u", 2)

    define_external_cost(ocp, param_manager)
    define_discrete_dynamics(ocp, param_manager)
    define_constraints(ocp, param_manager)

    ocp._test_pm = param_manager
    return ocp


@pytest.fixture(scope="session")
def acados_test_ocp_indefinite_hess(
    ocp_options: AcadosOcpOptions,
    nominal_params: tuple[_AcadosParameter, ...],
) -> AcadosOcp:
    """AcadosOcp whose nonlinear dynamics give an indefinite full Hessian.

    Same 4-state / 2-input model, params, and external cost as
    :func:`acados_test_ocp`, but the discrete dynamics carry a smooth nonlinear
    term (see :func:`define_nonlinear_discrete_dynamics`) and the inequality
    bounds are loosened (see :func:`define_loose_constraints`). Under an
    EXACT-hessian SQP the nonlinear term makes the Hessian of the Lagrangian
    indefinite, exercising the nonlinear-MPC backward path that the purely linear
    ``acados_test_ocp`` does not.
    """
    ocp = AcadosOcp()
    ocp.solver_options = ocp_options

    param_manager = AcadosParameterManager(N_horizon=ocp.solver_options.N_horizon)
    for param in nominal_params:
        param_manager.register_parameter(
            name=param.name,
            default=param.default,
            differentiable=(param.interface == "differentiable"),
            splits=param.splits,
        )

    ocp.model.name = "test_ocp_indefinite_hess"

    ocp.model.x = ca.SX.sym("x", 4)
    ocp.model.u = ca.SX.sym("u", 2)

    define_nonlinear_discrete_dynamics(ocp, param_manager)
    define_external_cost(ocp, param_manager)
    define_loose_constraints(ocp)

    ocp._test_pm = param_manager
    return ocp


@pytest.fixture(scope="session")
def acados_test_ocp_indefinite_hess_stagewise(
    ocp_options: AcadosOcpOptions,
    nominal_stagewise_params: tuple[_AcadosParameter, ...],
) -> AcadosOcp:
    """Like :func:`acados_test_ocp_indefinite_hess` but with stagewise params."""
    ocp = AcadosOcp()
    ocp.solver_options = ocp_options

    param_manager = AcadosParameterManager(N_horizon=ocp.solver_options.N_horizon)
    for param in nominal_stagewise_params:
        param_manager.register_parameter(
            name=param.name,
            default=param.default,
            differentiable=(param.interface == "differentiable"),
            splits=param.splits,
        )

    ocp.model.name = "test_ocp_indefinite_hess_stagewise"

    ocp.model.x = ca.SX.sym("x", 4)
    ocp.model.u = ca.SX.sym("u", 2)

    define_nonlinear_discrete_dynamics(ocp, param_manager)
    define_external_cost(ocp, param_manager)
    define_loose_constraints(ocp)

    ocp._test_pm = param_manager
    return ocp


@pytest.fixture(scope="session")
def _param_manager(acados_test_ocp: AcadosOcp) -> AcadosParameterManager:
    pm = acados_test_ocp._test_pm
    del acados_test_ocp._test_pm
    return pm


@pytest.fixture(scope="session")
def diff_mpc(
    acados_test_ocp: AcadosOcp, _param_manager: AcadosParameterManager
) -> AcadosDiffMpcTorch:
    return AcadosDiffMpcTorch(
        ocp=acados_test_ocp,
        parameter_manager=_param_manager,
        initializer=None,
        discount_factor=None,
        dtype=torch.float64,
    )


@pytest.fixture(scope="session")
def diff_mpc_jax(
    acados_test_ocp: AcadosOcp, _param_manager: AcadosParameterManager
) -> AcadosDiffMpcJax:
    return AcadosDiffMpcJax(
        ocp=acados_test_ocp,
        parameter_manager=_param_manager,
        initializer=None,
        discount_factor=None,
    )


@pytest.fixture(scope="session")
def _param_manager_stagewise(
    acados_test_ocp_with_stagewise_varying_params: AcadosOcp,
) -> AcadosParameterManager:
    pm = acados_test_ocp_with_stagewise_varying_params._test_pm
    del acados_test_ocp_with_stagewise_varying_params._test_pm
    return pm


@pytest.fixture(scope="session")
def diff_mpc_with_stagewise_varying_params(
    acados_test_ocp_with_stagewise_varying_params: AcadosOcp,
    _param_manager_stagewise: AcadosParameterManager,
) -> AcadosDiffMpcTorch:
    return AcadosDiffMpcTorch(
        ocp=acados_test_ocp_with_stagewise_varying_params,
        parameter_manager=_param_manager_stagewise,
        initializer=None,
        discount_factor=None,
        dtype=torch.float64,
    )


@pytest.fixture(scope="session")
def diff_mpc_jax_with_stagewise_varying_params(
    acados_test_ocp_with_stagewise_varying_params: AcadosOcp,
    _param_manager_stagewise: AcadosParameterManager,
) -> AcadosDiffMpcJax:
    return AcadosDiffMpcJax(
        ocp=acados_test_ocp_with_stagewise_varying_params,
        parameter_manager=_param_manager_stagewise,
        initializer=None,
        discount_factor=None,
    )


@pytest.fixture(scope="session")
def diff_mpc_indefinite_hess(
    acados_test_ocp_indefinite_hess: AcadosOcp,
) -> AcadosDiffMpcTorch:
    pm = acados_test_ocp_indefinite_hess._test_pm
    del acados_test_ocp_indefinite_hess._test_pm
    return AcadosDiffMpcTorch(
        ocp=acados_test_ocp_indefinite_hess,
        parameter_manager=pm,
        initializer=None,
        discount_factor=None,
        dtype=torch.float64,
    )


@pytest.fixture(scope="session")
def diff_mpc_indefinite_hess_stagewise(
    acados_test_ocp_indefinite_hess_stagewise: AcadosOcp,
) -> AcadosDiffMpcTorch:
    pm = acados_test_ocp_indefinite_hess_stagewise._test_pm
    del acados_test_ocp_indefinite_hess_stagewise._test_pm
    return AcadosDiffMpcTorch(
        ocp=acados_test_ocp_indefinite_hess_stagewise,
        parameter_manager=pm,
        initializer=None,
        discount_factor=None,
        dtype=torch.float64,
    )


@pytest.fixture(scope="session")
def export_dir(tmp_path_factory: pytest.TempPathFactory) -> str:
    """Fixture to create a temporary directory for exporting files."""
    return str(tmp_path_factory.mktemp("export_dir"))


@pytest.fixture(scope="session")
def rng() -> np.random.Generator:
    """Fixture to provide a random number generator."""
    return np.random.default_rng(42)
