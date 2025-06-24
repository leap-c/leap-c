from collections.abc import Callable
from itertools import chain

import casadi as ca
import numpy as np
import pytest
from acados_template import AcadosOcp, AcadosOcpOptions

from leap_c.ocp.acados.parameters import (
    AcadosParamManager,
    Parameter,
    find_param_in_p_or_p_global,
)
from leap_c.ocp.acados.torch import AcadosImplicitLayer


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


def _create_diag_matrix(
    _q_sqrt: np.ndarray | ca.SX,
) -> np.ndarray | ca.SX:
    if any(isinstance(i, ca.SX) for i in [_q_sqrt]):
        return ca.diag(_q_sqrt)
    return np.diag(_q_sqrt)


def get_cost_expr_ext_cost(ocp: AcadosOcp, **kwargs: dict[str, ca.SX]) -> ca.SX:
    x = ocp.model.x
    u = ocp.model.u

    try:
        xref = kwargs["xref"]
    except KeyError:
        xref = find_param_in_p_or_p_global(["xref"], ocp.model)["xref"]

    try:
        uref = kwargs["uref"]
    except KeyError:
        uref = find_param_in_p_or_p_global(["uref"], ocp.model)["uref"]

    try:
        q_diag = kwargs["q_diag"]
    except KeyError:
        q_diag = find_param_in_p_or_p_global(["q_diag"], ocp.model)["q_diag"]

    try:
        r_diag = kwargs["r_diag"]
    except KeyError:
        r_diag = find_param_in_p_or_p_global(["r_diag"], ocp.model)["r_diag"]

    Q_sqrt = _create_diag_matrix(q_diag)
    R_sqrt = _create_diag_matrix(r_diag)

    return 0.5 * (
        ca.mtimes([ca.transpose(x - xref), Q_sqrt.T, Q_sqrt, x - xref])
        + ca.mtimes([ca.transpose(u - uref), R_sqrt.T, R_sqrt, u - uref])
    )


def get_cost_expr_ext_cost_e(ocp: AcadosOcp, **kwargs: dict[str, ca.SX]) -> ca.SX:
    x = ocp.model.x

    Q_sqrt_e = _create_diag_matrix(
        find_param_in_p_or_p_global(["q_diag_e"], ocp.model)["q_diag_e"]
    )

    xref_e = find_param_in_p_or_p_global(["xref_e"], ocp.model)["xref_e"]

    return 0.5 * ca.mtimes([ca.transpose(x - xref_e), Q_sqrt_e.T, Q_sqrt_e, x - xref_e])


def get_cost_W(ocp: AcadosOcp) -> ca.SX:
    """Get the cost weight matrix W for the OCP."""
    q_diag = find_param_in_p_or_p_global(["q_diag"], ocp.model)["q_diag"]
    r_diag = find_param_in_p_or_p_global(["r_diag"], ocp.model)["r_diag"]

    return ca.diag(ca.vertcat(q_diag, r_diag))


def get_cost_yref(ocp: AcadosOcp) -> ca.SX:
    """Get the cost reference vector yref for the OCP."""
    xref = find_param_in_p_or_p_global(["xref"], ocp.model)["xref"]
    uref = find_param_in_p_or_p_global(["uref"], ocp.model)["uref"]

    return ca.vertcat(xref, uref)


def get_cost_yref_e(ocp: AcadosOcp) -> ca.SX:
    """Get the cost reference vector yref_e for the OCP."""
    return find_param_in_p_or_p_global(["xref_e"], ocp.model)["xref_e"]


def define_nonlinear_ls_cost(ocp: AcadosOcp) -> None:
    """Define the cost for the AcadosOcp as a nonlinear least squares cost."""
    ocp.cost.cost_type_0 = "NONLINEAR_LS"
    ocp.cost.cost_type = "NONLINEAR_LS"
    ocp.cost.cost_type_e = "NONLINEAR_LS"

    ocp.cost.W_0 = get_cost_W(ocp=ocp)
    ocp.cost.W = get_cost_W(ocp=ocp)
    ocp.cost.W_e = ocp.cost.W[: ocp.dims.nx, : ocp.dims.nx]

    ocp.cost.yref_0 = get_cost_yref(ocp=ocp)
    ocp.cost.yref = get_cost_yref(ocp=ocp)
    ocp.cost.yref_e = get_cost_yref_e(ocp=ocp)

    ocp.model.cost_y_expr_0 = ca.vertcat(ocp.model.x, ocp.model.u)
    ocp.model.cost_y_expr = ca.vertcat(ocp.model.x, ocp.model.u)
    ocp.model.cost_y_expr_e = ocp.model.x


def define_external_cost(ocp: AcadosOcp) -> None:
    """Define the cost for the AcadosOcp as an external cost."""
    ocp.cost.cost_type_0 = "EXTERNAL"
    ocp.cost.cost_type = "EXTERNAL"
    ocp.cost.cost_type_e = "EXTERNAL"
    ocp.model.cost_expr_ext_cost_0 = get_cost_expr_ext_cost(ocp=ocp)
    ocp.model.cost_expr_ext_cost = get_cost_expr_ext_cost(ocp=ocp)
    ocp.model.cost_expr_ext_cost_e = get_cost_expr_ext_cost_e(ocp=ocp)


@pytest.fixture(scope="session", params=["external", "nonlinear_ls"])
def ocp_cost_fun(request: pytest.FixtureRequest) -> Callable:
    """Fixture to define the cost type for the AcadosOcp."""
    if request.param == "external":
        return define_external_cost
    if request.param == "nonlinear_ls":
        return define_nonlinear_ls_cost

    class UnknownCostFunctionError(ValueError):
        def __init__(self) -> None:
            super().__init__("Unknown cost function requested.")

    raise UnknownCostFunctionError


@pytest.fixture(scope="session")
def N_horizon() -> int:
    """Fixture to define the number of steps in the horizon."""
    return 10


@pytest.fixture(scope="session", params=["exact", "gn"])
def ocp_options(N_horizon: int, request: pytest.FixtureRequest) -> AcadosOcpOptions:
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
    ocp_options.N_horizon = N_horizon

    return ocp_options


@pytest.fixture(scope="session")
def acados_test_ocp(
    ocp_cost_fun: Callable,
    ocp_options: AcadosOcpOptions,
    nominal_params: tuple[Parameter, ...],
) -> AcadosOcp:
    """Define a simple AcadosOcp for testing purposes."""
    name = "test_ocp"

    ocp = AcadosOcp()

    param_manager = AcadosParamManager(N_horizon=ocp_options.N_horizon)
    [param_manager.add(param) for param in nominal_params]
    param_manager.assign_to_ocp(ocp=ocp)

    ocp.solver_options = ocp_options

    ocp.model.name = name

    ocp.dims.nu = 2
    ocp.dims.nx = 4

    ocp.model.x = ca.SX.sym("x", ocp.dims.nx)
    ocp.model.u = ca.SX.sym("u", ocp.dims.nu)

    m = param_manager.get_sym(field_="m")
    cx = param_manager.get_sym(field_="cx")
    cy = param_manager.get_sym(field_="cy")
    dt = ocp.solver_options.tf / ocp.solver_options.N_horizon
    ocp.model.disc_dyn_expr = (
        get_A_disc(m=m, cx=cx, cy=cy, dt=dt) @ ocp.model.x
        + get_B_disc(m=m, cx=cx, cy=cy, dt=dt) @ ocp.model.u
    )

    # Define cost
    ocp_cost_fun(ocp)

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

    # Cast parameters to appropriate types for acados
    # TODO: Move to assign_to_ocp in AcadosParamManager. Requires a refactor that does
    # not rely on ocp_cost_fun(ocp)
    ocp.model.p = param_manager.get_flat("p")
    ocp.model.p_global = param_manager.get_flat("p_global")

    return ocp


@pytest.fixture(scope="session")
def implicit_layer(acados_test_ocp: AcadosOcp) -> AcadosImplicitLayer:
    return AcadosImplicitLayer(
        ocp=acados_test_ocp,
        initializer=None,
        sensitivity_ocp=None,
        discount_factor=None,
    )


@pytest.fixture(scope="session")
def nominal_params() -> tuple[Parameter, ...]:
    return (
        Parameter(
            name="m",
            value=np.array([1.0]),
            lower_bound=np.array([0.5]),
            upper_bound=np.array([1.5]),
            differentiable=False,
            varying=False,
        ),
        Parameter(
            name="cx",
            value=np.array([0.1]),
            lower_bound=np.array([0.05]),
            upper_bound=np.array([0.15]),
            differentiable=False,
            varying=False,
        ),
        Parameter(
            name="cy",
            value=np.array([0.1]),
            lower_bound=np.array([0.05]),
            upper_bound=np.array([0.15]),
            differentiable=False,
            varying=False,
        ),
        Parameter(
            name="q_diag",
            value=np.array([1.0, 1.0, 1.0, 1.0]),
            lower_bound=np.array([0.5, 0.5, 0.5, 0.5]),
            upper_bound=np.array([1.5, 1.5, 1.5, 1.5]),
            differentiable=True,
            varying=False,
        ),
        Parameter(
            name="r_diag",
            value=np.array([0.1, 0.1]),
            lower_bound=np.array([0.05, 0.05]),
            upper_bound=np.array([0.15, 0.15]),
            differentiable=True,
            varying=False,
        ),
        Parameter(
            name="q_diag_e",
            value=np.array([1.0, 1.0, 1.0, 1.0]),
            lower_bound=np.array([0.5, 0.5, 0.5, 0.5]),
            upper_bound=np.array([1.5, 1.5, 1.5, 1.5]),
            differentiable=True,
            varying=False,
        ),
        Parameter(
            name="xref",
            value=np.array([0.0, 0.0, 0.0, 0.0]),
            lower_bound=np.array([-1.0, -1.0, -1.0, -1.0]),
            upper_bound=np.array([1.0, 1.0, 1.0, 1.0]),
            differentiable=True,
            varying=False,
        ),
        Parameter(
            name="uref",
            value=np.array([0.0, 0.0]),
            lower_bound=np.array([-1.0, -1.0]),
            upper_bound=np.array([1.0, 1.0]),
            differentiable=True,
            varying=False,
        ),
        Parameter(
            name="xref_e",
            value=np.array([0.0, 0.0, 0.0, 0.0]),
            lower_bound=np.array([-1.0, -1.0, -1.0, -1.0]),
            upper_bound=np.array([1.0, 1.0, 1.0, 1.0]),
            differentiable=True,
            varying=False,
        ),
    )


@pytest.fixture(scope="session")
def nominal_varying_params() -> tuple[Parameter, ...]:
    return (
        Parameter(
            name="m",
            value=np.array([1.0]),
            lower_bound=np.array([0.5]),
            upper_bound=np.array([1.5]),
            differentiable=False,
            varying=False,
        ),
        Parameter(
            name="cx",
            value=np.array([0.1]),
            lower_bound=np.array([0.05]),
            upper_bound=np.array([0.15]),
            differentiable=False,
            varying=False,
        ),
        Parameter(
            name="cy",
            value=np.array([0.1]),
            lower_bound=np.array([0.05]),
            upper_bound=np.array([0.15]),
            differentiable=False,
            varying=False,
        ),
        Parameter(
            name="q_diag",
            value=np.array([1.0, 1.0, 1.0, 1.0]),
            lower_bound=np.array([0.5, 0.5, 0.5, 0.5]),
            upper_bound=np.array([1.5, 1.5, 1.5, 1.5]),
            differentiable=True,
            varying=True,
        ),
        Parameter(
            name="r_diag",
            value=np.array([0.1, 0.1]),
            lower_bound=np.array([0.05, 0.05]),
            upper_bound=np.array([0.15, 0.15]),
            differentiable=True,
            varying=False,
        ),
        Parameter(
            name="q_diag_e",
            value=np.array([1.0, 1.0, 1.0, 1.0]),
            lower_bound=np.array([0.5, 0.5, 0.5, 0.5]),
            upper_bound=np.array([1.5, 1.5, 1.5, 1.5]),
            differentiable=True,
            varying=False,
        ),
        Parameter(
            name="xref",
            value=np.array([0.1, 0.2, 0.3, 0.4]),
            lower_bound=np.array([-1.0, -1.0, -1.0, -1.0]),
            upper_bound=np.array([1.0, 1.0, 1.0, 1.0]),
            differentiable=True,
            varying=True,
        ),
        Parameter(
            name="uref",
            value=np.array([0.5, 0.6]),
            lower_bound=np.array([-1.0, -1.0]),
            upper_bound=np.array([1.0, 1.0]),
            differentiable=True,
            varying=True,
        ),
        Parameter(
            name="xref_e",
            value=np.array([0.0, 0.0, 0.0, 0.0]),
            lower_bound=np.array([-1.0, -1.0, -1.0, -1.0]),
            upper_bound=np.array([1.0, 1.0, 1.0, 1.0]),
            differentiable=True,
            varying=False,
        ),
    )


@pytest.fixture(scope="session")
def acados_param_manager(
    N_horizon: int,
) -> AcadosParamManager:
    return AcadosParamManager(N_horizon=N_horizon)


@pytest.fixture(scope="session")
def acados_test_ocp_with_stagewise_varying_params(
    ocp_options: AcadosOcpOptions,
    nominal_varying_params: tuple[Parameter, ...],
) -> AcadosOcp:
    """Define a simple AcadosOcp for testing purposes."""
    name = "test_ocp_with_stagewise_varying_params"

    ocp = AcadosOcp()

    ocp.solver_options = ocp_options

    param_manager = AcadosParamManager(N_horizon=ocp_options.N_horizon)
    [param_manager.add(param) for param in nominal_varying_params]
    param_manager.assign_to_ocp(ocp=ocp)

    ocp.model.name = name

    ocp.dims.nu = 2
    ocp.dims.nx = 4

    ocp.model.x = ca.SX.sym("x", ocp.dims.nx)
    ocp.model.u = ca.SX.sym("u", ocp.dims.nu)

    stage_cost = []
    r_diag = param_manager.get_sym(field_="r_diag")
    R_sqrt = _create_diag_matrix(r_diag)
    for stage in range(ocp.solver_options.N_horizon):
        indicator = param_manager.get_sym(field_="indicator", stage_=stage)
        xref = param_manager.get_sym(field_="xref", stage_=stage)
        uref = param_manager.get_sym(field_="uref", stage_=stage)
        q_diag = param_manager.get_sym(field_="q_diag", stage_=stage)
        Q_sqrt = _create_diag_matrix(q_diag)

        stage_cost.append(
            indicator
            * 0.5
            * (
                ca.mtimes(
                    [
                        ca.transpose(ocp.model.x - xref),
                        Q_sqrt.T,
                        Q_sqrt,
                        ocp.model.x - xref,
                    ]
                )
                + ca.mtimes(
                    [
                        ca.transpose(ocp.model.u - uref),
                        R_sqrt.T,
                        R_sqrt,
                        ocp.model.u - uref,
                    ]
                )
            )
        )

    ocp.cost.cost_type_0 = "EXTERNAL"
    ocp.model.cost_expr_ext_cost_0 = stage_cost[0]
    ocp.cost.cost_type = "EXTERNAL"
    ocp.model.cost_expr_ext_cost = ca.sum1(ca.vertcat(*stage_cost[1:]))
    ocp.cost.cost_type_e = "EXTERNAL"
    xref_e = param_manager.get_sym(field_="xref_e")
    q_diag_e = param_manager.get_sym(field_="q_diag_e")
    Q_sqrt_e = _create_diag_matrix(q_diag_e)

    ocp.model.cost_expr_ext_cost_e = 0.5 * ca.mtimes(
        [ca.transpose(ocp.model.x - xref_e), Q_sqrt_e.T, Q_sqrt_e, ocp.model.x - xref_e]
    )

    ####
    m = param_manager.get_sym(field_="m")
    cx = param_manager.get_sym(field_="cx")
    cy = param_manager.get_sym(field_="cy")
    dt = ocp.solver_options.tf / ocp.solver_options.N_horizon
    ocp.model.disc_dyn_expr = (
        get_A_disc(m=m, cx=cx, cy=cy, dt=dt) @ ocp.model.x
        + get_B_disc(m=m, cx=cx, cy=cy, dt=dt) @ ocp.model.u
    )

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

    # Cast parameters to appropriate types for acados
    ocp.model.p = param_manager.get_flat("p")
    ocp.model.p_global = param_manager.get_flat("p_global")

    return ocp


@pytest.fixture(scope="session")
def implicit_layer_with_stagewise_varying_params(
    acados_test_ocp_with_stagewise_varying_params: AcadosOcp,
) -> AcadosImplicitLayer:
    implicit_layer_ = AcadosImplicitLayer(
        ocp=acados_test_ocp_with_stagewise_varying_params,
        initializer=None,
        sensitivity_ocp=None,
        discount_factor=None,
    )

    # TODO: Setting the indicator variables for each stage. Do this via the param_manager
    for ocp_solver in chain(
        implicit_layer_.implicit_fun.forward_batch_solver.ocp_solvers,
        implicit_layer_.implicit_fun.backward_batch_solver.ocp_solvers,
    ):
        for stage_ in range(ocp_solver.acados_ocp.solver_options.N_horizon):
            idx_values_ = np.array(
                [
                    ocp_solver.acados_ocp.dims.np
                    - ocp_solver.acados_ocp.solver_options.N_horizon
                    + stage_
                ]
            )
            ocp_solver.set_params_sparse(
                stage_=stage_,
                idx_values_=idx_values_,
                param_values_=np.array([1.0]),
            )

        for stage_ in range(ocp_solver.acados_ocp.solver_options.N_horizon):
            print(f"stage: {stage_}; p: {ocp_solver.get(stage_=stage_, field_='p')}")

    return implicit_layer_


@pytest.fixture(scope="session")
def export_dir(tmp_path_factory: pytest.TempPathFactory) -> str:
    """Fixture to create a temporary directory for exporting files."""
    return str(tmp_path_factory.mktemp("export_dir"))


@pytest.fixture(scope="session")
def rng() -> np.random.Generator:
    """Fixture to provide a random number generator."""
    return np.random.default_rng(42)
