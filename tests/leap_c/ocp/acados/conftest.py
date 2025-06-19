from collections.abc import Callable

import casadi as ca
import numpy as np
import pytest
from acados_template import AcadosModel, AcadosOcp, AcadosOcpOptions
from casadi.tools import entry, struct_symSX

from leap_c.ocp.acados.torch import AcadosDiffMpc


def _process_params(
    params: list[str], nominal_param: dict[str, np.ndarray]
) -> tuple[list, list]:
    entries = []
    values = []
    for param in params:
        try:
            entries.append(entry(param, shape=nominal_param[param].shape))
            values.append(nominal_param[param].T.reshape(-1, 1))
        except AttributeError:
            entries.append(entry(param, shape=(1, 1)))
            values.append(np.array([nominal_param[param]]).reshape(-1, 1))
    return entries, values


def find_param_in_p_or_p_global(
    param_name: list[str], model: AcadosModel
) -> dict[str, ca.SX]:
    if model.p == []:
        return {key: model.p_global[key] for key in param_name}  # type:ignore
    if model.p_global is None:
        return {key: model.p[key] for key in param_name}  # type:ignore
    return {
        key: (model.p[key] if key in model.p.keys() else model.p_global[key])  # type:ignore  # noqa: SIM118
        for key in param_name
    }


def translate_learnable_param_to_p_global(
    nominal_param: dict[str, np.ndarray],
    learnable_param: list[str],
    ocp: AcadosOcp,
    verbosity: int = 0,
) -> AcadosOcp:
    if learnable_param:
        entries, values = _process_params(learnable_param, nominal_param)
        ocp.model.p_global = struct_symSX(entries)
        ocp.p_global_values = np.concatenate(values).flatten()

    non_learnable_params = [key for key in nominal_param if key not in learnable_param]
    if non_learnable_params:
        entries, values = _process_params(non_learnable_params, nominal_param)
        ocp.model.p = struct_symSX(entries)
        ocp.parameter_values = np.concatenate(values).flatten()

    if verbosity:
        print("learnable_params", learnable_param)
        print("non_learnable_params", non_learnable_params)
    return ocp


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
        )  # type: ignore

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
        )  # type: ignore

    return np.array(
        [
            [0, 0],
            [0, 0],
            [(m / cx) * (1 - np.exp(-cx * dt / m)), 0],
            [0, (m / cy) * (1 - np.exp(-cy * dt / m))],
        ]
    )


def get_disc_dyn_expr(
    ocp: AcadosOcp,
) -> ca.SX:
    x = ocp.model.x
    u = ocp.model.u

    m = find_param_in_p_or_p_global(["m"], ocp.model)["m"]
    cx = find_param_in_p_or_p_global(["cx"], ocp.model)["cx"]
    cy = find_param_in_p_or_p_global(["cy"], ocp.model)["cy"]
    dt = ocp.solver_options.tf / ocp.solver_options.N_horizon

    A = get_A_disc(m=m, cx=cx, cy=cy, dt=dt)
    B = get_B_disc(m=m, cx=cx, cy=cy, dt=dt)

    return A @ x + B @ u


def _create_diag_matrix(
    _q_sqrt: np.ndarray | ca.SX,
) -> np.ndarray | ca.SX:
    if any(isinstance(i, ca.SX) for i in [_q_sqrt]):
        return ca.diag(_q_sqrt)
    return np.diag(_q_sqrt)


def get_cost_expr_ext_cost(ocp: AcadosOcp) -> ca.SX:
    x = ocp.model.x
    u = ocp.model.u

    Q_sqrt = _create_diag_matrix(
        find_param_in_p_or_p_global(["q_diag"], ocp.model)["q_diag"]
    )
    R_sqrt = _create_diag_matrix(
        find_param_in_p_or_p_global(["r_diag"], ocp.model)["r_diag"]
    )

    xref = find_param_in_p_or_p_global(["xref"], ocp.model)["xref"]
    uref = find_param_in_p_or_p_global(["uref"], ocp.model)["uref"]

    return 0.5 * (
        ca.mtimes([ca.transpose(x - xref), Q_sqrt.T, Q_sqrt, x - xref])
        + ca.mtimes([ca.transpose(u - uref), R_sqrt.T, R_sqrt, u - uref])
    )


def get_cost_expr_ext_cost_e(ocp: AcadosOcp) -> ca.SX:
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
def acados_test_ocp(ocp_cost_fun: Callable, ocp_options: AcadosOcpOptions) -> AcadosOcp:
    """Define a simple AcadosOcp for testing purposes."""
    nominal_p_global = {
        "m": 1.0,
        "cx": 0.1,
        "cy": 0.1,
        "q_diag": np.array([1.0, 1.0, 1.0, 1.0]),
        "r_diag": np.array([0.1, 0.1]),
        "q_diag_e": np.array([1.0, 1.0, 1.0, 1.0]),
        "xref": np.array([0.0, 0.0, 0.0, 0.0]),
        "uref": np.array([0.0, 0.0]),
        "xref_e": np.array([0.0, 0.0, 0.0, 0.0]),
    }

    learnable_p_global = nominal_p_global.keys()

    # Remove from learnable parameters to test non-learnable parameters
    learnable_p_global = [p for p in learnable_p_global if p not in ["m", "cx", "cy"]]

    name = "test_ocp"

    ocp = AcadosOcp()

    ocp.solver_options = ocp_options

    ocp.model.name = name

    ocp.dims.nu = 2
    ocp.dims.nx = 4

    ocp.model.x = ca.SX.sym("x", ocp.dims.nx)
    ocp.model.u = ca.SX.sym("u", ocp.dims.nu)

    ocp = translate_learnable_param_to_p_global(
        nominal_param=nominal_p_global,
        learnable_param=learnable_p_global,
        ocp=ocp,
        verbosity=1,
    )

    ocp.model.disc_dyn_expr = get_disc_dyn_expr(ocp=ocp)

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
    if isinstance(ocp.model.p, struct_symSX):
        ocp.model.p = ocp.model.p.cat if ocp.model.p is not None else []

    if isinstance(ocp.model.p_global, struct_symSX):
        ocp.model.p_global = (
            ocp.model.p_global.cat if ocp.model.p_global is not None else None
        )

    return ocp


@pytest.fixture(scope="session")
def diff_mpc(acados_test_ocp: AcadosOcp) -> AcadosDiffMpc:
    return AcadosDiffMpc(
        ocp=acados_test_ocp,
        initializer=None,
        sensitivity_ocp=None,
        discount_factor=None,
    )


@pytest.fixture(scope="session")
def rng() -> np.random.Generator:
    """Fixture to provide a random number generator."""
    return np.random.default_rng(42)
