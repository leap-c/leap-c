# Test the following stuff:

# 1. batch solver is shared, forward solver and backward solver are the same (EXACT)
# 2. batch solver is not shared, forward solver and backward solver are different (GN, E)
# 3. sensitivities wrt p_global
# 4. ext_cost_p_global
# 5.
import casadi as ca
import numpy as np
import pytest
from acados_template import AcadosModel, AcadosOcp
from casadi.tools import entry, struct_symSX

# we need one version exact


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
        key: (model.p[key] if key in model.p.keys() else model.p_global[key])  # type:ignore
        for key in param_name
    }


def translate_learnable_param_to_p_global(
    nominal_param: dict[str, np.ndarray],
    learnable_param: list[str],
    ocp: AcadosOcp,
    verbose: bool = False,
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

    if verbose:
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


@pytest.fixture(scope="session")
def nominal_p_global() -> dict[str, np.ndarray]:
    """Nominal parameters for the AcadosOcp."""
    return {
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


@pytest.fixture(scope="session")
def learnable_p_global() -> list[str]:
    """Learnable parameters for the AcadosOcp."""
    return [
        "m",
        "cx",
        "cy",
        "q_diag",
        "r_diag",
        "q_diag_e",
        "xref",
        "uref",
        "xref_e",
    ]


@pytest.fixture(scope="session")
def x0() -> np.ndarray | None:
    """Define initial state for the AcadosOcp."""
    return None


def define_nonlinear_ls_cost():
    pass


def define_linear_ls_cost():
    pass


def define_external_cost():
    pass


def set_gn_solver_options(ocp: AcadosOcp) -> AcadosOcp:
    """Set the solver options for the OCP."""
    ocp.solver_options.integrator_type = "DISCRETE"
    ocp.solver_options.nlp_solver_type = "SQP"
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
    ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
    ocp.solver_options.qp_solver_ric_alg = 1
    ocp.solver_options.with_value_sens_wrt_params = True
    ocp.solver_options.with_solution_sens_wrt_params = True
    ocp.solver_options.with_batch_functionality = True

    return ocp


def set_solver_exact_options(ocp: AcadosOcp) -> AcadosOcp:
    """Configure the OCP solver options."""
    ocp.solver_options.integrator_type = "DISCRETE"
    ocp.solver_options.nlp_solver_type = "SQP"
    ocp.solver_options.hessian_approx = "EXACT"
    ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
    ocp.solver_options.qp_solver_ric_alg = 1
    ocp.solver_options.with_value_sens_wrt_params = True
    ocp.solver_options.with_solution_sens_wrt_params = True
    ocp.solver_options.with_batch_functionality = True

    return ocp


@pytest.fixture(scope="session")
def acados_test_ocp(
    x0: np.ndarray | None,
    nominal_p_global: dict[str, np.ndarray],
    learnable_p_global: list[str],
) -> AcadosOcp:
    """Define a simple AcadosOcp for testing purposes."""
    tf = 2.0
    N_horizon = 10
    name = "test_ocp"

    ocp = AcadosOcp()

    ocp.solver_options.tf = tf
    ocp.solver_options.N_horizon = N_horizon

    ocp.model.name = name

    ocp.dims.nu = 2
    ocp.dims.nx = 3

    ocp.model.x = ca.SX.sym("x", ocp.dims.nx)
    ocp.model.u = ca.SX.sym("u", ocp.dims.nu)

    ocp = translate_learnable_param_to_p_global(
        nominal_param=nominal_p_global,
        learnable_param=learnable_p_global,
        ocp=ocp,
    )

    ocp.model.disc_dyn_expr = get_disc_dyn_expr(ocp=ocp)
    ocp.model.cost_expr_ext_cost_0 = get_cost_expr_ext_cost(ocp=ocp)
    ocp.cost.cost_type_0 = "EXTERNAL"
    ocp.model.cost_expr_ext_cost = get_cost_expr_ext_cost(ocp=ocp)
    ocp.cost.cost_type = "EXTERNAL"
    ocp.model.cost_expr_ext_cost_e = get_cost_expr_ext_cost_e(ocp=ocp)
    ocp.cost.cost_type_e = "EXTERNAL"

    ocp.constraints.x0 = np.array([1.0, 1.0, 0.0, 0.0]) if x0 is None else x0

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
