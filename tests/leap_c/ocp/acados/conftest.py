from collections.abc import Callable
from itertools import chain

import casadi as ca
import numpy as np
import pytest
from acados_template import AcadosModel, AcadosOcp, AcadosOcpOptions
from casadi.tools import entry, struct_symSX

from leap_c.ocp.acados.torch import AcadosImplicitLayer


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
        return {key: model.p_global[key] for key in param_name}
    if model.p_global is None:
        return {key: model.p[key] for key in param_name}
    return {
        key: (model.p[key] if key in model.p.keys() else model.p_global[key])  # noqa: SIM118
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
def implicit_layer(acados_test_ocp: AcadosOcp) -> AcadosImplicitLayer:
    return AcadosImplicitLayer(
        ocp=acados_test_ocp,
        initializer=None,
        sensitivity_ocp=None,
        discount_factor=None,
    )


def is_stagewise_varying(param_key: str) -> bool:
    """
    Determine if a parameter is stage-wise varying based on its key pattern.

    Stage-wise varying parameters typically have keys in the format "label_stage_index",
    for example: "xref_0", "uref_5", "Q_2", etc.

    Args:
        param_key: The parameter key to check

    Returns:
        True if the parameter is stage-wise varying, False otherwise
    """
    # If there's no underscore, it's definitely not stage-wise
    if "_" not in param_key:
        return False

    # Split by underscore
    parts = param_key.split("_")

    # Check if the last part is a numeric stage index
    try:
        # Try to convert the last part to an integer
        int(parts[-1])

        # Make sure the base_name is not empty
        base_name = "_".join(parts[:-1])
        return bool(base_name)
    except ValueError:
        # The last part is not a numeric stage index
        return False


def categorize_parameters(
    nominal_params: dict[str, np.ndarray], nonlearnable_keys: set[str]
) -> dict[str, dict[str, np.ndarray]]:
    """Categorize parameters into learnable/nonlearnable and constant/varying."""
    learnable_keys = set(nominal_params.keys()) - nonlearnable_keys

    nonlearnable = {
        key: value for key, value in nominal_params.items() if key in nonlearnable_keys
    }

    learnable = {
        key: value for key, value in nominal_params.items() if key in learnable_keys
    }

    varying_learnable = {
        key: value for key, value in learnable.items() if is_stagewise_varying(key)
    }

    constant_learnable = {
        key: value for key, value in learnable.items() if key not in varying_learnable
    }

    return {
        "nonlearnable": nonlearnable,
        "learnable": {
            "constant": constant_learnable,
            "varying": varying_learnable,
        },
    }


def create_p_global_entries(
    params: dict[str, np.ndarray], N_horizon: int
) -> tuple[dict[str, list]]:
    """Create casadi struct entries for parameters."""
    labels = {
        "constant": list(params["constant"].keys()),
        "varying": list({"_".join(key.split("_")[:-1]) for key in params["varying"]}),
    }

    entries = []

    # Add constant parameter entries
    for label in labels["constant"]:
        param_shape = params["constant"][label].shape
        entries.append(entry(label, shape=param_shape))

    # Add varying parameter entries with shape consistency check
    for label in labels["varying"]:
        # Get all shapes for this parameter across stages
        shapes = {
            params["varying"][f"{label}_{stage}"].shape
            for stage in range(N_horizon)
            if f"{label}_{stage}" in params["varying"]
        }

        if not shapes:
            continue  # Skip if no matching parameters

        if len(shapes) > 1:
            msg = f"Parameter '{label}' has inconsistent shapes across stages: {shapes}"
            raise ValueError(msg)

        entries.append(entry(label, shape=shapes.pop(), repeat=N_horizon))

    return entries


def fill_p_global_values(
    params: dict[str, np.ndarray],
    p_global_values: struct_symSX,
) -> np.ndarray:
    """Fill parameter values in the CasADi structure."""
    # Fill constant values
    for key, value in params["constant"].items():
        p_global_values[key] = value

    # Fill varying values
    for key, value in params["varying"].items():
        label, stage = key.rsplit("_", 1)
        p_global_values[label, int(stage)] = value

    return p_global_values.cat.full().flatten()


def create_p_entries(
    params: dict[str, np.ndarray],
    N_horizon: int,
) -> tuple[dict[str, list], dict[str, list]]:
    """Create casadi struct entries for parameters."""
    entries = []

    # Add non-learnable parameter entries
    for key, value in params.items():
        entries.append(entry(key, shape=value.shape))

    # Add indicator entry
    entries.append(entry("indicator", shape=(1,), repeat=N_horizon))

    return entries


def fill_p_values(
    params: dict[str, np.ndarray],
    p_values: struct_symSX,
) -> np.ndarray:
    """Fill parameter values in the CasADi structure."""
    # Fill non-learnable values
    for key, value in params.items():
        p_values[key] = value

    return p_values.cat.full().flatten()


@pytest.fixture(scope="session")
def acados_test_ocp_with_stagewise_varying_params(
    ocp_options: AcadosOcpOptions,
) -> AcadosOcp:
    """Define a simple AcadosOcp for testing purposes."""
    name = "test_ocp_with_stagewise_varying_params"

    ocp = AcadosOcp()

    ocp.solver_options = ocp_options

    ocp.model.name = name

    ocp.dims.nu = 2
    ocp.dims.nx = 4

    ocp.model.x = ca.SX.sym("x", ocp.dims.nx)
    ocp.model.u = ca.SX.sym("u", ocp.dims.nu)

    nominal_params = {
        "m": np.array([1.0]),
        "cx": np.array([0.1]),
        "cy": np.array([0.1]),
        **{
            f"q_diag_{k}": np.array([1.0, 1.0, 1.0, 1.0])
            for k in range(ocp_options.N_horizon)
        },  # q_diag for initial stage and each intermediate stage
        "r_diag": np.array([0.1, 0.1]),
        "q_diag_e": np.array([1.0, 1.0, 1.0, 1.0]),
        **{
            f"xref_{k}": np.array([0.1, 0.2, 0.3, 0.4])
            for k in range(ocp_options.N_horizon)
        },  # xref for initial stage and each intermediate each stage
        **{
            f"uref_{k}": np.array([0.5, 0.6]) for k in range(ocp_options.N_horizon)
        },  # uref for initial stage and each intermediate each stage
        "xref_e": np.array([0.0, 0.0, 0.0, 0.0]),
    }

    # Categorize parameters
    nonlearnable_keys = {"cx", "cy"}

    params = categorize_parameters(
        nominal_params=nominal_params, nonlearnable_keys=nonlearnable_keys
    )

    ocp.model.p_global = struct_symSX(
        create_p_global_entries(
            params=params["learnable"],
            N_horizon=ocp.solver_options.N_horizon,
        )
    )

    ocp.p_global_values = fill_p_global_values(
        params=params["learnable"],
        p_global_values=ocp.model.p_global(0),
    )

    ocp.model.p = struct_symSX(
        create_p_entries(
            params=params["nonlearnable"],
            N_horizon=ocp.solver_options.N_horizon,
        )
    )

    ocp.parameter_values = fill_p_values(
        params=params["nonlearnable"],
        p_values=ocp.model.p(0),
    )

    # Define cost
    stage_cost = [
        ocp.model.p["indicator", stage]
        * get_cost_expr_ext_cost(
            ocp=ocp,
            xref=ocp.model.p_global["xref", stage],
            uref=ocp.model.p_global["uref", stage],
            q_diag=ocp.model.p_global["q_diag", stage],
        )
        for stage in range(ocp.solver_options.N_horizon)
    ]

    ocp.cost.cost_type_0 = "EXTERNAL"
    ocp.model.cost_expr_ext_cost_0 = stage_cost[0]
    ocp.cost.cost_type = "EXTERNAL"
    ocp.model.cost_expr_ext_cost = ca.sum1(ca.vertcat(*stage_cost))
    ocp.cost.cost_type_e = "EXTERNAL"
    ocp.model.cost_expr_ext_cost_e = get_cost_expr_ext_cost_e(ocp=ocp)

    ####

    ocp.model.disc_dyn_expr = get_disc_dyn_expr(ocp=ocp)

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
def implicit_layer_with_stagewise_varying_params(
    acados_test_ocp_with_stagewise_varying_params: AcadosOcp,
) -> AcadosImplicitLayer:
    implicit_layer_ = AcadosImplicitLayer(
        ocp=acados_test_ocp_with_stagewise_varying_params,
        initializer=None,
        sensitivity_ocp=None,
        discount_factor=None,
    )

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
