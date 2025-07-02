from itertools import chain

import casadi as ca
import numpy as np
import pytest
from acados_template import AcadosOcp, AcadosOcpOptions

from leap_c.ocp.acados.parameters import (
    AcadosParamManager,
    Parameter,
)
from leap_c.ocp.acados.torch import AcadosDiffMpc


@pytest.fixture(scope="session")
def nominal_params() -> tuple[Parameter, ...]:
    return (
        Parameter(
            name="m",
            value=np.array([1.0]),
            lower_bound=np.array([0.5]),
            upper_bound=np.array([1.5]),
            differentiable=False,
            stagewise=False,
            fix=False,
        ),
        Parameter(
            name="cx",
            value=np.array([0.1]),
            lower_bound=np.array([0.05]),
            upper_bound=np.array([0.15]),
            differentiable=False,
            stagewise=False,
            fix=False,
        ),
        Parameter(
            name="cy",
            value=np.array([0.1]),
            lower_bound=np.array([0.05]),
            upper_bound=np.array([0.15]),
            differentiable=False,
            stagewise=False,
            fix=False,
        ),
        Parameter(
            name="q_diag",
            value=np.array([1.0, 1.0, 1.0, 1.0]),
            lower_bound=np.array([0.5, 0.5, 0.5, 0.5]),
            upper_bound=np.array([1.5, 1.5, 1.5, 1.5]),
            differentiable=True,
            stagewise=False,
            fix=False,
        ),
        Parameter(
            name="r_diag",
            value=np.array([0.1, 0.1]),
            lower_bound=np.array([0.05, 0.05]),
            upper_bound=np.array([0.15, 0.15]),
            differentiable=True,
            stagewise=False,
            fix=False,
        ),
        Parameter(
            name="q_diag_e",
            value=np.array([1.0, 1.0, 1.0, 1.0]),
            lower_bound=np.array([0.5, 0.5, 0.5, 0.5]),
            upper_bound=np.array([1.5, 1.5, 1.5, 1.5]),
            differentiable=True,
            stagewise=False,
            fix=False,
        ),
        Parameter(
            name="xref",
            value=np.array([0.0, 0.0, 0.0, 0.0]),
            lower_bound=np.array([-1.0, -1.0, -1.0, -1.0]),
            upper_bound=np.array([1.0, 1.0, 1.0, 1.0]),
            differentiable=True,
            stagewise=False,
            fix=False,
        ),
        Parameter(
            name="uref",
            value=np.array([0.0, 0.0]),
            lower_bound=np.array([-1.0, -1.0]),
            upper_bound=np.array([1.0, 1.0]),
            differentiable=True,
            stagewise=False,
            fix=False,
        ),
        Parameter(
            name="xref_e",
            value=np.array([0.0, 0.0, 0.0, 0.0]),
            lower_bound=np.array([-1.0, -1.0, -1.0, -1.0]),
            upper_bound=np.array([1.0, 1.0, 1.0, 1.0]),
            differentiable=True,
            stagewise=False,
            fix=False,
        ),
    )


@pytest.fixture(scope="session")
def nominal_stagewise_params(
    nominal_params: tuple[Parameter, ...],
) -> tuple[Parameter, ...]:
    """Copy nominal_params and modify specific parameters to be stagewise."""
    # Override specific fields for stage-wise parameters
    stagewise_overrides = {
        "q_diag": {"stagewise": True},
        "xref": {"stagewise": True},
        "uref": {"stagewise": True},
    }

    modified_params = []
    for param in nominal_params:
        if param.name in stagewise_overrides:
            # Create new parameter with overridden fields
            kwargs = param._asdict()
            kwargs.update(stagewise_overrides[param.name])
            modified_params.append(Parameter(**kwargs))
        else:
            modified_params.append(param)

    return tuple(modified_params)


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
def acados_test_ocp_no_p_global(
    ocp_options: AcadosOcpOptions,
    nominal_params: tuple[Parameter, ...],
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

    # Make a copy of the nominal parameters where differentiable and stagewise is set to False everywhere
    params = tuple(
        Parameter(
            name=param.name,
            value=param.value,
            lower_bound=param.lower_bound,
            upper_bound=param.upper_bound,
            fix=True,
            differentiable=False,
            stagewise=False,
        )
        for param in nominal_params
    )

    param_manager = AcadosParamManager(
        params=params, N_horizon=ocp.solver_options.N_horizon
    )

    ocp.model.name = name

    ocp.model.x = ca.SX.sym("x", 4)
    ocp.model.u = ca.SX.sym("u", 2)

    ocp.dims.nx = ocp.model.x.shape[0]
    ocp.dims.nu = ocp.model.u.shape[0]

    kwargs = {
        "m": param_manager.get(field="m"),
        "cx": param_manager.get(field="cx"),
        "cy": param_manager.get(field="cy"),
        "dt": ocp.solver_options.tf / ocp.solver_options.N_horizon,
    }
    # Make sure all entries are floats or casadi SX
    # TODO: Move this into the AcadosParamManager
    for key, value in kwargs.items():
        if isinstance(value, np.ndarray):
            kwargs[key] = value.item() if value.size == 1 else ca.SX(value)

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
            param_manager.parameter_values["xref"].full().flatten(),
            param_manager.parameter_values["uref"].full().flatten(),
        ]
    )

    ocp.cost.W_0 = np.diag(
        np.concatenate(
            [
                param_manager.parameter_values["q_diag"].full().flatten(),
                param_manager.parameter_values["r_diag"].full().flatten(),
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
    ocp.cost.yref_e = param_manager.parameter_values["xref_e"].full().flatten()
    ocp.cost.W_e = np.diag(param_manager.parameter_values["q_diag_e"].full().flatten())
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


def define_external_cost(ocp: AcadosOcp, param_manager: AcadosParamManager):
    y = ca.vertcat(
        ocp.model.x,
        ocp.model.u,
    )
    yref = ca.vertcat(
        param_manager.get(field="xref"),
        param_manager.get(field="uref"),
    )
    W_sqrt = ca.diag(
        ca.vertcat(
            param_manager.get(field="q_diag"),
            param_manager.get(field="r_diag"),
        )
    )
    xref_e = param_manager.get(field="xref_e")
    Q_sqrt_e = ca.diag(param_manager.get(field="q_diag_e"))

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


def define_discrete_dynamics(ocp: AcadosOcp, param_manager: AcadosParamManager) -> None:
    kwargs = {
        "m": param_manager.get(field="m"),
        "cx": param_manager.get(field="cx"),
        "cy": param_manager.get(field="cy"),
        "dt": ocp.solver_options.tf / ocp.solver_options.N_horizon,
    }
    ocp.model.disc_dyn_expr = (
        get_A_disc(**kwargs) @ ocp.model.x + get_B_disc(**kwargs) @ ocp.model.u
    )


def define_constraints(ocp: AcadosOcp, param_manager: AcadosParamManager) -> None:
    """Define constraints for the OCP."""
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


@pytest.fixture(scope="session", params=["external", "nonlinear_ls"])
def acados_test_ocp(
    ocp_options: AcadosOcpOptions,
    nominal_params: tuple[Parameter, ...],
    request: pytest.FixtureRequest,
) -> AcadosOcp:
    """Define a simple AcadosOcp for testing purposes."""
    name = "test_ocp"

    ocp = AcadosOcp()

    ocp.solver_options = ocp_options

    param_manager = AcadosParamManager(
        params=nominal_params, N_horizon=ocp.solver_options.N_horizon
    )
    param_manager.assign_to_ocp(ocp)

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

    return ocp


# TODO: Remove this fixture once the nominal_stagewise_params can be used in acados_test_ocp
@pytest.fixture(scope="session")
def acados_test_ocp_with_stagewise_varying_params(
    ocp_options: AcadosOcpOptions,
    nominal_stagewise_params: tuple[Parameter, ...],
) -> AcadosOcp:
    """Define a simple AcadosOcp for testing purposes."""
    name = "test_ocp_with_stagewise_varying_params"

    ocp = AcadosOcp()

    ocp.solver_options = ocp_options

    param_manager = AcadosParamManager(
        params=nominal_stagewise_params, N_horizon=ocp.solver_options.N_horizon
    )
    param_manager.assign_to_ocp(ocp)

    ocp.model.name = name

    ocp.model.x = ca.SX.sym("x", 4)
    ocp.model.u = ca.SX.sym("u", 2)

    define_external_cost(ocp, param_manager)
    define_discrete_dynamics(ocp, param_manager)
    define_constraints(ocp, param_manager)

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
def diff_mpc_with_stagewise_varying_params(
    acados_test_ocp_with_stagewise_varying_params: AcadosOcp,
    nominal_stagewise_params: tuple[Parameter, ...],
    print_level: int = 0,
) -> AcadosDiffMpc:
    diff_mpc = AcadosDiffMpc(
        ocp=acados_test_ocp_with_stagewise_varying_params,
        initializer=None,
        sensitivity_ocp=None,
        discount_factor=None,
    )

    acados_param_manager = AcadosParamManager(
        params=nominal_stagewise_params,
        N_horizon=acados_test_ocp_with_stagewise_varying_params.solver_options.N_horizon,
    )

    # Get the default parameter values for each stage
    parameter_values = acados_param_manager.combine_parameter_values()

    for ocp_solver in chain(
        diff_mpc.diff_mpc_fun.forward_batch_solver.ocp_solvers,
        diff_mpc.diff_mpc_fun.backward_batch_solver.ocp_solvers,
    ):
        for batch in range(parameter_values.shape[0]):
            for stage in range(parameter_values.shape[1]):
                ocp_solver.set(stage, "p", parameter_values[batch, stage, :])

                if print_level > 0:
                    print(
                        f"stage: {stage}; p: {ocp_solver.get(stage_=stage, field_='p')}"
                    )

    return diff_mpc


@pytest.fixture(scope="session")
def export_dir(tmp_path_factory: pytest.TempPathFactory) -> str:
    """Fixture to create a temporary directory for exporting files."""
    return str(tmp_path_factory.mktemp("export_dir"))


@pytest.fixture(scope="session")
def rng() -> np.random.Generator:
    """Fixture to provide a random number generator."""
    return np.random.default_rng(42)
