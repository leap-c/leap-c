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


# @pytest.fixture(scope="session")
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

    param_manager = AcadosParamManager(params=nominal_params, ocp=ocp)

    ocp.model.name = name

    ocp.model.x = ca.SX.sym("x", 4)
    ocp.model.u = ca.SX.sym("u", 2)

    kwargs = {
        "m": param_manager.get(field_="m"),
        "cx": param_manager.get(field_="cx"),
        "cy": param_manager.get(field_="cy"),
        "dt": ocp.solver_options.tf / ocp.solver_options.N_horizon,
    }
    ocp.model.disc_dyn_expr = (
        get_A_disc(**kwargs) @ ocp.model.x + get_B_disc(**kwargs) @ ocp.model.u
    )

    # Define cost
    if request.param == "external":
        xref = param_manager.get(field_="xref")
        xref_e = param_manager.get(field_="xref_e")
        uref = param_manager.get(field_="uref")
        R_sqrt = ca.diag(param_manager.get(field_="r_diag"))
        Q_sqrt = ca.diag(param_manager.get(field_="q_diag"))
        Q_sqrt_e = ca.diag(param_manager.get(field_="q_diag_e"))

        # Initial stage costs
        ocp.cost.cost_type_0 = "EXTERNAL"
        ocp.model.cost_expr_ext_cost_0 = 0.5 * (
            ca.mtimes(
                [
                    ca.transpose(ocp.model.x - xref),
                    Q_sqrt,
                    ca.transpose(Q_sqrt),
                    ocp.model.x - xref,
                ]
            )
            + ca.mtimes(
                [
                    ca.transpose(ocp.model.u - uref),
                    R_sqrt,
                    ca.transpose(R_sqrt),
                    ocp.model.u - uref,
                ]
            )
        )

        # Intermediate stage costs
        ocp.cost.cost_type = ocp.cost.cost_type_0
        ocp.model.cost_expr_ext_cost = ocp.model.cost_expr_ext_cost_0
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

    if request.param == "nonlinear_ls":
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


@pytest.fixture(scope="session")
def diff_mpc(acados_test_ocp: AcadosOcp) -> AcadosDiffMpc:
    return AcadosDiffMpc(
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
            value=np.array([0.0, 0.0, 0.0, 0.0]),
            lower_bound=np.array([-1.0, -1.0, -1.0, -1.0]),
            upper_bound=np.array([1.0, 1.0, 1.0, 1.0]),
            differentiable=True,
            varying=True,
        ),
        Parameter(
            name="uref",
            value=np.array([0.0, 0.0]),
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
def nominal_varying_params_for_param_manager_tests() -> tuple[Parameter, ...]:
    return (
        Parameter(
            name="m",
            value=np.array([1.0]),
            lower_bound=np.array([0.5]),
            upper_bound=np.array([1.5]),
            differentiable=True,
            varying=True,
        ),
        Parameter(
            name="cx",
            value=np.array([0.1]),
            lower_bound=np.array([0.05]),
            upper_bound=np.array([0.15]),
            differentiable=True,
            varying=True,
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
            differentiable=False,
            varying=False,
        ),
        Parameter(
            name="r_diag",
            value=np.array([0.1, 0.1]),
            lower_bound=np.array([0.05, 0.05]),
            upper_bound=np.array([0.15, 0.15]),
            differentiable=False,
            varying=False,
        ),
        Parameter(
            name="q_diag_e",
            value=np.array([1.0, 1.0, 1.0, 1.0]),
            lower_bound=np.array([0.5, 0.5, 0.5, 0.5]),
            upper_bound=np.array([1.5, 1.5, 1.5, 1.5]),
            differentiable=False,
            varying=False,
        ),
        Parameter(
            name="xref",
            value=np.array([0.1, 0.2, 0.3, 0.4]),
            lower_bound=np.array([-1.0, -1.0, -1.0, -1.0]),
            upper_bound=np.array([1.0, 1.0, 1.0, 1.0]),
            differentiable=False,
            varying=False,
        ),
        Parameter(
            name="uref",
            value=np.array([0.5, 0.6]),
            lower_bound=np.array([-1.0, -1.0]),
            upper_bound=np.array([1.0, 1.0]),
            differentiable=False,
            varying=False,
        ),
        Parameter(
            name="xref_e",
            value=np.array([0.0, 0.0, 0.0, 0.0]),
            lower_bound=np.array([-1.0, -1.0, -1.0, -1.0]),
            upper_bound=np.array([1.0, 1.0, 1.0, 1.0]),
            differentiable=True,
            varying=True,
        ),
    )


@pytest.fixture(scope="session")
def acados_param_manager(
    nominal_varying_params: tuple[Parameter, ...],
) -> AcadosParamManager:
    ocp = AcadosOcp()
    ocp.solver_options.N_horizon = 30
    return AcadosParamManager(params=nominal_varying_params, ocp=ocp)


@pytest.fixture(scope="session")
def acados_test_ocp_with_stagewise_varying_params(
    ocp_options: AcadosOcpOptions,
    nominal_varying_params: tuple[Parameter, ...],
) -> AcadosOcp:
    """Define a simple AcadosOcp for testing purposes."""
    name = "test_ocp_with_stagewise_varying_params"

    ocp = AcadosOcp()

    ocp.solver_options = ocp_options

    param_manager = AcadosParamManager(params=nominal_varying_params, ocp=ocp)

    ocp.model.name = name

    ocp.model.x = ca.SX.sym("x", 4)
    ocp.model.u = ca.SX.sym("u", 2)

    stage_cost = []
    for stage in range(ocp.solver_options.N_horizon):
        y = ca.vertcat(
            ocp.model.x,
            ocp.model.u,
        )
        yref = ca.vertcat(
            param_manager.get(field_="xref", stage_=stage),
            param_manager.get(field_="uref", stage_=stage),
        )
        W_sqrt = ca.diag(
            ca.vertcat(
                param_manager.get(field_="q_diag", stage_=stage),
                param_manager.get(field_="r_diag"),
            )
        )

        stage_cost.append(
            param_manager.get(field_="indicator", stage_=stage)
            * 0.5
            * (
                ca.mtimes(
                    [ca.transpose(y - yref), W_sqrt, ca.transpose(W_sqrt), y - yref]
                )
            )
        )

    ocp.cost.cost_type_0 = "EXTERNAL"
    ocp.model.cost_expr_ext_cost_0 = stage_cost[0]
    ocp.cost.cost_type = "EXTERNAL"
    ocp.model.cost_expr_ext_cost = ca.sum1(ca.vertcat(*stage_cost[1:]))
    ocp.cost.cost_type_e = "EXTERNAL"
    xref_e = param_manager.get(field_="xref_e")
    q_diag_e = param_manager.get(field_="q_diag_e")
    Q_sqrt_e = ca.diag(q_diag_e)

    ocp.model.cost_expr_ext_cost_e = 0.5 * ca.mtimes(
        [
            ca.transpose(ocp.model.x - xref_e),
            Q_sqrt_e,
            ca.transpose(Q_sqrt_e),
            ocp.model.x - xref_e,
        ]
    )

    ####
    kwargs = {
        "m": param_manager.get(field_="m"),
        "cx": param_manager.get(field_="cx"),
        "cy": param_manager.get(field_="cy"),
        "dt": ocp.solver_options.tf / ocp.solver_options.N_horizon,
    }
    ocp.model.disc_dyn_expr = (
        get_A_disc(**kwargs) @ ocp.model.x + get_B_disc(**kwargs) @ ocp.model.u
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

    return ocp


@pytest.fixture(scope="session")
def diff_mpc_with_stagewise_varying_params(
    acados_test_ocp_with_stagewise_varying_params: AcadosOcp,
    nominal_varying_params: tuple[Parameter, ...],
    print_level: int = 0,
) -> AcadosDiffMpc:
    diff_mpc = AcadosDiffMpc(
        ocp=acados_test_ocp_with_stagewise_varying_params,
        initializer=None,
        sensitivity_ocp=None,
        discount_factor=None,
    )

    acados_param_manager = AcadosParamManager(
        params=nominal_varying_params,
        ocp=diff_mpc.diff_mpc_fun.forward_batch_solver.ocp_solvers[0].acados_ocp,
    )

    # Get the default parameter values for each stage
    p_global_values = acados_param_manager.p_global_values
    parameter_values = acados_param_manager.combine_parameter_values()

    for ocp_solver in chain(
        diff_mpc.diff_mpc_fun.forward_batch_solver.ocp_solvers,
        diff_mpc.diff_mpc_fun.backward_batch_solver.ocp_solvers,
    ):
        for batch in range(parameter_values.shape[0]):
            for stage in range(parameter_values.shape[1]):
                ocp_solver.set(stage, "p", parameter_values[batch, stage, :])

    if print_level > 0:
        print("Parameter values for each stage:")
        for stage_ in range(ocp_solver.acados_ocp.solver_options.N_horizon):
            print(f"stage: {stage_}; p: {ocp_solver.get(stage_=stage_, field_='p')}")

    return diff_mpc


@pytest.fixture(scope="session")
def export_dir(tmp_path_factory: pytest.TempPathFactory) -> str:
    """Fixture to create a temporary directory for exporting files."""
    return str(tmp_path_factory.mktemp("export_dir"))


@pytest.fixture(scope="session")
def rng() -> np.random.Generator:
    """Fixture to provide a random number generator."""
    return np.random.default_rng(42)
