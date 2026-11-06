import casadi as ca
import numpy as np
from acados_template import AcadosOcp

from leap_c.diff_mpc.utils.create_solver import _translate_cost_terms_to_external


def _create_ocp_with_external_stage_cost() -> AcadosOcp:
    """Minimal OCP with an external stage cost and no terminal cost set."""
    ocp = AcadosOcp()
    ocp.model.name = "translate_cost_test"
    ocp.model.x = ca.SX.sym("x", 1)
    ocp.model.u = ca.SX.sym("u", 1)
    ocp.model.disc_dyn_expr = ocp.model.x + ocp.model.u

    ocp.cost.cost_type = "EXTERNAL"
    ocp.model.cost_expr_ext_cost = ocp.model.x**2 + ocp.model.u**2

    ocp.solver_options.hessian_approx = "EXACT"
    return ocp


def test_translate_empty_terminal_ls_cost():
    """The default terminal cost (LINEAR_LS, ny_e = 0) becomes a zero external cost.

    acados' own translation of it fails with CasADi >= 3.8.
    """
    ocp = _create_ocp_with_external_stage_cost()

    _translate_cost_terms_to_external(ocp)

    assert ocp.cost.cost_type == "EXTERNAL"
    assert ocp.cost.cost_type_e == "EXTERNAL"
    assert ocp.model.cost_expr_ext_cost_e == 0


def test_translate_nonempty_terminal_ls_cost():
    ocp = _create_ocp_with_external_stage_cost()
    ocp.cost.cost_type_e = "LINEAR_LS"
    ocp.cost.Vx_e = np.eye(1)
    ocp.cost.W_e = np.eye(1)
    ocp.cost.yref_e = np.zeros(1)

    _translate_cost_terms_to_external(ocp)

    assert ocp.cost.cost_type_e == "EXTERNAL"
    assert ocp.model.cost_expr_ext_cost_e is not None
