from pathlib import Path
from unittest.mock import Mock

import numpy as np

from leap_c.ocp.acados.implicit import (
    AcadosImplicitCtx,
    AcadosImplicitFunction,
    SensitivityField,
)
from leap_c.ocp.acados.initializer import ZeroInitializer


def test_acados_implicit_ctx_initialization() -> None:
    """Test initialization of AcadosImplicitCtx."""
    iterate = None  # Mock or dummy object for AcadosOcpFlattenedBatchIterate
    status = np.array([0])
    log = {"cost": 0.0}
    solver_input = None  # Mock or dummy object for AcadosSolverInput

    ctx = AcadosImplicitCtx(
        iterate=iterate,
        status=status,
        log=log,
        solver_input=solver_input,
    )

    assert ctx.iterate == iterate
    assert np.array_equal(ctx.status, status)
    assert ctx.log == log
    assert ctx.solver_input == solver_input
    assert ctx.needs_input_grad is None
    assert ctx.du0_dp_global is None
    assert ctx.du0_dx0 is None
    assert ctx.dvalue_du is None
    assert ctx.dvalue_dx0 is None
    assert ctx.dx_dp_global is None
    assert ctx.du_dp_global is None


def test_sensitivity_field_literal() -> None:
    """Test SensitivityField literal values."""
    valid_fields = [
        "du0_dp_global",
        "dx_dp_global",
        "du_dp_global",
        "dvalue_dp_global",
        "dvalue_du0",
    ]

    for field in valid_fields:
        assert field in SensitivityField.__args__


def test_acados_implicit_function_initialization(acados_ocp) -> None:
    """Test initialization of AcadosImplicitFunction."""
    implicit_function = AcadosImplicitFunction(
        ocp=acados_ocp,
        initializer=None,
        sensitivity_ocp=None,
        discount_factor=None,
        export_directory="c_generated_code",
    )

    assert implicit_function.forward_batch_solver is not None
    assert implicit_function.backward_batch_solver is not None


def test_acados_implicit_function_forward_backward_sensitivity(
    acados_ocp,
    rng,
    n_batch: int = 5,
) -> None:
    """Test the forward method of AcadosImplicitFunction."""
    x0 = np.vstack(
        [
            rng.normal(
                loc=acados_ocp.constraints.x0, scale=0.1, size=acados_ocp.dims.nx
            )
            for _ in range(n_batch)
        ]
    )

    implicit_function = AcadosImplicitFunction(ocp=acados_ocp)
    ctx, sol_u0, sol_x, sol_u, sol_value = implicit_function.forward(x0=x0)

    assert isinstance(ctx, AcadosImplicitCtx)
    assert sol_u0 is not None
    assert sol_x is not None
    assert sol_u is not None
    assert sol_value is not None

    gradients = implicit_function.backward(
        ctx=ctx, u0_grad=None, x_grad=None, u_grad=None, value_grad=None
    )

    grad_u0 = gradients[2]
    grad_p_global = gradients[3]

    assert grad_u0 is not None
    assert grad_p_global is not None

    du0_dp_global = implicit_function.sensitivity(ctx, "du0_dp_global")

    assert du0_dp_global is not None
