from pathlib import Path

import numpy as np

from leap_c.ocp.acados.implicit import (
    AcadosImplicitCtx,
    AcadosImplicitFunction,
)
import torch

from leap_c.ocp.acados.torch import AcadosImplicitLayer

from leap_c.ocp.acados.initializer import ZeroInitializer

import conftest

from typing import Tuple, Callable
from dataclasses import dataclass

from acados_template import AcadosOcp


def test_initialization(implicit_layer):
    assert True


def test_file_management(implicit_layer, export_dir):
    pass


def test_ctx_loading(implicit_layer: AcadosImplicitLayer, export_dir):
    pass


def test_translations(test_ocp, export_dir):
    # Test the translations of the cost between [external, linear_ls, nonlinear_ls] to external
    pass


def test_statelessness(test_ocp):
    # See current MPC implementation. Needs rewrite.
    pass


def test_backup_functionality(test_ocp):
    # See current MPC implementation. Needs rewrite.
    pass


def test_closed_loop(acados_test_implicit_function):
    # Test the acados_example_ocp in closed loop. Do we need a reference fixture?
    # x0 = np.array([0.5, 0.5, 0.5, 0.5])
    # x = [x0]
    # u = []

    # p_global = acados_test_impl_fun.ocp.p_global_values

    # for step in range(100):
    #     u_star, _, status = learnable_linear_mpc.policy(x[-1], p_global=p_global)
    #     assert status == 0, f"Did not converge to a solution in step {step}"
    #     u.append(u_star)
    #     x.append(learnable_linear_mpc.ocp_batch_solver.ocp_solvers[0].get(1, "x"))
    #     assert status == 0

    # x = np.array(x)
    # u = np.array(u)

    # assert (
    #     np.median(x[-10:, 0]) <= 1e-1
    #     and np.median(x[-10:, 1]) <= 1e-1
    #     and np.median(u[-10:]) <= 1e-1
    # )
    pass


@dataclass
class GradCheckConfig:
    """Configuration for gradient checking parameters."""

    atol: float = 1e-2
    eps: float = 1e-4
    raise_exception: bool = True


@dataclass
class TestData:
    """Container for test data tensors."""

    x0: torch.Tensor
    u0: torch.Tensor
    p_global: torch.Tensor


def _setup_test_data(
    implicit_layer: AcadosImplicitLayer,
    n_batch: int,
    dtype: torch.dtype,
    noise_scale: float,
) -> TestData:
    """Set up test data tensors with proper gradients enabled."""
    ocp = implicit_layer.implicit_fun.ocp

    # Create a seeded generator
    generator = torch.Generator()
    generator.manual_seed(42)

    # Generate noisy global parameters
    loc = torch.tensor(ocp.p_global_values, dtype=dtype).unsqueeze(0).repeat(n_batch, 1)
    scale = noise_scale * loc
    p_global = torch.normal(mean=loc, std=scale, generator=generator).requires_grad_()

    # Setup initial state
    loc = torch.tensor(ocp.constraints.x0, dtype=dtype).unsqueeze(0).repeat(n_batch, 1)
    scale = noise_scale * loc
    x0_batch = torch.normal(mean=loc, std=scale, generator=generator).requires_grad_()

    # Setup initial control
    loc = (
        torch.tensor(np.zeros(ocp.dims.nu), dtype=dtype).unsqueeze(0).repeat(n_batch, 1)
    )
    scale = noise_scale
    u0_batch = torch.normal(mean=loc, std=scale, generator=generator).requires_grad_()

    assert x0_batch.shape == (n_batch, ocp.dims.nx), (
        f"x0 shape mismatch. Expected: {(n_batch, ocp.dims.nx)}, Got: {x0_batch.shape}"
    )
    assert u0_batch.shape == (n_batch, ocp.dims.nu), (
        f"u0 shape mismatch. Expected: {(n_batch, ocp.dims.nu)}, Got: {u0_batch.shape}"
    )
    assert p_global.shape == (n_batch, ocp.dims.np_global), (
        f"p_global shape mismatch. Expected: {(n_batch, ocp.dims.np_global)}, \
        Got: {p_global.shape}"
    )
    assert x0_batch.requires_grad, "x0_batch should require gradients"
    assert u0_batch.requires_grad, "u0_batch should require gradients"
    assert p_global.requires_grad, "p_global should require gradients"

    return TestData(x0=x0_batch, u0=u0_batch, p_global=p_global)


def test_forward(
    implicit_layer: AcadosImplicitLayer,
    n_batch: int = 4,
    dtype: torch.dtype = torch.float64,
    noise_scale: float = 0.05,
    verbosity: int = 0,
) -> None:
    """
    Test the forward method of AcadosImplicitFunction with different input combinations.

    Args:
        implicit_layer: The implicit layer to test
        n_batch: Number of batch samples
        dtype: PyTorch data type for tensors
        noise_scale: Scale factor for noise added to parameters
        test_all_variants: If True, test all forward call variants
        verbosity: Level of verbosity for test output
    """

    def _run_single_forward_test(
        implicit_layer: AcadosImplicitLayer,
        forward_kwargs: dict[str, torch.Tensor],
        expected_output_type: str,
        n_batch: int,
        acados_ocp: AcadosOcp,
    ) -> None:
        """Run a single forward test with given parameters."""
        # Call forward method
        ctx, u0, x, u, value = implicit_layer.forward(**forward_kwargs)

        # Validate solver status
        assert np.all(ctx.status == 0), (
            f"Forward method failed with status {ctx.status}"
        )

        # Validate output types
        assert isinstance(ctx, AcadosImplicitCtx), (
            "ctx should be an instance of AcadosImplicitCtx"
        )
        assert isinstance(u0, torch.Tensor), "u0 should be a torch.Tensor"
        assert isinstance(x, torch.Tensor), "x should be a torch.Tensor"
        assert isinstance(u, torch.Tensor), "u should be a torch.Tensor"
        assert isinstance(value, torch.Tensor), "value should be a torch.Tensor"

        expected_u0_shape = (n_batch, acados_ocp.dims.nu)
        assert u0.shape == expected_u0_shape, (
            f"u0 shape mismatch. Expected: {expected_u0_shape}, Got: {u0.shape}"
        )

        expected_x_shape = (
            n_batch,
            acados_ocp.dims.nx * (acados_ocp.solver_options.N_horizon + 1),
        )
        assert x.shape == expected_x_shape, (
            f"x shape mismatch. Expected: {expected_x_shape}, Got: {x.shape}"
        )

        expected_u_shape = (
            n_batch,
            acados_ocp.dims.nu * acados_ocp.solver_options.N_horizon,
        )
        assert u.shape == expected_u_shape, (
            f"u shape mismatch. Expected: {expected_u_shape}, Got: {u.shape}"
        )

        # Validate value shape (same for both V and Q)
        expected_value_shape = (n_batch,)
        assert value.shape == expected_value_shape, (
            f"{expected_output_type} shape mismatch. Expected: {expected_value_shape}, Got: {value.shape}"
        )

    acados_ocp = implicit_layer.implicit_fun.ocp
    n_batch = implicit_layer.implicit_fun.forward_batch_solver.N_batch_max

    # Setup test data
    test_data = _setup_test_data(implicit_layer, n_batch, dtype, noise_scale)

    # Define test cases
    test_cases = [
        {
            "name": "x0_only",
            "kwargs": {"x0": test_data.x0},
            "expected_output": "V",
        },
        {
            "name": "x0_and_u0",
            "kwargs": {"x0": test_data.x0, "u0": test_data.u0},
            "expected_output": "Q",
        },
        {
            "name": "x0_and_p_global",
            "kwargs": {"x0": test_data.x0, "p_global": test_data.p_global},
            "expected_output": "V",
        },
        {
            "name": "all_parameters",
            "kwargs": {
                "x0": test_data.x0,
                "u0": test_data.u0,
                "p_global": test_data.p_global,
            },
            "expected_output": "Q",
        },
    ]

    for test_case in test_cases:
        if verbosity > 0:
            print(f"Testing forward call: {test_case['name']}")
        _run_single_forward_test(
            implicit_layer,
            test_case["kwargs"],
            test_case["expected_output"],
            n_batch,
            acados_ocp,
        )


def test_sensitivity(
    implicit_layer: AcadosImplicitLayer,
    n_batch: int = 4,
    max_batch_size: int = 10,
    dtype: torch.dtype = torch.float64,
    noise_scale: float = 0.1,
) -> None:
    """
    Test sensitivity of AcadosImplicitLayer to changes in parameters.

    Args:
        implicit_layer: The implicit layer to test
        n_batch: Number of batch samples to generate
        max_batch_size: Maximum allowed batch size for performance
        dtype: PyTorch data type for tensors
        noise_scale: Scale factor for noise added to parameters
    """
    # Validate batch size
    if n_batch > max_batch_size:
        raise ValueError(
            f"Batch size {n_batch} exceeds maximum {max_batch_size}. "
            "Large batch sizes make the test very slow."
        )

    # Setup test data
    test_data = _setup_test_data(implicit_layer, n_batch, dtype, noise_scale)

    ctx, u0, x, u, value = implicit_layer.forward(x0=test_data.x0)

    field_list = ["dvalue_dp_global", "dvalue_dx0", "dvalue_du0"]

    results = {
        field: implicit_layer.sensitivity(ctx=ctx, field_name=field)
        for field in field_list
    }

    assert results["dvalue_dp_global"].shape == (
        n_batch,
        implicit_layer.ocp.dims.np_global,
    ), (
        f"dvalue_dp_global shape mismatch. Expected: \
            {(n_batch, implicit_layer.ocp.dims.np_global)}, "
        f"Got: {results['dvalue_dp_global'].shape}"
    )

    assert results["dvalue_dx0"].shape == (
        n_batch,
        implicit_layer.ocp.dims.nx,
    ), (
        f"dvalue_dx0 shape mismatch. Expected: {(n_batch, implicit_layer.ocp.dims.nx)},"
        f" "
        f"Got: {results['dvalue_dx0'].shape}"
    )

    assert results["dvalue_du0"].shape == (
        n_batch,
        implicit_layer.ocp.dims.nu,
    ), (
        f"dvalue_du0 shape mismatch. Expected: {(n_batch, implicit_layer.ocp.dims.nu)},"
        f" "
        f"Got: {results['dvalue_du0'].shape}"
    )


def test_backward(
    implicit_layer: AcadosImplicitLayer,
    n_batch: int = 4,
    max_batch_size: int = 10,
    dtype: torch.dtype = torch.float64,
    noise_scale: float = 0.1,
) -> None:
    """
    Test gradient correctness for AcadosImplicitLayer using finite differences.

    Args:
        implicit_layer: The implicit layer to test
        n_batch: Number of batch samples to generate
        max_batch_size: Maximum allowed batch size for performance
        dtype: PyTorch data type for tensors
        noise_scale: Scale factor for noise added to parameters
    """

    def _create_backward_test_function(
        forward_func: Callable, output_selector: Callable[[tuple], torch.Tensor]
    ) -> Callable:
        """Create a test function that returns (output, status) tuple."""

        def test_func(*args, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
            result = forward_func(*args, **kwargs)
            ctx = result[0]
            output = output_selector(result)
            return output, torch.tensor(ctx.status, dtype=torch.float64)

        return test_func

    # note: result 0:ctx, 1:u0, 2:x, 3:u, 4:value
    def _create_du0dx0_test(implicit_layer: AcadosImplicitLayer) -> Callable:
        """Create test function for du0/dx0 gradient."""

        def forward_func(x0):
            return implicit_layer.forward(x0=x0)

        return _create_backward_test_function(
            forward_func, lambda result: result[1]
        )  # u0

    def _create_dVdx0_test(implicit_layer: AcadosImplicitLayer) -> Callable:
        """Create test function for dV/dx0 gradient."""

        def forward_func(x0):
            return implicit_layer.forward(x0=x0)

        return _create_backward_test_function(
            forward_func, lambda result: result[4]
        )  # value

    def _create_dQdx0_test(
        implicit_layer: AcadosImplicitLayer, u0: torch.Tensor
    ) -> Callable:
        """Create test function for dQ/dx0 gradient."""

        def forward_func(x0):
            return implicit_layer.forward(x0=x0, u0=u0)

        return _create_backward_test_function(
            forward_func, lambda result: result[4]
        )  # value

    def _create_du0dp_global_test(
        implicit_layer: AcadosImplicitLayer, x0: torch.Tensor
    ) -> Callable:
        """Create test function for du0/dp_global gradient."""

        def forward_func(p_global):
            return implicit_layer.forward(x0=x0, p_global=p_global)

        return _create_backward_test_function(
            forward_func, lambda result: result[1]
        )  # u0

    def _create_dVdp_global_test(
        implicit_layer: AcadosImplicitLayer, x0: torch.Tensor
    ) -> Callable:
        """Create test function for dV/dp_global gradient."""

        def forward_func(p_global):
            return implicit_layer.forward(x0=x0, p_global=p_global)

        return _create_backward_test_function(
            forward_func, lambda result: result[4]
        )  # value

    def _create_dQdp_global_test(
        implicit_layer: AcadosImplicitLayer, x0: torch.Tensor, u0: torch.Tensor
    ) -> Callable:
        """Create test function for dQ/dp_global gradient."""

        def forward_func(p_global):
            return implicit_layer.forward(x0=x0, u0=u0, p_global=p_global)

        return _create_backward_test_function(
            forward_func, lambda result: result[4]
        )  # value

    def _create_dQdu0_test(
        implicit_layer: AcadosImplicitLayer, x0: torch.Tensor, p_global: torch.Tensor
    ) -> Callable:
        """Create test function for dQ/du0 gradient."""

        def forward_func(u0):
            return implicit_layer.forward(x0=x0, u0=u0, p_global=p_global)

        return _create_backward_test_function(
            forward_func, lambda result: result[4]
        )  # value
        # Validate batch size
        if n_batch > max_batch_size:
            raise ValueError(
                f"Batch size {n_batch} exceeds maximum {max_batch_size}. "
                "Large batch sizes make the test very slow."
            )

    # Setup test data
    test_data = _setup_test_data(implicit_layer, n_batch, dtype, noise_scale)

    # Define gradient check configurations
    configs = {
        "standard": GradCheckConfig(atol=1e-2, eps=1e-4),
        "high_tolerance": GradCheckConfig(atol=5e-2, eps=1e-4),
        "fine_eps": GradCheckConfig(atol=1e-2, eps=1e-6),
    }

    # Define test cases
    test_cases = [
        ("dV/dx0", _create_dVdx0_test(implicit_layer), test_data.x0, "standard"),
        ("du0/dx0", _create_du0dx0_test(implicit_layer), test_data.x0, "standard"),
        (
            "dQ/dx0",
            _create_dQdx0_test(implicit_layer, test_data.u0),
            test_data.x0,
            "standard",
        ),
        (
            "du0/dp_global",
            _create_du0dp_global_test(implicit_layer, test_data.x0),
            test_data.p_global,
            "standard",
        ),
        (
            "dV/dp_global",
            _create_dVdp_global_test(implicit_layer, test_data.x0),
            test_data.p_global,
            "high_tolerance",
        ),
        (
            "dQ/dp_global",
            _create_dQdp_global_test(implicit_layer, test_data.x0, test_data.u0),
            test_data.p_global,
            "fine_eps",
        ),
        (
            "dQ/du0",
            _create_dQdu0_test(implicit_layer, test_data.x0, test_data.p_global),
            test_data.u0,
            "standard",
        ),
    ]

    # Run gradient checks
    for test_name, test_func, test_input, config_name in test_cases:
        config = configs[config_name]
        try:
            torch.autograd.gradcheck(
                test_func,
                test_input,
                atol=config.atol,
                eps=config.eps,
                raise_exception=config.raise_exception,
            )
            print(f"✓ {test_name} gradient check passed")
        except Exception as e:
            print(f"✗ {test_name} gradient check failed: {e}")
            raise
