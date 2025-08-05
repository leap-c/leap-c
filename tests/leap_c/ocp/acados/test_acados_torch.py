import copy
import platform
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from acados_template import AcadosOcp

from leap_c.ocp.acados.torch import AcadosDiffMpc, AcadosDiffMpcCtx

# import matplotlib.pyplot as plt


def test_initialization_with_stagewise_varying_params(
    diff_mpc_with_stagewise_varying_params: AcadosDiffMpc,
) -> None:
    """
    Test the initialization of the AcadosImplicitLayer with stagewise varying
    parameters.
    """
    assert diff_mpc_with_stagewise_varying_params is not None, (
        "AcadosImplicitLayer with stagewise varying parameters should not be None."
    )


def test_initialization(diff_mpc: AcadosDiffMpc) -> None:
    assert diff_mpc is not None, (
        "AcadosDiffMpc should not be None after initialization."
    )


def test_file_management(diff_mpc: AcadosDiffMpc, tol: float = 1e-5) -> None:
    """
    Tests the file management behavior of AcadosDiffMpcFunction during solver
    reloading and code export.

    Args:
        diff_mpc: The differentiable mpc object containing
            the OCP (Optimal Control Problem) and related configurations.
        tol: The tolerance for comparing file modification times.
            Defaults to 1e-5.

    Raises:
        AssertionError: If any of the following conditions are not met:
            - The `c_generated_code` directory does not exist or is not a
                directory.
            - No `.so` files are found in the `c_generated_code` directory.
            - Reloading the solver modifies the `.so` files beyond the specified
                tolerance.
    """
    code_export_directory = Path(diff_mpc.ocp.code_export_directory)
    export_directory = code_export_directory.parent

    assert code_export_directory.exists(), "c_generated_code directory does not exist"
    assert code_export_directory.is_dir(), "c_generated_code is not a directory"

    # Get all files in the directory
    system = platform.system()
    if system == "Windows":
        extensions = ["*.dll"]
    elif system == "Darwin":
        extensions = ["*.dylib", "*.so"]  # Support both for robustness
    else:
        extensions = ["*.so"]

    files = []
    for ext in extensions:
        files.extend(f for f in code_export_directory.glob(ext) if f.is_file())

    assert len(files) > 0, "No *.so files found in c_generated_code directory"

    last_modified = files[0].stat().st_mtime

    # Should reload the solver
    AcadosDiffMpc(
        ocp=diff_mpc.ocp,
        initializer=diff_mpc.diff_mpc_fun.initializer,
        sensitivity_ocp=diff_mpc.diff_mpc_fun.backward_batch_solver.ocp_solvers[
            0
        ].acados_ocp,  # type: ignore
        export_directory=export_directory,
    )

    reloaded_last_modified = files[0].stat().st_mtime

    assert last_modified - reloaded_last_modified < tol, (
        "The reloaded initialization should not modify the library files."
    )


def test_statelessness(diff_mpc: AcadosDiffMpc) -> None:
    """
    Test the statelessness of AcadosDiffMpc by verifying that the
    layer produces consistent outputs for identical inputs and different outputs
    for modified parameters.

    This test ensures that:
    1. The layer's output changes when global and stagewise parameters are modified.
    2. The layer's output remains consistent for identical inputs, confirming
        stateless behavior.

    Args:
        diff_mpc: The implicit layer to be tested.

    Raises:
        AssertionError: If the layer does not produce different outputs for
                        different parameters or if it does not produce consistent
                        outputs for identical inputs.
    """
    x0 = np.tile(
        A=np.array([0.5, 0.5, 0.5, 0.5]),
        reps=(diff_mpc.diff_mpc_fun.forward_batch_solver.N_batch_max, 1),
    )
    u0 = np.tile(
        A=np.array([0.5, 0.5]),
        reps=(diff_mpc.diff_mpc_fun.forward_batch_solver.N_batch_max, 1),
    )

    p_global = diff_mpc.diff_mpc_fun.ocp.p_global_values

    assert p_global is not None

    p_global = p_global + np.ones(p_global.shape[0]) * 0.01

    p_global = np.tile(
        A=p_global,
        reps=(diff_mpc.diff_mpc_fun.forward_batch_solver.N_batch_max, 1),
    )

    p_stagewise = diff_mpc.diff_mpc_fun.ocp.parameter_values

    assert p_stagewise is not None

    p_stagewise = p_stagewise + np.ones(p_stagewise.shape) * 0.01

    p_stagewise = np.tile(
        A=p_stagewise,
        reps=(
            diff_mpc.diff_mpc_fun.forward_batch_solver.N_batch_max,
            diff_mpc.ocp.solver_options.N_horizon + 1,  # type: ignore
            1,
        ),
    )

    solutions = []
    solutions.append(diff_mpc(x0=x0, u0=u0))

    solutions.append(
        diff_mpc(
            x0=x0 - 0.01,
            u0=u0 - 0.01,
            p_global=p_global,
            p_stagewise=p_stagewise,
        )
    )

    solutions.append(diff_mpc(x0=x0, u0=u0))

    assert not np.allclose(solutions[0][-1], solutions[1][-1]), (
        "The solutions should be different due to different parameters."
    )

    for i, field_name in enumerate(["u0", "x", "u", "Q value"], start=1):
        np.testing.assert_allclose(
            solutions[0][i],
            solutions[2][i],
            err_msg=f"The solutions should have the same {field_name}.",
        )


def test_backup_functionality(diff_mpc: AcadosDiffMpc) -> None:
    """
    Test the backup functionality of AcadosDiffMpc.

    This test verifies that the backup mechanism in the implicit layer can
    restore a corrupted iterate to a valid state and produce consistent
    solutions. It simulates a scenario where the iterate is corrupted by
    setting its fields to NaN and ensures that the backup functionality
    restores the iterate correctly.

    Args:
        diff_mpc: The AcadosDiffMpc to be tested.

    Raises:
        AssertionError: If the solver does not converge or if the solutions
                        before and after restoration are not consistent.
    """
    reps = (diff_mpc.diff_mpc_fun.forward_batch_solver.N_batch_max, 1)
    x0 = np.tile(A=np.array([0.5, 0.5, 0.5, 0.5]), reps=reps)
    u0 = np.tile(A=np.array([0.5, 0.5]), reps=reps)

    solutions = []
    solutions.append(diff_mpc(x0=x0, u0=u0))

    assert np.all(solutions[-1][0].status == 0), (
        "Solver did not converge for all samples."
    )

    # Set the iterate to NaN for all fields
    # This simulates a scenario where the iterate is corrupted
    ctx = copy.deepcopy(solutions[-1][0])
    for field_name in ["x", "u", "z", "sl", "su", "pi", "lam"]:
        original_field = getattr(ctx.iterate, field_name)
        setattr(
            ctx.iterate,
            field_name,
            np.full_like(original_field, np.nan),
        )

    # Test that the backup function can restore the iterate
    solutions.append(diff_mpc(x0=x0, u0=u0, ctx=ctx))

    assert np.all(solutions[-1][0].status == 0), (
        "Solver did not converge for all samples."
    )

    for i, field_name in enumerate(["u0", "x", "u", "Q value"], start=1):
        np.testing.assert_allclose(
            solutions[0][i],
            solutions[1][i],
            err_msg=f"The solutions should have the same {field_name}.",
        )


def test_closed_loop(
    diff_mpc: AcadosDiffMpc,
    diff_mpc_with_stagewise_varying_params: AcadosDiffMpc,
    tol: float = 1e-1,
) -> None:
    """
    Tests the closed-loop behavior of a system controlled by AcadosDiffMpc.

    This function simulates a closed-loop system for 100 steps, where the control
    inputs are computed using the provided implicit layer. It verifies that the
    solver converges at each step and checks that the system states and control
    inputs stabilize within a specified threshold.

    Args:
        diff_mpc: The implicit layer representing the
            control system, which includes the solver and problem definition.
        tol: The tolerance for checking the stabilization of states and

    Raises:
        AssertionError: If the solver fails to converge at any step or if the
            median of the last 10 states or control inputs exceeds the specified
            threshold.
    """
    for diff_mpc_k in [
        diff_mpc_with_stagewise_varying_params,
        diff_mpc,
    ]:
        nx = diff_mpc_k.ocp.dims.nx

        x0 = np.array([0.5, 0.5, 0.5, 0.5])
        x = [x0]
        u = []

        # Need first dimension of inputs to be batch size
        n_batch = 1

        p_global = diff_mpc_k.ocp.p_global_values.reshape(
            n_batch, diff_mpc_k.ocp.dims.np_global
        )

        for step in range(100):
            # Need first dimension to be batch size
            x0 = np.array(x[-1].reshape(n_batch, nx))  # type: ignore
            ctx, u0, _, _, _ = diff_mpc_k.forward(x0=x0, p_global=p_global)  # type: ignore
            assert ctx.status == 0, f"Did not converge to a solution in step {step}"
            u.append(u0)
            x.append(
                diff_mpc_k.diff_mpc_fun.forward_batch_solver.ocp_solvers[0].get(1, "x")
            )

        x = np.array(x)
        u = np.array(u).reshape(100, 2)

        assert np.median(x[-10:, 0]) <= tol, "Median of x[-10:, 0] exceeds threshold"
        assert np.median(x[-10:, 1]) <= tol, "Median of x[-10:, 1] exceeds threshold"
        assert np.median(u[-10:]) <= tol, "Median of u[-10:] exceeds threshold"

    #     fig = plt.figure(figsize=(10, 5))
    #     ax1 = fig.add_subplot(121)
    #     ax1.plot(x[:, 0], label="x[0]")
    #     ax1.plot(x[:, 1], label="x[1]")
    #     ax1.set_title("State Trajectories")
    #     ax1.set_xlabel("Time Step")
    #     ax1.set_ylabel("Position")
    #     ax1.legend()
    #     ax2 = fig.add_subplot(122)
    #     ax2.plot(u[:, 0], label="u[0]")
    #     ax2.plot(u[:, 1], label="u[1]")
    #     ax2.set_title("Control Inputs")
    #     ax2.set_xlabel("Time Step")
    #     ax2.set_ylabel("Control")
    #     ax2.legend()
    #     plt.tight_layout()
    # plt.show()


@dataclass
class GradCheckConfig:
    """Configuration for gradient checking parameters."""

    atol: float = 1e-2
    rtol: float = 1e-3
    eps: float = 1e-4
    raise_exception: bool = True


@dataclass
class AcadosTestInputs:
    """Container for test data tensors."""

    x0: torch.Tensor
    u0: torch.Tensor
    p_global: torch.Tensor


def _setup_test_inputs(
    diff_mpc: AcadosDiffMpc,
    n_batch: int,
    dtype: torch.dtype,
    noise_scale: float,
) -> AcadosTestInputs:
    """Set up test data tensors with proper gradients enabled."""
    ocp = diff_mpc.ocp

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
        torch.tensor(np.zeros(ocp.dims.nu), dtype=dtype).unsqueeze(0).repeat(n_batch, 1)  # type: ignore
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

    return AcadosTestInputs(x0=x0_batch, u0=u0_batch, p_global=p_global)


def test_forward(
    diff_mpc: AcadosDiffMpc,
    diff_mpc_with_stagewise_varying_params: AcadosDiffMpc,
    n_batch: int = 4,
    dtype: torch.dtype = torch.float64,
    noise_scale: float = 0.05,
    verbosity: int = 0,
) -> None:
    """
    Test the forward method of AcadosDiffMpc with different input combinations.

    Args:
        diff_mpc: The differentiable mpc to test
        n_batch: Number of batch samples
        dtype: PyTorch data type for tensors
        noise_scale: Scale factor for noise added to parameters
        test_all_variants: If True, test all forward call variants
        verbosity: Level of verbosity for test output
    """

    def _run_single_forward_test(
        diff_mpc: AcadosDiffMpc,
        forward_kwargs: dict[str, torch.Tensor],
        expected_output_type: str,
        n_batch: int,
        acados_ocp: AcadosOcp,
    ) -> None:
        """Run a single forward test with given parameters."""
        # Call forward method
        ctx, u0, x, u, value = diff_mpc(**forward_kwargs)

        # Validate solver status
        assert np.all(ctx.status == 0), (
            f"Forward method failed with status {ctx.status}"
        )

        # Validate output types
        assert isinstance(ctx, AcadosDiffMpcCtx), (
            "ctx should be an instance of AcadosDiffMpcCtx"
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
            (acados_ocp.solver_options.N_horizon + 1),  # type: ignore
            acados_ocp.dims.nx,
        )
        assert x.shape == expected_x_shape, (
            f"x shape mismatch. Expected: {expected_x_shape}, Got: {x.shape}"
        )

        expected_u_shape = (
            n_batch,
            acados_ocp.solver_options.N_horizon,
            acados_ocp.dims.nu,
        )
        assert u.shape == expected_u_shape, (
            f"u shape mismatch. Expected: {expected_u_shape}, Got: {u.shape}"
        )

        # Validate value shape (same for both V and Q)
        expected_value_shape = (n_batch,)
        assert value.shape == expected_value_shape, (
            f"{expected_output_type} shape mismatch. Expected: {expected_value_shape}"
            f", Got: {value.shape}"
        )

    for diff_mpc_k in [diff_mpc_with_stagewise_varying_params, diff_mpc]:
        acados_ocp = diff_mpc_k.ocp
        n_batch = diff_mpc_k.diff_mpc_fun.forward_batch_solver.N_batch_max

        # Setup test data
        test_inputs = _setup_test_inputs(diff_mpc_k, n_batch, dtype, noise_scale)

        # Define test cases
        test_cases = [
            {
                "name": "x0_only",
                "kwargs": {"x0": test_inputs.x0},
                "expected_output": "V",
            },
            {
                "name": "x0_and_u0",
                "kwargs": {"x0": test_inputs.x0, "u0": test_inputs.u0},
                "expected_output": "Q",
            },
            {
                "name": "x0_and_p_global",
                "kwargs": {"x0": test_inputs.x0, "p_global": test_inputs.p_global},
                "expected_output": "V",
            },
            {
                "name": "all_parameters",
                "kwargs": {
                    "x0": test_inputs.x0,
                    "u0": test_inputs.u0,
                    "p_global": test_inputs.p_global,
                },
                "expected_output": "Q",
            },
        ]

        for test_case in test_cases:
            if verbosity > 0:
                print(f"Testing forward call: {test_case['name']}")

            _run_single_forward_test(
                diff_mpc_k,
                test_case["kwargs"],
                test_case["expected_output"],
                n_batch,
                acados_ocp,
            )


def test_sensitivity(
    diff_mpc: AcadosDiffMpc,
    diff_mpc_with_stagewise_varying_params: AcadosDiffMpc,
    n_batch: int = 4,
    max_batch_size: int = 10,
    dtype: torch.dtype = torch.float64,
    noise_scale: float = 0.1,
) -> None:
    """
    Test sensitivity of AcadosDiffMpc to changes in parameters.

    Args:
        diff_mpc: The differentiable mpc to test
        n_batch: Number of batch samples to generate
        max_batch_size: Maximum allowed batch size for performance
        dtype: PyTorch data type for tensors
        noise_scale: Scale factor for noise added to parameters
    """
    # Validate batch size
    if n_batch > max_batch_size:
        error_message = (
            f"Batch size {n_batch} exceeds maximum {max_batch_size}. "
            "Large batch sizes make the test very slow."
        )
        raise ValueError(error_message)

    for diff_mpc_k in [
        diff_mpc_with_stagewise_varying_params,
        diff_mpc,
    ]:
        # Setup test data
        test_inputs = _setup_test_inputs(diff_mpc_k, n_batch, dtype, noise_scale)

        ctx, u0, x, u, value = diff_mpc_k.forward(x0=test_inputs.x0)

        results = {
            field: diff_mpc_k.sensitivity(ctx=ctx, field_name=field)  # type: ignore
            for field in ["dvalue_dp_global", "dvalue_dx0"]
        }

        assert results["dvalue_dp_global"].shape == (
            n_batch,
            diff_mpc_k.ocp.dims.np_global,
        ), (
            f"dvalue_dp_global shape mismatch. Expected: \
                {(n_batch, diff_mpc_k.ocp.dims.np_global)}, "
            f"Got: {results['dvalue_dp_global'].shape}"
        )

        assert results["dvalue_dx0"].shape == (
            n_batch,
            diff_mpc_k.ocp.dims.nx,
        ), (
            f"dvalue_dx0 shape mismatch. Expected: {(n_batch, diff_mpc_k.ocp.dims.nx)},"
            f" "
            f"Got: {results['dvalue_dx0'].shape}"
        )

        ctx, u0, x, u, value = diff_mpc_k.forward(x0=test_inputs.x0, u0=test_inputs.u0)
        results["dvalue_du0"] = diff_mpc_k.sensitivity(ctx=ctx, field_name="dvalue_du0")

        assert results["dvalue_du0"].shape == (
            n_batch,
            diff_mpc_k.ocp.dims.nu,
        ), (
            f"dvalue_du0 shape mismatch. Expected: {(n_batch, diff_mpc_k.ocp.dims.nu)},"
            f" "
            f"Got: {results['dvalue_du0'].shape}"
        )


def test_backward(
    diff_mpc: AcadosDiffMpc,
    diff_mpc_with_stagewise_varying_params: AcadosDiffMpc,
    n_batch: int = 4,
    max_batch_size: int = 10,
    dtype: torch.dtype = torch.float64,
    noise_scale: float = 0.1,
) -> None:
    """
    Test backward pass of AcadosDiffMpc using finite differences.

    Args:
        diff_mpc: The differentiable mpc to test
        n_batch: Number of batch samples to generate
        max_batch_size: Maximum allowed batch size for performance
        dtype: PyTorch data type for tensors
        noise_scale: Scale factor for noise added to parameters
    """
    # Validate batch size
    if n_batch > max_batch_size:
        error_message = (
            f"Batch size {n_batch} exceeds maximum {max_batch_size}. "
            "Large batch sizes make the test very slow."
        )
        raise ValueError(error_message)

    def _create_backward_test_function(
        forward_func: Callable, output_selector: Callable[[tuple], torch.Tensor]
    ) -> Callable:
        """Create a test function that returns (output, status) tuple."""

        def test_func(*args, **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
            result = forward_func(*args, **kwargs)
            ctx = result[0]

            # Validate solver status
            assert np.all(ctx.status == 0), (
                f"Forward method failed with status {ctx.status}"
            )
            output = output_selector(result)
            return output, torch.tensor(ctx.status, dtype=torch.float64)

        return test_func

    # note: result 0:ctx, 1:u0, 2:x, 3:u, 4:value
    def _create_du0dx0_test(diff_mpc: AcadosDiffMpc) -> Callable:
        """Create test function for du0/dx0 gradient."""

        def forward_func(x0):
            return diff_mpc.forward(x0=x0)

        return _create_backward_test_function(
            forward_func, lambda result: result[1]
        )  # u0

    def _create_dVdx0_test(diff_mpc: AcadosDiffMpc) -> Callable:
        """Create test function for dV/dx0 gradient."""

        def forward_func(x0):
            return diff_mpc.forward(x0=x0)

        return _create_backward_test_function(
            forward_func, lambda result: result[4]
        )  # value

    def _create_dQdx0_test(diff_mpc: AcadosDiffMpc, u0: torch.Tensor) -> Callable:
        """Create test function for dQ/dx0 gradient."""

        def forward_func(x0):
            return diff_mpc.forward(x0=x0, u0=u0)

        return _create_backward_test_function(
            forward_func, lambda result: result[4]
        )  # value

    def _create_du0dp_global_test(
        diff_mpc: AcadosDiffMpc, x0: torch.Tensor
    ) -> Callable:
        """Create test function for du0/dp_global gradient."""

        def forward_func(p_global):
            return diff_mpc.forward(x0=x0, p_global=p_global)

        return _create_backward_test_function(
            forward_func, lambda result: result[1]
        )  # u0

    def _create_dVdp_global_test(diff_mpc: AcadosDiffMpc, x0: torch.Tensor) -> Callable:
        """Create test function for dV/dp_global gradient."""

        def forward_func(p_global):
            return diff_mpc.forward(x0=x0, p_global=p_global)

        return _create_backward_test_function(
            forward_func, lambda result: result[4]
        )  # value

    def _create_dQdp_global_test(
        diff_mpc: AcadosDiffMpc, x0: torch.Tensor, u0: torch.Tensor
    ) -> Callable:
        """Create test function for dQ/dp_global gradient."""

        def forward_func(p_global):
            return diff_mpc.forward(x0=x0, u0=u0, p_global=p_global)

        return _create_backward_test_function(
            forward_func, lambda result: result[4]
        )  # value

    def _create_dQdu0_test(
        diff_mpc: AcadosDiffMpc, x0: torch.Tensor, p_global: torch.Tensor
    ) -> Callable:
        """Create test function for dQ/du0 gradient."""

        def forward_func(u0):
            return diff_mpc.forward(x0=x0, u0=u0, p_global=p_global)

        return _create_backward_test_function(
            forward_func, lambda result: result[4]
        )  # value

    def _create_dxdp_global_test(diff_mpc: AcadosDiffMpc, x0: torch.Tensor) -> Callable:
        """Create test function for dx/dp_global gradient."""

        def forward_func(p_global):
            return diff_mpc.forward(x0=x0, p_global=p_global)

        return _create_backward_test_function(
            forward_func, lambda result: result[2]
        )  # x

    def _create_dudp_global_test(diff_mpc: AcadosDiffMpc, x0: torch.Tensor) -> Callable:
        """Create test function for du/dp_global gradient."""

        def forward_func(p_global):
            return diff_mpc.forward(x0=x0, p_global=p_global)

        return _create_backward_test_function(
            forward_func, lambda result: result[3]
        )  # u

    def _create_dhvacudp_global_test(
        diff_mpc: AcadosDiffMpc, x0: torch.Tensor
    ) -> Callable:
        """Create test function for dfakeu/dp_global gradient."""

        def forward_func(p_global):
            return diff_mpc.forward(x0=x0, p_global=p_global)

        return _create_backward_test_function(
            forward_func,
            lambda result: result[2],  # [:, 1, 3][:, None]
        )

    for i, diff_mpc_k in enumerate(
        [  # diff_mpc_with_stagewise_varying_params,
            diff_mpc
        ]
    ):
        print(f"Testing the {i}th Differentiable MPC Backward Pass")
        print("=========================================================")
        test_inputs = _setup_test_inputs(diff_mpc_k, n_batch, dtype, noise_scale)

        # TODO: Sensitivities with respect to different parameters have different scales
        # that lead to different tolerances and step sizes for the parameters. At the moment,
        # we use a single set of tolerances and step sizes for all parameters.

        test_cases = [
            (
                "dV/dx0",
                _create_dVdx0_test(diff_mpc_k),
                test_inputs.x0,
                GradCheckConfig(atol=1e-1, eps=1e-2),
            ),
            (
                "du0/dx0",
                _create_du0dx0_test(diff_mpc_k),
                test_inputs.x0,
                GradCheckConfig(atol=1e0, eps=1e-4),
            ),
            (
                "dQ/dx0",
                _create_dQdx0_test(diff_mpc_k, test_inputs.u0),
                test_inputs.x0,
                GradCheckConfig(atol=1e-2, eps=1e-2),
            ),
            (
                "du0/dp_global",
                _create_du0dp_global_test(diff_mpc_k, test_inputs.x0),
                test_inputs.p_global,
                GradCheckConfig(atol=1e-1, eps=1e-4),
            ),
            (
                "dV/dp_global",
                _create_dVdp_global_test(diff_mpc_k, test_inputs.x0),
                test_inputs.p_global,
                GradCheckConfig(atol=1e-2, eps=1e-2),
            ),
            (
                "dQ/dp_global",
                _create_dQdp_global_test(diff_mpc_k, test_inputs.x0, test_inputs.u0),
                test_inputs.p_global,
                GradCheckConfig(atol=1e-2, eps=1e-2),
            ),
            (
                "dQ/du0",
                _create_dQdu0_test(diff_mpc_k, test_inputs.x0, test_inputs.p_global),
                test_inputs.u0,
                GradCheckConfig(atol=1e-2, eps=1e-2),
            ),
            (
                "dx/dp_global",
                _create_dxdp_global_test(diff_mpc_k, test_inputs.x0),
                test_inputs.p_global,
                GradCheckConfig(atol=1e-2, eps=1e-4),
            ),
            (
                "du/dp_global",
                _create_dudp_global_test(diff_mpc_k, test_inputs.x0),
                test_inputs.p_global,
                GradCheckConfig(atol=4 * 1e-2, eps=1e-4),
            ),
        ]

        # Run gradient checks
        for test_name, test_func, test_input, config in test_cases:
            try:
                print(f"{test_name} gradient check running")
                torch.autograd.gradcheck(
                    func=test_func,
                    inputs=test_input,
                    atol=config.atol,
                    rtol=config.rtol,
                    eps=config.eps,
                    raise_exception=True,
                )
                print(f"✓ {test_name} gradient check passed")
            except Exception as e:
                print(f"✗ {test_name} gradient check failed: {e}")
                raise


# NOTE: Useful for debugging the sensitivities.
# I inserted it somewhere around "if not _allclose_with_type_promotion(a, n, rtol, atol):"
# in gradcheck.py .

# def tell_me_more(a, n, rtol, atol):
#     print("max:", a.max(), n.max())
#     print("min:", a.min(), n.min())
#     print("Max difference:", (a - n).abs().max())
#     neq = ~torch.isclose(a, n, rtol=rtol, atol=atol)
#     nonzero = ~torch.isclose(a, torch.zeros_like(a), rtol=rtol, atol=atol)
#     nnonzero = ~torch.isclose(a, torch.zeros_like(n), rtol=rtol, atol=atol)
#     print("Wrong elements: ", neq.sum())
#     print("Nonzero elements: ", (nonzero.sum()))
#     print("Indices of wrong elements: ", torch.nonzero(neq, as_tuple=True))
#     nonzero_ind_a = torch.nonzero(nonzero, as_tuple=True)
#     nonzero_ind_n = torch.nonzero(nnonzero, as_tuple=True)
#     for entry_a, entry_n in zip(nonzero_ind_a, nonzero_ind_n
#     ):
#         if not torch.allclose(entry_a, entry_n):
#             print("Indices of nonzero elements a: ", nonzero_ind_a)
#             print("Indices of nonzero elements n: ", nonzero_ind_n)
#             raise Exception("Nonzero indices do not match")
#         else:
#             print("Nonzero indices match")
