import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch

from leap_c.jax import AcadosDiffMpcJax
from leap_c.torch import AcadosDiffMpcTorch

jax.config.update("jax_enable_x64", True)


def test_initialization(diff_mpc_jax: AcadosDiffMpcJax) -> None:
    assert diff_mpc_jax is not None, "diff_mpc_jax should not be None after initialization."


def test_forward_no_params(diff_mpc_jax: AcadosDiffMpcJax) -> None:
    B = 2
    nx = diff_mpc_jax.diff_mpc_fun.ocp.dims.nx
    nu = diff_mpc_jax.diff_mpc_fun.ocp.dims.nu
    N = diff_mpc_jax.diff_mpc_fun.ocp.solver_options.N_horizon

    x0 = jnp.ones((B, nx))
    ctx, u_star, x, u, value = diff_mpc_jax(x0)

    assert ctx is not None
    assert u_star.shape == (B, nu)
    assert x.shape == (B, N + 1, nx)
    assert u.shape == (B, N, nu)
    assert value.shape == (B, 1)


def test_forward_with_u0(diff_mpc_jax: AcadosDiffMpcJax) -> None:
    B = 2
    nx = diff_mpc_jax.diff_mpc_fun.ocp.dims.nx
    nu = diff_mpc_jax.diff_mpc_fun.ocp.dims.nu
    N = diff_mpc_jax.diff_mpc_fun.ocp.solver_options.N_horizon

    x0 = jnp.ones((B, nx))
    u0 = jnp.zeros((B, nu))
    _, u_star, x, u, value = diff_mpc_jax(x0, u0=u0)

    assert u_star.shape == (B, nu)
    assert x.shape == (B, N + 1, nx)
    assert u.shape == (B, N, nu)
    assert value.shape == (B, 1)


def test_statelessness(diff_mpc_jax: AcadosDiffMpcJax) -> None:
    B = 2
    nx = diff_mpc_jax.diff_mpc_fun.ocp.dims.nx

    x0 = jnp.ones((B, nx))
    _, u_star1, _, _, value1 = diff_mpc_jax(x0)
    _, u_star2, _, _, value2 = diff_mpc_jax(x0)

    assert jnp.allclose(u_star1, u_star2)
    assert jnp.allclose(value1, value2)


def test_warmstart_with_ctx(diff_mpc_jax: AcadosDiffMpcJax) -> None:
    """Feeding back the ctx from a previous call should reproduce the same solution.

    The problem is already converged, and the solver iterate stored in ctx should come from the
    most recent solve.
    """
    B = 2
    nx = diff_mpc_jax.diff_mpc_fun.ocp.dims.nx

    x0 = jnp.ones((B, nx))
    ctx1, u_star1, _, _, value1 = diff_mpc_jax(x0)
    ctx2, u_star2, _, _, value2 = diff_mpc_jax(x0, ctx=ctx1)

    assert ctx2 is not ctx1
    assert jnp.allclose(u_star1, u_star2, atol=1e-6)
    assert jnp.allclose(value1, value2, atol=1e-6)


def test_grad_wrt_x0(diff_mpc_jax: AcadosDiffMpcJax) -> None:
    B = 2
    nx = diff_mpc_jax.diff_mpc_fun.ocp.dims.nx

    x0 = jnp.ones((B, nx))

    def loss_fn(x0_val):
        _, _, _, _, value = diff_mpc_jax(x0_val)
        return jnp.sum(value)

    grad_x0 = jax.grad(loss_fn)(x0)
    assert grad_x0.shape == (B, nx)
    assert jnp.any(grad_x0 != 0), "Gradient should be non-zero"


def test_grad_wrt_u0(diff_mpc_jax: AcadosDiffMpcJax) -> None:
    B = 2
    nx = diff_mpc_jax.diff_mpc_fun.ocp.dims.nx
    nu = diff_mpc_jax.diff_mpc_fun.ocp.dims.nu

    x0 = jnp.ones((B, nx))
    u0 = jnp.ones((B, nu))

    def loss_fn(u0_val):
        _, u_star, _, _, _ = diff_mpc_jax(x0, u0=u0_val)
        return jnp.sum(u_star)

    grad_u0 = jax.grad(loss_fn)(u0)
    assert grad_u0.shape == (B, nu)


def test_grad_wrt_params(diff_mpc_jax: AcadosDiffMpcJax) -> None:
    B = 2
    nx = diff_mpc_jax.diff_mpc_fun.ocp.dims.nx

    pm = diff_mpc_jax.parameter_manager
    diff_names = pm.differentiable_parameter_names
    if not diff_names:
        pytest.skip("No differentiable parameters registered")
    param_name = diff_names[0]
    default_val = pm.parameters[param_name].broadcasted_default(pm.N_horizon)

    x0 = jnp.ones((B, nx))
    param_val = jnp.array([default_val, default_val])

    def loss_fn(p_val):
        _, _, _, _, value = diff_mpc_jax(x0, params={param_name: p_val})
        return jnp.sum(value)

    grad_p = jax.grad(loss_fn)(param_val)
    assert grad_p.shape == param_val.shape


def test_forward_with_params_dict(diff_mpc_jax: AcadosDiffMpcJax) -> None:
    B = 2
    nx = diff_mpc_jax.diff_mpc_fun.ocp.dims.nx
    nu = diff_mpc_jax.diff_mpc_fun.ocp.dims.nu
    N = diff_mpc_jax.diff_mpc_fun.ocp.solver_options.N_horizon

    pm = diff_mpc_jax.parameter_manager
    params = {}
    for name in pm.non_differentiable_parameter_names:
        default = pm.parameters[name].broadcasted_default(pm.N_horizon)
        params[name] = np.broadcast_to(default, (B, pm.N_horizon + 1, *default.shape))

    x0 = jnp.ones((B, nx))
    _, u_star, x, u, value = diff_mpc_jax(x0, params=params)

    assert u_star.shape == (B, nu)
    assert x.shape == (B, N + 1, nx)
    assert u.shape == (B, N, nu)
    assert value.shape == (B, 1)


def test_forward_and_grad_with_stagewise_varying_params(
    diff_mpc_jax_with_stagewise_varying_params: AcadosDiffMpcJax,
) -> None:
    """Exercise the stage-varying differentiable parameter path (indicator-gated segments)."""
    diff_mpc = diff_mpc_jax_with_stagewise_varying_params
    B = 2
    nx = diff_mpc.diff_mpc_fun.ocp.dims.nx

    x0 = jnp.ones((B, nx))

    def loss_fn(x0_val):
        _, _, _, _, value = diff_mpc(x0_val)
        return jnp.sum(value)

    grad_x0 = jax.grad(loss_fn)(x0)
    assert grad_x0.shape == (B, nx)
    assert jnp.any(grad_x0 != 0), "Gradient should be non-zero"


def test_jax_grad_matches_torch_grad_wrt_x0(
    diff_mpc: AcadosDiffMpcTorch, diff_mpc_jax: AcadosDiffMpcJax
) -> None:
    """Cross-check jax gradients against torch's on the same OCP fixture.

    Both layers wrap the same acados solver; their gradients of the objective value w.r.t. x0
    must match numerically, not just be non-zero. This guards against composition bugs (e.g.
    wrong argument order into ``fun.forward``) that per-piece unit tests would miss.
    """
    nx = diff_mpc.diff_mpc_fun.ocp.dims.nx
    # Small perturbation around the OCP's nominal x0 ([1.0, 0.5, 0.0, 0.0]) so the trajectory
    # stays clear of the state box constraints -- near an active constraint boundary, SQP
    # threading/rounding nondeterminism can flip the active set and make the two solves
    # (though numerically independent) land on different local solutions, which is a solver
    # artifact unrelated to the jax/torch composition this test is meant to check.
    x0_np = np.array([[1.0, 0.5, 0.0, 0.0], [1.05, 0.45, 0.02, -0.02]])[:, :nx]

    x0_torch = torch.tensor(x0_np, dtype=torch.float64, requires_grad=True)
    _, _, _, _, value_torch = diff_mpc(x0_torch)
    value_torch.sum().backward()
    grad_x0_torch = x0_torch.grad.numpy()

    def loss_fn(x0_val):
        _, _, _, _, value = diff_mpc_jax(x0_val)
        return jnp.sum(value)

    grad_x0_jax = np.asarray(jax.grad(loss_fn)(jnp.asarray(x0_np)))

    np.testing.assert_allclose(grad_x0_jax, grad_x0_torch, rtol=1e-6, atol=1e-8)


def test_jax_grad_matches_torch_grad_wrt_param(
    diff_mpc: AcadosDiffMpcTorch, diff_mpc_jax: AcadosDiffMpcJax
) -> None:
    """Cross-check jax gradients against torch's, w.r.t. a differentiable parameter.

    Same rationale as :func:`test_jax_grad_matches_torch_grad_wrt_x0`, but this exercises the
    path where a bug in ``combine_differentiable_parameters_jax``'s array composition would
    surface.
    """
    nx = diff_mpc.diff_mpc_fun.ocp.dims.nx
    # See test_jax_grad_matches_torch_grad_wrt_x0 for why this stays near the nominal x0.
    x0_np = np.array([[1.0, 0.5, 0.0, 0.0], [1.05, 0.45, 0.02, -0.02]])[:, :nx]

    pm = diff_mpc.parameter_manager
    diff_names = pm.differentiable_parameter_names
    if not diff_names:
        pytest.skip("No differentiable parameters registered")
    param_name = diff_names[0]
    default_val = pm.parameters[param_name].broadcasted_default(pm.N_horizon)
    param_np = np.array([default_val, default_val])

    x0_torch = torch.tensor(x0_np, dtype=torch.float64)
    p_torch = torch.tensor(param_np, dtype=torch.float64, requires_grad=True)
    _, _, _, _, value_torch = diff_mpc(x0_torch, params={param_name: p_torch})
    value_torch.sum().backward()
    grad_p_torch = p_torch.grad.numpy()

    def loss_fn(p_val):
        _, _, _, _, value = diff_mpc_jax(jnp.asarray(x0_np), params={param_name: p_val})
        return jnp.sum(value)

    grad_p_jax = np.asarray(jax.grad(loss_fn)(jnp.asarray(param_np)))

    np.testing.assert_allclose(grad_p_jax, grad_p_torch, rtol=1e-6, atol=1e-8)
