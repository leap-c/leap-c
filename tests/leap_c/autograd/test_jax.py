from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np

from leap_c.autograd.function import DiffFunction
from leap_c.autograd.jax import create_jax_callable


class DummyFunction(DiffFunction):
    """Simple test function with known analytical derivatives.

    Forward returns: u_star = x0 + 1 (+ a warmstart offset taken from ``ctx.iterate``,
    if a previous ``ctx`` is passed in), value = sum(x0**2).
    Backward returns analytical VJP.
    """

    def forward(self, ctx, x0_np, u0_np, p_global_np, p_stagewise_np, p_stagewise_sparse_idx_np):
        B = x0_np.shape[0]
        warmstart_offset = ctx.iterate if ctx is not None else 0.0
        new_ctx = SimpleNamespace(x0=x0_np.copy(), iterate=x0_np.sum())
        u_star = x0_np + 1 + warmstart_offset
        x = np.broadcast_to(u_star[:, None, :], (B, 3, x0_np.shape[1]))
        u = np.broadcast_to(u_star[:, None, :], (B, 2, x0_np.shape[1]))
        value = np.sum(x0_np**2, axis=1, keepdims=True)
        return new_ctx, u_star, x, u, value

    def backward(self, ctx, u0_grad, x_grad, u_grad, value_grad):
        x0_np = ctx.x0
        grad_x0 = np.zeros_like(x0_np)
        if u0_grad is not None:
            grad_x0 += u0_grad
        if value_grad is not None:
            grad_x0 += 2 * x0_np * value_grad
        return grad_x0, None, None, None, None


def test_create_jax_callable_forward():
    fun = DummyFunction()
    solve = create_jax_callable(fun, N_horizon=2, nx=2, nu=2)

    x0 = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    u0 = jnp.array([[0.5, 0.5], [0.5, 0.5]])
    p_global = jnp.array([[0.0], [0.0]])
    p_stagewise = jnp.zeros((2, 3, 1))

    ctx, u_star, x, u, value = solve(x0, u0, p_global, p_stagewise)

    expected_u_star = x0 + 1
    expected_value = jnp.sum(x0**2, axis=1, keepdims=True)

    assert ctx is not None
    assert jnp.allclose(u_star, expected_u_star)
    assert jnp.allclose(value, expected_value)


def test_create_jax_callable_grad_wrt_x0():
    fun = DummyFunction()
    solve = create_jax_callable(fun, N_horizon=2, nx=2, nu=2)

    x0 = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    u0 = jnp.array([[0.5, 0.5], [0.5, 0.5]])
    p_global = jnp.array([[0.0], [0.0]])
    p_stagewise = jnp.zeros((2, 3, 1))

    def loss_fn(x0_val):
        _, u_star, x, u, value = solve(x0_val, u0, p_global, p_stagewise)
        return jnp.sum(u_star) + jnp.sum(value)

    grad_x0 = jax.grad(loss_fn)(x0)
    expected_grad = jnp.ones_like(x0) + 2 * x0
    assert jnp.allclose(grad_x0, expected_grad, atol=1e-6)


def test_create_jax_callable_ctx_warmstart():
    """Passing a previous ``ctx`` back in should reach ``fun.forward`` as the ``ctx`` argument."""
    fun = DummyFunction()
    solve = create_jax_callable(fun, N_horizon=2, nx=2, nu=2)

    x0 = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    u0 = jnp.array([[0.5, 0.5], [0.5, 0.5]])
    p_global = jnp.array([[0.0], [0.0]])
    p_stagewise = jnp.zeros((2, 3, 1))

    ctx1, u_star1, *_ = solve(x0, u0, p_global, p_stagewise)
    ctx2, u_star2, *_ = solve(x0, u0, p_global, p_stagewise, ctx=ctx1)

    assert jnp.allclose(u_star1, x0 + 1)
    assert jnp.allclose(u_star2, x0 + 1 + ctx1.iterate)


def test_create_jax_callable_does_not_cross_contaminate_ctx_across_calls():
    """Guard against a shared/nonlocal context.

    Two forward-only calls on the same ``solve`` before either is differentiated must not share
    context: each call's backward pass must use its own forward pass's context, not whichever call
    happened to run most recently.
    """
    fun = DummyFunction()
    solve = create_jax_callable(fun, N_horizon=2, nx=2, nu=2)

    x0_a = jnp.array([[1.0, 2.0]])
    x0_b = jnp.array([[10.0, 20.0]])
    u0 = jnp.array([[0.5, 0.5]])
    p_global = jnp.array([[0.0]])
    p_stagewise = jnp.zeros((1, 3, 1))

    def loss_fn(x0_a_val, x0_b_val):
        _, u_star_a, _, _, value_a = solve(x0_a_val, u0, p_global, p_stagewise)
        _, u_star_b, _, _, value_b = solve(x0_b_val, u0, p_global, p_stagewise)
        return jnp.sum(u_star_a) + jnp.sum(value_a) + jnp.sum(u_star_b) + jnp.sum(value_b)

    grad_a, grad_b = jax.grad(loss_fn, argnums=(0, 1))(x0_a, x0_b)

    assert jnp.allclose(grad_a, jnp.ones_like(x0_a) + 2 * x0_a, atol=1e-6)
    assert jnp.allclose(grad_b, jnp.ones_like(x0_b) + 2 * x0_b, atol=1e-6)


def test_create_jax_callable_grad_wrt_p_global():
    class DummyWithP(DiffFunction):
        def forward(
            self, ctx, x0_np, u0_np, p_global_np, p_stagewise_np, p_stagewise_sparse_idx_np
        ):
            B = x0_np.shape[0]
            new_ctx = SimpleNamespace(x0=x0_np.copy(), p_global=p_global_np.copy())
            u_star = x0_np + p_global_np[:, :2]
            x = np.broadcast_to(u_star[:, None, :], (B, 3, 2))
            u = np.broadcast_to(u_star[:, None, :], (B, 2, 2))
            value = np.sum(p_global_np**2, axis=1, keepdims=True)
            return new_ctx, u_star, x, u, value

        def backward(self, ctx, u0_grad, x_grad, u_grad, value_grad):
            x0_np = ctx.x0
            p_global_np = ctx.p_global
            grad_x0 = np.zeros_like(x0_np)
            grad_p_global = np.zeros_like(p_global_np)
            if u0_grad is not None:
                grad_x0 += u0_grad
                grad_p_global[:, :2] += u0_grad
            if value_grad is not None:
                grad_p_global += 2 * p_global_np * value_grad
            return grad_x0, None, grad_p_global, None, None

    fun = DummyWithP()
    solve = create_jax_callable(fun, N_horizon=2, nx=2, nu=2)

    x0 = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    u0 = jnp.array([[0.5, 0.5], [0.5, 0.5]])
    p_global = jnp.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    p_stagewise = jnp.zeros((2, 3, 1))

    def loss_fn(p_val):
        _, u_star, x, u, value = solve(x0, u0, p_val, p_stagewise)
        return jnp.sum(u_star) + jnp.sum(value)

    grad_p = jax.grad(loss_fn)(p_global)
    # d(u_star)/dp = [1, 1, 0], d(value)/dp = 2*p
    # grad_p[:, :2] = 1 + 2*p[:, :2], grad_p[:, 2] = 2*p[:, 2]
    expected_grad_p = jnp.zeros_like(p_global)
    expected_grad_p = expected_grad_p.at[:, :2].set(
        jnp.ones_like(p_global[:, :2]) + 2 * p_global[:, :2]
    )
    expected_grad_p = expected_grad_p.at[:, 2].set(2 * p_global[:, 2])

    assert jnp.allclose(grad_p, expected_grad_p, atol=1e-6)
