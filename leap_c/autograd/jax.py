"""Creates JAX callables with custom VJP rules from DiffFunction objects."""

from typing import Any, Callable

import numpy as np

from leap_c.autograd.function import DiffFunction
from leap_c.diff_mpc.function import AcadosDiffMpcCtx
from leap_c.utils.dependencies import require_jax

jax = require_jax()
from jax import custom_vjp, pure_callback  # noqa: E402
from jax.core import ShapedArray  # noqa: E402


def _to_np(val: Any) -> np.ndarray | None:
    if val is None:
        return None
    return np.asarray(val)


def _to_np_tuple(*vals: Any) -> tuple[np.ndarray | None, ...]:
    return tuple(_to_np(v) for v in vals)


def _to_jax(val: np.ndarray | None) -> "jax.Array | None":
    return None if val is None else jax.numpy.asarray(val)


def create_jax_callable(
    fun: DiffFunction,
    N_horizon: int,
    nx: int,
    nu: int,
) -> Callable[..., tuple[AcadosDiffMpcCtx, "jax.Array", "jax.Array", "jax.Array", "jax.Array"]]:
    """Creates a JAX-callable function with a custom VJP wrapping a ``DiffFunction``.

    The returned function accepts ``(x0, u0, p_global, p_stagewise, ctx=None)`` and returns
    ``(ctx, sol_u0, x, u, sol_value)``, mirroring
    :func:`leap_c.autograd.torch.create_autograd_function`: passing a ``ctx`` from a previous
    call warmstarts the solve, and the returned ``ctx`` can be fed into the next call.

    The impure acados C solver is isolated inside ``jax.pure_callback``, and gradients flow via
    ``jax.custom_vjp``. Every call builds a *fresh* ``custom_vjp``-wrapped primal closing over a
    per-call context box: the forward pass's context is threaded to its own matching backward pass
    via that box (``AcadosDiffMpcCtx`` isn't a valid JAX pytree, so it can't ride as a
    ``custom_vjp`` residual). Because the box and the VJP rule are rebuilt on every call rather than
    shared across calls, concurrent-in-time forward calls on the same layer (e.g. two rollouts
    before either is differentiated) never clobber each other's context; only the fixed cost of
    re-registering a VJP rule is paid per call, which is negligible next to the acados solve. Since
    ``pure_callback`` calls run eagerly, ``jax.grad`` works but ``jax.jit`` of the whole layer (or
    of ``jax.grad`` thereof) is not supported.

    ``pure_callback`` passes JAX arrays (not numpy) to the callback, so explicit
    ``np.asarray()`` conversion is performed before calling the core solver.

    ``needs_input_grad`` always requests sensitivities for ``x0`` and ``p_global`` (and for
    ``u0`` when it was fixed rather than left as the free variable), since ``custom_vjp`` gives
    no way to know in advance which of them the caller will actually differentiate through.
    Unlike the torch layer (which only computes the sensitivities PyTorch's autograd graph says
    are needed), this can trigger an acados sensitivity computation the OCP wasn't configured for
    (e.g. ``p_global`` gradients on an OCP built without ``with_solution_sens_wrt_params``) even
    if the caller only differentiates w.r.t. ``x0``.

    Args:
        fun: Framework-agnostic ``DiffFunction`` (e.g., ``AcadosDiffMpcFunction``).
        N_horizon: OCP horizon length.
        nx: State dimension.
        nu: Control dimension.

    Returns:
        A JAX-callable function whose gradient is defined by ``fun.backward``.
    """

    def _result_shapes(batch_size: int, dtype: Any) -> tuple[ShapedArray, ...]:
        return (
            ShapedArray((batch_size, nu), dtype),
            ShapedArray((batch_size, N_horizon + 1, nx), dtype),
            ShapedArray((batch_size, N_horizon, nu), dtype),
            ShapedArray((batch_size, 1), dtype),
        )

    def solve(
        x0: "jax.Array",
        u0: "jax.Array | None",
        p_global: "jax.Array",
        p_stagewise: "jax.Array",
        ctx: AcadosDiffMpcCtx | None = None,
    ) -> tuple[AcadosDiffMpcCtx, "jax.Array", "jax.Array", "jax.Array", "jax.Array"]:
        # Fresh per call: neither this box nor the custom_vjp function below are reused across
        # calls, so a forward pass here can never overwrite another call's context.
        ctx_box: dict[str, AcadosDiffMpcCtx] = {}

        @custom_vjp
        def _solve(x0, u0, p_global, p_stagewise):  # noqa: ANN001
            batch_size, dtype = x0.shape[0], x0.dtype

            def callback(x0_jax, u0_jax, p_global_jax, p_stagewise_jax):  # noqa: ANN001
                x0_np, u0_np, p_global_np, p_stagewise_np = _to_np_tuple(
                    x0_jax, u0_jax, p_global_jax, p_stagewise_jax
                )
                new_ctx, u_star, x, u, value = fun.forward(
                    ctx, x0_np, u0_np, p_global_np, p_stagewise_np, None
                )
                ctx_box["ctx"] = new_ctx
                return u_star, x, u, value

            return pure_callback(
                callback, _result_shapes(batch_size, dtype), x0, u0, p_global, p_stagewise
            )

        def _solve_fwd(x0, u0, p_global, p_stagewise):  # noqa: ANN001
            outputs = _solve(x0, u0, p_global, p_stagewise)
            # `u0 is not None` matters, not just "hardcode True": acados only supports a
            # `du0_dx0`-style sensitivity when `u0` was actually fixed (an equality constraint on
            # the initial control); requesting it while `u0` is the free variable (u0 is None)
            # raises inside acados. Unlike torch's autograd, `custom_vjp` can't tell us which
            # inputs the caller ultimately differentiates, so `x0`/`p_global` are always requested.
            ctx_box["ctx"].needs_input_grad = (False, True, u0 is not None, True, False, False)
            # `custom_vjp` residuals must be valid JAX pytrees, but `AcadosDiffMpcCtx` holds
            # numpy/solver-internal state that isn't one. Thread it to `_solve_bwd` via the
            # closure-captured `ctx_box` instead (safe here: fresh per `solve()` call, unlike a
            # module-level nonlocal shared across calls) and return an empty residual.
            return outputs, None

        def _solve_bwd(_residual: None, cotangents: tuple) -> tuple:
            np_cotangents = _to_np_tuple(*cotangents)
            grad_x0, grad_u0, grad_p_global, _, _ = fun.backward(ctx_box["ctx"], *np_cotangents)
            return _to_jax(grad_x0), _to_jax(grad_u0), _to_jax(grad_p_global), None

        _solve.defvjp(_solve_fwd, _solve_bwd)

        u_star, x, u, value = _solve(x0, u0, p_global, p_stagewise)
        return ctx_box["ctx"], u_star, x, u, value

    return solve
