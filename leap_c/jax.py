"""Central interface to use acados in JAX."""

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from acados_template import AcadosOcp

from leap_c.autograd.jax import create_jax_callable
from leap_c.diff_mpc.data import validate_forward_inputs
from leap_c.diff_mpc.function import AcadosDiffMpcCtx, AcadosDiffMpcFunction
from leap_c.diff_mpc.initializer import AcadosDiffMpcInitializer
from leap_c.parameters import AcadosParameterManager
from leap_c.utils.dependencies import require_jax
from leap_c.utils.repr import (
    format_diff_mpc_module_extra_repr,
    format_diff_mpc_module_repr,
)

if TYPE_CHECKING:
    import jax
else:
    jax = require_jax()
import jax.numpy as jnp  # noqa: E402


class AcadosDiffMpcJax:
    """JAX wrapper for differentiable MPC based on acados.

    This class wraps acados solvers to enable their use in differentiable JAX
    pipelines. It uses ``jax.custom_vjp`` together with ``jax.pure_callback`` to isolate
    the impure C solver. The resulting callable is compatible with ``jax.grad`` (but not
    ``jax.jit``-of-``jax.grad`` — see :mod:`leap_c.autograd.jax`).

    Accepts a plain ``AcadosOcp`` together with a parameter manager. The parameter manager's
    :meth:`~AcadosParameterManager.combine_differentiable_parameters_jax` is called in the forward
    pass to build a flat differentiable array from the ``params`` dict.

    .. warning::
        acados solves in double precision. JAX defaults to 32-bit arrays, so unless
        ``jax.config.update("jax_enable_x64", True)`` is set, ``x0``/outputs will silently lose
        precision relative to the torch layer (which defaults to ``torch.get_default_dtype()``,
        commonly float64). Enable x64 for numerically comparable results.

    Examples:
        Forward pass with differentiable parameters; gradients flow back through ``params``::

            >>> diff_mpc = AcadosDiffMpcJax(ocp, manager)
            >>> ctx, u0, x, u, value = diff_mpc(x0, params={"cost_weight": w})
            >>> jax.grad(lambda w: diff_mpc(x0, params={"cost_weight": w})[-1].sum())(w)

    Attributes:
        diff_mpc_fun: The differentiable MPC function wrapper for acados.
        parameter_manager: The parameter manager instance.
    """

    diff_mpc_fun: AcadosDiffMpcFunction
    parameter_manager: AcadosParameterManager

    def __init__(
        self,
        ocp: AcadosOcp,
        parameter_manager: AcadosParameterManager,
        initializer: AcadosDiffMpcInitializer | None = None,
        discount_factor: float | None = None,
        export_directory: Path | None = None,
        n_batch_init: int | None = None,
        num_threads_batch_solver: int | None = None,
        verbose: bool = True,
    ) -> None:
        """Initializes the AcadosDiffMpcJax module.

        Calls ``parameter_manager.assign_to_ocp(ocp)`` to synchronise CasADi symbols and default
        values onto the OCP, then creates the solvers.

        Args:
            ocp: The acados OCP object. Must not yet have ``model.p`` / ``model.p_global`` set
                (they will be set by ``assign_to_ocp``).
            parameter_manager: A parameter manager with registered parameters.
            initializer: The initializer used to provide initial guesses for the solver.
                Uses a zero iterate by default.
            discount_factor: An optional discount factor for the sensitivity problem.
            export_directory: An optional directory for generated C code.
            n_batch_init: Initially supported batch size. If ``None``, a default is used.
            num_threads_batch_solver: Number of parallel threads for the batch solver.
            verbose: Whether to print solver generation output.
        """
        parameter_manager.assign_to_ocp(ocp)
        self.diff_mpc_fun = AcadosDiffMpcFunction(
            ocp=ocp,
            initializer=initializer,
            discount_factor=discount_factor,
            export_directory=export_directory,
            n_batch_init=n_batch_init,
            num_threads_batch_solver=num_threads_batch_solver,
            verbose=verbose,
        )
        self.parameter_manager = parameter_manager
        self._solve_fn = create_jax_callable(
            self.diff_mpc_fun,
            N_horizon=ocp.solver_options.N_horizon,
            nx=ocp.dims.nx,
            nu=ocp.dims.nu,
        )

    def __call__(
        self,
        x0: "jax.Array",
        u0: "jax.Array | None" = None,
        params: dict[str, "jax.Array | np.ndarray"] | None = None,
        ctx: AcadosDiffMpcCtx | None = None,
    ) -> tuple[AcadosDiffMpcCtx, "jax.Array", "jax.Array", "jax.Array", "jax.Array"]:
        """Performs the forward pass by solving the provided problem instances.

        Passing a ``ctx`` returned by a previous call warmstarts the solver: its saved iterate is
        reused as the initial guess. When ``ctx`` is ``None``, the solver is initialized by the
        ``initializer`` provided at construction (a zero iterate by default).

        Setting ``u0`` fixates the first-stage control: instead of being a free decision
        variable, the first control is constrained to ``u0``. In that case the returned ``u0``
        matches the input ``u0`` and ``value`` is the cost of the constrained trajectory (a
        Q-value) rather than the optimal value (a V-value).

        Args:
            x0: Initial states with shape ``(B, x_dim)``.
            u0: Initial actions with shape ``(B, u_dim)``. Defaults to ``None`` (free variable).
            params: A dictionary containing named parameter overrides. Values may be JAX
                arrays (differentiable) or numpy arrays (non-differentiable).
            ctx: Context from a previous solve, used to warmstart. Defaults to ``None``.

        Returns:
            ctx: A new context object from solving the problems.
            u0: Solution of initial control input, shape ``(B, u_dim)``.
            x: State trajectory, shape ``(B, N+1, x_dim)``.
            u: Control trajectory, shape ``(B, N, u_dim)``.
            value: Cost value, shape ``(B, 1)``.
        """
        validate_forward_inputs(self.diff_mpc_fun.ocp, x0, u0)
        batch_size = x0.shape[0]

        differentiable_overwrites: dict[str, "jax.Array | np.ndarray"] = {}
        non_differentiable_overwrites: dict[str, np.ndarray] = {}
        if params is not None:
            for name in self.parameter_manager.differentiable_parameter_names:
                if name in params:
                    differentiable_overwrites[name] = params[name]
            for name in self.parameter_manager.non_differentiable_parameter_names:
                if name in params:
                    val = params[name]
                    non_differentiable_overwrites[name] = (
                        val if isinstance(val, np.ndarray) else np.asarray(val)
                    )

        p_global = self.parameter_manager.combine_differentiable_parameters_jax(
            batch_size=batch_size,
            **differentiable_overwrites,
        )

        p_stagewise_np = self.parameter_manager.combine_non_differentiable_parameters(
            batch_size=batch_size, **non_differentiable_overwrites
        )
        p_stagewise = jnp.asarray(p_stagewise_np)

        return self._solve_fn(x0, u0, p_global, p_stagewise, ctx=ctx)

    def extra_repr(self) -> str:
        return format_diff_mpc_module_extra_repr(
            ocp=self.diff_mpc_fun.ocp, parameter_manager=self.parameter_manager
        )

    def __repr__(self) -> str:
        return format_diff_mpc_module_repr(
            class_name=type(self).__name__,
            ocp=self.diff_mpc_fun.ocp,
            parameter_manager=self.parameter_manager,
        )
