"""Central interface to use acados in PyTorch."""

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from acados_template import AcadosOcp

from leap_c.autograd.torch import create_autograd_function
from leap_c.diff_mpc.data import validate_forward_inputs
from leap_c.diff_mpc.function import (
    AcadosDiffMpcCtx,
    AcadosDiffMpcFunction,
)
from leap_c.diff_mpc.initializer import AcadosDiffMpcInitializer
from leap_c.parameters import AcadosParameterManager
from leap_c.utils.dependencies import require_torch
from leap_c.utils.repr import (
    format_diff_mpc_module_extra_repr,
    format_diff_mpc_module_repr,
)

if TYPE_CHECKING:
    import torch
else:
    torch = require_torch()


class AcadosDiffMpcTorch(torch.nn.Module):
    """PyTorch module for differentiable MPC based on acados.

    This module wraps acados solvers to enable their use in end-to-end machine learning
    pipelines. It provides an autograd compatible forward method and supports sensitivity
    computation with respect to various inputs.

    Accepts a plain ``AcadosOcp`` together with a parameter manager. The parameter manager's
    :meth:`~AcadosParameterManager.combine_differentiable_parameters_torch` is called in the forward
    pass to build a flat differentiable tensor from the ``params`` dict.

    Examples:
        Forward pass with differentiable parameters; gradients flow back through ``params``::

            >>> diff_mpc = AcadosDiffMpcTorch(ocp, manager)
            >>> ctx, u_star, x, u, value = diff_mpc(x0, params={"cost_weight": w})
            >>> value.sum().backward()
            >>> w.grad  # d(value)/d(cost_weight)

        Sensitivities via :func:`torch.autograd.functional.jacobian`::

            >>> jac = torch.autograd.functional.jacobian(lambda x0: diff_mpc(x0)[1], x0)

    Attributes:
        diff_mpc_fun: The differentiable MPC function wrapper for acados.
        autograd_fun: A PyTorch autograd function created from ``diff_mpc_fun``.
        parameter_manager: The parameter manager instance.
    """

    diff_mpc_fun: AcadosDiffMpcFunction
    autograd_fun: type[torch.autograd.Function]
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
        dtype: torch.dtype | None = None,
        verbose: bool = True,
    ) -> None:
        """Initializes the AcadosDiffMpcTorch module.

        Calls ``parameter_manager.assign_to_ocp(ocp)`` to synchronise CasADi symbols and default
        values onto the OCP, then creates the solvers.

        Args:
            ocp: The acados OCP object.  Must not yet have ``model.p`` / ``model.p_global`` set
                (they will be set by ``assign_to_ocp``).
            parameter_manager: A parameter manager with registered parameters.
            initializer: The initializer used to provide initial guesses for the solver.
                Uses a zero iterate by default.
            discount_factor: An optional discount factor for the sensitivity problem.
            export_directory: An optional directory for generated C code.
            n_batch_init: Initially supported batch size.  If ``None``, a default is used.
            num_threads_batch_solver: Number of parallel threads for the batch solver.
            dtype: Output dtype.  Defaults to ``torch.get_default_dtype()``.
            verbose: Whether to print solver generation output.

        """
        super().__init__()
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
        self.autograd_fun = create_autograd_function(self.diff_mpc_fun)
        self.dtype = torch.get_default_dtype() if dtype is None else dtype

    def forward(
        self,
        x0: torch.Tensor,
        u0: torch.Tensor | None = None,
        params: dict[str, torch.Tensor | np.ndarray] | None = None,
        ctx: AcadosDiffMpcCtx | None = None,
    ) -> tuple[AcadosDiffMpcCtx, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Performs the forward pass by solving the provided problem instances.

        Passing a ``ctx`` returned by a previous call warmstarts the solver: its saved iterate
        is reused as the initial guess, which can speed up convergence across sequential solves.
        When ``ctx`` is ``None``, the solver is initialized by the ``initializer`` provided at
        construction (a zero iterate by default).

        Setting ``u0`` fixates the first-stage control: instead of being a free decision
        variable, the first control is constrained to ``u0``. In that case the returned
        ``u_star`` equals ``u0`` and ``value`` is the cost of the constrained trajectory
        (a Q-value) rather than the optimal value (a V-value).

        Args:
            x0: Initial states with shape ``(B, x_dim)``.
            u0: Initial actions with shape ``(B, u_dim)``. Defaults to ``None``.
            params: A dictionary containing named parameter overrides.  Values may be
                torch tensors (differentiable) or numpy arrays (non-differentiable).
            ctx: Context from a previous solve, used to warmstart. Defaults to ``None``.

        Returns:
            ctx: A new context object from solving the problems.
            u0: Solution of initial control input.
            x: The solution of the whole state trajectory.
            u: The solution of the whole control trajectory.
            value: The cost value of the computed trajectory.

        Examples:
            Warmstart a subsequent solve by feeding the previous ``ctx`` back in::

                >>> ctx, u_star, x, u, value = diff_mpc(x0)
                >>> ctx, u_star, x, u, value = diff_mpc(x0, ctx=ctx)

            Fixate the first-stage control with ``u0``::

                >>> ctx, u_star, x, u, value = diff_mpc(x0, u0=u0)
                >>> torch.allclose(u_star, u0)  # True (up to dtype cast)
        """
        validate_forward_inputs(self.diff_mpc_fun.ocp, x0, u0)
        batch_size = x0.shape[0]
        device = x0.device

        differentiable_overwrites: dict[str, torch.Tensor | np.ndarray] = {}
        non_differentiable_overwrites: dict[str, np.ndarray] = {}
        if params is not None:
            for name in self.parameter_manager.differentiable_parameter_names:
                if name in params:
                    differentiable_overwrites[name] = params[name]
            for name in self.parameter_manager.non_differentiable_parameter_names:
                if name in params:
                    val = params[name]
                    if isinstance(val, torch.Tensor) and val.requires_grad:
                        raise ValueError(
                            f"Parameter '{name}' was registered as non-differentiable "
                            "but received a tensor with requires_grad=True. "
                            "Either register it with differentiable=True, or pass "
                            ".detach() to suppress this error."
                        )
                    if isinstance(val, torch.Tensor):
                        val = val.detach().cpu().numpy()
                    non_differentiable_overwrites[name] = val

        p_global = self.parameter_manager.combine_differentiable_parameters_torch(
            batch_size=batch_size,
            device=device,
            dtype=x0.dtype,
            **differentiable_overwrites,
        )

        p_stagewise_np = self.parameter_manager.combine_non_differentiable_parameters(
            batch_size=batch_size, **non_differentiable_overwrites
        )
        p_stagewise = torch.from_numpy(p_stagewise_np).to(device=device, dtype=x0.dtype)

        ctx, u_star, x, u, value = self.autograd_fun.apply(ctx, x0, u0, p_global, p_stagewise, None)
        u_star = u_star.to(dtype=self.dtype)
        x = x.to(dtype=self.dtype)
        u = u.to(dtype=self.dtype)
        value = value.to(dtype=self.dtype)
        return ctx, u_star, x, u, value  # type:ignore

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
