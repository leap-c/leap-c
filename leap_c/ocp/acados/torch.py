"""Central interface to use acados in PyTorch."""

from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from acados_template import AcadosOcp

from leap_c.autograd.torch import create_autograd_function
from leap_c.ocp.acados.diff_mpc import (
    AcadosDiffMpcCtx,
    AcadosDiffMpcFunction,
)
from leap_c.ocp.acados.initializer import AcadosDiffMpcInitializer
from leap_c.ocp.acados.parameters import AcadosParameterManager

AcadosDiffMpcSensitivityOptions = Literal[
    "du0_dp_global",
    "dx_dp_global",
    "du_dp_global",
    "dvalue_dp_global",
    "dvalue_du0",
    "du0_dx0",
    "dvalue_dx0",
]


class AcadosDiffMpcTorch(torch.nn.Module):
    """PyTorch module for differentiable MPC based on acados.

    This module wraps acados solvers to enable their use in differentiable machine learning
    pipelines. It provides an autograd compatible forward method and supports sensitivity
    computation with respect to various inputs (see `AcadosDiffMpcCtx`).

    Accepts a plain ``AcadosOcp`` together with a parameter manager. The parameter manager's
    :meth:`~AcadosParameterManager.combine_learnable_parameters_torch` is called in the forward
    pass to build a flat differentiable tensor from the ``params`` dict.


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

        Builds flat ``p_global`` and ``p_stagewise`` arrays from the ``params`` dict using the
        parameter manager, then calls the underlying differentiable MPC function.  Gradients flow
        back through ``p_global`` to the individual parameter tensors provided in ``params``.

        Args:
            x0: Initial states with shape ``(B, x_dim)``.
            u0: Initial actions with shape ``(B, u_dim)``. Defaults to ``None``.
            params: A dictionary containing named parameter overrides.  Values may be
                torch tensors (learnable) or numpy arrays (non-learnable).
            ctx: An object for storing context. If provided, it will be used to warmstart the solve.

        Returns:
            ctx: A new context object from solving the problems.
            u0: Solution of initial control input.
            x: The solution of the whole state trajectory.
            u: The solution of the whole control trajectory.
            value: The cost value of the computed trajectory.
        """
        batch_size = x0.shape[0]
        device = x0.device

        # Separate learnable (tensor) and non-learnable (numpy) overrides
        learnable_overwrites: dict[str, torch.Tensor | np.ndarray] = {}
        non_learnable_overwrites: dict[str, np.ndarray] = {}
        if params is not None:
            for name in self.parameter_manager.learnable_parameter_names:
                if name in params:
                    learnable_overwrites[name] = params[name]
            for name in self.parameter_manager.non_learnable_parameter_names:
                if name in params:
                    val = params[name]
                    if isinstance(val, torch.Tensor):
                        val = val.detach().cpu().numpy()
                    non_learnable_overwrites[name] = val

        # Build flat p_global differentiably
        p_global = self.parameter_manager.combine_learnable_parameters_torch(
            batch_size=batch_size,
            device=device,
            dtype=x0.dtype,
            **learnable_overwrites,
        )

        # Build p_stagewise (non-learnable, no gradient needed)
        p_stagewise_np = self.parameter_manager.combine_non_learnable_parameters(
            batch_size=batch_size, **non_learnable_overwrites
        )
        p_stagewise = torch.from_numpy(p_stagewise_np).to(device=device, dtype=x0.dtype)

        ctx, u_star, x, u, value = self.autograd_fun.apply(ctx, x0, u0, p_global, p_stagewise, None)
        u_star = u_star.to(dtype=self.dtype)
        x = x.to(dtype=self.dtype)
        u = u.to(dtype=self.dtype)
        value = value.to(dtype=self.dtype)
        return ctx, u_star, x, u, value  # type:ignore

    # TODO (Dirk): This is probably not needed anymore...
    def set_constraint_bounds(
        self,
        lbx: np.ndarray,
        ubx: np.ndarray,
        stages: Sequence[int],
    ) -> None:
        """Set lbx/ubx constraints on the forward solvers for the given stages.

        Args:
            lbx: Lower bounds, shape ``(batch_size, len(stages), constraint_dim)``.
            ubx: Upper bounds, shape ``(batch_size, len(stages), constraint_dim)``.
            stages: Stage indices at which to apply the bounds.
        """
        solvers = self.diff_mpc_fun.forward_batch_solver.ocp_solvers
        batch_size = lbx.shape[0]
        for i in range(batch_size):
            solver = solvers[i]
            for j, stage in enumerate(stages):
                solver.constraints_set(stage, "lbx", lbx[i, j])
                solver.constraints_set(stage, "ubx", ubx[i, j])
