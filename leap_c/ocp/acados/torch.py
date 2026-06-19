"""Central interface to use acados in PyTorch."""

from collections.abc import Sequence
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from acados_template import AcadosOcp

from leap_c.autograd.torch import create_autograd_function
from leap_c.ocp.acados.diff_mpc import (
    AcadosDiffMpcCtx,
    AcadosDiffMpcFunction,
    AcadosDiffMpcSensitivityOptions,
)
from leap_c.ocp.acados.initializer import AcadosDiffMpcInitializer
from leap_c.ocp.acados.parameters import (
    AcadosParameterManager,
    _define_starts_and_ends,
)
from leap_c.planner import ParameterizedPlanner


class AcadosParameterManagerTorch(AcadosParameterManager):
    """Parameter manager using torch operations for differentiable parameter combination.

    Uses ``torch.cat``, indexing, and reshaping — all differentiable — so gradients
    flow back to the original parameter tensors in the ``params`` dict.
    """

    def combine_learnable_parameters(
        self,
        batch_size: int | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        **overwrites: torch.Tensor | np.ndarray,
    ) -> torch.Tensor:
        """Combine learnable parameters into a flat tensor, preserving differentiability.

        Args:
            batch_size: Batch size. Required.
            device: Target device for the output tensor. Required.
            dtype: Target dtype for the output tensor. Required.
            **overwrites: Named parameter overrides as tensors or numpy arrays.

        Returns:
            Tensor of shape ``(batch_size, N_learnable)``.
        """
        inferred_batch_size = next(iter(overwrites.values())).shape[0] if overwrites else None

        if batch_size is not None and inferred_batch_size is not None:
            if batch_size != inferred_batch_size:
                raise ValueError(
                    f"Provided batch_size={batch_size} does not match "
                    f"inferred batch_size={inferred_batch_size} from overwrites."
                )

        batch_size = inferred_batch_size if inferred_batch_size is not None else batch_size or 1

        batch_param = (
            torch.from_numpy(self.learnable_default_flat)
            .to(device, dtype)
            .expand(batch_size, -1)
            .clone()
        )

        if not overwrites:
            return batch_param

        for name, values in overwrites.items():
            if name not in self.parameters:
                raise ValueError(
                    f"Parameter '{name}' not found. "
                    f"Available parameters: {list(self.parameters.keys())}"
                )

            param = self.parameters[name]

            if param.interface != "learnable":
                raise ValueError(
                    f"Parameter '{name}' has interface '{param.interface}', "
                    "but only 'learnable' parameters can be used in this method."
                )

            if values.shape[0] != batch_size:
                raise ValueError(
                    f"Parameter '{name}' values have batch size {values.shape[0]}, "
                    f"but expected {batch_size}."
                )

            if param.is_stage_varying:
                Np1 = self.N_horizon + 1
                if isinstance(param.splits, list):
                    Np1 = param.splits[-1] + 1
                if values.shape[1] != Np1:
                    raise ValueError(
                        f"Parameter '{name}' is stage-varying and requires shape "
                        f"(batch_size, {Np1}, ...), but got shape {values.shape}."
                    )

        batch_param.requires_grad_()

        for name in self.learnable_parameter_names:
            param = self.parameters[name]
            if name in overwrites:
                val = overwrites[name]
                if isinstance(val, np.ndarray):
                    val = torch.from_numpy(val).to(device=device, dtype=dtype)
                elif isinstance(val, torch.Tensor):
                    val = val.to(device=device, dtype=dtype)
            else:
                val = None

            if param.is_stage_varying:
                starts, ends = _define_starts_and_ends(
                    splits=param.splits, N_horizon=self.N_horizon
                )
                for start, end in zip(starts, ends):
                    key = f"{name}_{start}_{end}"
                    s, e = self._learnable_parameter_store.indices[key]
                    if val is not None:
                        batch_param = torch.cat(
                            [
                                batch_param[:, :s],
                                val[:, start].reshape(batch_size, -1),
                                batch_param[:, e:],
                            ],
                            dim=-1,
                        )
            else:
                s, e = self._learnable_parameter_store.indices[name]
                if val is not None:
                    batch_param = torch.cat([batch_param[:, :s], val, batch_param[:, e:]], dim=-1)

        return batch_param


class AcadosDiffMpcTorch(ParameterizedPlanner[AcadosDiffMpcCtx]):
    """PyTorch module for differentiable MPC based on acados.

    This module wraps acados solvers to enable their use in differentiable machine learning
    pipelines. It provides an autograd compatible forward method and supports sensitivity
    computation with respect to various inputs (see `AcadosDiffMpcCtx`).

    Accepts a plain ``AcadosOcp`` together with a parameter manager.  The parameter manager's
    :meth:`~AcadosParameterManagerTorch.combine_learnable_parameters` is called in the forward pass
    to build a flat differentiable tensor from the ``params`` dict.

    Attributes:
        diff_mpc_fun: The differentiable MPC function wrapper for acados.
        autograd_fun: A PyTorch autograd function created from ``diff_mpc_fun``.
        parameter_manager: The parameter manager instance.
    """

    diff_mpc_fun: AcadosDiffMpcFunction
    autograd_fun: type[torch.autograd.Function]
    parameter_manager: AcadosParameterManagerTorch

    def __init__(
        self,
        ocp: AcadosOcp,
        parameter_manager: AcadosParameterManagerTorch,
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

    @property
    def param_space(self) -> gym.Space:
        """Describes the parameter space of the planner."""
        return self.parameter_manager.get_param_space()

    def default_param(self, obs: np.ndarray | None = None) -> np.ndarray:
        """Provides a default parameter configuration for the planner."""
        default = self.parameter_manager.learnable_default_flat
        if obs is None or obs.ndim <= 1:
            return default
        return np.broadcast_to(default, (*obs.shape[:-1], default.size))

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
        if params:
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
        p_global = self.parameter_manager.combine_learnable_parameters(
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

    def sensitivity(self, ctx, field_name: AcadosDiffMpcSensitivityOptions) -> np.ndarray:
        """Retrieves a specific sensitivity field from the context object.

        Args:
            ctx: The ctx object generated by the forward pass.
            field_name: The name of the sensitivity field to retrieve.
        """
        return self.diff_mpc_fun.sensitivity(ctx, field_name)
