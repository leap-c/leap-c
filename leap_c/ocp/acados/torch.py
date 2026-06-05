"""Central interface to use acados in PyTorch."""

from collections.abc import Sequence
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from leap_c.autograd.torch import create_autograd_function
from leap_c.ocp.acados.diff_mpc import (
    AcadosDiffMpcCtx,
    AcadosDiffMpcFunction,
    AcadosDiffMpcSensitivityOptions,
)
from leap_c.ocp.acados.diff_ocp import AcadosDiffOcp
from leap_c.ocp.acados.initializer import AcadosDiffMpcInitializer
from leap_c.planner import ParameterizedPlanner


class AcadosDiffMpcTorch(ParameterizedPlanner[AcadosDiffMpcCtx]):
    """PyTorch module for differentiable MPC based on acados.

    This module wraps acados solvers to enable their use in differentiable machine learning
    pipelines. It provides an autograd compatible forward method and supports sensitivity
    computation with respect to various inputs (see `AcadosDiffMpcCtx`).


    Attributes:
        diff_mpc_fun: The differentiable MPC function wrapper for acados.
        autograd_fun: A PyTorch autograd function created from `diff_mpc_fun`.
    """

    diff_mpc_fun: AcadosDiffMpcFunction
    autograd_fun: type[torch.autograd.Function]

    def __init__(
        self,
        ocp: AcadosDiffOcp,
        initializer: AcadosDiffMpcInitializer | None = None,
        discount_factor: float | None = None,
        export_directory: Path | None = None,
        n_batch_init: int | None = None,
        num_threads_batch_solver: int | None = None,
        dtype: torch.dtype | None = None,
        verbose: bool = True,
    ) -> None:
        """Initializes the AcadosDiffMpcTorch module.

        Args:
            ocp: The acados ocp object defining the optimal control problem structure.
            initializer: The initializer used to provide initial guesses for the solver, if none are
                provided explicitly or on a retry. Uses a zero iterate by default.
            discount_factor: An optional discount factor for the sensitivity problem.
                If none is provided, the default acados weighting will be used, i.e., `1/N_horizon`
                on the stage cost and `1` on the terminal cost.
            export_directory: An optional directory to which the generated C code will be exported.
                If none is provided, a unique temporary directory will be created used.
            n_batch_init: Initially supported batch size of the batch OCP solver.
                Using larger batches will trigger a delay for the creation of more solvers.
                If `None`, a default value is used.
            num_threads_batch_solver: Number of parallel threads to use for the batch OCP solver.
                If `None`, a default value is used.
            dtype: The output of the forward pass will automatically be cast to this type.
                If `None`, the default PyTorch dtype is used.
            verbose: Whether to print the output while generating solvers.

        """
        super().__init__()
        self.diff_mpc_fun = AcadosDiffMpcFunction(
            ocp=ocp,
            initializer=initializer,
            discount_factor=discount_factor,
            export_directory=export_directory,
            n_batch_init=n_batch_init,
            num_threads_batch_solver=num_threads_batch_solver,
            verbose=verbose,
        )
        self.parameter_manager = ocp.parameter_manager
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
        param: torch.Tensor | None = None,
        params: dict[str, torch.Tensor | np.ndarray] | None = None,
        ctx: AcadosDiffMpcCtx | None = None,
    ) -> tuple[AcadosDiffMpcCtx, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Performs the forward pass by solving the provided problem instances.

        In the background, PyTorch builds a computational graph that can be used for
        backpropagation. The context `ctx` is used to store intermediate values required for the
        backward pass, warmstart the solver for subsequent calls and to compute specific
        sensitivities.

        Args:
            x0: Initial states with shape `(B, x_dim)`.
            u0: Initial actions with shape `(B, u_dim)`. Defaults to `None`.
            param: Learnable flat parameter tensor, shape `(B, p_global_dim)`.
            params: A dictionary containing named parameter overrides. Values may be
                torch tensors or numpy arrays.
            ctx: An object for storing context. If provided, it will be used to warmstart the solve
                (e.g., by using the saved iterate). Defaults to `None`.

        Returns:
            ctx: A new context object from solving the problems.
            u0: Solution of initial control input.
            x: The solution of the whole state trajectory.
            u: The solution of the whole control trajectory.
            value: The cost value of the computed trajectory.
        """
        batch_size = x0.shape[0]
        device = x0.device
        dtype = x0.dtype

        # 1. Build p_global_tensor (learnable parameters)
        if param is not None:
            # Mode 1: Direct flat learnable tensor passed
            p_global_tensor = param
        else:
            # Mode 2: Build p_global from defaults and apply any overrides from params dict
            default_flat = torch.from_numpy(self.parameter_manager.learnable_default_flat).to(
                device=device, dtype=dtype
            )
            p_global_tensor = default_flat.unsqueeze(0).expand(batch_size, -1).clone()

            if params is not None:
                from leap_c.ocp.acados.parameters import _define_starts_and_ends

                # Overwrite learnable parameters from the dictionary
                for name in self.parameter_manager.global_parameter_names:
                    if name in params:
                        val = params[name]
                        if isinstance(val, np.ndarray):
                            val = torch.from_numpy(val).to(device=device, dtype=dtype)
                        param_def = self.parameter_manager.parameters[name]

                        if param_def.splits != "global":
                            # Stage-varying learnable
                            starts, ends = _define_starts_and_ends(
                                splits=param_def.splits,
                                N_horizon=self.parameter_manager.N_horizon,
                            )
                            val_reshaped = val.reshape(
                                batch_size, self.parameter_manager.N_horizon + 1, -1
                            )
                            for start, end in zip(starts, ends):
                                s, e = self.parameter_manager._learnable_parameter_store.indices[
                                    f"{name}_{start}_{end}"
                                ]
                                p_global_tensor[:, s:e] = val_reshaped[:, start, :]
                        else:
                            # Non-stage-varying learnable
                            s, e = self.parameter_manager._learnable_parameter_store.indices[name]
                            p_global_tensor[:, s:e] = val.reshape(batch_size, -1)

        # 2. Build p_stagewise_tensor (non-learnable parameters)
        non_learnable_overrides: dict[str, np.ndarray] = {}
        if params is not None:
            # Extract non-learnable parameters from the dictionary
            for name in self.parameter_manager.stagewise_parameter_names:
                if name in params:
                    val = params[name]
                    if isinstance(val, torch.Tensor):
                        val = val.detach().cpu().numpy()
                    non_learnable_overrides[name] = val

        p_stagewise_np = self.parameter_manager.combine_non_learnable_parameter_values(
            batch_size=batch_size, **non_learnable_overrides
        )
        p_stagewise_tensor = torch.from_numpy(np.array(p_stagewise_np, copy=True)).to(
            device=device, dtype=dtype
        )
        p_stagewise_sparse_idx = None

        ctx, u_star, x, u, value = self.autograd_fun.apply(
            ctx,
            x0,
            u0,
            p_global_tensor,  # type:ignore
            p_stagewise_tensor,
            p_stagewise_sparse_idx,
        )
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
