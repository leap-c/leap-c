"""Provides stochastic MPC actors."""

from dataclasses import dataclass, field
from math import prod
from typing import Any, Generic, Literal, NamedTuple, Self

import gymnasium as gym
import gymnasium.spaces as spaces
import torch
import torch.nn as nn

from leap_c.controller import CtxType, ParameterizedController
from leap_c.torch.nn.bounded_distributions import (
    BoundedDistribution,
    BoundedDistributionName,
    get_bounded_distribution,
)
from leap_c.torch.nn.extractor import Extractor, ExtractorName, get_extractor_cls
from leap_c.torch.nn.mlp import Mlp, MlpConfig
from leap_c.utils.gym import flatten_param_space


class StochasticMPCActorOutput(NamedTuple):
    """Output of the hierarchical MPC actor's forward pass.

    Attributes:
        param: The predicted parameters (which have been input into the controller).
        log_prob: The log-probability of the distribution that led to the action. Can be either from
          the parameter or action distribution, depending on the noise mode.
        stats: A dictionary containing statistics from internal modules.
        action: The action output by the controller (None if only_param=True).
        status: The status of the MPC solver (0 if successful).
        ctx: The context object containing information about the MPC solve.
    """

    param: torch.Tensor
    log_prob: torch.Tensor
    stats: dict[str, float]
    action: torch.Tensor | None = None
    status: torch.Tensor | None = None
    ctx: CtxType | None = None

    def __post_init__(self) -> None:
        if torch.isnan(self.param).any():
            raise ValueError("NaN detected in StochasticMPCActorOutput.param")
        if torch.isnan(self.log_prob).any():
            raise ValueError("NaN detected in StochasticMPCActorOutput.log_prob")
        if self.action is not None and torch.isnan(self.action).any():
            raise ValueError("NaN detected in StochasticMPCActorOutput.action")

    def select(self, mask: torch.Tensor) -> Self:
        """Select a subset of the output based on the given mask. Discards stats and ctx."""
        return StochasticMPCActorOutput(
            self.param[mask],
            self.log_prob[mask],
            None,
            self.action[mask] if self.action is not None else None,
            self.status[mask] if self.status is not None else None,
            None,
        )


@dataclass(kw_only=True)
class HierachicalMPCActorConfig:
    """Configuration for HierachicalMPCActor.

    Attributes:
        noise: Where to inject noise - "param" for parameter space, "action" for action space.
        extractor_name: The name of the feature extractor to use.
        mlp: The configuration for the MLP predicting distribution parameters.
        distribution_name: The name of the bounded distribution for sampling.
            In "param" mode: distribution for sampling parameters.
            In "action" mode: distribution for sampling actions (params are deterministic).
        distribution_kwargs: Additional keyword arguments for the distribution constructor.
        residual: Whether to use residual learning (param = default_param + learned_param).
            Only applicable in parameter noise mode (noise="param").
        entropy_correction: Whether to apply entropy correction based on the Jacobian.
            Only applicable in parameter noise mode (noise="param").
    """

    noise: Literal["param", "action"] = "param"
    extractor_name: ExtractorName = "identity"
    mlp: MlpConfig = field(default_factory=MlpConfig)
    distribution_name: BoundedDistributionName = "squashed_gaussian"
    distribution_kwargs: dict[str, Any] = field(default_factory=dict)
    residual: bool = False
    entropy_correction: bool = False


class HierachicalMPCActor(nn.Module, Generic[CtxType]):
    """Hierarchical MPC actor.

    Implements a two-level hierarchy:
    1. High-level policy predicts controller parameters
    2. Low-level controller computes actions from parameters

    Noise can be injected at either level, controlled by action_distribution_name:

    Parameter noise mode (noise="param"):
        - Stochastic parameters → Deterministic controller → Deterministic actions
        - Noise is added to controller parameters before execution
        - Supports residual learning (parameters as offsets from defaults)
        - Supports entropy correction based on action-to-parameter Jacobian

    Action noise mode (noise="action"):
        - Deterministic parameters → Deterministic controller → Stochastic actions
        - Controller parameters are predicted deterministically
        - Noise is added to controller output actions
        - For Beta distribution: MPC action serves as the anchor/mode

    Attributes:
        controller: The parameterized controller (low-level policy).
        extractor: Feature extractor for observations.
        mlp: MLP that outputs distribution parameters (high-level policy).
        param_distribution: Bounded distribution for parameters.
        action_distribution: Bounded distribution for actions (None in parameter noise mode).
        residual: Whether to use residual learning (parameter noise mode only).
        entropy_correction: Whether to apply Jacobian-based entropy correction
            (parameter noise mode only).
    """

    controller: ParameterizedController[CtxType]
    extractor: Extractor
    mlp: Mlp
    param_distribution: BoundedDistribution
    action_distribution: BoundedDistribution | None
    residual: bool
    entropy_correction: bool

    def __init__(
        self,
        cfg: HierachicalMPCActorConfig,
        observation_space: gym.Space,
        action_space: gym.Space,
        controller: ParameterizedController[CtxType],
    ) -> None:
        """Initialize HierachicalMPCActor.

        Args:
            cfg: Configuration for the hierarchical actor.
            observation_space: The observation space.
            action_space: The action space.
            controller: The parameterized controller (low-level policy).
        """
        super().__init__()

        self.controller = controller
        self.residual = cfg.residual and cfg.noise == "param"
        self.entropy_correction = cfg.entropy_correction and cfg.noise == "param"

        param_space: spaces.Box = flatten_param_space(controller.param_space)
        param_dim = param_space.shape[0]
        action_dim = prod(action_space.shape)

        # create extractor
        extractor_cls = get_extractor_cls(cfg.extractor_name)
        self.extractor = extractor_cls(observation_space)

        # create distributions based on noise mode
        if cfg.noise == "param":
            # parameter noise: distribution for parameters
            self.param_distribution = get_bounded_distribution(
                cfg.distribution_name, space=param_space, **cfg.distribution_kwargs
            )

            self.action_distribution = None
            # MLP outputs: param distribution parameters
            output_sizes = list(self.param_distribution.parameter_size(param_dim))
        else:
            # action noise: deterministic param transform + action distribution
            # Note: param_distribution is used deterministically
            self.param_distribution = get_bounded_distribution(
                "squashed_gaussian", space=param_space, **cfg.distribution_kwargs
            )
            self.action_distribution = get_bounded_distribution(
                cfg.distribution_name, space=action_space, **cfg.distribution_kwargs
            )
            # MLP outputs: (param_mean, action_dist_params...)
            action_param_size = self.action_distribution.parameter_size(action_dim)
            output_sizes = (param_dim,) + action_param_size

        self.cfg = cfg

        self.mlp = Mlp(
            input_sizes=self.extractor.output_size,
            output_sizes=output_sizes,
            mlp_cfg=cfg.mlp,
        )

    # TODO (Mazen): this should also be removed, the actors should produce a dictionary instead of
    # flat parameters.
    def _param_to_dict(self, flat_param: torch.Tensor) -> dict[str, torch.Tensor]:
        """Convert a flat parameter tensor to a dict for the dict-based controller API.

        The flat parameter tensor is the output of the distribution (RL system).
        This method splits it into named entries expected by the controller, using
        the planner's parameter manager to determine the structure.

        Args:
            flat_param: Flat parameter tensor ``(batch_size, total_learnable_dim)``.

        Returns:
            Dict mapping parameter names to their values. Stage-varying parameters
            get shape ``(batch_size, N_horizon + 1, pdim)``; global parameters
            get shape ``(batch_size, pdim)``.
        """
        planner = getattr(self.controller, "planner", None)
        if planner is None:
            raise TypeError(
                f"Cannot convert flat parameters to dict: controller "
                f"({type(self.controller).__name__}) has no `.planner` attribute."
            )
        manager = getattr(planner, "param_manager", None)
        if manager is None:
            raise TypeError(
                f"Cannot convert flat parameters to dict: planner "
                f"({type(planner).__name__}) has no `param_manager` attribute."
            )

        store = manager._learnable_parameter_store
        Np1 = manager.N_horizon + 1
        batch_size = flat_param.shape[0]

        param_dict = {}
        for name in manager.learnable_parameter_names:
            param_def = manager.parameters[name]
            pdim = param_def.default.size

            if param_def.is_stage_varying:
                val = torch.zeros(
                    batch_size,
                    Np1,
                    pdim,
                    dtype=flat_param.dtype,
                    device=flat_param.device,
                )
                for stored_name, (si, ei) in store.indices.items():
                    if stored_name.startswith(f"{name}_"):
                        suffix = stored_name[len(name) + 1 :]
                        start, end = (int(x) for x in suffix.split("_"))
                        block_val = flat_param[..., si:ei].reshape(batch_size, pdim)
                        val[:, start : end + 1, :] = block_val.unsqueeze(1)
                param_dict[name] = val
            else:
                si, ei = store.indices[name]
                param_dict[name] = flat_param[..., si:ei]

        return param_dict

    def forward(
        self,
        obs: torch.Tensor,
        ctx: CtxType | None = None,
        deterministic: bool = False,
        only_param: bool = False,
    ) -> StochasticMPCActorOutput:
        """Sample parameters/actions from the hierarchical policy.

        Args:
            obs: Observations.
            ctx: Optional controller context for warm-starting the low-level controller.
            deterministic: If True, use deterministic mode (no sampling from distributions).
            only_param: If True, only return parameters without computing actions.
                Only applicable in parameter noise mode.

        Returns:
            Actor output containing parameters, log probabilities, statistics, and actions.
        """
        e = self.extractor(obs)

        if self.action_distribution is None:
            # parameter noise mode
            dist_params = self.mlp(e)
            anchor = self.controller.default_param(obs) if self.residual else None

            param, log_prob, stats = self.param_distribution(
                *dist_params,
                deterministic=deterministic,
                anchor=anchor,
            )

            if only_param:
                return StochasticMPCActorOutput(param, log_prob, stats)

            # get action from controller
            param_dict = self._param_to_dict(param)
            ctx, action = self.controller(obs, param_dict, ctx=ctx)

            # Store distribution info in context for rendering/debugging
            ctx.param_distribution_info = {
                "distribution": self.param_distribution,
                "dist_params": dist_params,
                "deterministic": deterministic,
                "anchor": anchor,
            }

            # apply entropy correction if enabled
            if self.entropy_correction:
                # NOTE: Computing the full Jacobian, but we only need the diagonal.
                # Can be slow for large batches. Look into vectorized computations.
                j = torch.autograd.functional.jacobian(
                    lambda p: self.controller(obs, self._param_to_dict(p), ctx=ctx)[1],
                    param,
                )
                # j: (B, A, B, D) — extract per-sample Jacobians (B, A, D)
                b = param.shape[0]
                j = j[torch.arange(b), :, torch.arange(b), :]

                jtj = j @ j.transpose(1, 2)
                correction = (
                    torch.det(jtj + 1e-3 * torch.eye(jtj.shape[1], device=jtj.device)).sqrt().log()
                )
                log_prob -= correction.unsqueeze(1)

            # merge controller stats
            if ctx.log is not None:
                stats |= ctx.log

            return StochasticMPCActorOutput(param, log_prob, stats, action, ctx.status, ctx)

        # action noise mode
        param_mean, *action_dist_params = self.mlp(e)

        # transform parameters deterministically (no noise on params)
        param, _, param_stats = self.param_distribution(param_mean, deterministic=True, anchor=None)

        # get action from controller
        param_dict = self._param_to_dict(param)
        ctx, action_mpc = self.controller(obs, param_dict, ctx=ctx)

        # add noise to action - use MPC action as anchor
        action, log_prob, action_stats = self.action_distribution(
            *action_dist_params, deterministic=deterministic, anchor=action_mpc
        )

        # merge stats
        stats = param_stats | action_stats
        if ctx.log is not None:
            stats |= ctx.log

        return StochasticMPCActorOutput(param, log_prob, stats, action, ctx.status, ctx)
