"""Provides stochastic MPC actors."""

from dataclasses import dataclass, field
from typing import Generic, Literal, NamedTuple, Self

import gymnasium as gym
import gymnasium.spaces as spaces
import numpy as np
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
        residual: Whether to use residual learning (param = default_param + learned_param).
            Only applicable in parameter noise mode (noise="param").
        entropy_correction: Whether to apply entropy correction based on the Jacobian.
            Only applicable in parameter noise mode (noise="param").
    """

    noise: Literal["param", "action"] = "param"
    extractor_name: ExtractorName = "identity"
    mlp: MlpConfig = field(default_factory=MlpConfig)
    distribution_name: BoundedDistributionName = "squashed_gaussian"
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

        param_space: spaces.Box = controller.param_space
        param_dim = param_space.shape[0]
        action_dim = np.prod(action_space.shape)

        # create extractor
        extractor_cls = get_extractor_cls(cfg.extractor_name)
        self.extractor = extractor_cls(observation_space)

        # create distributions based on noise mode
        if cfg.noise == "param":
            # parameter noise: distribution for parameters
            self.param_distribution = get_bounded_distribution(
                cfg.distribution_name, space=param_space
            )

            self.action_distribution = None
            # MLP outputs: param distribution parameters
            output_sizes = list(self.param_distribution.parameter_size(param_dim))
        else:
            # action noise: deterministic param transform + action distribution
            # Note: param_distribution is used deterministically by passing log_std=None
            # in forward(), which transforms the mean without adding noise
            self.param_distribution = get_bounded_distribution(
                "squashed_gaussian", space=param_space
            )
            self.action_distribution = get_bounded_distribution(
                cfg.distribution_name, space=action_space
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
            ctx, action = self.controller(obs, param, ctx=ctx)

            # Store distribution info in context for rendering/debugging
            ctx.param_distribution_info = {
                "distribution": self.param_distribution,
                "dist_params": dist_params,
                "deterministic": deterministic,
                "anchor": anchor,
            }

            # apply entropy correction if enabled
            if self.entropy_correction:
                j = self.controller.jacobian_action_param(ctx)
                j = torch.from_numpy(j).to(param.device)
                jtj = j @ j.transpose(1, 2)
                correction = (
                    torch.det(jtj + 1e-3 * torch.eye(jtj.shape[1], device=jtj.device)).sqrt().log()
                )
                log_prob -= correction.unsqueeze(1)

            # merge controller stats
            if ctx.log is not None:
                stats = stats | ctx.log

            return StochasticMPCActorOutput(param, log_prob, stats, action, ctx.status, ctx)

        # action noise mode
        mlp_outputs = self.mlp(e)
        param_mean = mlp_outputs[0]
        action_dist_params = mlp_outputs[1:]

        # transform parameters deterministically (no noise on params)
        # log_std=None makes the distribution deterministic (just applies tanh + scaling)
        param, _, param_stats = self.param_distribution(param_mean, log_std=None, anchor=None)

        # get action from controller
        ctx, action_mpc = self.controller(obs, param, ctx=ctx)

        # add noise to action - use MPC action as anchor
        action, log_prob, action_stats = self.action_distribution(
            *action_dist_params, deterministic=deterministic, anchor=action_mpc
        )

        # merge stats
        stats = param_stats | action_stats
        if ctx.log is not None:
            stats = stats | ctx.log

        return StochasticMPCActorOutput(param, log_prob, stats, action, ctx.status, ctx)
