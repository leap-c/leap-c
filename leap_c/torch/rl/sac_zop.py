"""Provides a trainer for a SAC algorithm that sets parameters of a parameterized controller."""

from dataclasses import dataclass
from pathlib import Path
from typing import Generator, Generic, NamedTuple, Type

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
from leap_c.torch.nn.mlp import Mlp, MlpConfig, init_mlp_params_with_inverse_default
from leap_c.torch.rl.buffer import ReplayBuffer
from leap_c.torch.rl.sac import SacCritic, SacTrainerConfig
from leap_c.torch.rl.utils import soft_target_update
from leap_c.torch.utils.seed import mk_seed
from leap_c.trainer import Trainer
from leap_c.utils.gym import seed_env, wrap_env


class SacZopActorOutput(NamedTuple):
    """Output of the SAC-ZOP actor's forward pass.

    Attributes:
        param: The predicted parameters (which have been input into the controller).
        log_prob: The log-probability of the distribution that led to the action.
            NOTE: This log-probability is just a proxy for the true log-probability of the action,
            it is actually the log probability of the parameters that were input into the
            controller.
        stats: A dictionary containing several statistics of internal modules.
        action: The action output by the controller.
        ctx: The context object containing information about the MPC solve.
    """

    param: torch.Tensor
    log_prob: torch.Tensor
    stats: dict[str, float]
    action: torch.Tensor | None = None
    ctx: CtxType | None = None


@dataclass(kw_only=True)
class SacZopTrainerConfig(SacTrainerConfig):
    """Specific settings for the Zop trainer.

    Attributes:
        init_param_with_default: Whether to initialize the parameters of the controller such that
            the mean of the gaussian transformed by the squashing of the SquashedGaussian
            corresponds to the Parameter default values. Only works if
             1. the parameters are fixed nn.Parameters, and not predicted by a network
            (see MlpConfig hidden_dims).
             2. a SquashedGaussian distribution is used.

            If `True`, the default parameters according to `controller.default_param(None)` will be
            used, else the parameters will be initialized to the middle
            of the parameter bounds.
    """

    init_param_with_default: bool = True


class MpcSacActor(nn.Module, Generic[CtxType]):
    """An actor module for SAC-ZOP, containing a ParameterizedController.

    The ParameterizedController is used to compute actions, but does not need to support
    differentiating through it. Noise is injected in the parameter space.

    Attributes:
        extractor: The feature extractor module.
        controller: The parameterized controller.
        mlp: The MLP module that predicts the parameters of the Gaussian distribution.
        bounded_distribution: The bounded distribution module.
    """

    extractor: Extractor
    controller: ParameterizedController[CtxType]
    mlp: Mlp
    bounded_distribution: BoundedDistribution

    def __init__(
        self,
        extractor_cls: Type[Extractor],
        observation_space: gym.Space,
        controller: ParameterizedController[CtxType],
        distribution_name: BoundedDistributionName,
        mlp_cfg: MlpConfig,
        init_param_with_default: bool,
    ) -> None:
        """Instantiates the SAC-ZOP actor.

        Args:
            extractor_cls: The class used for extracting features from observations.
            observation_space: The observation space used to configure the extractor.
            controller: The differentiable parameterized controller used to compute actions from
                parameters.
            distribution_name: The name of the bounded distribution used to sample parameters.
            mlp_cfg: The configuration for the MLP used to predict parameters.
            init_param_with_default: Whether to initialize the parameters of the mlp such that the
                parameters transformed by the distribution correspond to the default parameters.
        """
        super().__init__()

        param_space: spaces.Box = controller.param_space
        param_dim = param_space.shape[0]

        self.extractor = extractor_cls(observation_space)
        self.controller = controller
        self.bounded_distribution = get_bounded_distribution(
            distribution_name, space=controller.param_space
        )
        self.mlp = Mlp(
            input_sizes=self.extractor.output_size,
            output_sizes=list(self.bounded_distribution.parameter_size(param_dim)),
            mlp_cfg=mlp_cfg,
        )
        if init_param_with_default:
            init_mlp_params_with_inverse_default(self.mlp, self.bounded_distribution, controller)

    def forward(
        self,
        obs: torch.Tensor,
        ctx: CtxType | None = None,
        deterministic: bool = False,
        only_param: bool = False,
    ) -> SacZopActorOutput:
        """Sample parameters from the policy and (optional) compute actions using the controller.

        The given observations are passed to the extractor to obtain features.
        These are used to predict a bounded distribution in the (learnable) parameter space of the
        controller using the MLP. Afterwards, this parameters are sampled from this distribution,
        and passed to the controller, which then computes the final actions.
        This forward pass does NOT support differentiation through the controller.

        Args:
            obs: The observations to compute the actions for.
            ctx: The optional context object containing information about the previous controller
                solve. Can be used, e.g., to warm-start the solver.
            deterministic: If `True`, use the mode of the distribution instead of sampling.
            only_param: If `True`, only return the predicted parameters and log-probabilities, but
                do not compute the action using the controller.

        Returns:
            SacZopActorOutput: The output of the actor containing parameters, log-probability,
                statistics, actions, and context.
        """
        e = self.extractor(obs)
        dist_params = self.mlp(e)

        param, log_prob, dist_stats = self.bounded_distribution(
            *dist_params, deterministic=deterministic
        )

        if only_param:
            return SacZopActorOutput(param, log_prob, dist_stats)

        with torch.no_grad():
            ctx, action = self.controller(obs, param, ctx=ctx)

        stats = dist_stats
        if ctx.log is not None:
            stats |= ctx.log
        return SacZopActorOutput(param, log_prob, stats, action, ctx)


class SacZopTrainer(Trainer[SacZopTrainerConfig, CtxType], Generic[CtxType]):
    """A trainer that implements SAC with a controller in the policy network.

    The controller is used to compute actions but without differentiating through it (SAC-ZOP).
    Uses parameter noise and a parameter critic.

    Attributes:
        train_env: The training environment.
        q: The Q-function approximator (critic).
        q_target: The target Q-function approximator.
        q_optim: The optimizer for the Q-function.
        pi: The policy network containing the parameterized controller (the actor).
        pi_optim: The optimizer for the policy network.
        log_alpha: The log of the temperature parameter.
        alpha_optim: The optimizer for the temperature parameter.
            If `None`, the temperature is fixed.
        target_entropy: The target entropy for the policy.
            If `None`, the temperature is fixed.
        entropy_norm: The normalization factor for the entropy term.
            Normalizes the entropy based on the ratio of parameter and action dimensions.
        buffer: The replay buffer used to store transitions.
    """

    train_env: gym.Env
    q: SacCritic
    q_target: SacCritic
    q_optim: torch.optim.Optimizer
    pi: MpcSacActor[CtxType]
    pi_optim: torch.optim.Optimizer
    log_alpha: nn.Parameter
    alpha_optim: torch.optim.Optimizer | None
    target_entropy: float | None
    entropy_norm: float
    buffer: ReplayBuffer

    def __init__(
        self,
        cfg: SacZopTrainerConfig,
        val_env: gym.Env,
        output_path: str | Path,
        device: str,
        train_env: gym.Env,
        controller: ParameterizedController[CtxType],
        extractor_cls: Type[Extractor] | ExtractorName = "identity",
    ) -> None:
        """Initializes the SAC-ZOP trainer.

        Args:
            cfg: The configuration for the trainer.
            val_env: The validation environment.
            output_path: The path to the output directory.
            device: The device on which the trainer is running.
            train_env: The training environment.
            controller: The controller to use in the policy.
            extractor_cls: The class used for extracting features from observations.
        """
        super().__init__(cfg, val_env, output_path, device)

        param_space: spaces.Box = controller.param_space
        observation_space = train_env.observation_space
        action_dim = np.prod(train_env.action_space.shape)
        param_dim = np.prod(param_space.shape)

        self.train_env = wrap_env(train_env)

        if isinstance(extractor_cls, str):
            extractor_cls = get_extractor_cls(extractor_cls)

        self.q = SacCritic(
            extractor_cls, param_space, observation_space, cfg.critic_mlp, cfg.num_critics
        )
        self.q_target = SacCritic(
            extractor_cls, param_space, observation_space, cfg.critic_mlp, cfg.num_critics
        )
        self.q_target.load_state_dict(self.q.state_dict())
        self.q_optim = torch.optim.Adam(self.q.parameters(), lr=cfg.lr_q)

        self.pi = MpcSacActor(
            extractor_cls,
            observation_space,
            controller,
            cfg.distribution_name,
            cfg.actor_mlp,
            cfg.init_param_with_default,
        )
        self.pi_optim = torch.optim.Adam(self.pi.parameters(), lr=cfg.lr_pi)

        self.log_alpha = nn.Parameter(torch.tensor(cfg.init_alpha).log())

        self.entropy_norm = param_dim / action_dim
        if cfg.lr_alpha is not None:
            self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=cfg.lr_alpha)
            self.target_entropy = -action_dim if cfg.target_entropy is None else cfg.target_entropy
        else:
            self.alpha_optim = None
            self.target_entropy = None

        self.buffer = ReplayBuffer(cfg.buffer_size, device=device)

    def train_loop(self) -> Generator[int, None, None]:
        is_terminated = is_truncated = True
        policy_ctx = None
        obs = None

        while True:
            if is_terminated or is_truncated:
                obs, _ = seed_env(self.train_env, mk_seed(self.rng), {"mode": "train"})
                policy_ctx = None
                is_terminated = is_truncated = False

            obs_batched = self.buffer.collate([obs])

            with torch.no_grad():
                pi_output: SacZopActorOutput = self.pi(obs_batched, policy_ctx, deterministic=False)
            assert pi_output.action is not None, "Expected action to be not `None`"
            action = pi_output.action.cpu().numpy()[0]
            param = pi_output.param.cpu().numpy()[0]

            self.report_stats("train_trajectory", {"action": action, "param": param}, verbose=True)
            self.report_stats("train_policy_rollout", pi_output.stats, verbose=True)

            obs_prime, reward, is_terminated, is_truncated, info = self.train_env.step(action)

            if "episode" in info or "task" in info:
                self.report_stats("train", info.get("episode", {}) | info.get("task", {}))

            self.buffer.put((obs, param, reward, obs_prime, is_terminated))

            obs = obs_prime
            policy_ctx = pi_output.ctx

            if (
                self.state.step >= self.cfg.train_start
                and len(self.buffer) >= self.cfg.batch_size
                and self.state.step % self.cfg.update_freq == 0
            ):
                # sample batch
                o, a, r, o_prime, te = self.buffer.sample(self.cfg.batch_size)

                # sample action
                pi_o = self.pi(o, None, only_param=True)
                a_pi = pi_o.param
                log_p = pi_o.log_prob / self.entropy_norm

                # update temperature
                if self.alpha_optim is not None:
                    alpha_loss = -torch.mean(
                        self.log_alpha.exp() * (log_p + self.target_entropy).detach()
                    )
                    self.alpha_optim.zero_grad()
                    alpha_loss.backward()
                    self.alpha_optim.step()

                # update critic
                alpha = self.log_alpha.exp().item()
                with torch.no_grad():
                    pi_o_prime = self.pi(o_prime, None, only_param=True)
                    q_target = torch.cat(self.q_target(o_prime, pi_o_prime.param), dim=1)
                    q_target = torch.min(q_target, dim=1, keepdim=True).values

                    # add entropy
                    factor = self.cfg.entropy_reward_bonus / self.entropy_norm
                    q_target = q_target - alpha * pi_o_prime.log_prob * factor

                    target = r[:, None] + self.cfg.gamma * (1 - te[:, None]) * q_target

                q = torch.cat(self.q(o, a), dim=1)
                q_loss = torch.mean((q - target).pow(2))

                self.q_optim.zero_grad()
                q_loss.backward()
                self.q_optim.step()

                # update actor
                q_pi = torch.cat(self.q(o, a_pi), dim=1)
                min_q_pi = torch.min(q_pi, dim=1, keepdim=True).values
                pi_loss = (alpha * log_p - min_q_pi).mean()

                self.pi_optim.zero_grad()
                pi_loss.backward()
                self.pi_optim.step()

                # soft updates
                if self.state.step % self.cfg.soft_update_freq == 0:
                    soft_target_update(self.q, self.q_target, self.cfg.tau)

                # report stats
                loss_stats = {
                    "q_loss": q_loss.item(),
                    "pi_loss": pi_loss.item(),
                    "alpha": alpha,
                    "q": q.mean().item(),
                    "q_target": target.mean().item(),
                    "entropy": -log_p.mean().item(),
                }
                self.report_stats("loss", loss_stats, verbose=True)

            yield 1

    def act(
        self, obs, deterministic: bool = False, state: CtxType | None = None
    ) -> tuple[np.ndarray, CtxType, dict[str, float]]:
        obs = self.buffer.collate([obs])
        with torch.no_grad():
            pi_output: SacZopActorOutput = self.pi(obs, state, deterministic=deterministic)
        assert pi_output.action is not None, "Expected action to be not `None`"
        action = pi_output.action.cpu().numpy()[0]
        return action, pi_output.ctx, pi_output.stats

    @property
    def optimizers(self) -> list[torch.optim.Optimizer]:
        optimizers = [self.q_optim, self.pi_optim]
        if self.alpha_optim is not None:
            optimizers.append(self.alpha_optim)
        return optimizers

    def periodic_ckpt_modules(self) -> list[str]:
        return ["q", "pi", "q_target", "log_alpha"]

    def singleton_ckpt_modules(self) -> list[str]:
        return ["buffer"]
