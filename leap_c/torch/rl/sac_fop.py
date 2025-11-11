"""Provides a trainer for a Soft Actor-Critic algorithm that uses a differentiable MPC
layer in the policy network."""

from dataclasses import dataclass
from pathlib import Path
from typing import Generator, Generic, Literal, NamedTuple, Self, Type

import gymnasium as gym
import gymnasium.spaces as spaces
import numpy as np
import torch
import torch.nn as nn

from leap_c.controller import CtxType, ParameterizedController
from leap_c.torch.nn.bounded_distributions import (
    BoundedDistribution,
    BoundedDistributionName,
    BoundedTransform,
    SquashedGaussian,
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


@dataclass(kw_only=True)
class SacFopTrainerConfig(SacTrainerConfig):
    """Specific settings for the Fop trainer.

    Attributes:
        noise: The type of noise to use for the policy.
            If `"param"`, the noise is added to the predicted parameters (before the controller);
            if `"action"`, the noise is added to the predicted actions (after the controller).
        entropy_correction: Whether to use the entropy correction term for the log-probability.
            When using parameter noise, the computed log-probability does not account for the
            transformation through the controller. The entropy correction adds a correction term
            based on the Jacobian of the action with respect to the parameters.
        init_param_with_default: Whether to initialize the parameters of the controller such that
            the mean of the gaussian transformed by the squashing of the SquashedGaussian
            corresponds to the Parameter default values. Only works if
            1. the parameters are fixed nn.Parameters, and not predicted by a network
            (see MlpConfig hidden_dims).
            2. a SquashedGaussian distribution is used.
            If true, the default parameters according to controller.default_param(None)
            will be used, else the parameters will be initialized to the middle
            of the parameter bounds.
    """

    noise: Literal["param", "action"] = "param"
    entropy_correction: bool = False
    init_param_with_default: bool = True


class SacFopActorOutput(NamedTuple):
    """Output of the SAC-FOP actor.

    Attributes:
        param: The predicted parameters (which have been input into the controller).
        log_prob: The log-probability of the distribution that led to the action.
            NOTE: This log-probability is just a proxy for the true log-probability of the action,
            if using parameter noise.
        stats: A dictionary containing several statistics of internal modules.
        action: The action output by the controller.
        status: The status of the MPC solver (`0` if successful).
        ctx: The context object containing information about the MPC solve.
    """

    param: torch.Tensor
    log_prob: torch.Tensor
    stats: dict[str, float]
    action: torch.Tensor
    status: torch.Tensor
    ctx: CtxType

    def select(self, mask: torch.Tensor) -> Self:
        """Select a subset of the output based on the given mask. Discards stats and ctx."""
        return SacFopActorOutput(
            self.param[mask], self.log_prob[mask], None, self.action[mask], self.status[mask], None
        )


class FopActor(nn.Module, Generic[CtxType]):
    """An actor module for SAC-FOP, containing a differentiable MPC layer and injecting noise in the
    parameter space.

    Attributes:
        controller: The differentiable parameterized controller used to compute actions from
            parameters.
        extractor: The feature extractor used to process observations before passing them to the MLP
            predicting parameters.
        mlp: The MLP used to predict the parameters of the controller from the observations.
        correction: Whether to use the entropy correction term for the log-probability.
        bounded_distribution: The bounded distribution used to sample parameters.
    """

    controller: ParameterizedController[CtxType]
    extractor: Extractor
    mlp: Mlp
    correction: bool
    bounded_distribution: BoundedDistribution

    def __init__(
        self,
        extractor: Extractor,
        mlp_cfg: MlpConfig,
        controller: ParameterizedController[CtxType],
        distribution_name: BoundedDistributionName,
        correction: bool,
        init_param_with_default: bool,
    ) -> None:
        """Initializes the FOP actor.

        Args:
            extractor: The feature extractor used to process observations before passing them to the
                MLP predicting parameters.
            mlp_cfg: The configuration for the MLP used to predict parameters.
            controller: The differentiable parameterized controller used to compute actions from
                parameters.
            distribution_name: The name of the bounded distribution
                used to sample parameters.
            correction: Whether to use the entropy correction term for the log-probability.
            init_param_with_default: Whether to initialize the parameters of the mlp such that the
                parameters transformed by the distribution correspond to the default parameters.
        """
        super().__init__()
        self.controller = controller
        self.extractor = extractor
        param_space = controller.param_space
        param_dim = param_space.shape[0]
        self.bounded_distribution = get_bounded_distribution(distribution_name, space=param_space)
        self.mlp = Mlp(
            input_sizes=self.extractor.output_size,
            output_sizes=list(self.bounded_distribution.parameter_size(param_dim)),
            mlp_cfg=mlp_cfg,
        )
        self.correction = correction
        if init_param_with_default:
            init_mlp_params_with_inverse_default(self.mlp, self.bounded_distribution, controller)

    def forward(
        self, obs: np.ndarray, ctx: CtxType | None = None, deterministic: bool = False
    ) -> SacFopActorOutput:
        """The given observations are passed to the extractor to obtain features.
        These are used to predict a bounded distribution in the (learnable) parameter space of the
        controller using the MLP. Afterwards, this parameters are sampled from this distribution,
        and passed to the controller, which then computes the final actions.

        Args:
            obs: The observations to compute the actions for.
            ctx: The optional context object containing information about the previous controller
                solve. Can be used, e.g., to warm-start the solver.
            deterministic: If `True`, use the mode of the distribution instead of sampling.
        """
        e = self.extractor(obs)
        dist_params = self.mlp(e)

        param, log_prob, dist_stats = self.bounded_distribution(
            *dist_params, deterministic=deterministic
        )

        ctx, action = self.controller(obs, param, ctx=ctx)

        if self.correction:
            j = self.controller.jacobian_action_param(ctx)
            j = torch.from_numpy(j).to(param.device)
            jtj = j @ j.transpose(1, 2)
            correction = (
                torch.det(jtj + 1e-3 * torch.eye(jtj.shape[1], device=jtj.device)).sqrt().log()
            )
            log_prob -= correction.unsqueeze(1)

        stats = dist_stats
        if ctx.log is not None:
            stats |= ctx.log
        return SacFopActorOutput(param, log_prob, stats, action, ctx.status, ctx)


class FoaActor(nn.Module, Generic[CtxType]):
    """An actor module for SAC-FOP, containing a differentiable MPC layer and injecting noise in the
    action space.

    Attributes:
        controller: The differentiable parameterized controller (MPC) used to compute actions.
        extractor: The feature extractor used to process observations before passing them to the MLP
            predicting parameters.
        mlp: The MLP used to predict the parameters of the controller from the observations.
        parameter_transform: The transformation used to map the MLP output to the parameter space.
            Ensures predicted parameters are within bounds.
        action_transform: The transformation used to map the controller output to the action space.
            Ensures predicted actions are within bounds.
        squashed_gaussian: The squashed Gaussian distribution used to sample parameters.
    """

    controller: ParameterizedController[CtxType]
    extractor: Extractor
    mlp: Mlp
    parameter_transform: BoundedTransform
    action_transform: BoundedTransform
    squashed_gaussian: SquashedGaussian

    def __init__(
        self,
        action_space: gym.spaces.Box,
        extractor: Extractor,
        mlp_cfg: MlpConfig,
        controller: ParameterizedController[CtxType],
        init_param_with_default: bool,
    ) -> None:
        """Instantiate the FOA actor.

        Args:
            action_space: The action space this actor should predict actions from.
            extractor: The feature extractor used to process observations before passing them to the
                MLP predicting parameters.
            mlp_cfg: The configuration for the MLP used to predict parameters.
            controller: The differentiable parameterized controller used to compute actions from
                parameters.
            init_param_with_default: Whether to initialize the parameters of the mlp such that the
                parameters transformed by the distribution correspond to the default parameters.
        """
        super().__init__()
        self.controller = controller
        self.extractor = extractor
        param_dim = controller.param_space.shape[0]
        action_dim = action_space.shape[0]
        self.mlp = Mlp(
            input_sizes=self.extractor.output_size,
            output_sizes=(param_dim, action_dim),
            mlp_cfg=mlp_cfg,
        )
        self.parameter_transform = BoundedTransform(self.controller.param_space)
        self.action_transform = BoundedTransform(action_space)
        self.squashed_gaussian = SquashedGaussian(action_space)
        if init_param_with_default:
            init_mlp_params_with_inverse_default(self.mlp, self.parameter_transform, controller)

    def forward(
        self, obs: np.ndarray, ctx: CtxType | None = None, deterministic: bool = False
    ) -> SacFopActorOutput:
        """The given observations are passed to the extractor to obtain features.
        These are used by the MLP to predict parameters, as well as a standard deviation.
        The parameters are passed to the controller to obtain actions. These actions are used
        together with the standard deviation to define a distribution in the action space.
        The final actions are then sampled from this distribution.

        Args:
            obs: The observations to compute the actions for.
            ctx: The optional context object containing information about the previous controller
                solve. Can be used, e.g., to warm-start the solver.
            deterministic: If `True`, use the mean of the distribution instead of sampling.
        """
        e = self.extractor(obs)
        mean, log_std = self.mlp(e)
        param = self.parameter_transform(mean)

        ctx, action_mpc = self.controller(obs, param, ctx=ctx)
        action_unbounded = self.action_transform.inverse(action_mpc)
        action_squashed, log_prob, gaussian_stats = self.squashed_gaussian(
            action_unbounded, log_std, deterministic=deterministic
        )

        stats = gaussian_stats
        if ctx.log is not None:
            stats |= ctx.log
        return SacFopActorOutput(param, log_prob, stats, action_squashed, ctx.status, ctx)


class SacFopTrainer(Trainer[SacFopTrainerConfig, CtxType], Generic[CtxType]):
    """A trainer implementing Soft Actor-Critic (SAC) that uses a differentiable controller layer in
    the policy network (SAC-FOP).
    Supports variants using parameter noise or action noise. Always uses an action critic.

    Attributes:
        train_env: The training environment.
        q: The Q-function approximator (critic).
        q_target: The target Q-function approximator.
        q_optim: The optimizer for the Q-function.
        pi: The policy network containing the parameterized controller (the actor).
        pi_optim: The optimizer for the policy network.
        log_alpha: The logarithm of the temperature parameter.
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
    pi: FopActor[CtxType] | FoaActor[CtxType]
    pi_optim: torch.optim.Optimizer
    log_alpha: nn.Parameter
    alpha_optim: torch.optim.Optimizer | None
    target_entropy: float | None
    entropy_norm: float
    buffer: ReplayBuffer

    def __init__(
        self,
        cfg: SacFopTrainerConfig,
        val_env: gym.Env,
        output_path: str | Path,
        device: str,
        train_env: gym.Env,
        controller: ParameterizedController[CtxType],
        extractor_cls: Type[Extractor] | ExtractorName = "identity",
    ) -> None:
        """Initializes the SAC-FOP trainer.

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
        action_space = train_env.action_space
        observation_space = train_env.observation_space
        action_dim = np.prod(action_space.shape)
        param_dim = np.prod(param_space.shape)

        self.train_env = wrap_env(train_env)

        if isinstance(extractor_cls, str):
            extractor_cls = get_extractor_cls(extractor_cls)

        self.q = SacCritic(
            extractor_cls, action_space, observation_space, cfg.critic_mlp, cfg.num_critics
        )
        self.q_target = SacCritic(
            extractor_cls, action_space, observation_space, cfg.critic_mlp, cfg.num_critics
        )
        self.q_target.load_state_dict(self.q.state_dict())
        self.q_optim = torch.optim.Adam(self.q.parameters(), lr=cfg.lr_q)

        if cfg.noise == "param":
            self.pi = FopActor[CtxType](
                extractor_cls(observation_space),
                cfg.actor_mlp,
                controller,
                cfg.distribution_name,
                cfg.entropy_correction,
                cfg.init_param_with_default,
            )
        elif cfg.noise == "action":
            if cfg.distribution_name != "squashed_gaussian":
                raise ValueError(
                    "When using action noise, the distribution must be 'squashed_gaussian'."
                )
            self.pi = FoaActor[CtxType](
                action_space,
                extractor_cls(observation_space),
                cfg.actor_mlp,
                controller,
                cfg.init_param_with_default,
            )
        else:
            raise ValueError(f"Unknown noise type: {cfg.noise}")

        self.pi_optim = torch.optim.Adam(self.pi.parameters(), lr=cfg.lr_pi)

        self.log_alpha = nn.Parameter(torch.tensor(cfg.init_alpha).log())

        self.entropy_norm = param_dim / action_dim if cfg.noise == "param" else 1.0
        if cfg.lr_alpha is not None:
            self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=cfg.lr_alpha)
            self.target_entropy = -action_dim if cfg.target_entropy is None else cfg.target_entropy
        else:
            self.alpha_optim = None
            self.target_entropy = None

        self.buffer = ReplayBuffer(
            cfg.buffer_size, device=device, collate_fn_map=controller.collate_fn_map
        )

    def train_loop(self) -> Generator[int, None, None]:
        is_terminated = is_truncated = True
        policy_state = None
        obs = None

        while True:
            if is_terminated or is_truncated:
                obs, _ = seed_env(self.train_env, mk_seed(self.rng), {"mode": "train"})
                policy_state = None
                is_terminated = is_truncated = False

            obs_batched = self.buffer.collate([obs])

            with torch.no_grad():
                pi_output: SacFopActorOutput = self.pi(
                    obs_batched, policy_state, deterministic=False
                )
            action = pi_output.action.cpu().numpy()[0]
            param = pi_output.param.cpu().numpy()[0]

            self.report_stats("train_trajectory", {"param": param, "action": action}, verbose=True)
            self.report_stats("train_policy_rollout", pi_output.stats, verbose=True)

            obs_prime, reward, is_terminated, is_truncated, info = self.train_env.step(action)

            if "episode" in info or "task" in info:
                self.report_stats("train", info.get("episode", {}) | info.get("task", {}))

            self.buffer.put(
                (
                    obs,
                    action,
                    reward,
                    obs_prime,
                    is_terminated,
                    pi_output.ctx,
                )
            )

            obs = obs_prime
            policy_state = pi_output.ctx

            if (
                self.state.step >= self.cfg.train_start
                and len(self.buffer) >= self.cfg.batch_size
                and self.state.step % self.cfg.update_freq == 0
            ):
                # sample batch
                o, a, r, o_prime, te, ps_sol = self.buffer.sample(self.cfg.batch_size)

                # sample action
                pi_o = self.pi(o, ps_sol)
                with torch.no_grad():
                    pi_o_prime = self.pi(o_prime, ps_sol)

                pi_o_stats = pi_o.stats

                # Only use samples where the MPC solver was successful for both
                # current and next action.
                mask_status = (pi_o.status == 0) & (pi_o_prime.status == 0)
                o = o[mask_status]
                a = a[mask_status]
                r = r[mask_status]
                o_prime = o_prime[mask_status]
                te = te[mask_status]
                pi_o = pi_o.select(mask_status)
                pi_o_prime = pi_o_prime.select(mask_status)

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
                    q_target = torch.cat(self.q_target(o_prime, pi_o_prime.action), dim=1)
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
                q_pi = torch.cat(self.q(o, pi_o.action), dim=1)
                min_q_pi = torch.min(q_pi, dim=1).values
                pi_loss = (alpha * log_p - min_q_pi).mean()

                self.pi_optim.zero_grad()
                pi_loss.backward()
                self.pi_optim.step()

                # soft updates
                if self.state.step % self.cfg.soft_update_freq == 0:
                    soft_target_update(self.q, self.q_target, self.cfg.tau)

                loss_stats = {
                    "q_loss": q_loss.item(),
                    "pi_loss": pi_loss.item(),
                    "alpha": alpha,
                    "q": q.mean().item(),
                    "q_target": target.mean().item(),
                    "masked_samples_perc": 1 - float(mask_status.mean().item()),
                    "entropy": -log_p.mean().item(),
                }
                self.report_stats("loss", loss_stats)
                self.report_stats("train_policy_update", pi_o_stats, verbose=True)

            yield 1

    def act(
        self, obs: np.ndarray, deterministic: bool = False, state: CtxType | None = None
    ) -> tuple[np.ndarray, CtxType, dict[str, float]]:
        obs = self.buffer.collate([obs])
        with torch.no_grad():
            pi_output: SacFopActorOutput = self.pi(obs, state, deterministic)
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
