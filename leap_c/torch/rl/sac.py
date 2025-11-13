from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Generator, Type

import gymnasium as gym
import gymnasium.spaces as spaces
import numpy as np
import torch
import torch.nn as nn

from leap_c.torch.nn.bounded_distributions import (
    BoundedDistribution,
    BoundedDistributionName,
    get_bounded_distribution,
)
from leap_c.torch.nn.extractor import Extractor, ExtractorName, get_extractor_cls
from leap_c.torch.nn.mlp import Mlp, MlpConfig
from leap_c.torch.nn.scale import min_max_scaling
from leap_c.torch.rl.buffer import ReplayBuffer
from leap_c.torch.rl.utils import soft_target_update
from leap_c.torch.utils.seed import mk_seed
from leap_c.trainer import Trainer, TrainerConfig
from leap_c.utils.gym import seed_env, wrap_env


@dataclass(kw_only=True)
class SacTrainerConfig(TrainerConfig):
    """Contains the necessary configuration for a SacTrainer.

    Attributes:
        critic_mlp: The configuration for the Q-networks (critics).
        actor_mlp: The configuration for the policy network.
        batch_size: The batch size for training.
        buffer_size: The size of the replay buffer.
        gamma: The discount factor.
        tau: The soft update factor for the target networks.
        soft_update_freq: The frequency of soft updates (in steps).
        lr_q: The learning rate for the Q-networks.
        lr_pi: The learning rate for the policy network.
        lr_alpha: The learning rate for the temperature parameter.
            Can be set to None to avoid updating the temperature.
        init_alpha: The initial temperature parameter.
        target_entropy: The minimum target entropy for the policy.
            If `None`, it is set automatically depending on dimensions of the action space.
        entropy_reward_bonus: Whether to add an entropy bonus to the reward.
        num_critics: The number of critic networks.
        update_freq: The frequency of updating the networks (in steps).
        distribution_name: The type of bounded distribution to use
            for sampling inside the policy.
    """

    critic_mlp: MlpConfig = field(default_factory=MlpConfig)
    actor_mlp: MlpConfig = field(default_factory=MlpConfig)
    batch_size: int = 64
    buffer_size: int = 1000000
    gamma: float = 0.99
    tau: float = 0.005
    soft_update_freq: int = 1
    lr_q: float = 1e-4
    lr_pi: float = 1e-4
    lr_alpha: float | None = 1e-3
    init_alpha: float = 0.01
    target_entropy: float | None = None
    entropy_reward_bonus: bool = True
    num_critics: int = 2
    update_freq: int = 4
    distribution_name: BoundedDistributionName = "squashed_gaussian"


class SacCritic(nn.Module):
    """A critic network for Soft Actor-Critic (SAC).

    Consists of multiple Q-networks that estimate the expected return for given state-action pairs.

    Attributes:
        extractor: A list of feature extractors for the observations.
        mlp: A list of multi-layer perceptrons (MLPs) that estimate Q-values.
        action_space: The action space of the environment (used for normalizing the actions).
    """

    extractor: nn.ModuleList
    mlp: nn.ModuleList
    action_space: spaces.Box

    def __init__(
        self,
        extractor_cls: Type[Extractor],
        action_space: spaces.Box,
        observation_space: spaces.Space,
        mlp_cfg: MlpConfig,
        num_critics: int,
    ) -> None:
        """Initializes the SAC critic network.

        Args:
            extractor_cls: The class used for extracting features from observations.
            action_space: The action space of the environment (used for normalizing the actions).
            observation_space: The observation space of the environment for the extractors.
            mlp_cfg: The configuration for the MLPs.
            num_critics: The number of critic networks to create.
        """
        super().__init__()

        action_dim = action_space.shape[0]

        self.extractor = nn.ModuleList(extractor_cls(observation_space) for _ in range(num_critics))
        self.mlp = nn.ModuleList(
            [
                Mlp(input_sizes=[qe.output_size, action_dim], output_sizes=1, mlp_cfg=mlp_cfg)
                for qe in self.extractor
            ]
        )
        self.action_space = action_space

    def forward(self, x: torch.Tensor, a: torch.Tensor) -> list[torch.Tensor]:
        """Returns a list of Q-value estimates for the given state-action pairs."""
        a_norm = min_max_scaling(a, self.action_space)
        return [mlp(qe(x), a_norm) for qe, mlp in zip(self.extractor, self.mlp)]


class SacActor(nn.Module):
    """An actor network for Soft Actor-Critic (SAC).

    Attributes:
        extractor: A feature extractor for the observations.
        mlp: A multi-layer perceptron (MLP) that outputs the mean and log standard deviation for the
            action distribution.
        bounded_distribution: A module that samples actions from a bounded distribution.
    """

    extractor: Extractor
    mlp: Mlp
    bounded_distribution: BoundedDistribution

    def __init__(
        self,
        extractor_cls: Type[Extractor],
        action_space: spaces.Box,
        observation_space: spaces.Space,
        distribution_name: BoundedDistributionName,
        mlp_cfg: MlpConfig,
    ) -> None:
        """Initializes the SAC actor network.

        Args:
            extractor_cls: The class used for extracting features from observations.
            action_space: The action space this actor should predict actions from.
            observation_space: The observation space of the environment for the extractor.
            distribution_name: The name of the bounded distribution to use for sampling actions.
            mlp_cfg: The configuration for the MLP.
        """
        super().__init__()

        action_dim = action_space.shape[0]

        self.extractor = extractor_cls(observation_space)
        self.bounded_distribution = get_bounded_distribution(distribution_name, space=action_space)
        self.mlp = Mlp(
            input_sizes=self.extractor.output_size,
            output_sizes=list(self.bounded_distribution.parameter_size(action_dim)),
            mlp_cfg=mlp_cfg,
        )

    def forward(
        self, obs: torch.Tensor, deterministic: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
        """Sample actions from the policy given observations.

        The given observations are passed to the extractor to obtain features.
        These are used by the MLP to predict parameters used to define a
        bounded distribution in the action space.
        The final actions are then sampled from this distribution.

        Args:
            obs: The observations to compute the actions for.
            ctx: The optional context object containing information about the previous controller
                solve. Can be used, e.g., to warm-start the solver.
            deterministic: If `True`, use the mode of the distribution instead of sampling.
        """
        e = self.extractor(obs)
        dist_params = self.mlp(e)

        action, log_prob, stats = self.bounded_distribution(
            *dist_params, deterministic=deterministic
        )

        return action, log_prob, stats


class SacTrainer(Trainer[SacTrainerConfig, Any]):
    """A trainer for Soft Actor-Critic (SAC).

    Attributes:
        train_env: The training environment.
        q: The Q-function approximator (critic).
        q_target: The target Q-function approximator.
        q_optim: The optimizer for the Q-function.
        pi: The policy network (the actor).
        pi_optim: The optimizer for the policy network.
        log_alpha: The logarithm of the temperature parameter.
        alpha_optim: The optimizer for the temperature parameter.
            If `None`, the temperature is fixed.
        target_entropy: The target entropy for the policy.
            If `None`, the temperature is fixed.
        buffer: The replay buffer used for storing and sampling experiences.
    """

    train_env: gym.Env
    q: SacCritic
    q_target: SacCritic
    q_optim: torch.optim.Optimizer
    pi: SacActor
    pi_optim: torch.optim.Optimizer
    log_alpha: nn.Parameter
    alpha_optim: torch.optim.Optimizer | None
    target_entropy: float | None
    buffer: ReplayBuffer

    def __init__(
        self,
        cfg: SacTrainerConfig,
        val_env: gym.Env,
        output_path: str | Path,
        device: str,
        train_env: gym.Env,
        extractor_cls: Type[Extractor] | ExtractorName = "identity",
    ) -> None:
        """Initializes the trainer with a configuration, output path, and device.

        Args:
            cfg: The configuration for the trainer.
            val_env: The validation environment.
            output_path: The path to the output directory.
            device: The device on which the trainer is running
            train_env: The training environment.
            extractor_cls: The class used for extracting features from observations.
        """
        super().__init__(cfg, val_env, output_path, device)

        self.train_env = wrap_env(train_env)
        action_space: spaces.Box = self.train_env.action_space
        observation_space = self.train_env.observation_space

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

        self.pi = SacActor(
            extractor_cls,
            action_space,
            observation_space,
            cfg.distribution_name,
            cfg.actor_mlp,
        )
        self.pi_optim = torch.optim.Adam(self.pi.parameters(), lr=cfg.lr_pi)

        self.log_alpha = nn.Parameter(torch.tensor(cfg.init_alpha).log())

        if self.cfg.lr_alpha is not None:
            self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=cfg.lr_alpha)
            action_dim = np.prod(self.train_env.action_space.shape)
            self.target_entropy = -action_dim if cfg.target_entropy is None else cfg.target_entropy
        else:
            self.alpha_optim = None
            self.target_entropy = None

        self.buffer = ReplayBuffer(cfg.buffer_size, device=device)

    def train_loop(self) -> Generator[int, None, None]:
        is_terminated = is_truncated = True

        while True:
            if is_terminated or is_truncated:
                obs, _ = seed_env(self.train_env, mk_seed(self.rng))
                is_terminated = is_truncated = False

            action, _, stats = self.act(obs)
            self.report_stats("train_trajectory", {"action": action}, verbose=True)
            self.report_stats("train_policy_rollout", stats, verbose=True)

            obs_prime, reward, is_terminated, is_truncated, info = self.train_env.step(action)

            if "episode" in info or "task" in info:
                self.report_stats("train", info.get("episode", {}) | info.get("task", {}))

            # TODO (Jasper): Add is_truncated to buffer.
            self.buffer.put((obs, action, reward, obs_prime, is_terminated))

            obs = obs_prime

            if (
                self.state.step >= self.cfg.train_start
                and len(self.buffer) >= self.cfg.batch_size
                and self.state.step % self.cfg.update_freq == 0
            ):
                # sample batch
                o, a, r, o_prime, te = self.buffer.sample(self.cfg.batch_size)

                # sample action
                a_pi, log_p, _ = self.pi(o)

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
                    a_pi_prime, log_p_prime, _ = self.pi(o_prime)
                    q_target = torch.cat(self.q_target(o_prime, a_pi_prime), dim=1)
                    q_target = torch.min(q_target, dim=1, keepdim=True).values

                    # add entropy
                    q_target = q_target - alpha * log_p_prime * self.cfg.entropy_reward_bonus

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
                self.report_stats("loss", loss_stats)

            yield 1

    def act(
        self, obs, deterministic: bool = False, state: Any = None
    ) -> tuple[np.ndarray, None, dict[str, float]]:
        obs = self.buffer.collate([obs])
        with torch.no_grad():
            action, _, stats = self.pi(obs, deterministic=deterministic)
        return action.cpu().numpy()[0], None, stats

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
