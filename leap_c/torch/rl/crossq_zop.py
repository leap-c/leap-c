"""Provides a trainer for the CrossQ-ZOP algorithm.

Implementation based on https://arxiv.org/abs/1902.05605
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Generator, Generic

import gymnasium as gym
import gymnasium.spaces as spaces
import numpy as np
import torch
import torch.nn as nn

from leap_c.controller import CtxType, ParameterizedController
from leap_c.torch.nn.extractor import ExtractorName, get_extractor_cls
from leap_c.torch.nn.mlp import MlpConfig
from leap_c.torch.rl.buffer import ReplayBuffer
from leap_c.torch.rl.mpc_actor import (
    HierachicalMPCActor,
    HierachicalMPCActorConfig,
    StochasticMPCActorOutput,
)
from leap_c.torch.rl.sac import SacCritic
from leap_c.torch.utils.seed import mk_seed
from leap_c.trainer import Trainer, TrainerConfig
from leap_c.utils.gym import seed_env, wrap_env


def recursive_cat(dict1, dict2):
    """Recursively stacks two nested dictionaries of tensors along dim=0.

    Args:
        dict1 (dict): First nested dictionary with tensors.
        dict2 (dict): Second nested dictionary with tensors.

    Returns:
        dict: Nested dictionary with stacked tensors.
    """
    if not isinstance(dict1, dict) or not isinstance(dict2, dict):
        raise ValueError("Both inputs must be dictionaries.")

    stacked_dict = {}
    for key in dict1.keys():
        if key in dict2:
            if isinstance(dict1[key], torch.Tensor) and isinstance(dict2[key], torch.Tensor):
                # Check if tensors have the same shape
                if dict1[key].shape == dict2[key].shape:
                    stacked_dict[key] = torch.cat((dict1[key], dict2[key]), dim=0)
                else:
                    raise ValueError(f"Tensors for key '{key}' have different shapes.")
            elif isinstance(dict1[key], dict) and isinstance(dict2[key], dict):
                # Recursively stack nested dictionaries
                stacked_dict[key] = recursive_cat(dict1[key], dict2[key])
            else:
                raise ValueError(
                    f"Mixed types for key '{key}'. Both must be tensors or dictionaries."
                )
        else:
            raise KeyError(f"Key '{key}' is present in the first dictionary but not in the second.")

    return stacked_dict


@dataclass(kw_only=True)
class CrossQZopConfig(TrainerConfig):
    """Configuration for CrossQ-ZOP trainer.

    Attributes:
        actor: Configuration for the HierachicalMPCActor. Default has batchnorm=True for CrossQ.
        critic_mlp: Q-network configuration. Default has batchnorm=True for CrossQ.
        batch_size: Training batch size.
        buffer_size: Replay buffer size.
        gamma: Discount factor.
        lr_q: Q-network learning rate.
        lr_pi: Policy learning rate.
        lr_alpha: Temperature learning rate.
        init_alpha: Initial temperature.
        target_entropy: Target entropy.
        entropy_reward_bonus: Add entropy bonus to reward.
        num_critics: Number of critics.
        update_freq: Update frequency.
        distribution_name: Bounded distribution type.
    """

    actor: HierachicalMPCActorConfig = field(
        default_factory=lambda: HierachicalMPCActorConfig(
            residual=False, mlp=MlpConfig(batchnorm=True)
        )
    )
    critic_mlp: MlpConfig = field(default_factory=lambda: MlpConfig(batchnorm=True))
    batch_size: int = 64
    buffer_size: int = 1_000_000
    gamma: float = 0.99
    lr_q: float = 1e-4
    lr_pi: float = 1e-4
    lr_alpha: float | None = 1e-3
    init_alpha: float = 0.01
    target_entropy: float | None = None
    entropy_reward_bonus: bool = True
    num_critics: int = 2
    update_freq: int = 4
    distribution_name: str = "squashed_gaussian"


class CrossQZop(Trainer[CrossQZopConfig, CtxType], Generic[CtxType]):
    """CrossQ-ZOP trainer.

    Implements CrossQ with zero-order policy optimization and parameter noise.
    Based on https://arxiv.org/abs/1902.05605

    Attributes:
        train_env: The training environment.
        q: The Q-function approximator (critic).
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
    q_optim: torch.optim.Optimizer
    pi: HierachicalMPCActor[CtxType]
    pi_optim: torch.optim.Optimizer
    log_alpha: nn.Parameter
    alpha_optim: torch.optim.Optimizer | None
    target_entropy: float | None
    entropy_norm: float
    buffer: ReplayBuffer

    def __init__(
        self,
        cfg: CrossQZopConfig,
        val_env: gym.Env | None,
        output_path: str | Path,
        device: str,
        train_env: gym.Env,
        controller: ParameterizedController[CtxType],
        extractor_cls: ExtractorName | None = None,
    ) -> None:
        """Initializes the CrossQ-ZOP trainer.

        Args:
            cfg: The configuration for the trainer.
            val_env: The validation environment. If None, training runs without evaluation.
            output_path: The path to the output directory.
            device: The device on which the trainer is running.
            train_env: The training environment.
            controller: The controller to use in the policy.
            extractor_cls: Deprecated. Use cfg.actor.extractor_name instead.
        """
        super().__init__(cfg, val_env, output_path, device)

        param_space: spaces.Box = controller.param_space
        observation_space = train_env.observation_space
        action_space = train_env.action_space
        action_dim = np.prod(action_space.shape)
        param_dim = np.prod(param_space.shape)

        self.train_env = wrap_env(train_env)

        # Handle deprecated extractor_cls parameter
        if extractor_cls is not None:
            cfg.actor.extractor_name = extractor_cls

        # Get extractor class for critic
        critic_extractor_cls = get_extractor_cls(cfg.actor.extractor_name)

        self.q = SacCritic(
            critic_extractor_cls, param_space, observation_space, cfg.critic_mlp, cfg.num_critics
        )
        self.q_optim = torch.optim.Adam(self.q.parameters(), lr=cfg.lr_q)

        self.pi = HierachicalMPCActor(cfg.actor, observation_space, action_space, controller)
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

    def train_loop(self) -> Generator[tuple[int, float], None, None]:
        is_terminated = is_truncated = True
        policy_ctx = None
        obs = None

        while True:
            if is_terminated or is_truncated:
                obs, _ = seed_env(self.train_env, mk_seed(self.rng), {"mode": "train"})
                policy_ctx = None
                is_terminated = is_truncated = False

            obs_batched = self.buffer.collate([obs])

            self.eval()
            with torch.no_grad():
                pi_output: StochasticMPCActorOutput = self.pi(
                    obs_batched, policy_ctx, deterministic=False
                )
            self.train()
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

                o_comb = recursive_cat(o, o_prime)
                a_comb = torch.cat([a, pi_o_prime.param], dim=0)

                qs = torch.cat(self.q(o_comb, a_comb), dim=1)
                half_idx = self.cfg.batch_size
                q = qs[:half_idx]
                q_target = qs[half_idx:]

                with torch.no_grad():
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

            yield 1, float(reward)

    def act(
        self, obs, deterministic: bool = False, state: CtxType | None = None
    ) -> tuple[np.ndarray, CtxType, dict[str, float]]:
        self.eval()
        obs = self.buffer.collate([obs])
        with torch.inference_mode():
            pi_output: StochasticMPCActorOutput = self.pi(obs, state, deterministic=deterministic)
        assert pi_output.action is not None, "Expected action to be not `None`"
        action = pi_output.action.cpu().numpy()[0]
        self.train()
        return action, pi_output.ctx, pi_output.stats

    @property
    def optimizers(self) -> list[torch.optim.Optimizer]:
        optimizers = [self.q_optim, self.pi_optim]
        if self.alpha_optim is not None:
            optimizers.append(self.alpha_optim)
        return optimizers

    def periodic_ckpt_modules(self) -> list[str]:
        return ["q", "pi", "log_alpha"]

    def singleton_ckpt_modules(self) -> list[str]:
        return ["buffer"]
