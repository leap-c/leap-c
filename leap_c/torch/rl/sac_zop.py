"""Provides a trainer for a SAC algorithm that sets parameters of a parameterized controller."""

from dataclasses import dataclass, field
from math import prod
from pathlib import Path
from typing import Generator, Generic

import numpy as np
import torch
import torch.nn as nn
from gymnasium import Env, spaces

from leap_c.controller import CtxType, ParameterizedController
from leap_c.torch.nn.extractor import ExtractorName, get_extractor_cls
from leap_c.torch.rl.buffer import ReplayBuffer
from leap_c.torch.rl.mpc_actor import (
    HierachicalMPCActor,
    HierachicalMPCActorConfig,
    StochasticMPCActorOutput,
)
from leap_c.torch.rl.sac import SacCritic, SacTrainerConfig
from leap_c.torch.rl.utils import soft_target_update
from leap_c.torch.utils.seed import mk_seed
from leap_c.trainer import Trainer
from leap_c.utils.gym import seed_env, wrap_env


@dataclass(kw_only=True)
class SacZopTrainerConfig(SacTrainerConfig):
    """Specific settings for the Zop trainer.

    Attributes:
        actor: Configuration for the HierachicalMPCActor.
    """

    actor: HierachicalMPCActorConfig = field(
        default_factory=lambda: HierachicalMPCActorConfig(residual=False)
    )


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

    train_env: Env
    q: SacCritic
    q_target: SacCritic
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
        cfg: SacZopTrainerConfig,
        val_env: Env | None,
        output_path: str | Path,
        device: int | str | torch.device,
        dtype: torch.dtype,
        train_env: Env,
        controller: ParameterizedController[CtxType],
        extractor_cls: ExtractorName | None = None,
    ) -> None:
        """Initializes the SAC-ZOP trainer.

        Args:
            cfg: The configuration for the trainer.
            val_env: The validation environment. If None, training runs without evaluation.
            output_path: The path to the output directory.
            device: The device on which the trainer is running.
            dtype: The data type to use for tensor computations.
            train_env: The training environment.
            controller: The controller to use in the policy.
            extractor_cls: Deprecated. Use cfg.actor.extractor_name instead.
        """
        super().__init__(cfg, val_env, output_path, device)

        param_space: spaces.Box = controller.param_space
        obs_space = train_env.observation_space
        act_space = train_env.action_space
        action_dim = prod(act_space.shape)
        param_dim = prod(param_space.shape)
        self.train_env = wrap_env(train_env)
        device = self.device

        # Handle deprecated extractor_cls parameter
        if extractor_cls is not None:
            cfg.actor.extractor_name = extractor_cls

        # Get extractor class for critic
        critic_extractor_cls = get_extractor_cls(cfg.actor.extractor_name)

        args = (critic_extractor_cls, param_space, obs_space, cfg.critic_mlp, cfg.num_critics)
        self.q = SacCritic(*args).to(device, dtype)
        self.q_target = SacCritic(*args).to(device, dtype)
        self.q_target.load_state_dict(self.q.state_dict())
        self.q_optim = torch.optim.Adam(self.q.parameters(), lr=cfg.lr_q)

        self.pi = HierachicalMPCActor(cfg.actor, obs_space, act_space, controller).to(device, dtype)
        self.pi_optim = torch.optim.Adam(self.pi.parameters(), lr=cfg.lr_pi)

        self.log_alpha = nn.Parameter(
            torch.scalar_tensor(cfg.init_alpha, device=device, dtype=dtype).log()
        )
        self.entropy_norm = param_dim / action_dim
        if cfg.lr_alpha is not None:
            self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=cfg.lr_alpha)
            self.target_entropy = -action_dim if cfg.target_entropy is None else cfg.target_entropy
        else:
            self.alpha_optim = None
            self.target_entropy = None

        self.buffer = ReplayBuffer(cfg.buffer_size, device, dtype)

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

            with torch.no_grad():
                pi_output: StochasticMPCActorOutput = self.pi(
                    obs_batched, policy_ctx, deterministic=False
                )
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
                q_loss = (q - target).square().mean()

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

            yield 1, float(reward)

    def act(
        self, obs, deterministic: bool = False, state: CtxType | None = None
    ) -> tuple[np.ndarray, CtxType, dict[str, float]]:
        obs = self.buffer.collate([obs])
        with torch.inference_mode():
            pi_output: StochasticMPCActorOutput = self.pi(obs, state, deterministic=deterministic)
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
