"""Provides a trainer for a SAC algorithm that uses a diff. MPC layer in the policy network."""

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
class SacFopTrainerConfig(SacTrainerConfig):
    """Specific settings for the Fop trainer.

    Attributes:
        actor: Configuration for the HierachicalMPCActor.
    """

    actor: HierachicalMPCActorConfig = field(default_factory=HierachicalMPCActorConfig)


class SacFopTrainer(Trainer[SacFopTrainerConfig, CtxType], Generic[CtxType]):
    """A trainer implementing Soft Actor-Critic (SAC) that uses a differentiable controller layer.

    The differentiable controller layer is in the policy network (SAC-FOP).

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
    pi: HierachicalMPCActor[CtxType]
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
        extractor_cls: ExtractorName | None = None,
    ) -> None:
        """Initializes the SAC-FOP trainer.

        Args:
            cfg: The configuration for the trainer.
            val_env: The validation environment.
            output_path: The path to the output directory.
            device: The device on which the trainer is running.
            train_env: The training environment.
            controller: The controller to use in the policy.
            extractor_cls: Deprecated. Use cfg.actor.extractor_name instead.
        """
        super().__init__(cfg, val_env, output_path, device)

        param_space: spaces.Box = controller.param_space
        action_space = train_env.action_space
        observation_space = train_env.observation_space
        action_dim = np.prod(action_space.shape)
        param_dim = np.prod(param_space.shape)

        self.train_env = wrap_env(train_env)

        # Handle deprecated extractor_cls parameter
        if extractor_cls is not None:
            cfg.actor.extractor_name = extractor_cls

        # Get extractor class for critic
        critic_extractor_cls = get_extractor_cls(cfg.actor.extractor_name)

        self.q = SacCritic(
            critic_extractor_cls, action_space, observation_space, cfg.critic_mlp, cfg.num_critics
        )
        self.q_target = SacCritic(
            critic_extractor_cls, action_space, observation_space, cfg.critic_mlp, cfg.num_critics
        )
        self.q_target.load_state_dict(self.q.state_dict())
        self.q_optim = torch.optim.Adam(self.q.parameters(), lr=cfg.lr_q)

        self.pi = HierachicalMPCActor(cfg.actor, observation_space, action_space, controller)

        self.pi_optim = torch.optim.Adam(self.pi.parameters(), lr=cfg.lr_pi)

        self.log_alpha = nn.Parameter(torch.tensor(cfg.init_alpha).log())

        self.entropy_norm = param_dim / action_dim if cfg.actor.noise == "param" else 1.0
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
                pi_output: StochasticMPCActorOutput = self.pi(
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

                # Log the gradients of the solution map wrt. params
                dudp = self.pi.controller.jacobian_action_param(ctx=pi_o.ctx)
                zero_grads = np.abs(dudp[mask_status]).sum(axis=(-2, -1)) > 0

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
                    "zero_dudp_perc": 1 - float(zero_grads.mean().item()),
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
            pi_output: StochasticMPCActorOutput = self.pi(obs, state, deterministic)
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
