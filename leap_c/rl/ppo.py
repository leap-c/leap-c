from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from leap_c.nn.mlp import MLP, MlpConfig
from leap_c.registry import register_trainer
from leap_c.rl.rollout_buffer import RolloutBuffer
from leap_c.task import Task
from leap_c.trainer import BaseConfig, LogConfig, TrainConfig, Trainer, ValConfig


@dataclass(kw_only=True)
class PpoAlgorithmConfig:
    """Contains the necessary information for a PpoTrainer.

    Attributes:
        critic_mlp: The configuration for the critic network.
        actor_mlp: The configuration for the actor network.
        num_steps: The number of steps per rollout.
        lr_q: The learning rate for the critic network.
        lr_pi: The learning rate for the actor network.
    """

    critic_mlp: MlpConfig = field(default_factory=MlpConfig)
    actor_mlp: MlpConfig = field(default_factory=MlpConfig)
    num_steps: int = 128
    lr_q: float = 2.5e-4
    lr_pi: float = 2.5e-4
    anneal_lr: bool = True
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    norm_adv: bool = True
    clip_vloss: bool = True
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float = None
    num_mini_batches: int = 4
    update_epochs: int = 4

@dataclass(kw_only=True)
class PpoBaseConfig(BaseConfig):
    """Contains the necessary information for a Trainer.

    Attributes:
        ppo: The Ppo algorithm configuration.
        train: The training configuration.
        val: The validation configuration.
        log: The logging configuration.
        seed: The seed for the trainer.
    """

    ppo: PpoAlgorithmConfig = field(default_factory=PpoAlgorithmConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    val: ValConfig = field(default_factory=ValConfig)
    log: LogConfig = field(default_factory=LogConfig)
    seed: int = 0


class PpoCritic(nn.Module):
    def __init__(self, task: Task, env: gym.Env, mlp_cfg: MlpConfig):
        super().__init__()

        self.extractor = task.create_extractor(env)

        self.mlp = MLP(
            input_sizes=self.extractor.output_size,
            output_sizes=1,
            mlp_cfg=mlp_cfg,
        )

    def forward(self, x: torch.Tensor):
        e = self.extractor(x)
        value = self.mlp(e)

        return value


class PpoActor(nn.Module):
    scale: torch.Tensor
    loc: torch.Tensor

    def __init__(self, task, env, mlp_cfg: MlpConfig):
        super().__init__()

        self.extractor = task.create_extractor(env)
        action_dim = env.action_space.n

        self.mlp = MLP(
            input_sizes=self.extractor.output_size,
            output_sizes=int(action_dim),
            mlp_cfg=mlp_cfg,
        )

    def forward(self, x: torch.Tensor, deterministic: bool = False, action=None):
        e = self.extractor(x)
        logits = self.mlp(e)
        probs = Categorical(logits=logits)

        if action is None:
            action = probs.mode if deterministic else probs.sample()

        return action, probs.log_prob(action), { "entropy" : probs.entropy() }


@register_trainer("ppo", PpoBaseConfig())
class PpoTrainer(Trainer):
    cfg: PpoBaseConfig

    def __init__(
        self, task: Task, output_path: str | Path, device: str, cfg: PpoBaseConfig
    ):
        """Initializes the trainer with a configuration, output path, and device.

        Args:
            task: The task to be solved by the trainer.
            output_path: The path to the output directory.
            device: The device on which the trainer is running
            cfg: The configuration for the trainer.
        """
        super().__init__(task, output_path, device, cfg)

        assert isinstance(self.train_env.action_space, gym.spaces.Discrete),\
            "only discrete action space is supported"

        self.q = PpoCritic(task, self.train_env, cfg.ppo.critic_mlp)
        self.q_optim = torch.optim.Adam(self.q.parameters(), lr=cfg.ppo.lr_q)
        self.q_lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            self.q_optim,
            start_factor=1.0,
            end_factor=0.0,
            total_iters=self.cfg.train.steps // self.cfg.ppo.num_steps
        )

        self.pi = PpoActor(task, self.train_env, cfg.ppo.actor_mlp)
        self.pi_optim = torch.optim.Adam(self.pi.parameters(), lr=cfg.ppo.lr_pi)
        self.pi_lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            self.pi_optim,
            start_factor=1.0,
            end_factor=0.0,
            total_iters=self.cfg.train.steps // self.cfg.ppo.num_steps
        )

        self.buffer = RolloutBuffer(
            cfg.ppo.num_steps,
            self.train_env.observation_space.shape,
            self.train_env.action_space.shape,
            device=device,
            collate_fn_map=task.collate_fn_map
        )

    def train_loop(self) -> Iterator[int]:
        obs, _ = self.train_env.reset()
        is_terminated = is_truncated = False

        while True:
            if is_terminated or is_truncated:
                obs, _ = self.train_env.reset()

            #region Rollout Collection
            obs_collate = self.task.collate([obs], self.device)
            with torch.no_grad():
                value = self.q(obs_collate).cpu().numpy()[0]
            action, log_prob, stats = self.act(obs)
            self.report_stats("train_trajectory", {"action": action}, self.state.step)

            obs_prime, reward, is_terminated, is_truncated, info = self.train_env.step(
                action
            )
            done = float(is_terminated or is_truncated)
            self.buffer[self.state.step % self.cfg.ppo.num_steps] = (
                obs, action, log_prob, reward, done, value
            )
            #endregion

            if (self.state.step + 1) % self.cfg.ppo.num_steps == 0:
                #region Generalized Advantage Estimation (GAE)
                advantages = torch.zeros(self.cfg.ppo.num_steps, device=self.device)
                returns = torch.zeros(self.cfg.ppo.num_steps, device=self.device)
                with torch.no_grad():
                    for t in reversed(range(self.cfg.ppo.num_steps)):
                        obs_prime_collate = self.task.collate([obs_prime], self.device)
                        value_prime = self.q(obs_prime_collate) if t == self.cfg.ppo.num_steps - 1\
                            else self.buffer[t + 1][5]
                        obs, action, log_prob, reward, done, value = self.buffer[t]

                        # TD Error: δ = r + γ * V' - V
                        delta = reward + self.cfg.ppo.gamma * value_prime * (1.0 - done.item()) - value

                        # GAE: A = δ + γ * λ * A'
                        advantage_prime = 0.0 if t == self.cfg.ppo.num_steps - 1 else advantages[t + 1]
                        advantages[t] = delta + self.cfg.ppo.gamma * self.cfg.ppo.gae_lambda\
                                    * (1.0 - done.item()) * advantage_prime

                        # Returns: G = A + V
                        returns[t] = advantages[t] + value
                #endregion

                #region Loss Calculation and Parameter Optimization
                mini_batch_size = self.cfg.ppo.num_steps // self.cfg.ppo.num_mini_batches

                for epoch in range(self.cfg.ppo.update_epochs):
                    for start in range(0, self.cfg.ppo.num_steps, mini_batch_size):
                        end = start + mini_batch_size
                        observations, actions, log_probs, rewards, dones, values = self.buffer[start:end]

                        new_values = self.q(observations)
                        _, new_log_probs, stats = self.pi(observations, action=actions)

                        log_ratio = new_log_probs - log_probs
                        ratio = log_ratio.exp()

                        with torch.no_grad():
                            approx_kl = ((ratio - 1) - log_ratio).mean()

                        mb_advantages = advantages[start:end]
                        if self.cfg.ppo.norm_adv:
                            mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                        pg_loss1 = -mb_advantages * ratio
                        pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self.cfg.ppo.clip_coef, 1 + self.cfg.ppo.clip_coef)
                        pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                        new_values = new_values.view(-1)
                        mb_returns = returns[start:end]
                        if self.cfg.ppo.clip_vloss:
                            v_loss_unclipped = (new_values - mb_returns) ** 2
                            v_clipped = values + torch.clamp(
                                new_values - values,
                                -self.cfg.ppo.clip_coef,
                                self.cfg.ppo.clip_coef,
                            )
                            v_loss_clipped = (v_clipped - mb_returns) ** 2
                            v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                            v_loss = 0.5 * v_loss_max.mean()
                        else:
                            v_loss = 0.5 * ((new_values - mb_returns) ** 2).mean()

                        entropy_loss = stats["entropy"].mean()
                        loss = pg_loss - self.cfg.ppo.ent_coef * entropy_loss + v_loss * self.cfg.ppo.vf_coef

                        self.report_stats("train", {"loss": loss.item()}, self.state.step)

                        self.q_optim.zero_grad()
                        self.pi_optim.zero_grad()

                        loss.backward()

                        nn.utils.clip_grad_norm_(self.q.parameters(), self.cfg.ppo.max_grad_norm)
                        nn.utils.clip_grad_norm_(self.pi.parameters(), self.cfg.ppo.max_grad_norm)

                        self.q_optim.step()
                        self.pi_optim.step()

                    if self.cfg.ppo.target_kl is not None and approx_kl > self.cfg.ppo.target_kl:
                        break
                # endregion

                if self.cfg.ppo.anneal_lr:
                    self.q_lr_scheduler.step()
                    self.pi_lr_scheduler.step()

            obs = obs_prime
            yield 1

    def act(
        self, obs, deterministic: bool = False, state=None
    ) -> tuple[np.ndarray, None, dict[str, float]]:
        obs = self.task.collate([obs], self.device)
        with torch.no_grad():
            action, log_prob, stats = self.pi(obs, deterministic=deterministic)
        return action.cpu().numpy()[0], log_prob, stats

    @property
    def optimizers(self) -> list[torch.optim.Optimizer]:
        return [self.q_optim, self.pi_optim]

    def save(self, path: str | Path | None = None) -> None:
        """Save the trainer state in a checkpoint folder."""

        torch.save(self.buffer, self.output_path / "buffer.pt")
        return super().save()

    def load(self, path: str | Path) -> None:
        """Loads the state of a trainer from the output_path."""

        self.buffer = torch.load(self.output_path / "buffer.pt")
        return super().load(path)
