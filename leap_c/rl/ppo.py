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
    clipping_epsilon: float = 0.2
    l_vf_weight: float = 0.25
    l_ent_weight: float = 0.01
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


class ClippedSurrogateLoss(nn.Module):
    def __init__(self, epsilon: float):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, new_log_prob: torch.Tensor, old_log_prob: torch.Tensor, advantage: torch.Tensor) -> torch.Tensor:
        ratio = torch.exp(new_log_prob - old_log_prob)
        unclipped_loss = -advantage * ratio
        clipped_loss = -advantage * torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon)
        return torch.max(unclipped_loss, clipped_loss)


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
            total_iters=cfg.train.steps // cfg.ppo.num_steps
        )

        self.pi = PpoActor(task, self.train_env, cfg.ppo.actor_mlp)
        self.pi_optim = torch.optim.Adam(self.pi.parameters(), lr=cfg.ppo.lr_pi)
        self.pi_lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            self.pi_optim,
            start_factor=1.0,
            end_factor=0.0,
            total_iters=cfg.train.steps // cfg.ppo.num_steps
        )

        self.clipped_loss = ClippedSurrogateLoss(cfg.ppo.clipping_epsilon)
        self.mse_loss = nn.MSELoss()

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
                    losses = []
                    for start in range(0, self.cfg.ppo.num_steps, mini_batch_size):
                        end = start + mini_batch_size
                        observations, actions, log_probs, rewards, dones, values = self.buffer[start:end]

                        new_values = self.q(observations)
                        _, new_log_probs, stats = self.pi(observations, action=actions)

                        # Calculating Loss
                        l_clip = self.clipped_loss(new_log_probs, log_probs, advantages[start:end]).mean()
                        l_vf = self.mse_loss(new_values.view(-1), returns[start:end])
                        l_ent = -stats["entropy"].mean()

                        loss = l_clip + self.cfg.ppo.l_ent_weight * l_ent + self.cfg.ppo.l_vf_weight * l_vf
                        losses.append(loss.item())

                        # Updating Parameters
                        self.q_optim.zero_grad()
                        self.pi_optim.zero_grad()
                        loss.backward()
                        self.q_optim.step()
                        self.pi_optim.step()
                    self.report_stats("train", {"epoch": epoch + 1, "loss": sum(losses)}, self.state.step)
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
        for key, value in stats.items():
            if isinstance(value, torch.Tensor):
                stats[key] = value.cpu().numpy()
        return action.cpu().numpy()[0], log_prob, stats

    @property
    def optimizers(self) -> list[torch.optim.Optimizer]:
        return [self.q_optim, self.pi_optim]
