from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal

from leap_c.nn.mlp import MLP, MlpConfig
from leap_c.registry import register_trainer
from leap_c.rl.replay_buffer import ReplayBuffer
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
    l_vf_weight: float = 0.5
    l_ent_weight: float = 0.01
    num_mini_batches: int = 4
    update_epochs: int = 4
    normalize_advantages: bool = True
    clip_value_loss: bool = True
    max_grad_norm: float = 0.5


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


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class PpoCritic(nn.Module):
    def __init__(self, task: Task, env: gym.vector.SyncVectorEnv, mlp_cfg: MlpConfig):
        super().__init__()

        self.extractor = task.create_extractor(env)

        # self.mlp = MLP(
        #     input_sizes=self.extractor.output_size,
        #     output_sizes=1,
        #     mlp_cfg=mlp_cfg,
        # )
        self.mlp = nn.Sequential(
            layer_init(nn.Linear(self.extractor.output_size, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )

    def forward(self, x: torch.Tensor):
        e = self.extractor(x)
        value = self.mlp(e)

        return value.squeeze(-1)


class PpoActor(nn.Module):
    scale: torch.Tensor
    loc: torch.Tensor

    def __init__(self, task, env, mlp_cfg: MlpConfig):
        super().__init__()

        self.extractor = task.create_extractor(env)
        space = env.single_action_space
        action_dim = space.shape[0]

        # self.mlp = MLP(
        #     input_sizes=self.extractor.output_size,
        #     output_sizes=action_dim, # type: ignore
        #     mlp_cfg=mlp_cfg,
        # )
        self.mlp = nn.Sequential(
            layer_init(nn.Linear(self.extractor.output_size, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, action_dim), std=0.01),
        )
        self.log_std = nn.Parameter(torch.zeros(1, action_dim))

        # loc = (space.high + space.low) / 2.0
        # scale = (space.high - space.low) / 2.0
        #
        # loc = torch.tensor(loc, dtype=torch.float32)
        # scale = torch.tensor(scale, dtype=torch.float32)
        #
        # self.register_buffer("loc", loc)
        # self.register_buffer("scale", scale)

    def forward(self, x: torch.Tensor, deterministic: bool = False, action=None):
        e = self.extractor(x)
        mean = self.mlp(e)
        std = self.log_std.expand_as(mean).exp()

        # probs = SquashedGaussianButBetter(mean, std, self.loc, self.scale)
        probs = Normal(mean, std)

        if action is None:
            action = probs.mode if deterministic else probs.sample()

        log_prob = probs.log_prob(action).sum(dim=1)
        entropy = probs.entropy().sum(dim=1)

        return action, log_prob, {
            "entropy": entropy
        }


class ClippedSurrogateLoss(nn.Module):
    def __init__(self, epsilon: float):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, new_log_prob: torch.Tensor, old_log_prob: torch.Tensor, advantage: torch.Tensor) -> torch.Tensor:
        ratio = torch.exp(new_log_prob - old_log_prob)
        unclipped_loss = -advantage * ratio
        clipped_loss = -advantage * torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon)
        return torch.max(unclipped_loss, clipped_loss)


class ValueSquaredErrorLoss(nn.Module):
    def __init__(self, clipped: bool = False, epsilon: float = 0.2):
        super().__init__()
        self.clipped = clipped
        self.epsilon = epsilon

    def forward(self, new_values: torch.Tensor, old_values: torch.Tensor, returns: torch.Tensor) -> torch.Tensor:
        if self.clipped:
            unclipped_loss = (new_values - returns) ** 2
            clipped_values = old_values + torch.clamp(new_values - old_values, -self.epsilon, self.epsilon)
            clipped_loss = (clipped_values - returns) ** 2
            return 0.5 * torch.max(unclipped_loss, clipped_loss).mean()
        else:
            return 0.5 * ((new_values - returns) ** 2).mean()


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

        assert isinstance(self.train_env, gym.vector.SyncVectorEnv), "Only vectorized tasks are supported"

        self.q = PpoCritic(task, self.train_env, cfg.ppo.critic_mlp)
        self.q_optim = torch.optim.Adam(self.q.parameters(), lr=cfg.ppo.lr_q)
        self.q_lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            self.q_optim,
            start_factor=1.0,
            end_factor=0.0,
            total_iters=cfg.train.steps // (cfg.ppo.num_steps * cfg.train.num_envs)
        )

        self.pi = PpoActor(task, self.train_env, cfg.ppo.actor_mlp)
        self.pi_optim = torch.optim.Adam(self.pi.parameters(), lr=cfg.ppo.lr_pi)
        self.pi_lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            self.pi_optim,
            start_factor=1.0,
            end_factor=0.0,
            total_iters=cfg.train.steps // (cfg.ppo.num_steps * cfg.train.num_envs)
        )

        self.clipped_loss = ClippedSurrogateLoss(cfg.ppo.clipping_epsilon)
        self.value_loss = ValueSquaredErrorLoss(cfg.ppo.clip_value_loss, cfg.ppo.clipping_epsilon)

        self.buffer = ReplayBuffer(cfg.ppo.num_steps, device)

    def train_loop(self) -> Iterator[int]:
        obs, _ = self.train_env.reset(seed=self.cfg.seed)

        while True:
            #region Rollout Collection
            obs_collate = self.task.collate(obs, self.device)
            with torch.no_grad():
                action, log_prob, _ = self.pi(obs_collate)
                value = self.q(obs_collate).cpu().numpy()
                action = action.cpu().numpy()
                log_prob = log_prob.cpu().numpy()
            self.report_stats("train_trajectory", {"action": action}, self.state.step)

            obs_prime, reward, is_terminated, is_truncated, info = self.train_env.step(
                action
            )
            self.buffer.put((
                obs,
                action,
                log_prob,
                reward,
                obs_prime,
                is_terminated,
                np.logical_or(is_terminated, is_truncated),
                value
            ))
            if "episode" in info:
                idx = info["episode"]["_r"].argmax()
                self.report_stats("train", {
                    "episodic_return": float(info["episode"]["r"][idx]),
                    "episodic_length": int(info["episode"]["l"][idx]),
                }, self.state.step)
            #endregion

            obs = obs_prime

            if (self.state.step + self.cfg.train.num_envs) % (self.cfg.ppo.num_steps * self.cfg.train.num_envs) == 0:
                #region Generalized Advantage Estimation (GAE)
                advantages = torch.zeros((self.cfg.ppo.num_steps, self.cfg.train.num_envs), device=self.device)
                returns = torch.zeros((self.cfg.ppo.num_steps, self.cfg.train.num_envs), device=self.device)
                with torch.no_grad():
                    for t in reversed(range(self.cfg.ppo.num_steps)):
                        _, _, _, reward, obs_prime, termination, done, value = self.buffer[t]

                        reward = reward.squeeze(0)
                        obs_prime = obs_prime.squeeze(0)
                        termination = termination.squeeze(0)
                        done = done.squeeze(0)
                        value = value.squeeze(0)

                        value_prime = self.q(obs_prime)

                        # TD Error: δ = r + γ * V' - V
                        delta = reward + self.cfg.ppo.gamma * value_prime * (1.0 - termination) - value

                        # GAE: A = δ + γ * λ * A'
                        advantage_prime = advantages[t + 1] if t != self.cfg.ppo.num_steps - 1\
                            else torch.zeros(self.cfg.train.num_envs, device=self.device)
                        advantages[t] = delta + self.cfg.ppo.gamma * self.cfg.ppo.gae_lambda\
                                    * (1.0 - done) * advantage_prime

                        # Returns: G = A + V
                        returns[t] = advantages[t] + value
                #endregion

                #region Loss Calculation and Parameter Optimization
                mini_batch_size = (self.cfg.ppo.num_steps * self.cfg.train.num_envs) // self.cfg.ppo.num_mini_batches
                indices = np.arange(self.cfg.ppo.num_steps * self.cfg.train.num_envs)
                for epoch in range(self.cfg.ppo.update_epochs):
                    np.random.shuffle(indices)
                    for start in range(0, self.cfg.ppo.num_steps * self.cfg.train.num_envs, mini_batch_size):
                        end = start + mini_batch_size
                        mb_indices = indices[start:end]
                        observations, actions, log_probs, _, _, _, _, values = self.buffer[mb_indices]

                        observations = observations.flatten(start_dim=0, end_dim=1)
                        actions = actions.flatten(start_dim=0, end_dim=1)
                        log_probs = log_probs.flatten(start_dim=0, end_dim=1)

                        mb_advantages = advantages[mb_indices].flatten()
                        mb_returns = returns[mb_indices].flatten()

                        if self.cfg.ppo.normalize_advantages:
                            mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                        new_values = self.q(observations)
                        _, new_log_probs, stats = self.pi(observations, action=actions)

                        # Calculating Loss
                        l_clip = self.clipped_loss(new_log_probs, log_probs, mb_advantages).mean()
                        l_vf = self.value_loss(new_values, values, mb_returns)
                        l_ent = -stats["entropy"].mean()

                        loss = l_clip + self.cfg.ppo.l_ent_weight * l_ent + self.cfg.ppo.l_vf_weight * l_vf

                        # Updating Parameters
                        self.q_optim.zero_grad()
                        self.pi_optim.zero_grad()
                        loss.backward()
                        nn.utils.clip_grad_norm_(self.q.parameters(), self.cfg.ppo.max_grad_norm)
                        nn.utils.clip_grad_norm_(self.pi.parameters(), self.cfg.ppo.max_grad_norm)
                        self.q_optim.step()
                        self.pi_optim.step()

                self.report_stats("train", {
                    "policy_loss": l_clip.item(),
                    "value_loss": l_vf.item(),
                    "entropy": -l_ent.item(),
                    "learning_rate": self.q_optim.param_groups[0]["lr"]
                }, self.state.step)
                # endregion

                if self.cfg.ppo.anneal_lr:
                    self.q_lr_scheduler.step()
                    self.pi_lr_scheduler.step()

                self.buffer.clear()

            yield self.cfg.train.num_envs

    def act(
        self, obs, deterministic: bool = False, state=None
    ) -> tuple[np.ndarray, None, dict[str, float]]:
        obs = self.task.collate([obs], self.device)
        with torch.no_grad():
            action, log_prob, stats = self.pi(obs, deterministic=deterministic)
        for key, value in stats.items():
            if isinstance(value, torch.Tensor):
                stats[key] = value.cpu().numpy()
        return action.cpu().numpy()[0], None, stats

    @property
    def optimizers(self) -> list[torch.optim.Optimizer]:
        return [self.q_optim, self.pi_optim]
