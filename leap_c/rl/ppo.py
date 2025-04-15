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
    def __init__(
        self,
        task: Task,
        env: gym.Env,
        mlp_cfg: MlpConfig,
    ):
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

        self.pi = PpoActor(task, self.train_env, cfg.ppo.actor_mlp)
        self.pi_optim = torch.optim.Adam(self.pi.parameters(), lr=cfg.ppo.lr_pi)

        self.buffer = RolloutBuffer(
            cfg.ppo.num_steps,
            self.train_env.observation_space.shape,
            self.train_env.action_space.shape,
            device=device,
            collate_fn_map=task.collate_fn_map
        )

    def train_loop(self) -> Iterator[int]:
        is_terminated = is_truncated = True

        while True:
            if is_terminated or is_truncated:
                obs, _ = self.train_env.reset()

            value = self.value(obs)
            action, log_prob, stats = self.act(obs)
            self.report_stats("train_trajectory", {"action": action}, self.state.step)

            obs_prime, reward, is_terminated, is_truncated, info = self.train_env.step(
                action
            )
            self.buffer[self.state.step % self.cfg.ppo.num_steps] = (
                obs, action, log_prob, reward, is_terminated or is_truncated, value
            )

            obs = obs_prime

            if (self.state.step + 1) % self.cfg.ppo.num_steps == 0:
                # TODO: data have been collected
                pass

            yield 1

    def act(
        self, obs, deterministic: bool = False, state=None
    ) -> tuple[np.ndarray, None, dict[str, float]]:
        obs = self.task.collate([obs], self.device)
        with torch.no_grad():
            action, log_prob, stats = self.pi(obs, deterministic=deterministic)
        return action.cpu().numpy()[0], log_prob, stats

    def value(self, obs) -> float:
        obs = self.task.collate([obs], self.device)
        with torch.no_grad():
            value = self.q(obs)
        return value.cpu().numpy()[0]

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
