from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple
import bisect

import wandb
import numpy as np
import torch
from torch import nn
import pandas as pd

from leap_c.task import Task
from leap_c.rollout import episode_rollout


@dataclass(kw_only=True)
class TrainConfig:
    """Contains the necessary information for the training loop.

    Args:
        steps: The number of steps in the training loop.
        start: The number of training steps before training starts.
    """

    steps: int = 100000
    start: int = 0


@dataclass(kw_only=True)
class LogConfig:
    """Contains the necessary information for logging.

    Args:
        train_interval: The interval at which training statistics will be logged.
        train_window: The moving window size for the training statistics.
        val_window: The moving window size for the validation statistics (note that
            this does not consider the training step but the number of validation episodes).
    """

    train_interval: int = 1000
    train_window: int = 1000

    val_window: int = 1

    csv_logger: bool = False
    wandb_logger: bool = False


@dataclass(kw_only=True)
class ValConfig:
    """Contains the necessary information for validation.

    Args:
        interval: The interval at which validation episodes will be run.
        num_rollouts: The number of rollouts during validation.
        deterministic: If True, the policy will act deterministically during validation.
        ckpt_modus: How to save the model, which can be "best", "last" or "all".
        render_mode: The mode in which the episodes will be rendered.
        render_deterministic: If True, the episodes will be rendered deterministically (e.g., no exploration).
        render_interval_exploration: The interval at which exploration episodes will be rendered.
        render_interval_validation: The interval at which validation episodes will be rendered.
    """

    interval: int = 50000
    num_rollouts: int = 1
    deterministic: bool = True

    ckpt_modus: str = "best"

    num_render_rollouts: int = 1
    render_mode: str | None = "rgb_array"  # rgb_array or human
    render_deterministic: bool = True


@dataclass(kw_only=True)
class BaseConfig:
    """Contains the necessary information for a Trainer.

    Attributes:
        train: The training configuration.
        val: The validation configuration.
        log: The logging configuration.
        seed: The seed for the trainer.
    """

    train: TrainConfig
    val: ValConfig
    log: LogConfig
    seed: int


@dataclass(kw_only=True)
class TrainerState:
    """The state of a trainer.

    Contains all the necessary information to save and load a trainer state
    and to calculate the training statistics. Thus everything that is not
    stored by the torch state dict.

    Attributes:
        step: The current step of the training loop.
        train_logs: A dictionary containing the logs of the training update.
        train_logs_time: A deque containing the time of the training updates.
        val_logs: A dictionary containing the logs of the validation.
        scores: A list containing the scores of the validation episodes.
        min_score: The minimum score of the validation episodes
    """

    step: int = 0
    train_logs: dict = field(default_factory=lambda: defaultdict(list))
    train_logs_time: list[int] = field(default_factory=list)
    val_logs: dict = field(default_factory=lambda: defaultdict(list))
    scores: list[int] = field(default_factory=list)
    min_score: float = float("inf")


class Trainer(ABC, nn.Module):
    """A trainer provides the implementation of an algorithm.

    It is responsible for training the components of the algorithm and
    for interacting with the environment.

    Attributes:
        task: The task to be solved by the trainer.
        cfg: The configuration for the trainer.
        output_path: The path to the output directory.
        device: The device on which the trainer is running.
        train_logs: A dictionary containing the logs of the training update.
        train_logs_time: A deque containing the time of the training updates.
        val_logs: A dictionary containing the logs of the validation.
    """

    def __init__(self, task: Task, cfg: BaseConfig, output_path: str, device: str):
        """Initializes the trainer with a configuration, output path, and device.

        Args:
            task: The task to be solved by the trainer.
            cfg: The configuration for the trainer.
            output_path: The path to the output directory.
            device: The device on which the trainer is running
        """
        super().__init__()

        self.task = task
        self.cfg = cfg
        self.device = device

        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)

        # env
        # COMMENT: In the future we could allow different envs.
        self.train_env = task.env_factory()
        self.train_env.reset(seed=cfg.seed)
        self.eval_env = task.env_factory()
        self.eval_env.reset(seed=cfg.seed)

        # trainer state
        self.state = TrainerState()

        # init wandb
        if cfg.log.wandb_logger:
            wandb.init(project="leap", dir=self.output_path / "wandb")

    @abstractmethod
    def train_step(self) -> dict[str, float]:
        """A single training step.

        Returns:
            A dictionary containing the training statistics.
        """
        ...

    @abstractmethod
    def act(self, obs, deterministic: bool = False) -> np.ndarray:
        """Act based on the observation.

        This is intended for rollouts (= interaction with the environment).

        Args:
            obs (Any): The observation for which the action should be determined.
            deterministic (bool): If True, the action is drawn deterministically.

        Returns:
            The action to take.
        """
        ...

    def report_stats(self, group: str, stats: dict[str, float], step: int):
        """Report the statistics of the training loop.

        Args:
            group: The group of the statistics.
            stats: The statistics to be reported.
            step: The current step of the training loop
        """

        if self.cfg.log.wandb_logger:
            wandb.log(stats, step=step)

        if self.cfg.log.csv_logger:
            csv_path = self.output_path / f"{group}_log.csv"

            if csv_path.exists():
                kw = {"mode": "a", "header": False}
            else:
                kw = {"mode": "w", "header": True}

            df = pd.DataFrame(stats, index=[step])  # type: ignore
            df.to_csv(csv_path, **kw)

    def loop(self) -> float:
        """Call this function in your script to start the training loop."""

        for self.state.step in range(self.state.step, self.cfg.train.steps):

            # train step
            train_logs = self.train_step()

            # update train logs
            if train_logs is not None:
                for key, value in train_logs.items():
                    self.state.train_logs[key].append(value)
                    self.state.train_logs_time.append(self.state.step)

            # do validation
            if self.state.step % self.cfg.val.interval == 0:
                val_score, val_stats = self.validate()
                self.scores.append(val_score)

                if val_score > self.state.min_score:
                    self.state.min_score = val_score

                    if self.cfg.val.ckpt_modus == "best":
                        self.save()

                for key, value in val_stats.items():
                    self.state.val_logs[key].append(value)

                smoothed_stats = {
                    key: np.mean(self.state.val_logs[key][-self.cfg.log.val_window :])
                    for key in self.state.val_logs
                }
                self.report_stats("val", smoothed_stats, self.state.step)

            # do logging
            if self.state.step % self.cfg.log.train_interval == 0:
                # find window length for smoothing
                time_index = bisect.bisect_left(
                    self.state.train_logs_time,
                    self.state.step - self.cfg.log.train_window,
                )
                smoothed_stats = {
                    key: np.mean(self.state.train_logs[key][time_index:])
                    for key in self.state.train_logs
                }
                self.report_stats("train", smoothed_stats, self.state.step)

            # save model
            if self.cfg.val.ckpt_modus != "best":
                self.save()

        return self.state.min_score  # Return last validation score for testing purposes

    def validate(self) -> Tuple[float, dict[str, float]]:
        """Do a deterministic validation run of the policy and
        return the mean of the cumulative reward over all validation episodes."""

        policy = lambda obs: self.act(obs, deterministic=self.cfg.val.deterministic)

        parts = []

        for _ in range(self.cfg.val.num_rollouts):
            p = episode_rollout(policy, self.eval_env, render_human=False)
            parts.append(p)

        stats = {key: float(np.mean([p[key] for p in parts])) for key in parts[0]}

        return float(stats["score"]), stats

    def _ckpt_path(self, name: str) -> Path:
        if self.cfg.val.ckpt_modus == "best":
            return self.output_path / "ckpts" / f"best_{name}.pth"

        return self.output_path / "ckpts" / f"{self.step}_{name}.pth"

    def save(self) -> None:
        """Save the trainer state in a checkpoint folder."""

        torch.save(self.state_dict(), self._ckpt_path("model"))
        torch.save(self.state, self._ckpt_path("trainer"))

    def load(self) -> None:
        """Loads the state of a trainer from the output_path."""

        self.load_state_dict(torch.load(self._ckpt_path("model")))
        self.state = torch.load(self._ckpt_path("trainer"))
