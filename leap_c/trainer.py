from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Any
import bisect

import wandb
import numpy as np
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
from yaml import dump

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
            This is calculated by the number of training steps.
        val_window: The moving window size for the validation statistics (note that
            this does not consider the training step but the number of validation episodes).
        csv_logger: If True, the statistics will be logged to a CSV file.
        tensorboard_logger: If True, the statistics will be logged to TensorBoard.
        wandb_logger: If True, the statistics will be logged to Weights & Biases.
    """

    train_interval: int = 1000
    train_window: int = 1000

    val_window: int = 1

    csv_logger: bool = True
    tensorboard_logger: bool = True
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
    num_rollouts: int = 20
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
        timestamps: A dictionary containing the timestamps of the statistics.
        logs: A dictionary of dictionaries containing the statistics.
        scores: A list containing the scores of the validation episodes.
        min_score: The minimum score of the validation episodes
    """

    step: int = 0
    timestamps: dict = field(default_factory=lambda: defaultdict(list))
    logs: dict = field(default_factory=lambda: defaultdict(lambda: defaultdict(list)))
    scores: list[float] = field(default_factory=list)
    min_score: float = float("inf")


class Trainer(ABC, nn.Module):
    """A trainer provides the implementation of an algorithm.

    It is responsible for training the components of the algorithm and
    for interacting with the environment.

    Attributes:
        task: The task to be solved by the trainer.
        cfg: The configuration for the trainer.
        output_path: The path to the output directory.
        train_env: The training environment.
        eval_env: The evaluation environment.
        state: The state of the trainer.
        device: The device on which the trainer is running.
        optimizers: The optimizers of the trainer.
    """

    def __init__(
        self, task: Task, output_path: str | Path, device: str, cfg: BaseConfig
    ):
        """Initializes the trainer with a configuration, output path, and device.

        Args:
            task: The task to be solved by the trainer.
            output_path: The path to the output directory.
            device: The device on which the trainer is running
            cfg: The configuration for the trainer.
        """
        super().__init__()

        self.task = task
        task.seed = cfg.seed
        self.cfg = cfg
        self.device = device

        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)

        # envs
        self.train_env = self.task.train_env
        self.eval_env = self.task.eval_env

        # trainer state
        self.state = TrainerState()

        # init wandb
        if cfg.log.wandb_logger:
            wandb.init(project="leap", dir=self.output_path / "wandb")

        # tensorboard
        if cfg.log.tensorboard_logger:
            self.writer = SummaryWriter(self.output_path)

        # log dataclass config as yaml
        with open(self.output_path / "config.yaml", "w") as f:
            dump(cfg, f)

    @abstractmethod
    def train_loop(self) -> Iterator[int]:
        """The main training loop.

        For simplicity, we use an Iterator here, to make the training loop as simple as
        possible. To make your own code compatible use the yield statement to return the
        number of steps your train loop did. If yield not always returns 1, the val-
        idation might be performed not exactly at the specified interval.

        Returns:
           The number of steps the training loop did.
        """
        ...

    @abstractmethod
    def act(
        self, obs, deterministic: bool = False, state=None
    ) -> tuple[np.ndarray, Any | None]:
        """Act based on the observation.

        This is intended for rollouts (= interaction with the environment).

        Args:
            obs (Any): The observation for which the action should be determined.
            deterministic (bool): If True, the action is drawn deterministically.
            state: The state of the policy. If the policy is recurrent or includes
                an MPC planner.

        Returns:
            The action and the state of the policy.
        """
        ...

    @property
    def optimizers(self) -> list[torch.optim.Optimizer]:
        """If provided optimizers are also checkpointed."""
        return []

    def init_policy_state(self) -> Any | None:
        """Initializes the state of the policy.

        This could be useful for recurrent policies or policies that include an MPC planner.

        Returns:
            The initial state of the policy.
        """
        return None

    def report_stats(
        self,
        group: str,
        stats: dict[str, float],
        timestamp: int,
        window_size: int | None = None,
    ):
        """Report the statistics of the training loop.

        Args:
            group: The group of the statistics.
            stats: The statistics to be reported.
            timestamp: The timestamp of the logging entry.
            window_size: The window size for smoothing the statistics.
        """

        self.state.timestamps[group].append(timestamp)
        for key, value in stats.items():
            self.state.logs[group][key].append(value)

        if window_size is not None:
            window_idx = bisect.bisect_left(
                self.state.timestamps[group], timestamp - window_size
            )
            stats = {
                key: float(np.mean(values[-window_idx:]))
                for key, values in self.state.logs[group].items()
            }

        if group == "train" or group == "val":
            print(f"Step: {timestamp}, {group}: {stats}")

        if self.cfg.log.wandb_logger:
            wandb.log(stats, step=timestamp)

        if self.cfg.log.tensorboard_logger:
            for key, value in stats.items():
                self.writer.add_scalar(f"{group}/{key}", value, timestamp)

        if self.cfg.log.csv_logger:
            csv_path = self.output_path / f"{group}_log.csv"

            if csv_path.exists():
                kw = {"mode": "a", "header": False}
            else:
                kw = {"mode": "w", "header": True}

            df = pd.DataFrame(stats, index=[timestamp])  # type: ignore
            df.to_csv(csv_path, **kw)

    def run(self) -> float:
        """Call this function in your script to start the training loop."""
        train_loop_iter = self.train_loop()

        while self.state.step < self.cfg.train.steps:
            # train
            self.state.step += next(train_loop_iter)

            # validate
            if self.state.step // self.cfg.val.interval > len(self.state.scores):
                val_score = self.validate()
                self.state.scores.append(val_score)

                if val_score > self.state.min_score:
                    self.state.min_score = val_score
                    if self.cfg.val.ckpt_modus == "best":
                        self.save()

            # save model
            if self.cfg.val.ckpt_modus != "best":
                self.save()

        return self.state.min_score  # Return last validation score for testing purposes

    def validate(self) -> float:
        """Do a deterministic validation run of the policy and
        return the mean of the cumulative reward over all validation episodes."""

        def create_policy_fn():
            policy_state = self.init_policy_state()

            def policy_fn(obs):
                nonlocal policy_state
                action, policy_state = self.act(
                    obs, deterministic=self.cfg.val.deterministic, state=policy_state
                )
                return action

            return policy_fn

        policy = create_policy_fn()

        parts = []

        for idx in range(self.cfg.val.num_rollouts):
            if idx < self.cfg.val.num_render_rollouts:
                video_folder = self.output_path / "video"
                video_folder.mkdir(exist_ok=True)
                video_path = video_folder / f"{self.state.step}_{idx}.mp4"
            else:
                video_path = None

            p = episode_rollout(
                policy, self.eval_env, render_human=False, video_path=video_path
            )
            parts.append(p)

        stats = {key: float(np.mean([p[key] for p in parts])) for key in parts[0]}
        self.report_stats("val", stats, self.state.step, self.cfg.log.val_window)

        return float(stats["score"])

    def _ckpt_path(self, name: str, suffix: str) -> Path:
        if self.cfg.val.ckpt_modus == "best":
            return self.output_path / "ckpts" / f"best_{name}.{suffix}"
        elif self.cfg.val.ckpt_modus == "last":
            return self.output_path / "ckpts" / f"last_{name}.{suffix}"

        return self.output_path / "ckpts" / f"{self.step}_{name}.{suffix}"

    def save(self) -> None:
        """Save the trainer state in a checkpoint folder."""

        torch.save(self.state_dict(), self._ckpt_path("model", "pth"))
        torch.save(self.state, self._ckpt_path("trainer", "pkl"))

        if self.optimizers:
            state_dict = {
                f"optimizer_{i}": opt.state_dict()
                for i, opt in enumerate(self.optimizers)
            }
            torch.save(state_dict, self._ckpt_path("optimizers", "pth"))

    def load(self) -> None:
        """Loads the state of a trainer from the output_path."""

        self.load_state_dict(torch.load(self._ckpt_path("model", "pth")))
        self.state = torch.load(self._ckpt_path("trainer", "pkl"))

        if self.optimizers:
            state_dict = torch.load(self._ckpt_path("optimizers", "pth"))
            for i, opt in enumerate(self.optimizers):
                opt.load_state_dict(state_dict[f"optimizer_{i}"])
