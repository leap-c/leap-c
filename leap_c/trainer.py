import bisect
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, DefaultDict, Iterator

import numpy as np
import pandas as pd
import torch
import wandb
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from yaml import safe_dump

from leap_c.rollout import episode_rollout
from leap_c.task import Task
from leap_c.utils import add_prefix_extend, set_seed


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
        log_actions: If True, the actions from interacting with environments will be logged.
        csv_logger: If True, the statistics will be logged to a CSV file.
        tensorboard_logger: If True, the statistics will be logged to TensorBoard.
        wandb_logger: If True, the statistics will be logged to Weights & Biases.
        wandb_init_kwargs: The kwargs to pass to wandb.init. If "dir" is not specified, it is set to output path / "wandb".
    """

    train_interval: int = 1000
    train_window: int = 1000

    val_window: int = 1

    log_actions: bool = False

    csv_logger: bool = True
    tensorboard_logger: bool = True
    wandb_logger: bool = False
    wandb_init_kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass(kw_only=True)
class ValConfig:
    """Contains the necessary information for validation.

    Args:
        interval: The interval at which validation episodes will be run.
        num_rollouts: The number of rollouts during validation.
        deterministic: If True, the policy will act deterministically during validation.
        ckpt_modus: How to save the model, which can be "best", "last", "all" or "none".
        render_mode: The mode in which the episodes will be rendered.
        render_deterministic: If True, the episodes will be rendered deterministically (e.g., no exploration).
        render_interval_exploration: The interval at which exploration episodes will be rendered.
        render_interval_validation: The interval at which validation episodes will be rendered.
        report_score: Whether to report the cummulative score or the final evaluation score.
    """

    interval: int = 10000
    num_rollouts: int = 10
    deterministic: bool = True

    ckpt_modus: str = "best"

    num_render_rollouts: int = 1
    render_mode: str | None = "rgb_array"  # rgb_array or human
    render_deterministic: bool = True

    report_score: str = "cum"  # "final"


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


def set_to_test_cfg(cfg: BaseConfig) -> BaseConfig:
    """Set the configuration to test mode.

    Args:
        cfg: The configuration to be modified.

    Returns:
        The modified configuration.
    """
    cfg.train.steps = 10
    cfg.val.num_rollouts = 1
    cfg.val.interval = 10
    cfg.val.num_render_rollouts = 0
    cfg.val.ckpt_modus = "none"
    cfg.log.csv_logger = False
    cfg.log.tensorboard_logger = False
    cfg.log.wandb_logger = False
    return cfg


def defaultdict_list() -> DefaultDict[str, list]:
    """Returns a defaultdict with a list as default value.

    We make this explicit to avoid issues with pickling."""
    return defaultdict(list)


def nested_defaultdict_list() -> DefaultDict[str, DefaultDict[str, list]]:
    """Returns a nested defaultdict with a list as default value.

    We make this explicit to avoid issues with pickling."""
    return defaultdict(defaultdict_list)


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
        max_score: The maximum score of the validation episodes.
    """

    step: int = 0
    timestamps: dict = field(default_factory=defaultdict_list)
    logs: dict = field(default_factory=nested_defaultdict_list)
    scores: list[float] = field(default_factory=list)
    max_score: float = -float("inf")


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
        self.cfg = cfg
        self.device = device

        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)

        # envs
        self.train_env = self.task.create_train_env(seed=cfg.seed)
        self.eval_env = self.task.create_eval_env(seed=cfg.seed)

        # trainer state
        self.state = TrainerState()

        # init wandb
        if cfg.log.wandb_logger:
            if not cfg.log.wandb_init_kwargs.get("dir", False):  # type:ignore
                wandbdir = self.output_path / "wandb"
                wandbdir.mkdir(exist_ok=True)
                cfg.log.wandb_init_kwargs["dir"] = str(wandbdir)
            wandb.init(**cfg.log.wandb_init_kwargs)

        # tensorboard
        if cfg.log.tensorboard_logger:
            self.writer = SummaryWriter(self.output_path)

        # log dataclass config as yaml
        with open(self.output_path / "config.yaml", "w") as f:
            safe_dump(asdict(cfg), f)

        # seed
        set_seed(cfg.seed)

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
    ) -> tuple[np.ndarray, Any | None, dict[str, float] | None]:
        """Act based on the observation.

        This is intended for rollouts (= interaction with the environment).

        Args:
            obs (Any): The observation for which the action should be determined.
            deterministic (bool): If True, the action is drawn deterministically.
            state: The state of the policy. If the policy is recurrent or includes
                an MPC planner. Note, that at the start of an episode, the state
                assumed to be None.

        Returns:
            The action, the state of the policy and potential solving stats.
        """
        ...

    @property
    def optimizers(self) -> list[torch.optim.Optimizer]:
        """If provided optimizers are also checkpointed."""
        return []

    def report_stats(
        self,
        group: str,
        stats: dict[str, float | np.ndarray],
        timestamp: int | None = None,
        window_size: int | None = None,
    ):
        """Report the statistics of the training loop.

        If the statistics are a numpy array, the array is split into multiple
        statistics of the form `key_{i}`.

        Args:
            group: The group of the statistics.
            stats: The statistics to be reported.
            timestamp: The timestamp of the logging entry. If None, the step
                saved in the trainer state is used.
            window_size: The window size for smoothing the statistics.
        """
        if timestamp is None:
            timestamp = self.state.step

        for key, value in list(stats.items()):
            if not isinstance(value, np.ndarray):
                continue

            if value.size == 1:
                stats[key] = float(value)
                continue

            assert value.ndim == 1, "Only 1D arrays are supported."

            stats.pop(key)
            for i, v in enumerate(value):
                stats[f"{key}_{i}"] = float(v)

        self.state.timestamps[group].append(timestamp)
        for key, value in stats.items():
            self.state.logs[group][key].append(value)

        if window_size is not None:
            window_idx = bisect.bisect_left(
                self.state.timestamps[group],
                timestamp - window_size,  # type: ignore
            )
            stats = {
                key: float(np.mean(values[-window_idx:]))
                for key, values in self.state.logs[group].items()
            }

        if group == "train" or group == "val":
            print(f"Step: {timestamp}, {group}: {stats}")

        if self.cfg.log.wandb_logger:
            newstats = {}
            add_prefix_extend(prefix=group + "/", extended=newstats, extending=stats)
            wandb.log(newstats, step=timestamp)

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
        if self.cfg.val.report_score not in ["cum", "final"]:
            raise RuntimeError(
                f"report_score is {self.cfg.val.report_score} but can be 'cum' or 'final'"
            )

        self.to(self.device)

        train_loop_iter = self.train_loop()

        while self.state.step < self.cfg.train.steps:
            # train
            self.state.step += next(train_loop_iter)

            # validate
            if self.state.step // self.cfg.val.interval > len(self.state.scores):
                val_score = self.validate()
                self.state.scores.append(val_score)

                if val_score > self.state.max_score:
                    self.state.max_score = val_score
                    if self.cfg.val.ckpt_modus == "best":
                        self.save()

                # save model
                if self.cfg.val.ckpt_modus in ["last", "all"]:
                    self.save()

        if self.cfg.val.report_score == "cum":
            return sum(self.state.scores)
        return self.state.max_score

    def validate(self) -> float:
        """Do a deterministic validation run of the policy and
        return the mean of the cumulative reward over all validation episodes."""

        def create_policy_fn():
            policy_state = None

            def policy_fn(obs):
                nonlocal policy_state

                action, policy_state, policy_stats = self.act(
                    obs, deterministic=self.cfg.val.deterministic, state=policy_state
                )
                return action, policy_stats

            return policy_fn

        policy = create_policy_fn()

        parts_rollout = []
        parts_policy = []

        for idx in range(self.cfg.val.num_rollouts):
            if idx < self.cfg.val.num_render_rollouts:
                video_folder = self.output_path / "video"
                video_folder.mkdir(exist_ok=True)
                video_path = video_folder / f"{self.state.step}_{idx}.mp4"
            else:
                video_path = None

            r, p = episode_rollout(
                policy, self.eval_env, render_human=False, video_path=video_path
            )
            parts_rollout.append(r)
            parts_policy.append(p)

        stats_rollout = {
            key: float(np.mean([p[key] for p in parts_rollout]))
            for key in parts_rollout[0]
        }
        self.report_stats(
            "val", stats_rollout, self.state.step, self.cfg.log.val_window
        )

        if parts_policy[0]:
            stats_policy = {
                key: float(np.mean(np.concatenate([p[key] for p in parts_policy])))
                for key in parts_policy[0]
            }
            self.report_stats("val_policy", stats_policy, self.state.step)

        return float(stats_rollout["score"])

    def _ckpt_path(
        self,
        name: str,
        suffix: str,
        basedir: str | Path | None = None,
        singleton: bool = False,
    ) -> Path:
        """Returns the path to a checkpoint file."""
        if basedir is None:
            basedir = self.output_path

        basedir = Path(basedir)
        (basedir / "ckpts").mkdir(exist_ok=True)

        all_but_singleton = (
            True if self.cfg.val.ckpt_modus == "all" and singleton else False
        )

        if self.cfg.val.ckpt_modus == "best":
            return basedir / "ckpts" / f"best_{name}.{suffix}"
        elif self.cfg.val.ckpt_modus == "last" or all_but_singleton:
            return basedir / "ckpts" / f"last_{name}.{suffix}"

        return basedir / "ckpts" / f"{self.state.step}_{name}.{suffix}"

    def periodic_ckpt_modules(self) -> list[str]:
        """Returns the modules that should be checkpointed periodically.

        This is used for example for tracking policy parameters over time.
        """
        return []

    def singleton_ckpt_modules(self) -> list[str]:
        """Returns the modules that should be checkpointed only once.

        Replay Buffers often should not be stored multiple times as there is overlap.
        """
        return []

    def save(self, path: str | Path | None = None) -> None:
        """Save the trainer state in a checkpoint folder.

        If the path is None, the checkpoint is saved in the output path of the trainer.
        The state_dict is split into different parts. For example if the trainer has
        as submodule "pi" and "q", the state_dict is saved separately as "pi.ckpt" and
        "q.ckpt". Additionally, the optimizers are saved as "optimizers.ckpt" and the
        trainer state is saved as "trainer_state.ckpt".

        Args:
            path: The folder where to save the checkpoint.
        """

        # split the state_dict into seperate parts
        for name in self.periodic_ckpt_modules():
            state_dict = getattr(self, name).state_dict()
            torch.save(state_dict, self._ckpt_path(name, "ckpt", path))
        for name in self.singleton_ckpt_modules():
            state_dict = getattr(self, name).state_dict()
            torch.save(state_dict, self._ckpt_path(name, "ckpt", path, singleton=True))

        torch.save(self.state, self._ckpt_path("trainer_state", "ckpt", path))

        if self.optimizers:
            state_dict = {
                f"optimizer_{i}": opt.state_dict()
                for i, opt in enumerate(self.optimizers)
            }
            torch.save(state_dict, self._ckpt_path("optimizers", "ckpt", path))

    def load(self, path: str | Path) -> None:
        """Loads the state of a trainer from the output_path.

        Args:
            path: The path to the checkpoint folder.
        """
        basedir = Path(path)

        # load
        for name in self.periodic_ckpt_modules():
            state_dict = torch.load(self._ckpt_path(name, "ckpt", basedir))
            getattr(self, name).load_state_dict(state_dict)

        for name in self.singleton_ckpt_modules():
            state_dict = torch.load(
                self._ckpt_path(name, "ckpt", basedir), weights_only=False
            )
            getattr(self, name).load_state_dict(state_dict)

        self.state = torch.load(
            self._ckpt_path("trainer_state", "ckpt", basedir), weights_only=False
        )

        if self.optimizers:
            state_dict = torch.load(self._ckpt_path("optimizers", "ckpt", basedir))
            for i, opt in enumerate(self.optimizers):
                opt.load_state_dict(state_dict[f"optimizer_{i}"])
