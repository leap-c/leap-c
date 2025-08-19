from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Generic, Iterator, Literal, TypeVar, List, get_args

import numpy as np
import torch
import gymnasium as gym
from torch import nn
from yaml import safe_dump

from leap_c.utils.logger import Logger, LoggerConfig
from leap_c.utils.rollout import episode_rollout
from leap_c.utils.gym import wrap_env, WrapperType, seed_env
from leap_c.torch.utils.seed import set_seed


TrainerConfigType = TypeVar("TrainerConfigType", bound="TrainerConfig")
ValReportScoreOptions = Literal["cum", "final", "best"]


@dataclass(kw_only=True)
class TrainerConfig:
    """Contains the necessary information for the training loop.

    Args:
        seed: The seed for the training.
        train_steps: The number of steps in the training loop.
        train_start: The number of training steps before training starts.
        val_interval: The interval at which validation episodes will be run.
        val_num_rollouts: The number of rollouts during validation.
        val_deterministic: If True, the policy will act deterministically during validation.
        val_render_mode: The mode in which the episodes will be rendered.
        val_render_deterministic: If True, the episodes will be rendered deterministically (e.g., no exploration).
        val_report_score: Whether to report the cummulative score, the final evaluation score or the best evaluation score.
        ckpt_modus: How to save the model, which can be "best", "last", "all" or "none".
    """

    # reproducibility
    seed: int = 0

    # configuration for the training loop
    train_steps: int = 100000
    train_start: int = 0

    # validation configuration
    val_interval: int = 10000
    val_num_rollouts: int = 10
    val_deterministic: bool = True
    val_num_render_rollouts: int = 1
    val_render_mode: str | None = "rgb_array"  # rgb_array or human
    val_render_deterministic: bool = True
    val_report_score: ValReportScoreOptions = "cum"

    # checkpointing configuration
    ckpt_modus: Literal["best", "last", "all", "none"] = "best"

    # logging configuration
    log: LoggerConfig = field(default_factory=lambda: LoggerConfig())


@dataclass(kw_only=True)
class TrainerState:
    """The state of a trainer.

    Attributes:
        step: The current step of the training loop.
        timestamps: A dictionary containing the timestamps of the statistics.
        max_score: The maximum score of the validation episodes.
    """

    step: int = 0
    scores: list[float] = field(default_factory=list)
    max_score: float = -float("inf")


class Trainer(ABC, nn.Module, Generic[TrainerConfigType]):
    """A trainer provides the implementation of an algorithm.

    It is responsible for training the components of the algorithm and
    for interacting with the environment.

    Attributes:
        cfg: The configuration for the trainer.
        output_path: The path to the output directory.
        eval_env: The evaluation environment.
        state: The state of the trainer.
        device: The device on which the trainer is running.
        optimizers: The optimizers of the trainer.
    """

    def __init__(
        self,
        cfg: TrainerConfigType,
        eval_env: gym.Env,
        output_path: str | Path,
        device: str,
        wrappers: List[WrapperType] | None = None,
    ):
        """Initializes the trainer with a configuration, output path, and device.

        Args:
            cfg: The configuration for the trainer.
            eval_env: The evaluation environment.
            output_path: The path to the output directory.
            device: The device on which the trainer is running
        """
        super().__init__()

        self.cfg = cfg
        self.device = device

        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)

        # envs
        self.eval_env = seed_env(
            wrap_env(eval_env, wrappers=wrappers), seed=self.cfg.seed
        )

        # trainer state
        self.state = TrainerState()

        # logger
        self.logger = Logger(self.cfg.log, self.output_path)

        # log dataclass config as yaml
        with open(self.output_path / "config.yaml", "w") as f:
            safe_dump(asdict(self.cfg), f)

        # seed
        set_seed(self.cfg.seed)

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
        verbose: bool = False,
        with_smoothing: bool = True,
    ):
        """Report the statistics of the training loop.

        If the statistics are a numpy array, the array is split into multiple
        statistics of the form `key_{i}`.

        Args:
            group: The group of the statistics.
            stats: The statistics to be reported.
            verbose: If True, the statistics will only be logged in verbosity mode.
            with_smoothing: If True, the statistics are smoothed with a moving window.
                This also results in the statistics being only reported at specific
                intervals.
        """
        self.logger(group, stats, self.state.step, verbose, with_smoothing)

    def run(self) -> float:
        """Call this function in your script to start the training loop."""
        if self.cfg.val_report_score not in get_args(ValReportScoreOptions):
            raise RuntimeError(
                f"report_score is '{self.cfg.val_report_score}' but has to be one of {get_args(ValReportScoreOptions)}"
            )

        self.to(self.device)

        train_loop_iter = self.train_loop()

        # initial policy validation
        val_score = self.validate()
        self.state.scores.append(val_score)
        self.state.max_score = val_score

        while self.state.step < self.cfg.train_steps:
            # train
            self.state.step += next(train_loop_iter)

            # validate
            if self.state.step // self.cfg.val_interval > len(self.state.scores) - 1:
                val_score = self.validate()
                self.state.scores.append(val_score)

                if val_score > self.state.max_score:
                    self.state.max_score = val_score
                    if self.cfg.ckpt_modus == "best":
                        self.save()

                # save model
                if self.cfg.ckpt_modus in ["last", "all"]:
                    self.save()

        self.logger.close()

        match self.cfg.val_report_score:
            case "cum":
                return sum(self.state.scores)
            case "final":
                return self.state.scores[-1]
            case "best":
                return self.state.max_score

    def validate(self) -> float:
        """Do a deterministic validation run of the policy and
        return the mean of the cumulative reward over all validation episodes."""

        def create_policy_fn():
            policy_state = None

            def policy_fn(obs):
                nonlocal policy_state

                action, policy_state, policy_stats = self.act(
                    obs, deterministic=self.cfg.val_deterministic, state=policy_state
                )
                return action, policy_stats

            return policy_fn

        policy = create_policy_fn()

        parts_rollout = []
        parts_policy = []

        rollouts = episode_rollout(
            policy,
            self.eval_env,
            self.cfg.val_num_rollouts,
            self.cfg.val_num_render_rollouts,
            render_human=False,
            video_folder=self.output_path / "video",
            name_prefix=f"{self.state.step}",
        )

        for r, p in rollouts:
            parts_rollout.append(r)
            parts_policy.append(p)

        stats_rollout = {
            key: float(np.mean([p[key] for p in parts_rollout]))
            for key in parts_rollout[0]
        }
        self.report_stats("val", stats_rollout, with_smoothing=False)  # type:ignore

        if parts_policy[0]:
            stats_policy = {
                key: float(np.mean(np.concatenate([p[key] for p in parts_policy])))
                for key in parts_policy[0]
            }
            self.report_stats("val_policy", stats_policy, with_smoothing=False)  # type:ignore

        print(f"Validation at {self.state.step}:")
        for key, value in stats_rollout.items():
            print(f"  {key}: {value:.3f}")

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
            True if self.cfg.ckpt_modus == "all" and singleton else False
        )

        if self.cfg.ckpt_modus == "best":
            return basedir / "ckpts" / f"best_{name}.{suffix}"
        elif self.cfg.ckpt_modus == "last" or all_but_singleton:
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

        def save_element(name: str, elem: Any, path: Path, singleton: bool = False):
            """Saves an element to the checkpoint path."""
            if isinstance(elem, nn.Module):
                state_dict = elem.state_dict()
                torch.save(state_dict, self._ckpt_path(name, "ckpt", path, singleton))
            else:
                torch.save(elem, self._ckpt_path(name, "ckpt", path, singleton))

        # split the state_dict into seperate parts
        for name in self.periodic_ckpt_modules():
            save_element(name, getattr(self, name), self.output_path)
        for name in self.singleton_ckpt_modules():
            save_element(name, getattr(self, name), self.output_path, singleton=True)

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

        def load_element(name: str, path: Path, singleton: bool = False):
            """Loads an element from the checkpoint path."""
            if isinstance(getattr(self, name), nn.Module):
                state_dict = torch.load(
                    self._ckpt_path(name, "ckpt", path, singleton), weights_only=False
                )
                getattr(self, name).load_state_dict(state_dict)
            else:
                elem = torch.load(
                    self._ckpt_path(name, "ckpt", path, singleton), weights_only=False
                )
                setattr(self, name, elem)

        # load
        for name in self.periodic_ckpt_modules():
            load_element(name, basedir)

        for name in self.singleton_ckpt_modules():
            load_element(name, basedir, singleton=True)

        self.state = torch.load(
            self._ckpt_path("trainer_state", "ckpt", basedir), weights_only=False
        )

        if self.optimizers:
            state_dict = torch.load(self._ckpt_path("optimizers", "ckpt", basedir))
            for i, opt in enumerate(self.optimizers):
                opt.load_state_dict(state_dict[f"optimizer_{i}"])
