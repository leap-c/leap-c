from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Generator, Generic, Literal, TypeVar, get_args

import gymnasium as gym
import numpy as np
import torch
from yaml import safe_dump

from leap_c.controller import CtxType
from leap_c.torch.utils.seed import set_seed
from leap_c.utils.gym import WrapperType, wrap_env
from leap_c.utils.logger import Logger, LoggerConfig
from leap_c.utils.rollout import episode_rollout

ValReportScoreOptions = Literal["cum", "final", "best"]


@dataclass(kw_only=True)
class TrainerConfig:
    """Contains the necessary information for the training loop.

    Args:
        seed: The seed for the training.
        train_steps: The number of steps in the training loop.
        train_start: The number of training steps before training starts (e.g., to collect some data
            first).
        val_freq: The frequency (in steps) at which validation episodes will be run.
        val_num_rollouts: The number of episode rollouts during validation.
        val_deterministic: If True, the policy will act deterministically during validation.
        val_render_mode: The mode in which the episodes will be rendered.
        val_report_score: Whether to report the cummulative score,
            the final evaluation score or the best evaluation score of the validation rollouts.
        ckpt_modus: Models are potentially saved after each validation. This controls which of the
            models to save:
            - `"best"`: only the best model according to the validation score is saved
            - `"last"`: only the model of the last validation is saved
            - `"all"`: the models of every validation are saved
            - `"none"`: no models are saved.
        log: The configuration for the logger.
    """

    # reproducibility
    seed: int = 0

    # configuration for the training loop
    train_steps: int = 100_000
    train_start: int = 0

    # validation configuration
    val_freq: int = 10_000
    val_num_rollouts: int = 10
    val_deterministic: bool = True
    val_num_render_rollouts: int = 1
    val_render_mode: str | None = "rgb_array"  # rgb_array or human
    val_report_score: ValReportScoreOptions = "cum"

    # checkpointing configuration
    ckpt_modus: Literal["best", "last", "all", "none"] = "best"

    # logging configuration
    log: LoggerConfig = field(default_factory=LoggerConfig)


TrainerConfigType = TypeVar("TrainerConfigType", bound=TrainerConfig)


@dataclass(kw_only=True)
class TrainerState:
    """The state of a trainer.

    Attributes:
        step: The current step of the training loop.
        scores: A list containing the scores of the validation episodes.
        max_score: The maximum score of the validation episodes.
    """

    step: int = 0
    scores: list[float] = field(default_factory=list)
    max_score: float = float("-inf")


class Trainer(ABC, torch.nn.Module, Generic[TrainerConfigType, CtxType]):
    """A trainer provides the implementation of an algorithm.

    It is responsible for training the components of the algorithm and
    for interacting with the environment.

    Attributes:
        cfg: The configuration for the trainer.
        output_path: The path to the output directory.
        eval_env: The evaluation/validation environment.
        state: The state of the trainer.
        device: The device on which the trainer is running.
        logger: The logger for the trainer.
    """

    cfg: TrainerConfigType
    output_path: Path
    eval_env: gym.Env
    state: TrainerState
    device: str
    logger: Logger

    def __init__(
        self,
        cfg: TrainerConfigType,
        eval_env: gym.Env,
        output_path: str | Path,
        device: int | str | torch.device,
        wrappers: list[WrapperType] | None = None,
    ) -> None:
        """Initializes the trainer with a configuration, output path, and device.

        Args:
            cfg: The configuration for the trainer.
            eval_env: The evaluation/validation environment.
            output_path: The path to the output directory.
            device: The device on which the trainer is running.
            wrappers: Optional list of wrappers to apply to the environment.
        """
        super().__init__()

        self.cfg = cfg
        self.device = torch.device(device)

        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)

        # envs
        self.eval_env = wrap_env(eval_env, wrappers=wrappers)

        # trainer state
        self.state = TrainerState()

        # logger
        self.logger = Logger(self.cfg.log, self.output_path)

        # log dataclass config as yaml
        with open(self.output_path / "config.yaml", "w") as f:
            safe_dump(asdict(self.cfg), f)

        # seed
        self.rng = set_seed(self.cfg.seed)

    @abstractmethod
    def train_loop(self) -> Generator[int, None, None]:
        """The main training loop.

        For simplicity, we use an Iterator here, to make the training loop as simple as
        possible. To make your own code compatible use the yield statement to return the
        number of steps your train loop did. If yield does not always return `1`, the validation
        might be performed not exactly at the specified interval.

        Yields:
           The number of steps the training loop did.
        """

    @abstractmethod
    def act(
        self, obs: np.ndarray, deterministic: bool = False, state: CtxType | None = None
    ) -> tuple[np.ndarray, CtxType | None, dict[str, float] | None]:
        """Act based on the observation.

        This is intended for rollouts (= interaction with the environment).

        Args:
            obs (Any): The observation for which the action should be determined.
            deterministic (bool): If `True`, the action is drawn deterministically.
            state: The state of the policy. Useful, if, e.g., the policy is recurrent or includes
                an MPC planner which might want to pass warmstarting information.
                Note, that at the start of an episode, the state is assumed to be `None`.

        Returns:
            The action, the state of the policy and potential solving stats.
        """

    @property
    def optimizers(self) -> list[torch.optim.Optimizer]:
        """If provided, optimizers are also checkpointed."""
        return []

    def report_stats(
        self,
        group: str,
        stats: dict[str, float | np.ndarray],
        verbose: bool = False,
        with_smoothing: bool = True,
    ) -> None:
        """Report the statistics of the training loop.

        If the statistics are a numpy array, the array is split into multiple
        statistics of the form `"key_{i}"`.

        Args:
            group: The group of the statistics.
            stats: The statistics to be reported.
            verbose: If `True`, the statistics will only be logged in verbosity mode.
            with_smoothing: If `True`, the statistics are smoothed with a moving window.
                This also results in the statistics being only reported at specific
                intervals.
        """
        self.logger(group, stats, self.state.step, verbose, with_smoothing)

    def run(self) -> float:
        """Call this function in your script to start the training loop."""
        if self.cfg.val_report_score not in get_args(ValReportScoreOptions):
            raise RuntimeError(
                f"report_score is '{self.cfg.val_report_score}' "
                f"but has to be one of {get_args(ValReportScoreOptions)}"
            )

        with self.logger:
            self.to(self.device)
            train_loop_iter = self.train_loop()

            # initial policy validation
            self.eval()  # set to eval mode
            with torch.inference_mode():
                val_score = self.validate()
            self.train()  # set back to train mode
            self.state.scores.append(val_score)
            self.state.max_score = val_score

            while self.state.step < self.cfg.train_steps:
                # train
                self.state.step += next(train_loop_iter)

                # validate
                if self.state.step // self.cfg.val_freq >= len(self.state.scores):
                    self.eval()  # set to eval mode
                    with torch.inference_mode():
                        val_score = self.validate()
                    self.train()  # set back to train mode
                    self.state.scores.append(val_score)

                    if val_score > self.state.max_score:
                        self.state.max_score = val_score
                        if self.cfg.ckpt_modus == "best":
                            self.save()

                    # save model
                    if self.cfg.ckpt_modus in ("last", "all"):
                        self.save()

        match self.cfg.val_report_score:
            case "cum":
                return sum(self.state.scores)
            case "final":
                return self.state.scores[-1]
            case "best":
                return self.state.max_score

    def validate(self) -> float:
        """Validate the policy.

        The validation runs the policy deterministically and returns the mean of the cumulative
        reward over all validation episodes.

        Returns:
            The mean return over all validation episodes.

        Note:
            This method neither sets the trainer to eval mode nor disables gradient computations. It
            is the caller's responsibility to do so via `trainer.eval()` as well as `torch.no_grad`
            or `torch.inference_mode` context managers.
        """

        def create_policy_fn():
            policy_state: CtxType | None = None

            def policy_fn(obs):
                nonlocal policy_state

                action, policy_state, policy_stats = self.act(
                    obs, deterministic=self.cfg.val_deterministic, state=policy_state
                )
                return action, policy_state, policy_stats

            return policy_fn

        rollouts = episode_rollout(
            policy=create_policy_fn(),
            env=self.eval_env,
            episodes=self.cfg.val_num_rollouts,
            render_episodes=self.cfg.val_num_render_rollouts,
            render_human=self.cfg.val_render_mode == "human",
            video_folder=self.output_path / "video",
            name_prefix=f"{self.state.step}",
        )

        parts_rollout = []
        parts_policy = []
        for r, p in rollouts:
            parts_rollout.append(r)
            parts_policy.append(p)

        stats_rollout = {
            key: float(np.mean([p[key] for p in parts_rollout])) for key in parts_rollout[0]
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
        ckpt_dir = (self.output_path if basedir is None else Path(basedir)) / "ckpts"
        ckpt_dir.mkdir(exist_ok=True)

        if self.cfg.ckpt_modus == "best":
            return ckpt_dir / f"best_{name}.{suffix}"
        elif self.cfg.ckpt_modus == "last" or (self.cfg.ckpt_modus == "all" and singleton):
            return ckpt_dir / f"last_{name}.{suffix}"

        return ckpt_dir / f"{self.state.step}_{name}.{suffix}"

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

        If the path is `None`, the checkpoint is saved in the output path of the trainer. The
        `state_dict` is split into different parts. For example if the trainer has as submodule
        `"pi"` and `"q"`, the `state_dict` is saved separately as `"pi.ckpt"` and `"q.ckpt"`.
        Additionally, the optimizers are saved as `"optimizers.ckpt"` and the trainer state is saved
        as `"trainer_state.ckpt"`.

        Args:
            path: The folder where to save the checkpoint.
        """

        def save_element(name: str, elem: Any, path: Path, singleton: bool = False) -> None:
            """Saves an element to the checkpoint path."""
            to_be_saved = elem.state_dict() if isinstance(elem, torch.nn.Module) else elem
            torch.save(to_be_saved, self._ckpt_path(name, "ckpt", path, singleton))

        # split the state_dict into seperate parts
        for name in self.periodic_ckpt_modules():
            save_element(name, getattr(self, name), self.output_path)
        for name in self.singleton_ckpt_modules():
            save_element(name, getattr(self, name), self.output_path, singleton=True)

        torch.save(self.state, self._ckpt_path("trainer_state", "ckpt", path))

        if self.optimizers:
            state_dict = {
                f"optimizer_{i}": opt.state_dict() for i, opt in enumerate(self.optimizers)
            }
            torch.save(state_dict, self._ckpt_path("optimizers", "ckpt", path))

    def load(self, path: str | Path) -> None:
        """Loads the state of a trainer from disk.

        Args:
            path: The path to the checkpoint folder.
        """
        basedir = Path(path)

        def load_element(name: str, path: Path, singleton: bool = False) -> None:
            """Loads an element from the checkpoint path."""
            obj = torch.load(self._ckpt_path(name, "ckpt", path, singleton), weights_only=False)
            if isinstance(getattr(self, name), torch.nn.Module):
                getattr(self, name).load_state_dict(obj)
            else:
                setattr(self, name, obj)

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
