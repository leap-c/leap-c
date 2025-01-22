from abc import ABC, abstractmethod
import contextlib
from collections import defaultdict
from dataclasses import dataclass
import os
from typing import ContextManager

from gymnasium import Env
from gymnasium.utils.save_video import save_video
import numpy as np
import torch

from leap_c.util import add_prefix_extend, create_dir_if_not_exists


@dataclass(kw_only=True)
class TrainConfig:
    """Contains the necessary information for the training loop.

    Args:
        steps: The number of steps in the training loop.
        start: The number of training steps before training starts.
        update_interval: The interval at which the model will be updated.
    """
    steps: int = 100000
    start: int = 0
    update_interval: int = 1


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

    render_mode: str | None = "rgb_array"  # rgb_array or human
    render_deterministic: bool = True
    render_interval_exploration: int = 50000
    render_interval_validation: int = 50000


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


@dataclass(kw_only=True)
class BaseConfig:
    """Contains the necessary information for a Trainer."""
    train: TrainConfig
    val: ValConfig
    seed: int


class Trainer(ABC):
    """Interface for a trainer."""

    def __init__(self, cfg: BaseConfig, device: str, output_path: str):
        super().__init__()
        self.cfg = cfg
        self.device = device
        self.output_path = output_path

        self.logs = defaultdict(list)

    @abstractmethod
    def act(self, obs, deterministic: bool = False) -> np.ndarray:
        """Act based on the observation.

        This is intended for rollouts (= interaction with the environment).

        Args:
            obs: The observation for which the action should be determined.
            deterministic: If True, the action is drawn deterministically.
        """
        ...

    @abstractmethod
    def train_step(self):
        """One step of training the components."""
        ...

    @abstractmethod
    def save(self, save_directory: str):
        """Save the models in the given directory."""
        raise NotImplementedError()

    @abstractmethod
    def load(self, save_directory: str):
        """Load the models from the given directory. Is ment to be exactly compatible with save."""
        raise NotImplementedError()

    def validate(
        self,
        ocp_env: Env,
        n_val_rollouts: int,
    ) -> float:
        """Do a deterministic validation run of the policy and
        return the mean of the cumulative reward over all validation episodes."""
        scores = []
        for _ in range(n_val_rollouts):
            info = self.episode_rollout(ocp_env, True, torch.no_grad(), config)
            score = info["score"]
            scores.append(score)

        return sum(scores) / n_val_rollouts

    def episode_rollout(
        self,
        ocp_env: Env,
        validation: bool,
        grad_or_no_grad: ContextManager,
        config: BaseTrainerConfig,
    ) -> dict:
        """Rollout an episode (including putting transitions into the replay buffer) and return the cumulative reward.
        Parameters:
            ocp_env: The gym environment.
            validation: If True, the policy will act as if this is validation (e.g., turning off exploration).
            grad_or_no_grad: A context manager in which to perform the rollout. E.g., torch.no_grad().
            config: The configuration for the training loop.

        Returns:
            A dictionary containing information about the rollout, at containing the keys

            "score" for the cumulative score
            "length" for the length of this episode (how many steps were taken until termination/truncation)
        """
        score = 0
        count = 0
        obs, info = ocp_env.reset(seed=config.seed)

        terminated = False
        truncated = False

        if (
            validation
            and self.total_validation_rollouts % config.render_interval_validation == 0
        ):
            render_this = True
            video_name = "validation"
            episode_index = self.total_validation_rollouts
        elif (
            not validation
            and self.total_exploration_rollouts % config.render_interval_exploration
            == 0
        ):
            render_this = True
            video_name = "exploration"
            episode_index = self.total_exploration_rollouts
        else:
            render_this = False

        if render_this:
            frames = []
        with grad_or_no_grad:
            while not terminated and not truncated:
                a, stats = self.act(obs, deterministic=validation)
                obs_prime, r, terminated, truncated, info = ocp_env.step(a)
                if render_this:
                    frames.append(info["frame"])
                self.replay_buffer.put((obs, a, r, obs_prime, terminated))  # type:ignore
                score += r  # type: ignore
                obs = obs_prime
                count += 1
        if validation:
            self.total_validation_rollouts += 1
        else:
            self.total_exploration_rollouts += 1

        if (
            render_this and config.render_mode == "rgb_array"
        ):  # human mode does not return frames
            save_video(
                frames,
                video_folder=config.video_directory_path,  # type:ignore
                episode_trigger=lambda x: True,
                name_prefix=video_name,
                episode_index=episode_index,
                fps=ocp_env.metadata["render_fps"],
            )
        return dict(score=score, length=count)

    def training_loop(
        self,
        ocp_env: Env,
        config: BaseTrainerConfig,
    ) -> float:
        """Call this function in your script to start the training loop.
        Saving works by calling the save method of the trainer object every
        save_interval many episodes or when validation returns a new best score.

        Parameters:
            ocp_env: The gym environment.
            config: The configuration for the training loop.
        """

        if config.no_grad_during_rollout:
            grad_or_no_grad = torch.no_grad()
        else:
            grad_or_no_grad = contextlib.nullcontext()
        max_val_score = -np.inf

        for n_epi in range(config.max_episodes):
            exploration_stats = dict()
            info = self.episode_rollout(ocp_env, False, grad_or_no_grad, config)
            add_prefix_extend("exploration_", exploration_stats, info)
            if (
                self.replay_buffer.size()
                > config.dont_train_until_this_many_transitions
            ):
                self.log(exploration_stats, commit=False)
                for i in range(config.training_steps_per_episode):
                    training_stats = self.train()
                    self.log(training_stats, commit=True)
                if config.crude_memory_debugging:
                    very_crude_debug_memory_leak()

                if n_epi % config.val_interval == 0:
                    avg_val_score = self.validate(
                        ocp_env=ocp_env,
                        n_val_rollouts=config.n_val_rollouts,
                        config=config,
                    )
                    self.log({"val_score": avg_val_score}, commit=False)
                    print("avg_val_score: ", avg_val_score)
                    if avg_val_score > max_val_score:
                        save_directory_for_models = os.path.join(
                            config.save_directory_path,
                            "val_score_"
                            + str(avg_val_score)
                            + "_episode_"
                            + str(n_epi),
                        )
                        create_dir_if_not_exists(save_directory_for_models)
                        max_val_score = avg_val_score
                        self.save(save_directory_for_models)
            else:
                self.log(exploration_stats, commit=True)
            if n_epi % config.save_interval == 0:
                save_directory_for_models = os.path.join(
                    config.save_directory_path, "episode_" + str(n_epi)
                )
                create_dir_if_not_exists(save_directory_for_models)
                self.save(save_directory_for_models)
        ocp_env.close()
        return max_val_score  # Return last validation score for testing purposes
