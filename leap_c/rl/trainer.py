import contextlib
import gc
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, ContextManager

import numpy as np
import torch
from gymnasium import Env
from gymnasium.utils.save_video import save_video
from leap_c.rl.replay_buffer import ReplayBuffer
from leap_c.util import add_prefix_extend, create_dir_if_not_exists


def very_crude_debug_memory_leak():
    debug_file = os.path.join(os.getcwd(), "debug.txt")
    with open(debug_file, "a") as f:
        for obj in gc.get_objects():
            try:
                if torch.is_tensor(obj) or (
                    hasattr(obj, "data") and torch.is_tensor(obj.data)
                ):
                    f.write(str(type(obj)) + "\t" + str(obj.size()) + "\n")
            except:
                pass
        f.write("====End of Episode====" + "\n")


@dataclass(kw_only=True)
class BaseTrainerConfig:
    """Contains the necessary information for the training loop.
    Attributes:
        seed: The seed for reproducibility.
        device: The device on which the models will be trained.
        save_directory_path: The path to a directory where the models will be saved.
        render_mode: The render_mode of the environment. If None, no rendering will be done.
        video_directory_path: The path to a directory where videos will be saved.
            Only does something if render_mode is rgb_array.
        render_interval_exploration: Everytime current_exploration_episodes%render_interval_exploration == 0,
            the episode is being rendered.
        render_interval_validation: Everytime current_validation_episodes%render_interval_validation == 0,
            the episode is being rendered.
        save_interval: The interval in which the models will be saved, additional to models being saved when validation hits a new high.
        max_episodes: The maximum number of episodes to rollout.
        training_per_episode: The number of training steps per episode.
        max_eps_length: The maximum number of steps in an episode.
        dont_train_until_this_many_transitions: The number of transitions that should be gathered in the replay buffer before training starts.
        val_interval: Every this many episodes a validation will be done.
        n_val_rollouts: Number of rollouts during validation.
        no_grad_during_rollout: If True, no gradients will be calculated during the rollout (for efficiency).
        crude_memory_debugging: If True, a very crude memory debugging will be run after every training and the results will be appended into "debug.txt" in the current working directory."""

    seed: int
    device: str
    save_directory_path: str

    render_mode: str | None  # rgb_array or human
    video_directory_path: str | None = None
    render_interval_exploration: int
    render_interval_validation: int

    max_episodes: int
    training_steps_per_episode: int
    dont_train_until_this_many_transitions: int
    val_interval: int
    n_val_rollouts: int
    save_interval: int
    no_grad_during_rollout: bool
    crude_memory_debugging: bool


class Trainer(ABC):
    """Interface for a trainer."""

    replay_buffer: ReplayBuffer
    device: str
    # Used for rendering
    total_validation_rollouts: int = 0
    total_exploration_rollouts: int = 0

    @abstractmethod
    def act(self, obs: Any, deterministic: bool = False) -> np.ndarray:
        """Act based on the observation. This is intended for rollouts (= interaction with the environment).
        Parameters:
            obs: The observation for which the action should be determined.
            deterministic: If True, the action is drawn deterministically.
        """
        raise NotImplementedError()

    @abstractmethod
    def train(self):
        """One step of training the components."""
        raise NotImplementedError()

    @abstractmethod
    def log(self, stats: dict[str, Any], commit: bool):
        """Log the given statistics as one step.
        Parameters:
            stats: A dictionary containing the statistics of the training step.
            commit: Whether to commit the logged statistics to the current step.
                Should be False when more statistics are to be added in a separate call.
            save_directory: The directory where the logs should be saved.
        """
        raise NotImplementedError()

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
        config: BaseTrainerConfig,
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
            if len(self.replay_buffer) > config.dont_train_until_this_many_transitions:
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
