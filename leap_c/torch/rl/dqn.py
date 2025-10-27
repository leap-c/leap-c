"""Provides a trainer for a Deep Q-Network (DQN) algorithm."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Generator

import numpy as np
import torch
import torch.nn as nn
from gymnasium import Env
from gymnasium.spaces import Discrete
from torch import Tensor
from torch.nn.functional import mse_loss

from leap_c.torch.nn.extractor import Extractor, ExtractorName, get_extractor_cls
from leap_c.torch.nn.mlp import Mlp, MlpConfig
from leap_c.torch.rl.buffer import ReplayBuffer
from leap_c.torch.rl.utils import soft_target_update
from leap_c.torch.utils.seed import mk_seed
from leap_c.trainer import Trainer, TrainerConfig
from leap_c.utils.gym import seed_env, wrap_env


@dataclass(kw_only=True)
class DqnTrainerConfig(TrainerConfig):
    """Contains the necessary configuration for a `DqnTrainer`.

    Attributes:
        batch_size: The batch size for training.
        buffer_size: The size of the replay buffer.
        critic_mlp: The configuration for the Q-network (critic).
        gamma: The discount factor.
        lr: The learning rate for the Q-network.
        update_freq: The frequency of updating the network (in steps).
        soft_update_freq: The frequency of soft updates (in steps).
        tau: The soft update factor for the target network.
        gradient_steps: Number of gradient steps to perform per update.
        start_exploration: The starting epsilon for exploration. Must be `>= end_exploration`.
        end_exploration: The ending epsilon for exploration. Must be `<= start_exploration`
        exploration_fraction: The fraction of `train_steps` it takes from start to end

    Notes:
        Default values inspired by CleanRL's DQN implementation.
    """

    batch_size: int = 128
    buffer_size: int = 10_000

    critic_mlp: MlpConfig = field(default_factory=MlpConfig)

    gamma: float = 0.99
    lr: float = 2.5e-4

    update_freq: int = 10
    soft_update_freq: int = 500
    tau: float = 1.0
    gradient_steps: int = 1

    start_exploration: float = 1.0
    end_exploration: float = 0.05
    exploration_fraction: float = 0.5

    def __post_init__(self) -> None:
        if self.start_exploration < self.end_exploration:
            raise ValueError(
                "`start_exploration` must be greater than or equal to `end_exploration`."
            )


class DqnCritic(nn.Module):
    """A critic network for Deep Q-Network (DQN).

    Attributes:
        extractor: A feature extractor for the observations.
        mlp: A Multi-Layer Perceptron (MLP) that estimates Q-values.
    """

    extractor: Extractor
    mlp: Mlp

    def __init__(self, extractor: Extractor, action_space: Discrete, mlp_cfg: MlpConfig) -> None:
        """Initializes the DQN critic network.

        Args:
            extractor: The extractor that returns features from observations.
            action_space: The discrete action space this critic should predict values over.
            observation_space: The observation space of the environment for the extractors.
            mlp_cfg: The configuration for the MLP.
        """
        super().__init__()
        self.extractor = extractor
        self.mlp = Mlp(
            input_sizes=self.extractor.output_size,
            output_sizes=int(action_space.n),
            mlp_cfg=mlp_cfg,
        )

    def forward(self, obs: Tensor) -> Tensor:
        """Returns the Q-value estimates for the given observation."""
        return self.mlp(self.extractor(obs))


def _linear_schedule(start: float, end: float, duration: float, t: float) -> float:
    """Returns a linearly scheduled value between `start` and `end` (with `start` >= `end`), over
    `duration` steps, for the current time step `t`."""
    slope = (end - start) / duration
    return max(slope * t + start, end)


class DqnTrainer(Trainer[DqnTrainerConfig]):
    """A trainer for Deep Q-Network (DQN).

    Attributes:
        train_env (Env): The training environment.
        q (DqnCritic): The Q-function critic approximator.
        q_target (DqnCritic): The target Q-function critic approximator.
        optim (torch.optim.Optimizer): The optimizer for the Q-function.
        buffer (ReplayBuffer): The replay buffer used for storing and sampling experiences.

    Notes:
        DQN only supports discrete action spaces.
    """

    train_env: Env
    q: DqnCritic
    q_target: DqnCritic
    optim: torch.optim.Optimizer
    buffer: ReplayBuffer

    def __init__(
        self,
        cfg: DqnTrainerConfig,
        train_env: Env,
        eval_env: Env,
        output_path: str | Path,
        device: int | str | torch.device,
        extractor_cls: type[Extractor] | ExtractorName = "identity",
    ) -> None:
        """Initializes the trainer with a configuration, output path, and device.

        Args:
            cfg (DqnTrainerConfig): The configuration for the trainer.
            train_env (Env): The training environment.
            eval_env (Env): The evaluation environment.
            output_path (str or Path): The path to the output directory (e.g., logs).
            device (int, str or torch.device): The device on which the trainer is running.
            extractor_cls (type of Extractor or {"identity", "scaling"}): The class used for
                extracting features from observations.

        Raises:
            ValueError: If the action space is not discrete.
        """
        action_space = train_env.action_space
        observation_space = train_env.observation_space
        if not isinstance(action_space, Discrete) or not isinstance(
            eval_env.action_space, Discrete
        ):
            raise ValueError("DQN only supports discrete action spaces.")

        super().__init__(cfg, eval_env, output_path, device)

        self.train_env = wrap_env(train_env)
        if isinstance(extractor_cls, str):
            extractor_cls = get_extractor_cls(extractor_cls)

        self.q = DqnCritic(extractor_cls(observation_space), action_space, cfg.critic_mlp)
        self.q_target = DqnCritic(extractor_cls(observation_space), action_space, cfg.critic_mlp)
        self.q_target.load_state_dict(self.q.state_dict())
        self.optim = torch.optim.Adam(self.q.parameters(), lr=cfg.lr)
        self.buffer = ReplayBuffer(cfg.buffer_size, device=self.device)

    @property
    def optimizers(self) -> list[torch.optim.Optimizer]:
        return [self.optim]

    def periodic_ckpt_modules(self) -> list[str]:
        return ["q", "q_target"]

    def singleton_ckpt_modules(self) -> list[str]:
        return ["buffer"]

    def act(
        self, obs: np.ndarray, deterministic: bool = False, state: Any | None = None
    ) -> tuple[int, None, dict[str, float]]:
        """Runs the Q-Network to obtain an action for the given observation.

        Args:
            obs (array): The observation/initial condition to run through the network.
            deterministic (bool, optional): Whether to act deterministically or stochastically. In
                the latter case, epsilon-greedy exploration is used. Defaults to `False`.
            state (Any): Not used.

        Returns:
            A tuple containing the action to be taken, `None`, and a dict of statistics.
        """
        action_space: Discrete = self.train_env.action_space

        if not deterministic and self.rng.uniform() < (
            epsilon := _linear_schedule(
                self.cfg.start_exploration,
                self.cfg.end_exploration,
                self.cfg.exploration_fraction * self.cfg.train_steps,
                self.state.step,
            )
        ):
            action = action_space.sample()
            stats = {"epsilon": epsilon}

        else:
            with torch.inference_mode():
                q_values: Tensor = self.q(self.buffer.collate((obs,)))
            action = action_space.start + q_values.argmax(dim=1).numpy(force=True)
            stats = {}

        return int(action), None, stats

    def train_loop(self) -> Generator[int]:
        env = self.train_env
        buffer = self.buffer
        state = self.state
        cfg = self.cfg

        # reset before training loop
        obs, _ = seed_env(env, mk_seed(self.rng))
        terminated = truncated = False

        while True:
            # compute an (possibly random) action for the current observation
            action, _, stats = self.act(obs, deterministic=False)
            self.report_stats("train_trajectory", {"action": action}, True)
            self.report_stats("train_policy_rollout", stats, True)

            # step the environment and add transition to buffer
            obs_next, reward, terminated, truncated, info = env.step(action)
            buffer.put((obs, action, reward, obs_next, terminated))
            if "episode" in info or "task" in info:
                self.report_stats("train", info.get("episode", {}) | info.get("task", {}))

            # shift to next step
            obs = obs_next

            # update, if time is ripe
            if (
                state.step >= cfg.train_start
                and state.step % cfg.update_freq == 0
                and len(buffer) >= cfg.batch_size
            ):
                self._update()

            # yield execution to `trainer.run()`
            yield 1

            # reset env if episode ended
            if terminated or truncated:
                obs, _ = seed_env(env, mk_seed(self.rng))
                terminated = truncated = False

    def _update(self) -> None:
        """Performs `cfg.gradient_steps` updates of the critic network based on the squared TD error
        between the current action-value estimate and the target estimate."""
        action_space_start = self.train_env.action_space.start

        losses: list[float] = []
        for _ in range(self.cfg.gradient_steps):
            # sample a batch of transitions
            obs, act, r, obs_next, terminated = self.buffer.sample(self.cfg.batch_size)

            # compute TD target
            with torch.no_grad():
                q_next = self.q_target(obs_next).amax(dim=1)
                target = r + self.cfg.gamma * (1.0 - terminated) * q_next

            # compute current Q estimates and loss
            a_idx = (act - action_space_start).unsqueeze(1).long()
            estimate = self.q(obs).gather(1, a_idx).flatten()
            loss = mse_loss(target, estimate)
            losses.append(loss.item())

            # optimize
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            # soft updates
            if self.state.step % self.cfg.soft_update_freq == 0:
                soft_target_update(self.q, self.q_target, self.cfg.tau)

        avg_td_loss = sum(losses) / len(losses)
        self.report_stats("loss", {"avg_td_loss": avg_td_loss}, with_smoothing=False)
