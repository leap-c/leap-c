from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Type

import gymnasium as gym
import gymnasium.spaces as spaces
import numpy as np
import torch
import torch.nn as nn

from leap_c.torch.nn.extractor import ExtractorName, Extractor, get_extractor_cls
from leap_c.torch.nn.gaussian import SquashedGaussian
from leap_c.torch.nn.mlp import MLP, MlpConfig
from leap_c.torch.nn.scale import min_max_scaling
from leap_c.torch.rl.buffer import ReplayBuffer
from leap_c.torch.rl.utils import soft_target_update
from leap_c.trainer import Trainer, TrainerConfig
from leap_c.utils.gym import wrap_env, seed_env


@dataclass(kw_only=True)
class SacTrainerConfig(TrainerConfig):
    """Contains the necessary configuration for a SacTrainer.

    Attributes:
        critic_mlp: The configuration for the critic networks.
        actor_mlp: The configuration for the actor network.
        batch_size: The batch size for training.
        buffer_size: The size of the replay buffer.
        gamma: The discount factor.
        tau: The soft update factor.
        soft_update_freq: The frequency of soft updates.
        lr_q: The learning rate for the Q networks.
        lr_pi: The learning rate for the policy network.
        lr_alpha: The learning rate for the temperature parameter.
        init_alpha: The initial temperature parameter.
        target_entropy: The minimum target entropy for the policy. If None, it
            is set automatically depending on dimensions of the action space.
        entropy_reward_bonus: Whether to add an entropy bonus to the reward.
        num_critics: The number of critic networks.
        report_loss_freq: The frequency of reporting the loss.
        update_freq: The frequency of updating the networks.
    """

    critic_mlp: MlpConfig = field(default_factory=MlpConfig)
    actor_mlp: MlpConfig = field(default_factory=MlpConfig)
    batch_size: int = 64
    buffer_size: int = 1000000
    gamma: float = 0.99
    tau: float = 0.005
    soft_update_freq: int = 1
    lr_q: float = 1e-4
    lr_pi: float = 1e-4
    lr_alpha: float | None = 1e-3
    init_alpha: float = 0.01
    target_entropy: float | None = None
    entropy_reward_bonus: bool = True
    num_critics: int = 2
    report_loss_freq: int = 100
    update_freq: int = 4


class SacCritic(nn.Module):
    def __init__(
        self,
        extractor_cls: Type[Extractor],
        action_space: spaces.Box,
        observation_space: spaces.Space,
        mlp_cfg: MlpConfig,
        num_critics: int,
    ):
        super().__init__()

        action_dim = action_space.shape[0]  # type: ignore

        self.extractor = nn.ModuleList(
            extractor_cls(observation_space) for _ in range(num_critics)
        )
        self.mlp = nn.ModuleList(
            [
                MLP(
                    input_sizes=[qe.output_size, action_dim],  # type: ignore
                    output_sizes=1,
                    mlp_cfg=mlp_cfg,
                )
                for qe in self.extractor
            ]
        )
        self.action_space = action_space

    def forward(self, x: torch.Tensor, a: torch.Tensor):
        a_norm = min_max_scaling(a, self.action_space)  # type: ignore
        return [mlp(qe(x), a_norm) for qe, mlp in zip(self.extractor, self.mlp)]


class SacActor(nn.Module):
    def __init__(
        self,
        extractor_cls: Type[Extractor],
        action_space: spaces.Box,
        observation_space: spaces.Space,
        mlp_cfg: MlpConfig,
    ):
        super().__init__()

        action_dim = action_space.shape[0]  # type: ignore

        self.extractor = extractor_cls(observation_space)
        self.mlp = MLP(
            input_sizes=self.extractor.output_size,
            output_sizes=(action_dim, action_dim),  # type: ignore
            mlp_cfg=mlp_cfg,
        )
        self.squashed_gaussian = SquashedGaussian(action_space)

    def forward(self, x: torch.Tensor, deterministic=False):
        e = self.extractor(x)
        mean, log_std = self.mlp(e)

        action, log_prob, stats = self.squashed_gaussian(mean, log_std, deterministic)

        return action, log_prob, stats


class SacTrainer(Trainer[SacTrainerConfig]):
    def __init__(
        self,
        cfg: SacTrainerConfig,
        val_env: gym.Env,
        output_path: str | Path,
        device: str,
        train_env: gym.Env,
        extractor_cls: Type[Extractor] | ExtractorName = "identity",
    ):
        """Initializes the trainer with a configuration, output path, and device.

        Args:
            cfg: The configuration for the trainer.
            val_env: The validation environment.
            output_path: The path to the output directory.
            device: The device on which the trainer is running
            train_env: The training environment.
            extractor_cls: The class used for extracting features from observations.
        """
        super().__init__(cfg, val_env, output_path, device)

        self.train_env = seed_env(wrap_env(train_env), seed=self.cfg.seed)
        action_space: spaces.Box = self.train_env.action_space  # type: ignore
        observation_space = self.train_env.observation_space

        if isinstance(extractor_cls, str):
            extractor_cls = get_extractor_cls(extractor_cls)

        self.q = SacCritic(
            extractor_cls,  # type: ignore
            action_space,
            observation_space,
            cfg.critic_mlp,
            cfg.num_critics,
        )
        self.q_target = SacCritic(
            extractor_cls,  # type: ignore
            action_space,
            observation_space,
            cfg.critic_mlp,
            cfg.num_critics,
        )
        self.q_target.load_state_dict(self.q.state_dict())
        self.q_optim = torch.optim.Adam(self.q.parameters(), lr=cfg.lr_q)

        self.pi = SacActor(
            extractor_cls, action_space, observation_space, cfg.actor_mlp
        )  # type: ignore
        self.pi_optim = torch.optim.Adam(self.pi.parameters(), lr=cfg.lr_pi)

        self.log_alpha = nn.Parameter(torch.tensor(cfg.init_alpha).log())  # type: ignore

        if self.cfg.lr_alpha is not None:
            self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=cfg.lr_alpha)  # type: ignore
            action_dim = np.prod(self.train_env.action_space.shape)  # type: ignore
            self.target_entropy = (
                -action_dim if cfg.target_entropy is None else cfg.target_entropy
            )
        else:
            self.alpha_optim = None
            self.target_entropy = None

        self.buffer = ReplayBuffer(cfg.buffer_size, device=device)

    def train_loop(self) -> Iterator[int]:
        is_terminated = is_truncated = True

        while True:
            if is_terminated or is_truncated:
                obs, _ = self.train_env.reset()
                is_terminated = is_truncated = False

            action, _, stats = self.act(obs)  # type: ignore
            self.report_stats("train_trajectory", {"action": action}, verbose=True)
            self.report_stats("train_policy_rollout", stats, verbose=True)  # type: ignore

            obs_prime, reward, is_terminated, is_truncated, info = self.train_env.step(
                action
            )

            if "episode" in info or "task" in info:
                self.report_stats(
                    "train", {**info.get("episode", {}), **info.get("task", {})}
                )

            # TODO (Jasper): Add is_truncated to buffer.
            self.buffer.put((obs, action, reward, obs_prime, is_terminated))  # type: ignore

            obs = obs_prime

            if (
                self.state.step >= self.cfg.train_start
                and len(self.buffer) >= self.cfg.batch_size
                and self.state.step % self.cfg.update_freq == 0
            ):
                # sample batch
                o, a, r, o_prime, te = self.buffer.sample(self.cfg.batch_size)

                # sample action
                a_pi, log_p, _ = self.pi(o)

                # update temperature
                if self.alpha_optim is not None:
                    alpha_loss = -torch.mean(
                        self.log_alpha.exp() * (log_p + self.target_entropy).detach()
                    )
                    self.alpha_optim.zero_grad()
                    alpha_loss.backward()
                    self.alpha_optim.step()

                # update critic
                alpha = self.log_alpha.exp().item()
                with torch.no_grad():
                    a_pi_prime, log_p_prime, _ = self.pi(o_prime)
                    q_target = torch.cat(self.q_target(o_prime, a_pi_prime), dim=1)
                    q_target = torch.min(q_target, dim=1, keepdim=True).values

                    # add entropy
                    q_target = (
                        q_target - alpha * log_p_prime * self.cfg.entropy_reward_bonus
                    )

                    target = r[:, None] + self.cfg.gamma * (1 - te[:, None]) * q_target

                q = torch.cat(self.q(o, a), dim=1)
                q_loss = torch.mean((q - target).pow(2))

                self.q_optim.zero_grad()
                q_loss.backward()
                self.q_optim.step()

                # update actor
                q_pi = torch.cat(self.q(o, a_pi), dim=1)
                min_q_pi = torch.min(q_pi, dim=1, keepdim=True).values
                pi_loss = (alpha * log_p - min_q_pi).mean()

                self.pi_optim.zero_grad()
                pi_loss.backward()
                self.pi_optim.step()

                # soft updates
                soft_target_update(self.q, self.q_target, self.cfg.tau)

                # report stats
                loss_stats = {
                    "q_loss": q_loss.item(),
                    "pi_loss": pi_loss.item(),
                    "alpha": alpha,
                    "q": q.mean().item(),
                    "q_target": target.mean().item(),
                    "entropy": -log_p.mean().item(),
                }
                self.report_stats("loss", loss_stats)

            yield 1

    def act(
        self, obs, deterministic: bool = False, state=None
    ) -> tuple[np.ndarray, None, dict[str, float]]:
        obs = self.buffer.collate([obs])
        with torch.no_grad():
            action, _, stats = self.pi(obs, deterministic=deterministic)
        return action.cpu().numpy()[0], None, stats

    @property
    def optimizers(self) -> list[torch.optim.Optimizer]:
        if self.alpha_optim is None:
            return [self.q_optim, self.pi_optim]

        return [self.q_optim, self.pi_optim, self.alpha_optim]

    def periodic_ckpt_modules(self) -> list[str]:
        return ["q", "pi", "q_target", "log_alpha"]

    def singleton_ckpt_modules(self) -> list[str]:
        return ["buffer"]
