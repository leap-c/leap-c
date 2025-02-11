"""Provides a trainer for a Soft Actor-Critic algorithm that uses a differentiable MPC
layer for the policy network."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
import torch.nn as nn

from leap_c.mpc import MPCBatchedState
from leap_c.nn.gaussian import Gaussian
from leap_c.nn.modules import MPCSolutionModule
from leap_c.nn.mlp import MLP, MLPConfig
from leap_c.rl.replay_buffer import ReplayBuffer
from leap_c.registry import register_trainer
from leap_c.task import Task
from leap_c.rl.sac import SACCritic, SACAlgorithmConfig
from leap_c.trainer import (
    BaseConfig,
    LogConfig,
    TrainConfig,
    Trainer,
    ValConfig,
)


@dataclass(kw_only=True)
class SACFOUBaseConfig(BaseConfig):
    """Contains the necessary information for a Trainer.

    Attributes:
        sac: The SAC algorithm configuration.
        train: The training configuration.
        val: The validation configuration.
        log: The logging configuration.
        seed: The seed for the trainer.
    """

    sac: SACAlgorithmConfig = field(default_factory=SACAlgorithmConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    val: ValConfig = field(default_factory=ValConfig)
    log: LogConfig = field(default_factory=LogConfig)
    seed: int = 0


class MPCSACActor(nn.Module):
    def __init__(
        self,
        task,
        mlp_cfg: MLPConfig,
    ):
        super().__init__()

        param_space = task.param_space

        self.extractor = task.create_extractor()
        self.mlp = MLP(
            input_sizes=self.extractor.output_size,
            output_sizes=(param_space.shape[0], param_space.shape[0]),  # type: ignore
            mlp_cfg=mlp_cfg,
        )
        self.gaussian = Gaussian(param_space)
        self.mpc = task.mpc
        self.prepare_mpc_input = task.prepare_mpc_input

    def forward(self, obs, mpc_state: MPCBatchedState, deterministic=False):
        e = self.extractor(obs)
        mean, log_std = self.mlp(e)

        param, log_prob = self.gaussian(mean, log_std, deterministic=deterministic)

        mpc_input = self.prepare_mpc_input(obs, param)
        mpc_output, stats = self.mpc(mpc_input, mpc_state)

        return mpc_output.u0, log_prob, stats


@register_trainer("sac_fou", SACFOUBaseConfig())
class SACFOUTrainer(Trainer):
    cfg: SACFOUBaseConfig

    def __init__(
        self, task: Task, output_path: str | Path, device: str, cfg: SACFOUBaseConfig
    ):
        """Initializes the trainer with a configuration, output path, and device.

        Args:
            task: The task to be solved by the trainer.
            output_path: The path to the output directory.
            device: The device on which the trainer is running
            cfg: The configuration for the trainer.
        """
        super().__init__(task, output_path, device, cfg)

        self.q = SACCritic(task, cfg.sac.critic_mlp, cfg.sac.num_critics).to(device)
        self.q_target = SACCritic(task, cfg.sac.critic_mlp, cfg.sac.num_critics).to(
            device
        )
        self.q_target.load_state_dict(self.q.state_dict())
        self.q_optim = torch.optim.Adam(self.q.parameters(), lr=cfg.sac.lr_q)

        self.pi = MPCSACActor(task, cfg.sac.actor_mlp).to(device)
        self.pi_optim = torch.optim.Adam(self.pi.parameters(), lr=cfg.sac.lr_pi)

        self.log_alpha = nn.Parameter(torch.tensor(0.0))  # type: ignore
        self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=cfg.sac.lr_alpha)  # type: ignore

        self.buffer = ReplayBuffer(cfg.sac.buffer_size, device=device)

        self.to(device)

    def train_loop(self) -> Iterator[int]:
        is_terminated = is_truncated = True
        episode_return = episode_length = np.inf
        policy_state = None

        while True:
            if is_terminated or is_truncated:
                obs, _ = self.train_env.reset()
                if episode_length < np.inf:
                    stats = {
                        "episode_return": episode_return,
                        "episode_length": episode_length,
                    }
                    self.report_stats("train", stats, self.state.step)
                policy_state = self.init_policy_state()
                is_terminated = is_truncated = False
                episode_return = episode_length = 0

            action, policy_state_prime = self.act(obs, policy_state)  # type: ignore
            obs_prime, reward, is_terminated, is_truncated, _ = self.train_env.step(
                action
            )

            episode_return += float(reward)
            episode_length += 1

            # TODO (Jasper): Add is_truncated to buffer.
            self.buffer.put((obs, policy_state, action, reward, obs_prime, policy_state_prime, is_terminated))  # type: ignore

            obs = obs_prime

            if (
                self.state.step >= self.cfg.train.start
                and len(self.buffer) >= self.cfg.sac.batch_size
                and self.state.step % self.cfg.sac.update_freq == 0
            ):
                # sample batch
                o, ps, a, r, o_prime, ps_prime, te = self.buffer.sample(
                    self.cfg.sac.batch_size
                )

                # sample action
                a_pi, log_p = self.pi(o, ps)
                log_p = log_p.sum(dim=-1).unsqueeze(-1)

                # update temperature
                target_entropy = -np.prod(self.train_env.action_space.shape)  # type: ignore
                alpha_loss = -torch.mean(
                    self.log_alpha.exp() * (log_p + target_entropy).detach()
                )
                self.alpha_optim.zero_grad()
                alpha_loss.backward()
                self.alpha_optim.step()

                # update critic
                alpha = self.log_alpha.exp().item()
                with torch.no_grad():
                    a_pi_prime, _ = self.pi(o_prime, ps_prime)
                    q_target = torch.cat(self.q_target(o_prime, a_pi_prime), dim=1)
                    q_target = torch.min(q_target, dim=1).values
                    # add entropy
                    q_target = q_target - alpha * log_p

                    target = r + self.cfg.sac.gamma * (1 - te) * q_target

                q = torch.cat(self.q(o, a), dim=1)
                q_loss = torch.mean((q - target[:, None]).pow(2))

                self.q_optim.zero_grad()
                q_loss.backward()
                self.q_optim.step()

                # update actor
                q_pi = torch.cat(self.q(o, a_pi), dim=1)
                min_q_pi = torch.min(q_pi, dim=1).values
                pi_loss = (alpha * log_p - min_q_pi).mean()

                self.pi_optim.zero_grad()
                pi_loss.backward()
                self.pi_optim.step()

                # soft updates
                for q, q_target in zip(self.q.parameters(), self.q_target.parameters()):
                    q_target.data = (
                        self.cfg.sac.tau * q.data
                        + (1 - self.cfg.sac.tau) * q_target.data
                    )

                report_freq = self.cfg.sac.report_loss_freq * self.cfg.sac.update_freq

                if self.state.step % report_freq == 0:
                    loss_stats = {
                        "q_loss": q_loss.item(),
                        "pi_loss": pi_loss.item(),
                        "alpha": alpha,
                        "q": q.mean().item(),
                        "q_target": target.mean().item(),
                    }
                    self.report_stats("loss", loss_stats, self.state.step + 1)

            yield 1

    def act(
        self, obs, deterministic: bool = False, state=None
    ) -> tuple[np.ndarray, Any | None]:
        obs = self.task.collate([obs], device=self.device)

        with torch.no_grad():
            action, state, _ = self.pi(obs, state, deterministic=deterministic)

        return action.cpu().numpy(), state

    @property
    def optimizers(self) -> list[torch.optim.Optimizer]:
        return [self.q_optim, self.pi_optim, self.alpha_optim]

    def save(self) -> None:
        """Save the trainer state in a checkpoint folder."""

        torch.save(self.buffer, self.output_path / "buffer.pt")
        return super().save()

    def load(self) -> None:
        """Loads the state of a trainer from the output_path."""

        self.buffer = torch.load(self.output_path / "buffer.pt")
        return super().load()
