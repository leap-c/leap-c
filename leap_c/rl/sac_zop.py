"""Provides a trainer for a Soft Actor-Critic algorithm that uses a differentiable MPC
layer for the policy network."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, NamedTuple

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn

from leap_c.mpc import MpcBatchedState
from leap_c.nn.mlp import MLP, MlpConfig
from leap_c.nn.modules import MpcSolutionModule
from leap_c.nn.gaussian import SquashedGaussian
from leap_c.registry import register_trainer
from leap_c.rl.replay_buffer import ReplayBuffer
from leap_c.task import Task
from leap_c.trainer import (
    BaseConfig,
    LogConfig,
    TrainConfig,
    Trainer,
    ValConfig,
)


@dataclass(kw_only=True)
class SacZopAlgorithmConfig:
    """Contains the necessary information for a SacTrainer.

    Attributes:
        critic_mlp: The configuration for the critic networks.
        actor_mlp: The configuration for the policy network.
        batch_size: The batch size for training.
        buffer_size: The size of the replay buffer.
        gamma: The discount factor.
        tau: The soft update factor.
        soft_update_freq: The frequency of soft updates.
        lr_q: The learning rate for the Q networks.
        lr_pi: The learning rate for the policy network.
        lr_alpha: The learning rate for the temperature parameter.
        init_alpha: The initial value for the temperature parameter.
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
    lr_pi: float = 3e-4  # this can be lowered to make it more stable
    lr_alpha: float = 1e-3  # 1e-4
    init_alpha: float = 0.1
    entropy_reward_bonus: bool = True
    num_critics: int = 2
    report_loss_freq: int = 100
    update_freq: int = 1


@dataclass(kw_only=True)
class SacZopBaseConfig(BaseConfig):
    """Contains the necessary information for a Trainer.

    Attributes:
        sac: The Sac algorithm configuration.
        train: The training configuration.
        val: The validation configuration.
        log: The logging configuration.
        seed: The seed for the trainer.
    """

    sac: SacZopAlgorithmConfig = field(default_factory=SacZopAlgorithmConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    val: ValConfig = field(default_factory=ValConfig)
    log: LogConfig = field(default_factory=LogConfig)
    seed: int = 0


class SacCritic(nn.Module):
    def __init__(
        self,
        task: Task,
        env: gym.Env,
        mlp_cfg: MlpConfig,
        num_critics: int,
    ):
        super().__init__()

        action_dim = task.param_space.shape[0]  # type: ignore

        self.extractor = nn.ModuleList(
            [task.create_extractor(env) for _ in range(num_critics)]
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

    def forward(self, x: torch.Tensor, a: torch.Tensor):
        return [mlp(qe(x), a) for qe, mlp in zip(self.extractor, self.mlp)]


class SacZopActorOutput(NamedTuple):
    param: torch.Tensor
    log_prob: torch.Tensor
    stats: dict[str, float]
    action: torch.Tensor | None = None
    status: torch.Tensor | None = None
    state_solution: MpcBatchedState | None = None


class MpcSacActor(nn.Module):
    def __init__(
        self,
        task: Task,
        env: gym.Env,
        mlp_cfg: MlpConfig,
    ):
        super().__init__()

        param_space = task.param_space

        self.extractor = task.create_extractor(env)
        self.mlp = MLP(
            input_sizes=self.extractor.output_size,
            output_sizes=(param_space.shape[0], param_space.shape[0]),  # type:ignore
            mlp_cfg=mlp_cfg,
        )

        self.mpc: MpcSolutionModule = task.mpc  # type:ignore
        self.prepare_mpc_input = task.prepare_mpc_input

        self.squashed_gaussian = SquashedGaussian(param_space)  # type:ignore

    def forward(
        self,
        obs,
        mpc_state: None | MpcBatchedState,
        deterministic: bool = False,
        only_param: bool = False,
    ) -> SacZopActorOutput:
        e = self.extractor(obs)
        mean, log_std = self.mlp(e)

        param, log_prob, gauss_stats = self.squashed_gaussian(
            mean, log_std, deterministic
        )

        if only_param:
            return SacZopActorOutput(param, log_prob, gauss_stats)

        mpc_input = self.prepare_mpc_input(obs, param)

        # TODO: We have to catch and probably replace the state_solution somewhere,
        #       if its not a converged solution
        with torch.no_grad():
            mpc_output, state_solution, mpc_stats = self.mpc(mpc_input, mpc_state)

        return SacZopActorOutput(
            param,
            log_prob,
            {**gauss_stats, **mpc_stats},
            mpc_output.u0,
            mpc_output.status,
            state_solution,  # type: ignore
        )


@register_trainer("sac_zop", SacZopBaseConfig())
class SacZopTrainer(Trainer):
    cfg: SacZopBaseConfig

    def __init__(
        self, task: Task, output_path: str | Path, device: str, cfg: SacZopBaseConfig
    ):
        """Initializes the trainer with a configuration, output path, and device.

        Args:
            task: The task to be solved by the trainer.
            output_path: The path to the output directory.
            device: The device on which the trainer is running
            cfg: The configuration for the trainer.
        """
        super().__init__(task, output_path, device, cfg)

        self.q = SacCritic(
            task, self.train_env, cfg.sac.critic_mlp, cfg.sac.num_critics
        )
        self.q_target = SacCritic(
            task, self.train_env, cfg.sac.critic_mlp, cfg.sac.num_critics
        )
        self.q_target.load_state_dict(self.q.state_dict())
        self.q_optim = torch.optim.Adam(self.q.parameters(), lr=cfg.sac.lr_q)

        self.pi = MpcSacActor(task, self.train_env, cfg.sac.actor_mlp)
        self.pi_optim = torch.optim.Adam(self.pi.parameters(), lr=cfg.sac.lr_pi)

        self.log_alpha = nn.Parameter(torch.tensor(cfg.sac.init_alpha).log())  # type: ignore
        self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=cfg.sac.lr_alpha)  # type: ignore
        action_dim = np.prod(self.train_env.action_space.shape)  # type: ignore
        param_dim = np.prod(task.param_space.shape)  # type: ignore
        self.target_normalized_entropy = -action_dim
        self.entropy_norm = param_dim / action_dim

        self.buffer = ReplayBuffer(cfg.sac.buffer_size, device=device)

    def train_loop(self) -> Iterator[int]:
        is_terminated = is_truncated = True
        policy_state = None
        obs = None

        while True:
            if is_terminated or is_truncated:
                obs, _ = self.train_env.reset()
                policy_state = None
                is_terminated = is_truncated = False

            obs_batched = self.task.collate([obs], device=self.device)

            with torch.no_grad():
                pi_output = self.pi(obs_batched, policy_state, deterministic=False)
                action = pi_output.action.cpu().numpy()[0]
                param = pi_output.param.cpu().numpy()[0]

            self.report_stats("train_trajectory", {"action": action, "param": param})
            self.report_stats("train_policy_rollout", pi_output.stats)


            obs_prime, reward, is_terminated, is_truncated, info = self.train_env.step(action)
  
            if "episode" in info:
                self.report_stats("train", info["episode"])

            self.buffer.put(
                (
                    obs,
                    param,
                    reward,
                    obs_prime,
                    is_terminated,
                )
            )  # type: ignore

            obs = obs_prime
            policy_state = pi_output.state_solution

            if (
                self.state.step >= self.cfg.train.start
                and len(self.buffer) >= self.cfg.sac.batch_size
                and self.state.step % self.cfg.sac.update_freq == 0
            ):
                # sample batch
                o, a, r, o_prime, te = self.buffer.sample(self.cfg.sac.batch_size)

                # sample action
                pi_o = self.pi(o, None, only_param=True)
                a_pi = pi_o.param
                log_p = pi_o.log_prob / self.entropy_norm

                # update temperature
                alpha_loss = -torch.mean(
                    self.log_alpha.exp()
                    * (log_p + self.target_normalized_entropy).detach()
                )
                self.alpha_optim.zero_grad()
                alpha_loss.backward()
                self.alpha_optim.step()

                # update critic
                alpha = self.log_alpha.exp().item()
                with torch.no_grad():
                    pi_o_prime = self.pi(o_prime, None, only_param=True)
                    q_target = torch.cat(
                        self.q_target(o_prime, pi_o_prime.param), dim=1
                    )
                    q_target = torch.min(q_target, dim=1, keepdim=True).values

                    # add entropy
                    factor = self.cfg.sac.entropy_reward_bonus / self.entropy_norm
                    q_target = (
                        q_target - alpha * pi_o_prime.log_prob * factor
                    )

                    target = (
                        r[:, None] + self.cfg.sac.gamma * (1 - te[:, None]) * q_target
                    )

                q = torch.cat(self.q(o, a), dim=1)
                q_loss = torch.mean((q - target).pow(2))

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
                        "entropy": -log_p.mean().item(),
                    }
                    self.report_stats("loss", loss_stats, self.state.step + 1)

            yield 1

    def act(
        self, obs, deterministic: bool = False, state=None
    ) -> tuple[np.ndarray, Any, dict[str, float]]:
        obs = self.task.collate([obs], device=self.device)

        with torch.no_grad():
            pi_output = self.pi(obs, state, deterministic=deterministic)

        action = pi_output.action.cpu().numpy()[0]

        return action, pi_output.state_solution, pi_output.stats

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
