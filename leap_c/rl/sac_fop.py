"""Provides a trainer for a Soft Actor-Critic algorithm that uses a differentiable MPC
layer for the policy network."""

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator

import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym

from leap_c.mpc import MPCBatchedState
from leap_c.nn.mlp import MLP, MLPConfig
from leap_c.nn.modules import MPCSolutionModule
from leap_c.registry import register_trainer
from leap_c.rl.replay_buffer import ReplayBuffer
from leap_c.rl.sac import SacAlgorithmConfig, SacCritic
from leap_c.task import Task
from leap_c.trainer import (
    BaseConfig,
    LogConfig,
    TrainConfig,
    Trainer,
    ValConfig,
)


LOG_STD_MIN = -4
LOG_STD_MAX = 2


@dataclass(kw_only=True)
class SacFopBaseConfig(BaseConfig):
    """Contains the necessary information for a Trainer.

    Attributes:
        sac: The Sac algorithm configuration.
        train: The training configuration.
        val: The validation configuration.
        log: The logging configuration.
        seed: The seed for the trainer.
    """

    sac: SacAlgorithmConfig = field(default_factory=SacAlgorithmConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    val: ValConfig = field(default_factory=ValConfig)
    log: LogConfig = field(default_factory=LogConfig)
    seed: int = 0


class MPCSacActor(nn.Module):
    def __init__(
        self,
        task: Task,
        env: gym.Env,
        mlp_cfg: MLPConfig,
        prepare_mpc_state: (
            Callable[[torch.Tensor, torch.Tensor, MPCBatchedState], MPCBatchedState]
            | None
        ) = None,
    ):
        super().__init__()

        param_space = task.param_space

        self.extractor = task.create_extractor(env)
        self.mlp = MLP(
            input_sizes=self.extractor.output_size,
            output_sizes=(param_space.shape[0], param_space.shape[0]),  # type:ignore
            mlp_cfg=mlp_cfg,
        )

        self.mpc: MPCSolutionModule = task.mpc  # type:ignore
        self.prepare_mpc_input = task.prepare_mpc_input
        self.prepare_mpc_state = prepare_mpc_state

        # add scaling params for tanh [-1, 1] -> [low, high]
        loc = (param_space.high + param_space.low) / 2.0  # type: ignore
        scale = (param_space.high - param_space.low) / 2.0  # type: ignore
        loc = torch.tensor(loc, dtype=torch.float32)
        scale = torch.tensor(scale, dtype=torch.float32)
        self.register_buffer("loc", loc)
        self.register_buffer("scale", scale)

    def forward(self, obs, mpc_state: MPCBatchedState, deterministic=False):
        e = self.extractor(obs)
        mean, log_std = self.mlp(e)

        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)

        if deterministic:
            action = mean
        else:
            action = mean + std * torch.randn_like(mean)

        log_prob = (
            -0.5 * ((action - mean) / std).pow(2) - log_std - np.log(np.sqrt(2) * np.pi)
        )

        action = torch.tanh(action)

        log_prob -= torch.log(self.scale[None, :] * (1 - action.pow(2)) + 1e-6)

        param = action * self.scale[None, :] + self.loc[None, :]

        # param_labels = self.mpc.mpc.param_labels

        # if action.shape[0] == 1:
        #     stats = {
        #         param_labels[k]: element for k, element in enumerate(param.squeeze())
        #     }
        #     self.trainer["trainer"].report_stats(
        #         "action",
        #         stats,
        #         self.trainer["trainer"].state.step,
        #     )

        mpc_input = self.prepare_mpc_input(obs, param)
        if self.prepare_mpc_state is not None:
            mpc_input, mpc_state = self.prepare_mpc_state(
                obs, param, mpc_state
            )  # type:ignore

        # TODO: We have to catch and probably replace the state_solution somewhere,
        #       if its not a converged solution
        mpc_output, state_solution, stats = self.mpc(mpc_input, mpc_state)

        return (
            mpc_output.u0,
            log_prob,
            mpc_output.status,
            state_solution,
            param,
            stats,
        )


@register_trainer("sac_fop", SacFopBaseConfig())
class SacFopTrainer(Trainer):
    cfg: SacFopBaseConfig

    def __init__(
        self, task: Task, output_path: str | Path, device: str, cfg: SacFopBaseConfig
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
        ).to(device)
        self.q_target = SacCritic(
            task, self.train_env, cfg.sac.critic_mlp, cfg.sac.num_critics
        ).to(device)
        self.q_target.load_state_dict(self.q.state_dict())
        self.q_optim = torch.optim.Adam(self.q.parameters(), lr=cfg.sac.lr_q)

        self.pi = MPCSacActor(task, self.train_env, cfg.sac.actor_mlp).to(device)
        self.pi_optim = torch.optim.Adam(self.pi.parameters(), lr=cfg.sac.lr_pi)

        self.log_alpha = nn.Parameter(torch.tensor(0.0))  # type: ignore
        self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=cfg.sac.lr_alpha)  # type: ignore

        self.buffer = ReplayBuffer(cfg.sac.buffer_size, device=device)

        self.to(device)

    def init_policy_state(self) -> Any:
        return (
            self.pi.mpc.mpc.ocp_solver.store_iterate_to_flat_obj()
        )  # State full of zeros

    def train_loop(self) -> Iterator[int]:
        is_terminated = is_truncated = True
        episode_return = episode_length = np.inf
        episode_act_stats = defaultdict(list)
        policy_state = self.init_policy_state()
        obs = None

        while True:
            if is_terminated or is_truncated:
                obs, _ = self.train_env.reset()
                if episode_length < np.inf:
                    mpc_episode_stats = {k: float(np.mean(v)) for k, v in episode_act_stats.items()}
                    episode_stats = {
                        "episode_return": episode_return,
                        "episode_length": episode_length,
                    }
                    self.report_stats("train", episode_stats, self.state.step)
                    self.report_stats("train_policy_rollout", mpc_episode_stats, self.state.step)
                policy_state = self.init_policy_state()
                is_terminated = is_truncated = False
                episode_return = episode_length = 0
                episode_act_stats = defaultdict(list)

            obs_batched = self.task.collate([obs], device=self.device)

            with torch.no_grad():
                action, _, _, policy_state_prime, _, act_stats = self.pi(
                    obs_batched, policy_state, deterministic=False
                )
                action = action.cpu().numpy()[0]

            for key, value in act_stats.items():
                episode_act_stats[key].append(value)

            obs_prime, reward, is_terminated, is_truncated, _ = self.train_env.step(
                action
            )

            episode_return += float(reward)
            episode_length += 1

            self.buffer.put(
                (
                    obs,
                    action,
                    reward,
                    obs_prime,
                    is_terminated,
                    is_truncated,
                    policy_state,
                    policy_state_prime,
                )
            )  # type: ignore

            obs = obs_prime
            policy_state = policy_state_prime

            if (
                self.state.step >= self.cfg.train.start
                and len(self.buffer) >= self.cfg.sac.batch_size
                and self.state.step % self.cfg.sac.update_freq == 0
            ):
                # sample batch
                o, a, r, o_prime, te, tr, ps, ps_prime = self.buffer.sample(
                    self.cfg.sac.batch_size
                )

                # sample action
                a_pi, log_p, status, state_sol, param, mpc_stats = self.pi(o, ps_prime)
                log_p = log_p.sum(dim=-1).unsqueeze(-1)

                # update temperature
                target_entropy = -np.prod(self.task.param_space.shape)  # type: ignore
                alpha_loss = -torch.mean(
                    self.log_alpha.exp() * (log_p + target_entropy).detach()
                )
                self.alpha_optim.zero_grad()
                alpha_loss.backward()
                self.alpha_optim.step()

                # update critic
                alpha = self.log_alpha.exp().item()
                with torch.no_grad():
                    a_pi_prime, log_p_prime, *_ = self.pi(o_prime, ps_prime)
                    q_target = torch.cat(self.q_target(o_prime, a_pi_prime), dim=1)
                    q_target = torch.min(q_target, dim=1).values

                    # add entropy
                    q_target = q_target - alpha * log_p_prime[:, 0]

                    target = r + self.cfg.sac.gamma * (1 - te) * q_target

                q = torch.cat(self.q(o, a), dim=1)
                q_loss = torch.mean((q - target[:, None]).pow(2))

                self.q_optim.zero_grad()
                q_loss.backward()
                self.q_optim.step()

                # update actor
                mask_status = status == 0
                q_pi = torch.cat(self.q(o, a_pi), dim=1)
                min_q_pi = torch.min(q_pi, dim=1).values
                pi_loss = (alpha * log_p - min_q_pi)[mask_status].mean()

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
                        "masked_samples": (status != 0).float().mean().item(),
                    }
                    self.report_stats("loss", loss_stats, self.state.step + 1)
                    self.report_stats("train_policy_update", mpc_stats, self.state.step + 1)

            yield 1

    def act(
        self, obs, deterministic: bool = False, state=None
    ) -> tuple[np.ndarray, Any, dict[str, float]]:
        obs = self.task.collate([obs], device=self.device)

        with torch.no_grad():
            action, _, _, state_prime, _, stats = self.pi(
                obs, state, deterministic=deterministic
            )

        action = action.cpu().numpy()[0]

        return action, state_prime, stats

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
