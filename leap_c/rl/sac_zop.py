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
from leap_c.nn.gaussian import SquashedGaussian
from leap_c.nn.mlp import MLP, MlpConfig
from leap_c.nn.modules import MpcSolutionModule
from leap_c.nn.utils import min_max_scaling
from leap_c.registry import register_trainer
from leap_c.rl.replay_buffer import ReplayBuffer
from leap_c.rl.utils import soft_target_update
from leap_c.rl.sac import SacBaseConfig
from leap_c.task import Task
from leap_c.trainer import Trainer


NUM_THREADS_ACADOS_BATCH = 4


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
        self.param_space = task.param_space

    def forward(self, x: torch.Tensor, p: torch.Tensor):
        p_norm = min_max_scaling(p, self.param_space)  # type: ignore
        return [mlp(qe(x), p_norm) for qe, mlp in zip(self.extractor, self.mlp)]


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


@register_trainer("sac_zop", SacBaseConfig())
class SacZopTrainer(Trainer):
    cfg: SacBaseConfig

    def __init__(
        self, task: Task, output_path: str | Path, device: str, cfg: SacBaseConfig
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
        self.pi.mpc.mpc.num_threads_batch_methods = NUM_THREADS_ACADOS_BATCH
        self.pi_optim = torch.optim.Adam(self.pi.parameters(), lr=cfg.sac.lr_pi)

        self.log_alpha = nn.Parameter(torch.tensor(cfg.sac.init_alpha).log())  # type: ignore

        action_dim = np.prod(self.train_env.action_space.shape)  # type: ignore
        param_dim = np.prod(task.param_space.shape)  # type: ignore
        self.entropy_norm = param_dim / action_dim
        if cfg.sac.lr_alpha is not None:
            self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=cfg.sac.lr_alpha)  # type: ignore
            self.target_entropy = (
                -action_dim
                if cfg.sac.target_entropy is None
                else cfg.sac.target_entropy
            )
        else:
            self.alpha_optim = None
            self.target_entropy = None

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

            obs_prime, reward, is_terminated, is_truncated, info = self.train_env.step(
                action
            )

            if "episode" in info:
                stats = info["episode"]
                if "task" in info:
                    stats.update(info["task"])
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
                    pi_o_prime = self.pi(o_prime, None, only_param=True)
                    q_target = torch.cat(
                        self.q_target(o_prime, pi_o_prime.param), dim=1
                    )
                    q_target = torch.min(q_target, dim=1, keepdim=True).values

                    # add entropy
                    factor = self.cfg.sac.entropy_reward_bonus / self.entropy_norm
                    q_target = q_target - alpha * pi_o_prime.log_prob * factor

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
                min_q_pi = torch.min(q_pi, dim=1, keepdim=True).values
                pi_loss = (alpha * log_p - min_q_pi).mean()

                self.pi_optim.zero_grad()
                pi_loss.backward()
                self.pi_optim.step()

                # soft updates
                soft_target_update(self.q, self.q_target, self.cfg.sac.tau)

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
        if self.alpha_optim is None:
            return [self.q_optim, self.pi_optim]

        return [self.q_optim, self.pi_optim, self.alpha_optim]

    def periodic_ckpt_modules(self) -> list[str]:
        return ["q", "pi", "q_target"]

    def singleton_ckpt_modules(self) -> list[str]:
        return ["buffer"]
