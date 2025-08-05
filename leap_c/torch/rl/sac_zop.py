"""Provides a trainer for a Soft Actor-Critic algorithm that uses a differentiable MPC
layer for the policy network."""

from pathlib import Path
from typing import Any, Iterator, NamedTuple, Type

import gymnasium.spaces as spaces
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn

from leap_c.controller import ParameterizedController
from leap_c.torch.nn.extractor import Extractor, ExtractorName, get_extractor_cls
from leap_c.torch.nn.gaussian import SquashedGaussian
from leap_c.torch.nn.mlp import MLP, MlpConfig
from leap_c.torch.rl.buffer import ReplayBuffer
from leap_c.torch.rl.utils import soft_target_update
from leap_c.torch.rl.sac import SacTrainerConfig, SacCritic
from leap_c.trainer import Trainer
from leap_c.utils.gym import wrap_env, seed_env


class SacZopActorOutput(NamedTuple):
    param: torch.Tensor
    log_prob: torch.Tensor
    stats: dict[str, float]
    action: torch.Tensor | None = None
    ctx: Any = None


class MpcSacActor(nn.Module):
    def __init__(
        self,
        extractor_cls: Type[Extractor],
        observation_space: gym.Space,
        controller: ParameterizedController,
        mlp_cfg: MlpConfig,
    ):
        super().__init__()

        param_space: spaces.Box = controller.param_space  # type:ignore

        self.extractor = extractor_cls(observation_space)
        self.controller = controller
        self.mlp = MLP(
            input_sizes=self.extractor.output_size,
            output_sizes=(param_space.shape[0], param_space.shape[0]),  # type:ignore
            mlp_cfg=mlp_cfg,
        )
        self.squashed_gaussian = SquashedGaussian(param_space)  # type:ignore

    def forward(
        self,
        obs,
        ctx=None,
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

        with torch.no_grad():
            ctx, action = self.controller(obs, param, ctx=ctx)

        return SacZopActorOutput(
            param,
            log_prob,
            gauss_stats,
            action,
            ctx,
        )


class SacZopTrainer(Trainer[SacTrainerConfig]):
    def __init__(
        self,
        cfg: SacTrainerConfig,
        val_env: gym.Env,
        output_path: str | Path,
        device: str,
        train_env: gym.Env,
        controller: ParameterizedController,
        extractor_cls: Type[Extractor] | ExtractorName = "identity",
    ):
        """Initializes the SAC ZOP trainer.

        Args:
            cfg: The configuration for the trainer.
            val_env: The validation environment.
            output_path: The path to the output directory.
            device: The device on which the trainer is running
            train_env: The training environment.
            controller: The controller to use for the policy.
            extractor_cls: The class used for extracting features from observations.
        """
        super().__init__(cfg, val_env, output_path, device)

        param_space: spaces.Box = controller.param_space  # type: ignore
        observation_space = train_env.observation_space
        action_dim = np.prod(train_env.action_space.shape)  # type: ignore
        param_dim = np.prod(param_space.shape)

        self.train_env = seed_env(wrap_env(train_env), seed=self.cfg.seed)
        self.controller = controller

        if isinstance(extractor_cls, str):
            extractor_cls = get_extractor_cls(extractor_cls)

        self.q = SacCritic(
            extractor_cls,  # type: ignore
            param_space,
            observation_space,
            cfg.critic_mlp,
            cfg.num_critics,
        )
        self.q_target = SacCritic(
            extractor_cls,  # type: ignore
            param_space,
            observation_space,
            cfg.critic_mlp,
            cfg.num_critics,
        )
        self.q_target.load_state_dict(self.q.state_dict())
        self.q_optim = torch.optim.Adam(self.q.parameters(), lr=cfg.lr_q)

        self.pi = MpcSacActor(
            extractor_cls,  # type: ignore
            observation_space,
            controller,
            cfg.actor_mlp,
        )
        self.pi_optim = torch.optim.Adam(self.pi.parameters(), lr=cfg.lr_pi)

        self.log_alpha = nn.Parameter(torch.tensor(cfg.init_alpha).log())  # type: ignore

        self.entropy_norm = param_dim / action_dim
        if cfg.lr_alpha is not None:
            self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=cfg.lr_alpha)  # type: ignore
            self.target_entropy = (
                -action_dim if cfg.target_entropy is None else cfg.target_entropy
            )
        else:
            self.alpha_optim = None
            self.target_entropy = None

        self.buffer = ReplayBuffer(cfg.buffer_size, device=device)

    def train_loop(self) -> Iterator[int]:
        is_terminated = is_truncated = True
        policy_ctx = None
        obs = None

        while True:
            if is_terminated or is_truncated:
                obs, _ = self.train_env.reset(options={"mode": "train"})
                policy_ctx = None
                is_terminated = is_truncated = False

            obs_batched = self.buffer.collate([obs])

            with torch.no_grad():
                pi_output = self.pi(obs_batched, policy_ctx, deterministic=False)
                action = pi_output.action.cpu().numpy()[0]
                param = pi_output.param.cpu().numpy()[0]

            self.report_stats(
                "train_trajectory", {"action": action, "param": param}, verbose=True
            )
            self.report_stats("train_policy_rollout", pi_output.stats, verbose=True)

            obs_prime, reward, is_terminated, is_truncated, info = self.train_env.step(
                action
            )

            if "episode" in info or "task" in info:
                self.report_stats(
                    "train", {**info.get("episode", {}), **info.get("task", {})}
                )

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
            policy_ctx = pi_output.ctx

            if (
                self.state.step >= self.cfg.train_start
                and len(self.buffer) >= self.cfg.batch_size
                and self.state.step % self.cfg.update_freq == 0
            ):
                # sample batch
                o, a, r, o_prime, te = self.buffer.sample(self.cfg.batch_size)

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
                    factor = self.cfg.entropy_reward_bonus / self.entropy_norm
                    q_target = q_target - alpha * pi_o_prime.log_prob * factor

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
                self.report_stats("loss", loss_stats, verbose=True)

            yield 1

    def act(
        self, obs, deterministic: bool = False, state=None
    ) -> tuple[np.ndarray, Any, dict[str, float]]:
        obs = self.buffer.collate([obs])

        with torch.no_grad():
            pi_output = self.pi(obs, state, deterministic=deterministic)

        action = pi_output.action.cpu().numpy()[0]

        return action, pi_output.ctx, pi_output.stats

    @property
    def optimizers(self) -> list[torch.optim.Optimizer]:
        if self.alpha_optim is None:
            return [self.q_optim, self.pi_optim]

        return [self.q_optim, self.pi_optim, self.alpha_optim]

    def periodic_ckpt_modules(self) -> list[str]:
        return ["q", "pi", "q_target", "log_alpha"]

    def singleton_ckpt_modules(self) -> list[str]:
        return ["buffer"]
