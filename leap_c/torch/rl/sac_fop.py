"""Provides a trainer for a Soft Actor-Critic algorithm that uses a differentiable MPC
layer for the policy network."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, NamedTuple, Literal, Type

import gymnasium as gym
import gymnasium.spaces as spaces
import numpy as np
import torch
import torch.nn as nn

from leap_c.controller import ParameterizedController
from leap_c.torch.nn.extractor import Extractor, ExtractorName, get_extractor_cls
from leap_c.torch.nn.gaussian import SquashedGaussian, BoundedTransform
from leap_c.torch.nn.mlp import MLP, MlpConfig
from leap_c.torch.rl.buffer import ReplayBuffer
from leap_c.torch.rl.sac import SacTrainerConfig, SacCritic
from leap_c.torch.rl.utils import soft_target_update
from leap_c.trainer import Trainer
from leap_c.utils.gym import wrap_env, seed_env


@dataclass(kw_only=True)
class SacFopTrainerConfig(SacTrainerConfig):
    """Specific settings for the Fop trainer."""

    noise: Literal["param", "action"] = "param"
    entropy_correction: bool = False


class SacFopActorOutput(NamedTuple):
    param: torch.Tensor
    log_prob: torch.Tensor
    stats: dict[str, float]
    action: torch.Tensor
    status: torch.Tensor
    ctx: Any | None

    def select(self, mask: torch.Tensor) -> "SacFopActorOutput":
        return SacFopActorOutput(
            self.param[mask],
            self.log_prob[mask],
            None,  # type:ignore
            self.action[mask],
            self.status[mask],
            None,
        )


class FopActor(nn.Module):
    def __init__(
        self,
        extractor: Extractor,
        mlp_cfg: MlpConfig,
        controller: ParameterizedController,
        correction: bool = True,
    ):
        super().__init__()
        self.controller = controller
        self.extractor = extractor
        param_dim = controller.param_space.shape[0]
        self.mlp = MLP(
            input_sizes=self.extractor.output_size,
            output_sizes=(param_dim, param_dim),  # type:ignore
            mlp_cfg=mlp_cfg,
        )
        self.correction = correction
        self.squashed_gaussian = SquashedGaussian(controller.param_space)  # type:ignore

    def forward(self, obs, ctx=None, deterministic=False) -> SacFopActorOutput:
        e = self.extractor(obs)
        mean, log_std = self.mlp(e)

        param, log_prob, gaussian_stats = self.squashed_gaussian(
            mean, log_std, deterministic=deterministic
        )

        ctx, action = self.controller(obs, param, ctx=ctx)

        if self.correction:
            j = self.controller.jacobian_action_param(ctx)
            j = torch.from_numpy(j).to(param.device)  # type:ignore
            jtj = j @ j.transpose(1, 2)
            correction = (
                torch.det(jtj + 1e-3 * torch.eye(jtj.shape[1], device=jtj.device))
                .sqrt()
                .log()
            )
            log_prob -= correction.unsqueeze(1)

        return SacFopActorOutput(
            param,
            log_prob,
            {**gaussian_stats, **ctx.log},
            action,
            ctx.status,
            ctx,
        )


class FoaActor(nn.Module):
    def __init__(
        self,
        env: gym.Env,
        extractor: Extractor,
        mlp_cfg: MlpConfig,
        controller: ParameterizedController,
    ):
        super().__init__()
        self.env = env
        self.controller = controller
        self.extractor = extractor
        param_dim = controller.param_space.shape[0]  # type:ignore
        action_dim = env.action_space.shape[0]  # type:ignore
        self.mlp = MLP(
            input_sizes=self.extractor.output_size,
            output_sizes=(param_dim, action_dim),  # type:ignore
            mlp_cfg=mlp_cfg,
        )
        self.parameter_transform = BoundedTransform(
            self.controller.param_space  # type:ignore
        )  # type:ignore
        self.action_transform = BoundedTransform(self.env.action_space)  # type:ignore
        self.squashed_gaussian = SquashedGaussian(self.env.action_space)  # type:ignore

    def forward(self, obs, ctx=None, deterministic=False) -> SacFopActorOutput:
        e = self.extractor(obs)
        mean, log_std = self.mlp(e)
        param = self.parameter_transform(mean)

        ctx, action_mpc = self.controller(obs, param, ctx=ctx)
        action_unbounded = self.action_transform.inverse(action_mpc)
        action_squashed, log_prob, gaussian_stats = self.squashed_gaussian(
            action_unbounded, log_std, deterministic=deterministic
        )
        return SacFopActorOutput(
            param,
            log_prob,
            {**gaussian_stats, **ctx.log},
            action_squashed,
            ctx.status,
            ctx,
        )


class SacFopTrainer(Trainer[SacFopTrainerConfig]):
    def __init__(
        self,
        cfg: SacFopTrainerConfig,
        val_env: gym.Env,
        output_path: str | Path,
        device: str,
        train_env: gym.Env,
        controller: ParameterizedController,
        extractor_cls: Type[Extractor] | ExtractorName = "identity",
    ):
        """Initializes the SAC FOP trainer.

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
            train_env.action_space,
            observation_space,
            cfg.critic_mlp,
            cfg.num_critics,
        )
        self.q_target = SacCritic(
            extractor_cls,  # type: ignore
            train_env.action_space,
            observation_space,
            cfg.critic_mlp,
            cfg.num_critics,
        )
        self.q_target.load_state_dict(self.q.state_dict())
        self.q_optim = torch.optim.Adam(self.q.parameters(), lr=cfg.lr_q)

        if cfg.noise == "param":
            self.pi = FopActor(
                extractor_cls(observation_space),  # type: ignore
                cfg.actor_mlp,
                controller,
                correction=cfg.entropy_correction,
            )
        elif cfg.noise == "action":
            self.pi = FoaActor(
                train_env,
                extractor_cls(observation_space),  # type: ignore
                cfg.actor_mlp,
                controller,
            )
        else:
            raise ValueError(f"Unknown noise type: {cfg.noise}")

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

        self.buffer = ReplayBuffer(
            cfg.buffer_size, device=device, collate_fn_map=controller.collate_fn_map
        )

    def train_loop(self) -> Iterator[int]:
        is_terminated = is_truncated = True
        policy_state = None
        obs = None

        while True:
            if is_terminated or is_truncated:
                obs, _ = self.train_env.reset(options={"mode": "train"})
                policy_state = None
                is_terminated = is_truncated = False

            obs_batched = self.buffer.collate([obs])

            with torch.no_grad():
                # TODO (Jasper): Argument order is not consistent
                pi_output: SacFopActorOutput = self.pi(
                    obs_batched, policy_state, deterministic=False
                )
                action = pi_output.action.cpu().numpy()[0]
                param = pi_output.param.cpu().numpy()[0]

            self.report_stats(
                "train_trajectory", {"param": param, "action": action}, verbose=True
            )
            self.report_stats("train_policy_rollout", pi_output.stats, verbose=True)  # type: ignore

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
                    action,
                    reward,
                    obs_prime,
                    is_terminated,
                    pi_output.ctx,
                )
            )  # type: ignore

            obs = obs_prime
            policy_state = pi_output.ctx

            if (
                self.state.step >= self.cfg.train_start
                and len(self.buffer) >= self.cfg.batch_size
                and self.state.step % self.cfg.update_freq == 0
            ):
                # sample batch
                o, a, r, o_prime, te, ps_sol = self.buffer.sample(self.cfg.batch_size)

                # sample action
                pi_o = self.pi(o, ps_sol)
                with torch.no_grad():
                    pi_o_prime = self.pi(o_prime, ps_sol)

                pi_o_stats = pi_o.stats

                # compute mask
                mask_status = (pi_o.status == 0) & (pi_o_prime.status == 0)

                # reduce batch
                o = o[mask_status]
                a = a[mask_status]
                r = r[mask_status]
                o_prime = o_prime[mask_status]
                te = te[mask_status]
                pi_o = pi_o.select(mask_status)
                pi_o_prime = pi_o_prime.select(mask_status)

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
                    q_target = torch.cat(
                        self.q_target(o_prime, pi_o_prime.action), dim=1
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
                # mask_status = pi_o.status == 0
                q_pi = torch.cat(self.q(o, pi_o.action), dim=1)
                min_q_pi = torch.min(q_pi, dim=1).values
                pi_loss = (alpha * log_p - min_q_pi).mean()

                self.pi_optim.zero_grad()
                pi_loss.backward()
                self.pi_optim.step()

                # soft updates
                soft_target_update(self.q, self.q_target, self.cfg.tau)

                loss_stats = {
                    "q_loss": q_loss.item(),
                    "pi_loss": pi_loss.item(),
                    "alpha": alpha,
                    "q": q.mean().item(),
                    "q_target": target.mean().item(),
                    "masked_samples_perc": 1 - float(mask_status.mean().item()),
                    "entropy": -log_p.mean().item(),
                }
                self.report_stats("loss", loss_stats)
                self.report_stats("train_policy_update", pi_o_stats, verbose=True)

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
