"""Provides a trainer for a Soft Actor-Critic algorithm that uses a differentiable MPC
layer for the policy network."""

from pathlib import Path
from typing import Any, Iterator, NamedTuple, Type

import gymnasium as gym
import gymnasium.spaces as spaces
import numpy as np
import torch
import torch.nn as nn

from leap_c.controller import ParameterizedController
from leap_c.torch.nn.extractor import Extractor, ExtractorName, get_extractor_cls
from leap_c.torch.nn.gaussian import SquashedGaussian
from leap_c.torch.nn.mlp import Mlp, MlpConfig
from leap_c.torch.rl.buffer import ReplayBuffer
from leap_c.torch.rl.sac import SacCritic, SacTrainerConfig
from leap_c.torch.rl.utils import soft_target_update
from leap_c.torch.utils.seed import mk_seed
from leap_c.trainer import Trainer
from leap_c.utils.gym import seed_env, wrap_env


class SacZopActorOutput(NamedTuple):
    """
    Attributes:
        param: The predicted parameters (which have been input into the controller).
        log_prob: The log-probability of the distribution that led to the action.
            NOTE: This log-probability is just a proxy for the true log-probability of the action,
            it is actually the log probability of the parameters that were
            input into the controller.
        stats: A dictionary containing several statistics of internal modules.
        action: The action output by the controller.
        ctx: The context object containing information about the MPC solve.
    """

    param: torch.Tensor
    log_prob: torch.Tensor
    stats: dict[str, float]
    action: torch.Tensor | None = None
    ctx: Any = None


class MpcSacActor(nn.Module):
    """An actor module for SAC-ZOP, containing a ParameterizedController to compute actions,
    but not differentiating through it, and injecting noise in the parameter space.

    Attributes:
        extractor: The feature extractor module.
        controller: The parameterized controller.
        mlp: The MLP module that predicts the parameters of the Gaussian distribution.
        squashed_gaussian: The squashed Gaussian distribution module.
    """

    extractor: Extractor
    controller: ParameterizedController
    mlp: Mlp
    squashed_gaussian: SquashedGaussian

    def __init__(
        self,
        extractor_cls: Type[Extractor],
        observation_space: gym.Space,
        controller: ParameterizedController,
        mlp_cfg: MlpConfig,
    ):
        """
        Args:
            extractor_cls: The class used for extracting features from observations.
            observation_space: The observation space used to configure the extractor.
            controller: The differentiable parameterized controller used to compute actions
                from parameters.
            mlp_cfg: The configuration for the MLP used to predict parameters.
        """
        super().__init__()

        param_space: spaces.Box = controller.param_space  # type:ignore

        self.extractor = extractor_cls(observation_space)
        self.controller = controller
        self.mlp = Mlp(
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
        """The given observations are passed to the extractor to obtain features.
        These are used to predict a distribution in the (learnable)
        parameter space of the controller using the MLP. Afterwards, this parameters are sampled
        from this distribution, and passed to the controller, which then computes the final actions.
        This forward pass does NOT support differentiation through the controller.

        Args:
            obs: The observations to compute the actions for.
            ctx: The optional context object containing information about the
                previous controller solve. Can be used, e.g., to warm-start the solver.
            deterministic: If true, use the mean of the distribution instead of sampling.
            only_param: If true, only return the predicted parameters and
                log-probabilities, but do not compute the action using the controller.
        """
        e = self.extractor(obs)
        mean, log_std = self.mlp(e)

        param, log_prob, gauss_stats = self.squashed_gaussian(mean, log_std, deterministic)

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
    """A trainer that implements Soft Actor-Critic (SAC) with a controller
    in the policy network, but without differentiating through it (SAC-ZOP).
    Uses parameter noise and a parameter critic.

    Attributes:
        train_env: The training environment.
        q: The Q-function approximator (critic).
        q_target: The target Q-function approximator.
        q_optim: The optimizer for the Q-function.
        pi: The policy network containing the parameterized controller (the actor).
        pi_optim: The optimizer for the policy network.
        log_alpha: The log of the temperature parameter.
        alpha_optim: The optimizer for the temperature parameter.
            Is None, if the temperature is fixed.
        target_entropy: The target entropy for the policy.
            Is None, if the temperature is fixed.
        entropy_norm: The normalization factor for the entropy term.
            Normalizes the entropy based on the ratio of parameter and action dimensions.
        buffer: The replay buffer used to store transitions.
    """

    train_env: gym.Env
    q: SacCritic
    q_target: SacCritic
    q_optim: torch.optim.Optimizer
    pi: MpcSacActor
    pi_optim: torch.optim.Optimizer
    log_alpha: nn.Parameter
    alpha_optim: torch.optim.Optimizer | None
    target_entropy: float | None
    entropy_norm: float
    buffer: ReplayBuffer

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
        """Initializes the SAC-ZOP trainer.

        Args:
            cfg: The configuration for the trainer.
            val_env: The validation environment.
            output_path: The path to the output directory.
            device: The device on which the trainer is running.
            train_env: The training environment.
            controller: The controller to use in the policy.
            extractor_cls: The class used for extracting features from observations.
        """
        super().__init__(cfg, val_env, output_path, device)

        param_space: spaces.Box = controller.param_space  # type: ignore
        observation_space = train_env.observation_space
        action_dim = np.prod(train_env.action_space.shape)  # type: ignore
        param_dim = np.prod(param_space.shape)

        self.train_env = wrap_env(train_env)

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
            self.target_entropy = -action_dim if cfg.target_entropy is None else cfg.target_entropy
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
                obs, _ = seed_env(self.train_env, mk_seed(self.rng), {"mode": "train"})
                policy_ctx = None
                is_terminated = is_truncated = False

            obs_batched = self.buffer.collate([obs])

            with torch.no_grad():
                pi_output = self.pi(obs_batched, policy_ctx, deterministic=False)
                action = pi_output.action.cpu().numpy()[0]
                param = pi_output.param.cpu().numpy()[0]

            self.report_stats("train_trajectory", {"action": action, "param": param}, verbose=True)
            self.report_stats("train_policy_rollout", pi_output.stats, verbose=True)

            obs_prime, reward, is_terminated, is_truncated, info = self.train_env.step(action)

            if "episode" in info or "task" in info:
                self.report_stats("train", {**info.get("episode", {}), **info.get("task", {})})

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
                    q_target = torch.cat(self.q_target(o_prime, pi_o_prime.param), dim=1)
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
