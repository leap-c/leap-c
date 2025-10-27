"""Provides a trainer for a Deep Q-Network (DQN) algorithm that uses a differentiable MPC layer in
the critic network.

In particular, the action-value function approximator consists of a multi-layer perceptron (MLP)
that provides the parametrization of an MPC controller conditioned on the current observation, and a
last layer made of a differentiable MPC controller. The controller then computes the estimated
optimal action-value function and policy.

Note that, by leveraging an MPC in the last layer of the action-value function approximator, this
DQN-MPC implementation can deal with continuous action spaces."""

from dataclasses import dataclass
from pathlib import Path
from typing import Generator, NamedTuple
from warnings import warn

import numpy as np
import torch
from gymnasium import Env
from gymnasium.spaces import Box
from torch import Tensor, nn
from torch.nn.functional import mse_loss

from leap_c.controller import ParameterizedController
from leap_c.ocp.acados.diff_mpc import AcadosDiffMpcCtx
from leap_c.torch.nn.bounded_distributions import BoundedTransform
from leap_c.torch.nn.extractor import Extractor, ExtractorName, get_extractor_cls
from leap_c.torch.nn.mlp import Mlp, MlpConfig, init_mlp_params_with_inverse_default
from leap_c.torch.rl.buffer import ReplayBuffer
from leap_c.torch.rl.dqn import DqnTrainerConfig, _linear_schedule
from leap_c.torch.rl.utils import soft_target_update
from leap_c.torch.utils.seed import mk_seed
from leap_c.trainer import Trainer
from leap_c.utils.gym import seed_env, wrap_env


@dataclass(kw_only=True)
class DqnMpcTrainerConfig(DqnTrainerConfig):
    """Specific settings for the DQN-MPC trainer.

    Attributes:
        init_param_with_default: Whether to initialize the learnable MPC parameters such that their
            values, when mapped to the parameter space (see `DqnMpcCritic.param_transform`),
            correspond to the default parameters of the MPC controller (see
            `DqnMpcCritic.controller.default_param`). Only valid in case no MLP is used (e.g.,
            `DqnMpcTrainerConfig.critic_mlp.hidden_dims is None`).
        num_threads_batch_solver: Number of threads to be used by the batched acados solver. The
            number of required batches will be at most `DqnTrainerConfig.batch_size`.
    """

    init_param_with_default: bool = True
    num_threads_batch_solver: int = 2**3


class DqnMpcCriticOutput(NamedTuple):
    """Output of the DQN-MPC critic.

    Attributes:
        param: The estimated parameters for which the MPC was solved.
        action: The optimal action output by the MPC controller.
        value: The objective value at the optimal solution.
        ctx: The context object containing information about the MPC solution.
    """

    param: Tensor
    action: Tensor
    value: Tensor
    ctx: AcadosDiffMpcCtx

    @property
    def status(self) -> np.ndarray:
        """Extracts the solver status from the context.

        Returns:
            The solver status.
        """
        return self.ctx.status

    @property
    def stats(self) -> dict[str, float]:
        """Extracts statistics from the context.

        Returns:
            A dictionary containing the solver statistics.
        """
        return self.ctx.log or {}


class DqnMpcCritic(nn.Module):
    """The critic for the DQN-MPC algorithm.

    Technically speaking, it acts both as a critic and as a policy provider.

    Attributes:
        extractor: A feature extractor for the observations.
        mlp: A Multi-Layer Perceptron (MLP) to provide estimates for the MPC parameters.
        controller: The differentiable parameterized MPC controller used in the critic.
        parameter_transform: Transform that maps the MLP (unbounded) outputs to the space of valid
            controller parameters.
    """

    extractor: Extractor
    mlp: Mlp
    controller: ParameterizedController
    parameter_transform: BoundedTransform

    def __init__(
        self,
        extractor: Extractor,
        mlp_cfg: MlpConfig,
        controller: ParameterizedController,
        init_param_with_default: bool = True,
    ) -> None:
        f"""Initializes the DQN-MPC critic.

        Args:
            extractor: The extractor that returns features from observations.
            mlp_cfg: The configuration for the MLP.
            controller: The differentiable parameterized controller to be used.
            init_param_with_default: Whether to initialize the learnable MPC parameters such that
                their values, when mapped to the parameter space (see
                `DqnMpcCritic.param_transform`), correspond to the default parameters of the MPC
                controller (see `controller.default_param`). Only valid in case no MLP is used
                (e.g., `mlp_cfg.hidden_dims is None`).

        Raises:
            ValueError: If the controller's parameter space is not a bounded `{Box.__name__}` space.
        """
        if (
            not isinstance(space := controller.param_space, Box)
            or not space.bounded_above.all()
            or not space.bounded_below.all()
        ):
            raise ValueError(
                f"`{self.__class__.__name__}` only supports bounded `{Box.__name__}` parameter "
                "spaces."
            )
        super().__init__()

        self.extractor = extractor
        self.controller = controller
        self.mlp = Mlp(
            input_sizes=extractor.output_size, output_sizes=space.shape[0], mlp_cfg=mlp_cfg
        )
        self.parameter_transform = BoundedTransform(controller.param_space)
        if init_param_with_default:
            init_mlp_params_with_inverse_default(self.mlp, self.parameter_transform, controller)

    def forward(
        self, obs: Tensor, action: Tensor | None = None, ctx: AcadosDiffMpcCtx | None = None
    ) -> DqnMpcCriticOutput:
        param = self.parameter_transform(self.mlp(self.extractor(obs)))
        ctx, action, value = self.controller(obs, param, action, ctx)
        return DqnMpcCriticOutput(param, action, value, ctx)


class DqnMpcTrainer(Trainer[DqnMpcTrainerConfig]):
    """A trainer implementing Deep Q-Network (DQN) that uses a differentiable controller layer in
    the critic network (DQN-MPC).

    Attributes:
        train_env (Env): The training environment.
        q (DqnMpcCritic): The critic.
        q_target (DqnMpcCritic): The target critic.
        optim (torch.optim.Optimizer): Optimizer for the critic.
        buffer (ReplayBuffer): The replay buffer used for storing and sampling experiences.
        action_space (Box): Bounded action space of the environment the critic is trained on.
    """

    train_env: Env
    q: DqnMpcCritic
    q_target: DqnMpcCritic
    optim: torch.optim.Optimizer
    buffer: ReplayBuffer
    action_space: Box

    def __init__(
        self,
        cfg: DqnMpcTrainerConfig,
        train_env: Env,
        eval_env: Env,
        controller: ParameterizedController,
        output_path: str | Path,
        device: int | str | torch.device,
        extractor_cls: type[Extractor] | ExtractorName = "identity",
    ) -> None:
        f"""Initializes the DQN-MPC trainer.

        Args:
            cfg (DqnMpcTrainerConfig): The configuration for the trainer.
            train_env (Env): The training environment.
            eval_env (Env): The evaluation environment.
            controller (ParameterizedController): The differentiable MPC controller used as last
                layer in the policy value function approximator.
            output_path (str or Path): The path to the output directory (e.g., logs).
            device (int, str or torch.device): The device on which the trainer is running.
            extractor_cls (Extractor type or {"identity", "scaling"}): The class used for extracting
                features from observations.

        Raises:
            ValueError: If `train_env`'s action space space is not a bounded `{Box.__name__}` space.
        """
        if (
            not isinstance(action_space := train_env.action_space, Box)
            or not action_space.bounded_above.all()
            or not action_space.bounded_below.all()
        ):
            raise ValueError(
                f"`{self.__class__.__name__}` only supports bounded `{Box.__name__}` action spaces."
            )
        super().__init__(cfg, eval_env, output_path, device)

        self.train_env = wrap_env(train_env)
        if isinstance(extractor_cls, str):
            extractor_cls = get_extractor_cls(extractor_cls)
        extractor = extractor_cls(train_env.observation_space)
        self.action_space = action_space

        self.q = DqnMpcCritic(extractor, cfg.critic_mlp, controller, cfg.init_param_with_default)
        self.q_target = DqnMpcCritic(
            extractor, cfg.critic_mlp, controller, cfg.init_param_with_default
        )
        self.q_target.load_state_dict(self.q.state_dict())
        self.optim = torch.optim.Adam(self.q.parameters(), lr=cfg.lr)
        self.buffer = ReplayBuffer(
            cfg.buffer_size, device, collate_fn_map=controller.collate_fn_map
        )

    @property
    def optimizers(self) -> list[torch.optim.Optimizer]:
        return [self.optim]

    def periodic_ckpt_modules(self) -> list[str]:
        return ["q", "q_target"]

    def singleton_ckpt_modules(self) -> list[str]:
        return ["buffer"]

    def act(
        self,
        obs: np.ndarray,
        deterministic: bool = False,
        state: AcadosDiffMpcCtx | None = None,
    ) -> tuple[np.ndarray, AcadosDiffMpcCtx, dict[str, float]]:
        """Runs the DQN-MPC to obtain an action for the given observation.

        Args:
            obs (array): The observation/initial condition to run the actor for.
            deterministic (bool, optional): Whether to act deterministically or stochastically. In
                the latter case, epsilon-greedy exploration is used. Defaults to `False`.
            state (AcadosDiffMpcCtx, optional): The context (previous solution) to be passed to the
                controller. Defaults to `None`.

        Returns:
            A tuple containing the action to be taken, the context returned by the controller, and
            a dict of statistics.
        """
        if not deterministic and self.rng.uniform() < (
            epsilon := _linear_schedule(
                self.cfg.start_exploration,
                self.cfg.end_exploration,
                self.cfg.exploration_fraction * self.cfg.train_steps,
                self.state.step,
            )
        ):
            space = self.action_space
            action = self.rng.uniform(space.low, space.high, size=space.shape)
            ctx = state  # pass through previous context
            stats = {"epsilon": epsilon}

        else:
            with torch.inference_mode():
                output: DqnMpcCriticOutput = self.q(self.buffer.collate((obs,)), None, state)
            action = output.action.squeeze(0).numpy(force=True).astype(obs.dtype, copy=False)
            ctx = output.ctx
            stats = output.stats

        return action, ctx, stats

    def train_loop(self) -> Generator[int]:
        env = self.train_env
        buffer = self.buffer
        state = self.state
        cfg = self.cfg

        # reset before training loop
        obs, _ = seed_env(env, mk_seed(self.rng))
        ctx: AcadosDiffMpcCtx | None = None
        terminated = truncated = False

        while True:
            # compute an (possibly random) action for the current observation (HACK: we force the
            # very first MPC of each episode to be solved so that `ctx` is not `None`; otherwise,
            # the buffer collate function would fail when trying to collate `None` contexts)
            action, ctx, stats = self.act(obs, ctx is None, ctx)
            self.report_stats("train_trajectory", {"action": action}, True)
            self.report_stats("train_policy_rollout", stats, True)

            # step the environment and add transition to buffer
            obs_next, reward, terminated, truncated, info = env.step(action)
            buffer.put((obs, action, reward, obs_next, terminated, ctx))
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
                ctx = None
                terminated = truncated = False

    def _update(self) -> None:
        """Performs `cfg.gradient_steps` updates of the DQN-MPC ageant based on the squared TD error
        between the current action-value estimate and the target estimate."""
        dtype = self.buffer.tensor_dtype
        device = self.device

        losses: list[float] = []
        for i in range(self.cfg.gradient_steps):
            # sample a batch of transitions - negate the rewards to convert them to costs, since the
            # MPC controller final layer provides cost-to-go estimates rather than return estimates
            obs, act, r, obs_next, terminated, ctx = self.buffer.sample(self.cfg.batch_size)
            cost = r.neg()

            # compute samples of current action value distribution
            output: DqnMpcCriticOutput = self.q(obs, act, ctx)
            q_ok = torch.as_tensor(output.status, dtype=dtype, device=device) == 0.0

            # remove batch entries where the action-value MPC failed by masking via `q_ok` - does
            # not matter whether the next state is terminal or not; since Q(s,a) failed, we need to
            # skip these entries
            if not q_ok.any():
                warn(f"All action-value MPCs failed at gradient step {i}; skipping the update.")
                continue
            estimate = output.value[q_ok]  # remove entries where Q(s,a) failed
            cost = cost[q_ok]
            obs_next = obs_next[q_ok]
            terminated = terminated[q_ok]

            # compute target - solve state-value MPC only for the remaining non-terminal next states
            target = cost.clone()
            if (nonterm := (terminated == 0.0)).any():
                # compute the state value function at next state (only for remaining non-terminal)
                with torch.inference_mode():
                    output_next: DqnMpcCriticOutput = self.q_target(obs_next[nonterm], None, ctx)
                v_ok = torch.as_tensor(output_next.status, dtype=dtype, device=device) == 0.0

                # remove batch entries where the state-value MPC failed by masking via `v_ok`
                if not v_ok.any():
                    warn(f"All state-value MPCs failed at gradient step {i}; skipping the update.")
                    continue
                estimate_next = output_next.value[v_ok]

                # finally, compute the target as cost + gamma * V(s') only for those non-terminal
                # batch entries for which both Q(s,a) and V(s') succeeded
                keep = torch.full(nonterm.shape, True, dtype=torch.bool, device=device)
                keep[nonterm] = v_ok  # flags entries with V ok (Q was already handled above)
                target = target[keep]  # contains terminal as well as non-terminal (with V ok)
                estimate = estimate[keep]  # same as above
                nonterm = nonterm[keep]  # among V ok, discerns which are non-terminal
                target[nonterm] += self.cfg.gamma * estimate_next

            # compute TD loss between current action value estimate and target. Note that, if
            # execution reaches here, it is guaranteed that at least one batch entry has had a
            # successful Q(s,a) solution and, if there is at least one non-terminal state, a
            # successful V(s') solution. So, the loss is well-defined.
            loss = mse_loss(target, estimate)
            losses.append(loss.item())

            # optimize
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            # soft updates
            if self.state.step % self.cfg.soft_update_freq == 0:
                soft_target_update(self.q, self.q_target, self.cfg.tau)

        avg_td_loss = sum(losses) / len(losses) if losses else float("nan")
        self.report_stats("loss", {"avg_td_loss": avg_td_loss}, with_smoothing=False)
