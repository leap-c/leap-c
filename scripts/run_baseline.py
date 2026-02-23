"""Main script to run baselines (controller or random) with default parameters.

By default, runs validation episodes. For building comparison with RL methods,
use the `--only-train` flag to run training episodes instead. This will report
stats similar to the RL training. This is especially useful for high variance
environments (e.g. hvac), where a lot of validation episode are required to get
a good estimate of performance.
"""

from argparse import ArgumentParser
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Generator, Literal

import gymnasium as gym
import torch
from numpy import ndarray

from leap_c.controller import CtxType, ParameterizedController
from leap_c.examples import ExampleControllerName, ExampleEnvName, create_controller, create_env
from leap_c.run import (
    default_controller_code_path,
    default_name,
    default_output_path,
    init_run,
    validate_torch_dtype_arg,
)
from leap_c.torch.rl.buffer import ReplayBuffer
from leap_c.torch.utils.seed import mk_seed
from leap_c.trainer import Trainer, TrainerConfig
from leap_c.utils.gym import seed_env, wrap_env


@dataclass
class BaselineTrainerConfig(TrainerConfig):
    """Configuration for running baseline experiments."""

    pass


@dataclass
class RunBaselineConfig:
    """Configuration for running baseline experiments.

    Attributes:
        env: The environment name.
        controller: The controller name (used if policy_type is 'controller').
        policy_type: The type of policy to run ('controller' or 'random').
        trainer: The trainer configuration.
    """

    env: ExampleEnvName = "cartpole"
    controller: ExampleControllerName | None = None
    policy_type: Literal["controller", "random"] = "controller"
    trainer: BaselineTrainerConfig = field(default_factory=BaselineTrainerConfig)


class BaselineTrainer(Trainer[BaselineTrainerConfig, Any]):
    """A trainer that runs a baseline policy (controller or random).

    Supports two modes:
    - Validation only (default): Runs validation episodes only
    - Training: Runs training episodes and reports stats similar to RL algorithms.

    Attributes:
        controller: The parameterized controller to use (if policy_type is 'controller').
        collate_fn: The function used to collate observations and actions.
        train_env: The training environment (None for validation-only mode).
    """

    train_env: gym.Env | None

    def __init__(
        self,
        cfg: BaselineTrainerConfig,
        val_env: gym.Env | None,
        output_path: str | Path,
        device: int | str | torch.device,
        dtype: torch.dtype,
        policy_type: Literal["controller", "random"],
        controller: ParameterizedController[CtxType] | None = None,
        train_env: gym.Env | None = None,
    ) -> None:
        """Initializes the trainer.

        Args:
            cfg: The trainer configuration.
            val_env: The validation environment.
            output_path: The path to save outputs to.
            device: The device to use.
            dtype: The data type to use.
            policy_type: The type of policy to run.
            controller: The parameterized controller to use (if policy_type is 'controller').
            train_env: The training environment. If None, only validation is performed.
        """
        super().__init__(cfg, val_env, output_path, device)
        self.policy_type = policy_type
        self.controller = controller
        self.train_env = wrap_env(train_env) if train_env is not None else None

        if self.policy_type == "controller":
            assert self.controller is not None, "Expected controller to be provided!"
            self.collate_fn = ReplayBuffer(1, device, dtype, controller.collate_fn_map).collate
        else:
            self.collate_fn = None

    def train_loop(self) -> Generator[tuple[int, float], None, None]:
        """Run training episodes, reporting stats similar to SAC training."""
        if self.train_env is None:
            # validation-only mode: just yield dummy values
            while True:
                yield 1, 0.0

        is_terminated = is_truncated = True
        policy_ctx = None
        obs = None

        while True:
            if is_terminated or is_truncated:
                obs, _ = seed_env(self.train_env, mk_seed(self.rng), {"mode": "train"})
                policy_ctx = None
                is_terminated = is_truncated = False

            if self.policy_type == "random":
                action = self.train_env.action_space.sample()
            else:
                obs_batched: torch.Tensor = self.collate_fn([obs])
                default_param = self.controller.default_param(obs_batched)
                default_param_tensor = torch.as_tensor(
                    default_param, device=self.device, dtype=obs_batched.dtype
                )
                policy_ctx, action_tensor = self.controller(
                    obs_batched, default_param_tensor, ctx=policy_ctx
                )
                action = action_tensor[0].cpu().numpy()

            obs_prime, reward, is_terminated, is_truncated, info = self.train_env.step(action)

            if "episode" in info or "task" in info:
                self.report_stats("train", info.get("episode", {}) | info.get("task", {}))

            obs = obs_prime

            yield 1, float(reward)

    def act(
        self, obs: ndarray, deterministic: bool = False, state: Any | None = None
    ) -> tuple[ndarray, Any, dict[str, float] | None]:
        """Use the policy (controller or random)."""
        if self.policy_type == "random":
            # Use eval_env action space for validation
            return self.eval_env.action_space.sample(), None, None

        obs_batched = self.collate_fn([obs])
        default_param = self.controller.default_param(obs_batched)
        default_param = torch.from_numpy(default_param).to(self.device)
        ctx, action = self.controller(obs_batched, default_param, ctx=state)
        action = action.cpu().numpy()[0]
        return action, ctx, ctx.log


def create_cfg(
    env: ExampleEnvName,
    controller: ExampleControllerName | None,
    seed: int,
    only_train: bool = False,
    ckpt_modus: Literal["best", "last", "all", "none"] = "none",
    policy_type: Literal["controller", "random"] = "controller",
) -> RunBaselineConfig:
    """Return the default configuration for running baseline experiments.

    Args:
        env: The environment name.
        controller: The controller name.
        seed: The random seed.
        only_train: Whether to run training episodes.
        ckpt_modus: The checkpoint mode.
        policy_type: The type of policy to run.
    """
    cfg = RunBaselineConfig()
    cfg.env = env
    cfg.policy_type = policy_type

    if policy_type == "controller":
        cfg.controller = controller if controller is not None else env
    else:
        cfg.controller = controller  # Can be None for random

    # ---- Section: cfg.trainer ----
    cfg.trainer.seed = seed
    cfg.trainer.train_start = 0
    cfg.trainer.val_num_rollouts = 20 if env != "hvac" else 100
    cfg.trainer.val_deterministic = True
    cfg.trainer.val_num_render_rollouts = 0
    cfg.trainer.val_render_mode = "rgb_array"
    cfg.trainer.val_report_score = "cum"
    cfg.trainer.ckpt_modus = ckpt_modus

    if env == "hvac":
        cfg.trainer.log.cumulative_metrics = [
            "train/money_spent",
            "train/energy_kwh",
            "train/constraint_violation",
        ]

    if only_train:
        cfg.trainer.train_steps = 1_000_000 if env == "pointmass" else 200_000
        cfg.trainer.val_freq = 10_000 if env != "hvac" else 50_000
    else:
        cfg.trainer.train_steps = 1
        cfg.trainer.val_freq = 1

    # ---- Section: cfg.trainer.log ----
    cfg.trainer.log.verbose = True
    cfg.trainer.log.interval = 1_000
    cfg.trainer.log.window = 10_000
    cfg.trainer.log.csv_logger = True
    cfg.trainer.log.tensorboard_logger = True
    cfg.trainer.log.wandb_logger = False
    cfg.trainer.log.wandb_init_kwargs = {}

    return cfg


def run_baseline(
    cfg: RunBaselineConfig,
    output_path: str | Path,
    device: int | str | torch.device,
    dtype: torch.dtype,
    reuse_code_dir: Path | None = None,
    only_train: bool = False,
) -> float:
    """Run the baseline.

    Args:
        cfg: The configuration for running the baseline.
        output_path: The path to save outputs to.
            If it already exists, the run will continue from the last checkpoint.
        device: The device to use.
        dtype: The data type to use.
        reuse_code_dir: The directory to reuse compiled code from, if any.
        only_train: Whether to run training episodes.
    """
    val_env = create_env(cfg.env, render_mode="rgb_array") if not only_train else None
    train_env = create_env(cfg.env) if only_train else None

    controller = None
    if cfg.policy_type == "controller":
        controller = create_controller(cfg.controller, reuse_code_dir)

    trainer = BaselineTrainer(
        cfg=cfg.trainer,
        val_env=val_env,
        output_path=output_path,
        device=device,
        dtype=dtype,
        policy_type=cfg.policy_type,
        controller=controller,
        train_env=train_env,
    )
    init_run(trainer, cfg, output_path)
    return trainer.run()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--output_path", type=Path, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--dtype", type=validate_torch_dtype_arg, default="float32")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--env", type=str, default="cartpole")
    parser.add_argument("--controller", type=str, default=None)
    parser.add_argument(
        "--policy-type",
        type=str,
        default="controller",
        choices=["controller", "random"],
        help="The type of policy to run.",
    )
    parser.add_argument(
        "--only-train",
        action="store_true",
        help="Run training episodes over time (for comparison with RL methods). "
        "Without this flag, validation episodes are run instead.",
    )
    parser.add_argument(
        "-r",
        "--reuse_code",
        action="store_true",
        help="Reuse compiled code. The first time this is run, it will compile the code.",
    )
    parser.add_argument("--reuse_code_dir", type=Path, default=None)
    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-project", type=str, default="leap-c")
    args = parser.parse_args()

    cfg = create_cfg(
        args.env,
        args.controller,
        args.seed,
        args.only_train,
        policy_type=args.policy_type,
    )

    if args.use_wandb:
        config_dict = asdict(cfg)
        cfg.trainer.log.wandb_logger = True
        cfg.trainer.log.wandb_init_kwargs = {
            "entity": args.wandb_entity,
            "project": args.wandb_project,
            "name": default_name(
                args.seed, tags=["baseline", args.policy_type, args.env, str(args.controller)]
            ),
            "config": config_dict,
        }

    if args.output_path is None:
        output_path = default_output_path(
            seed=args.seed,
            tags=["baseline", args.policy_type, args.env, str(args.controller)],
        )
    else:
        output_path = args.output_path

    if args.reuse_code and args.reuse_code_dir is None:
        reuse_code_dir = default_controller_code_path()
    elif args.reuse_code_dir is not None:
        reuse_code_dir = args.reuse_code_dir
    else:
        reuse_code_dir = None

    run_baseline(cfg, output_path, args.device, args.dtype, reuse_code_dir, args.only_train)
