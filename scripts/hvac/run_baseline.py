"""Main script to run baselines (controller or random) with default parameters.

By default, runs validation episodes. For building comparison with RL methods,
use the `--only-train` flag to run training episodes instead. This will report
stats similar to the RL training. This is especially useful for high variance
environments (e.g. hvac), where a lot of validation episode are required to get
a good estimate of performance.
"""

import shutil
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Generator, Literal, get_args

import gymnasium as gym
import numpy as np
import pandas as pd
import torch
from numpy import ndarray

from leap_c.controller import CtxType, ParameterizedController
from leap_c.examples import ExampleControllerName, ExampleEnvName, create_controller, create_env
from leap_c.examples.hvac.dataset import DataConfig, HvacDataset
from leap_c.examples.hvac.env import StochasticThreeStateRcEnv
from leap_c.examples.hvac.utils import set_temperature_limits
from leap_c.run import (
    default_controller_code_path,
    default_name,
    default_output_path,
    init_run,
    validate_torch_device_arg,
    validate_torch_dtype_arg,
)
from leap_c.torch.rl.buffer import ReplayBuffer
from leap_c.torch.utils.seed import mk_seed
from leap_c.trainer import Trainer, TrainerConfig
from leap_c.utils.gym import seed_env, wrap_env


@dataclass
class BaselineTrainerConfig(TrainerConfig):
    """Configuration for running baseline experiments."""

    param_ckpt: str | None = None
    val_step_print_interval: int = 50


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
        self._val_records: list[dict] = []

        if self.policy_type == "controller":
            assert self.controller is not None, "Expected controller to be provided!"
            self.collate_fn = ReplayBuffer(1, device, dtype, controller.collate_fn_map).collate
        else:
            self.collate_fn = None

        self.loaded_param = None
        if self.cfg.param_ckpt is not None:
            data = np.load(self.cfg.param_ckpt, allow_pickle=True)
            if "best_param" in data:
                self.loaded_param = data["best_param"]
            elif "best_config" in data:
                config = data["best_config"]
                if isinstance(config, np.ndarray) and config.dtype == object:
                    config = config.item()
                if isinstance(config, dict) and all(k.startswith("param_") for k in config.keys()):
                    n_params = len(config)
                    self.loaded_param = np.array([config[f"param_{i}"] for i in range(n_params)])
                else:
                    self.loaded_param = config
            else:
                raise ValueError(
                    f"Could not find 'best_param' or 'best_config' in {self.cfg.param_ckpt}"
                )

            if not isinstance(self.loaded_param, np.ndarray):
                self.loaded_param = np.asarray(self.loaded_param)
            self.loaded_param = self.loaded_param.astype(np.float32)

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
                obs_batched = self.collate_fn([obs])
                if self.loaded_param is not None:
                    param = self.loaded_param
                else:
                    param = self.controller.default_param(obs_batched)
                param_tensor = torch.from_numpy(param).to(self.device)
                policy_ctx, action_tensor = self.controller(
                    obs_batched, param_tensor, ctx=policy_ctx
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
        if self.loaded_param is not None:
            param = self.loaded_param
        else:
            param = self.controller.default_param(obs_batched)
        param_tensor = torch.from_numpy(param).to(self.device)
        ctx, action = self.controller(obs_batched, param_tensor, ctx=state)
        action = action.cpu().numpy()[0]
        return action, ctx, ctx.log

    def _make_val_step_callback(self):
        env = self.eval_env.unwrapped if self.eval_env is not None else None
        if not isinstance(env, StochasticThreeStateRcEnv):
            return super()._make_val_step_callback()

        interval = self.cfg.val_step_print_interval
        records = self._val_records

        def callback(step: int, obs, action, reward, info, ctx) -> None:
            state = obs["state"]
            ti_c = float(state[0]) - 273.15
            th_c = float(state[1]) - 273.15
            te_c = float(state[2]) - 273.15
            qh = float(obs["time"]["quarter_hour"][0])
            lb_K, ub_K = set_temperature_limits(np.array([qh]))
            task = info.get("task", {})
            solver_status = (
                int(ctx.status.flat[0]) if ctx is not None and hasattr(ctx, "status") else -1
            )
            records.append(
                {
                    "datetime": info.get("datetime", ""),
                    "step": step,
                    "Ti_C": ti_c,
                    "Th_C": th_c,
                    "Te_C": te_c,
                    "T_lb_C": float(lb_K[0]) - 273.15,
                    "T_ub_C": float(ub_K[0]) - 273.15,
                    "ambient_temp_C": float(obs["forecast"]["temperature"][0]) - 273.15,
                    "solar_W_m2": float(obs["forecast"]["solar"][0]),
                    "price": float(task.get("price", float("nan"))),
                    "action_W": float(action[0]),
                    "reward": float(reward),
                    "energy_kwh": float(task.get("energy_kwh", float("nan"))),
                    "money_spent": float(task.get("money_spent", float("nan"))),
                    "comfort_reward": float(task.get("comfort_reward", float("nan"))),
                    "energy_reward": float(task.get("energy_reward", float("nan"))),
                    "constraint_violation_K": float(task.get("constraint_violation", float("nan"))),
                    "success": int(task.get("success", 0)),
                    "solver_status": solver_status,
                }
            )
            if interval > 0 and step % interval == 0:
                print(f"  val step {step:4d} | Ti={ti_c:.1f}°C  Th={th_c:.1f}°C  Te={te_c:.1f}°C")

        return callback

    def validate(self) -> float:
        self._val_records.clear()
        score = super().validate()
        if self._val_records:
            df = pd.DataFrame(self._val_records)
            csv_path = self.output_path / f"val_timeseries_step{self.state.step}.csv"
            df.to_csv(csv_path, index=False)
            print(f"Timeseries saved to: {csv_path}")
        return score


def create_cfg(
    env: ExampleEnvName,
    controller: ExampleControllerName | None,
    seed: int,
    only_train: bool = False,
    ckpt_modus: Literal["best", "last", "all", "none"] = "none",
    policy_type: Literal["controller", "random"] = "controller",
    param_ckpt: Path | None = None,
) -> RunBaselineConfig:
    """Return the default configuration for running baseline experiments.

    Args:
        env: The environment name.
        controller: The controller name.
        seed: The random seed.
        only_train: Whether to run training episodes.
        ckpt_modus: The checkpoint mode.
        policy_type: The type of policy to run.
        param_ckpt: The parameter checkpoint to load.
    """
    cfg = RunBaselineConfig()
    cfg.env = env
    cfg.policy_type = policy_type
    cfg.trainer.param_ckpt = str(param_ckpt) if param_ckpt is not None else None

    if policy_type == "controller":
        cfg.controller = controller if controller is not None else env
    else:
        cfg.controller = controller  # Can be None for random

    # ---- Section: cfg.trainer ----
    cfg.trainer.seed = seed
    cfg.trainer.train_start = 0
    cfg.trainer.val_num_rollouts = 1
    cfg.trainer.val_deterministic = True
    cfg.trainer.val_num_render_rollouts = 0
    cfg.trainer.val_render_mode = None
    cfg.trainer.val_report_score = "cum"
    cfg.trainer.ckpt_modus = ckpt_modus

    if env == "hvac":
        cfg.trainer.log.cumulative_metrics = [
            "train/money_spent",
            "train/energy_kwh",
            "train/constraint_violation",
        ]

    cfg.trainer.train_steps = 0
    cfg.trainer.val_freq = 1

    # ---- Section: cfg.trainer.log ----
    cfg.trainer.log.verbose = True
    cfg.trainer.log.interval = 100
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
    val_episode_steps: int | None = 1000,
    overwrite: bool = False,
) -> float:
    """Run the baseline.

    Args:
        cfg: The configuration for running the baseline.
        output_path: The path to save outputs to.
            If it already exists, the run will continue from the last checkpoint
            unless `overwrite` is True.
        device: The device to use.
        dtype: The data type to use.
        reuse_code_dir: The directory to reuse compiled code from, if any.
        only_train: Whether to run training episodes.
        val_episode_steps: Override the episode length (in steps) for hvac validation.
            Each step is a 15-min interval, so 1000 steps = 250 hours.
        overwrite: If True, delete an existing output directory before running.
    """
    if overwrite and Path(output_path).exists():
        shutil.rmtree(output_path)

    hvac_kwargs: dict = {}
    if cfg.env in ("hvac", "hvac_continual") and val_episode_steps is not None:
        max_hours = val_episode_steps / 4  # 15-min intervals → hours
        hvac_kwargs["dataset"] = HvacDataset(cfg=DataConfig(max_hours=int(max_hours)))

    val_env = create_env(cfg.env, **hvac_kwargs) if not only_train else None
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
    parser = ArgumentParser(
        description="Training of baseline controllers.",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    group = parser.add_argument_group("Run settings")
    group.add_argument(
        "--output-path", type=Path, default=None, help="Path to outputs (e.g., logs)."
    )
    group.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete an existing output directory before running (implies a fresh start).",
    )
    group.add_argument(
        "--device", type=validate_torch_device_arg, default="cpu", help="Device to run on."
    )
    group.add_argument(
        "--dtype",
        type=validate_torch_dtype_arg,
        default="float32",
        help="Data type to use during training and evaluation.",
    )
    group.add_argument("--seed", type=int, default=0, help="RNG seed.")
    group.add_argument(
        "-r",
        "--reuse-code",
        action="store_true",
        help="Reuse compiled code. The first time this is run, it will compile the code.",
    )
    group.add_argument(
        "--reuse-code-dir", type=Path, default=None, help="Directory for compiled code."
    )
    group = parser.add_argument_group("Train and eval")
    group.add_argument(
        "--env",
        type=str,
        choices=get_args(ExampleEnvName),
        default="cartpole",
        help="Environment to train on.",
    )
    group.add_argument(
        "--controller",
        type=str,
        choices=get_args(ExampleControllerName),
        default=None,
        help="MPC controller to use as actor. If not provided, it is taken from `--env`.",
    )
    group.add_argument(
        "--policy-type",
        type=str,
        default="controller",
        choices=["controller", "random"],
        help="The type of policy to run. If `random`, the controller will not be used.",
    )
    group.add_argument(
        "--only-train",
        action="store_true",
        help="Run training episodes over time (for comparison with RL methods). "
        "Without this flag, validation episodes are run instead.",
    )
    group.add_argument(
        "--param-ckpt",
        type=Path,
        default=None,
        help="Controller parameters to load from a SMAC run.",
    )
    group.add_argument(
        "--val-episode-steps",
        type=int,
        default=None,
        help="Override episode length (steps) for hvac validation. "
        "Each step is a 15-min interval (e.g. 1000 steps = 250 hours).",
    )

    group = parser.add_argument_group("W&B logging")
    group.add_argument("--use-wandb", action="store_true", help="Whether to use W&B logging.")
    group.add_argument("--wandb-entity", type=str, default=None, help="W&B entity name.")
    group.add_argument("--wandb-project", type=str, default="leap-c", help="W&B project name.")
    group.add_argument("--wandb-group", type=str, default="baseline", help="W&B group name.")

    args = parser.parse_args()

    cfg = create_cfg(
        args.env,
        args.controller,
        args.seed,
        args.only_train,
        policy_type=args.policy_type,
        param_ckpt=args.param_ckpt,
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

    run_baseline(
        cfg,
        output_path,
        args.device,
        args.dtype,
        reuse_code_dir,
        args.only_train,
        args.val_episode_steps,
        args.overwrite,
    )
