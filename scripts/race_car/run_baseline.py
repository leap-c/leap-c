"""Run baseline (MPC controller or random policy) for the race-car environment.

Wires the channel-logging trainer machinery in ``channels.py`` up against
``RaceCarEnv`` / ``RaceCarPlanner``.

Example:
-------
    # MPC controller, single validation lap:
    python scripts/race_car/run_baseline.py --output-path output/race_car_baseline

    # Random policy:
    python scripts/race_car/run_baseline.py --policy-type random
"""

from __future__ import annotations

import shutil
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Generator, Literal

import gymnasium as gym
import numpy as np
import torch
from channels import ChannelLogger, default_channels, header_from
from numpy import ndarray
from sac_race_car_mixin import add_reward_cli, build_reward_from_args

from leap_c.controller import CtxType, ParameterizedController
from leap_c.examples import ExampleControllerName, create_controller
from leap_c.examples.race_car.env import RaceCarEnv, RaceCarEnvConfig, RaceCarRewardConfig
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
    """Trainer configuration for race-car baseline runs."""

    param_ckpt: str | None = None
    val_step_print_interval: int = 50
    compute_sensitivities: bool = False


@dataclass
class RunBaselineConfig:
    env: str = "race_car"
    controller: ExampleControllerName | None = None
    policy_type: Literal["controller", "random"] = "controller"
    trainer: BaselineTrainerConfig = field(default_factory=BaselineTrainerConfig)
    reward: RaceCarRewardConfig = field(default_factory=RaceCarRewardConfig)


class BaselineTrainer(Trainer[BaselineTrainerConfig, Any]):
    """Trainer that runs a fixed-parameter MPC or random baseline for race_car."""

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
        super().__init__(cfg, val_env, output_path, device)
        self.policy_type = policy_type
        self.controller = controller
        self.train_env = wrap_env(train_env) if train_env is not None else None
        self._val_logger: ChannelLogger | None = None
        self._pre_obs: Any = None  # most recent pre-step obs passed into act()

        if self.policy_type == "controller":
            assert self.controller is not None
            self.collate_fn = ReplayBuffer(1, device, dtype, controller.collate_fn_map).collate
        else:
            self.collate_fn = None

        self.loaded_param: np.ndarray | None = None
        if self.cfg.param_ckpt is not None:
            data = np.load(self.cfg.param_ckpt, allow_pickle=True)
            if "best_param" in data:
                self.loaded_param = np.asarray(data["best_param"]).astype(np.float32)
            else:
                raise ValueError(f"Could not find 'best_param' in {self.cfg.param_ckpt}")

    def train_loop(self) -> Generator[tuple[int, float], None, None]:
        if self.train_env is None:
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
                param = (
                    self.loaded_param
                    if self.loaded_param is not None
                    else self.controller.default_param(obs_batched)
                )
                param_tensor = torch.from_numpy(param).to(self.device)
                if param_tensor.ndim == 1:
                    param_tensor = param_tensor.unsqueeze(0)
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
        self._pre_obs = obs
        if self.policy_type == "random":
            return self.eval_env.action_space.sample(), None, None
        obs_batched = self.collate_fn([obs])
        param = (
            self.loaded_param
            if self.loaded_param is not None
            else self.controller.default_param(obs_batched)
        )
        param_tensor = torch.from_numpy(param).to(self.device)
        if param_tensor.ndim == 1:
            param_tensor = param_tensor.unsqueeze(0)
        ctx, action = self.controller(obs_batched, param_tensor, ctx=state)
        return action.cpu().numpy()[0], ctx, ctx.log

    def _make_val_step_callback(self):
        env = self.eval_env.unwrapped if self.eval_env is not None else None
        planner = getattr(self.controller, "planner", None)
        if not isinstance(env, RaceCarEnv) or planner is None:
            return super()._make_val_step_callback()

        self._val_logger = ChannelLogger(
            default_channels(env, planner, compute_sensitivities=self.cfg.compute_sensitivities)
        )
        interval = self.cfg.val_step_print_interval

        def callback(step: int, obs_post, action, reward, info, ctx) -> None:
            obs_pre = self._pre_obs
            self._val_logger.record(obs_pre, info, action, ctx, reward)
            if interval > 0 and step % interval == 0:
                s_ = float(np.asarray(obs_pre).flat[0])
                n_ = float(np.asarray(obs_pre).flat[1])
                v_ = float(np.asarray(obs_pre).flat[3])
                print(
                    f"  val step {step:4d} | s={s_:+6.2f} m | n={n_ * 100:+5.1f} cm "
                    f"| v={v_:.3f} m/s"
                )

        return callback

    def validate(self) -> float:
        self._val_logger = None
        score = super().validate()
        if self._val_logger is None or not (self._val_logger.scalars or self._val_logger.sequences):
            return score
        env = self.eval_env.unwrapped
        planner = self.controller.planner
        header = header_from(env, planner)
        npz_path, json_path = self._val_logger.save(self.output_path, self.state.step, header)
        print(f"Channel log saved to: {npz_path}")
        print(f"Channel metadata saved to: {json_path}")
        return score


def create_cfg(
    controller: ExampleControllerName | None,
    seed: int,
    policy_type: Literal["controller", "random"] = "controller",
    param_ckpt: Path | None = None,
) -> RunBaselineConfig:
    cfg = RunBaselineConfig()
    cfg.env = "race_car"
    cfg.policy_type = policy_type
    cfg.controller = controller if controller is not None else "race_car"
    cfg.trainer.param_ckpt = str(param_ckpt) if param_ckpt is not None else None
    cfg.trainer.seed = seed
    cfg.trainer.train_start = 0
    cfg.trainer.val_num_rollouts = 1
    cfg.trainer.val_deterministic = True
    cfg.trainer.val_num_render_rollouts = 0
    cfg.trainer.val_render_mode = None
    cfg.trainer.val_report_score = "cum"
    cfg.trainer.ckpt_modus = "none"
    cfg.trainer.train_steps = 0
    cfg.trainer.val_freq = 1
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
    overwrite: bool = False,
    max_steps: int = 4000,
) -> float:
    if overwrite and Path(output_path).exists():
        shutil.rmtree(output_path)

    env_cfg = RaceCarEnvConfig(max_steps=max_steps, reward=cfg.reward)
    val_env = RaceCarEnv(cfg=env_cfg)

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
    )
    init_run(trainer, cfg, output_path)
    return trainer.run()


if __name__ == "__main__":
    parser = ArgumentParser(
        description="race_car baseline: MPC controller or random policy.",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    g = parser.add_argument_group("Run settings")
    g.add_argument("--output-path", type=Path, default=None)
    g.add_argument("--overwrite", action="store_true")
    g.add_argument("--device", type=validate_torch_device_arg, default="cpu")
    g.add_argument("--dtype", type=validate_torch_dtype_arg, default="float64")
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("-r", "--reuse-code", action="store_true")
    g.add_argument("--reuse-code-dir", type=Path, default=None)

    g = parser.add_argument_group("Train and eval")
    g.add_argument("--controller", type=str, default="race_car")
    g.add_argument(
        "--policy-type", type=str, default="controller", choices=["controller", "random"]
    )
    g.add_argument("--max-steps", type=int, default=4000)
    g.add_argument("--param-ckpt", type=Path, default=None)
    g.add_argument(
        "--compute-sensitivities",
        action="store_true",
        help="Compute du_dp sensitivities at each validation step (slower; saves to NPZ).",
    )

    add_reward_cli(parser)

    g = parser.add_argument_group("W&B logging")
    g.add_argument("--use-wandb", action="store_true")
    g.add_argument("--wandb-entity", type=str, default=None)
    g.add_argument("--wandb-project", type=str, default="leap-c")
    g.add_argument("--wandb-group", type=str, default="baseline-race_car")

    args = parser.parse_args()

    cfg = create_cfg(
        controller=args.controller,
        seed=args.seed,
        policy_type=args.policy_type,
        param_ckpt=args.param_ckpt,
    )
    cfg.trainer.compute_sensitivities = args.compute_sensitivities
    cfg.reward = build_reward_from_args(args)

    if args.use_wandb:
        config_dict = asdict(cfg)
        cfg.trainer.log.wandb_logger = True
        cfg.trainer.log.wandb_init_kwargs = {
            "entity": args.wandb_entity,
            "project": args.wandb_project,
            "name": default_name(
                args.seed, tags=["baseline", args.policy_type, "race_car", str(args.controller)]
            ),
            "config": config_dict,
        }

    if args.output_path is None:
        output_path = default_output_path(
            seed=args.seed,
            tags=["baseline", args.policy_type, "race_car", str(args.controller)],
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
        args.overwrite,
        max_steps=args.max_steps,
    )
