"""Run baseline (MPC controller or random policy) for the i4b environment.

By default runs a single validation episode.  Use ``--only-train`` to run
training episodes instead, which reports rolling stats comparable to RL runs.

Example:
-------
    # MPC controller, 3-day validation episode:
    python scripts/i4b/run_baseline.py --days 3

    # Random policy:
    python scripts/i4b/run_baseline.py --policy-type random
"""

import shutil
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Generator, Literal

import gymnasium as gym
import numpy as np
import pandas as pd
import torch
from numpy import ndarray

from leap_c.controller import CtxType, ParameterizedController
from leap_c.examples import ExampleControllerName, create_controller, create_env
from leap_c.examples.i4b.env import I4bEnv
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

# Default episode length for validation
_DEFAULT_DAYS = 3


@dataclass
class BaselineTrainerConfig(TrainerConfig):
    """Trainer configuration for i4b baseline runs."""

    param_ckpt: str | None = None
    val_step_print_interval: int = 16


@dataclass
class RunBaselineConfig:
    """Top-level configuration for an i4b baseline run.

    Attributes:
        env: The environment name (always "i4b").
        controller: Controller name used when policy_type is "controller".
        policy_type: "controller" runs the MPC; "random" samples uniformly.
        trainer: Trainer hyper-parameters.
    """

    env: str = "i4b"
    controller: ExampleControllerName | None = None
    policy_type: Literal["controller", "random"] = "controller"
    trainer: BaselineTrainerConfig = field(default_factory=BaselineTrainerConfig)


class BaselineTrainer(Trainer[BaselineTrainerConfig, Any]):
    """Trainer that runs a fixed-parameter MPC or random baseline for i4b."""

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
        self._val_records: list[dict] = []
        self._val_x_trajs: list[np.ndarray] = []  # per-step (N+1, nx) state trajectories
        self._val_u_trajs: list[np.ndarray] = []  # per-step (N, nu) control trajectories

        if self.policy_type == "controller":
            assert self.controller is not None
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
        if self.policy_type == "random":
            return self.eval_env.action_space.sample(), None, None

        obs_batched = self.collate_fn([obs])
        param = (
            self.loaded_param
            if self.loaded_param is not None
            else self.controller.default_param(obs_batched)
        )
        param_tensor = torch.from_numpy(param).to(self.device)
        ctx, action = self.controller(obs_batched, param_tensor, ctx=state)
        action = action.cpu().numpy()[0]
        return action, ctx, ctx.log

    def _make_val_step_callback(self):
        env = self.eval_env.unwrapped if self.eval_env is not None else None
        if not isinstance(env, I4bEnv):
            return super()._make_val_step_callback()

        interval = self.cfg.val_step_print_interval
        records = self._val_records
        x_trajs = self._val_x_trajs
        u_trajs = self._val_u_trajs

        planner = getattr(getattr(self.controller, "planner", None), "cfg", None)
        N_horizon = getattr(planner, "N_horizon", None)
        nx = (
            getattr(self.controller.planner, "_nx", None)
            if hasattr(self.controller, "planner")
            else None
        )
        obs_keys = list(env.state_keys)  # e.g. ["T_room", "T_wall", "T_hp_ret"]

        def callback(step: int, obs, action, reward, info, ctx) -> None:
            T_room = float(info.get("T_room", float("nan")))
            T_hp_sup = float(info.get("T_hp_sup", float("nan")))
            E_el_kWh = float(info.get("E_el_kWh", float("nan")))
            T_set_lower = float(obs["setpoints"]["T_set_lower"].flat[0])
            T_set_upper = float(obs["setpoints"]["T_set_upper"].flat[0])
            T_amb = float(obs["disturbances"]["T_amb"].flat[0])
            solver_status = (
                int(ctx.status.flat[0]) if ctx is not None and hasattr(ctx, "status") else -1
            )
            records.append(
                {
                    "step": step,
                    "T_room": T_room,
                    "T_set_lower": T_set_lower,
                    "T_set_upper": T_set_upper,
                    "T_amb": T_amb,
                    "T_hp_sup": T_hp_sup,
                    "E_el_kWh": E_el_kWh,
                    "reward": float(reward),
                    "solver_status": solver_status,
                }
            )
            for i, key in enumerate(obs_keys):
                records[-1][key] = float(obs["state"].flat[i])

            if ctx is not None and hasattr(ctx, "stats") and ctx.stats:
                for key in ctx.stats[0].keys():
                    if key.startswith("time") or key in ["sqp_iter"]:
                        records[-1][f"solver_{key}"] = float(ctx.stats[0][key])

            if ctx is not None and hasattr(ctx, "iterate") and ctx.iterate is not None:
                # ctx.iterate.x shape: (1, (N+1)*nx) — reshape to (N+1, nx)
                # ctx.iterate.u shape: (1, N*nu)     — reshape to (N, nu)
                x_trajs.append(ctx.iterate.x[0].reshape(N_horizon + 1, nx))
                u_trajs.append(ctx.iterate.u[0].reshape(N_horizon, -1))

            if interval > 0 and step % interval == 0:
                flag = "" if T_set_lower <= T_room <= T_set_upper else " [!]"
                print(
                    f"  val step {step:4d} | T_room={T_room:.1f}°C"
                    f"  [{T_set_lower:.1f}, {T_set_upper:.1f}]°C{flag}"
                    f"  T_HP={T_hp_sup:.1f}°C  E={E_el_kWh * 1e3:.0f} Wh"
                )

        return callback

    def validate(self) -> float:
        self._val_records.clear()
        self._val_x_trajs.clear()
        self._val_u_trajs.clear()
        score = super().validate()
        step = self.state.step
        if self._val_records:
            df = pd.DataFrame(self._val_records)
            csv_path = self.output_path / f"val_timeseries_step{step}.csv"
            df.to_csv(csv_path, index=False)
            print(f"Timeseries saved to: {csv_path}")
        if self._val_x_trajs:
            env = self.eval_env.unwrapped if self.eval_env is not None else None
            state_names = list(env.state_keys) if isinstance(env, I4bEnv) else []
            npz_path = self.output_path / f"val_mpc_trajectories_step{step}.npz"
            np.savez_compressed(
                npz_path,
                x=np.stack(self._val_x_trajs),  # (T, N+1, nx)
                u=np.stack(self._val_u_trajs),  # (T, N, nu)
                state_names=np.array(state_names),
            )
            print(f"MPC trajectories saved to: {npz_path}")
        return score


def create_cfg(
    controller: ExampleControllerName | None,
    seed: int,
    only_train: bool = False,
    policy_type: Literal["controller", "random"] = "controller",
    param_ckpt: Path | None = None,
) -> RunBaselineConfig:
    """Return the default configuration for an i4b baseline run."""
    cfg = RunBaselineConfig()
    cfg.env = "i4b"
    cfg.policy_type = policy_type
    cfg.trainer.param_ckpt = str(param_ckpt) if param_ckpt is not None else None

    cfg.controller = "i4b"

    # (
    #     (controller if controller is not None else "i4b")
    #     if policy_type == "controller"
    #     else controller
    # )

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
    only_train: bool = False,
    days: int = _DEFAULT_DAYS,
    overwrite: bool = False,
) -> float:
    """Run the i4b baseline.

    Args:
        cfg: Run configuration.
        output_path: Directory for logs and checkpoints.
        device: Torch device.
        dtype: Torch dtype.
        reuse_code_dir: Directory with pre-compiled acados code, if any.
        only_train: Run training episodes instead of validation.
        days: Episode length in days for validation (default: 3).
        overwrite: Delete existing output directory before running.
    """
    if overwrite and Path(output_path).exists():
        shutil.rmtree(output_path)

    val_env = create_env(cfg.env) if not only_train else None
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
        description="i4b baseline: MPC controller or random policy.",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    group = parser.add_argument_group("Run settings")
    group.add_argument("--output-path", type=Path, default=None)
    group.add_argument("--overwrite", action="store_true")
    group.add_argument("--device", type=validate_torch_device_arg, default="cpu")
    group.add_argument("--dtype", type=validate_torch_dtype_arg, default="float64")
    group.add_argument("--seed", type=int, default=0)
    group.add_argument(
        "-r", "--reuse-code", action="store_true", help="Reuse compiled acados code."
    )
    group.add_argument("--reuse-code-dir", type=Path, default=None)

    group = parser.add_argument_group("Train and eval")
    group.add_argument(
        "--controller", type=str, default="i4b", help="Controller name (default: 'i4b')."
    )
    group.add_argument(
        "--policy-type",
        type=str,
        default="controller",
        choices=["controller", "random"],
    )
    group.add_argument(
        "--only-train", action="store_true", help="Run training episodes (for RL comparison)."
    )
    group.add_argument(
        "--days", type=int, default=_DEFAULT_DAYS, help="Episode length in days for validation."
    )
    group.add_argument("--param-ckpt", type=Path, default=None)

    group = parser.add_argument_group("W&B logging")
    group.add_argument("--use-wandb", action="store_true")
    group.add_argument("--wandb-entity", type=str, default=None)
    group.add_argument("--wandb-project", type=str, default="leap-c")
    group.add_argument("--wandb-group", type=str, default="baseline-i4b")

    args = parser.parse_args()

    cfg = create_cfg(
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
                args.seed, tags=["baseline", args.policy_type, "i4b", str(args.controller)]
            ),
            "config": config_dict,
        }

    if args.output_path is None:
        output_path = default_output_path(
            seed=args.seed,
            tags=["baseline", args.policy_type, "i4b", str(args.controller)],
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
        args.days,
        args.overwrite,
    )
