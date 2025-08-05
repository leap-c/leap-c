"""Main script to run controller with default parameters."""

from argparse import ArgumentParser
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator
import numpy as np
import gymnasium as gym

from leap_c.examples import create_env, create_controller
from leap_c.torch.rl.buffer import ReplayBuffer
from leap_c.run import default_controller_code_path, default_output_path, init_run
from leap_c.trainer import Trainer, TrainerConfig
from leap_c.controller import ParameterizedController


@dataclass
class ControllerTrainerConfig(TrainerConfig):
    """Configuration for running controller without training."""

    # Override defaults to skip training
    train_steps: int = 1  # No training
    val_interval: int = 1  # Validate immediately


@dataclass
class RunControllerConfig:
    """Configuration for running controller experiments."""

    env: str = "cartpole"
    controller: str = "cartpole"
    trainer: ControllerTrainerConfig = field(default_factory=ControllerTrainerConfig)


class ControllerTrainer(Trainer[ControllerTrainerConfig]):
    """A trainer that just runs the controller with default parameters."""

    def __init__(
        self,
        cfg: ControllerTrainerConfig,
        val_env: gym.Env,
        output_path: str | Path,
        device: str,
        controller: ParameterizedController,
    ):
        super().__init__(cfg, val_env, output_path, device)
        self.controller = controller

        buffer = ReplayBuffer(1, device, collate_fn_map=controller.collate_fn_map)
        self.collate_fn = buffer.collate

    def train_loop(self) -> Iterator[int]:
        """No training - just return immediately."""
        yield 1

    def act(
        self, obs, deterministic: bool = False, state=None
    ) -> tuple[np.ndarray, Any, dict[str, float]]:
        """Use controller with default parameters."""
        obs_batched = self.collate_fn([obs])

        default_param = self.controller.default_param(obs)

        param_batched = self.collate_fn([default_param])

        ctx, action = self.controller(obs_batched, param_batched, ctx=state)

        action = action.cpu().numpy()[0]

        return action, ctx, {}


def create_cfg() -> RunControllerConfig:
    """Return the default configuration for running controller experiments."""
    cfg = RunControllerConfig()
    cfg.env = "cartpole"
    cfg.controller = "cartpole"

    # ---- Section: cfg.trainer ----
    cfg.trainer.seed = 0
    cfg.trainer.train_steps = 1  # No training
    cfg.trainer.train_start = 0
    cfg.trainer.val_interval = 1  # Validate immediately
    cfg.trainer.val_num_rollouts = 20
    cfg.trainer.val_deterministic = True
    cfg.trainer.val_num_render_rollouts = 0
    cfg.trainer.val_render_mode = "rgb_array"
    cfg.trainer.val_render_deterministic = True
    cfg.trainer.val_report_score = "cum"
    cfg.trainer.ckpt_modus = "none"  # No checkpoints needed

    # ---- Section: cfg.trainer.log ----
    cfg.trainer.log.verbose = True
    cfg.trainer.log.interval = 1000
    cfg.trainer.log.window = 10000
    cfg.trainer.log.csv_logger = True
    cfg.trainer.log.tensorboard_logger = True
    cfg.trainer.log.wandb_logger = False
    cfg.trainer.log.wandb_init_kwargs = {}

    return cfg


def run_controller(
    cfg: RunControllerConfig,
    output_path: str | Path,
    device: str = "cpu",
    reuse_code_dir: Path | None = None,
) -> float:
    trainer = ControllerTrainer(
        val_env=create_env(cfg.env, render_mode="rgb_array"),
        controller=create_controller(
            cfg.controller, reuse_code_base_dir=reuse_code_dir
        ),
        output_path=output_path,
        device=device,
        cfg=cfg.trainer,
    )
    init_run(trainer, cfg, output_path)

    print(f"Running controller '{cfg.controller}' on environment '{cfg.env}'")

    final_score = trainer.run()
    print(f"Final validation score: {final_score}")

    return final_score


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--output_path", type=Path, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--env", type=str, default="cartpole")
    parser.add_argument("--controller", type=str, default=None)
    parser.add_argument(
        "-r",
        "--reuse_code",
        action="store_true",
        help="Reuse compiled code. The first time this is run, it will compile the code.",
    )
    parser.add_argument("--reuse_code_dir", type=Path, default=None)
    args = parser.parse_args()

    output_path = default_output_path(
        seed=args.seed, tags=["controller", args.env, args.controller]
    )

    cfg = create_cfg()
    cfg.trainer.seed = args.seed
    cfg.env = args.env
    cfg.controller = args.controller if args.controller else args.env

    if args.reuse_code and args.reuse_code_dir is None:
        reuse_code_dir = default_controller_code_path() if args.reuse_code else None
    elif args.reuse_code_dir is not None:
        reuse_code_dir = args.reuse_code_dir
    else:
        reuse_code_dir = None

    run_controller(
        cfg,
        output_path,
        device=args.device,
        reuse_code_dir=reuse_code_dir,
    )
