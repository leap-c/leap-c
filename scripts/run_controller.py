"""Main script to run controller with default parameters."""
from argparse import ArgumentParser
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator
import numpy as np
import gymnasium as gym

from leap_c.examples import create_env, create_controller
from leap_c.torch.rl.buffer import ReplayBuffer
from leap_c.run import default_output_path, init_run
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
    
    env_name: str = "cartpole"
    controller_name: str = "cartpole"
    device: str = "cpu"
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
        self.default_param = controller.default_param

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
        param_batched = self.collate_fn([self.default_param])
        
        ctx, action = self.controller(obs_batched, param_batched, ctx=state)

        action = action.cpu().numpy()[0]
        
        return action, ctx, {}


def run_controller(
    output_path: str | Path,
    env_name: str,
    controller_name: str,
    seed: int = 0,
    device: str = "cpu",
    verbose: bool = True,
) -> float:
    # ---- Configuration ----
    cfg = RunControllerConfig(env_name=env_name, device=device)
    cfg.env_name = env_name
    cfg.trainer.seed = seed
    cfg.controller_name = controller_name

    # ---- Section: cfg.trainer ----
    cfg.trainer.seed = seed
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
    cfg.trainer.log.verbose = verbose
    cfg.trainer.log.interval = 1000
    cfg.trainer.log.window = 10000
    cfg.trainer.log.csv_logger = True
    cfg.trainer.log.tensorboard_logger = True
    cfg.trainer.log.wandb_logger = False
    cfg.trainer.log.wandb_init_kwargs = {}

    trainer = ControllerTrainer(
        val_env=create_env(cfg.env_name, render_mode="rgb_array"),
        controller=create_controller(cfg.controller_name),
        output_path=output_path,
        device=device,
        cfg=cfg.trainer,
    )
    init_run(trainer, cfg, output_path)

    print(f"Running controller '{controller_name}' on environment '{env_name}'")
    print(f"Default parameters: {trainer.default_param}")

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
    args = parser.parse_args()

    output_path = default_output_path(seed=args.seed, tags=["controller", args.env, args.controller])

    if args.controller is None:
        args.controller = args.env

    run_controller(
        output_path,
        env_name=args.env,
        controller_name=args.controller,
        seed=args.seed,
        device=args.device,
    )
