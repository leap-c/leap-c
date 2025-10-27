"""Main script to run DQN experiments."""

from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from dataclasses import asdict, dataclass, field
from pathlib import Path

from leap_c.examples import ExampleEnvName, create_env
from leap_c.run import default_name, default_output_path, init_run
from leap_c.torch.nn.extractor import ExtractorName
from leap_c.torch.rl.dqn import DqnTrainer, DqnTrainerConfig


@dataclass
class RunDqnConfig:
    """Configuration for running DQN experiments.

    Attributes:
        env: The environment name.
        trainer: The trainer configuration.
        extractor: The feature extractor to use.
    """

    env: ExampleEnvName = "CartPole-v1"
    trainer: DqnTrainerConfig = field(default_factory=DqnTrainerConfig)
    extractor: ExtractorName = "identity"


def create_cfg(env: str, seed: int) -> RunDqnConfig:
    """Return the default configuration for running DQN experiments.

    Default values inspired by CleanRL's DQN implementation."""

    # ---- Configuration ----
    cfg = RunDqnConfig()
    cfg.env = env
    cfg.extractor = "identity" if env != "hvac" else "scaling"

    # ---- Section: cfg.trainer (base) ----
    cfg.trainer.seed = seed
    cfg.trainer.train_steps = 500_000
    cfg.trainer.train_start = 10_000
    cfg.trainer.val_freq = 10_000
    cfg.trainer.val_num_rollouts = 20
    cfg.trainer.val_deterministic = True
    cfg.trainer.val_num_render_rollouts = 0
    cfg.trainer.val_render_mode = "rgb_array"
    cfg.trainer.val_report_score = "cum"
    cfg.trainer.ckpt_modus = "best"

    # ---- Section: cfg.trainer (DQN) ----
    cfg.trainer.batch_size = 128
    cfg.trainer.buffer_size = 10_000
    cfg.trainer.gamma = 0.99
    cfg.trainer.lr = 2.5e-4
    cfg.trainer.update_freq = 10
    cfg.trainer.soft_update_freq = 500
    cfg.trainer.tau = 1.0
    cfg.trainer.gradient_steps = 1
    cfg.trainer.start_exploration = 1.0
    cfg.trainer.end_exploration = 0.05
    cfg.trainer.exploration_fraction = 0.5

    # ---- Section: cfg.trainer.critic_mlp ----
    cfg.trainer.critic_mlp.hidden_dims = (120, 84, 84)
    cfg.trainer.critic_mlp.activation = "relu"
    cfg.trainer.critic_mlp.weight_init = "orthogonal"

    # ---- Section: cfg.trainer.log ----
    cfg.trainer.log.verbose = True
    cfg.trainer.log.interval = 1_000
    cfg.trainer.log.window = 10_000
    cfg.trainer.log.csv_logger = True
    cfg.trainer.log.tensorboard_logger = False
    cfg.trainer.log.wandb_logger = False
    cfg.trainer.log.wandb_init_kwargs = {}
    return cfg


def run_dqn(cfg: RunDqnConfig, output_path: str | Path, device: str = "cuda") -> float:
    """Run the DQN trainer.

    Args:
        cfg: The configuration for running the controller.
        output_path: The path to save outputs to.
            If it already exists, the run will continue from the last checkpoint.
        device: The device to use.

    Raises:
        ValueError: If the action space cannot be discretized (in case it is not discrete).
    """
    train_env = create_env(cfg.env)
    eval_env = create_env(cfg.env, render_mode="rgb_array")
    trainer = DqnTrainer(cfg.trainer, train_env, eval_env, output_path, device, cfg.extractor)
    init_run(trainer, cfg, output_path)
    return trainer.run()


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Training of DQN agents.", formatter_class=ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--output-path", type=Path, default=None, help="Path to outputs (e.g., logs)."
    )
    parser.add_argument("--device", type=str, default="cpu", help="Device to run on.")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed.")
    parser.add_argument("--env", type=str, default="CartPole-v1", help="Environment to train on.")
    parser.add_argument("--use-wandb", action="store_true", help="Whether to use W&B logging.")
    parser.add_argument("--wandb-entity", type=str, default=None, help="W&B entity name.")
    parser.add_argument("--wandb-project", type=str, default="leap-c", help="W&B project name.")
    args = parser.parse_args()

    cfg = create_cfg(args.env, args.seed)

    method = "dqn"
    if args.use_wandb:
        config_dict = asdict(cfg)
        cfg.trainer.log.wandb_logger = True
        cfg.trainer.log.wandb_init_kwargs = {
            "entity": args.wandb_entity,
            "project": args.wandb_project,
            "name": default_name(args.seed, tags=[method, args.env]),
            "config": config_dict,
        }

    output_path = (
        args.output_path
        if args.output_path
        else default_output_path(seed=args.seed, tags=[method, args.env])
    )

    run_dqn(cfg, output_path, args.device)
