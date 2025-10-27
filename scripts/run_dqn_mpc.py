"""Main script to run DQN-MPC experiments."""

from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from dataclasses import asdict, dataclass, field
from pathlib import Path

from leap_c.examples import ExampleControllerName, ExampleEnvName, create_controller, create_env
from leap_c.run import default_controller_code_path, default_name, default_output_path, init_run
from leap_c.torch.nn.extractor import ExtractorName
from leap_c.torch.rl.dqn_mpc import DqnMpcTrainer, DqnMpcTrainerConfig


@dataclass
class RunDqnMpcConfig:
    """Configuration for running DQN-MPC experiments.

    Attributes:
        env: The environment name.
        controller: The controller name.
        trainer: The trainer configuration.
        extractor: The feature extractor to use.
    """

    env: ExampleEnvName = "cartpole"
    controller: ExampleControllerName = "cartpole"
    trainer: DqnMpcTrainerConfig = field(default_factory=DqnMpcTrainerConfig)
    extractor: ExtractorName = "identity"


def create_cfg(env: str, controller: str, seed: int) -> RunDqnMpcConfig:
    # ---- Configuration ----
    cfg = RunDqnMpcConfig()
    cfg.env = env
    cfg.controller = controller if controller is not None else env
    cfg.extractor = "identity" if env != "hvac" else "scaling"

    # ---- Section: cfg.trainer (base) ----
    cfg.trainer.seed = seed
    cfg.trainer.train_steps = 200_000
    cfg.trainer.train_start = 20_000
    cfg.trainer.val_freq = 10_000
    cfg.trainer.val_num_rollouts = 20
    cfg.trainer.val_deterministic = True
    cfg.trainer.val_num_render_rollouts = 1
    cfg.trainer.val_render_mode = "rgb_array"
    cfg.trainer.val_report_score = "cum"
    cfg.trainer.ckpt_modus = "best"

    # ---- Section: cfg.trainer (DQN-MPC) ----
    cfg.trainer.batch_size = 128
    cfg.trainer.buffer_size = 100_000
    cfg.trainer.gamma = 0.99
    cfg.trainer.lr = 3e-5
    cfg.trainer.update_freq = 4
    cfg.trainer.soft_update_freq = 1
    cfg.trainer.tau = 5e-3
    cfg.trainer.gradient_steps = 1
    cfg.trainer.start_exploration = 1.0
    cfg.trainer.end_exploration = 0.05
    cfg.trainer.exploration_fraction = 0.5
    cfg.trainer.init_param_with_default = True
    cfg.trainer.num_threads_batch_solver = 2**2  # NOTE: adjust based on available compute

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


def run_dqn_mpc(
    cfg: RunDqnMpcConfig,
    output_path: str | Path,
    device: str = "cuda",
    reuse_code_dir: Path | None = None,
) -> float:
    """Run the DQN-MPC trainer.

    Args:
        cfg: The configuration for running the controller.
        output_path: The path to save outputs to.
            If it already exists, the run will continue from the last checkpoint.
        device: The device to use.
        reuse_code_dir: The directory to reuse compiled code from, if any.
    """
    train_env = create_env(cfg.env)
    eval_env = create_env(cfg.env, render_mode="rgb_array")
    controller = create_controller(cfg.controller, reuse_code_dir)
    trainer = DqnMpcTrainer(
        cfg.trainer, train_env, eval_env, controller, output_path, device, cfg.extractor
    )
    init_run(trainer, cfg, output_path)
    return trainer.run()


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Training of DQN-MPC agents.", formatter_class=ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--output-path", type=Path, default=None, help="Path to outputs (e.g., logs)."
    )
    parser.add_argument("--device", type=str, default="cpu", help="Device to run on.")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed.")
    parser.add_argument("--env", type=str, default="cartpole", help="Environment to train on.")
    parser.add_argument(
        "--controller",
        type=str,
        default=None,
        help="Controller to use in the environment. If `None`, will be set equal to `--env`.",
    )
    parser.add_argument(
        "-r",
        "--reuse-code",
        action="store_true",
        help="Reuse compiled code. The first time this is run, it will compile the code anyhow.",
    )
    parser.add_argument(
        "--reuse-code-dir", type=Path, default=None, help="Directory for compiled code."
    )
    parser.add_argument("--use-wandb", action="store_true", help="Whether to use W&B logging.")
    parser.add_argument("--wandb-entity", type=str, default=None, help="W&B entity name.")
    parser.add_argument("--wandb-project", type=str, default="leap-c", help="W&B project name.")
    args = parser.parse_args()

    cfg = create_cfg(args.env, args.controller, args.seed)

    method = "dqn_mpc"
    if args.use_wandb:
        config_dict = asdict(cfg)
        cfg.trainer.log.wandb_logger = True
        cfg.trainer.log.wandb_init_kwargs = {
            "entity": args.wandb_entity,
            "project": args.wandb_project,
            "name": default_name(args.seed, tags=[method, args.env, args.controller]),
            "config": config_dict,
        }

    output_path = (
        args.output_path
        if args.output_path
        else default_output_path(seed=args.seed, tags=[method, args.env, args.controller])
    )

    if args.reuse_code and args.reuse_code_dir is None:
        reuse_code_dir = default_controller_code_path()
    elif args.reuse_code_dir is not None:
        reuse_code_dir = args.reuse_code_dir
    else:
        reuse_code_dir = None

    run_dqn_mpc(cfg, output_path, args.device, reuse_code_dir)
