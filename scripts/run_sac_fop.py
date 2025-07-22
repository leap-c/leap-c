"""Main script to run experiments."""
from argparse import ArgumentParser
from dataclasses import dataclass, field
from pathlib import Path

from leap_c.examples import create_env, create_controller
from leap_c.run import default_controller_code_path, default_output_path, init_run
from leap_c.torch.rl.sac_fop import SacFopTrainer, SacFopTrainerConfig


@dataclass
class RunSacFopConfig:
    """Configuration for running SAC experiments."""

    env_name: str = "cartpole"
    controller_name: str = "cartpole"
    device: str = "cuda"  # or 'cpu'
    trainer: SacFopTrainerConfig = field(default_factory=SacFopTrainerConfig)


def run_sac_fop(
    trainer_output_path: str | Path,
    env_name: str,
    controller_name: str,
    seed: int = 0,
    device: str = "cuda",
    verbose: bool = True,
    reuse_code_dir: Path | None = None,
) -> float:
    # ---- Configuration ----
    cfg = RunSacFopConfig(env_name=env_name, device=device)
    cfg.env_name = env_name
    cfg.trainer.seed = seed
    cfg.controller_name = controller_name

    # ---- Section: cfg.trainer ----
    cfg.trainer.seed = 0
    cfg.trainer.train_steps = 1000000
    cfg.trainer.train_start = 0
    cfg.trainer.val_interval = 10000
    cfg.trainer.val_num_rollouts = 20
    cfg.trainer.val_deterministic = True
    cfg.trainer.val_num_render_rollouts = 1
    cfg.trainer.val_render_mode = "rgb_array"
    cfg.trainer.val_render_deterministic = True
    cfg.trainer.val_report_score = "cum"
    cfg.trainer.ckpt_modus = "best"
    cfg.trainer.batch_size = 64
    cfg.trainer.buffer_size = 1000000
    cfg.trainer.gamma = 0.99
    cfg.trainer.tau = 0.005
    cfg.trainer.soft_update_freq = 1
    cfg.trainer.lr_q = 0.001
    cfg.trainer.lr_pi = 0.001
    cfg.trainer.lr_alpha = 0.001
    cfg.trainer.init_alpha = 0.02
    cfg.trainer.target_entropy = None
    cfg.trainer.entropy_reward_bonus = True
    cfg.trainer.num_critics = 2
    cfg.trainer.report_loss_freq = 100
    cfg.trainer.update_freq = 4
    cfg.trainer.noise = 'param'
    cfg.trainer.entropy_correction = False

    # ---- Section: cfg.trainer.log ----
    cfg.trainer.log.verbose = verbose
    cfg.trainer.log.interval = 1000
    cfg.trainer.log.window = 10000
    cfg.trainer.log.csv_logger = True
    cfg.trainer.log.tensorboard_logger = True
    cfg.trainer.log.wandb_logger = False
    cfg.trainer.log.wandb_init_kwargs = {}

    # ---- Section: cfg.trainer.critic_mlp ----
    cfg.trainer.critic_mlp.hidden_dims = (256, 256, 256)
    cfg.trainer.critic_mlp.activation = "relu"
    cfg.trainer.critic_mlp.weight_init = "orthogonal"

    # ---- Section: cfg.trainer.actor_mlp ----
    cfg.trainer.actor_mlp.hidden_dims = (256, 256, 256)
    cfg.trainer.actor_mlp.activation = "relu"
    cfg.trainer.actor_mlp.weight_init = "orthogonal"

    trainer = SacFopTrainer(
        val_env=create_env(cfg.env_name, render_mode="rgb_array"),
        train_env=create_env(cfg.env_name),
        controller=create_controller(cfg.controller_name, reuse_code_dir),
        output_path=trainer_output_path,
        device=device,
        cfg=cfg.trainer,
    )
    init_run(trainer, cfg, trainer_output_path)

    return trainer.run()


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

    if args.controller is None:
        args.controller = args.env

    if args.output_path is None:
        trainer_output_path = default_output_path(
            seed=args.seed, tags=["sac_fop", args.env, args.controller]
        )
    else:
        trainer_output_path = args.output_path

    if args.reuse_code and args.reuse_code_dir is None:
        reuse_code_dir = (
            default_controller_code_path() if args.reuse_code else None
        )
    elif args.reuse_code_dir is not None:
        reuse_code_dir = args.reuse_code_dir
    else:
        reuse_code_dir = None

    run_sac_fop(
        trainer_output_path,
        env_name=args.env,
        controller_name=args.controller,
        seed=args.seed,
        device=args.device,
        verbose=True,
        reuse_code_dir=reuse_code_dir,
    )
