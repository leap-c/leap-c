"""Main script to run experiments."""
import datetime
from argparse import ArgumentParser
from pathlib import Path

from leap_c.run import main
from leap_c.torch.rl import SacBaseConfig


parser = ArgumentParser()
parser.add_argument("--output_path", type=Path, default=None)
parser.add_argument("--task", type=str, default="point_mass")
parser.add_argument("--trainer", type=str, default="sac_fop")
parser.add_argument("--device", type=str, default="cpu")
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()

cfg = SacBaseConfig()
cfg.train.steps = 1000000
cfg.train.start = 0
cfg.val.interval = 10000
cfg.val.num_rollouts = 10
cfg.val.deterministic = True
cfg.val.ckpt_modus = 'best'
cfg.val.num_render_rollouts = 1
cfg.val.render_mode = 'rgb_array'
cfg.val.render_deterministic = True
cfg.val.report_score = 'cum'
cfg.log.verbose = True
cfg.log.interval = 1000
cfg.log.window = 10000
cfg.log.csv_logger = True
cfg.log.tensorboard_logger = True
cfg.log.wandb_logger = False
cfg.log.wandb_init_kwargs = {}
cfg.seed = 0
cfg.sac.critic_mlp.hidden_dims = (256, 256, 256)
cfg.sac.critic_mlp.activation = 'relu'
cfg.sac.critic_mlp.weight_init = 'orthogonal'
cfg.sac.actor_mlp.hidden_dims = (256, 256, 256)
cfg.sac.actor_mlp.activation = 'relu'
cfg.sac.actor_mlp.weight_init = 'orthogonal'
cfg.sac.batch_size = 64
cfg.sac.buffer_size = 1000000
cfg.sac.gamma = 0.99
cfg.sac.tau = 0.005
cfg.sac.soft_update_freq = 1
cfg.sac.lr_q = 0.0001
cfg.sac.lr_pi = 0.0003
cfg.sac.lr_alpha = 0.001
cfg.sac.init_alpha = 0.1
cfg.sac.target_entropy = None
cfg.sac.entropy_reward_bonus = True
cfg.sac.num_critics = 2
cfg.sac.report_loss_freq = 100
cfg.sac.update_freq = 4

time_str = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
output_path = Path(f"output/{args.task}/{args.trainer}_{args.seed}_{time_str}")

main(args.trainer, args.task, cfg, output_path, args.device)
