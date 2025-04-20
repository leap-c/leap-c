from pathlib import Path

from leap_c.nn.mlp import MlpConfig
from leap_c.run import main
from leap_c.rl.ppo import PpoBaseConfig

if __name__ == "__main__":
    cfg = PpoBaseConfig()
    cfg.ppo.actor_mlp = MlpConfig(
        hidden_dims=(64, 64),
        activation="tanh",
        weight_init="orthogonal",
    )
    cfg.ppo.critic_mlp = MlpConfig(
        hidden_dims=(64, 64),
        activation="tanh",
        weight_init="orthogonal",
    )
    cfg.ppo.num_steps = 256
    cfg.ppo.num_mini_batches = 8
    cfg.ppo.update_epochs = 10
    cfg.train.steps = 500 * 256
    cfg.val.interval = 50 * 256

    output_path = Path(f"output/pendulum_swingup/ppo")

    main("ppo", "pendulum_swingup", cfg, output_path, "cuda")
