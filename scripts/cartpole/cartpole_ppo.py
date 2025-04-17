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
    cfg.ppo.num_steps = 128
    cfg.train.steps = 1000 * 128
    cfg.val.interval = 100 * 128

    output_path = Path(f"output/cartpole/ppo")

    main("ppo", "cartpole", cfg, output_path, "cuda")
