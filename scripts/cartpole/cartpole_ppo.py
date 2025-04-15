from pathlib import Path

from leap_c.run import main
from leap_c.rl.ppo import PpoBaseConfig

if __name__ == "__main__":
    cfg = PpoBaseConfig()
    cfg.ppo.num_steps = 5
    cfg.train.steps = 15

    output_path = Path(f"output/cartpole/ppo")

    main("ppo", "cartpole", cfg, output_path, "cpu")
