"""Tiny SAC-FOP smoke run for the race_car pipeline.

Run with:
    python scripts/race_car/_smoke_sac_fop.py
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_sac_fop import create_cfg, run_sac_fop  # noqa: E402


def main() -> None:
    cfg = create_cfg(controller="race_car", seed=0, variant="fop", ckpt_modus="last")
    cfg.trainer.train_steps = 3_000
    cfg.trainer.train_start = 500
    cfg.trainer.val_freq = 1_500
    cfg.trainer.val_num_rollouts = 1
    cfg.trainer.val_num_render_rollouts = 1
    cfg.trainer.val_render_mode = "rgb_array"
    cfg.trainer.log.interval = 200
    cfg.max_steps = 300

    out = Path("output/race_car_sac_fop_smoke")
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    run_sac_fop(
        cfg=cfg,
        output_path=out,
        device=torch.device("cpu"),
        dtype=torch.float32,
        reuse_code_dir=Path("output/controller_code"),
        with_val=True,
    )

    print("\n=== smoke outputs ===")
    for p in sorted(out.rglob("*")):
        if p.is_file():
            print(p.relative_to(out), p.stat().st_size)


if __name__ == "__main__":
    main()
