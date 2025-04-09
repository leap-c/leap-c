from pathlib import Path

from leap_c.resurrect import resurrect_cfg, resurrect_task, resurrect_trainer
from leap_c.rl.sac import SacTrainer, SacBaseConfig
from leap_c.examples.pointmass.task import PointMassTask


def test_resurrect_cfg(resurrect_dir: Path):
    cfg = resurrect_cfg(resurrect_dir)

    assert isinstance(cfg, SacBaseConfig)
    assert cfg.sac.lr_q == 12345


def test_resurrect_task(resurrect_dir: Path):
    task = resurrect_task(resurrect_dir)

    assert isinstance(task, PointMassTask)


def test_resurrect_trainer(resurrect_dir: Path):
    cfg = resurrect_cfg(resurrect_dir)

    trainer = resurrect_trainer(resurrect_dir, device="cpu")

    assert isinstance(trainer, SacTrainer)
    assert isinstance(trainer.task, PointMassTask)
    assert trainer.cfg == cfg
    assert trainer.state.step == 2
