from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from leap_c.examples.pointmass.task import PointMassEasyTask as PointMassTask
from leap_c.registry import create_default_cfg
from leap_c.utils.resurrect import resurrect_cfg, resurrect_task, resurrect_trainer
from leap_c.rl.sac import SacBaseConfig, SacTrainer
from leap_c.run import main


@pytest.fixture(scope="session")
def resurrect_dir():
    """Fixture for the SAC trainer."""
    cfg: SacBaseConfig = create_default_cfg("sac")  # type: ignore

    cfg.sac.lr_q = 12345
    cfg.train.steps = 3
    cfg.val.interval = 2
    cfg.val.num_rollouts = 1
    cfg.val.num_render_rollouts = 0

    with TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        main("sac", "point_mass_easy", cfg, tmpdir, "cpu")

        yield tmpdir


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
