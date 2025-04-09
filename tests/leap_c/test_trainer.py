from pathlib import Path
from tempfile import TemporaryDirectory

from leap_c.trainer import Trainer


def test_trainer_checkpointing(trainer: Trainer):
    """Test the checkpointing functionality of the Trainer class.

    This test verifies that the Trainer class can correctly save and load
    checkpoints, including the state of the model, optimizer, and other
    training parameters.
    """

    with TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        orig_step = trainer.state.step
        orig_param = next(trainer.parameters()).data.clone()

        # save a checkpoint
        trainer.save(tmpdir)

        # change something in trainer state
        trainer.state.step = 1000

        # change a parameter in a model
        param = next(trainer.parameters())
        param.data = param.data + 1

        # load the checkpoint
        trainer.load(tmpdir)
        # check if the step is restored
        assert trainer.state.step == orig_step
        # check if the parameter is restored
        param = next(trainer.parameters())
        assert param.data.equal(orig_param)
