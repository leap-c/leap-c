from pathlib import Path
from tempfile import TemporaryDirectory

from leap_c.torch.rl.sac import SacTrainer, SacTrainerConfig
from leap_c.examples.cartpole.env import CartPoleEnv


def test_trainer_checkpointing():
    """
    Test the checkpointing functionality of the Trainer class.

    This test verifies that the Trainer class can correctly save and load
    checkpoints, including the state of the model, optimizer, and other
    training parameters.
    """
    with TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        val_env = CartPoleEnv()
        train_env = CartPoleEnv()

        trainer = SacTrainer(
            cfg=SacTrainerConfig(),
            val_env=val_env,
            output_path=tmpdir,
            device="cpu",
            train_env=train_env,
        )

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
