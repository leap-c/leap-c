from pathlib import Path
from tempfile import TemporaryDirectory

from leap_c.examples.cartpole.env import CartPoleEnv
from leap_c.torch.rl.sac import SacTrainer, SacTrainerConfig


def test_trainer_checkpointing():
    """Test the checkpointing functionality of the Trainer class.

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


def test_trainer_run_with_eval_env():
    """Test that the trainer can run with a validation environment.

    This test verifies that training works correctly when an eval_env is provided,
    using validation episodes to compute scores.
    """
    with TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        val_env = CartPoleEnv()
        train_env = CartPoleEnv()

        cfg = SacTrainerConfig(
            train_steps=10,
            train_start=0,
            val_freq=5,
            val_num_rollouts=2,
            ckpt_modus="best",
        )

        trainer = SacTrainer(
            cfg=cfg,
            val_env=val_env,
            output_path=tmpdir,
            device="cpu",
            train_env=train_env,
        )

        # Should run without errors
        score = trainer.run()

        # Score should be a float (validation score)
        assert isinstance(score, float)
        # Training should have completed
        assert trainer.state.step >= cfg.train_steps
        # Should have validation scores recorded
        assert len(trainer.state.scores) > 0
