import os
import shutil

import numpy as np
import pytest
from leap_c.examples.linear_system import LinearSystemOcpEnv
from leap_c.logging import NumberLogger
from leap_c.scripts.rl.linear_system_sac import (
    LinearSystemSACConfig,
    LinearSystemSACTrainer,
    Scenario,
    create_actor,
    create_mpc,
    create_qnet,
    create_replay_buffer,
    run_linear_system_sac,
    standard_config_dict,
)
from leap_c.util import create_dir_if_not_exists


def test_standard_sac_does_it_run():
    scenario = Scenario.STANDARD_SAC
    seed = 31337
    cwd = os.getcwd()
    savefile_dir_path = os.path.join(cwd, "test_standard_sac")
    create_dir_if_not_exists(savefile_dir_path)
    config_dict = standard_config_dict(scenario, savefile_dir_path)
    G = np.array([[1.0, 0.25], [0.0, 1.0]])
    config_dict = {
        **config_dict,
        **dict(  # type:ignore
            seed=seed,
            dont_train_until_this_many_transitions=4,
            max_episodes=10,
            training_steps_per_episode=2,
            batch_size=4,
            val_interval=8,
            render_mode="rgb_array",
            render_interval_exploration=5,
            render_interval_validation=2,
            default_params={
                "A": np.array([[1.0, 0.25], [0.0, 1.0]]),
                "B": np.array([[0.03125], [0.25]]),
                "Q": G.T @ G,  # np.identity(2),
                "R": np.identity(1),
                "b": np.array([[0.0], [0.0]]),
                "f": np.array([[0.0], [0.0], [0.0]]),
                "V_0": np.array([1e-3]),
            },
        ),
    }
    val = run_linear_system_sac(
        scenario=scenario,
        savefile_directory_path=savefile_dir_path,
        config_kwargs=config_dict,
    )
    assert val is not None  # Just to check that some validation occurred

    config = LinearSystemSACConfig(**config_dict)
    mpc = create_mpc(config)

    env = LinearSystemOcpEnv(mpc, dt=config.dt, max_time=config.max_time)
    config.a_dim = env.action_space.shape[0]  # type:ignore
    config.s_dim = env.state_space.shape[0]
    config.p_learnable_dim = env.p_learnable_space.shape[0]  # type:ignore
    config.target_entropy = -config.a_dim

    actor = create_actor(config, scenario, mpc)
    qnet1 = create_qnet(config)
    qnet2 = create_qnet(config)
    qnet1_target = create_qnet(config)
    qnet2_target = create_qnet(config)
    buffer = create_replay_buffer(config)
    logger = NumberLogger(
        save_directory_path=savefile_dir_path,
        save_frequency=1,
        moving_average_width=2,
    )
    trainer = LinearSystemSACTrainer(
        actor, qnet1, qnet2, qnet1_target, qnet2_target, buffer, logger, config
    )
    iwannaload = os.path.join(savefile_dir_path, "episode_0")
    trainer.load(iwannaload)  # Test running loading.
    # Clean up after yourself
    # NOTE: Comment this out if you want to see the rendering.
    shutil.rmtree(savefile_dir_path)


def test_fou_sac_does_it_run():
    scenario = Scenario.FO_U_SAC
    seed = 31337
    cwd = os.getcwd()
    savefile_dir_path = os.path.join(cwd, "test_fou_sac")
    create_dir_if_not_exists(savefile_dir_path)
    config_dict = standard_config_dict(scenario, savefile_dir_path)
    config_dict = {
        **config_dict,
        **dict(  # type:ignore
            seed=seed,
            dont_train_until_this_many_transitions=4,
            max_episodes=10,
            training_steps_per_episode=2,
            batch_size=4,
            val_interval=8,
            render_mode="rgb_array",
            render_interval_exploration=5,
            render_interval_validation=2,
        ),
    }
    val = run_linear_system_sac(
        scenario=scenario,
        savefile_directory_path=savefile_dir_path,
        config_kwargs=config_dict,
    )
    assert val is not None  # Just to check that some validation occurred
    config = LinearSystemSACConfig(**config_dict)
    # NOTE: Assumes these things do not have changed in the standard version.

    mpc = create_mpc(config)

    env = LinearSystemOcpEnv(mpc, dt=config.dt, max_time=config.max_time)
    config.a_dim = env.action_space.shape[0]  # type:ignore
    config.s_dim = env.state_space.shape[0]
    config.p_learnable_dim = env.p_learnable_space.shape[0]  # type:ignore
    config.target_entropy = -config.a_dim

    actor = create_actor(config, scenario, mpc)
    qnet1 = create_qnet(config)
    qnet2 = create_qnet(config)
    qnet1_target = create_qnet(config)
    qnet2_target = create_qnet(config)
    buffer = create_replay_buffer(config)
    logger = NumberLogger(
        save_directory_path=savefile_dir_path,
        save_frequency=1,
        moving_average_width=2,
    )
    trainer = LinearSystemSACTrainer(
        actor, qnet1, qnet2, qnet1_target, qnet2_target, buffer, logger, config
    )
    iwannaload = os.path.join(savefile_dir_path, "episode_0")
    trainer.load(iwannaload)  # Test running loading.
    # Clean up after yourself
    # NOTE: Comment this out if you want to see the rendering.
    shutil.rmtree(savefile_dir_path)


if __name__ == "main":
    pytest.main([__file__])
