import gymnasium as gym
from gymnasium.wrappers import OrderEnforcing, RecordEpisodeStatistics



def wrap_train_env(env: gym.Env, seed: int = 0) -> gym.Env:
    """Wraps a gymnasium environment for training.

    Args:
        env: The environment to wrap.
        seed: The seed for the environment.

    Returns:
        gym.Env: The wrapped environment for training.
    """
    env = RecordEpisodeStatistics(env, buffer_length=1)
    env = OrderEnforcing(env)
    env.reset(seed=seed)
    env.observation_space.seed(seed)
    env.action_space.seed(seed)

    return env


def wrap_eval_env(env: gym.Env, seed: int = 1) -> gym.Env:
    """Wraps a gymnasium environment for evaluation.

    Args:
        env: The environment to wrap.
        seed: The seed for the environment.

    Returns:
        gym.Env: The wrapped environment for evaluation.
    """
    env = OrderEnforcing(env)
    env.reset(seed=seed)
    env.observation_space.seed(seed)
    env.action_space.seed(seed)

    return env
