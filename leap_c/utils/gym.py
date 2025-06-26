from typing import List, Callable

import gymnasium as gym
from gymnasium.core import ObsType, ActType
from gymnasium.wrappers import OrderEnforcing, RecordEpisodeStatistics

WrapperType = Callable[[gym.Env[ObsType, ActType]], gym.Env[ObsType, ActType]]


def wrap_env(env: gym.Env, wrappers: List[WrapperType] | None = None) -> gym.Env:
    """Wraps a gymnasium environment.

    Args:
        env: The environment to wrap.
        wrappers: A list of wrappers to apply to the environment.

    Returns:
        gym.Env: The wrapped environment.
    """
    env = RecordEpisodeStatistics(env, buffer_length=1)
    env = OrderEnforcing(env)

    if wrappers is None:
        wrappers = []
    for wrapper in wrappers:
        env = wrapper(env)

    return env


def seed_env(env: gym.Env, seed: int = 0) -> gym.Env:
    """Seeds the environment.

    Args:
        env: The environment to seed.
        seed: The seed to use.

    Returns:
        gym.Env: The seeded environment.
    """
    env.reset(seed=seed)
    env.observation_space.seed(seed)
    env.action_space.seed(seed)

    return env
