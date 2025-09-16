from typing import Any, Callable, TypeAlias

from gymnasium import Env
from gymnasium.core import ActType, ObsType
from gymnasium.wrappers import OrderEnforcing, RecordEpisodeStatistics

WrapperType: TypeAlias = Callable[[Env[ObsType, ActType]], Env[ObsType, ActType]]


def wrap_env(
    env: Env[ObsType, ActType], wrappers: list[WrapperType] | None = None
) -> Env[ObsType, ActType]:
    """Wraps a gymnasium environment.

    Args:
        env: The environment to wrap.
        wrappers: A list of wrappers to apply to the environment.

    Returns:
        gymnasium.Env: The wrapped environment.
    """
    env = RecordEpisodeStatistics(env, buffer_length=1)
    env = OrderEnforcing(env)
    if wrappers:
        for wrapper in wrappers:
            env = wrapper(env)
    return env


def seed_env(
    env: Env[ObsType, ActType], seed: int = 0, options: dict[str, Any] | None = None
) -> tuple[ObsType, dict[str, Any]]:
    """Seeds the environment.

    Args:
        env: The environment to seed.
        seed: The seed to use.
        options: Additional options to pass to `env.reset`.

    Returns:
        tuple: The output of `env.reset`, i.e., the initial observation and info dictionary.
    """
    out = env.reset(seed=seed, options=options)
    env.observation_space.seed(seed)
    env.action_space.seed(seed)
    return out
