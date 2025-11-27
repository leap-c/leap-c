from typing import Any, Callable, TypeAlias

import numpy as np
from gymnasium import Env, spaces
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
    env.observation_space.seed(seed)
    env.action_space.seed(seed)
    return env.reset(seed=seed, options=options)


def check_params_not_in_space(
    param: np.ndarray,
    param_space: spaces.Box,
) -> list[tuple[int, float, float, float]]:
    """Check which parameters are not within the param_space bounds.

    Args:
        param: Array of parameter values
        param_space: Parameter space with bounds

    Returns:
        List of tuples (index, param_value, low_bound, high_bound) for out-of-bounds params
    """
    if param_space.contains(param):
        return []

    out_of_bounds = []
    low = param_space.low
    high = param_space.high

    for i, (p, l, h) in enumerate(zip(param, low, high)):
        if p < l or p > h:
            out_of_bounds.append((i, p, l, h))

    return out_of_bounds
