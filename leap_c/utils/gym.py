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
    unwrapped_ = env.unwrapped
    env_ = env
    already_wrapped_with_record_stats = already_wrapped_with_order_enf = False
    while env_ is not unwrapped_:
        if isinstance(env_, RecordEpisodeStatistics) and env_.time_queue.maxlen == 1:
            already_wrapped_with_record_stats = True
        if isinstance(env_, OrderEnforcing):
            already_wrapped_with_order_enf = True
        env_ = env_.env

    if not already_wrapped_with_record_stats:
        env = RecordEpisodeStatistics(env, buffer_length=1)
    if not already_wrapped_with_order_enf:
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
