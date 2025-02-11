import gymnasium as gym
from leap_c.task import Task
from leap_c.registry import register_task


@register_task("half_cheetah")
class HalfCheetahTask(Task):
    def __init__(self):
        env_factory = lambda: gym.make("HalfCheetah-v5")
        super().__init__(None, env_factory)  # type: ignore

