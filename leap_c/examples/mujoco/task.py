import gymnasium as gym
from leap_c.task import Task
from leap_c.registry import register_task


@register_task("half_cheetah")
class HalfCheetahTask(Task):
    def __init__(self):
        super().__init__(None)

    def create_env(self, train: bool = True) -> gym.Env:
        return gym.make("HalfCheetah-v5")

