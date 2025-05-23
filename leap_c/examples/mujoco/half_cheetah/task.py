from typing import Any, Optional

import gymnasium as gym
import torch

from leap_c.acados.mpc import MpcInput
from leap_c.registry import register_task
from leap_c.task import Task


@register_task("half_cheetah")
class HalfCheetahTask(Task):
    def __init__(self):
        super().__init__(None)

    def create_env(self, train: bool = True) -> gym.Env:
        return gym.make("HalfCheetah-v5") if train else gym.make("HalfCheetah-v5", render_mode="rgb_array")

    def prepare_mpc_input(
        self,
        obs: Any,
        param_nn: Optional[torch.Tensor] = None,
        action: Optional[torch.Tensor] = None,
    ) -> MpcInput:
        return MpcInput(obs, param_nn, action)
