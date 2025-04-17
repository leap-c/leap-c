from typing import Any, Optional

import gymnasium as gym
import torch

from leap_c.mpc import MpcInput
from leap_c.nn.extractor import Extractor, IdentityExtractor
from leap_c.registry import register_task
from leap_c.task import Task


@register_task("cartpole")
class CartPoleTask(Task):
    def __init__(self):
        super().__init__(mpc=None)

    def create_env(self, train: bool) -> gym.Env:
        return gym.make("CartPole-v1") if train else gym.make("CartPole-v1", render_mode="rgb_array")

    def prepare_mpc_input(
        self,
        obs: Any,
        param_nn: Optional[torch.Tensor] = None,
        action: Optional[torch.Tensor] = None,
    ) -> MpcInput:
        raise NotImplementedError("This task does not support MPC input preparation.")

    def create_extractor(self, env: gym.Env) -> Extractor:
        return IdentityExtractor(env)
