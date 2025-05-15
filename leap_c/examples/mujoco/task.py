from typing import Any, Optional

import gymnasium as gym
import torch
import numpy as np

from leap_c.mpc import MpcInput
from leap_c.task import Task
from leap_c.registry import register_task


@register_task("half_cheetah")
class HalfCheetahTask(Task):
    def __init__(self):
        super().__init__(None)

    def create_env(self, train: bool = True) -> gym.Env:
        env = gym.make("HalfCheetah-v4") if train else gym.make("HalfCheetah-v4", render_mode="rgb_array")
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.ClipAction(env)
        env = gym.wrappers.NormalizeObservation(env)
        env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10), None)
        env = gym.wrappers.NormalizeReward(env, gamma=0.99)
        env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
        return env

    def prepare_mpc_input(
        self,
        obs: Any,
        param_nn: Optional[torch.Tensor] = None,
        action: Optional[torch.Tensor] = None,
    ) -> MpcInput:
        return MpcInput(obs, param_nn, action)
