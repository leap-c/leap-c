from typing import Callable, Optional

import gymnasium as gym

from leap_c.controller import ParameterizedController
from leap_c.examples.cartpole.controller import CartPoleController
from leap_c.examples.cartpole.env import (
    CartPoleBalanceEnv,
    CartPoleEnv,
)
from leap_c.registry import register_task
from leap_c.task import HierarchicalControllerMixin, ReinforcementLearningMixin, Task


@register_task("cartpole")
class CartPoleSwingup(Task, ReinforcementLearningMixin, HierarchicalControllerMixin):
    """Swing-up task for the pendulum on a cart system.
    The task is to swing up the pendulum from a downward position to the upright position
    (and balance it there)."""

    def __init__(self):
        super().__init__()

    def _create_env(self, train: bool) -> gym.Env:
        return CartPoleEnv(render_mode="rgb_array" if train else None)

    def create_parameterized_controller(
        self, collate_state_fn: Optional[Callable] = None
    ) -> ParameterizedController:
        return CartPoleController(collate_state_fn)


@register_task("cartpole_balance")
class CartPoleBalance(CartPoleSwingup):
    """The same as PendulumOnCartSwingup, but the starting position of the pendulum is upright, making the task a balancing task."""

    def _create_env(self, train: bool) -> gym.Env:
        return CartPoleBalanceEnv(render_mode="rgb_array" if train else None)
