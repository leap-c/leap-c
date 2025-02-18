from typing import Any, Optional

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
from functools import cached_property

from leap_c.examples.pointmass.env import (
    PointMassEnv,
    WindField,
    WindTunnel,
    WindTunnelParam,
    VortexParam,
    VortexWind,
    InverseVortexWind,
    RandomWind,
    RandomWindParam,
    VariationWind,
    VariationWindParam,
    BaseWind,
    BaseWindParam,
)
from leap_c.examples.pointmass.mpc import PointMassMPC
from leap_c.mpc import MPCInput, MPCParameter
from leap_c.nn.modules import MPCSolutionModule
from leap_c.registry import register_task
from leap_c.task import Task


@register_task("point_mass")
class PointMassTask(Task):
    def __init__(self):
        mpc = PointMassMPC(
            learnable_params=[
                # "m",
                # "cx",
                # "cy",
                "q_diag",
                # "r_diag",
                # "q_diag_e",
                "xref",
                "uref",
                # "xref_e",
                # "u_wind",
            ]
        )
        mpc_layer = MPCSolutionModule(mpc)

        super().__init__(mpc_layer, PointMassEnv)

        self.param_low = 0.5 * mpc.ocp.p_global_values
        self.param_high = 1.5 * mpc.ocp.p_global_values

        # TODO: Handle params that are nominally zero
        for i, p in enumerate(mpc.ocp.p_global_values):
            if p == 0:
                self.param_low[i] = -10.0
                self.param_high[i] = 10.0

    @property
    def param_space(self) -> spaces.Box:
        # low = np.array([0.5, 0.0])
        # high = np.array([2.5, 0.5])
        low = self.param_low
        high = self.param_high
        return spaces.Box(low=low, high=high, dtype=np.float32)

    @cached_property
    def train_env(self) -> gym.Env:
        env = PointMassEnv(
            max_time=10.0,
            init_state_dist={
                "low": np.array([1.0, 0.0, 0.0, 0.0]),
                "high": np.array([5.0, 5.0, 0.0, 0.0]),
            },
        )
        env.reset(seed=self.seed)
        return env

    @cached_property
    def eval_env(self) -> gym.Env:
        env = PointMassEnv(
            max_time=10.0,
            init_state_dist={
                "low": np.array([5.0, 3.0, 0.0, 0.0]),
                "high": np.array([5.0, 5.0, 0.0, 0.0]),
            },
        )
        env.reset(seed=self.seed)
        return env

    def prepare_mpc_input(
        self,
        obs: Any,
        param_nn: Optional[torch.Tensor] = None,
    ) -> MPCInput:
        mpc_param = MPCParameter(p_global=param_nn)  # type: ignore

        return MPCInput(x0=obs, parameters=mpc_param)


@register_task("point_mass_homo_center")
class PointMassTaskHomoCenter(PointMassTask):
    def __init__(self):
        super().__init__()

    @cached_property
    def train_env(self) -> gym.Env:
        env = PointMassEnv(
            init_state_dist={
                "low": np.array([1.0, -5.0, 0.0, 0.0]),
                "high": np.array([5.0, 5.0, 0.0, 0.0]),
            },
            observation_space=spaces.Box(
                low=np.array([0.0, -5.0, -50.0, -50.0]),
                high=np.array([8.0, +5.0, 50.0, 50.0]),
                dtype=np.float64,
            ),
            max_time=10.0,
            Fmax=10.0,
        )
        env.reset(seed=self.seed)
        return env

    @cached_property
    def eval_env(self) -> gym.Env:
        env = PointMassEnv(
            init_state_dist={
                "low": np.array([5.0, -1.0, 0.0, 0.0]),
                "high": np.array([5.0, 1.0, 0.0, 0.0]),
            },
            observation_space=spaces.Box(
                low=np.array([0.0, -5.0, -50.0, -50.0]),
                high=np.array([8.0, +5.0, 50.0, 50.0]),
                dtype=np.float64,
            ),
            max_time=10.0,
            Fmax=10.0,
        )
        env.reset(seed=self.seed)
        return env


@register_task("point_mass_vortex")
class PointMassTaskVortex(PointMassTask):
    def __init__(self):
        super().__init__()

    @cached_property
    def train_env(self) -> gym.Env:
        env = PointMassEnv(
            init_state_dist={
                "low": np.array([1.0, -5.0, 0.0, 0.0]),
                "high": np.array([5.0, 5.0, 0.0, 0.0]),
            },
            wind_field=WindField(
                [
                    # BaseWind(param=BaseWindParam(magnitude=(-1.0, 1.0))),
                    RandomWind(param=RandomWindParam()),
                    VariationWind(param=VariationWindParam()),
                    VortexWind(param=VortexParam(center=(2.5, 0.0))),
                    # WindTunnel(
                    #     param=WindTunnelParam(
                    #         center=(0, 0), magnitude=(0, 3.0), decay=(0.0, 0.1)
                    #     )
                    # ),
                ]
            ),
        )
        env.reset(seed=self.seed)
        return env

    @cached_property
    def eval_env(self) -> gym.Env:
        env = PointMassEnv(
            init_state_dist={
                "low": np.array([5.0, -1.0, 0.0, 0.0]),
                "high": np.array([5.0, 1.0, 0.0, 0.0]),
            },
            wind_field=WindField(
                [
                    # BaseWind(param=BaseWindParam(magnitude=(-1.0, 1.0))),
                    RandomWind(param=RandomWindParam()),
                    VariationWind(param=VariationWindParam()),
                    VortexWind(param=VortexParam(center=(2.5, 0.0))),
                    # WindTunnel(
                    #     param=WindTunnelParam(
                    #         center=(0, 0), magnitude=(0, 3.0), decay=(0.0, 0.1)
                    #     )
                    # ),
                ]
            ),
        )
        env.reset(seed=self.seed)
        return env


@register_task("point_mass_inverse_vortex")
class PointMassTaskInverseVortex(PointMassTask):
    def __init__(self):
        super().__init__()

    @cached_property
    def train_env(self) -> gym.Env:
        env = PointMassEnv(
            init_state_dist={
                "low": np.array([1.0, -5.0, 0.0, 0.0]),
                "high": np.array([5.0, 5.0, 0.0, 0.0]),
            },
            wind_field=WindField(
                [
                    # BaseWind(param=BaseWindParam(magnitude=(-1.0, 1.0))),
                    RandomWind(param=RandomWindParam()),
                    VariationWind(param=VariationWindParam()),
                    InverseVortexWind(param=VortexParam(center=(2.5, 0.0))),
                    # WindTunnel(
                    #     param=WindTunnelParam(
                    #         center=(0, 0), magnitude=(0, 3.0), decay=(0.0, 0.1)
                    #     )
                    # ),
                ]
            ),
        )
        env.reset(seed=self.seed)
        return env

    @cached_property
    def eval_env(self) -> gym.Env:
        env = PointMassEnv(
            init_state_dist={
                "low": np.array([5.0, -1.0, 0.0, 0.0]),
                "high": np.array([5.0, 1.0, 0.0, 0.0]),
            },
            wind_field=WindField(
                [
                    # BaseWind(param=BaseWindParam(magnitude=(-1.0, 1.0))),
                    RandomWind(param=RandomWindParam()),
                    VariationWind(param=VariationWindParam()),
                    VortexWind(param=VortexParam(center=(2.5, 0.0))),
                    # WindTunnel(
                    #     param=WindTunnelParam(
                    #         center=(0, 0), magnitude=(0, 3.0), decay=(0.0, 0.1)
                    #     )
                    # ),
                ]
            ),
        )
        env.reset(seed=self.seed)
        return env
