from leap_c.task import Task
import numpy as np
from typing import Any, Optional
from leap_c.examples.pointmass.mpc import PointMassMPC as MPC
from leap_c.examples.pointmass.env import PointMassEnv as Env
from leap_c.mpc import MPCInput, MPCSingleState


class PointMassTask(Task):
    def __init__(self, mpc: MPC, env: Env):
        super().__init__(mpc, env)

    def prepare_nn_input(self, obs: Any) -> np.ndarray:
        return np.array(obs)

    def prepare_mpc_input(
        self,
        obs: Any,
        param_nn: Optional[np.ndarray] = None,
    ) -> MPCInput:
        state = MPCSingleState(x=obs)
        return MPCInput(state=state, param_nn=param_nn)
