from typing import Any

import numpy as np
from leap_c.task import Task
from leap_c.util import tensor_to_numpy
from numpy import ndarray

from ...mpc import MPCInput


# TODO: Think of a better name
class SwingUpShortHorizonFOU(Task):
    """Swing-up task for the pendulum on a cart system.
    The task is to swing up the pendulum from a downward position to the upright position
    (and balance it there)."""

    def prepare_nn_input(self, obs: Any) -> ndarray:
        # TODO: Where is the tensor conversion currently happening while exploring
        # (while training it is happening in the buffer already. Should it even, or do we want to
        # do it here now?)?
        return obs

    def prepare_mpc_input(self, obs: Any, param_nn: ndarray | None = None) -> MPCInput:
        x0 = tensor_to_numpy(obs).astype(np.float64)
        # TODO: We probably should add an action to this function in the interface,
        # in case someone wants to use MPC Q functions? Also to prepare_nn_input.
        # (Or do we want to ignore MPC Q functions for the moment?)

        # TODO: I think there needs to be clarification how precisely a Task definition should be.
        # E.g., in prepare_mpc_input I have to define the MPCInput, in particular how the parameters
        # look like. But I don't even have the information about what p_global actually means.
        # Do I have to scale them? To what bounds? Which ones do I need to put into
        # p_global, p_stagewise, p_W, p_yref (If any)?
        # => I think either the task should define the mpc itself, or we introduce something that
        # keeps track of what the parameters mean, what params are global, etc., in the MPC class (MPC).
        # I prefer the former (I think the latter would be a param manager? :D)


#        return MPCInput(x0=x0, parameters=param_nn)
