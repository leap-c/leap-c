from abc import ABC, abstractmethod
from copy import deepcopy
from functools import partial
from pathlib import Path

import casadi as ca
import numpy as np
from acados_template import (
    AcadosOcp,
    AcadosSim,
    AcadosSimOptions,
    AcadosSimSolver,
)

from leap_c.acados.mpc import Mpc, MpcParameter
from leap_c.acados.file_manager import AcadosFileManager


def create_dynamics_from_mpc(
    mpc: Mpc, export_dir: None | Path = None, sens_forw=False, dt=None
) -> "Dynamics":
    """Create a dynamics object from an MPC object."""
    ocp: AcadosOcp = mpc.ocp

    # if there is a discrete dynamics function, use it
    if ocp.model.disc_dyn_expr is not None:
        return CasadiDynamics(mpc)

    # otherwise we create a AcadosSim object
    assert ocp.model.f_expl_expr is not None

    # create sim opts
    sim_opts = AcadosSimOptions()

    # assert sim_method_num_stages all equal
    # does this makes sense?
    sim_opts.num_steps = ocp.solver_options.sim_method_num_steps
    sim_opts.newton_iter = ocp.solver_options.sim_method_newton_iter
    sim_opts.integrator_type = ocp.solver_options.integrator_type

    # create sim
    sim = AcadosSim()
    sim.solver_options = sim_opts
    sim.solver_options.sens_forw = sens_forw
    sim.model = deepcopy(ocp.model)
    sim.parameter_values = ocp.parameter_values

    if dt is None:
        dt = ocp.solver_options.tf / ocp.solver_options.N_horizon  # type: ignore

    sim.solver_options.T = dt

    return SimDynamics(sim, export_dir=export_dir)


class Dynamics(ABC):
    @abstractmethod
    def __call__(
        self,
        x: np.ndarray,
        u: np.ndarray,
        p: np.ndarray | None,
        with_sens: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Step the dynamics.

        Args:
            x: The current state.
            u: The current input.
            p: The current parameter.
            with_sens: Whether to return the sensitivity.

        Returns:
            The next state.
        """
        ...


class SimDynamics(Dynamics):
    def __init__(
        self, sim: AcadosSim, export_dir: Path | None = None, cleanup: bool = True
    ):
        self.sim = sim
        afm = AcadosFileManager(export_dir, cleanup=cleanup)
        self._sim_solver_fn = partial(afm.setup_acados_sim_solver, sim)
        self._sim_solver = None

    @property
    def sim_solver(self) -> AcadosSimSolver:
        if self._sim_solver is None:
            self._sim_solver = self._sim_solver_fn()
        return self._sim_solver

    def __call__(
        self,
        x: np.ndarray,
        u: np.ndarray,
        p: np.ndarray | None = None,
        with_sens: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Step the dynamics.

        Args:
            x: The current state.
            u: The current input.
            p: The current parameter.
            with_sens: Whether to return the sensitivity.

        Returns:
            The next state and the sensitivity if with_sens is True.
        """
        self.sim_solver.set("x", x)
        self.sim_solver.set("u", u)

        if p is not None:
            self.sim_solver.set("p", p)

        self.sim_solver.solve()

        if not with_sens:
            return self.sim_solver.get("x")  # type: ignore

        Sx = self.sim_solver.get("Sx")
        Su = self.sim_solver.get("Su")

        return self.sim_solver.get("x"), Sx, Su  # type: ignore


class CasadiDynamics(Dynamics):
    def __init__(self, mpc: Mpc):
        self.mpc = mpc

        ocp = mpc.ocp
        inputs = [ocp.model.x, ocp.model.u]
        names = ["x", "u"]

        if self.mpc.default_p_global is not None:
            inputs.append(ocp.model.p_global)
            names.append("p_global")

        if self.mpc.default_p_stagewise is not None:
            inputs.append(ocp.model.p)
            names.append("p")

        expr = ocp.model.disc_dyn_expr

        self.dyn_fn = ca.Function("dyn", inputs, [expr], names, ["x_next"])

        # generate the jacobian
        inputs_cat = ca.vertcat(*inputs)
        jac_expr = ca.jacobian(expr, inputs_cat)

        self.jac_fn = ca.Function("jtimes", [inputs_cat], [jac_expr])

    def __call__(
        self,
        x: np.ndarray,
        u: np.ndarray,
        p: MpcParameter | None = None,
        with_sens: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Step the dynamics.

        If the parameter is not provided, the current parameter of the MPC object is used.

        Args:
            x: The current state.
            u: The current input.
            p: The current parameter.
            with_sens: Whether to return the sensitivity.

        Returns:
            The next state.
        """
        x_shape = x.shape
        batched = True if len(x_shape) > 1 else False

        inputs = [x, u]

        p_global, p_stage = self.mpc.fetch_param(p, 0)

        if p_global is not None:
            if batched:
                p_global = np.tile(p_global, (x_shape[0], 1))
            inputs.append(p_global)

        if p_stage is not None:
            if batched:
                p_stage = np.tile(p_stage, (x_shape[0], 1))
            inputs.append(p_stage)

        # transpose inputs to match casadi batch format
        inputs = [i.T for i in inputs]

        output = self.dyn_fn(*inputs).full().T.reshape(x_shape)  # type: ignore

        if not with_sens:
            return output

        # compute the jacobian
        sizes = [i.shape[0] for i in inputs]
        splits = np.cumsum(sizes)[:-1]

        inputs_cat = np.concatenate(inputs, axis=0)

        jac = self.jac_fn(inputs_cat).full().T  # type: ignore
        if len(x_shape) > 1:
            jac = jac.reshape(x_shape[0], -1, x_shape[1])
            Sx, Su = np.split(jac, splits, axis=1)[:2]
        else:
            Sx, Su = np.split(jac, splits, axis=0)[:2]

        return output, Sx, Su
