from itertools import chain
from typing import Any
from pathlib import Path

import casadi as ca
import gymnasium as gym
import numpy as np
import torch
from acados_template import AcadosOcp
from env import StochasticThreeStateRcEnv
from scipy.constants import convert_temperature
from util import transcribe_discrete_state_space
from env import decompose_observation

from leap_c.controller import ParameterizedController
from leap_c.examples.hvac.config import make_default_hvac_params
from leap_c.ocp.acados.parameters import AcadosParamManager, Parameter
from leap_c.ocp.acados.torch import AcadosDiffMpc


class HvacController(ParameterizedController):
    def __init__(
        self,
        params: tuple[Parameter, ...] | None = None,
        N_horizon: int = 96,  # Using discrete dynamics with 15 minutes time steps,
        diff_mpc_kwargs: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.params = params = params or make_default_hvac_params()
        self.param_manager = AcadosParamManager(
            params=self.params,
            N_horizon=N_horizon,
        )

        self.ocp = export_parametric_ocp(
            param_manager=self.param_manager,
            N_horizon=N_horizon,
        )
        self.diff_mpc = AcadosDiffMpc(self.ocp, **diff_mpc_kwargs)

        #####
        # TODO: Initialization of indicator variables. Should we move this to the AcadosParamManager?
        batch_size = self.diff_mpc.diff_mpc_fun.forward_batch_solver.N_batch_max
        parameter_values = self.param_manager.combine_parameter_values(
            batch_size=batch_size
        )

        for ocp_solver in chain(
            self.diff_mpc.diff_mpc_fun.forward_batch_solver.ocp_solvers,
            self.diff_mpc.diff_mpc_fun.backward_batch_solver.ocp_solvers,
        ):
            for batch in range(batch_size):
                for stage in range(N_horizon + 1):
                    ocp_solver.set(stage, "p", parameter_values[batch, stage, :])
        #####

    def forward(self, obs, param, ctx=None) -> tuple[Any, torch.Tensor]:
        x0 = torch.as_tensor(decompose_observation(obs)[2:5], dtype=torch.float64)

        p_global = torch.as_tensor(param, dtype=torch.float64)
        ctx, u0, x, u, value = self.diff_mpc(
            x0.unsqueeze(0), p_global=p_global.unsqueeze(0), ctx=ctx
        )
        return ctx, u0

    def jacobian_action_param(self, ctx) -> np.ndarray:
        return self.diff_mpc.sensitivity(ctx, field_name="du0_dp_global")

    def param_space(self) -> gym.Space:
        lb, ub = self.param_manager.get_p_global_bounds()
        return gym.spaces.Box(low=lb, high=ub, dtype=np.float64)

    def default_param(self) -> np.ndarray:
        # TODO: Move cat.full().flatten() to AcadosParamManager
        return self.param_manager.p_global_values.cat.full().flatten()


def export_parametric_ocp(
    param_manager: AcadosParamManager,
    N_horizon: int,
    name: str = "hvac",
    x0: np.ndarray | None = None,
) -> AcadosOcp:
    """
    Export the HVAC OCP.

    Args:
        param_manager: The parameter manager containing the parameters for the OCP.
        N_horizon: Number of time steps in the horizon.
        name: Name of the OCP model.
        x0: Initial state. If None, a default value is used.

    Returns:
        AcadosOcp: The configured OCP object.
    """
    ocp = AcadosOcp()

    param_manager.assign_to_ocp(ocp)

    # Model
    ocp.model.name = name

    ocp.model.x = ca.vertcat(
        ca.SX.sym("Ti"),  # Indoor air temperature
        ca.SX.sym("Th"),  # Radiator temperature
        ca.SX.sym("Te"),  # Envelope temperature
    )

    ocp.model.u = ca.SX.sym("qh")  # Heat input to radiator

    Ad, Bd, Ed = transcribe_discrete_state_space(
        Ad=np.zeros((3, 3)),
        Bd=np.zeros((3, 1)),
        Ed=np.zeros((3, 2)),
        dt=900.0,  # 15 minutes in seconds
        params={
            key: param_manager.get(key)
            for key in ["Ch", "Ci", "Ce", "Rhi", "Rie", "Rea", "gAw"]
        },
    )

    ocp.model.disc_dyn_expr = Ad @ ocp.model.x + Bd @ ocp.model.u + Ed @ ocp.model.p[:2]

    # Cost function
    ocp.cost.cost_type = "EXTERNAL"
    ocp.model.cost_expr_ext_cost = param_manager.get("price") * ocp.model.u

    # Constraints
    ocp.constraints.x0 = x0 or np.array(
        [convert_temperature(17.0, "celsius", "kelvin")] * 3
    )

    ocp.constraints.lbu = np.array([0.0])
    ocp.constraints.ubu = np.array([5000.0])  # [W]
    ocp.constraints.idxbu = np.array([0])

    ocp.constraints.lbx = convert_temperature(np.array([17.0]), "celsius", "kelvin")
    ocp.constraints.ubx = convert_temperature(np.array([25.0]), "celsius", "kelvin")
    ocp.constraints.idxbx = np.array([0])

    ocp.constraints.idxsbx = np.array([0])
    ocp.cost.zl = 1e4 * np.ones((ocp.constraints.idxsbx.size,))
    ocp.cost.Zl = 1e4 * np.ones((ocp.constraints.idxsbx.size,))
    ocp.cost.zu = 1e4 * np.ones((ocp.constraints.idxsbx.size,))
    ocp.cost.Zu = 1e4 * np.ones((ocp.constraints.idxsbx.size,))

    # Solver options
    ocp.solver_options.tf = N_horizon
    ocp.solver_options.N_horizon = N_horizon
    ocp.solver_options.integrator_type = "DISCRETE"
    ocp.solver_options.nlp_solver_type = "SQP"
    ocp.solver_options.hessian_approx = "EXACT"
    ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
    # ocp.solver_options.qp_solver_cond_N = 4

    return ocp


if __name__ == "__main__":
    horizon_hours = 24
    N_horizon = horizon_hours * 4  # 4 time steps per hour
    env = StochasticThreeStateRcEnv(
        step_size=900.0,  # 15 minutes in seconds
        horizon_hours=24,
        enable_noise=True,
    )

    obs, _ = env.reset()

    x0 = torch.as_tensor(decompose_observation(obs)[2:5], dtype=torch.float64)

    param_manager = AcadosParamManager(
        params=make_default_hvac_params(),
        N_horizon=N_horizon,
    )

    Ta_forecast, solar_forecast, price_forecast = decompose_observation(obs)[5:8]

    # TODO: Move this into the param_manager?
    param = param_manager.p_global_values(0)
    for stage in range(N_horizon + 1):
        param["Ta", stage] = Ta_forecast[stage]
        param["Phi_s", stage] = solar_forecast[stage]
        param["price", stage] = price_forecast[stage]
    param = param.cat.full().flatten()

    controller = HvacController(
        N_horizon=N_horizon,
        diff_mpc_kwargs={
            "export_directory": Path("hvac_mpc_export"),
        },
    )

    ctx, u0 = controller.forward(obs=obs, param=param)

    print("ctx", ctx)
    print("u0", u0)
