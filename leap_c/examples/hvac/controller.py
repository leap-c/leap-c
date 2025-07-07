from pathlib import Path
from typing import Any

import casadi as ca
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
from itertools import chain
from acados_template import AcadosOcp
from env import StochasticThreeStateRcEnv, decompose_observation
from scipy.constants import convert_temperature
from util import transcribe_discrete_state_space

from leap_c.controller import ParameterizedController
from leap_c.examples.hvac.config import make_default_hvac_params
from leap_c.ocp.acados.parameters import AcadosParamManager, Parameter
from leap_c.ocp.acados.torch import AcadosDiffMpc


class HvacController(ParameterizedController):
    def __init__(
        self,
        params: tuple[Parameter, ...] | None = None,
        N_horizon: int = 96,  # 24 hours in 15 minutes time steps
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

    def forward(self, obs, param, ctx=None) -> tuple[Any, torch.Tensor]:
        # NOTE: obs includes datetime information,
        # which is why we cast elements to dtype np.float64
        x0 = torch.as_tensor(np.array(obs[:, 2:5], dtype=np.float64))
        p_global = torch.as_tensor(param, dtype=torch.float64).unsqueeze(0)

        lb, ub = set_temperature_limits(decompose_observation(obs)[-1])

        p_stagewise = self.param_manager.combine_parameter_values(
            lb_Ti=lb.reshape(1, -1, 1),
            ub_Ti=ub.reshape(1, -1, 1),
        )

        ctx, u0, x, u, value = self.diff_mpc(
            x0,
            p_global=p_global,
            p_stagewise=p_stagewise,
            ctx=ctx,
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
            for key in [
                "Ch",
                "Ci",
                "Ce",
                "Rhi",
                "Rie",
                "Rea",
                "gAw",
            ]
        },
    )

    d = ca.vertcat(
        param_manager.get("Ta"),  # Ambient temperature
        param_manager.get("Phi_s"),  # Solar radiation
    )
    ocp.model.disc_dyn_expr = Ad @ ocp.model.x + Bd @ ocp.model.u + Ed @ d

    # Cost function
    ocp.cost.cost_type = "EXTERNAL"
    ocp.model.cost_expr_ext_cost = param_manager.get("price") * ocp.model.u

    # Constraints
    ocp.constraints.x0 = x0 or np.array(
        [convert_temperature(20.0, "celsius", "kelvin")] * 3
    )

    ocp.model.con_h_expr = ca.vertcat(
        ocp.model.x[0] - param_manager.get("lb_Ti"),
        param_manager.get("ub_Ti") - ocp.model.x[0],
    )
    ocp.constraints.lh = np.array([0.0, 00])
    ocp.constraints.uh = np.array([1e3, 1e3])

    ocp.constraints.idxsh = np.array([0, 1])
    ocp.cost.zl = 1e4 * np.ones((ocp.constraints.idxsh.size,))
    ocp.cost.Zl = 1e4 * np.ones((ocp.constraints.idxsh.size,))
    ocp.cost.zu = 1e4 * np.ones((ocp.constraints.idxsh.size,))
    ocp.cost.Zu = 1e4 * np.ones((ocp.constraints.idxsh.size,))

    ocp.constraints.lbu = np.array([0.0])
    ocp.constraints.ubu = np.array([5000.0])  # [W]
    ocp.constraints.idxbu = np.array([0])

    # Solver options
    ocp.solver_options.tf = N_horizon
    ocp.solver_options.N_horizon = N_horizon
    ocp.solver_options.integrator_type = "DISCRETE"
    ocp.solver_options.nlp_solver_type = "SQP"
    ocp.solver_options.hessian_approx = "EXACT"
    ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"

    return ocp


# def get_comfort_bounds(tuple[pd.datetime]


def set_temperature_limits(
    time: np.ndarray[np.datetime64],
    night_start_hour: int = 21,
    night_end_hour: int = 8,
) -> tuple[np.ndarray[np.float64], np.ndarray[np.float64]]:
    """Set temperature limits based on the time of day."""
    if len(time) == 0:
        raise ValueError("Time array cannot be empty")

    n_points = len(time)

    # Extract hours using numpy datetime operations
    hours = (time.astype("datetime64[h]") - time.astype("datetime64[D]")).astype(int)

    # Vectorized night detection
    night_idx = (hours >= night_start_hour) | (hours < night_end_hour)

    # Initialize and set values
    lb = np.where(
        night_idx,
        convert_temperature(15.0, "celsius", "kelvin"),
        convert_temperature(19.0, "celsius", "kelvin"),
    )
    ub = np.where(
        night_idx,
        convert_temperature(21.0, "celsius", "kelvin"),
        convert_temperature(23.0, "celsius", "kelvin"),
    )

    return lb, ub


if __name__ == "__main__":
    horizon_hours = 24
    N_horizon = horizon_hours * 4  # 4 time steps per hour
    env = StochasticThreeStateRcEnv(
        step_size=900.0,  # 15 minutes in seconds
        horizon_hours=24,
        enable_noise=True,
    )

    obs, _ = env.reset(
        state_0=np.array([convert_temperature(20.0, "celsius", "kelvin")] * 3)
    )

    x0 = torch.as_tensor(decompose_observation(obs)[2:5], dtype=torch.float64)

    param_manager = AcadosParamManager(
        params=make_default_hvac_params(),
        N_horizon=N_horizon,
    )

    Ta_forecast, solar_forecast, price_forecast, time = decompose_observation(obs)[5:]

    # TODO: Move this into the param_manager?
    param = param_manager.p_global_values(0)
    for stage in range(N_horizon + 1):
        param["Ta", stage] = Ta_forecast[:, stage]
        param["Phi_s", stage] = solar_forecast[:, stage]
        param["price", stage] = price_forecast[:, stage]
    param = param.cat.full().flatten()

    controller = HvacController(
        N_horizon=N_horizon,
        diff_mpc_kwargs={
            "export_directory": Path("hvac_mpc_export"),
        },
    )

    # Adjust to match shape (batch_size, ...)
    obs = obs.reshape(1, -1)  # Reshape to match expected input shape

    ctx, u0 = controller.forward(obs=obs, param=param)

    x = ctx.iterate.x.reshape(-1, 3)
    u = ctx.iterate.u.reshape(-1, 1)

    time = time.flatten()

    plt.figure(figsize=(12, 8))
    plt.subplot(3, 1, 1)
    plt.ylabel("Temperature (K)")
    plt.step(
        time,
        convert_temperature(x[:, 0], "kelvin", "celsius"),
        label="Indoor (Ti)",
    )
    # plt.plot(
    #     time,
    #     convert_temperature(x[:, 1], "kelvin", "celsius"),
    #     label="Radiator (Th)",
    # )
    plt.step(
        time,
        convert_temperature(x[:, 2], "kelvin", "celsius"),
        label="Envelope (Te)",
    )
    plt.step(
        time,
        convert_temperature(Ta_forecast.reshape(-1), "kelvin", "celsius"),
        label="Ambient (Ta)",
    )
    plt.grid(visible=True)
    plt.legend()
    plt.subplot(3, 1, 2)
    plt.ylabel("Solar Radiation (W/m²)")
    plt.step(time, solar_forecast.reshape(-1), label="Solar Radiation")
    plt.grid(visible=True)
    plt.legend()
    plt.subplot(3, 1, 3)
    plt.ylabel("Control Input (W)")
    plt.step(time[:-1], u[:, 0], label="Heat Input (u)")
    plt.xlabel("Time (s)")
    plt.grid(visible=True)
    plt.legend()
    plt.tight_layout()
    plt.show()
