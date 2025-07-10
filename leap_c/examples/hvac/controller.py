from pathlib import Path
from typing import Any
from itertools import chain

import casadi as ca
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
from acados_template import ACADOS_INFTY
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

        self.N_horizon = N_horizon

        self.qh = 0.0
        self.dqh = 0.0

    def forward(self, obs, param: Any = None, ctx=None) -> tuple[Any, torch.Tensor]:
        # NOTE: obs includes datetime information,
        # which is why we cast elements to dtype np.float64
        x0 = torch.as_tensor(np.array(obs[:, 2:5], dtype=np.float64))

        # Append  [self.qh, self.dqh] to x0
        x0 = torch.cat(
            [
                x0,
                torch.as_tensor([[self.qh, self.dqh]], dtype=torch.float64),
            ],
            dim=1,
        )

        if param is None:
            # Use default parameters if none are provided
            # NOTE: The SAC controller would modify the forecasted parameters
            param = self.param_manager.p_global_values(0)
            Ta_forecast, solar_forecast, price_forecast = decompose_observation(obs)[5:]
            for stage in range(self.N_horizon + 1):
                param["Ta", stage] = Ta_forecast[:, stage]
                param["Phi_s", stage] = solar_forecast[:, stage]
                param["price", stage] = price_forecast[:, stage]
                param["q_dqh", stage] = 1.0  # weight on rate of change of heater power
                param["q_ddqh", stage] = 1.0  # weight on acceleration of heater power
            param=param.cat.full().flatten()


        p_global = torch.as_tensor(param, dtype=torch.float64).unsqueeze(0)

        batch_size = x0.shape[0]

        quarter_hours = np.array(
            [
                np.arange(obs[i, 0], obs[i, 0] + N_horizon + 1) % N_horizon
                for i in range(batch_size)
            ]
        )

        lb, ub = set_temperature_limits(quarter_hours=quarter_hours)

        p_stagewise = self.param_manager.combine_parameter_values(
            lb_Ti=lb.reshape(batch_size, -1, 1),
            ub_Ti=ub.reshape(batch_size, -1, 1),
        )

        ctx, u0, x, u, value = self.diff_mpc(
            x0,
            p_global=p_global,
            p_stagewise=p_stagewise,
            ctx=ctx,
        )


        self.qh = x[:, 2*self.ocp.dims.nx-2]
        self.dqh = x[:, 2*self.ocp.dims.nx-1]

        action = np.array(self.qh.detach().numpy(), dtype=np.float32).reshape(-1)

        return ctx, action

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

    dt: float = 900.0  # Time step in seconds (15 minutes)

    ocp = AcadosOcp()

    param_manager.assign_to_ocp(ocp)

    # Model
    ocp.model.name = name

    ocp.model.x = ca.vertcat(
        ca.SX.sym("Ti"),  # Indoor air temperature
        ca.SX.sym("Th"),  # Radiator temperature
        ca.SX.sym("Te"),  # Envelope temperature
    )

    qh = ca.SX.sym("qh")  # Heat input to radiator
    dqh = ca.SX.sym("dqh")  # Increment Heat input to radiator
    ddqh = ca.SX.sym("ddqh")  # Increment Heat input to radiator

    Ad, Bd, Ed = transcribe_discrete_state_space(
        Ad=np.zeros((3, 3)),
        Bd=np.zeros((3, 1)),
        Ed=np.zeros((3, 2)),
        dt=dt,
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
    ocp.model.disc_dyn_expr = Ad @ ocp.model.x + Bd @ qh + Ed @ d

    # Augment the model with double integrator for the control input
    ocp.model.x = ca.vertcat(ocp.model.x, qh, dqh)
    ocp.model.disc_dyn_expr = ca.vertcat(
        ocp.model.disc_dyn_expr,
        qh + dt * dqh + 0.5*dt**2*ddqh,
        dqh + dt * ddqh,

    )
    ocp.model.u = ddqh

    # Cost function
    ocp.cost.cost_type = "EXTERNAL"
    ocp.model.cost_expr_ext_cost = (
        0.25 * param_manager.get("price") * qh
        + param_manager.get("q_dqh") * (dqh) ** 2
        + param_manager.get("q_ddqh") * (ddqh) ** 2
    )

    ocp.cost.cost_type_e = "EXTERNAL"
    ocp.model.cost_expr_ext_cost_e = (
        0.25 * param_manager.get("price") * qh
        + param_manager.get("q_dqh") * (dqh) ** 2
    )


    # Constraints
    ocp.constraints.x0 = x0 or np.array(
        [convert_temperature(20.0, "celsius", "kelvin")] * 3 + [0.0, 0.0]
    )

    # Comfort constraints
    ocp.model.con_h_expr = ca.vertcat(
        ocp.model.x[0] - param_manager.get("lb_Ti"),
        param_manager.get("ub_Ti") - ocp.model.x[0],
    )
    ocp.constraints.lh = np.array([0.0, 00])
    ocp.constraints.uh = np.array([ACADOS_INFTY, ACADOS_INFTY])

    ocp.constraints.idxsh = np.array([0, 1])
    ocp.cost.zl = 1e2 * np.ones((ocp.constraints.idxsh.size,))
    ocp.cost.Zl = 1e2 * np.ones((ocp.constraints.idxsh.size,))
    ocp.cost.zu = 1e2 * np.ones((ocp.constraints.idxsh.size,))
    ocp.cost.Zu = 1e2 * np.ones((ocp.constraints.idxsh.size,))

    ocp.constraints.lbx = np.array([0.0])
    ocp.constraints.ubx = np.array([5000.0])
    ocp.constraints.idxbx = np.array([3])  # qh

    # Solver options
    ocp.solver_options.tf = N_horizon
    ocp.solver_options.N_horizon = N_horizon
    ocp.solver_options.integrator_type = "DISCRETE"
    ocp.solver_options.nlp_solver_type = "SQP"
    ocp.solver_options.hessian_approx = "EXACT"
    ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"

    return ocp

def set_temperature_limits(
    quarter_hours: np.ndarray,
    night_start_hour: int = 22,
    night_end_hour: int = 8,
) -> tuple[np.ndarray[np.float64], np.ndarray[np.float64]]:
    """Set temperature limits based on the time of day."""
    hours = np.floor(quarter_hours / 4)

    # Vectorized night detection
    night_idx = (hours >= night_start_hour) | (hours < night_end_hour)

    # Initialize and set values
    lb = np.where(
        night_idx,
        convert_temperature(12.0, "celsius", "kelvin"),
        convert_temperature(19.0, "celsius", "kelvin"),
    )
    ub = np.where(
        night_idx,
        convert_temperature(25.0, "celsius", "kelvin"),
        convert_temperature(22.0, "celsius", "kelvin"),
    )
    return lb, ub


def plot_ocp_results(
    time: np.ndarray[np.datetime64],
    obs: np.ndarray,
    ctx: Any,
    figsize: tuple[float, float] = (12, 10),
    save_path: str | None = None,
) -> plt.Figure:
    """
    Plot the OCP solution results in a figure with three vertically stacked subplots.

    Args:
        obs: Observation data
        ctx: Context containing the OCP iterate
        dt: Time step in seconds
        figsize: Figure size (width, height)
        save_path: Optional path to save the figure

    Returns:
        matplotlib Figure object
    """
    x = ctx.iterate.x.reshape(-1, 5)
    # u = ctx.iterate.u.reshape(-1, 1)
    u = x[:, 4]
    # Create time vectors

    # Create figure with subplots
    fig, axes = plt.subplots(4, 1, figsize=figsize, sharex=True)
    fig.suptitle(
        "Thermal Building Control - OCP Solution", fontsize=16, fontweight="bold"
    )

    # Subplot 1: Thermal States

    # Convert temperatures to Celsius for plotting
    Ti_celsius = convert_temperature(x[:, 0], "kelvin", "celsius")
    Th_celsius = convert_temperature(x[:, 1], "kelvin", "celsius")
    Te_celsius = convert_temperature(x[:, 2], "kelvin", "celsius")

    Ta_forecast, solar_forecast, price_forecast = decompose_observation(obs=obs)[5:]
    solar_forecast = solar_forecast.reshape(-1)
    price_forecast = price_forecast.reshape(-1)
    time = time.reshape(-1)

    quarter_hours = np.arange(obs[0], obs[0] + len(time)) % len(time)
    T_lower, T_upper = set_temperature_limits(quarter_hours=quarter_hours)

    T_lower_celsius = convert_temperature(T_lower.reshape(-1), "kelvin", "celsius")
    T_upper_celsius = convert_temperature(T_upper.reshape(-1), "kelvin", "celsius")
    Ta_celsius = convert_temperature(Ta_forecast.reshape(-1), "kelvin", "celsius")

    ax0 = axes[0]
    ax0.fill_between(
        time,
        T_lower_celsius,
        T_upper_celsius,
        alpha=0.2,
        color="lightgreen",
        label="Comfort zone",
    )

    # Plot comfort bounds as dashed lines
    ax0.step(time, T_lower_celsius, "g--", alpha=0.7, label="Lower bound")
    ax0.step(time, T_upper_celsius, "g--", alpha=0.7, label="Upper bound")

    # Plot state trajectories
    ax0.step(time, Ti_celsius, "b-", linewidth=2, label="Indoor temp. (Ti)")
    ax0.step(
        time,
        Te_celsius,
        "orange",
        linewidth=2,
        label="Envelope temp. (Te)",
    )

    ax0.set_ylabel("Temperature [°C]", fontsize=12)
    ax0.legend(loc="best")
    ax0.grid(visible=True, alpha=0.3)
    ax0.set_title("Indoor/Envelope Temperature", fontsize=14, fontweight="bold")

    # Subplot 1: Heater Temperature
    ax1 = axes[1]
    ax1.step(time, Th_celsius, "b-", linewidth=2, label="Radiator temp. (Th)")
    ax1.set_ylabel("Temperature [°C]", fontsize=12)
    ax1.grid(visible=True, alpha=0.3)
    ax1.set_title("Heater Temperature", fontsize=14, fontweight="bold")

    # Subplot 2: Disturbance Signals (twin axes)
    ax2 = axes[2]

    # Outdoor temperature (left y-axis)
    ax2.step(
        time,
        Ta_celsius,
        "b-",
        where="post",
        linewidth=2,
        label="Outdoor temp.",
    )
    ax2.set_ylabel("Outdoor Temperature [°C]", color="b", fontsize=12)
    ax2.tick_params(axis="y", labelcolor="b")

    # Solar radiation (right y-axis)
    ax2_twin = ax2.twinx()
    ax2_twin.step(
        time,
        solar_forecast,
        color="orange",
        where="post",
        linewidth=2,
        label="Solar radiation",
    )
    ax2_twin.set_ylabel("Solar Radiation [W/m²]", color="orange", fontsize=12)
    ax2_twin.tick_params(axis="y", labelcolor="orange")

    ax2.grid(visible=True, alpha=0.3)
    ax2.set_title("Exogeneous Signals", fontsize=14, fontweight="bold")

    # Subplot 3: Control Input
    ax3 = axes[3]

    # Plot control as step function
    ax3.step(
        time,
        u,
        "b-",
        where="post",
        linewidth=2,
        label="Heat input",
    )

    ax3.set_xlabel("Time [hours]", fontsize=12)
    ax3.set_ylabel("Heat Input [W]", color="b", fontsize=12)
    ax3.grid(visible=True, alpha=0.3)
    ax3.set_title("Control Input", fontsize=14, fontweight="bold")

    # Set y-axis lower limit to 0 for better visualization
    ax3.set_ylim(bottom=0)

    ax3_twin = ax3.twinx()
    # Add energy cost as a secondary y-axis
    ax3_twin.step(
        time,
        price_forecast,
        color="orange",
        where="post",
        linewidth=2,
        label="Energy cost (scaled)",
    )
    ax3_twin.set_ylabel("Energy Price [EUR/kWh]", color="orange", fontsize=12)
    ax3_twin.tick_params(axis="y", labelcolor="orange")
    ax3_twin.grid(visible=False)  # Disable grid for twin axis
    ax3_twin.set_ylim(bottom=0)  # Set lower limit to 0 for energy cost

    # Adjust layout
    plt.tight_layout()

    # Add summary statistics as text
    dt = 900.0  # Time step in seconds (15 minutes)
    total_energy_kWh = u.sum() * dt / 3600 / 1000  # Convert to kWh
    max_comfort_violation = max(
        ctx.iterate.sl.reshape(-1, 2).max(),
        ctx.iterate.su.reshape(-1, 2).max(),
    )

    stats_text = (
        f"Total Energy: {total_energy_kWh:.1f} kWh | "
        f"Max Comfort Violation: {max_comfort_violation:.2f} K"
    )

    fig.text(
        0.78,
        0.02,
        stats_text,
        ha="center",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "lightgray", "alpha": 0.8},
    )

    # Save figure if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Figure saved to: {save_path}")

    return fig


def plot_simulation(time: np.ndarray, obs: np.ndarray, action: np.ndarray) -> plt.Figure:

    quarter_hours, day, Ti, Th, Te, Ta_forecast, solar_forecast, price_forecast = decompose_observation(obs=obs)

    # Convert temperatures to Celsius for plotting
    Ti_celsius = convert_temperature(Ti, "kelvin", "celsius")
    Th_celsius = convert_temperature(Th, "kelvin", "celsius")
    Te_celsius = convert_temperature(Te, "kelvin", "celsius")
    Ta_celsius = convert_temperature(Ta_forecast[:, 0], "kelvin", "celsius")

    Ti_lower, Ti_upper = set_temperature_limits(quarter_hours)
    Ti_lower_celsius = convert_temperature(Ti_lower, "kelvin", "celsius")
    Ti_upper_celsius = convert_temperature(Ti_upper, "kelvin", "celsius")

    qh = action
    solar = solar_forecast[:, 0]
    price = price_forecast[:, 0]

    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    fig.suptitle(
        "Thermal Building Control - OCP Solution",
        fontsize=16,
        fontweight="bold",
    )

    ax0 = axes[0]
    ax0.fill_between(
        time,
        Ti_lower_celsius,
        Ti_upper_celsius,
        alpha=0.2,
        color="lightgreen",
        label="Comfort zone",
    )

    # Plot comfort bounds as dashed lines
    ax0.step(time, Ti_lower_celsius, "g--", alpha=0.7, label="Lower bound")
    ax0.step(time, Ti_upper_celsius, "g--", alpha=0.7, label="Upper bound")

    # Plot state trajectories
    ax0.step(time, Ti_celsius, "b-", linewidth=2, label="Indoor temp. (Ti)")
    ax0.step(
        time,
        Te_celsius,
        "orange",
        linewidth=2,
        label="Envelope temp. (Te)",
    )

    ax0.set_ylabel("Temperature [°C]", fontsize=12)
    ax0.legend(loc="best")
    ax0.grid(visible=True, alpha=0.3)
    ax0.set_title("Indoor/Envelope Temperature", fontsize=14, fontweight="bold")

    # Subplot 1: Heater Temperature
    ax1 = axes[1]
    ax1.step(time, Th_celsius, "b-", linewidth=2, label="Radiator temp. (Th)")
    ax1.set_ylabel("Temperature [°C]", fontsize=12)
    ax1.grid(visible=True, alpha=0.3)
    ax1.set_title("Heater Temperature", fontsize=14, fontweight="bold")

    # Subplot 2: Disturbance Signals (twin axes)
    ax2 = axes[2]

    # Outdoor temperature (left y-axis)
    ax2.step(
        time,
        Ta_celsius,
        "b-",
        where="post",
        linewidth=2,
        label="Outdoor temp.",
    )
    ax2.set_ylabel("Outdoor Temperature [°C]", color="b", fontsize=12)
    ax2.tick_params(axis="y", labelcolor="b")

    # Solar radiation (right y-axis)
    ax2_twin = ax2.twinx()
    ax2_twin.step(
        time,
        solar,
        color="orange",
        where="post",
        linewidth=2,
        label="Solar radiation",
    )
    ax2_twin.set_ylabel("Solar Radiation [W/m²]", color="orange", fontsize=12)
    ax2_twin.tick_params(axis="y", labelcolor="orange")

    ax2.grid(visible=True, alpha=0.3)
    ax2.set_title("Exogeneous Signals", fontsize=14, fontweight="bold")

    # Subplot 3: Control Input
    ax3 = axes[3]

    # Plot control as step function
    ax3.step(
        time,
        qh,
        "b-",
        where="post",
        linewidth=2,
        label="Heat input",
    )

    ax3.set_xlabel("Time [hours]", fontsize=12)
    ax3.set_ylabel("Heat Input [W]", color="b", fontsize=12)
    ax3.grid(visible=True, alpha=0.3)
    ax3.set_title("Control Input", fontsize=14, fontweight="bold")

    # Set y-axis lower limit to 0 for better visualization
    ax3.set_ylim(bottom=0)

    ax3_twin = ax3.twinx()
    # Add energy cost as a secondary y-axis
    ax3_twin.step(
        time,
        price,
        color="orange",
        where="post",
        linewidth=2,
        label="Energy cost (scaled)",
    )
    ax3_twin.set_ylabel("Energy Price [EUR/kWh]", color="orange", fontsize=12)
    ax3_twin.tick_params(axis="y", labelcolor="orange")
    ax3_twin.grid(visible=False)  # Disable grid for twin axis
    ax3_twin.set_ylim(bottom=0)  # Set lower limit to 0 for energy cost

    # Adjust layout
    plt.tight_layout()

    # Add summary statistics as text
    dt = 900.0  # Time step in seconds (15 minutes)
    total_energy_kWh = qh.sum() * dt / 3600 / 1000  # Convert to kWh
    # max_comfort_violation = max(
    #     ctx.iterate.sl.reshape(-1, 2).max(),
    #     ctx.iterate.su.reshape(-1, 2).max(),
    # )

    stats_text = (
        f"Total Energy: {total_energy_kWh:.1f} kWh | "
        # f"Max Comfort Violation: {max_comfort_violation:.2f} K"
    )

    fig.text(
        0.78,
        0.02,
        stats_text,
        ha="center",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "lightgray", "alpha": 0.8},
    )

    return fig

if __name__ == "__main__":
    horizon_hours = 24
    N_horizon = horizon_hours * 4  # 4 time steps per hour
    env = StochasticThreeStateRcEnv(
        step_size=900.0,  # 15 minutes in seconds
        horizon_hours=horizon_hours,
    )

    controller = HvacController(
        N_horizon=N_horizon,
        diff_mpc_kwargs={
            "export_directory": Path("hvac_mpc_export"),
        },
    )

    n_steps = 1 * 24 * 4  # days * hours * 4 time steps per hour

    obs, info = env.reset()

    obs = np.tile(obs, (n_steps + 1, 1))
    time = np.empty(n_steps)
    action = np.zeros((n_steps, 1), dtype=np.float32)

    for k in range(n_steps):
        time[k] = info["time_forecast"][0]
        _, action[k] = controller.forward(obs=obs[k, :].reshape(1, -1))
        obs[k+1, :], _, _, _, info, _ = env.step(action=action[k])

    obs = obs[:-1, :]

    plot_simulation(time, obs, action)

    plt.show()
