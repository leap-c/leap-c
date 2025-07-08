from pathlib import Path
from typing import Any

import casadi as ca
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
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

        batch_size = x0.shape[0]

        lb, ub = set_temperature_limits(decompose_observation(obs)[-1])

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

        action = np.array(u0.detach().numpy(), dtype=np.float32).reshape(-1)

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
    ocp.model.cost_expr_ext_cost = 0.25 * param_manager.get("price") * ocp.model.u

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
    night_start_hour: int = 22,
    night_end_hour: int = 8,
) -> tuple[np.ndarray[np.float64], np.ndarray[np.float64]]:
    """Set temperature limits based on the time of day."""
    if len(time) == 0:
        raise ValueError("Time array cannot be empty")

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


def plot_ocp_results(
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
    x = ctx.iterate.x.reshape(-1, 3)
    u = ctx.iterate.u.reshape(-1, 1)
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

    Ta_forecast, solar_forecast, price_forecast, time = decompose_observation(obs=obs)[
        5:
    ]
    solar_forecast = solar_forecast.reshape(-1)
    price_forecast = price_forecast.reshape(-1)
    time = time.reshape(-1)

    T_lower, T_upper = set_temperature_limits(time)

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
        time[:-1],
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


if __name__ == "__main__":
    horizon_hours = 24
    N_horizon = horizon_hours * 4  # 4 time steps per hour
    env = StochasticThreeStateRcEnv(
        step_size=900.0,  # 15 minutes in seconds
        horizon_hours=24,
        enable_noise=True,
    )

    param_manager = AcadosParamManager(
        params=make_default_hvac_params(),
        N_horizon=N_horizon,
    )

    controller = HvacController(
        N_horizon=N_horizon,
        diff_mpc_kwargs={
            "export_directory": Path("hvac_mpc_export"),
        },
    )

    days = 10
    n_steps = days * 24 * 4  # 4 time steps per hour

    obs, _ = env.reset(
        state_0=np.array([convert_temperature(20.0, "celsius", "kelvin")] * 3)
    )

    results = {
        key: np.empty(
            n_steps,
        )
        for key in [
            "Ti",
            "Th",
            "Te",
            "qh",
            "Ta",
            "Phi_s",
            "price",
        ]
    }

    results["time"] = []

    print(obs.shape)

    for k in range(n_steps):
        # NOTE: The SAC controller would modify the forecasted parameters
        Ti, Th, Te, Ta_forecast, solar_forecast, price_forecast, time = (
            decompose_observation(obs.reshape(1, -1))[2:]
        )

        param = param_manager.p_global_values(0)
        for stage in range(N_horizon + 1):
            param["Ta", stage] = Ta_forecast[:, stage]
            param["Phi_s", stage] = solar_forecast[:, stage]
            param["price", stage] = price_forecast[:, stage]
        param = param.cat.full().flatten()

        ctx, action = controller.forward(obs=obs.reshape(1, -1), param=param)

        if False:
            plot_ocp_results(obs=obs, ctx=ctx)
            plt.show()

        results["Ti"][k] = Ti[0]
        results["Th"][k] = Th[0]
        results["Te"][k] = Te[0]
        results["qh"][k] = action[0]
        results["Ta"][k] = Ta_forecast[0, 0]
        results["Phi_s"][k] = solar_forecast[0, 0]
        results["price"][k] = price_forecast[0, 0]
        results["time"].append(time[0, 0])

        obs = env.step(action=action)

    time = np.array(results["time"])
    Ti_lower, Ti_upper = set_temperature_limits(time)
    # Convert temperatures to Celsius for plotting
    Ti_celsius = convert_temperature(results["Ti"], "kelvin", "celsius")
    Th_celsius = convert_temperature(results["Th"], "kelvin", "celsius")
    Te_celsius = convert_temperature(results["Te"], "kelvin", "celsius")
    Ta_celsius = convert_temperature(results["Ta"], "kelvin", "celsius")
    Ti_lower_celsius = convert_temperature(Ti_lower.reshape(-1), "kelvin", "celsius")
    Ti_upper_celsius = convert_temperature(Ti_upper.reshape(-1), "kelvin", "celsius")
    qh = results["qh"]
    solar = results["Phi_s"]
    price = results["price"]

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

    plt.show()
