from pathlib import Path
from typing import Any, NamedTuple

import pandas as pd
import casadi as ca
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
from acados_template import ACADOS_INFTY
from acados_template import AcadosOcp
from .env import StochasticThreeStateRcEnv, decompose_observation
from scipy.constants import convert_temperature
from .util import transcribe_discrete_state_space

from leap_c.controller import ParameterizedController
from leap_c.examples.hvac.config import make_default_hvac_params
from leap_c.ocp.acados.parameters import AcadosParamManager, Parameter
from leap_c.ocp.acados.torch import AcadosDiffMpc, AcadosDiffMpcCtx
from leap_c.ocp.acados.diff_mpc import collate_acados_diff_mpc_ctx

from .util import set_temperature_limits


class HvacControllerCtx(NamedTuple):
    diff_mpc_ctx: AcadosDiffMpcCtx
    qh: torch.Tensor
    dqh: torch.Tensor

    @property
    def status(self):
        return self.diff_mpc_ctx.status

    @property
    def log(self):
        return self.diff_mpc_ctx.log

    @property
    def du0_dp_global(self):
        return self.diff_mpc_ctx.du0_dp_global


class HvacController(ParameterizedController):
    collate_fn_map = {AcadosDiffMpcCtx: collate_acados_diff_mpc_ctx}

    def __init__(
        self,
        params: tuple[Parameter, ...] | None = None,
        stagewise: bool = False,
        N_horizon: int = 96,  # 24 hours in 15 minutes time steps
        diff_mpc_kwargs: dict[str, Any] | None = None,
        export_directory: Path | None = None,
    ) -> None:
        super().__init__()

        self.stagewise = stagewise

        self.param_manager = AcadosParamManager(
            params=params or make_default_hvac_params(stagewise),
            N_horizon=N_horizon,
        )

        self.ocp = export_parametric_ocp(
            param_manager=self.param_manager,
            N_horizon=N_horizon,
        )

        if diff_mpc_kwargs is None:
            diff_mpc_kwargs = {}

        self.diff_mpc = AcadosDiffMpc(
            self.ocp, **diff_mpc_kwargs, export_directory=export_directory
        )

    def forward(self, obs, param: Any = None, ctx=None) -> tuple[Any, torch.Tensor]:
        batch_size = obs.shape[0]

        if ctx is None:
            qh = torch.zeros((batch_size, 1), dtype=torch.float64, device=obs.device)
            dqh = torch.zeros((batch_size, 1), dtype=torch.float64, device=obs.device)
            diff_mpc_ctx = None
        else:
            qh = ctx.qh
            dqh = ctx.dqh
            if qh.ndim == 1:
                qh = qh.unsqueeze(0)
            if dqh.ndim == 1:
                dqh = dqh.unsqueeze(0)

            diff_mpc_ctx = ctx.diff_mpc_ctx

        x0 = torch.cat(
            [
                obs[:, 2:5],
                qh,
                dqh,
            ],
            dim=1,
        )

        N_horizon = self.ocp.solver_options.N_horizon
        quarter_hours = np.array(
            [
                np.arange(
                    obs[i, 0].cpu().numpy(), obs[i, 0].cpu().numpy() + N_horizon + 1
                )
                % N_horizon
                for i in range(batch_size)
            ]
        )

        lb, ub = set_temperature_limits(quarter_hours=quarter_hours)

        p_stagewise = self.param_manager.combine_parameter_values(
            lb_Ti=lb.reshape(batch_size, -1, 1),
            ub_Ti=ub.reshape(batch_size, -1, 1),
        )

        diff_mpc_ctx, u0, x, u, value = self.diff_mpc(
            x0,
            p_global=param,
            p_stagewise=p_stagewise,
            ctx=diff_mpc_ctx,
        )

        ctx = HvacControllerCtx(
            diff_mpc_ctx,
            qh=x[:, 1, 3].detach(),
            dqh=x[:, 1, 4].detach(),
        )

        return ctx, x[:, 1, 3][:, None]

    def jacobian_action_param(self, ctx) -> np.ndarray:
        return self.diff_mpc.sensitivity(ctx.diff_mpc_ctx, field_name="du0_dp_global")

    @property
    def param_space(self) -> gym.Space:
        lb, ub = self.param_manager.get_p_global_bounds()
        return gym.spaces.Box(low=lb, high=ub, dtype=np.float64)

    def default_param(self, obs) -> np.ndarray | None:
        if self.stagewise:
            param = self.param_manager.p_global_values(0)

            N_horizon = self.ocp.solver_options.N_horizon
            Ta_forecast, solar_forecast, price_forecast = decompose_observation(obs)[5:]

            for stage in range(N_horizon + 1):
                param["Ta", stage] = Ta_forecast[stage]
                param["Phi_s", stage] = solar_forecast[stage]
                param["price", stage] = price_forecast[stage]
                # TODO: Retrieve these from the parameter manager after its refactored
                param["q_dqh", stage] = 1.0  # weight on rate of change of heater power
                param["q_ddqh", stage] = 1.0  # weight on acceleration of heater power
                param["q_Ti", stage] = 0.001  # weight on acceleration of heater power
                param["ref_Ti", stage] = convert_temperature(
                    21.0, "celsius", "kelvin"
                )  # weight on acceleration of heater power
            return param.cat.full().flatten()

        return self.param_manager.p_global_values.cat.full().flatten()  # type:ignore


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
        qh + dt * dqh + 0.5 * dt**2 * ddqh,
        dqh + dt * ddqh,
    )
    ocp.model.u = ddqh

    # Cost function
    ocp.cost.cost_type = "EXTERNAL"
    ocp.model.cost_expr_ext_cost = (
        0.25 * param_manager.get("price") * qh
        + param_manager.get("q_Ti")
        * (param_manager.get("ref_Ti") - ocp.model.x[0]) ** 2
        + param_manager.get("q_dqh") * (dqh) ** 2
        + param_manager.get("q_ddqh") * (ddqh) ** 2
    )

    ocp.cost.cost_type_e = "EXTERNAL"
    ocp.model.cost_expr_ext_cost_e = (
        0.25 * param_manager.get("price") * qh
        + param_manager.get("q_Ti")
        * (param_manager.get("ref_Ti") - ocp.model.x[0]) ** 2
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
    ocp.constraints.lh = np.array([0.0, 0.0])
    ocp.constraints.uh = np.array([ACADOS_INFTY, ACADOS_INFTY])

    ocp.constraints.idxsh = np.array([0, 1])
    ocp.cost.zl = 1e2 * np.ones((ocp.constraints.idxsh.size,))
    ocp.cost.Zl = 1e2 * np.ones((ocp.constraints.idxsh.size,))
    ocp.cost.zu = 1e2 * np.ones((ocp.constraints.idxsh.size,))
    ocp.cost.Zu = 1e2 * np.ones((ocp.constraints.idxsh.size,))

    ocp.constraints.lbx = np.array([-5000.0])
    ocp.constraints.ubx = np.array([5000.0])  # Watt
    ocp.constraints.idxbx = np.array([3])  # qh

    # Solver options
    ocp.solver_options.tf = N_horizon
    ocp.solver_options.N_horizon = N_horizon
    ocp.solver_options.integrator_type = "DISCRETE"
    ocp.solver_options.nlp_solver_type = "SQP"
    ocp.solver_options.hessian_approx = "EXACT"
    ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"

    return ocp


def _create_base_plot(
    figsize: tuple[float, float] = (12, 10),
) -> tuple[plt.Figure, list]:
    """Create base figure and axes for thermal building control plots."""
    fig, axes = plt.subplots(4, 1, figsize=figsize, sharex=True)
    fig.suptitle(
        "Thermal Building Control - OCP Solution", fontsize=16, fontweight="bold"
    )
    return fig, axes


def _plot_temperature_subplot(
    ax: plt.Axes,
    time: np.ndarray,
    Ti_celsius: np.ndarray,
    Te_celsius: np.ndarray,
    Ti_lower_celsius: np.ndarray,
    Ti_upper_celsius: np.ndarray,
) -> None:
    """Plot temperature data with comfort zone on given axes."""
    ax.fill_between(
        time,
        Ti_lower_celsius,
        Ti_upper_celsius,
        alpha=0.2,
        color="lightgreen",
        label="Comfort zone",
    )

    # Plot comfort bounds as dashed lines
    ax.step(time, Ti_lower_celsius, "g--", alpha=0.7, label="Lower bound")
    ax.step(time, Ti_upper_celsius, "g--", alpha=0.7, label="Upper bound")

    # Plot state trajectories
    ax.step(time, Ti_celsius, "b-", linewidth=2, label="Indoor temp. (Ti)")
    ax.step(
        time,
        Te_celsius,
        "orange",
        linewidth=2,
        label="Envelope temp. (Te)",
    )

    ax.set_ylabel("Temperature [°C]", fontsize=12)
    ax.legend(loc="best")
    ax.grid(visible=True, alpha=0.3)
    ax.set_title("Indoor/Envelope Temperature", fontsize=14, fontweight="bold")


def _plot_heater_subplot(
    ax: plt.Axes, time: np.ndarray, Th_celsius: np.ndarray
) -> None:
    """Plot heater temperature on given axes."""
    ax.step(time, Th_celsius, "b-", linewidth=2, label="Radiator temp. (Th)")
    ax.set_ylabel("Temperature [°C]", fontsize=12)
    ax.grid(visible=True, alpha=0.3)
    ax.set_title("Heater Temperature", fontsize=14, fontweight="bold")


def _plot_disturbance_subplot(
    ax: plt.Axes, time: np.ndarray, Ta_celsius: np.ndarray, solar: np.ndarray
) -> None:
    """Plot disturbance signals (outdoor temperature and solar radiation) on given axes."""
    # Outdoor temperature (left y-axis)
    ax.step(
        time,
        Ta_celsius,
        "b-",
        where="post",
        linewidth=2,
        label="Outdoor temp.",
    )
    ax.set_ylabel("Outdoor Temperature [°C]", color="b", fontsize=12)
    ax.tick_params(axis="y", labelcolor="b")

    # Solar radiation (right y-axis)
    ax_twin = ax.twinx()
    ax_twin.step(
        time,
        solar,
        color="orange",
        where="post",
        linewidth=2,
        label="Solar radiation",
    )
    ax_twin.set_ylabel("Solar Radiation [W/m²]", color="orange", fontsize=12)
    ax_twin.tick_params(axis="y", labelcolor="orange")

    ax.grid(visible=True, alpha=0.3)
    ax.set_title("Exogeneous Signals", fontsize=14, fontweight="bold")


def _plot_control_subplot(
    ax: plt.Axes, time: np.ndarray, control_input: np.ndarray, price: np.ndarray
) -> None:
    """Plot control input and energy price on given axes."""
    # Plot control as step function
    ax.step(
        time,
        control_input,
        "b-",
        where="post",
        linewidth=2,
        label="Heat input",
    )

    ax.set_xlabel("Time [hours]", fontsize=12)
    ax.set_ylabel("Heat Input [W]", color="b", fontsize=12)
    ax.grid(visible=True, alpha=0.3)
    ax.set_title("Control Input", fontsize=14, fontweight="bold")
    ax.set_ylim(bottom=0)

    # Add energy cost as a secondary y-axis
    ax_twin = ax.twinx()
    ax_twin.step(
        time,
        price,
        color="orange",
        where="post",
        linewidth=2,
        label="Energy cost (scaled)",
    )
    ax_twin.set_ylabel("Energy Price [EUR/kWh]", color="orange", fontsize=12)
    ax_twin.tick_params(axis="y", labelcolor="orange")
    ax_twin.grid(visible=False)
    ax_twin.set_ylim(bottom=0)


def _add_summary_stats(
    fig: plt.Figure, control_input: np.ndarray, ctx: Any = None
) -> None:
    """Add summary statistics text to the figure."""
    dt = 900.0  # Time step in seconds (15 minutes)
    total_energy_kWh = control_input.sum() * dt / 3600 / 1000  # Convert to kWh

    if ctx is not None:
        max_comfort_violation = max(
            ctx.iterate.sl.reshape(-1, 2).max(),
            ctx.iterate.su.reshape(-1, 2).max(),
        )
        stats_text = (
            f"Total Energy: {total_energy_kWh:.1f} kWh | "
            f"Max Comfort Violation: {max_comfort_violation:.2f} K"
        )
    else:
        stats_text = f"Total Energy: {total_energy_kWh:.1f} kWh"

    fig.text(
        0.78,
        0.02,
        stats_text,
        ha="center",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "lightgray", "alpha": 0.8},
    )


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
    u = x[:, 4]

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

    # Create base plot
    fig, axes = _create_base_plot(figsize)

    # Plot each subplot using helper functions
    _plot_temperature_subplot(
        axes[0], time, Ti_celsius, Te_celsius, T_lower_celsius, T_upper_celsius
    )
    _plot_heater_subplot(axes[1], time, Th_celsius)
    _plot_disturbance_subplot(axes[2], time, Ta_celsius, solar_forecast)
    _plot_control_subplot(axes[3], time, u, price_forecast)

    # Adjust layout and add summary stats
    plt.tight_layout()
    _add_summary_stats(fig, u, ctx)

    # Save figure if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Figure saved to: {save_path}")

    return fig


def plot_simulation(
    time: np.ndarray, obs: np.ndarray, action: np.ndarray
) -> plt.Figure:
    """
    Plot simulation results in a figure with four vertically stacked subplots.

    Args:
        time: Time array
        obs: Observation data
        action: Control action array

    Returns:
        matplotlib Figure object
    """
    quarter_hours, day, Ti, Th, Te, Ta_forecast, solar_forecast, price_forecast = (
        decompose_observation(obs=obs)
    )

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

    # Create base plot
    fig, axes = _create_base_plot()

    # Plot each subplot using helper functions
    _plot_temperature_subplot(
        axes[0], time, Ti_celsius, Te_celsius, Ti_lower_celsius, Ti_upper_celsius
    )
    _plot_heater_subplot(axes[1], time, Th_celsius)
    _plot_disturbance_subplot(axes[2], time, Ta_celsius, solar)
    _plot_control_subplot(axes[3], time, qh, price)

    # Adjust layout and add summary stats
    plt.tight_layout()
    _add_summary_stats(fig, qh)

    return fig


if __name__ == "__main__":
    horizon_hours = 24
    N_horizon = horizon_hours * 4  # 4 time steps per hour

    start_time = pd.Timestamp("2021-01-01 00:00:00+0100", tz="UTC+01:00")
    env = StochasticThreeStateRcEnv(
        step_size=900.0,  # 15 minutes in seconds
        horizon_hours=horizon_hours,
        start_time=start_time,
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
    action = np.zeros((n_steps, 1), dtype=np.float32)
    time = []

    for k in range(n_steps):
        time.append(info["time_forecast"][0])
        _, action[k] = controller.forward(obs=obs[k, :].reshape(1, -1))
        obs[k + 1, :], _, _, _, info, _ = env.step(action=action[k])

    time = np.array(time)
    obs = obs[:-1, :]

    plot_simulation(time, obs, action)

    plt.show()
