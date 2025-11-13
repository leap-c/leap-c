from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import torch

from leap_c.examples.hvac.acados_ocp import export_parametric_ocp, make_default_hvac_params
from leap_c.examples.hvac.util import set_temperature_limits
from leap_c.ocp.acados.parameters import AcadosParameter, AcadosParameterManager
from leap_c.ocp.acados.planner import AcadosPlanner
from leap_c.ocp.acados.torch import AcadosDiffMpcCtx, AcadosDiffMpcTorch


class HvacPlannerCtx(NamedTuple):
    """An extension of the AcadosDiffMpcCtx to also store the heater states.

    Attributes:
        diff_mpc_ctx: The underlying AcadosDiffMpcCtx from the MPC solver.
        qh: Current heating power of the radiator [W].
        dqh: Current rate of change of heating power [W/s].
    """

    diff_mpc_ctx: AcadosDiffMpcCtx
    qh: torch.Tensor
    dqh: torch.Tensor

    @property
    def status(self):
        return self.diff_mpc_ctx.status

    @property
    def log(self):
        return self.diff_mpc_ctx.log


@dataclass(kw_only=True)
class HvacPlannerConfig:
    """Configuration for the HVAC planner.

    Attributes:
        N_horizon: The number of steps in the MPC horizon
            (default: 96, i.e., 24 hours in 15-minute steps).
        stagewise: Whether to use stage-wise parameters for forecasts
            (ambient temperature, solar radiation, prices).
    """

    N_horizon: int = 96  # 24 hours in 15 minutes time steps
    stagewise: bool = False


class HvacPlanner(AcadosPlanner[HvacPlannerCtx]):
    """acados-based planner for the HVAC system.

    The first part of the state corresponds to the first part of the observation of the
    StochasticThreeStateRcEnv environment, i.e., the indoor temperature Ti,
    the radiator temperature Th, and the envelope temperature Te.
    Appended to this state are the action "qh" from the environment
    (the heating power of the radiator), and its derivative "dqh". Hence, the action of
    this planner is "ddqh", the acceleration of the heating power.

    The cost function takes the form of
        0.25 * price * qh
        + q_Ti * (ref_Ti - ocp.model.x[0]) ** 2
        + q_dqh * (dqh) ** 2
        + q_ddqh * (ddqh) ** 2,
    i.e., a linear price term combined with weighted quadratic penalties on
    the room temperature residuals, the rate of change of the heating power,
    and the acceleration of the heating power.

    The dynamics correspond partly to the dynamics also found in the environment.
    The differences are:
    - The dynamics here do not include the noise.
    - In case the ambient temperature, the solar radiation and the prices are not learned,
    they are set to a default value, instead of the data being used.
    - The action "qh" from the environment is part of the state here.
    To make setting of the radiator heating power smoother,
    a double integrator is added to the dynamics (hence, the action in this planner is ddqh,
    the acceleration of the heating power).

    The inequality constraints are box constraints on the room temperature
    (comfort bounds, soft/slacked), and the heating power qh (hard).

    Attributes:
        cfg: Configuration object containing horizon length and other settings.
        stagewise: Whether to use stage-wise parameters.
    """

    cfg: HvacPlannerConfig

    def __init__(
        self,
        cfg: HvacPlannerConfig | None = None,
        params: tuple[AcadosParameter, ...] | None = None,
        diff_mpc_kwargs: dict[str, Any] | None = None,
        export_directory: Path | None = None,
    ) -> None:
        """Initializes the HvacPlanner.

        Args:
            cfg: A configuration object containing high-level settings for the
                MPC problem, such as horizon length. If not provided,
                a default config is used.
            params: An optional tuple of parameters to define the
                ocp object. If not provided, default parameters for the HVAC
                system will be created based on the cfg.
            diff_mpc_kwargs: Optional keyword arguments to pass to AcadosDiffMpcTorch.
            export_directory: An optional directory path where the generated
                `acados` solver code will be exported.
        """
        self.cfg = HvacPlannerConfig() if cfg is None else cfg
        self.stagewise = self.cfg.stagewise

        params = (
            make_default_hvac_params(
                stagewise=self.cfg.stagewise,
                N_horizon=self.cfg.N_horizon,
            )
            if params is None
            else params
        )

        param_manager = AcadosParameterManager(parameters=params, N_horizon=self.cfg.N_horizon)

        ocp = export_parametric_ocp(
            param_manager=param_manager,
            N_horizon=self.cfg.N_horizon,
        )

        if diff_mpc_kwargs is None:
            diff_mpc_kwargs = {}

        diff_mpc = AcadosDiffMpcTorch(ocp, **diff_mpc_kwargs, export_directory=export_directory)

        super().__init__(param_manager=param_manager, diff_mpc=diff_mpc)

    def forward(
        self,
        obs: torch.Tensor,
        action: torch.Tensor | None = None,
        param: torch.Tensor | None = None,
        ctx: HvacPlannerCtx | None = None,
    ) -> tuple[
        HvacPlannerCtx, torch.Tensor | None, torch.Tensor, torch.Tensor | None, torch.Tensor
    ]:
        """Computes the MPC solution for the HVAC system.

        Args:
            obs: Observation from the environment containing:
                - quarter_hour: Current quarter hour (0-95)
                - day_of_year: Current day of year
                - Ti, Th, Te: Indoor, radiator, and envelope temperatures
                - Forecasts: Ambient temperature, solar radiation, and price forecasts
            action: Not used in this planner.
            param: Learnable parameters for the MPC.
            ctx: Optional context from previous invocation containing heater states.

        Returns:
            ctx: Updated context containing solver state and heater power states.
            u0: First control action (ddqh).
            x: State trajectory (Ti, Th, Te, qh, dqh) over the horizon.
            u: Control trajectory (ddqh) over the horizon.
            value: Cost value of the trajectory.
        """
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

        # Construct initial state for MPC: [Ti, Th, Te, qh, dqh]
        x0 = torch.cat(
            [
                obs[:, 2:5],  # [Ti, Th, Te]
                qh,
                dqh,
            ],
            dim=1,
        )

        N_horizon = self.diff_mpc.diff_mpc_fun.ocp.solver_options.N_horizon
        quarter_hours = np.array(
            [
                np.arange(obs[i, 0].cpu().numpy(), obs[i, 0].cpu().numpy() + N_horizon + 1)
                % N_horizon
                for i in range(batch_size)
            ]
        )

        # Set time-varying temperature comfort bounds
        lb, ub = set_temperature_limits(quarter_hours=quarter_hours)

        # NOTE: In case we want to pass the data of exogenous influences to the planner,
        # we can do it here
        p_stagewise = self.param_manager.combine_non_learnable_parameter_values(
            lb_Ti=lb.reshape(batch_size, -1, 1),
            ub_Ti=ub.reshape(batch_size, -1, 1),
        )

        diff_mpc_ctx, u0, x, u, value = self.diff_mpc(
            x0,
            action,
            param,
            p_stagewise,
            ctx=diff_mpc_ctx,
        )

        ctx = HvacPlannerCtx(
            diff_mpc_ctx,
            qh=x[:, 1, 3].detach(),
            dqh=x[:, 1, 4].detach(),
        )

        return ctx, u0, x, u, value

    def default_param(self, obs: np.ndarray | None) -> np.ndarray:
        """Provides default parameters for the HVAC planner.

        If stagewise=True and obs is provided with forecasts, uses the forecasts
        for ambient temperature, solar radiation, and prices.

        Args:
            obs: Optional observation array containing forecasts.

        Returns:
            Default parameter array.
        """
        param = self.param_manager.learnable_parameters(
            self.param_manager.learnable_parameters_default.cat.full().flatten()
        )

        if not self.stagewise or obs is None:
            return param.cat.full().flatten()

        Ta_forecast, solar_forecast, price_forecast = decompose_observation(obs)[5:]

        if isinstance(Ta_forecast, torch.Tensor):
            Ta_forecast = Ta_forecast.cpu().numpy().flatten()
        if isinstance(solar_forecast, torch.Tensor):
            solar_forecast = solar_forecast.cpu().numpy().flatten()
        if isinstance(price_forecast, torch.Tensor):
            price_forecast = price_forecast.cpu().numpy().flatten()

        N_horizon = self.diff_mpc.ocp.solver_options.N_horizon
        for stage in range(N_horizon + 1):
            param[f"Ta_{stage}_{stage}"] = Ta_forecast[stage]
            param[f"Phi_s_{stage}_{stage}"] = solar_forecast[stage]
            param[f"price_{stage}_{stage}"] = price_forecast[stage]

        return param.cat.full().flatten()


def decompose_observation(obs: np.ndarray) -> tuple:
    """Decompose the observation vector into its components.

    Args:
        obs: Observation vector from the environment.

    Returns:
        Tuple containing:
        - quarter_hour: Current quarter hour of the day (0-95)
        - day_of_year: Current day of the year (1-365)
        - Ti: Indoor air temperature in Kelvin
        - Th: Radiator temperature in Kelvin
        - Te: Envelope temperature in Kelvin
        - Ta_forecast: Ambient temperature forecast for the next N steps
        - solar_forecast: Solar radiation forecast for the next N steps
        - price_forecast: Electricity price forecast for the next N steps
    """
    if obs.ndim > 1:
        N_forecast = (obs.shape[1] - 5) // 3

        quarter_hour = obs[:, 0]
        day_of_year = obs[:, 1]
        Ti = obs[:, 2]
        Th = obs[:, 3]
        Te = obs[:, 4]

        Ta_forecast = obs[:, 5 : 5 + 1 * N_forecast]
        solar_forecast = obs[:, 5 + 1 * N_forecast : 5 + 2 * N_forecast]
        price_forecast = obs[:, 5 + 2 * N_forecast : 5 + 3 * N_forecast]

        for forecast in [
            Ta_forecast,
            solar_forecast,
            price_forecast,
        ]:
            assert forecast.shape[1] == N_forecast, (
                f"Expected {N_forecast} forecasts, got {forecast.shape[1]}"
            )
    else:
        N_forecast = (len(obs) - 5) // 3

        quarter_hour = obs[0]
        day_of_year = obs[1]
        Ti = obs[2]
        Th = obs[3]
        Te = obs[4]

        Ta_forecast = obs[5 : 5 + 1 * N_forecast]
        solar_forecast = obs[5 + 1 * N_forecast : 5 + 2 * N_forecast]
        price_forecast = obs[5 + 2 * N_forecast : 5 + 3 * N_forecast]

        for forecast in [
            Ta_forecast,
            solar_forecast,
            price_forecast,
        ]:
            assert len(forecast) == N_forecast, (
                f"Expected {N_forecast} forecasts, got {len(forecast)}"
            )

    return (
        quarter_hour,
        day_of_year,
        Ti,
        Th,
        Te,
        Ta_forecast,
        solar_forecast,
        price_forecast,
    )
