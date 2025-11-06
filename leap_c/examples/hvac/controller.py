from pathlib import Path
from typing import Any, NamedTuple

import casadi as ca
import gymnasium as gym
import numpy as np
import torch
from acados_template import ACADOS_INFTY, AcadosOcp
from scipy.constants import convert_temperature

from leap_c.controller import ParameterizedController
from leap_c.examples.hvac.config import make_default_hvac_params
from leap_c.ocp.acados.diff_mpc import collate_acados_diff_mpc_ctx
from leap_c.ocp.acados.parameters import AcadosParameter, AcadosParameterManager
from leap_c.ocp.acados.torch import AcadosDiffMpcCtx, AcadosDiffMpcTorch

from .util import set_temperature_limits, transcribe_discrete_state_space


class HvacControllerCtx(NamedTuple):
    """An extension of the AcadosDiffMpcCtx to also store the heater states."""

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


class HvacController(ParameterizedController[HvacControllerCtx]):
    """acados-based controller for the HVAC system.
    The first part of the state corresponds to the first part of the observation of the
    StochasticThreeStateRcEnv environment, i.e., the indoor temperature Ti,
    the radiator temperature Th, and the envelope temperature Te.
    Appended to this state are the action "qh" from the environment
    (the heating power of the radiator), and its derivative "dqh". Hence, the action of
    this controller is "ddqh", the acceleration of the heating power.

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
    a double integrator is added to the dynamics (hence, the action in this controller is ddqh,
    the acceleration of the heating power).

    The inequality constraints are box constraints on the room temperature
    (comfort bounds, soft/slacked), and the heating power qh (hard).

    Attributes:
        param_manager: For managing the parameters of the OCP.
        ocp: The AcadosOcp object representing the optimal control problem.
        diff_mpc: The AcadosDiffMpcTorch object for solving the OCP and computing sensitivities.
        stagewise: Whether to use stage-wise parameters.
        collate_fn_map: A mapping for collating contexts in batch processing.
    """

    collate_fn_map = {AcadosDiffMpcCtx: collate_acados_diff_mpc_ctx}

    def __init__(
        self,
        params: tuple[AcadosParameter, ...] | None = None,
        stagewise: bool = False,
        N_horizon: int = 96,  # 24 hours in 15 minutes time steps
        diff_mpc_kwargs: dict[str, Any] | None = None,
        export_directory: Path | None = None,
    ) -> None:
        super().__init__()

        self.stagewise = stagewise

        self.param_manager = AcadosParameterManager(
            parameters=params
            or make_default_hvac_params(
                stagewise=stagewise,
                N_horizon=N_horizon,
            ),
            N_horizon=N_horizon,
        )

        self.ocp = export_parametric_ocp(
            param_manager=self.param_manager,
            N_horizon=N_horizon,
        )

        if diff_mpc_kwargs is None:
            diff_mpc_kwargs = {}

        self.diff_mpc = AcadosDiffMpcTorch(
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
                np.arange(obs[i, 0].cpu().numpy(), obs[i, 0].cpu().numpy() + N_horizon + 1)
                % N_horizon
                for i in range(batch_size)
            ]
        )

        lb, ub = set_temperature_limits(quarter_hours=quarter_hours)

        # NOTE: In case we want to pass the data of exogenous influences to the controller,
        # we can do it here
        p_stagewise = self.param_manager.combine_non_learnable_parameter_values(
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

    def jacobian_action_param(self, ctx: HvacControllerCtx) -> np.ndarray:
        return self.diff_mpc.sensitivity(ctx.diff_mpc_ctx, field_name="du0_dp_global")

    @property
    def param_space(self) -> gym.Space:
        return self.param_manager.get_param_space(dtype=np.float64)

    def default_param(self, obs) -> np.ndarray | None:
        param = self.param_manager.learnable_parameters(
            self.param_manager.learnable_parameters_default.cat.full().flatten()
        )

        if not self.stagewise:
            return param.cat.full().flatten()

        Ta_forecast, solar_forecast, price_forecast = decompose_observation(obs)[5:]

        if isinstance(Ta_forecast, torch.Tensor):
            Ta_forecast = Ta_forecast.cpu().numpy().flatten()
        if isinstance(solar_forecast, torch.Tensor):
            solar_forecast = solar_forecast.cpu().numpy().flatten()
        if isinstance(price_forecast, torch.Tensor):
            price_forecast = price_forecast.cpu().numpy().flatten()

        for stage in range(self.ocp.solver_options.N_horizon + 1):
            param[f"Ta_{stage}_{stage}"] = Ta_forecast[stage]
            param[f"Phi_s_{stage}_{stage}"] = solar_forecast[stage]
            param[f"price_{stage}_{stage}"] = price_forecast[stage]

        return param.cat.full().flatten()

    def _extract_param_by_prefix(self, structured_param, key_prefix: str) -> np.ndarray:
        """
        Extract parameters from structured_param that start with the given key prefix.

        Args:
            structured_param: The structured parameter object
            key_prefix: The prefix to filter parameter keys (e.g., "Ta_", "Phi_s_", "price_")

        Returns:
            np.ndarray: Vertically concatenated array of parameter values
        """
        keys = [key for key in structured_param.keys() if key.startswith(key_prefix)]
        return ca.vertcat(*[structured_param[key] for key in keys]).full()


def export_parametric_ocp(
    param_manager: AcadosParameterManager,
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
    dqh = ca.SX.sym("dqh")  # Velocity of heat input to radiator
    ddqh = ca.SX.sym("ddqh")  # Acceleration of heat input to radiator

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
        + param_manager.get("q_Ti") * (param_manager.get("ref_Ti") - ocp.model.x[0]) ** 2
        + param_manager.get("q_dqh") * (dqh) ** 2
        + param_manager.get("q_ddqh") * (ddqh) ** 2
    )

    ocp.cost.cost_type_e = "EXTERNAL"
    ocp.model.cost_expr_ext_cost_e = (
        0.25 * param_manager.get("price") * qh
        + param_manager.get("q_Ti") * (param_manager.get("ref_Ti") - ocp.model.x[0]) ** 2
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

    ocp.constraints.lbx = np.array([0.0])  # Can only consume power
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


def decompose_observation(obs: np.ndarray) -> tuple:
    """
    Decompose the observation vector into its components.

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

        # Cast to appropriate types
        # quarter_hour = quarter_hour.astype(np.int32)
        # day_of_year = day_of_year.astype(np.int32)
        # Ti = Ti.astype(np.float32)
        # Th = Th.astype(np.float32)
        # Te = Te.astype(np.float32)
        # Ta_forecast = Ta_forecast.astype(np.float32)
        # solar_forecast = solar_forecast.astype(np.float32)
        # price_forecast = price_forecast.astype(np.float32)

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
