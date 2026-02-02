from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import torch
from scipy.constants import convert_temperature

from leap_c.examples.hvac.acados_ocp import (
    HvacAcadosParamGranularity,
    HvacAcadosParamInterface,
    export_parametric_ocp,
    make_default_hvac_params,
)
from leap_c.examples.hvac.utils import set_temperature_limits
from leap_c.ocp.acados.data import (
    collate_acados_flattened_batch_iterate_fn,
    collate_acados_ocp_solver_input,
)
from leap_c.ocp.acados.parameters import AcadosParameter, AcadosParameterManager
from leap_c.ocp.acados.planner import AcadosPlanner
from leap_c.ocp.acados.torch import AcadosDiffMpcCtx, AcadosDiffMpcTorch


@dataclass(kw_only=True)
class HvacPlannerCtx(AcadosDiffMpcCtx):
    """An extension of the AcadosDiffMpcCtx to also store the heater states.

    Attributes:
        qh: Current heating power of the radiator [W].
        dqh: Current rate of change of heating power [W/s].
        render_info: Optional dictionary containing data for rendering (parameters, actions, etc.).
    """

    qh: torch.Tensor
    dqh: torch.Tensor
    render_info: dict[str, Any] | None = None


def collate_acados_diff_mpc_ctx(
    batch: Sequence[AcadosDiffMpcCtx],
    collate_fn_map: dict[str, Callable] | None = None,
) -> AcadosDiffMpcCtx:
    """Collates a batch of AcadosDiffMpcCtx objects into a single object."""
    return AcadosDiffMpcCtx(
        iterate=collate_acados_flattened_batch_iterate_fn([ctx.iterate for ctx in batch]),
        log=None,
        status=np.array([ctx.status for ctx in batch]),
        solver_input=collate_acados_ocp_solver_input([ctx.solver_input for ctx in batch]),
        qh=torch.stack([ctx.qh for ctx in batch], dim=0),
        dqh=torch.stack([ctx.dqh for ctx in batch], dim=0),
    )


@dataclass(kw_only=True)
class HvacPlannerConfig:
    """Configuration for the HVAC planner.

    Attributes:
        N_horizon: The number of steps in the MPC horizon
            (default: 96, i.e., 24 hours in 15-minute steps).
        param_interface: Determines the exposed parameter interface of the planner.
        param_granularity: Determines the granularity of the parameters
        dtype: Type the planner output tensors will automatically be cast to.
    """

    N_horizon: int = 24 * 4 - 1  # 24 hours in 15 minutes time steps
    param_interface: HvacAcadosParamInterface = "reference"
    param_granularity: HvacAcadosParamGranularity = "global"

    dtype: torch.dtype = torch.float32


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

        params = (
            make_default_hvac_params(
                interface=self.cfg.param_interface,
                granularity=self.cfg.param_granularity,
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

        diff_mpc = AcadosDiffMpcTorch(
            ocp, **diff_mpc_kwargs, export_directory=export_directory, dtype=self.cfg.dtype
        )

        super().__init__(param_manager=param_manager, diff_mpc=diff_mpc)

    def forward(
        self,
        obs: dict[str, torch.Tensor | dict[str, torch.Tensor]],
        action: torch.Tensor | None = None,
        param: torch.Tensor | None = None,
        ctx: HvacPlannerCtx | None = None,
    ) -> tuple[
        HvacPlannerCtx, torch.Tensor | None, torch.Tensor, torch.Tensor | None, torch.Tensor
    ]:
        """Computes the MPC solution for the HVAC system.

        Args:
            obs: Dict observation from the environment containing:
                - time: dict with {quarter_hour, day_of_year, day_of_week}
                - state: tensor of shape (batch, 3) containing [Ti, Th, Te]
                - forecast: dict with {temperature, solar, price}
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
        batch_size = obs["time"]["quarter_hour"].shape[0]
        device = obs["time"]["quarter_hour"].device

        # Use default parameters if none provided and there are learnable parameters
        if param is None and self.param_manager.learnable_parameters.size > 0:
            default_flat = torch.from_numpy(
                self.param_manager.learnable_parameters_default.cat.full().flatten()
            ).to(device)
            param = default_flat.unsqueeze(0).expand(batch_size, -1)

        if not isinstance(ctx, HvacPlannerCtx):
            qh = torch.zeros((batch_size, 1), dtype=torch.float64, device=device)
            dqh = torch.zeros((batch_size, 1), dtype=torch.float64, device=device)
        else:
            qh = ctx.qh
            dqh = ctx.dqh

        # Set time-varying temperature comfort bounds from quarter_hour
        quarter_hours = obs["time"]["quarter_hour"]  # (batch_size, 1)
        # TODO (Jasper): Understand the wrapping logic here
        lb, ub = set_temperature_limits(
            quarter_hours=np.array(
                [
                    np.arange(
                        quarter_hours[i, 0].cpu().numpy(),
                        quarter_hours[i, 0].cpu().numpy() + self.cfg.N_horizon + 1,
                    )
                    % 96  # Wrap around 96 quarter hours per day
                    for i in range(batch_size)
                ]
            )
        )

        overwrites = {
            "lb_Ti": lb.reshape(batch_size, -1, 1),
            "ub_Ti": ub.reshape(batch_size, -1, 1),
        }

        sub_param: dict[str, torch.Tensor] = {}

        render_info = {
            "lb_Ti": overwrites["lb_Ti"],
            "ub_Ti": overwrites["ub_Ti"],
        }

        for key in self.param_manager.parameters.keys():
            if self.param_manager.parameters[key].interface == "fix":
                continue

            if not self.param_manager.has_learnable_param_pattern(f"{key}*"):
                # If the forecast parameter is not learned, set it from the observation
                if key in ["temperature", "solar", "price"]:
                    # Get forecast from obs dict
                    forecast_data = obs["forecast"][key]  # (batch_size, N_forecast)
                    # Truncate/pad to match horizon + 1 stages
                    n_stages = self.cfg.N_horizon + 1
                    overwrites[key] = (
                        forecast_data[:, :n_stages]
                        .reshape(batch_size, -1, 1)
                        .detach()
                        .cpu()
                        .numpy()
                    )
                    render_info[key] = overwrites[key]
            else:
                # If the forecast parameter is learned, extract its structured representation
                sub_param[key] = self.param_manager.get_labeled_learnable_parameters(
                    param, label=key
                )
                render_info[key] = sub_param[key].detach().cpu().numpy()

        p_stagewise = self.param_manager.combine_non_learnable_parameter_values(**overwrites)

        # Build state: [Ti, Th, Te, qh, dqh]
        thermal_state = obs["state"]  # (batch_size, 3) - [Ti, Th, Te]
        state = torch.cat([thermal_state, qh, dqh], dim=1)  # type: ignore[arg-type]

        diff_mpc_ctx, _, x, u, value = self.diff_mpc(
            state,
            action,
            param,
            p_stagewise,
            ctx=ctx,
        )

        render_info["Ti"] = x[:, :, 0].detach().cpu().numpy()  # Indoor temperature trajectory
        render_info["Th"] = x[:, :, 1].detach().cpu().numpy()  # Radiator temperature trajectory
        render_info["Te"] = x[:, :, 2].detach().cpu().numpy()  # Envelope temperature trajectory
        render_info["qh"] = x[:, :, 3].detach().cpu().numpy()  # Heater power trajectory
        render_info["dqh"] = x[:, :, 4].detach().cpu().numpy()
        render_info["u_trajectory"] = u.detach().cpu().numpy()  # Full action trajectory
        render_info["ddqh"] = render_info["u_trajectory"]

        # TODO: Assuming ref_Ti and q_Ti are the only parameters that are learnable
        render_info["ref_Ti_min"] = convert_temperature(
            self.param_manager.parameters["ref_Ti"].space.low, "kelvin", "celsius"
        )
        render_info["ref_Ti_max"] = convert_temperature(
            self.param_manager.parameters["ref_Ti"].space.high, "kelvin", "celsius"
        )
        render_info["q_Ti_min"] = self.param_manager.parameters["q_Ti"].space.low
        render_info["q_Ti_max"] = self.param_manager.parameters["q_Ti"].space.high

        # Dynamics parameters
        dynamics_params = ["gAw", "Rea", "Rhi", "Rie", "Ch", "Ci", "Ce"]
        for key in dynamics_params:
            if self.param_manager.has_learnable_param_pattern(f"{key}*"):
                render_info[key] = (
                    self.param_manager.get_labeled_learnable_parameters(param, label=key)
                    .detach()
                    .cpu()
                    .numpy()
                )
            else:
                # Use default value if not learnable
                render_info[key] = self.param_manager.parameters[key].default

        for key in ["temperature", "ref_Ti", "lb_Ti", "ub_Ti", "Ti", "Th", "Te"]:
            render_info[key] = convert_temperature(
                val=render_info[key],
                old_scale="kelvin",
                new_scale="celsius",
            )

        ctx = HvacPlannerCtx(
            **vars(diff_mpc_ctx),
            qh=x[:, 1, 3].detach()[:, None],
            dqh=x[:, 1, 4].detach()[:, None],
            render_info=render_info,
        )

        return ctx, x[:, 1, 3][:, None], x, u, value

    def default_param(
        self, obs: dict[str, np.ndarray | dict[str, np.ndarray]] | None
    ) -> np.ndarray:
        """Provides default parameters for the HVAC planner.

        # TODO (Jasper)
        If stagewise=True the forecast parameters will be set from obs.

        Args:
            obs: Optional Dict observation containing forecasts.

        Returns:
            Default parameter array.

        NOTE: For stagewise parameters, we assume that the forecast stages
        match the stages in the parameters. No block-wise varying parameters
        are supported here.
        """
        if obs is None:
            return self.param_manager.learnable_parameters_default.cat.full().flatten()

        # Convert tensors to numpy if needed
        def to_numpy(x):
            if isinstance(x, torch.Tensor):
                return x.detach().cpu().numpy()
            return x

        # Get forecasts from dict observation
        temp_forecast = to_numpy(obs["forecast"]["temperature"])
        solar_forecast = to_numpy(obs["forecast"]["solar"])
        price_forecast = to_numpy(obs["forecast"]["price"])

        # Handle both single and batched observations uniformly
        # Ensure forecasts are 2D: (n_batch, n_forecast)
        if temp_forecast.ndim == 1:
            temp_forecast = temp_forecast[None, :]
            solar_forecast = solar_forecast[None, :]
            price_forecast = price_forecast[None, :]
            was_single = True
        else:
            was_single = False

        n_batch = temp_forecast.shape[0]
        n_stages = self.cfg.N_horizon + 1

        # Truncate forecasts to match the horizon length
        temp_forecast = temp_forecast[:, :n_stages]
        solar_forecast = solar_forecast[:, :n_stages]
        price_forecast = price_forecast[:, :n_stages]

        # Prepare overwrites dict for learnable forecast parameters
        overwrites = {}
        forecast_data = {
            "temperature": temp_forecast,
            "solar": solar_forecast,
            "price": price_forecast,
        }

        for key, data in forecast_data.items():
            if self.param_manager.has_learnable_param_pattern(f"{key}*"):
                # This parameter is learnable, add to overwrites
                overwrites[key] = data

        # Use parameter manager's efficient method
        batch_param = self.param_manager.combine_default_learnable_parameter_values(
            batch_size=n_batch, **overwrites
        )

        # Return single array if input was single observation
        return batch_param[0] if was_single else batch_param
