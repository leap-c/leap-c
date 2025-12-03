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
    """

    N_horizon: int = 24 * 4 - 1  # 24 hours in 15 minutes time steps
    param_interface: HvacAcadosParamInterface = "reference"
    param_granularity: HvacAcadosParamGranularity = "global"


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

        # Use default parameters if none provided and there are learnable parameters
        if param is None and self.param_manager.learnable_parameters.size > 0:
            default_flat = torch.from_numpy(
                self.param_manager.learnable_parameters_default.cat.full().flatten()
            ).to(obs.device)
            param = default_flat.unsqueeze(0).expand(batch_size, -1)

        if not isinstance(ctx, HvacPlannerCtx):
            qh = torch.zeros((batch_size, 1), dtype=torch.float64, device=obs.device)
            dqh = torch.zeros((batch_size, 1), dtype=torch.float64, device=obs.device)
        else:
            qh = ctx.qh
            dqh = ctx.dqh

        # Set time-varying temperature comfort bounds. Reconstruct quarter hours
        # from observation.
        # TODO: Should we pass the quarter hours with an info dict instead?
        lb, ub = set_temperature_limits(
            quarter_hours=np.array(
                [
                    np.arange(
                        obs[i, 0].cpu().numpy(), obs[i, 0].cpu().numpy() + self.cfg.N_horizon + 1
                    )
                    % self.cfg.N_horizon
                    for i in range(batch_size)
                ]
            )
        )

        overwrites = {
            "lb_Ti": lb.reshape(batch_size, -1, 1),
            "ub_Ti": ub.reshape(batch_size, -1, 1),
        }

        obs_idx = {
            "quarter_hour": 0,
            "day_of_year": 1,
            "Ti": 2,
            "Th": 3,
            "Te": 4,
            "temperature": slice(5, 5 + self.cfg.N_horizon + 1),
            "solar": slice(5 + self.cfg.N_horizon + 1, 5 + 2 * self.cfg.N_horizon + 2),
            "price": slice(5 + 2 * self.cfg.N_horizon + 2, 5 + 3 * self.cfg.N_horizon + 3),
        }

        sub_param: dict[str, torch.Tensor] = {}

        render_info = {
            "lb_Ti": overwrites["lb_Ti"],
            "ub_Ti": overwrites["ub_Ti"],
        }

        for key in [
            "temperature",
            "solar",
            "price",
            "ref_Ti",
            "q_Ti",
            "q_dqh",
            "q_ddqh",
        ]:
            if not self.param_manager.has_learnable_param_pattern(f"{key}*"):
                # If the forecast parameter is not learned, set it from the observation
                overwrites[key] = (
                    obs[:, obs_idx[key]].reshape(batch_size, -1, 1).detach().cpu().numpy()
                )
                render_info[key] = overwrites[key]
            else:
                # If the forecast parameter is learned, extract its structured representation
                sub_param[key] = self.param_manager.get_labeled_learnable_parameters(
                    param, label=key
                )
                render_info[key] = sub_param[key].detach().cpu().numpy()

        p_stagewise = self.param_manager.combine_non_learnable_parameter_values(**overwrites)

        diff_mpc_ctx, _, x, u, value = self.diff_mpc(
            torch.cat([obs[:, 2:5], qh, dqh], dim=1),  # [Ti, Th, Te, qh, dqh]
            action,
            param,
            p_stagewise,
            ctx=ctx,
        )

        for key in ["temperature", "ref_Ti", "lb_Ti", "ub_Ti"]:
            render_info[key] = convert_temperature(
                val=render_info[key],
                old_scale="kelvin",
                new_scale="celsius",
            )

        # Prepare render info if parameters are available
        render_info["qh"] = x[:, :, 3].detach().cpu().numpy()  # Heater power trajectory
        render_info["dqh"] = x[:, :, 4].detach().cpu().numpy()  # Heater power trajectory
        render_info["u_trajectory"] = u.detach().cpu().numpy()  # Full action trajectory
        render_info["ddqh"] = render_info["u_trajectory"]

        ctx = HvacPlannerCtx(
            **vars(diff_mpc_ctx),
            qh=x[:, 1, 3].detach()[:, None],
            dqh=x[:, 1, 4].detach()[:, None],
            render_info=render_info,
        )

        return ctx, x[:, 1, 3][:, None], x, u, value

    def default_param(self, obs: np.ndarray | None) -> np.ndarray:
        """Provides default parameters for the HVAC planner.

        # TODO (Jasper)
        If stagewise=True the forecast parameters will be set from obs.

        Args:
            obs: Optional observation array containing forecasts.

        Returns:
            Default parameter array.

        NOTE: For stagewise parameters, we assume that the forecast stages
        match the stages in the parameters. No block-wise varying parameters
        are supported here.
        """
        if isinstance(obs, torch.Tensor):
            obs = obs.detach().cpu().numpy()

        # Handle both single and batched observations uniformly
        # Ensure obs is 2D: (n_batch, obs_dim)
        obs_2d = obs if obs.ndim == 2 else obs[None, :]
        n_batch = obs_2d.shape[0]

        # Extract forecasts: shape (n_batch, n_stages, 3)
        n_stages = self.cfg.N_horizon + 1
        forecasts = np.asarray(obs_2d[:, 5:]).reshape(n_batch, 3, -1).transpose(0, 2, 1)

        # Truncate forecasts to match the horizon length
        forecasts = forecasts[:, :n_stages, :]

        # Prepare overwrites dict for learnable forecast parameters
        overwrites = {}
        forecast_keys = ["temperature", "solar", "price"]

        for j, key in enumerate(forecast_keys):
            if self.param_manager.has_learnable_param_pattern(f"{key}*"):
                # This parameter is learnable, add to overwrites
                overwrites[key] = forecasts[:, :, j]

        # Use parameter manager's efficient method
        batch_param = self.param_manager.combine_default_learnable_parameter_values(
            batch_size=n_batch, **overwrites
        )

        # Return single array if input was single observation
        return batch_param[0] if obs.ndim == 1 else batch_param
