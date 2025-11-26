from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import torch

from leap_c.examples.hvac.acados_ocp import export_parametric_ocp, make_default_hvac_params
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

        # Set temperature forecasts if they are not learned
        if not self.param_manager.has_learnable_param_pattern("temperature*"):
            start_idx = 5
            overwrites["temperature"] = obs[
                :, start_idx : start_idx + self.cfg.N_horizon + 1
            ].reshape(batch_size, -1, 1)

        # Set solar radiation forecasts if they are not learned
        if not self.param_manager.has_learnable_param_pattern("solar*"):
            start_idx = 5 + self.cfg.N_horizon + 1
            overwrites["solar"] = obs[:, start_idx : start_idx + self.cfg.N_horizon + 1].reshape(
                batch_size, -1, 1
            )

        # Set price forecasts if they are not learned
        if not self.param_manager.has_learnable_param_pattern("price*"):
            start_idx = 5 + 2 * (self.cfg.N_horizon + 1)
            overwrites["price"] = obs[:, start_idx : start_idx + self.cfg.N_horizon + 1].reshape(
                batch_size, -1, 1
            )

        p_stagewise = self.param_manager.combine_non_learnable_parameter_values(**overwrites)

        diff_mpc_ctx, _, x, u, value = self.diff_mpc(
            torch.cat([obs[:, 2:5], qh, dqh], dim=1),  # [Ti, Th, Te, qh, dqh]
            action,
            param,
            p_stagewise,
            ctx=ctx,
        )

        # Prepare render info if parameters are available
        render_info = None
        if param is not None:
            render_info = {
                "ddqh": u[:, 0, 0].detach().cpu().numpy(),  # First action (acceleration)
                "u_trajectory": u.detach().cpu().numpy(),  # Full action trajectory
            }
            # Extract learnable parameters if they exist
            try:
                for param_name in ["q_Ti", "q_dqh", "q_ddqh", "ref_Ti"]:
                    if self.param_manager.has_learnable_param_pattern(param_name):
                        idx = self.param_manager.get_learnable_param_index(param_name)
                        if idx is not None:
                            render_info[param_name] = param[:, idx].detach().cpu().numpy()
            except Exception:
                pass  # If parameter extraction fails, just skip

        ctx = HvacPlannerCtx(
            **vars(diff_mpc_ctx),
            qh=x[:, 1, 3].detach()[:, None],
            dqh=x[:, 1, 4].detach()[:, None],
            render_info=render_info,
        )

        return ctx, x[:, 1, 3][:, None], x, u, value

    def default_param(self, obs: np.ndarray | None) -> np.ndarray:
        """Provides default parameters for the HVAC planner.

        If stagewise=True the forecast parameters will be set from obs.

        Args:
            obs: Optional observation array containing forecasts.

        Returns:
            Default parameter array.

        NOTE: For stagewise parameters, we assume that the forecast stages
        match the stages in the parameters. No block-wise varying parameters
        are supported here.
        """
        param = self.param_manager.learnable_parameters(
            self.param_manager.learnable_parameters_default.cat.full().flatten()
        )

        if not self.cfg.stagewise or obs is None:
            default_param = param.cat.full().flatten()
            # If obs is batched, repeat default params for each batch
            if obs is not None and obs.ndim == 2:
                n_batch = obs.shape[0]
                return np.tile(default_param, (n_batch, 1))
            return default_param

        if obs.ndim == 1:
            # Single observation case
            forecasts = obs[5:].reshape(3, -1).T
            for j, key in enumerate(["temperature", "solar", "price"]):
                if self.param_manager.has_learnable_param_pattern(f"{key}"):
                    # Check if parameter is stagewise by checking if stagewise version exists
                    if self.param_manager.has_learnable_param_pattern(f"{key}_*_*"):
                        # Stagewise parameter
                        for stage in range(
                            self.diff_mpc.diff_mpc_fun.ocp.solver_options.N_horizon + 1
                        ):
                            try:
                                param[f"{key}_{stage}_{stage}"] = forecasts[stage, j]
                            except KeyError as e:
                                raise KeyError(
                                    f"Learnable parameter '{key}_{stage}_{stage}' not found."
                                ) from e
                    else:
                        # Non-stagewise parameter - set single value
                        try:
                            param[f"{key}"] = forecasts[0, j]
                        except KeyError as e:
                            raise KeyError(f"Learnable parameter '{key}' not found.") from e

            return param.cat.full().flatten()
        else:
            # Batched observation case: obs has shape (n_batch, obs_dim)
            n_batch = obs.shape[0]
            forecasts = obs[:, 5:].reshape(n_batch, 3, -1).transpose(0, 2, 1)

            # Get n_param from the flattened default parameters
            n_param = len(param.cat.full().flatten())
            batch_param = np.zeros((n_batch, n_param))

            for i in range(n_batch):
                param = self.param_manager.learnable_parameters(
                    self.param_manager.learnable_parameters_default.cat.full().flatten()
                )
                # forecasts now has shape (n_batch, N_stages, 3)
                for j, key in enumerate(["temperature", "solar", "price"]):
                    if self.param_manager.has_learnable_param_pattern(f"{key}"):
                        # Check if parameter is stagewise by checking if stagewise version exists
                        if self.param_manager.has_learnable_param_pattern(f"{key}_*_*"):
                            # Stagewise parameter
                            for stage in range(
                                self.diff_mpc.diff_mpc_fun.ocp.solver_options.N_horizon + 1
                            ):
                                try:
                                    param[f"{key}_{stage}_{stage}"] = forecasts[i, stage, j]
                                except KeyError as e:
                                    raise KeyError(
                                        f"Learnable parameter '{key}_{stage}_{stage}' not found."
                                    ) from e
                        else:
                            # Non-stagewise parameter - set single value
                            try:
                                param[f"{key}"] = forecasts[i, 0, j]
                            except KeyError as e:
                                raise KeyError(f"Learnable parameter '{key}' not found.") from e

                batch_param[i, :] = param.cat.full().flatten()

            return batch_param
