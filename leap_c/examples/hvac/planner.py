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
    """An extension of the AcadosDiffMpcCtx to also store the heater state.

    Attributes:
        qh: Current heating power of the radiator [kW], used as qh_prev in the next MPC call.
        render_info: Optional dictionary containing data for rendering (parameters, actions, etc.).
    """

    qh: torch.Tensor
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
    )


@dataclass(kw_only=True)
class HvacPlannerConfig:
    """Configuration for the HVAC planner.

    Attributes:
        N_horizon: The number of steps in the MPC horizon
            (default: 96, i.e., 24 hours in 15-minute steps).
        param_interface: Determines the exposed parameter interface of the planner.
        param_granularity: Determines the granularity of the parameters.
        discount_factor: discount factor along the MPC horizon.
            If `None`, it defaults to the behavior of `AcadosOcpOptions.cost_scaling`.
        n_batch_init: Initially supported batch size of the batch OCP solver.
            Using larger batches will trigger a delay for the creation of more solvers.
            If `None`, a default value is used.
        num_threads_batch_solver: Number of parallel threads to use for the batch OCP solver.
            If `None`, a default value is used.
        dtype: Type the planner output tensors will automatically be cast to. If `None`, PyTorch
            default dtype is used.
    """

    N_horizon: int = 24 * 4 - 1  # 24 hours in 15 minutes time steps
    param_interface: HvacAcadosParamInterface = "reference_dynamics"
    param_granularity: HvacAcadosParamGranularity = "global"

    discount_factor: float | None = None
    n_batch_init: int | None = None
    num_threads_batch_solver: int | None = None
    dtype: torch.dtype | None = None


class HvacPlanner(AcadosPlanner[HvacPlannerCtx]):
    """acados-based planner for the HVAC system.

    The first part of the state corresponds to the first part of the observation of the
    StochasticThreeStateRcEnv environment, i.e., the indoor temperature Ti,
    the radiator temperature Th, and the envelope temperature Te.
    Appended to this state is "qh_prev", the heating power applied at the previous step [kW].
    The action of this planner is "qh" [kW], the heating power directly.

    The cost function takes the form of
        0.25 * price * qh
        + q_dqh * (qh - qh_prev) ** 2,
    i.e., a linear price term combined with a quadratic penalty on the rate of change
    of heating power.

    The dynamics correspond partly to the dynamics also found in the environment.

    The differences are:
    - The dynamics here do not include the noise.
    - In case the ambient temperature, the solar radiation and the prices are not learned,
    they are set to a default value, instead of the data being used.

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
            ocp,
            discount_factor=self.cfg.discount_factor,
            export_directory=export_directory,
            n_batch_init=self.cfg.n_batch_init,
            num_threads_batch_solver=self.cfg.num_threads_batch_solver,
            dtype=self.cfg.dtype,
            **diff_mpc_kwargs,
        )
        super().__init__(param_manager=param_manager, diff_mpc=diff_mpc)

    def _set_stagewise_constraint_bounds(
        self,
        lb: np.ndarray,
        ub: np.ndarray,
        batch_size: int,
    ) -> None:
        """Set per-stage Ti comfort bounds on the forward solver via constraints_set.

        Args:
            lb: Lower bounds in Kelvin, shape (batch_size, N_horizon + 1).
            ub: Upper bounds in Kelvin, shape (batch_size, N_horizon + 1).
            batch_size: Number of problem instances in the batch.
        """
        N = self.cfg.N_horizon
        # Stages 1..N; stage 0 is fixed by x0 and is not constrained here.
        stages = list(range(1, N + 1))
        # lb/ub shape: (batch_size, N+1) → select stages 1..N → (batch_size, N, 1)
        lbx = lb[:batch_size, 1:, np.newaxis]
        ubx = ub[:batch_size, 1:, np.newaxis]
        self.diff_mpc.set_constraint_bounds(lbx, ubx, stages)

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
            ctx: Updated context containing solver state and heater power.
            u0: First control action (qh) in Watts [W], for the environment.
            x: State trajectory (Ti, Th, Te, qh_prev) over the horizon.
            u: Control trajectory (qh in kW) over the horizon.
            value: Cost value of the trajectory.
        """
        batch_size = obs["time"]["quarter_hour"].shape[0]
        device = obs["time"]["quarter_hour"].device

        # Use default parameters if none provided and there are learnable parameters
        if param is None and self.param_manager._learnable_parameter_store.size > 0:
            default_flat = torch.from_numpy(self.param_manager.learnable_default_flat).to(device)
            param = default_flat.unsqueeze(0).expand(batch_size, -1)

        if not isinstance(ctx, HvacPlannerCtx):
            qh = torch.zeros((batch_size, 1), dtype=torch.float64, device=device)
        else:
            qh = ctx.qh

        # Set time-varying temperature comfort bounds from quarter_hour
        quarter_hours = obs["time"]["quarter_hour"]  # (batch_size, 1)
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
        )  # lb, ub shape: (batch_size, N_horizon + 1), values in Kelvin

        # Apply per-stage comfort bounds directly via constraints_set.
        # Passing them as OCP parameters has no effect because the OCP model never
        # references lb_Ti / ub_Ti via param_manager.get().
        self._set_stagewise_constraint_bounds(lb, ub, batch_size)

        overwrites: dict = {}

        sub_param: dict[str, torch.Tensor] = {}

        render_info = {
            "lb_Ti": lb.reshape(batch_size, -1, 1),
            "ub_Ti": ub.reshape(batch_size, -1, 1),
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

        p_stagewise = self.param_manager.combine_non_learnable_parameter_values(
            batch_size=batch_size, **overwrites
        )

        # Build state: [Ti, Th, Te, qh_prev]
        thermal_state = obs["state"]  # (batch_size, 3) - [Ti, Th, Te]
        state = torch.cat([thermal_state, qh], dim=1)  # type: ignore[arg-type]

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
        render_info["qh"] = x[:, :, 3].detach().cpu().numpy()  # Heater power trajectory [kW]
        render_info["u_trajectory"] = u.detach().cpu().numpy()  # Full action trajectory (qh [kW])

        # TODO: Assuming ref_Ti and log_q_Ti are the only parameters that are learnable
        render_info["ref_Ti_min"] = convert_temperature(
            self.param_manager.parameters["ref_Ti"].space.low, "kelvin", "celsius"
        )
        render_info["ref_Ti_max"] = convert_temperature(
            self.param_manager.parameters["ref_Ti"].space.high, "kelvin", "celsius"
        )
        render_info["log_q_Ti_min"] = self.param_manager.parameters["log_q_Ti"].space.low
        render_info["log_q_Ti_max"] = self.param_manager.parameters["log_q_Ti"].space.high

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
            qh=u[:, 0, :].detach(),  # first input qh [kW], used as qh_prev next step
            render_info=render_info,
        )

        # u[:, 0, :] is qh in kW; environment expects W
        return ctx, u[:, 0, :] * 1000.0, x, u, value

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
            return self.param_manager.learnable_default_flat

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
