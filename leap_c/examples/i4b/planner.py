import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from leap_c.examples.i4b.acados_ocp import export_parametric_ocp, make_i4b_params
from leap_c.ocp.acados.parameters import AcadosParameter, AcadosParameterManager
from leap_c.ocp.acados.planner import AcadosPlanner
from leap_c.ocp.acados.torch import AcadosDiffMpcCtx, AcadosDiffMpcTorch

# i4b imports - extend path so planner can be used standalone
_I4B_ROOT = Path(__file__).resolve().parents[3] / "external" / "i4b"
if str(_I4B_ROOT) not in sys.path:
    sys.path.insert(0, str(_I4B_ROOT))

from src.models.model_buildings import Building  # noqa: E402
from src.models.model_hvac import Heatpump  # noqa: E402


@dataclass(kw_only=True)
class I4bPlannerCtx(AcadosDiffMpcCtx):
    """An extension of the AcadosDiffMpcCtx for the I4b planner."""


@dataclass(kw_only=True)
class I4bPlannerConfig:
    """Configuration for the I4b planner.

    Attributes:
        N_horizon: Number of shooting intervals (default: 96 = 24 h in 15-min steps).
        ws: Quadratic weight on soft-constraint slack variables for T_room comfort.
        delta_t: Sampling time in seconds (default 900 = 15 min).
        discount_factor: Discount factor along the MPC horizon.
        n_batch_init: Initially supported batch size for the batch OCP solver.
        num_threads_batch_solver: Number of parallel threads for the batch solver.
        dtype: Output tensor dtype. Uses PyTorch default if None.
    """

    N_horizon: int = 96  # 96 x 900 s = 24 h
    ws: float = 0.1
    delta_t: float = 900.0
    T_set_upper: float = 26.0
    discount_factor: float | None = None
    n_batch_init: int | None = None
    num_threads_batch_solver: int | None = None
    dtype: torch.dtype | None = None


# Normalisation constants matching I4bEnv (action_low=0, action_high=65)
_ACTION_MID = 32.5
_ACTION_HALF = 32.5


def _fc_to_staged(
    fc: torch.Tensor | None,
    current: np.ndarray,
    batch_size: int,
    N: int,
) -> np.ndarray:
    """Convert a forecast tensor to a (B, N+1, 1) stagewise numpy array.

    Args:
        fc: Forecast tensor of shape (B, N_fc) or None.
        current: Current value array of shape (B, 1), used when fc is None or too short.
        batch_size: Batch size B.
        N: MPC horizon length (number of intervals).

    Returns:
        Array of shape (B, N+1, 1).
    """
    if fc is None:
        return np.broadcast_to(current[:, np.newaxis, :], (batch_size, N + 1, 1)).copy()
    fc_np = fc.detach().cpu().numpy()[:, :, np.newaxis]  # (B, N_fc, 1)
    N_fc = fc_np.shape[1]
    if N_fc >= N + 1:
        return fc_np[:, : N + 1, :].copy()
    # Pad with last forecast value to cover remaining stages.
    pad = np.broadcast_to(fc_np[:, -1:, :], (batch_size, N + 1 - N_fc, 1)).copy()
    return np.concatenate([fc_np, pad], axis=1)


class I4bPlanner(AcadosPlanner[AcadosDiffMpcCtx]):
    """Acados-based MPC planner for the I4b building heat-pump control environment.

    The observation fed to ``forward`` must follow the layout produced by ``I4bEnv``
    (a ``spaces.Dict``):
        obs["state"]                       – building thermal states, shape (B, nx)
        obs["disturbances"]["T_amb"]       – ambient temperature, shape (B, 1)
        obs["disturbances"]["Qdot_gains"]  – heat gains, shape (B, 1)
        obs["setpoints"]["T_set_upper"]    – upper comfort bound, shape (B, 1)

    The planner extracts the building state and current disturbances from the
    observation, broadcasts the disturbances over the entire horizon, and calls the
    acados OCP solver.  The returned action is normalized to [-1, 1] to match the
    ``I4bEnv`` action space.

    Args:
        building_model: Instantiated Building model.  Must match the model used in
            ``I4bEnv`` to ensure dynamics consistency.
        hp_model: Instantiated Heatpump model (``Heatpump_AW`` or
            ``Heatpump_Vitocal``).
        cfg: High-level planner configuration.  Defaults are used if not provided.
        params: Optional parameter list; ``make_i4b_params()`` is used if None.
        diff_mpc_kwargs: Extra keyword arguments forwarded to ``AcadosDiffMpcTorch``.
        export_directory: Directory for generated acados C code.
    """

    cfg: I4bPlannerConfig

    def __init__(
        self,
        building_model: Building,
        hp_model: Heatpump,
        cfg: I4bPlannerConfig | None = None,
        params: list[AcadosParameter] | None = None,
        diff_mpc_kwargs: dict[str, Any] | None = None,
        export_directory: Path | None = None,
    ):
        self.cfg = I4bPlannerConfig() if cfg is None else cfg
        self._nx = len(building_model.state_keys)

        if params is None:
            params = make_i4b_params(self.cfg.N_horizon)

        param_manager = AcadosParameterManager(parameters=params, N_horizon=self.cfg.N_horizon)

        ocp = export_parametric_ocp(
            param_manager=param_manager,
            N_horizon=self.cfg.N_horizon,
            building_model=building_model,
            hp_model=hp_model,
            ws=self.cfg.ws,
            delta_t=self.cfg.delta_t,
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

    def forward(
        self,
        obs: dict[str, torch.Tensor | dict],
        action: torch.Tensor | None = None,
        param: torch.Tensor | None = None,
        ctx: AcadosDiffMpcCtx | None = None,
    ) -> tuple[
        AcadosDiffMpcCtx,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Solve the MPC problem and return the optimal normalised action.

        Args:
            obs: Dict observation as produced by ``I4bEnv``.  Must contain:
                ``obs["state"]`` – shape (batch_size, nx),
                ``obs["disturbances"]["T_amb"]`` – shape (batch_size, 1),
                ``obs["disturbances"]["Qdot_gains"]`` – shape (batch_size, 1),
                ``obs["setpoints"]["T_set_upper"]`` – shape (batch_size, 1).
            action: Warm-start action (optional).
            param: Learnable parameters (unused; all params are non-learnable).
            ctx: Previous solver context for warm-starting.

        Returns:
            ctx: Updated solver context.
            u0_norm: First control action normalised to [-1, 1], shape (batch_size, 1).
            x: State trajectory, shape (batch_size, N_horizon+1, nx).
            u: Control trajectory (T_HP [degC]), shape (batch_size, N_horizon, 1).
            value: Optimal cost, shape (batch_size, 1).
        """
        x0 = obs["state"]  # (B, nx)
        batch_size = x0.shape[0]
        N = self.cfg.N_horizon

        T_amb_now = obs["disturbances"]["T_amb"].detach().cpu().numpy()  # (B, 1)
        Qdot_gains_now = obs["disturbances"]["Qdot_gains"].detach().cpu().numpy()  # (B, 1)
        T_set_lower_now = obs["setpoints"]["T_set_lower"].detach().cpu().numpy()  # (B, 1)
        T_set_upper_now = obs["setpoints"]["T_set_upper"].detach().cpu().numpy()  # (B, 1)

        fc = obs.get("forecast", {})

        # Build (B, N+1, 1) stagewise arrays from forecast where available,
        # falling back to the current value broadcast over all stages.
        T_amb_staged = _fc_to_staged(fc.get("T_amb"), T_amb_now, batch_size, N)
        T_set_lower_staged = _fc_to_staged(fc.get("T_set_lower"), T_set_lower_now, batch_size, N)
        T_set_upper_staged = _fc_to_staged(fc.get("T_set_upper"), T_set_upper_now, batch_size, N)
        Qdot_gains_staged = np.broadcast_to(
            Qdot_gains_now[:, np.newaxis, :], (batch_size, N + 1, 1)
        ).copy()

        p_stagewise = self.param_manager.combine_non_learnable_parameter_values(
            T_amb=T_amb_staged,
            Qdot_gains=Qdot_gains_staged,
            T_set_lower=T_set_lower_staged,
            T_set_upper=T_set_upper_staged,
        )

        # Supply default learnable parameters (int_gains = 0) when not provided.
        if param is None and self.param_manager.learnable_parameters.size > 0:
            device = x0.device
            default_flat = torch.from_numpy(
                self.param_manager.learnable_parameters_default.cat.full().flatten()
            ).to(device)
            param = default_flat.unsqueeze(0).expand(batch_size, -1)

        diff_mpc_ctx, _, x, u, value = self.diff_mpc(x0, action, param, p_stagewise, ctx=ctx)

        # u[:, 0, 0] is the optimal T_HP [degC] at stage 0.
        T_HP_opt = u[:, 0, 0:1]  # (B, 1)
        u0_norm = (T_HP_opt - _ACTION_MID) / _ACTION_HALF
        u0_norm = torch.clamp(u0_norm, -1.0, 1.0)

        return diff_mpc_ctx, u0_norm, x, u, value

    def default_param(self, obs: dict | np.ndarray | None) -> np.ndarray:
        default = self.param_manager.learnable_parameters_default.cat.full().flatten()
        if obs is None:
            return default
        state = obs["state"] if isinstance(obs, dict) else obs
        if state.ndim <= 1:
            return default
        return np.broadcast_to(default, (*state.shape[:-1], default.size))
