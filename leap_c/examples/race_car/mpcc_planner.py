"""Differentiable MPCC planner for the race-car environment.

Parallel implementation of ``RaceCarPlanner`` that uses Model Predictive Contouring
Control instead of the upstream-style ``s_ref`` tracking. Shares ``RaceCarEnv``,
the 2-vector action ``[derD, derDelta]``, and the ``LMS_Track.txt`` reference;
diverges only in the OCP formulation (see ``mpcc_acados_ocp.py``).

Two frame variants share this planner, selected via ``MpccPlannerConfig.frame``:

- ``cartesian`` (default): paper-style formulation. Physical state is in Cartesian
  coordinates and contouring/lag errors are computed against reference splines
  ``x^d(theta), y^d(theta), psi^d(theta)``.
- ``frenet``: same MPCC objective and virtual-progress augmentation, but with
  the existing Frenet bicycle dynamics. Useful as an ablation that isolates
  the cost-shape change from the frame change.

The OCP runs on the augmented state ``[x_phys (6), theta, v_theta]`` and the
augmented control ``[derD, derDelta, dv_theta]``. ``forward`` converts the Frenet
observation from the env into the augmented initial state and strips the
``dv_theta`` channel from the returned action so the planner output matches
``RaceCarEnv.action_space`` directly.

References:
----------
- ``mpcc/main.tex`` (project-local): the MPCC algorithm description.
- Liniger, A., Domahidi, A., Morari, M. (2015). "Optimization-based autonomous
  racing of 1:43 scale RC cars." Optimal Control Applications and Methods.
"""

from dataclasses import dataclass, field
from pathlib import Path

import torch

from leap_c.examples.race_car.bicycle_model import DEFAULT_TRACK_FILE
from leap_c.examples.race_car.mpcc_acados_ocp import (
    MpccNlpSolver,
    MpccParamInterface,
    create_mpcc_params,
    export_mpcc_ocp,
)
from leap_c.examples.race_car.mpcc_model import (
    DV_THETA_MAX_DEFAULT,
    DV_THETA_MIN_DEFAULT,
    V_THETA_MAX_DEFAULT,
    MpccFrame,
    build_mpcc_path_splines,
    frenet_obs_to_mpcc_state,
)
from leap_c.ocp.acados.parameters import AcadosParameter, AcadosParameterManager
from leap_c.ocp.acados.planner import AcadosPlanner
from leap_c.ocp.acados.torch import AcadosDiffMpcCtx, AcadosDiffMpcTorch


@dataclass(kw_only=True)
class MpccPlannerConfig:
    """Configuration for the MPCC race-car planner.

    Attributes:
        frame: Reference frame for the MPCC formulation. ``cartesian`` (default)
            follows ``mpcc/main.tex`` and computes contouring/lag errors via
            tangent projection. ``frenet`` keeps the existing Frenet bicycle and
            reduces ``e_c -> n``, ``e_l -> s - theta``.
        N_horizon: Number of MPC shooting intervals.
        T_horizon: Horizon duration [s].
        param_interface: ``"global"`` (one set of cost weights) or
            ``"stagewise"`` (per-stage cost weights).
        v_theta_max: Upper bound on the virtual progress velocity ``v_theta``
            (the lower bound is fixed at 0 for non-reversal).
        dv_theta_min, dv_theta_max: Hard input bounds on ``dv_theta``.
        track_file: Track reference file used by the spline builders.
        discount_factor: Discount factor along the MPC horizon. ``None`` defers
            to acados defaults (no discount).
        n_batch_init: Initially supported batch size for the batch OCP solver.
        num_threads_batch_solver: Threads for the batch solver.
        dtype: Output tensor dtype.
    """

    frame: MpccFrame = "cartesian"
    N_horizon: int = 50
    T_horizon: float = 1.0
    param_interface: MpccParamInterface = "global"
    v_theta_max: float = V_THETA_MAX_DEFAULT
    dv_theta_min: float = DV_THETA_MIN_DEFAULT
    dv_theta_max: float = DV_THETA_MAX_DEFAULT
    nlp_solver: MpccNlpSolver = "SQP_RTI"
    track_file: Path = field(default_factory=lambda: DEFAULT_TRACK_FILE)
    discount_factor: float | None = None
    n_batch_init: int | None = None
    num_threads_batch_solver: int | None = None
    dtype: torch.dtype | None = None


class MpccPlanner(AcadosPlanner[AcadosDiffMpcCtx]):
    """Model-predictive contouring-control planner for the race-car env.

    Augmented state ``[x_phys (6), theta, v_theta]`` with ``nx = 8``; augmented
    control ``[derD, derDelta, dv_theta]`` with ``nu = 3``. The first two
    channels of the returned control are sliced to match ``RaceCarEnv.action_space``.
    """

    cfg: MpccPlannerConfig

    def __init__(
        self,
        cfg: MpccPlannerConfig | None = None,
        params: list[AcadosParameter] | None = None,
        export_directory: Path | None = None,
    ) -> None:
        self.cfg = MpccPlannerConfig() if cfg is None else cfg
        params = (
            create_mpcc_params(
                param_interface=self.cfg.param_interface,
                N_horizon=self.cfg.N_horizon,
            )
            if params is None
            else params
        )
        param_manager = AcadosParameterManager(parameters=params, N_horizon=self.cfg.N_horizon)
        ocp = export_mpcc_ocp(
            param_manager=param_manager,
            frame=self.cfg.frame,
            track_file=self.cfg.track_file,
            N_horizon=self.cfg.N_horizon,
            T_horizon=self.cfg.T_horizon,
            v_theta_max=self.cfg.v_theta_max,
            dv_theta_min=self.cfg.dv_theta_min,
            dv_theta_max=self.cfg.dv_theta_max,
            nlp_solver=self.cfg.nlp_solver,
        )
        diff_mpc = AcadosDiffMpcTorch(
            ocp,
            discount_factor=self.cfg.discount_factor,
            export_directory=export_directory,
            n_batch_init=self.cfg.n_batch_init,
            num_threads_batch_solver=self.cfg.num_threads_batch_solver,
            dtype=self.cfg.dtype,
        )
        self._track_geom = build_mpcc_path_splines(self.cfg.track_file)
        super().__init__(param_manager=param_manager, diff_mpc=diff_mpc)

    def forward(
        self,
        obs: torch.Tensor,
        action: torch.Tensor | None = None,
        param: torch.Tensor | None = None,
        ctx: AcadosDiffMpcCtx | None = None,
    ) -> tuple[AcadosDiffMpcCtx, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Solve the MPCC OCP and return the env-applicable first control.

        Args:
            obs: ``(B, 6)`` Frenet state ``[s, n, alpha, v, D, delta]`` from
                ``RaceCarEnv``.
            action: Optional warm-start action passed through to the diff_mpc.
            param: Optional learnable parameter override. Defaults to the
                parameter manager's defaults broadcast to the batch dim.
            ctx: Previous solver context for warm-starting.

        Returns:
            Tuple ``(ctx, u0, x_plan, u_plan, value)``. ``u0`` and ``u_plan`` are
            sliced to two channels ``[derD, derDelta]`` so the action matches
            ``RaceCarEnv.action_space`` directly; ``x_plan`` retains the full
            augmented 8-state for diagnostics. The full 3-channel input plan
            (including ``dv_theta``) is still available via ``ctx.iterate.u``.
        """
        device = obs.device
        dtype = obs.dtype

        obs_np = obs.detach().cpu().numpy()
        x0_aug = frenet_obs_to_mpcc_state(obs_np, frame=self.cfg.frame, track_geom=self._track_geom)
        x0_aug_t = torch.as_tensor(x0_aug, dtype=dtype, device=device).contiguous()

        if param is None and self.param_manager.learnable_default_flat.size > 0:
            default_flat = torch.from_numpy(self.param_manager.learnable_default_flat).to(
                dtype=dtype, device=device
            )
            param = default_flat.unsqueeze(0).expand(obs.shape[0], -1).contiguous()

        p_stagewise = self.param_manager.combine_non_learnable_parameter_values(obs.shape[0])

        ctx, u0, x_plan, u_plan, value = self.diff_mpc(
            x0_aug_t, action, param, p_stagewise, ctx=ctx
        )

        u0_env = u0[..., :2].contiguous() if u0 is not None else u0
        u_plan_env = u_plan[..., :2].contiguous() if u_plan is not None else u_plan
        return ctx, u0_env, x_plan, u_plan_env, value
