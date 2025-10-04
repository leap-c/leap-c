from dataclasses import dataclass
from pathlib import Path
from typing import Any
import sys
import os

import gymnasium as gym
import numpy as np
import torch

from leap_c.controller import ParameterizedController
from leap_c.examples.race_cars.acados_ocp import (
    RaceCarAcadosCostType,
    RaceCarAcadosParamInterface,
    create_race_car_params,
    export_parametric_ocp,
)
from leap_c.ocp.acados.diff_mpc import AcadosDiffMpcCtx, collate_acados_diff_mpc_ctx
from leap_c.ocp.acados.parameters import AcadosParameter, AcadosParameterManager
from leap_c.ocp.acados.torch import AcadosDiffMpc

from leap_c.examples.race_cars.tracks.readDataFcn import getTrack


@dataclass(kw_only=True)
class RaceCarControllerConfig:
    """Configuration for the Race Car controller.

    Attributes:
        N_horizon: The number of steps in the MPC horizon.
        T_horizon: The prediction horizon duration in seconds.
        track_file: Track data file to use.
        cost_type: The type of cost to use, either "EXTERNAL" or "NONLINEAR_LS".
        param_interface: Determines the exposed parameter interface.
    """

    N_horizon: int = 50
    T_horizon: float = 1.0
    track_file: str = "LMS_Track.txt"  # track data file

    cost_type: RaceCarAcadosCostType = "NONLINEAR_LS"  # NONLINEAR_LS allows learnable Q matrix
    param_interface: RaceCarAcadosParamInterface = "stagewise"  # stagewise for racing


class RaceCarController(ParameterizedController):
    collate_fn_map = {AcadosDiffMpcCtx: collate_acados_diff_mpc_ctx}

    def __init__(
        self,
        cfg: RaceCarControllerConfig | None = None,
        params: list[AcadosParameter] | None = None,
        export_directory: Path | None = None,
    ):
        super().__init__()
        self.cfg = RaceCarControllerConfig() if cfg is None else cfg
        params = (
            create_race_car_params(
                param_interface=self.cfg.param_interface,
                N_horizon=self.cfg.N_horizon,
                track_file=self.cfg.track_file,
            )
            if params is None
            else params
        )

        self.param_manager = AcadosParameterManager(
            parameters=params,
            N_horizon=self.cfg.N_horizon
        )

        self.ocp = export_parametric_ocp(
            param_manager=self.param_manager,
            cost_type=self.cfg.cost_type,
            name="racecar",
            track_file=self.cfg.track_file,
            N_horizon=self.cfg.N_horizon,
            T_horizon=self.cfg.T_horizon,
        )

        self.diff_mpc = AcadosDiffMpc(self.ocp, export_directory=export_directory)
        
        Sref, _, _, _, _ = getTrack(self.cfg.track_file)
        self.track_length = Sref[-1]

        # Initialize reference tracking variables (matching initial state from env)
        self._current_s = -2.0  # Match env.init_state() s value
        self._lookahead_distance = 3.0

    def set_racing_references(self, current_state: np.ndarray, lookahead_distance: float = 3.0):
        self._current_s = current_state[0]
        self._lookahead_distance = lookahead_distance

    def forward(self, obs, param, ctx=None) -> tuple[Any, torch.Tensor]:
        batch_size = obs.shape[0]
        N = self.cfg.N_horizon

        s_current_batch = obs[:, 0].detach().cpu().numpy() if isinstance(obs, torch.Tensor) else obs[:, 0]

        sref_N_batch = s_current_batch + self._lookahead_distance

        yref_batch = np.zeros((batch_size, N + 1, 8))

        interp_weights = np.arange(N) / N

        s_ref_all = s_current_batch[:, None] + (sref_N_batch - s_current_batch)[:, None] * interp_weights[None, :]

        yref_batch[:, :N, 0] = s_ref_all
        yref_batch[:, N, :] = yref_batch[:, N-1, :]

        yref_e_batch = np.zeros((batch_size, N + 1, 6))
        yref_e_batch[:, :, 0] = sref_N_batch[:, None]

        p_stagewise = self.param_manager.combine_non_learnable_parameter_values(
            batch_size=batch_size,
            yref=yref_batch,
            yref_e=yref_e_batch,
        )

        ctx, u0, x, u, value = self.diff_mpc(
            obs, p_global=param, p_stagewise=p_stagewise, ctx=ctx
        )

        return ctx, u0

    def jacobian_action_param(self, ctx) -> np.ndarray:
        return self.diff_mpc.sensitivity(ctx, field_name="du0_dp_global")

    @property
    def param_space(self) -> gym.Space:
        return self.param_manager.get_param_space(dtype=np.float32)

    def default_param(self, obs) -> np.ndarray:
        return self.param_manager.learnable_parameters_default.cat.full().flatten()
    
    def get_lap_progress(self, state: np.ndarray) -> float:
        s = state[0]
        return min(100.0, max(0.0, (s / self.track_length) * 100.0))
    
    def is_lap_complete(self, state: np.ndarray) -> bool:
        s = state[0]
        return s >= self.track_length
    
    def get_racing_metrics(self, state_trajectory: np.ndarray, dt: float) -> dict:
        if len(state_trajectory) == 0:
            return {}
        
        states = np.array(state_trajectory)
        s_vals = states[:, 0]  # progress
        n_vals = states[:, 1]  # lateral deviation
        v_vals = states[:, 3]  # velocity
        
        # Lap time (if lap completed)
        lap_complete = np.any(s_vals >= self.track_length)
        lap_time = None
        if lap_complete:
            lap_idx = np.where(s_vals >= self.track_length)[0][0]
            lap_time = lap_idx * dt
        
        # Average metrics
        avg_speed = np.mean(v_vals)
        avg_lateral_dev = np.mean(np.abs(n_vals))
        max_lateral_dev = np.max(np.abs(n_vals))
        
        # Progress metrics
        total_progress = s_vals[-1] - s_vals[0]
        progress_rate = total_progress / (len(states) * dt) if len(states) > 1 else 0.0
        
        return {
            'lap_complete': lap_complete,
            'lap_time': lap_time,
            'total_progress': total_progress,
            'progress_rate': progress_rate,
            'avg_speed': avg_speed,
            'avg_lateral_deviation': avg_lateral_dev,
            'max_lateral_deviation': max_lateral_dev,
            'final_progress_pct': self.get_lap_progress(states[-1]),
        }