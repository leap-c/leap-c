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

    cost_type: RaceCarAcadosCostType = "NONLINEAR_LS"
    param_interface: RaceCarAcadosParamInterface = "stagewise"  # stagewise for racing


class RaceCarController(ParameterizedController):
    """acados-based race car controller with learnable Q matrix parameters."""

    collate_fn_map = {AcadosDiffMpcCtx: collate_acados_diff_mpc_ctx}

    def __init__(
        self,
        cfg: RaceCarControllerConfig | None = None,
        params: list[AcadosParameter] | None = None,
        export_directory: Path | None = None,
    ):
        """Initializes the RaceCarController.

        Args:
            cfg: Configuration object containing MPC settings.
            params: Optional list of Parameter objects. If not provided,
                default parameters will be created.
            export_directory: Optional directory path where the generated
                acados solver code will be exported.
        """
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
            parameters=params, N_horizon=self.cfg.N_horizon
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
        
        # Load track data for reference generation
        try:
            self.Sref, self.Xref, self.Yref, self.Psiref, self.kapparef = getTrack(self.cfg.track_file)
            self.track_length = self.Sref[-1]
        except:
            print("Warning: Could not load track data")
            self.track_length = 10.0

        # Initialize reference tracking variables
        self._current_s = 0.0
        self._lookahead_distance = 3.0

    def set_racing_references(self, current_state: np.ndarray, lookahead_distance: float = 3.0):
        """Set racing reference trajectory based on current state.

        Args:
            current_state: Current race car state [s, n, alpha, v, D, delta]
            lookahead_distance: How far ahead to set the reference progress

        Note: References will be set in the forward() method via combine_non_learnable_parameter_values
        """
        # Store current state for use in forward()
        self._current_s = current_state[0]
        self._lookahead_distance = lookahead_distance

    def forward(self, obs, param, ctx=None) -> tuple[Any, torch.Tensor]:
        """Forward pass through the controller.

        Args:
            obs: Current observation/state tensor [batch_size, 6]
            param: Learnable parameters (Q matrix elements) [batch_size, n_params]
            ctx: Optional context from previous call

        Returns:
            Tuple of (context, control_action)
        """
        batch_size = obs.shape[0]

        # Set racing references based on current state
        # For batch processing, use the first sample to set references
        if batch_size > 0:
            current_state = obs[0].detach().cpu().numpy()
            self.set_racing_references(current_state)

        # Generate reference trajectory
        N = self.cfg.N_horizon
        dt = self.cfg.T_horizon / N
        s_current = self._current_s
        lookahead = self._lookahead_distance

        # Combine non-learnable parameters without overwriting
        # References are fixed in param_manager defaults
        p_stagewise = self.param_manager.combine_non_learnable_parameter_values(
            batch_size=batch_size
        )

        # Call the differentiable MPC solver
        ctx, u0, x, u, value = self.diff_mpc(
            obs, p_global=param, p_stagewise=p_stagewise, ctx=ctx
        )

        return ctx, u0

    def jacobian_action_param(self, ctx) -> np.ndarray:
        """Compute Jacobian of action w.r.t. learnable parameters."""
        return self.diff_mpc.sensitivity(ctx, field_name="du0_dp_global")

    @property
    def param_space(self) -> gym.Space:
        """Get the parameter space for learnable parameters."""
        return self.param_manager.get_param_space(dtype=np.float32)

    def default_param(self, obs) -> np.ndarray:
        """Get default learnable parameters."""
        return self.param_manager.learnable_parameters_default.cat.full().flatten()
    
    def get_predicted_trajectory(self, ctx) -> tuple[np.ndarray, np.ndarray]:
        """Get the predicted state and control trajectories from the last solve.
        
        Args:
            ctx: Context from the last forward call
            
        Returns:
            Tuple of (predicted_states, predicted_controls)
        """
        if ctx is None or not hasattr(ctx, 'x') or not hasattr(ctx, 'u'):
            return np.array([]), np.array([])
            
        # Extract trajectories from context
        x_pred = ctx.x.detach().cpu().numpy()  # Shape: (batch, N+1, nx)
        u_pred = ctx.u.detach().cpu().numpy()  # Shape: (batch, N, nu)
        
        return x_pred, u_pred
    
    def get_lap_progress(self, state: np.ndarray) -> float:
        """Get lap progress as a percentage.
        
        Args:
            state: Race car state [s, n, alpha, v, D, delta]
            
        Returns:
            Lap progress as percentage (0-100%)
        """
        s = state[0]
        return min(100.0, max(0.0, (s / self.track_length) * 100.0))
    
    def is_lap_complete(self, state: np.ndarray) -> bool:
        """Check if a lap has been completed.
        
        Args:
            state: Race car state [s, n, alpha, v, D, delta]
            
        Returns:
            True if lap is complete
        """
        s = state[0]
        return s >= self.track_length
    
    def get_racing_metrics(self, state_trajectory: np.ndarray, dt: float) -> dict:
        """Calculate racing performance metrics.
        
        Args:
            state_trajectory: Array of states over time [T, 6]
            dt: Time step
            
        Returns:
            Dictionary of racing metrics
        """
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