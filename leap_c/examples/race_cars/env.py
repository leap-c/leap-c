from dataclasses import dataclass
from typing import Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces


from leap_c.examples.race_cars.bicycle_model import bicycle_model
from leap_c.examples.race_cars.tracks.readDataFcn import getTrack
from leap_c.examples.race_cars.time2spatial import transformProj2Orig, transformOrig2Proj
from leap_c.examples.race_cars.plotFcn import plotTrackProj, plotRes, plotalat


@dataclass(kw_only=True)
class RaceCarEnvConfig:
    """Configuration for the Race Car environment."""

    track_file: str = "LMS_Track.txt"

    dt: float = 0.02
    max_time: float = 20.0

    # Race car physical parameters (from bicycle_model.py)
    m: float = 0.043
    C1: float = 0.5
    C2: float = 15.5
    Cm1: float = 0.28
    Cm2: float = 0.05
    Cr0: float = 0.011
    Cr2: float = 0.006

    # Performance thresholds
    max_lateral_accel: float = 4.0
    max_longitudinal_accel: float = 4.0
    track_width: float = 0.12


class RaceCarEnv(gym.Env):
    """
    Race car environment using the acados race car bicycle model.

    This environment uses spatial coordinates (s, n) where:
    - s: progress along the center line of the track
    - n: lateral deviation from the center line

    State: [s, n, alpha, v, D, delta] where:
    - s: progress along track [m]
    - n: lateral deviation [m] 
    - alpha: heading angle relative to track [rad]
    - v: velocity [m/s]
    - D: throttle/brake input [-1, 1]
    - delta: steering angle [rad]

    Action: [derD, derDelta] where:
    - derD: throttle/brake rate [-10, 10]
    - derDelta: steering rate [-2, 2]

    The objective is to complete laps as quickly as possible while staying on track.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 20}

    def __init__(
        self, render_mode: str | None = None, cfg: RaceCarEnvConfig | None = None
    ):
        self.cfg = RaceCarEnvConfig() if cfg is None else cfg

        self.model, self.constraint = bicycle_model(self.cfg.track_file)
        self.track_data = getTrack(self.cfg.track_file)
        self.Sref, self.Xref, self.Yref, self.Psiref, self.kapparef = self.track_data
        self.track_length = self.Sref[-1]

        # Define observation and action spaces
        # State bounds: [s, n, alpha, v, D, delta]
        high_state = np.array([
            self.track_length * 2,  # s (allow multiple laps)
            self.cfg.track_width,   # n (lateral deviation)
            2 * np.pi,              # alpha (heading angle)
            10.0,                   # v (velocity, reasonable max)
            1.0,                    # D (throttle)
            0.4,                    # delta (steering angle)
        ], dtype=np.float32)

        low_state = np.array([
            -self.track_length,     # s (can start before 0)
            -self.cfg.track_width,  # n
            -2 * np.pi,             # alpha
            0.0,                    # v (non-negative velocity)
            -1.0,                   # D
            -0.4,                   # delta
        ], dtype=np.float32)

        # Action bounds: [derD, derDelta]
        high_action = np.array([self.model.dthrottle_max, self.model.ddelta_max], dtype=np.float32)
        low_action = np.array([self.model.dthrottle_min, self.model.ddelta_min], dtype=np.float32)

        self.observation_space = spaces.Box(low_state, high_state, dtype=np.float32)
        self.action_space = spaces.Box(low_action, high_action, dtype=np.float32)

        # Environment state
        self.reset_needed = True
        self.t = 0.0
        self.state = None
        self.lap_count = 0
        self.best_lap_time = float('inf')
        self.current_lap_start_time = 0.0

        # Performance tracking
        self.state_trajectory = []
        self.action_trajectory = []
        self.lateral_accels = []
        self.longitudinal_accels = []

        # Rendering
        if not (render_mode is None or render_mode in self.metadata["render_modes"]):
            raise ValueError(
                f"render_mode must be one of {self.metadata['render_modes']}"
            )
        self.render_mode = render_mode
        self.window = None
        self.clock = None

    def _compute_accelerations(self, state, action):
        """Compute lateral and longitudinal accelerations for constraint checking."""
        s, n, alpha, v, D, delta = state
        derD, derDelta = action
        
        # Race car dynamics (from bicycle_model.py)
        Fxd = (self.cfg.Cm1 - self.cfg.Cm2 * v) * D - self.cfg.Cr2 * v * v - self.cfg.Cr0 * np.tanh(5 * v)
        
        # Lateral and longitudinal accelerations
        a_lat = self.cfg.C2 * v * v * delta + Fxd * np.sin(self.cfg.C1 * delta) / self.cfg.m
        a_long = Fxd / self.cfg.m * np.cos(self.cfg.C1 * delta)
        
        return a_lat, a_long

    def _dynamics_step(self, state, action, dt):
        """Integrate race car dynamics for one time step."""
        # This is a simplified integration - in practice you'd use the full CasADi model
        s, n, alpha, v, D, delta = state
        derD, derDelta = action
        
        # Get track curvature at current position
        # Simple interpolation - in practice use the CasADi interpolant
        s_idx = np.clip(int(s), 0, len(self.kapparef) - 1)
        kappa = self.kapparef[s_idx] if s >= 0 else self.kapparef[0]
        
        # Race car dynamics (simplified version of bicycle_model equations)
        Fxd = (self.cfg.Cm1 - self.cfg.Cm2 * v) * D - self.cfg.Cr2 * v * v - self.cfg.Cr0 * np.tanh(5 * v)
        
        # State derivatives
        sdot = (v * np.cos(alpha + self.cfg.C1 * delta)) / (1 - kappa * n)
        ndot = v * np.sin(alpha + self.cfg.C1 * delta)
        alphadot = v * self.cfg.C2 * delta - kappa * sdot
        vdot = Fxd / self.cfg.m * np.cos(self.cfg.C1 * delta)
        Ddot = derD
        deltadot = derDelta
        
        # Euler integration
        new_state = state + dt * np.array([sdot, ndot, alphadot, vdot, Ddot, deltadot])
        
        # Apply state bounds
        new_state[3] = max(0.0, new_state[3])  # velocity >= 0
        new_state[4] = np.clip(new_state[4], -1.0, 1.0)  # throttle bounds
        new_state[5] = np.clip(new_state[5], -0.4, 0.4)  # steering bounds
        
        return new_state

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Execute one step of the race car dynamics."""
        if self.reset_needed:
            raise Exception("Call reset before using the step method.")
        
        # Clip action to bounds
        action = np.clip(action, self.action_space.low, self.action_space.high)
        
        # Check acceleration constraints before applying action
        a_lat, a_long = self._compute_accelerations(self.state, action)
        
        # Update state using dynamics
        self.state = self._dynamics_step(self.state, action, self.cfg.dt)
        self.t += self.cfg.dt
        
        # Track trajectory
        self.state_trajectory.append(self.state.copy())
        self.action_trajectory.append(action.copy())
        self.lateral_accels.append(a_lat)
        self.longitudinal_accels.append(a_long)
        
        # Calculate reward
        reward = self._calculate_reward(action, a_lat, a_long)
        
        # Check termination conditions
        terminated, truncated, info = self._check_termination()
        
        self.reset_needed = terminated or truncated
        
        return self.state.copy(), reward, terminated, truncated, info

    def _calculate_reward(self, action, a_lat, a_long):
        """Calculate reward based on racing performance."""
        s, n, alpha, v, D, delta = self.state
        
        # Progress reward (main component)
        progress_reward = v * np.cos(alpha) * self.cfg.dt  # reward for forward progress
        
        # Penalty for lateral deviation from center line
        lateral_penalty = -10.0 * (n / self.cfg.track_width) ** 2
        
        # Penalty for excessive control inputs (encourage smooth driving)
        control_penalty = -0.1 * (action[0]**2 + action[1]**2)
        
        # Penalty for constraint violations
        constraint_penalty = 0.0
        if abs(a_lat) > self.cfg.max_lateral_accel:
            constraint_penalty -= 50.0 * (abs(a_lat) - self.cfg.max_lateral_accel)
        if abs(a_long) > self.cfg.max_longitudinal_accel:
            constraint_penalty -= 50.0 * (abs(a_long) - self.cfg.max_longitudinal_accel)
        if abs(n) > self.cfg.track_width:
            constraint_penalty -= 100.0 * (abs(n) - self.cfg.track_width)
        
        total_reward = progress_reward + lateral_penalty + control_penalty + constraint_penalty
        
        return total_reward

    def _check_termination(self):
        """Check if episode should terminate."""
        s, n, alpha, v, D, delta = self.state
        
        info = {}
        terminated = False
        truncated = False
        
        # Check if vehicle went off track (too far laterally)
        if abs(n) > self.cfg.track_width:
            terminated = True
            info = {"task": {"violation": True, "success": False, "reason": "off_track"}}
        
        # Check if vehicle completed a lap
        if s > self.track_length and self.lap_count == 0:
            self.lap_count += 1
            lap_time = self.t - self.current_lap_start_time
            if lap_time < self.best_lap_time:
                self.best_lap_time = lap_time
            
            info["lap"] = {
                "lap_time": lap_time,
                "best_lap_time": self.best_lap_time,
                "average_speed": self.track_length / lap_time if lap_time > 0 else 0.0
            }
            
            # Successful lap completion
            terminated = True
            info["task"] = {"violation": False, "success": True, "reason": "lap_complete"}
        
        # Check time limit
        if self.t > self.cfg.max_time:
            truncated = True
            info["task"] = {"violation": False, "success": False, "reason": "time_limit"}
        
        # Check if vehicle is going backwards significantly
        if v < 0.1 and self.t > 5.0:  # Very slow after initial period
            terminated = True
            info["task"] = {"violation": True, "success": False, "reason": "too_slow"}
        
        return terminated, truncated, info

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[np.ndarray, dict]:
        if seed is not None:
            super().reset(seed=seed)
            self.observation_space.seed(seed)
            self.action_space.seed(seed)
        
        self.t = 0.0
        self.state = self.init_state(options)
        self.reset_needed = False
        self.lap_count = 0
        self.current_lap_start_time = 0.0
        
        # Clear trajectories
        self.state_trajectory = [self.state.copy()]
        self.action_trajectory = []
        self.lateral_accels = []
        self.longitudinal_accels = []
        
        return self.state.copy(), {}

    def init_state(self, options: Optional[dict] = None) -> np.ndarray:
        """Initialize race car at the starting line."""
        # Start at beginning of track with some small perturbations for exploration
        s0 = -2.0  # Start slightly before the start line (as in original example)
        n0 = 0.0 + self.np_random.uniform(-0.02, 0.02)  # Small lateral perturbation
        alpha0 = 0.0 + self.np_random.uniform(-0.1, 0.1)  # Small heading perturbation
        v0 = 0.0  # Start from rest
        D0 = 0.0  # No throttle initially
        delta0 = 0.0  # Straight steering
        
        return np.array([s0, n0, alpha0, v0, D0, delta0], dtype=np.float32)

    def render(self):
        if self.render_mode is None:
            return
        
        import pygame
        from pygame import gfxdraw
        
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((1000, 800))
            pygame.display.set_caption("Race Car MPC")
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        if self.state is None:
            return None

        # Create canvas
        canvas = pygame.Surface((1000, 800))
        canvas.fill((255, 255, 255))  # white background
        
        # Get current state
        s, n, alpha, v, D, delta = self.state
        
        # Scale for rendering
        scale = 300  # pixels per meter
        center_x, center_y = 500, 400
        
        try:
            # Transform current position to x, y coordinates
            x_pos, y_pos, _, _ = transformProj2Orig(
                np.array([s]), np.array([n]), 
                np.array([alpha]), np.array([v]), 
                self.cfg.track_file
            )
            car_x = int(center_x + x_pos[0] * scale)
            car_y = int(center_y - y_pos[0] * scale)  # flip Y for screen coordinates
            
            # Draw track boundaries
            if hasattr(self, 'Xref') and hasattr(self, 'Yref'):
                # Draw center line
                center_points = []
                for i in range(0, len(self.Xref), 5):  # Sample every 5th point
                    x = int(center_x + self.Xref[i] * scale)
                    y = int(center_y - self.Yref[i] * scale)
                    if 0 <= x < 1000 and 0 <= y < 800:
                        center_points.append((x, y))
                
                if len(center_points) > 1:
                    pygame.draw.lines(canvas, (100, 100, 100), False, center_points, 2)
                
                # Draw track boundaries
                distance = self.cfg.track_width
                left_points = []
                right_points = []
                
                for i in range(0, len(self.Xref), 5):
                    # Left boundary
                    x_left = self.Xref[i] - distance * np.sin(self.Psiref[i])
                    y_left = self.Yref[i] + distance * np.cos(self.Psiref[i])
                    x = int(center_x + x_left * scale)
                    y = int(center_y - y_left * scale)
                    if 0 <= x < 1000 and 0 <= y < 800:
                        left_points.append((x, y))
                    
                    # Right boundary
                    x_right = self.Xref[i] + distance * np.sin(self.Psiref[i])
                    y_right = self.Yref[i] - distance * np.cos(self.Psiref[i])
                    x = int(center_x + x_right * scale)
                    y = int(center_y - y_right * scale)
                    if 0 <= x < 1000 and 0 <= y < 800:
                        right_points.append((x, y))
                
                if len(left_points) > 1:
                    pygame.draw.lines(canvas, (0, 0, 0), False, left_points, 3)
                if len(right_points) > 1:
                    pygame.draw.lines(canvas, (0, 0, 0), False, right_points, 3)
            
            # Draw race car trajectory
            if len(self.state_trajectory) > 1:
                traj_points = []
                states = np.array(self.state_trajectory[-100:])  # Last 100 points
                
                for state in states:
                    try:
                        x_traj, y_traj, _, _ = transformProj2Orig(
                            np.array([state[0]]), np.array([state[1]]), 
                            np.array([state[2]]), np.array([state[3]]), 
                            self.cfg.track_file
                        )
                        x = int(center_x + x_traj[0] * scale)
                        y = int(center_y - y_traj[0] * scale)
                        if 0 <= x < 1000 and 0 <= y < 800:
                            traj_points.append((x, y))
                    except:
                        continue
                
                if len(traj_points) > 1:
                    pygame.draw.lines(canvas, (255, 0, 0), False, traj_points, 2)
            
            # Draw race car as oriented rectangle
            car_length = 20
            car_width = 8
            
            # Car corners in local coordinates
            corners = [
                (-car_length//2, -car_width//2),
                (car_length//2, -car_width//2), 
                (car_length//2, car_width//2),
                (-car_length//2, car_width//2)
            ]
            
            # Rotate and translate corners
            cos_alpha = np.cos(alpha)
            sin_alpha = np.sin(alpha)
            rotated_corners = []
            
            for cx, cy in corners:
                rx = cx * cos_alpha - cy * sin_alpha
                ry = cx * sin_alpha + cy * cos_alpha
                rotated_corners.append((car_x + rx, car_y + ry))
            
            # Draw car body
            pygame.draw.polygon(canvas, (0, 0, 255), rotated_corners)
            pygame.draw.polygon(canvas, (0, 0, 0), rotated_corners, 2)
            
            # Draw velocity vector
            vel_length = min(v * 10, 50)  # Scale velocity for display
            vel_end_x = car_x + vel_length * cos_alpha
            vel_end_y = car_y + vel_length * sin_alpha
            pygame.draw.line(canvas, (0, 255, 0), (car_x, car_y), (vel_end_x, vel_end_y), 3)
            
        except Exception as e:
            # Fallback: draw car at center with basic info
            car_x, car_y = center_x, center_y
            pygame.draw.circle(canvas, (0, 0, 255), (car_x, car_y), 10)
        
        # Draw HUD information
        font = pygame.font.Font(None, 36)
        
        # Race info
        info_texts = [
            f"Time: {self.t:.2f}s",
            f"Speed: {v:.2f}m/s",
            f"Progress: {s:.1f}m ({(s/self.track_length*100):.1f}%)",
            f"Lateral: {n:.3f}m",
            f"Throttle: {D:.2f}",
            f"Steering: {delta:.3f}rad",
        ]
        
        for i, text in enumerate(info_texts):
            color = (0, 0, 0)
            if i == 2:  # Progress - green if good
                color = (0, 150, 0) if s > 0 else (150, 0, 0)
            elif i == 3:  # Lateral - red if too far
                color = (150, 0, 0) if abs(n) > self.cfg.track_width * 0.8 else (0, 0, 0)
            
            text_surface = font.render(text, True, color)
            canvas.blit(text_surface, (10, 10 + i * 30))
        
        # Draw progress bar
        progress_width = 300
        progress_height = 20
        progress_x = 650
        progress_y = 50
        
        # Background
        pygame.draw.rect(canvas, (200, 200, 200), (progress_x, progress_y, progress_width, progress_height))
        
        # Progress
        progress_pct = min(s / self.track_length, 1.0)
        if progress_pct > 0:
            pygame.draw.rect(canvas, (0, 255, 0), 
                           (progress_x, progress_y, int(progress_width * progress_pct), progress_height))
        
        # Border
        pygame.draw.rect(canvas, (0, 0, 0), (progress_x, progress_y, progress_width, progress_height), 2)
        
        # Progress text
        progress_text = font.render(f"Lap Progress: {progress_pct*100:.1f}%", True, (0, 0, 0))
        canvas.blit(progress_text, (progress_x, progress_y - 25))
        
        # Draw recent control inputs as bar charts
        if len(self.action_trajectory) > 0:
            recent_actions = self.action_trajectory[-20:]  # Last 20 actions
            
            # Throttle rate chart
            chart_x, chart_y = 650, 150
            chart_width, chart_height = 150, 100
            
            pygame.draw.rect(canvas, (240, 240, 240), (chart_x, chart_y, chart_width, chart_height))
            pygame.draw.rect(canvas, (0, 0, 0), (chart_x, chart_y, chart_width, chart_height), 2)
            
            if len(recent_actions) > 1:
                bar_width = chart_width // len(recent_actions)
                for i, action in enumerate(recent_actions):
                    throttle_rate = action[0]  # derD
                    bar_height = int(abs(throttle_rate) * chart_height / 20)  # Scale to chart
                    bar_x = chart_x + i * bar_width
                    bar_y = chart_y + chart_height // 2
                    
                    color = (255, 0, 0) if throttle_rate < 0 else (0, 255, 0)
                    if throttle_rate >= 0:
                        pygame.draw.rect(canvas, color, (bar_x, bar_y - bar_height, bar_width-1, bar_height))
                    else:
                        pygame.draw.rect(canvas, color, (bar_x, bar_y, bar_width-1, bar_height))
            
            # Chart label
            chart_label = font.render("Throttle Rate", True, (0, 0, 0))
            canvas.blit(chart_label, (chart_x, chart_y - 25))
        
        # Flip canvas (pygame uses different coordinate system)
        canvas = pygame.transform.flip(canvas, False, True)
        
        if self.render_mode == "human":
            self.window.blit(canvas, (0, 0))
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2))

    def close(self):
        if self.window is not None:
            import pygame
            pygame.display.quit()
            pygame.quit()