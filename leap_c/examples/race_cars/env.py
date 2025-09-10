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
    max_longitudinal_accel: float = 10.0
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

    def __init__(self, render_mode: str | None = None, cfg: RaceCarEnvConfig | None = None):
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
            raise ValueError(f"render_mode must be one of {self.metadata['render_modes']}")
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
        """Calculate reward based on racing performance: go fast while staying on centerline."""
        s, n, alpha, v, D, delta = self.state
        
        # 1. Speed reward - encourage high velocity
        # Use quadratic reward to strongly encourage speed
        target_speed = 3.0  # target racing speed [m/s]
        speed_reward = 2.0 * min(v / target_speed, 1.0)  # reward up to target speed
        if v > target_speed:
            speed_reward += 0.5 * (v - target_speed)  # bonus for exceeding target
        
        # 2. Progress reward - reward forward motion along track
        # Only reward progress if reasonably aligned with track
        alignment_factor = max(0.0, np.cos(alpha))  # cos(alpha) in [0,1] for good alignment
        progress_reward = v * alignment_factor * self.cfg.dt * 10.0
        
        # 3. Centerline following reward - strongly encourage staying on centerline
        # Use exponential decay to heavily penalize deviations
        centerline_reward = 5.0 * np.exp(-10.0 * (n / self.cfg.track_width) ** 2)
        
        # 4. Heading alignment reward - encourage car to be aligned with track
        heading_reward = 2.0 * np.exp(-5.0 * alpha ** 2)
        
        # 5. Smooth driving bonus - encourage smooth control inputs
        # smoothness_bonus = 1.0 * np.exp(-0.5 * (action[0]**2 + action[1]**2))
        
        # 6. Penalties for constraint violations (safety constraints)
        violation_penalty = 0.0
        
        # Severe penalty for going off track
        if abs(n) > self.cfg.track_width:
            violation_penalty -= 200.0 * (abs(n) - self.cfg.track_width)
        
        # Penalty for excessive accelerations (vehicle limits)
        if abs(a_lat) > self.cfg.max_lateral_accel:
            violation_penalty -= 100.0 * (abs(a_lat) - self.cfg.max_lateral_accel)
        if abs(a_long) > self.cfg.max_longitudinal_accel:
            violation_penalty -= 100.0 * (abs(a_long) - self.cfg.max_longitudinal_accel)
        
        # Penalty for driving backwards
        if v < 0.1:
            violation_penalty -= 10.0
        
        # Total reward combines all components
        total_reward = (speed_reward + progress_reward + centerline_reward + 
                       heading_reward + violation_penalty)
        
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
            info = {"task": {"violation": True, "success": False}}
        
        # Check if vehicle completed a lap
        elif s > self.track_length and self.lap_count == 0:
            self.lap_count += 1
            lap_time = self.t - self.current_lap_start_time
            if lap_time < self.best_lap_time:
                self.best_lap_time = lap_time
            
            # Add numeric lap statistics
            info["lap_time"] = lap_time
            info["best_lap_time"] = self.best_lap_time
            info["average_speed"] = self.track_length / lap_time if lap_time > 0 else 0.0
            
            # Successful lap completion
            terminated = True
            info["task"] = {"violation": False, "success": True}
        
        # Check time limit
        elif self.t > self.cfg.max_time:
            truncated = True
            info["task"] = {"violation": False, "success": False}
        
        # Check if vehicle is going backwards significantly
        elif v < 0.1 and self.t > 5.0:  # Very slow after initial period
            terminated = True
            info["task"] = {"violation": True, "success": False}
        
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
        
        # Initialize pygame
        pygame.init()
        pygame.font.init()
        
        if self.window is None and self.render_mode == "human":
            pygame.display.init()
            self.window = pygame.display.set_mode((1000, 800))
            pygame.display.set_caption("Race Car Environment")
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        if self.state is None:
            return None

        # Create canvas
        canvas = pygame.Surface((1000, 800))
        canvas.fill((255, 255, 255))  # white background
        
        # Get current state
        s, n, alpha, v, D, delta = self.state
        
        # Transform current car position to x,y coordinates
        try:
            x_car, y_car, _, _ = transformProj2Orig(
                np.array([s]), np.array([n]), 
                np.array([alpha]), np.array([v]), 
                self.cfg.track_file
            )
            x_car, y_car = x_car[0], y_car[0]
        except:
            # Fallback if transformation fails
            x_car, y_car = 0.0, 0.0
        
        # Set up coordinate system for rendering
        # Scale and center the track
        scale = 200  # pixels per meter
        center_x, center_y = 500, 400
        
        # Draw the full track using loaded track data
        if hasattr(self, 'Xref') and hasattr(self, 'Yref') and hasattr(self, 'Psiref'):
            # Draw center line
            center_points = []
            for i in range(len(self.Xref)):
                x = int(center_x + self.Xref[i] * scale)
                y = int(center_y - self.Yref[i] * scale)  # flip Y for screen coordinates
                if 0 <= x < 1000 and 0 <= y < 800:
                    center_points.append((x, y))
            
            if len(center_points) > 1:
                pygame.draw.lines(canvas, (128, 128, 128), False, center_points, 2)
            
            # Draw track boundaries (following the acados example)
            distance = self.cfg.track_width
            left_points = []
            right_points = []
            
            for i in range(len(self.Xref)):
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
        
        # Draw car position on track
        car_screen_x = int(center_x + x_car * scale)
        car_screen_y = int(center_y - y_car * scale)
        
        # Ensure car is visible on screen
        car_screen_x = max(10, min(990, car_screen_x))
        car_screen_y = max(10, min(790, car_screen_y))
        
        # Draw car as oriented rectangle
        car_length = 15
        car_width = 8
        
        # Car corners in local coordinates
        corners = [
            (-car_length//2, -car_width//2),
            (car_length//2, -car_width//2), 
            (car_length//2, car_width//2),
            (-car_length//2, car_width//2)
        ]
        
        # Get track heading at current position for car orientation
        track_heading = 0.0
        if hasattr(self, 'Psiref') and s >= 0 and s < len(self.Psiref):
            idx = min(int(s), len(self.Psiref) - 1)
            track_heading = self.Psiref[idx] if idx >= 0 else 0.0
        
        # Car's absolute heading
        car_heading = track_heading + alpha
        
        # Rotate and translate corners
        cos_heading = np.cos(car_heading)
        sin_heading = np.sin(car_heading)
        rotated_corners = []
        
        for cx, cy in corners:
            rx = cx * cos_heading - cy * sin_heading
            ry = cx * sin_heading + cy * cos_heading
            rotated_corners.append((car_screen_x + rx, car_screen_y + ry))
        
        # Draw car body
        pygame.draw.polygon(canvas, (0, 0, 255), rotated_corners)
        pygame.draw.polygon(canvas, (0, 0, 0), rotated_corners, 2)
        
        # Draw velocity vector
        vel_length = min(abs(v) * 10, 40)
        vel_end_x = car_screen_x + vel_length * cos_heading
        vel_end_y = car_screen_y + vel_length * sin_heading
        pygame.draw.line(canvas, (0, 255, 0), (car_screen_x, car_screen_y), (vel_end_x, vel_end_y), 3)
        
        # Draw trajectory
        if len(self.state_trajectory) > 1:
            traj_points = []
            for state in self.state_trajectory[-100:]:  # Last 100 points
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
        
        # Draw HUD information
        font = pygame.font.Font(None, 24)
        
        info_texts = [
            f"Time: {self.t:.2f}s",
            f"Speed: {v:.2f}m/s", 
            f"Position: s={s:.1f}m, n={n:.3f}m",
            f"Heading: {alpha:.3f}rad",
            f"Throttle: {D:.2f}",
            f"Steering: {delta:.3f}rad",
            f"Progress: {(s/self.track_length*100):.1f}%" if s >= 0 else "Progress: 0.0%"
        ]
        
        for i, text in enumerate(info_texts):
            color = (0, 0, 0)
            if i == 2 and abs(n) > self.cfg.track_width * 0.8:
                color = (255, 0, 0)  # Red if near track boundaries
            
            text_surface = font.render(text, True, color)
            canvas.blit(text_surface, (10, 10 + i * 25))
        
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