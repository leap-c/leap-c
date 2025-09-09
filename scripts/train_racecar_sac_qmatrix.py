#!/usr/bin/env python3
"""
Training script for learning Q matrix parameters in race car MPC using SAC.

This script demonstrates how to use SAC to learn optimal cost matrix parameters
for a race car to achieve fast lap times while staying on track.
"""

import argparse
import logging
from dataclasses import dataclass, field
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from leap_c.examples.race_cars.controller import RaceCarController, RaceCarControllerConfig
from leap_c.examples.race_cars.env import RaceCarEnv, RaceCarEnvConfig
from leap_c.torch.rl.sac import SacTrainer, SacTrainerConfig
from leap_c.torch.nn.mlp import MlpConfig
from leap_c.utils.logger import Logger


@dataclass(kw_only=True)
class RaceCarQMatrixTrainerConfig(SacTrainerConfig):
    """Configuration for training SAC to learn Q matrix parameters for racing."""
    
    # Environment config
    racecar_env: RaceCarEnvConfig = field(default_factory=RaceCarEnvConfig)
    controller: RaceCarControllerConfig = field(default_factory=RaceCarControllerConfig)
    
    # Training specific
    max_episodes: int = 2000
    evaluation_freq: int = 100
    save_freq: int = 200
    
    # SAC hyperparameters tuned for parameter learning
    lr_q: float = 3e-4
    lr_pi: float = 1e-4
    lr_alpha: float = 1e-4
    init_alpha: float = 0.2  # Higher for more exploration in racing
    batch_size: int = 256
    gamma: float = 0.99
    tau: float = 0.005
    buffer_size: int = 200000
    train_start: int = 2000


class RaceCarQMatrixEnv(gym.Wrapper):
    """
    Wrapper environment that treats Q matrix parameters as actions for racing.
    
    The agent (SAC) outputs Q matrix parameters, which are used by the MPC controller
    to generate race car controls. The reward is based on racing performance:
    speed, lap completion, and staying on track.
    """
    
    def __init__(self, racecar_env: RaceCarEnv, controller: RaceCarController):
        super().__init__(racecar_env)
        self.controller = controller
        self.racecar_env = racecar_env
        
        # Action space is now the Q matrix parameter space
        self.action_space = controller.param_space
        
        # Keep original observation space (race car state)
        self.observation_space = racecar_env.observation_space
        
        # Track MPC context
        self.mpc_ctx = None
        
        # Performance tracking
        self.episode_rewards = []
        self.best_lap_time = float('inf')
        self.episode_start_time = 0.0
        
    def reset(self, **kwargs):
        obs, info = self.racecar_env.reset(**kwargs)
        self.mpc_ctx = None
        self.episode_rewards = []
        self.episode_start_time = self.racecar_env.t
        
        return obs, info
    
    def step(self, q_params):
        """
        Step function where action is Q matrix parameters.
        
        Args:
            q_params: Q matrix parameters from SAC policy
            
        Returns:
            Standard gym step return tuple
        """
        # Convert Q parameters to torch tensor
        q_params_tensor = torch.tensor(q_params, dtype=torch.float32).unsqueeze(0)
        
        # Get current race car state
        current_obs = torch.tensor(self.env.state, dtype=torch.float32).unsqueeze(0)
        
        # Use MPC controller with learned Q parameters to get race car control
        try:
            self.mpc_ctx, vehicle_control = self.controller.forward(
                current_obs, q_params_tensor, self.mpc_ctx
            )
            
            # Extract control action and apply to race car
            vehicle_action = vehicle_control.detach().cpu().numpy()[0]
        except Exception as e:
            # If MPC fails, use safe fallback action
            print(f"MPC solver failed: {e}")
            vehicle_action = np.array([0.0, 0.0])  # Safe action: no throttle/steering change
        
        # Step the race car environment
        next_obs, base_reward, terminated, truncated, info = self.racecar_env.step(vehicle_action)
        
        # Enhanced reward calculation for racing
        reward = self._calculate_racing_reward(next_obs, base_reward, q_params, info)
        
        # Track episode rewards
        self.episode_rewards.append(reward)
        
        # Add Q-learning specific info
        if terminated or truncated:
            racing_metrics = self._get_episode_metrics()
            info['q_learning'] = {
                'q_params': q_params.tolist(),
                'episode_reward': np.sum(self.episode_rewards),
                'racing_metrics': racing_metrics,
            }
            
            # Update best lap time
            if racing_metrics.get('lap_complete') and racing_metrics.get('lap_time'):
                lap_time = racing_metrics['lap_time']
                if lap_time < self.best_lap_time:
                    self.best_lap_time = lap_time
                    info['q_learning']['new_best_lap'] = True
        
        return next_obs, reward, terminated, truncated, info
    
    def _calculate_racing_reward(self, state, base_reward, q_params, info):
        """Calculate reward optimized for racing performance."""
        s, n, alpha, v, D, delta = state
        
        # Base reward from environment (progress-based)
        reward = base_reward
        
        # Racing-specific reward components
        
        # 1. Speed reward (encourage high speeds)
        speed_reward = v * 0.5  # Reward proportional to speed
        
        # 2. Progress reward (main component for racing)
        progress_reward = 10.0 if s > getattr(self, 'last_s', s) else 0.0
        self.last_s = s
        
        # 3. Lap completion bonus
        if info.get('lap', {}).get('lap_time'):
            lap_time = info['lap']['lap_time']
            # Big bonus inversely proportional to lap time
            lap_bonus = 1000.0 / max(lap_time, 1.0)
            reward += lap_bonus
        
        # 4. Efficiency penalty (encourage reasonable Q parameters)
        # Penalize extremely large Q parameters that may cause instability
        param_penalty = 0.01 * np.sum(q_params**2)
        
        # 5. Track boundary penalty (already in base reward but emphasize)
        if abs(n) > 0.1:  # Close to track boundary
            boundary_penalty = -50.0 * (abs(n) - 0.1)
            reward += boundary_penalty
        
        # 6. Termination penalties
        if info.get('task', {}).get('violation'):
            reason = info['task'].get('reason', 'unknown')
            if reason == 'off_track':
                reward -= 200.0  # Heavy penalty for going off track
            elif reason == 'too_slow':
                reward -= 100.0  # Penalty for being too slow
        
        total_reward = reward + speed_reward + progress_reward - param_penalty
        
        return total_reward
    
    def _get_episode_metrics(self):
        """Get racing metrics for the episode."""
        if hasattr(self.racecar_env, 'state_trajectory') and len(self.racecar_env.state_trajectory) > 0:
            return self.controller.get_racing_metrics(
                self.racecar_env.state_trajectory, 
                self.racecar_env.cfg.dt
            )
        return {}


def create_environments(cfg: RaceCarQMatrixTrainerConfig):
    """Create training and validation environments."""
    
    # Create base race car environments
    train_racecar_env = RaceCarEnv(cfg=cfg.racecar_env)
    val_racecar_env = RaceCarEnv(cfg=cfg.racecar_env)
    
    # Create MPC controllers
    train_controller = RaceCarController(cfg=cfg.controller)
    val_controller = RaceCarController(cfg=cfg.controller)
    
    # Wrap in Q-matrix learning environments
    train_env = RaceCarQMatrixEnv(train_racecar_env, train_controller)
    val_env = RaceCarQMatrixEnv(val_racecar_env, val_controller)
    
    return train_env, val_env


def main():
    parser = argparse.ArgumentParser(description="Train SAC to learn Q matrix for race car MPC")
    parser.add_argument("--output-dir", type=str, default="./outputs/racecar_q_learning",
                       help="Output directory for logs and models")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use (cpu/cuda)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--max-steps", type=int, default=200000, help="Maximum training steps")
    parser.add_argument("--track-file", type=str, default="LMS_Track.txt", help="Track file to use")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=getattr(logging, args.log_level.upper()))
    logger = logging.getLogger(__name__)
    
    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Training configuration
    cfg = RaceCarQMatrixTrainerConfig(
        seed=args.seed,
        train_steps=args.max_steps,
        
        # Environment config
        racecar_env=RaceCarEnvConfig(
            track_file=args.track_file,
            max_time=30.0,  # Longer time for racing
            dt=0.02,        # Higher frequency for racing
        ),
        
        # Controller config
        controller=RaceCarControllerConfig(
            track_file=args.track_file,
            N_horizon=50,   # Long horizon for racing
            T_horizon=1.0,  # 1 second prediction
        ),
        
        # MLP configs for SAC networks
        critic_mlp=MlpConfig(
            hidden_dims=[512, 512, 256],
            activation="relu"
        ),
        actor_mlp=MlpConfig(
            hidden_dims=[512, 512, 256], 
            activation="relu"
        ),
    )
    
    logger.info(f"Training configuration: {cfg}")
    
    # Create environments
    logger.info("Creating racing environments...")
    try:
        train_env, val_env = create_environments(cfg)
        
        logger.info(f"Action space (Q params): {train_env.action_space}")
        logger.info(f"Observation space (race car state): {train_env.observation_space}")
        logger.info(f"Track file: {cfg.racecar_env.track_file}")
        
    except Exception as e:
        logger.error(f"Failed to create environments: {e}")
        logger.error("Make sure the acados race car example is available")
        return
    
    # Create trainer
    logger.info("Creating SAC trainer for race car Q-learning...")
    trainer = SacTrainer(
        cfg=cfg,
        train_env=train_env,
        val_env=val_env,
        output_path=output_path,
        device=args.device,
        extractor_cls="identity"  # Use identity extractor for state vector
    )
    
    # Start training
    logger.info("Starting race car Q-matrix learning...")
    try:
        trainer.train()
        logger.info("Training completed successfully!")
        
        # Save final model with racing-specific info
        final_model_path = output_path / "final_racecar_model.pt"
        torch.save({
            'policy': trainer.pi.state_dict(),
            'critic': trainer.q.state_dict(),
            'config': cfg,
            'track_file': args.track_file,
            'best_lap_time': getattr(train_env, 'best_lap_time', float('inf')),
        }, final_model_path)
        logger.info(f"Final race car model saved to {final_model_path}")
        
        # Print racing statistics
        if hasattr(train_env, 'best_lap_time') and train_env.best_lap_time < float('inf'):
            logger.info(f"Best lap time achieved: {train_env.best_lap_time:.3f}s")
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        raise


if __name__ == "__main__":
    main()