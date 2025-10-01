#!/usr/bin/env python3
"""
Training script for learning Q matrix parameters in race car MPC using SAC.

This script demonstrates how to use SAC to learn optimal cost matrix parameters
for a race car to achieve fast lap times while staying on track.
"""

import logging
import argparse
from pathlib import Path
from dataclasses import dataclass, field

import torch
import numpy as np
import gymnasium as gym

from leap_c.examples.race_cars.env import RaceCarEnv, RaceCarEnvConfig
from leap_c.examples import create_controller
from leap_c.torch.rl.sac import SacTrainer, SacTrainerConfig
from leap_c.torch.rl.sac_fop import SacFopTrainer, SacFopTrainerConfig
from leap_c.torch.nn.mlp import MlpConfig
from leap_c.run import default_output_path, init_run, default_controller_code_path


@dataclass(kw_only=True)
class RaceCarQMatrixTrainerConfig(SacFopTrainerConfig):
    racecar_env: RaceCarEnvConfig = field(default_factory=RaceCarEnvConfig)

    # Training specific
    max_episodes: int = 2000
    evaluation_freq: int = 100
    save_freq: int = 200

    # SAC hyperparameters tuned for parameter learning
    lr_q: float = 3e-4
    lr_pi: float = 2e-4
    lr_alpha: float = 2e-4
    init_alpha: float = 0.5  # Higher for more exploration in racing
    batch_size: int = 256
    gamma: float = 0.99
    tau: float = 0.005
    buffer_size: int = 1_000_000
    train_start: int = 1000
    noise: str = "param"  # For FOP mode
    entropy_correction: bool = False  # For FOP mode

class RaceCarRLEnv(gym.Wrapper):
    def __init__(self, racecar_env: RaceCarEnv):
        super().__init__(racecar_env)
        self.racecar_env = racecar_env

        self.action_space = racecar_env.action_space
        self.observation_space = racecar_env.observation_space
        
        # Performance tracking
        self.episode_rewards = []
        self.best_lap_time = float('inf')
        self.episode_start_time = 0.0
        self.last_s = 0.0  # Track progress

    def reset(self, **kwargs):
        obs, info = self.racecar_env.reset(**kwargs)
        self.episode_rewards = []
        self.episode_start_time = self.racecar_env.t
        self.last_s = obs[0]  # s coordinate

        return obs, info
    
    def step(self, action):
        next_obs, reward, terminated, truncated, info = self.racecar_env.step(action)

        # Track episode rewards
        self.episode_rewards.append(reward)

        # Add RL-specific info
        if terminated or truncated:
            racing_metrics = self._get_episode_metrics()
            info['rl_training'] = {
                'action': action.tolist(),
                'episode_reward': np.sum(self.episode_rewards),
                'racing_metrics': racing_metrics,
            }

            # Check for lap completion based on progress
            current_s = next_obs[0]
            if current_s > 8.0:  # Approximate lap completion for LMS track
                lap_time = self.racecar_env.t - self.episode_start_time
                if lap_time < self.best_lap_time:
                    self.best_lap_time = lap_time
                    info['rl_training']['new_best_lap'] = True
                    info['rl_training']['lap_time'] = lap_time

        return next_obs, reward, terminated, truncated, info
    
    def _calculate_racing_reward(self, state, base_reward, action, info):
        """
        THIS CODE IS WRITTEN BY CLAUDE
        SO NOW CURRENTLY NOT USED
        """
        s, n, alpha, v, D, delta = state

        reward = base_reward

        # Racing-specific reward components
        # 2. Progress reward (main component for racing)
        progress_reward = 10.0 if s > self.last_s else 0.0
        self.last_s = s

        # 3. Lap completion bonus
        if s > 8.7: # LMS track
            lap_time = self.racecar_env.t - self.episode_start_time
            if lap_time > 0:
                lap_bonus = 500.0 / max(lap_time, 1.0)
                reward += lap_bonus

        # 4. Control smoothness penalty
        # Penalize large control changes for smooth driving
        control_penalty = 0.01 * np.sum(action**2)

        total_reward = reward + progress_reward - control_penalty

        return total_reward
    
    def _get_episode_metrics(self):
        if hasattr(self.racecar_env, 'x_trajectory') and len(self.racecar_env.x_trajectory) > 0:
            trajectory = np.array(self.racecar_env.x_trajectory)
            max_s = np.max(trajectory[:, 0])  # Max progress along track
            avg_speed = np.mean(trajectory[:, 3])  # Average speed
            max_lateral_dev = np.max(np.abs(trajectory[:, 1]))  # Max lateral deviation

            return {
                'max_progress': max_s,
                'avg_speed': avg_speed,
                'max_lateral_deviation': max_lateral_dev,
                'episode_length': len(trajectory),
                'lap_complete': max_s > 8.0,
            }
        return {}


def create_environments(cfg: RaceCarQMatrixTrainerConfig):
    """Create training and validation environments."""

    # Create base race car environments (kappa_ref will be auto-loaded from track)
    train_racecar_env = RaceCarEnv(render_mode=None, cfg=cfg.racecar_env)
    val_racecar_env = RaceCarEnv(render_mode='rgb_array', cfg=cfg.racecar_env)

    # Wrap in RL training environments
    train_env = RaceCarRLEnv(train_racecar_env)
    val_env = RaceCarRLEnv(val_racecar_env)

    return train_env, val_env


def main():
    parser = argparse.ArgumentParser(description="Train SAC for race car control (end-to-end or MPC Q-matrix learning)")
    parser.add_argument("--output-path", type=Path, default=None,
                       help="Output directory for logs and models")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use (cpu/cuda)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--max-steps", type=int, default=2_000_000, help="Maximum training steps")
    parser.add_argument("--track-file", type=str, default="LMS_Track.txt", help="Track file to use")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level")
    parser.add_argument("--mode", type=str, default="fop", choices=["end-to-end", "fop"],
                       help="Training mode: 'end-to-end' (direct control) or 'fop' (MPC Q-matrix learning)")
    parser.add_argument("--controller", type=str, default="race_car",
                       help="Controller name for FOP mode (e.g., 'race_car', 'race_car_stagewise')")
    parser.add_argument("--reuse-code", action="store_true",
                       help="Reuse compiled MPC code (for FOP mode)")
    parser.add_argument("--reuse-code-dir", type=Path, default=None,
                       help="Directory to reuse compiled code from")

    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper()))
    logger = logging.getLogger(__name__)

    # Create output directory using standard pattern
    mode_tag = "fop" if args.mode == "fop" else "end-to-end"
    output_path = args.output_path if args.output_path else default_output_path(seed=args.seed, tags=["sac", mode_tag, "racecar"])
    output_path.mkdir(parents=True, exist_ok=True)

    # Create training configuration
    cfg = RaceCarQMatrixTrainerConfig(
        seed=args.seed,
        train_steps=args.max_steps,
        racecar_env=RaceCarEnvConfig(max_time=30.0, dt=0.02),
        critic_mlp=MlpConfig(hidden_dims=[512, 512, 256], activation="relu"),
        actor_mlp=MlpConfig(hidden_dims=[512, 512, 256], activation="relu"),
    )

    logger.info(f"Training mode: {args.mode}")
    logger.info(f"Training configuration: {cfg}")

    # Create trainer based on mode
    if args.mode == "fop":
        # FOP mode: Learn MPC Q-matrix parameters
        logger.info("Creating FOP trainer for MPC Q-matrix learning...")

        # Setup reuse code directory
        if args.reuse_code and args.reuse_code_dir is None:
            reuse_code_dir = default_controller_code_path()
        elif args.reuse_code_dir is not None:
            reuse_code_dir = args.reuse_code_dir
        else:
            reuse_code_dir = None

        # Create controller
        controller = create_controller(args.controller, reuse_code_base_dir=reuse_code_dir)
        logger.info(f"Controller: {args.controller}")
        logger.info(f"Parameter space: {controller.param_space}")

        # Create environments (raw RaceCarEnv, no wrapper)
        # Use 'human' for real-time visualization (slow) or None for fast training
        train_env = RaceCarEnv(render_mode='human', cfg=cfg.racecar_env)
        val_env = RaceCarEnv(render_mode='rgb_array', cfg=cfg.racecar_env)

        logger.info(f"Observation space: {train_env.observation_space}")
        logger.info(f"MPC parameter space: {controller.param_space}")

        trainer = SacFopTrainer(
            cfg=cfg,
            train_env=train_env,
            val_env=val_env,
            controller=controller,
            output_path=output_path,
            device=args.device,
            extractor_cls="identity"
        )
    else:
        # End-to-end mode: Direct vehicle control
        logger.info("Creating SAC trainer for end-to-end control learning...")
        train_env, val_env = create_environments(cfg)
        logger.info(f"Action space (vehicle control): {train_env.action_space}")
        logger.info(f"Observation space (race car state): {train_env.observation_space}")

        trainer = SacTrainer(
            cfg=cfg,
            train_env=train_env,
            val_env=val_env,
            output_path=output_path,
            device=args.device,
            extractor_cls="identity"
        )

    logger.info(f"Track file: {args.track_file}")

    # Initialize run
    init_run(trainer, cfg, output_path)

    # Start training
    logger.info("Starting race car control learning...")
    try:
        final_score = trainer.run()
        logger.info(f"Training completed successfully! Final score: {final_score}")
        final_model_path = output_path / "final_racecar_model.pt"
        torch.save({
            'policy': trainer.pi.state_dict(),
            'critic': trainer.q.state_dict(),
            'config': cfg,
            'track_file': args.track_file,
            'best_lap_time': getattr(train_env, 'best_lap_time', float('inf')),
        }, final_model_path)
        logger.info(f"Final race car model saved to {final_model_path}")
        if hasattr(train_env, 'best_lap_time') and train_env.best_lap_time < float('inf'):
            logger.info(f"Best lap time achieved: {train_env.best_lap_time:.3f}s")
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    finally:
        logger.info("Training finished. Cleaning up...")

if __name__ == "__main__":
    main()