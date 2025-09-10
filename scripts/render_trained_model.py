"""Script to load and render a trained SAC model."""

import argparse
import time
from pathlib import Path

import torch
import numpy as np
from gymnasium.utils.save_video import save_video

from leap_c.examples import create_env
from leap_c.torch.rl.sac import SacTrainer, SacTrainerConfig


def load_trained_model(checkpoint_dir: Path, env_name: str, device: str = "cpu"):
    """Load a trained SAC model from checkpoint directory."""
    
    # Create trainer with same config as training
    cfg = SacTrainerConfig()
    val_env = create_env(env_name, render_mode="rgb_array")
    train_env = create_env(env_name)
    
    trainer = SacTrainer(
        cfg=cfg,
        val_env=val_env,
        output_path="temp",
        device=device,
        train_env=train_env,
        extractor_cls="identity",
    )
    
    # Load from checkpoint directory (the trainer's load method expects the base directory)
    trainer.load(checkpoint_dir)
    trainer.eval()
    
    return trainer, val_env


def render_policy(trainer, env, num_episodes=5, max_steps=1000, save_video_path=None):
    """Render the trained policy."""
    
    frames = []
    episode_returns = []
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        episode_return = 0
        step_count = 0
        
        print(f"Episode {episode + 1}/{num_episodes}")
        
        while step_count < max_steps:
            # Get action from trained policy
            with torch.no_grad():
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(trainer.device)
                action, _, _ = trainer.pi(obs_tensor, deterministic=True)
                action = action.squeeze(0).cpu().numpy()
            
            # Take step
            obs, reward, terminated, truncated, info = env.step(action)
            episode_return += reward
            step_count += 1
            
            # Render frame
            if env.render_mode == "rgb_array":
                frame = env.render()
                if frame is not None:
                    frames.append(frame)
            elif env.render_mode == "human":
                env.render()
                time.sleep(0.02)  # Small delay for visualization
            
            if terminated or truncated:
                break
        
        episode_returns.append(episode_return)
        print(f"  Return: {episode_return:.3f}, Steps: {step_count}")
        
        if info:
            print(f"  Info: {info}")
    
    print(f"\nAverage return: {np.mean(episode_returns):.3f} ± {np.std(episode_returns):.3f}")
    
    # Save video if specified
    if save_video_path and frames:
        print(f"Saving video to {save_video_path}")
        save_video(
            frames,
            video_folder=str(save_video_path.parent),
            name_prefix=save_video_path.stem,
            fps=env.metadata.get("render_fps", 30),
        )


def main():
    parser = argparse.ArgumentParser(description="Render trained SAC model")
    parser.add_argument("checkpoint_dir", type=Path, help="Path to model checkpoint directory (contains ckpts/)")
    parser.add_argument("--env", type=str, default="cartpole", help="Environment name")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use")
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes to render")
    parser.add_argument("--max_steps", type=int, default=1000, help="Maximum steps per episode")
    parser.add_argument("--render_mode", type=str, default="human", choices=["human", "rgb_array"])
    parser.add_argument("--save_video", type=Path, help="Path to save video (only with rgb_array mode)")
    
    args = parser.parse_args()
    
    if not args.checkpoint_dir.exists():
        print(f"Checkpoint directory not found: {args.checkpoint_dir}")
        return
        
    # Check if it's a checkpoint directory or if we need to look for ckpts subdirectory
    if (args.checkpoint_dir / "ckpts").exists():
        checkpoint_dir = args.checkpoint_dir
    elif args.checkpoint_dir.name == "ckpts" and args.checkpoint_dir.exists():
        checkpoint_dir = args.checkpoint_dir.parent
    else:
        print(f"Could not find checkpoint files in: {args.checkpoint_dir}")
        print("Expected either a directory containing 'ckpts/' or the 'ckpts/' directory itself")
        return
    
    print(f"Loading model from: {checkpoint_dir}")
    print(f"Environment: {args.env}")
    
    # Load model
    trainer, env = load_trained_model(checkpoint_dir, args.env, args.device)
    
    # Set render mode
    env.render_mode = args.render_mode
    
    # Render policy
    render_policy(
        trainer, env, 
        num_episodes=args.episodes,
        max_steps=args.max_steps,
        save_video_path=args.save_video
    )
    
    env.close()


if __name__ == "__main__":
    main()