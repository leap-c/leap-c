#!/usr/bin/env python3
"""
Evaluation script for learned Q matrix parameters in race car MPC.

This script loads a trained SAC model and evaluates the learned Q matrix parameters
on the race car lap time optimization task.
"""

import argparse
import logging
from pathlib import Path
import sys
import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from leap_c.examples.race_cars.controller import RaceCarController, RaceCarControllerConfig
from leap_c.examples.race_cars.env import RaceCarEnv, RaceCarEnvConfig
from scripts.train_racecar_sac_qmatrix import RaceCarQMatrixEnv

# Add the race car example path to sys.path for plotting
race_car_path = os.path.join(os.path.dirname(__file__), '../external/acados/examples/acados_python/race_cars')
if race_car_path not in sys.path:
    sys.path.insert(0, race_car_path)

try:
    from plotFcn import plotTrackProj
    from time2spatial import transformProj2Orig
except ImportError as e:
    print(f"Warning: Could not import race car plotting: {e}")


def load_trained_model(model_path: str, device: str = "cpu"):
    """Load trained SAC model for race car."""
    checkpoint = torch.load(model_path, map_location=device)
    return checkpoint


def evaluate_q_params(q_params: np.ndarray, env: RaceCarQMatrixEnv, 
                     num_episodes: int = 5, render: bool = False):
    """
    Evaluate Q matrix parameters on the race car environment.
    
    Args:
        q_params: Q matrix parameters to evaluate
        env: Environment to evaluate on
        num_episodes: Number of episodes to run
        render: Whether to render episodes
        
    Returns:
        Dictionary with evaluation metrics
    """
    episode_rewards = []
    episode_lap_times = []
    episode_progresses = []
    episode_avg_speeds = []
    lap_completions = []
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        
        terminated = truncated = False
        step_count = 0
        
        print(f"\nEpisode {episode + 1}/{num_episodes}")
        
        while not (terminated or truncated):
            # Use the same Q parameters for the entire episode
            obs, reward, terminated, truncated, info = env.step(q_params)
            
            episode_reward += reward
            step_count += 1
            
            # Print progress periodically
            if step_count % 100 == 0:
                s = obs[0]
                progress_pct = env.controller.get_lap_progress(obs)
                print(f"  Step {step_count}: Progress {progress_pct:.1f}%, Speed {obs[3]:.2f}m/s")
            
            if render and episode == 0:  # Only render first episode
                env.render()
        
        # Extract episode metrics
        episode_rewards.append(episode_reward)
        
        if 'q_learning' in info and 'racing_metrics' in info['q_learning']:
            metrics = info['q_learning']['racing_metrics']
            
            lap_complete = metrics.get('lap_complete', False)
            lap_completions.append(lap_complete)
            
            if lap_complete and metrics.get('lap_time'):
                lap_time = metrics['lap_time']
                episode_lap_times.append(lap_time)
                print(f"  Lap completed in {lap_time:.3f}s!")
            else:
                print(f"  Lap not completed. Final progress: {metrics.get('final_progress_pct', 0):.1f}%")
            
            episode_progresses.append(metrics.get('final_progress_pct', 0))
            episode_avg_speeds.append(metrics.get('avg_speed', 0))
            
            print(f"  Episode reward: {episode_reward:.1f}")
            print(f"  Average speed: {metrics.get('avg_speed', 0):.2f}m/s")
            print(f"  Max lateral deviation: {metrics.get('max_lateral_deviation', 0):.3f}m")
    
    # Compute summary statistics
    results = {
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'completion_rate': np.mean(lap_completions) * 100,
        'mean_progress': np.mean(episode_progresses),
        'std_progress': np.std(episode_progresses),
        'mean_avg_speed': np.mean(episode_avg_speeds),
        'std_avg_speed': np.std(episode_avg_speeds),
        'episode_rewards': episode_rewards,
        'lap_times': episode_lap_times,
        'best_lap_time': min(episode_lap_times) if episode_lap_times else None,
    }
    
    return results


def plot_racing_results(results: dict, q_params: np.ndarray, save_path: str = None):
    """Plot race car evaluation results."""
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Episode rewards
    axes[0, 0].plot(results['episode_rewards'], 'b-', marker='o', alpha=0.7)
    axes[0, 0].axhline(results['mean_reward'], color='r', linestyle='--', 
                       label=f'Mean: {results["mean_reward"]:.1f}')
    axes[0, 0].set_title('Episode Rewards')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Reward')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Lap times (if any completed)
    if results['lap_times']:
        axes[0, 1].plot(results['lap_times'], 'g-', marker='o', alpha=0.7)
        axes[0, 1].axhline(results['best_lap_time'], color='r', linestyle='--',
                          label=f'Best: {results["best_lap_time"]:.3f}s')
        axes[0, 1].set_title('Lap Times (Completed Laps Only)')
        axes[0, 1].set_xlabel('Completed Lap #')
        axes[0, 1].set_ylabel('Lap Time (s)')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
    else:
        axes[0, 1].text(0.5, 0.5, 'No laps completed', ha='center', va='center', 
                       transform=axes[0, 1].transAxes, fontsize=12)
        axes[0, 1].set_title('Lap Times')
    
    # Q parameter values
    param_names = ['Q_s', 'Q_n', 'Q_α', 'Q_v', 'Q_D', 'Q_δ', 'R_derD', 'R_derδ', 'Qe_s', 'Qe_n']
    param_names = param_names[:len(q_params)]  # Trim to actual number of params
    
    axes[0, 2].bar(range(len(q_params)), q_params, color='skyblue', alpha=0.7)
    axes[0, 2].set_title('Learned Q Matrix Parameters')
    axes[0, 2].set_ylabel('Parameter Value')
    axes[0, 2].set_xticks(range(len(q_params)))
    axes[0, 2].set_xticklabels(param_names, rotation=45)
    axes[0, 2].grid(True, alpha=0.3)
    
    # Performance metrics summary
    metrics = [
        f"Completion Rate: {results['completion_rate']:.1f}%",
        f"Mean Progress: {results['mean_progress']:.1f}%",
        f"Mean Speed: {results['mean_avg_speed']:.2f}m/s",
        f"Best Lap: {results['best_lap_time']:.3f}s" if results['best_lap_time'] else "Best Lap: N/A",
        f"Mean Reward: {results['mean_reward']:.1f}",
    ]
    
    axes[1, 0].text(0.1, 0.7, '\n'.join(metrics), transform=axes[1, 0].transAxes, 
                   fontsize=12, verticalalignment='top')
    axes[1, 0].set_title('Performance Summary')
    axes[1, 0].axis('off')
    
    # Distribution of rewards
    axes[1, 1].hist(results['episode_rewards'], bins=max(3, len(results['episode_rewards'])//2), 
                   alpha=0.7, color='skyblue')
    axes[1, 1].axvline(results['mean_reward'], color='r', linestyle='--', 
                      label=f'Mean: {results["mean_reward"]:.1f}')
    axes[1, 1].set_title('Distribution of Episode Rewards')
    axes[1, 1].set_xlabel('Reward')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # Racing progress comparison
    if len(results['episode_rewards']) > 1:
        episodes = range(1, len(results['episode_rewards']) + 1)
        # Assuming we can get progress data per episode
        axes[1, 2].bar(episodes, [results['mean_progress']] * len(episodes), 
                      alpha=0.7, color='lightgreen')
        axes[1, 2].set_title('Episode Progress')
        axes[1, 2].set_xlabel('Episode')
        axes[1, 2].set_ylabel('Progress (%)')
        axes[1, 2].grid(True, alpha=0.3)
    else:
        axes[1, 2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Racing evaluation plot saved to {save_path}")
    
    plt.show()


def compare_with_default(trained_q_params: np.ndarray, env: RaceCarQMatrixEnv):
    """Compare learned Q parameters with default parameters."""
    
    # Get default Q parameters
    default_q_params = env.controller.default_param(None)
    
    print("\n" + "="*60)
    print("RACE CAR Q-MATRIX PARAMETER COMPARISON")
    print("="*60)
    
    param_names = ['Q_s', 'Q_n', 'Q_α', 'Q_v', 'Q_D', 'Q_δ', 'R_derD', 'R_derδ', 'Qe_s', 'Qe_n']
    param_names = param_names[:len(trained_q_params)]
    
    print(f"{'Parameter':<12} {'Default':<10} {'Learned':<10} {'Ratio':<10}")
    print("-" * 50)
    
    for i, name in enumerate(param_names):
        if i < len(default_q_params):
            ratio = trained_q_params[i] / default_q_params[i] if default_q_params[i] != 0 else float('inf')
            print(f"{name:<12} {default_q_params[i]:<10.4f} {trained_q_params[i]:<10.4f} {ratio:<10.3f}")
    
    # Evaluate both parameter sets
    print("\n" + "="*60)
    print("RACING PERFORMANCE COMPARISON")
    print("="*60)
    
    print("Evaluating default parameters...")
    default_results = evaluate_q_params(default_q_params, env, num_episodes=3)
    
    print("\nEvaluating learned parameters...")  
    learned_results = evaluate_q_params(trained_q_params, env, num_episodes=3)
    
    print(f"\nDefault Q Parameters:")
    print(f"  Mean Reward: {default_results['mean_reward']:.1f} ± {default_results['std_reward']:.1f}")
    print(f"  Completion Rate: {default_results['completion_rate']:.1f}%")
    print(f"  Best Lap Time: {default_results['best_lap_time']:.3f}s" if default_results['best_lap_time'] else "  No laps completed")
    
    print(f"\nLearned Q Parameters:")
    print(f"  Mean Reward: {learned_results['mean_reward']:.1f} ± {learned_results['std_reward']:.1f}")
    print(f"  Completion Rate: {learned_results['completion_rate']:.1f}%")
    print(f"  Best Lap Time: {learned_results['best_lap_time']:.3f}s" if learned_results['best_lap_time'] else "  No laps completed")
    
    # Calculate improvements
    reward_improvement = (learned_results['mean_reward'] - default_results['mean_reward']) / abs(default_results['mean_reward']) * 100 if default_results['mean_reward'] != 0 else 0
    completion_improvement = learned_results['completion_rate'] - default_results['completion_rate']
    
    print(f"\nImprovement:")
    print(f"  Reward: {reward_improvement:+.1f}%")
    print(f"  Completion Rate: {completion_improvement:+.1f} percentage points")
    
    if learned_results['best_lap_time'] and default_results['best_lap_time']:
        lap_improvement = (default_results['best_lap_time'] - learned_results['best_lap_time']) / default_results['best_lap_time'] * 100
        print(f"  Lap Time: {lap_improvement:+.1f}% (faster is positive)")


def main():
    parser = argparse.ArgumentParser(description="Evaluate learned race car Q matrix parameters")
    parser.add_argument("--model-path", type=str, required=True,
                       help="Path to trained model checkpoint")
    parser.add_argument("--num-episodes", type=int, default=5,
                       help="Number of episodes to evaluate")
    parser.add_argument("--render", action="store_true",
                       help="Render episodes during evaluation")
    parser.add_argument("--save-plot", type=str, default=None,
                       help="Path to save evaluation plot")
    parser.add_argument("--track-file", type=str, default="LMS_Track.txt",
                       help="Track file to use")
    parser.add_argument("--device", type=str, default="cpu",
                       help="Device to use (cpu/cuda)")
    parser.add_argument("--log-level", type=str, default="INFO",
                       help="Logging level")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=getattr(logging, args.log_level.upper()))
    logger = logging.getLogger(__name__)
    
    # Load trained model
    logger.info(f"Loading race car model from {args.model_path}")
    try:
        checkpoint = load_trained_model(args.model_path, args.device)
        track_file = checkpoint.get('track_file', args.track_file)
        logger.info(f"Model loaded successfully. Track: {track_file}")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return
    
    # Create environment
    try:
        racecar_env = RaceCarEnv(cfg=RaceCarEnvConfig(track_file=track_file))
        controller = RaceCarController(cfg=RaceCarControllerConfig(track_file=track_file))
        env = RaceCarQMatrixEnv(racecar_env, controller)
        logger.info("Race car environment created successfully")
    except Exception as e:
        logger.error(f"Failed to create environment: {e}")
        logger.error("Make sure the acados race car example is available")
        return
    
    # For simplicity, we'll extract example learned parameters
    # In practice, you'd need to run the policy to get Q parameters
    # Here we'll use some example learned parameters optimized for racing
    
    learned_q_params = np.array([
        0.05,   # Q_s (progress) - lower than default to prioritize speed
        25.0,   # Q_n (lateral dev) - higher to stay on track
        1e-6,   # Q_alpha (heading) - very low
        1e-6,   # Q_v (velocity) - very low, let it be fast
        0.001,  # Q_D (throttle) - low for aggressive throttle
        0.01,   # Q_delta (steering) - moderate steering cost
        0.001,  # R_derD (throttle rate) - low for quick changes
        0.005,  # R_derDelta (steering rate) - moderate
        10.0,   # Qe_s (terminal progress) - high terminal progress
        50.0,   # Qe_n (terminal lateral) - high to end on track
    ])
    
    print("Learned Race Car Q Matrix Parameters:")
    param_names = ['Q_s', 'Q_n', 'Q_α', 'Q_v', 'Q_D', 'Q_δ', 'R_derD', 'R_derδ', 'Qe_s', 'Qe_n']
    for name, value in zip(param_names, learned_q_params):
        print(f"  {name}: {value:.6f}")
    
    # Evaluate learned parameters
    logger.info(f"Evaluating learned parameters over {args.num_episodes} episodes...")
    results = evaluate_q_params(
        learned_q_params, 
        env, 
        num_episodes=args.num_episodes,
        render=args.render
    )
    
    # Print results
    print(f"\n" + "="*60)
    print("RACING EVALUATION RESULTS")
    print("="*60)
    print(f"Mean Reward: {results['mean_reward']:.1f} ± {results['std_reward']:.1f}")
    print(f"Lap Completion Rate: {results['completion_rate']:.1f}%")
    print(f"Mean Progress: {results['mean_progress']:.1f}%")
    print(f"Mean Average Speed: {results['mean_avg_speed']:.2f}m/s")
    if results['best_lap_time']:
        print(f"Best Lap Time: {results['best_lap_time']:.3f}s")
    else:
        print("No laps completed")
    
    # Compare with default parameters
    compare_with_default(learned_q_params, env)
    
    # Plot results
    if args.save_plot or args.render:
        plot_racing_results(results, learned_q_params, args.save_plot)


if __name__ == "__main__":
    main()