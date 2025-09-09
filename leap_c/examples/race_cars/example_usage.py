#!/usr/bin/env python3
"""
Example usage of race car model with learnable Q matrix parameters.

This script demonstrates:
1. Creating a race car environment using the acados race car model
2. Setting up an MPC controller with learnable Q matrix
3. Running racing simulations with different Q parameter settings
4. Showing how the Q parameters affect racing performance
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import sys
import os

from leap_c.examples.race_cars.env import RaceCarEnv, RaceCarEnvConfig
from leap_c.examples.race_cars.controller import RaceCarController, RaceCarControllerConfig

# Add the race car example path to sys.path
race_car_path = os.path.join(os.path.dirname(__file__), '../../../external/acados/examples/acados_python/race_cars')
if race_car_path not in sys.path:
    sys.path.insert(0, race_car_path)

try:
    from plotFcn import plotTrackProj
    from time2spatial import transformProj2Orig
    from tracks.readDataFcn import getTrack
except ImportError as e:
    print(f"Warning: Could not import race car modules: {e}")
    print("Make sure the acados race car example is available")


def run_racing_simulation(controller: RaceCarController, env: RaceCarEnv, 
                         q_params: np.ndarray, max_steps: int = 1000):
    """
    Run a racing simulation with given Q matrix parameters.
    
    Args:
        controller: MPC controller
        env: Race car environment  
        q_params: Q matrix parameters
        max_steps: Maximum number of simulation steps
        
    Returns:
        Dictionary with simulation results
    """
    # Reset environment
    obs, _ = env.reset()
    
    # Convert Q parameters to torch tensor
    q_params_tensor = torch.tensor(q_params, dtype=torch.float32).unsqueeze(0)
    
    # Storage for results
    states = [obs.copy()]
    controls = []
    rewards = []
    lap_complete = False
    
    # MPC context
    mpc_ctx = None
    
    print(f"Starting race simulation...")
    
    for step in range(max_steps):
        # Get current observation as tensor
        current_obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        
        try:
            # Get control from MPC with learned Q parameters
            mpc_ctx, control_tensor = controller.forward(current_obs, q_params_tensor, mpc_ctx)
            control = control_tensor.detach().cpu().numpy()[0]
        except Exception as e:
            print(f"MPC failed at step {step}: {e}")
            control = np.array([0.0, 0.0])  # Safe fallback
        
        # Apply control to environment
        next_obs, reward, terminated, truncated, info = env.step(control)
        
        # Store results
        states.append(next_obs.copy())
        controls.append(control.copy())
        rewards.append(reward)
        
        # Check for lap completion
        if controller.is_lap_complete(next_obs):
            lap_complete = True
            lap_time = env.t
            print(f"Lap completed in {lap_time:.3f}s!")
            break
        
        # Progress update
        if step % 100 == 0:
            progress = controller.get_lap_progress(next_obs)
            print(f"Step {step}: Progress {progress:.1f}%, Speed {next_obs[3]:.2f}m/s")
        
        obs = next_obs
        
        if terminated or truncated:
            print(f"Simulation ended: {info.get('task', {}).get('reason', 'unknown')}")
            break
    
    # Get racing metrics
    racing_metrics = controller.get_racing_metrics(states, env.cfg.dt)
    
    return {
        'states': np.array(states),
        'controls': np.array(controls) if controls else np.array([]),
        'rewards': np.array(rewards),
        'lap_complete': lap_complete,
        'total_reward': np.sum(rewards),
        'final_progress': controller.get_lap_progress(states[-1]),
        'racing_metrics': racing_metrics,
        'simulation_time': env.t,
    }


def plot_racing_comparison(results_list: list, labels: list, q_params_list: list, track_file: str = "LMS_Track.txt"):
    """Plot comparison of different Q parameter settings for racing."""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
    
    # Load track data for visualization
    try:
        Sref, Xref, Yref, Psiref, kapparef = getTrack(track_file)
        track_available = True
    except:
        track_available = False
        print("Warning: Could not load track data for visualization")
    
    # Plot trajectories in spatial coordinates
    if track_available:
        # Plot track boundaries and center line
        track_width = 0.12
        Xboundleft = Xref - track_width * np.sin(Psiref)
        Yboundleft = Yref + track_width * np.cos(Psiref)
        Xboundright = Xref + track_width * np.sin(Psiref)
        Yboundright = Yref - track_width * np.cos(Psiref)
        
        axes[0, 0].plot(Xref, Yref, 'k--', linewidth=2, label='Center line')
        axes[0, 0].plot(Xboundleft, Yboundleft, 'k-', linewidth=1)
        axes[0, 0].plot(Xboundright, Yboundright, 'k-', linewidth=1)
    
    for i, (results, label, color) in enumerate(zip(results_list, labels, colors[:len(results_list)])):
        states = results['states']
        s_vals = states[:, 0]  # progress
        n_vals = states[:, 1]  # lateral deviation
        v_vals = states[:, 3]  # velocity
        
        # Transform to X-Y coordinates if possible
        if track_available:
            try:
                x_vals, y_vals, _, _ = transformProj2Orig(s_vals, n_vals, states[:, 2], v_vals, track_file)
                axes[0, 0].plot(x_vals, y_vals, color=color, label=f'{label} (Progress: {results["final_progress"]:.1f}%)', alpha=0.8)
                axes[0, 0].scatter(x_vals[-1], y_vals[-1], color=color, s=100, marker='x')  # End point
            except:
                # Fallback: plot in s-n coordinates
                axes[0, 0].plot(s_vals, n_vals, color=color, label=f'{label} (s-n coords)', alpha=0.8)
        else:
            axes[0, 0].plot(s_vals, n_vals, color=color, label=f'{label}', alpha=0.8)
    
    axes[0, 0].set_title('Racing Trajectories')
    axes[0, 0].set_xlabel('X Position (m)' if track_available else 'Progress s (m)')
    axes[0, 0].set_ylabel('Y Position (m)' if track_available else 'Lateral n (m)')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    axes[0, 0].set_aspect('equal')
    
    # Plot velocity profiles
    for i, (results, label, color) in enumerate(zip(results_list, labels, colors[:len(results_list)])):
        states = results['states']
        time_vals = np.linspace(0, results['simulation_time'], len(states))
        axes[0, 1].plot(time_vals, states[:, 3], color=color, label=label, alpha=0.8)
    
    axes[0, 1].set_title('Velocity Profiles')
    axes[0, 1].set_xlabel('Time (s)')
    axes[0, 1].set_ylabel('Velocity (m/s)')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Plot control inputs (throttle rate)
    for i, (results, label, color) in enumerate(zip(results_list, labels, colors[:len(results_list)])):
        controls = results['controls']
        if len(controls) > 0:
            time_vals = np.linspace(0, results['simulation_time'], len(controls))
            axes[0, 2].plot(time_vals, controls[:, 0], color=color, label=f'{label} (throttle)', alpha=0.8)
    
    axes[0, 2].set_title('Throttle Control Rate')
    axes[0, 2].set_xlabel('Time (s)')
    axes[0, 2].set_ylabel('Throttle Rate')
    axes[0, 2].legend()
    axes[0, 2].grid(True)
    
    # Plot Q parameter comparison
    param_names = ['Q_s', 'Q_n', 'Q_α', 'Q_v', 'Q_D', 'Q_δ', 'R_derD', 'R_derδ', 'Qe_s', 'Qe_n']
    n_params = len(q_params_list[0]) if q_params_list else 0
    param_names = param_names[:n_params]
    
    x = np.arange(len(param_names))
    width = 0.15
    
    for i, (q_params, label, color) in enumerate(zip(q_params_list, labels, colors[:len(q_params_list)])):
        offset = (i - len(q_params_list)/2 + 0.5) * width
        axes[1, 0].bar(x + offset, q_params, width, label=label, color=color, alpha=0.7)
    
    axes[1, 0].set_title('Q Matrix Parameter Comparison')
    axes[1, 0].set_ylabel('Parameter Value')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(param_names, rotation=45)
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_yscale('log')  # Log scale for better visibility
    
    # Performance metrics comparison
    metrics_names = ['Total Reward', 'Final Progress (%)', 'Avg Speed (m/s)', 'Lap Time (s)']
    
    for i, (results, label) in enumerate(zip(results_list, labels)):
        metrics_values = [
            results['total_reward'],
            results['final_progress'],
            results['racing_metrics'].get('avg_speed', 0),
            results['racing_metrics'].get('lap_time', 0) if results['lap_complete'] else 0,
        ]
        
        x_pos = np.arange(len(metrics_names))
        axes[1, 1].bar(x_pos + i*0.2, metrics_values, 0.2, label=label, alpha=0.7)
    
    axes[1, 1].set_title('Racing Performance Metrics')
    axes[1, 1].set_xticks(x_pos + 0.2)
    axes[1, 1].set_xticklabels(metrics_names, rotation=45)
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # Progress over time
    for i, (results, label, color) in enumerate(zip(results_list, labels, colors[:len(results_list)])):
        states = results['states']
        time_vals = np.linspace(0, results['simulation_time'], len(states))
        progress_vals = [controller.get_lap_progress(state) for state in states]
        axes[1, 2].plot(time_vals, progress_vals, color=color, label=label, alpha=0.8)
    
    axes[1, 2].set_title('Lap Progress Over Time')
    axes[1, 2].set_xlabel('Time (s)')
    axes[1, 2].set_ylabel('Progress (%)')
    axes[1, 2].legend()
    axes[1, 2].grid(True)
    axes[1, 2].set_ylim(0, 110)
    
    plt.tight_layout()
    plt.show()


def main():
    print("Race Car MPC with Learnable Q Matrix - Example Usage")
    print("=" * 70)
    
    track_file = "LMS_Track.txt"
    
    # Create environment and controller
    try:
        env_config = RaceCarEnvConfig(track_file=track_file, max_time=30.0)
        controller_config = RaceCarControllerConfig(
            track_file=track_file, 
            N_horizon=50, 
            T_horizon=1.0
        )
        
        env = RaceCarEnv(cfg=env_config)
        controller = RaceCarController(cfg=controller_config)
        
        print(f"Race car environment created successfully")
        print(f"Track: {track_file}")
        print(f"MPC horizon: {controller_config.N_horizon} steps, {controller_config.T_horizon} seconds")
        print(f"Parameter space: {controller.param_space}")
        
    except Exception as e:
        print(f"Error creating race car environment: {e}")
        print("Make sure the acados race car example is available")
        return
    
    # Test different Q parameter settings for racing
    print("\nTesting different Q parameter configurations for racing...")
    
    # Default parameters
    default_params = controller.default_param(None)
    print(f"Default parameters shape: {default_params.shape}")
    
    # Conservative racing parameters (prioritize safety)
    conservative_params = np.array([
        0.1,    # Q_s - moderate progress weight
        50.0,   # Q_n - high lateral deviation penalty (stay on track!)
        1e-4,   # Q_alpha - low heading penalty
        1e-4,   # Q_v - low velocity penalty
        0.01,   # Q_D - moderate throttle penalty
        0.02,   # Q_delta - moderate steering penalty
        0.01,   # R_derD - moderate throttle rate penalty
        0.02,   # R_derDelta - moderate steering rate penalty
        20.0,   # Qe_s - high terminal progress
        100.0,  # Qe_n - very high terminal lateral penalty
    ])
    
    # Aggressive racing parameters (prioritize speed)
    aggressive_params = np.array([
        0.01,   # Q_s - low progress weight (let speed drive progress)
        5.0,    # Q_n - lower lateral penalty (more risk taking)
        1e-6,   # Q_alpha - very low heading penalty
        1e-6,   # Q_v - very low velocity penalty (go fast!)
        0.001,  # Q_D - very low throttle penalty
        0.005,  # Q_delta - low steering penalty
        0.001,  # R_derD - very low throttle rate penalty
        0.005,  # R_derDelta - low steering rate penalty
        5.0,    # Qe_s - moderate terminal progress
        20.0,   # Qe_n - moderate terminal lateral penalty
    ])
    
    # Balanced racing parameters
    balanced_params = np.array([
        0.05,   # Q_s - balanced progress weight
        15.0,   # Q_n - balanced lateral penalty
        1e-5,   # Q_alpha - very low heading penalty
        1e-5,   # Q_v - very low velocity penalty
        0.005,  # Q_D - low throttle penalty
        0.01,   # Q_delta - low steering penalty
        0.005,  # R_derD - low throttle rate penalty
        0.01,   # R_derDelta - low steering rate penalty
        10.0,   # Qe_s - balanced terminal progress
        50.0,   # Qe_n - high terminal lateral penalty
    ])
    
    q_params_list = [default_params, conservative_params, aggressive_params, balanced_params]
    labels = ['Default', 'Conservative Racing', 'Aggressive Racing', 'Balanced Racing']
    
    # Run simulations
    results_list = []
    for q_params, label in zip(q_params_list, labels):
        print(f"\nRunning racing simulation with {label} parameters...")
        try:
            results = run_racing_simulation(controller, env, q_params, max_steps=1500)
            results_list.append(results)
            
            print(f"  Total reward: {results['total_reward']:.1f}")
            print(f"  Final progress: {results['final_progress']:.1f}%")
            print(f"  Lap completed: {results['lap_complete']}")
            if results['lap_complete']:
                lap_time = results['racing_metrics'].get('lap_time', 0)
                print(f"  Lap time: {lap_time:.3f}s")
                avg_speed = results['racing_metrics'].get('avg_speed', 0)
                print(f"  Average speed: {avg_speed:.2f}m/s")
            
        except Exception as e:
            print(f"  Simulation failed: {e}")
            # Create dummy results to maintain list consistency
            results_list.append({
                'states': np.array([[0,0,0,0,0,0]]),
                'controls': np.array([]),
                'rewards': np.array([0]),
                'lap_complete': False,
                'total_reward': 0,
                'final_progress': 0,
                'racing_metrics': {},
                'simulation_time': 0,
            })
    
    # Plot comparison
    if results_list:
        print("\nGenerating racing comparison plots...")
        try:
            plot_racing_comparison(results_list, labels, q_params_list, track_file)
        except Exception as e:
            print(f"Plotting failed: {e}")
    
    print("\nRace car example completed!")
    print("\nKey insights:")
    print("- Conservative params: Better track following, potentially slower lap times")
    print("- Aggressive params: Higher speeds but risk going off track")  
    print("- Balanced params: Good compromise between speed and safety")
    print("\nSAC can learn optimal Q parameters by balancing these trade-offs!")
    print("The goal is to find parameters that maximize lap completion rate")
    print("while minimizing lap times - a challenging multi-objective optimization!")


if __name__ == "__main__":
    main()