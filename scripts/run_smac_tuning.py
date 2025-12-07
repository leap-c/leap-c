"""SMAC-based hyperparameter tuning for controller parameters.

This script uses SMAC3 (Sequential Model-based Algorithm Configuration) from the
Freiburg AutoML group to optimize controller parameters.

SMAC uses Bayesian optimization with random forests as the surrogate model,
making it particularly effective for hyperparameter optimization.

Requirements:
    pip install smac>=2.0.0

Usage:
    python scripts/run_smac_tuning.py --env hvac --controller hvac --n_trials 100
"""

from argparse import ArgumentParser
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from ConfigSpace import ConfigurationSpace, Float
from smac import HyperparameterOptimizationFacade, Scenario
from smac.runhistory.dataclasses import TrialValue

from leap_c.examples import ExampleControllerName, ExampleEnvName, create_controller, create_env
from leap_c.run import default_controller_code_path, default_output_path


@dataclass
class SmacTuningConfig:
    """Configuration for SMAC-based hyperparameter tuning.

    Attributes:
        env: The environment name.
        controller: The controller name.
        n_trials: Number of SMAC trials to run.
        n_rollouts: Number of rollouts per evaluation.
        max_steps: Maximum steps per rollout (None for no limit).
        seed: Random seed.
        deterministic: Whether to use deterministic actions.
        output_dir: Directory to save SMAC results.
        wandb_logger: Whether to log results to wandb.
        wandb_init_kwargs: Keyword arguments for wandb.init().
    """

    env: ExampleEnvName = "hvac"
    controller: ExampleControllerName = "hvac"
    n_trials: int = 100
    n_rollouts: int = 10
    max_steps: int | None = None
    seed: int = 42
    deterministic: bool = True
    output_dir: Path = field(default_factory=lambda: Path("output/smac_tuning"))
    wandb_logger: bool = False
    wandb_init_kwargs: dict[str, Any] = field(default_factory=dict)


class ControllerTuner:
    """Tunes controller parameters using SMAC.

    This class wraps the controller evaluation in a format suitable for SMAC optimization.

    Attributes:
        cfg: The tuning configuration.
        env: The environment instance.
        controller: The controller instance.
        param_space: The parameter space (from gymnasium).
        config_space: The SMAC configuration space.
    """

    def __init__(self, cfg: SmacTuningConfig, reuse_code_dir: Path | None = None) -> None:
        """Initialize the tuner.

        Args:
            cfg: The tuning configuration.
            reuse_code_dir: Directory to reuse compiled code from, if any.
        """
        self.cfg = cfg
        self.device = "cpu"
        self.wandb_run = None

        # Initialize wandb if enabled
        if cfg.wandb_logger:
            try:
                import wandb
                self.wandb_run = wandb.init(**cfg.wandb_init_kwargs)
                print(f"Wandb run initialized: {self.wandb_run.url}")
            except ImportError:
                print("Warning: wandb not installed. Skipping wandb logging.")
                cfg.wandb_logger = False

        # Create environment and controller
        self.env = create_env(cfg.env)
        self.controller = create_controller(cfg.controller, reuse_code_base_dir=reuse_code_dir)

        # Random number generator for evaluation seeds (different each trial to avoid overfitting)
        self.eval_rng = np.random.default_rng(cfg.seed)
        self.trial_counter = 0

        # Get parameter space from controller
        self.param_space = self.controller.param_space

        # Build SMAC configuration space from gymnasium space
        self.config_space = self._build_config_space()

        print(f"Parameter space: {self.param_space}")
        print(f"Configuration space: {self.config_space}")

    def _build_config_space(self) -> ConfigurationSpace:
        """Build SMAC ConfigurationSpace from gymnasium parameter space.

        Returns:
            A SMAC ConfigurationSpace object.
        """
        cs = ConfigurationSpace(seed=self.cfg.seed)

        # Get bounds from gymnasium Box space
        low = self.param_space.low.flatten()
        high = self.param_space.high.flatten()

        # Create hyperparameters for each parameter dimension
        for i, (lo, hi) in enumerate(zip(low, high)):
            # Handle infinite bounds by setting reasonable defaults
            if np.isinf(lo):
                lo = -1e6
            if np.isinf(hi):
                hi = 1e6

            hp = Float(
                name=f"param_{i}",
                bounds=(float(lo), float(hi)),
                default=float((lo + hi) / 2),
            )
            cs.add(hp)

        return cs

    def _config_to_param(self, config: dict[str, float]) -> torch.Tensor:
        """Convert SMAC configuration to parameter tensor.

        Args:
            config: SMAC configuration dictionary.

        Returns:
            Parameter tensor of shape (1, n_params).
        """
        n_params = len(config)
        param = np.array([config[f"param_{i}"] for i in range(n_params)])
        return torch.from_numpy(param).unsqueeze(0).to(self.device)

    def evaluate(self, config: dict[str, float], seed: int = 0) -> float:
        """Evaluate a parameter configuration.

        Args:
            config: SMAC configuration to evaluate.
            seed: Random seed for this evaluation (used as base, but we add randomness
                  to avoid overfitting to specific scenarios).

        Returns:
            The negative mean reward (SMAC minimizes, so we negate).
        """
        param = self._config_to_param(config)

        total_reward = 0.0
        n_successful = 0
        
        # Generate random seeds for this trial to avoid overfitting to specific scenarios
        # Each trial gets different random scenarios
        self.trial_counter += 1
        trial_seeds = self.eval_rng.integers(0, 2**31, size=self.cfg.n_rollouts)

        for rollout_idx in range(self.cfg.n_rollouts):
            rollout_seed = int(trial_seeds[rollout_idx])
            obs, info = self.env.reset(seed=rollout_seed)
            
            done = False
            truncated = False
            episode_reward = 0.0
            step = 0
            ctx = None

            while not (done or truncated):
                # Prepare observation
                obs_tensor = torch.from_numpy(obs).unsqueeze(0).to(self.device)

                # Get action from controller
                try:
                    ctx, action = self.controller(obs_tensor, param, ctx=ctx)
                    action = action.cpu().numpy()[0]
                except Exception as e:
                    print(f"Controller error: {e}")
                    break

                # Step environment
                obs, reward, done, truncated, info = self.env.step(action)
                episode_reward += reward
                step += 1

                # Check max steps
                if self.cfg.max_steps is not None and step >= self.cfg.max_steps:
                    break

            total_reward += episode_reward
            n_successful += 1

        if n_successful == 0:
            return 1e10  # Return large cost if all rollouts failed

        mean_reward = total_reward / n_successful

        # SMAC minimizes, so return negative reward
        return -mean_reward

    def _validate_with_fixed_seeds(
        self, config: dict[str, float], n_rollouts: int, base_seed: int
    ) -> float:
        """Validate a configuration with fixed seeds for reproducibility.

        Unlike evaluate(), this uses deterministic seeds for fair comparison.

        Args:
            config: SMAC configuration to validate.
            n_rollouts: Number of rollouts to run.
            base_seed: Base seed for reproducible evaluation.

        Returns:
            The mean reward over all rollouts.
        """
        param = self._config_to_param(config)

        total_reward = 0.0
        n_successful = 0

        for rollout_idx in range(n_rollouts):
            rollout_seed = base_seed + rollout_idx
            obs, info = self.env.reset(seed=rollout_seed)

            done = False
            truncated = False
            episode_reward = 0.0
            step = 0
            ctx = None

            while not (done or truncated):
                obs_tensor = torch.from_numpy(obs).unsqueeze(0).to(self.device)

                try:
                    ctx, action = self.controller(obs_tensor, param, ctx=ctx)
                    action = action.cpu().numpy()[0]
                except Exception as e:
                    print(f"Controller error: {e}")
                    break

                obs, reward, done, truncated, info = self.env.step(action)
                episode_reward += reward
                step += 1

                if self.cfg.max_steps is not None and step >= self.cfg.max_steps:
                    break

            total_reward += episode_reward
            n_successful += 1

        if n_successful == 0:
            return -1e10  # Return large negative reward if all rollouts failed

        return total_reward / n_successful

    def run(self) -> dict[str, Any]:
        """Run the SMAC optimization.

        Returns:
            Dictionary containing the best configuration and optimization history.
        """
        # Create output directory
        self.cfg.output_dir.mkdir(parents=True, exist_ok=True)

        # Define SMAC scenario
        scenario = Scenario(
            configspace=self.config_space,
            name="controller_tuning",
            output_directory=self.cfg.output_dir,
            deterministic=self.cfg.deterministic,
            n_trials=self.cfg.n_trials,
            seed=self.cfg.seed,
        )

        # Create SMAC facade with Hyperparameter Optimization
        smac = HyperparameterOptimizationFacade(
            scenario=scenario,
            target_function=self.evaluate,
            overwrite=True,
        )

        # Run optimization
        print(f"\nStarting SMAC optimization with {self.cfg.n_trials} trials...")
        print(f"Using random seeds per trial to avoid overfitting to specific scenarios.")
        incumbent = smac.optimize()

        # Get best configuration
        best_config = dict(incumbent)
        best_param = self._config_to_param(best_config)

        # Final validation with FIXED seeds for reproducibility (different from training)
        # Use more rollouts for reliable final estimate
        print("\nFinal validation with fixed seeds (for reproducibility)...")
        n_val_rollouts = max(self.cfg.n_rollouts * 5, 50)  # Use more rollouts for final validation
        final_reward = self._validate_with_fixed_seeds(
            best_config, 
            n_rollouts=n_val_rollouts,
            base_seed=self.cfg.seed + 100000  # Different seed range than training
        )

        print(f"\n{'='*60}")
        print("SMAC Optimization Complete!")
        print(f"{'='*60}")
        print(f"Best configuration: {best_config}")
        print(f"Best parameter tensor: {best_param}")
        print(f"Best validation reward (over {n_val_rollouts} rollouts): {final_reward:.4f}")
        print(f"Results saved to: {self.cfg.output_dir}")

        # Log best result to wandb
        if self.cfg.wandb_logger and self.wandb_run is not None:
            import wandb
            wandb.log({
                "best_validation_reward": final_reward,
                "n_trials": self.cfg.n_trials,
            })
            # Log best parameters as a summary
            wandb.run.summary["best_validation_reward"] = final_reward
            wandb.run.summary["best_config"] = best_config
            for i, (key, value) in enumerate(best_config.items()):
                wandb.run.summary[f"best_{key}"] = value
            print("Best result logged to wandb.")

        return {
            "best_config": best_config,
            "best_param": best_param.numpy(),
            "best_reward": final_reward,
            "scenario": scenario,
        }

    def close(self) -> None:
        """Clean up resources."""
        self.env.close()
        
        # Finish wandb run
        if self.cfg.wandb_logger and self.wandb_run is not None:
            import wandb
            wandb.finish()


def main() -> None:
    """Main entry point for SMAC tuning."""
    parser = ArgumentParser(description="SMAC-based hyperparameter tuning for controllers")
    parser.add_argument("--env", type=str, default="hvac", help="Environment name")
    parser.add_argument("--controller", type=str, default=None, help="Controller name (defaults to env)")
    parser.add_argument("--n_trials", type=int, default=100, help="Number of SMAC trials")
    parser.add_argument("--n_rollouts", type=int, default=10, help="Rollouts per evaluation")
    parser.add_argument("--max_steps", type=int, default=None, help="Max steps per rollout")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output_dir", type=Path, default=None, help="Output directory")
    parser.add_argument(
        "-r", "--reuse_code", action="store_true",
        help="Reuse compiled code for faster startup"
    )
    parser.add_argument("--reuse_code_dir", type=Path, default=None)
    parser.add_argument("--use-wandb", action="store_true", help="Log results to wandb")
    parser.add_argument("--wandb-entity", type=str, default=None, help="Wandb entity")
    parser.add_argument("--wandb-project", type=str, default="leap-c-smac", help="Wandb project")
    args = parser.parse_args()

    # Set defaults
    controller = args.controller if args.controller else args.env
    output_dir = args.output_dir if args.output_dir else default_output_path(
        seed=args.seed, tags=["smac", args.env, controller]
    )

    cfg = SmacTuningConfig(
        env=args.env,
        controller=controller,
        n_trials=args.n_trials,
        n_rollouts=args.n_rollouts,
        max_steps=args.max_steps,
        seed=args.seed,
        output_dir=output_dir,
        wandb_logger=args.use_wandb,
        wandb_init_kwargs={
            "entity": args.wandb_entity,
            "project": args.wandb_project,
            "name": f"smac_{args.env}_{controller}_seed{args.seed}",
            "config": {
                "env": args.env,
                "controller": controller,
                "n_trials": args.n_trials,
                "n_rollouts": args.n_rollouts,
                "max_steps": args.max_steps,
                "seed": args.seed,
            },
        } if args.use_wandb else {},
    )

    # Determine code reuse directory
    if args.reuse_code and args.reuse_code_dir is None:
        reuse_code_dir = default_controller_code_path()
    elif args.reuse_code_dir is not None:
        reuse_code_dir = args.reuse_code_dir
    else:
        reuse_code_dir = None

    # Run tuning
    tuner = ControllerTuner(cfg, reuse_code_dir=reuse_code_dir)
    try:
        results = tuner.run()
        
        # Save results
        results_file = cfg.output_dir / "best_params.npz"
        np.savez(
            results_file,
            best_config=results["best_config"],
            best_param=results["best_param"],
            best_reward=results["best_reward"],
        )
        print(f"Best parameters saved to: {results_file}")
        
    finally:
        tuner.close()


if __name__ == "__main__":
    main()
