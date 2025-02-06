from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from conftest import generate_batch_constant

from leap_c.mpc import MPC
from leap_c.util import SX_to_labels


def compute_diff_solve_batch_solve(
    results: Dict[str, Dict[str, Dict[str, np.ndarray]]],
    mpc: MPC,
):
    keys = ["state_value", "state_action_value", "policy"]
    fields = ["value", "gradient"]

    diff_solve_batch_solve = {
        key: {field: np.zeros((mpc.n_batch, mpc.p_global_dim)) for field in fields}
        for key in ["state_value", "state_action_value", "policy"]
    }

    for key in keys:
        for field in fields:
            diff_solve_batch_solve[key][field] = (
                results["_solve"][key][field] - results["_batch_solve"][key][field]
            )

    return diff_solve_batch_solve


def print_median_table(
    diff_solve_batch_solve: dict,
    param_names: list[str],
    keys: list[str] = ["state_value", "state_action_value", "policy"],
    fields: list[str] = ["value", "gradient"],
) -> None:
    """
    Print a formatted table of median absolute differences with parameter names as headers.

    Args:
        diff_solve_batch_solve (dict): Nested dictionary containing differences
        param_names (list[str]): List of parameter names
        keys (list[str]): List of keys to analyze
        fields (list[str]): List of fields to analyze
    """
    # Import pandas for nice table formatting

    pd.set_option("display.float_format", lambda x: "{:.2e}".format(x))

    # Create rows for the table
    rows = []
    for key in keys:
        for field in fields:
            data = diff_solve_batch_solve[key][field]

            # Check for NaN values
            if np.isnan(data).any():
                print(f"\nWarning: NaN values found in {key} - {field}")

            # Calculate median absolute differences
            median_values = np.median(np.abs(data), axis=0)

            # Create row dictionary
            row = {"Component": f"{key} - {field}"}
            row.update(
                {param: value for param, value in zip(param_names, median_values)}
            )
            rows.append(row)

    # Create DataFrame and display
    df = pd.DataFrame(rows)
    df = df.set_index("Component")

    print("\nMedian Absolute Differences:")
    print(df)

    # Additional statistics
    print("\nSummary:")
    print(f"Largest median difference: {df.max().max():.2e}")
    print(f"Smallest median difference: {df.min().min():.2e}")

    # Identify parameters with largest differences
    max_param = df.max().idxmax()
    print(f"Parameter with largest median difference: {max_param}")


def plot_solve_differences(
    results: Dict[str, Dict[str, Dict[str, np.ndarray]]],
    mpc: MPC,
    keys: list[str] = ["state_value", "state_action_value", "policy"],
    fields: list[str] = ["value", "gradient"],
    fig_size: tuple = (15, 10),
    scale="semilog",
) -> plt.Figure:
    """
    Create a grid of plots showing the differences between solve and batch_solve results.

    Args:
        diff_solve_batch_solve (dict): Nested dictionary containing differences
        keys (list[str]): List of keys to plot
        fields (list[str]): List of fields to plot
        fig_size (tuple): Figure size in inches

    Returns:
        matplotlib.figure.Figure: The created figure
    """

    diff_solve_batch_solve = compute_diff_solve_batch_solve(results, mpc)

    # Print summary statistics
    # analyze_nan_ratios(diff_solve_batch_solve)
    # Create subplot grid
    n_rows = len(keys)
    n_cols = len(fields)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=fig_size)

    # If only one row or column, make axes 2D
    if n_rows == 1:
        axes = axes[np.newaxis, :]
    if n_cols == 1:
        axes = axes[:, np.newaxis]

    # Plot each combination of key and field
    for i, key in enumerate(keys):
        for j, field in enumerate(fields):
            ax = axes[i, j]
            data = diff_solve_batch_solve[key][field]

            # Check if data includes nan
            if np.isnan(data).any():
                print(f"NaN values found in {key} - {field}")

            # Create parameter names based on dimension

            # print(f"Plotting {key} - {field} differences:")
            # print("median absolute difference:")
            param_names = SX_to_labels(mpc.ocp_solver.acados_ocp.model.p_global)
            # print(np.median(data, axis=0))

            # Plot using the provided function
            plot_batch_params(
                batch_data=data,
                param_names=param_names,
                title=f"{key} - {field}",
                ylabel="Difference"
                if j == 0
                else None,  # Only add ylabel for first column
                alpha=0.5,
                ax=ax,
                scale=scale,
            )

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Give a super title to the fig
    fig.suptitle("Differences between _solve and _batch_solve results", fontsize=16)
    return fig


def print_difference_statistics(diff_solve_batch_solve: dict) -> None:
    """
    Print summary statistics for the differences.

    Args:
        diff_solve_batch_solve (dict): Nested dictionary containing differences
    """
    for key in diff_solve_batch_solve:
        print(f"\n{key}:")
        for field in diff_solve_batch_solve[key]:
            data = diff_solve_batch_solve[key][field]
            abs_data = np.abs(data)
            print(f"  {field}:")
            print(f"    Max absolute difference: {np.max(abs_data):.2e}")
            print(f"    Mean absolute difference: {np.mean(abs_data):.2e}")
            print(f"    Median absolute difference: {np.median(abs_data):.2e}")


def plot_gradient_comparison_grid(
    results: Dict[str, Dict[str, Dict[str, np.ndarray]]],
    mpc: MPC,
    n_width: int = 4,
) -> plt.Figure:
    """
    Plot gradient comparisons in a grid of subplots.

    Args:
        results (dict): Dictionary containing gradient results
        mpc: MPC object containing model information
        n_width (int): Number of subplots horizontally
    """
    n_params = mpc.p_global_dim

    # Calculate number of rows needed
    n_rows = int(np.ceil(n_params / n_width))

    # Create figure and subplots
    fig, axes = plt.subplots(n_rows, n_width, figsize=(5 * n_width, 4 * n_rows))

    # Flatten axes for easier indexing if there are multiple rows
    if n_rows > 1:
        axes = axes.flatten()
    elif n_rows == 1 and n_width > 1:
        axes = axes.flatten()
    else:
        axes = [axes]

    # Create each subplot
    for i_param in range(n_params):
        ax = axes[i_param]

        # Plot gradients
        ax.plot(
            results["_batch_solve"]["state_value"]["gradient"][:, i_param],
            label="gradient",
        )
        ax.plot(
            results["_batch_solve"]["state_value"]["gradient_fd"][:, i_param],
            label="gradient_fd",
        )

        # Add labels and title
        ax.set_title(SX_to_labels(mpc.ocp_solver.acados_ocp.model.p_global)[i_param])
        ax.set_ylabel("gradient")
        ax.set_xlabel("batch index")
        ax.grid(True)
        ax.legend()

    # Hide any unused subplots
    for i in range(n_params, len(axes)):
        axes[i].set_visible(False)

    # Adjust layout to prevent overlap
    plt.tight_layout()
    return fig


def plot_batch_params(
    batch_data: np.ndarray,
    param_names: list[str] = None,
    title: str = None,
    ylabel: str = None,
    alpha: float = 0.5,
    ax: plt.Axes = None,
    scale: str = "semilog",
) -> plt.Axes:
    """
    Create a plot showing values as dots arranged vertically for each parameter index.

    Args:
        batch_data (np.ndarray): Array of shape (n_batch, n_param) containing the data to plot
        param_names (list, optional): List of parameter names for x-axis labels
        title (str, optional): Plot title
        ylabel (str, optional): Label for y-axis
        alpha (float, optional): Transparency of dots
        ax (matplotlib.axes.Axes, optional): Axes to plot on. If None, current axes will be used.
    """
    if ax is None:
        ax = plt.gca()

    n_batch, n_param = batch_data.shape

    # For each parameter
    for param_idx in range(n_param):
        # Get all batch values for this parameter
        param_values = batch_data[:, param_idx]

        # Create slight random x-offset for better visibility when values overlap
        x = np.random.normal(param_idx, 0.02, size=n_batch)

        # Calculate and plot median
        median_value = np.median(param_values)
        if scale == "semilog":
            ax.semilogy(x, param_values, "o", alpha=alpha, markersize=6)
            ax.semilogy(
                [param_idx - 0.2, param_idx + 0.2],  # x coordinates for horizontal line
                [median_value, median_value],  # y coordinates for horizontal line
                color="red",
                linewidth=2,
                solid_capstyle="butt",
            )
        else:
            ax.plot(x, param_values, "o", alpha=alpha, markersize=6)
            ax.plot(
                [param_idx - 0.2, param_idx + 0.2],  # x coordinates for horizontal line
                [median_value, median_value],  # y coordinates for horizontal line
                color="red",
                linewidth=2,
                solid_capstyle="butt",
            )

    # Set x-axis ticks and labels
    if param_names is not None:
        ax.set_xticks(range(n_param))
        ax.set_xticklabels(param_names, rotation=45)

    # Add labels and title
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)

    # Add grid
    ax.grid(True, axis="y", linestyle="--", alpha=0.7)

    return ax


def plot_gradient_diff_comparison(
    results: Dict[str, Dict[str, Dict[str, np.ndarray]]],
    mpc: MPC,
    figsize: Tuple[float, float] = (15, 8),
    scale="semilog",
) -> plt.Figure:
    """
    Plot gradient differences for different methods and keys in a 3x2 grid.

    Args:
        results (dict): Dictionary containing gradient results
        mpc: MPC object containing model information
        figsize (tuple): Figure size (width, height)
    """
    # Create figure with 3x2 grid
    fig, axes = plt.subplots(2, 3, figsize=figsize)

    # Get parameter names once
    param_names = SX_to_labels(mpc.ocp_solver.acados_ocp.model.p_global)

    # Keys for columns
    keys = ["state_value", "state_action_value", "policy"]

    # Methods for rows
    methods = ["_solve", "_batch_solve"]

    # Create plots
    for col, key in enumerate(keys):
        for row, method in enumerate(methods):
            plot_batch_params(
                results[method][key]["gradient_diff"],
                param_names=param_names,
                ylabel="gradient difference"
                if col == 0
                else "",  # Only show ylabel on leftmost plots
                title=f"{key} via {method}".replace("_", " "),
                ax=axes[row, col],
                scale=scale,
            )

            # Add horizontal line at y=0 for reference
            axes[row, col].axhline(y=0, color="k", linestyle="--", alpha=0.3)

    # Adjust layout to prevent overlap
    plt.tight_layout()
    return fig


def initialize_results(
    mpc: MPC,
    n_col: int,
) -> Dict[str, Dict[str, Dict[str, np.ndarray]]]:
    """Initialize the results dictionary structure."""
    results = {}
    for method in ["_solve", "_batch_solve"]:
        results[method] = {}
        for key in ["policy", "state_value", "state_action_value"]:
            results[method][key] = {
                "value": np.zeros(shape=(mpc.n_batch, n_col)),
                "gradient": np.zeros(shape=(mpc.n_batch, n_col)),
                "gradient_fd": np.zeros(shape=(mpc.n_batch, n_col)),
                "gradient_diff": np.zeros(shape=(mpc.n_batch, n_col)),
            }
    return results


def run_batch_solve(
    mpc: MPC,
    x0: np.ndarray,
    u0: np.ndarray,
    p_global: np.ndarray,
    i_param: int,
    results: Dict[str, Dict[str, Dict[str, np.ndarray]]],
):
    """Execute batch solve computations for all value types."""
    computations = {
        "state_value": lambda: mpc.state_value(state=x0, p_global=p_global, sens=True),
        "state_action_value": lambda: mpc.state_action_value(
            state=x0, action=u0, p_global=p_global, sens=True
        ),
        "policy": lambda: mpc.policy(
            state=x0, p_global=p_global, sens=True, use_adj_sens=True
        ),
    }

    # TODO: This assumes nu = 1 for the policy. Modify to cover the case where nu > 1
    for key, compute_func in computations.items():
        batch_value, batch_gradient, status = compute_func()
        results["_batch_solve"][key]["value"][:, i_param] = (
            batch_value if key != "policy" else batch_value
        ).flatten()
        results["_batch_solve"][key]["gradient"][:, i_param] = (
            batch_gradient[:, i_param]
            if key != "policy"
            else batch_gradient[:, 0, i_param]
        )
        assert np.all(
            status == 0
        ), f"Some samples in the Batch solve did not converge for {key}"


def run_individual_solve(
    mpc: MPC,
    x0: np.ndarray,
    u0: np.ndarray,
    p_global: np.ndarray,
    i_param: int,
    k: int,
    results: Dict[str, Dict[str, Dict[str, np.ndarray]]],
):
    """Execute individual solve computations for all value types."""
    computations = {
        "state_value": lambda: mpc.state_value(
            state=x0[k, :], p_global=p_global[k, :], sens=True
        ),
        "state_action_value": lambda: mpc.state_action_value(
            state=x0[k, :], action=u0[k, :], p_global=p_global[k, :], sens=True
        ),
        "policy": lambda: mpc.policy(
            state=x0[k, :], p_global=p_global[k, :], sens=True, use_adj_sens=True
        ),
    }

    # TODO: This assumes nu = 1 for the policy. Modify to cover the case where nu > 1
    for key, compute_func in computations.items():
        solve_value, solve_gradient, status = compute_func()
        results["_solve"][key]["value"][k, i_param] = solve_value[0]
        results["_solve"][key]["gradient"][k, i_param] = (
            solve_gradient[i_param]
            if key != "policy"
            else solve_gradient.reshape(-1)[i_param]
        )
        assert np.all(status == 0), f"Individual solve did not converge for {key}"


def find_varying_column(arr: np.ndarray) -> np.ndarray:
    """Find the single column that varies across rows."""
    return arr[:, np.where(np.std(arr, axis=0) > 1e-6)[0][0]]


def compute_finite_differences_for_parametric_sensitivities(
    results: Dict[str, Dict[str, Dict[str, np.ndarray]]],
    x_init: np.ndarray,
    u_init: np.ndarray,
    p_global: np.ndarray,
    i_param: int,
):
    """Compute finite differences and gradient differences."""

    # Find the varying value
    for method in ["_solve", "_batch_solve"]:
        for key in ["state_value", "state_action_value", "policy"]:
            results[method][key]["gradient_fd"][:, i_param] = np.gradient(
                results[method][key]["value"][:, i_param],
                p_global[:, i_param],
                edge_order=2,
            )

            results[method][key]["gradient_diff"][:, i_param] = np.abs(
                results[method][key]["gradient_fd"][:, i_param]
                - results[method][key]["gradient"][:, i_param]
            )


def verify_shapes(
    results: Dict[str, Dict[str, Dict[str, np.ndarray]]],
    mpc: MPC,
):
    n_col = results["_solve"]["state_value"]["value"].shape[1]
    # TODO: This assumes nu = 1 for the policy. Modify to cover the case where nu > 1
    """Verify the shapes of all result arrays."""
    for method in ["_solve", "_batch_solve"]:
        for key in ["state_value", "state_action_value", "policy"]:
            assert results[method][key]["value"].shape[0] == mpc.n_batch
            assert results[method][key]["gradient"].shape == (
                mpc.n_batch,
                n_col,
            )
            assert results[method][key]["gradient_fd"].shape == (
                mpc.n_batch,
                n_col,
            )


def validate_mpc_results_on_batch_p_global(
    results: Dict[str, Dict[str, Dict[str, np.ndarray]]],
    mpc: MPC,
    plot: bool = False,
):
    """Validates MPC results by comparing different solving methods and their gradients.

    This function performs several validation checks on MPC results:
    1. Verifies shapes of the result arrays
    2. Compares gradients from different solving methods against finite difference approximations
    3. Checks consistency between single and batch solving methods
    4. Checks for NaN values in the results
    5. Optionally visualizes the results through plots

    Args:
        results (Dict[str, Dict[str, Dict[str, np.ndarray]]]): Nested dictionary containing MPC results
            with following structure:
            - First level: solving method ('_solve' or '_batch_solve')
            - Second level: result type ('state_value', 'state_action_value', 'policy')
            - Third level: specific values ('gradient', 'gradient_fd', etc.)
        mpc (MPC): MPC object containing the problem definition and solver settings
        plot (bool, optional): If True, generates visualization plots. Defaults to False.

    Raises:
        AssertionError: If any of the following conditions are not met:
            - Gradients differ more than tolerance from finite difference approximations
            - Significant differences between solve and batch_solve methods
            - NaN values in results

    Returns:
        None: Function performs validation through assertions and prints statistics
    """
    verify_shapes(
        results=results,
        mpc=mpc,
    )

    tol = {"state_value": 1e-3, "state_action_value": 1e-3, "policy": 1e-2}
    for method in ["_solve", "_batch_solve"]:
        for key in ["state_value", "state_action_value"]:
            assert (
                np.median(
                    np.abs(
                        results[method]["state_value"]["gradient"]
                        - results[method]["state_value"]["gradient_fd"]
                    )
                )
                < tol[key]
            ), f"Method {method} key {key} gradient not close enough, maximal difference is {np.abs(results[method][key]['gradient'] - results[method][key]['gradient_fd']).max()}"

    diff_solve_batch_solve = compute_diff_solve_batch_solve(
        results=results,
        mpc=mpc,
    )

    # Stack diff_solve_batch_solve into a single array and compute the norm
    diff_solve_batch_solve_norm = np.linalg.norm(
        np.stack(
            [
                diff_solve_batch_solve[key][field]
                for key in diff_solve_batch_solve
                for field in diff_solve_batch_solve[key]
            ]
        )
    )

    assert (
        diff_solve_batch_solve_norm == 0.0
    ), f"Norm difference between _solve and _batch_solve should be 0, but is: {diff_solve_batch_solve_norm}"

    for key in diff_solve_batch_solve.keys():
        for field in diff_solve_batch_solve[key].keys():
            assert (
                np.median(np.abs(diff_solve_batch_solve[key][field])) < 1e-5
            ), f"Method {method} key {key} gradient not close enough, maximal difference is {np.abs(results[method][key]['gradient'] - results[method][key]['gradient_fd']).max()}"

    print_difference_statistics(diff_solve_batch_solve)
    keys = ["state_value", "state_action_value", "policy"]
    fields = ["value", "gradient"]
    for i, key in enumerate(keys):
        for j, field in enumerate(fields):
            data = diff_solve_batch_solve[key][field]

            # Check if data includes nan
            assert not np.isnan(data).any(), f"NaN values found in {key} - {field}"

            # Create parameter names based on dimension
            param_names = SX_to_labels(mpc.ocp_solver.acados_ocp.model.p_global)
            print_median_table(diff_solve_batch_solve, param_names, keys, fields)

    # Plot results if requested
    if plot:
        _ = plot_gradient_diff_comparison(results, mpc, scale="semilog")
        _ = plot_gradient_comparison_grid(results, mpc)
        _ = plot_solve_differences(results, mpc, scale="semilog")
        plt.show()


def run_test_mpc_solve_and_batch_solve_on_batch_p_global(
    mpc: MPC,
    p_global: np.ndarray,
    plot: bool = False,
) -> None:
    """Run MPC solver tests comparing batch and individual solutions with parametric sensitivities.

    This function performs a comprehensive test of MPC solutions by:
    1. Running batch solves for multiple parameter sets
    2. Running sequential individual solves for comparison
    3. Computing finite differences for parametric sensitivity validation

    Args:
        mpc (MPC): MPC controller instance to be tested
        p_global (np.ndarray): Global parameters array with shape (n_p, n_horizon, n_param_sets)
        plot (bool, optional): Flag to enable result plotting. Defaults to False.

    Returns:
        None

    Notes:
        - The function initializes solutions with constant values for states and inputs
        - Results are validated by comparing batch and individual solutions
        - Parametric sensitivities are verified using finite differences
    """
    x_init = generate_batch_constant(mpc.ocp.constraints.x0, p_global.shape)
    u_init = generate_batch_constant(
        mpc.ocp.constraints.lbu
        + 0.5 * (mpc.ocp.constraints.ubu - mpc.ocp.constraints.lbu),
        p_global.shape,
    )

    # Initialize results structure
    results = initialize_results(mpc, n_col=p_global.shape[2])

    # Main computation loop
    for i_param in range(p_global.shape[2]):
        # print(SX_to_labels(mpc.ocp_solver.acados_ocp.model.p_global)[i_param])

        # Run batch and individual solves
        run_batch_solve(
            mpc,
            x_init[:, :, i_param],
            u_init[:, :, i_param],
            p_global[:, :, i_param],
            i_param,
            results,
        )
        for k in range(mpc.n_batch):
            run_individual_solve(
                mpc,
                x_init[:, :, i_param],
                u_init[:, :, i_param],
                p_global[:, :, i_param],
                i_param,
                k,
                results,
            )

        # Compute finite differences
        compute_finite_differences_for_parametric_sensitivities(
            results,
            x_init[:, :, i_param],
            u_init[:, :, i_param],
            p_global[:, :, i_param],
            i_param,
        )

    validate_mpc_results_on_batch_p_global(results, mpc, plot)


def test_linear_mpc_parametric_sensitivities(
    learnable_linear_mpc: MPC, linear_mpc_p_global: np.ndarray
):
    run_test_mpc_solve_and_batch_solve_on_batch_p_global(
        learnable_linear_mpc, linear_mpc_p_global, plot=False
    )


if __name__ == "__main__":
    pytest.main([__file__])
    # n_batch = 8
    # mpc = PendulumOnCartMPC(
    #     learnable_params=["M", "m", "g", "l", "Q", "R"], n_batch=n_batch
    # )
    # p_global = generate_batch_variation(
    #     mpc.ocp_solver.acados_ocp.p_global_values, n_batch=n_batch
    # )

    # run_test_mpc_solve_and_batch_solve_on_batch_p_global(mpc, p_global, plot=True)
