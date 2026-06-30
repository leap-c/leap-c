"""Welcome to leap-c!

This notebook demonstrates solving a CartPole MPC problem using leap-c's
differentiable acados layer.
"""

import marimo

__generated_with = "0.13.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # leap-c — CartPole MPC

        This notebook solves a CartPole optimal control problem using
        `leap-c`'s differentiable acados wrapper.

        The CartPole is a classic control problem: balance a pole on a
        moving cart by applying horizontal forces.
        """
    )
    return


@app.cell
def _():
    import torch

    from leap_c.examples import create_planner
    return create_planner, torch


@app.cell
def _(create_planner):
    # Create the CartPole planner with default configuration.
    # This builds the acados OCP, generates C code, and compiles the solver.
    planner = create_planner("cartpole")
    return (planner,)


@app.cell
def _(mo, planner, torch):
    # Initial state: [cart_pos, pole_angle, cart_vel, pole_angular_vel]
    # Pole is slightly off upright (pi radians).
    x0 = torch.tensor([[0.0, 3.0, 0.0, 0.0]])

    # Solve the MPC with default parameters.
    # Returns: context, first action, state trajectory, control trajectory, value.
    ctx, u0, x_traj, u_traj, value = planner.forward(obs=x0)

    mo.md(
        f"""
        **Solver status:** `{ctx.status.tolist()}` (0 = success)

        **First action (force):** `{u0.tolist()}`

        **MPC value:** `{value.item():.4f}`
        """
    )
    return ctx, u_traj, value, x0, x_traj, u0


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Planned trajectory

        The plots below show the state and control trajectories predicted by the
        MPC solver over the full horizon.
        """
    )
    return


@app.cell
def _(x_traj):
    import matplotlib.pyplot as plt

    # x_traj shape: (batch, horizon+1, state_dim)
    # State: [cart_pos, pole_angle, cart_vel, pole_angular_vel]
    traj = x_traj[0].detach().numpy()
    n_state_steps = traj.shape[0]
    state_time = [i * 0.25 / 5 for i in range(n_state_steps)]  # T_horizon/N_horizon

    state_fig, state_axes = plt.subplots(2, 2, figsize=(10, 6))
    state_labels = [
        "Cart Position [m]",
        "Pole Angle [rad]",
        "Cart Velocity [m/s]",
        "Pole Ang. Vel. [rad/s]",
    ]

    for ax, label, data in zip(state_axes.flat, state_labels, traj.T):
        ax.plot(state_time, data, "-o", markersize=3)
        ax.set_xlabel("Time [s]")
        ax.set_ylabel(label)
        ax.grid(True)

    state_fig.suptitle("Planned State Trajectory")
    state_fig.tight_layout()
    state_fig
    return plt, state_fig


@app.cell
def _(plt, u_traj):
    # u_traj shape: (batch, horizon, control_dim)
    # Control: [force on cart]
    u = u_traj[0].detach().numpy()
    n_control_steps = u.shape[0]
    control_time = [i * 0.25 / 5 for i in range(n_control_steps)]

    control_fig, control_ax = plt.subplots(figsize=(8, 3))
    control_ax.plot(control_time, u, "-o", markersize=3, color="tab:orange")
    control_ax.set_xlabel("Time [s]")
    control_ax.set_ylabel("Force [N]")
    control_ax.set_title("Planned Control Trajectory")
    control_ax.grid(True)
    control_fig.tight_layout()
    control_fig
    return (control_fig,)


if __name__ == "__main__":
    app.run()
