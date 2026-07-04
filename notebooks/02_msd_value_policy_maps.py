"""Part 2 — the MPC as a value function and a policy.

Because the mass-spring-damper state is only two-dimensional, we can solve the
MPC from a whole grid of initial states in one batched call and plot the value
function and the policy as full 2-D maps — and watch how a physical parameter
(the spring stiffness) reshapes both.

The OCP builder is imported from ``nb_utils.msd``; it is the exact function
built step by step in ``01_msd_build_and_solve.py``.
"""

import marimo

__generated_with = "0.23.11"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 02 — value function and policy over the state space

    An MPC controller defines two functions of the initial state $x_0$:

    - the **value function** $V(x_0)$ — the optimal cost-to-go, and
    - the **policy** $\pi(x_0) = u_0^{*}(x_0)$ — the first optimal action.

    Batched solving makes both cheap to compute and visualise: a grid of initial states
    goes in as one batch, one map comes out per output. This notebook adds a
    twist: we solve the grid for **several spring stiffnesses at once** (grid
    × stiffness is still just one batch) and use a slider to explore the
    resulting maps.
    """)
    return


@app.cell
def _():
    import matplotlib.pyplot as plt
    import numpy as np
    import torch

    from nb_utils.msd import build_msd_ocp

    from leap_c.torch import AcadosDiffMpcTorch

    torch.set_default_dtype(torch.float64)
    return AcadosDiffMpcTorch, build_msd_ocp, np, plt, torch


@app.cell(hide_code=True)
def _(mo):
    mo.image(
        src=str(mo.notebook_dir() / "assets" / "mass_spring_damper.svg"),
        width=340,
        caption=(
            "Same system as notebook 01: state x = [p, v], control u = F. "
            "The stiffness k of the spring is one of the registered parameters."
        ),
    )
    return


@app.cell
def _(AcadosDiffMpcTorch, build_msd_ocp, np, torch):
    N_HORIZON = 50
    DT = 0.1
    GRID_N = 8  # maps are evaluated on a GRID_N x GRID_N grid of initial states
    stiffness_values = np.linspace(0.5, 4.0, 5)  # default k = 2.0 sits mid-sweep

    ocp, manager = build_msd_ocp(N_horizon=N_HORIZON, dt=DT)

    # One batch holds the full grid for every stiffness value.
    diff_mpc = AcadosDiffMpcTorch(
        ocp,
        manager,
        dtype=torch.float64,
        n_batch_init=GRID_N * GRID_N * len(stiffness_values),
        verbose=False,
    )
    return GRID_N, diff_mpc, stiffness_values


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## One batched solve for all maps

    The batch is laid out as `stiffness × grid`: the grid is repeated once
    per stiffness value, and the `stiffness` parameter column repeats each
    value over its grid block.
    """)
    return


@app.cell
def _(GRID_N, diff_mpc, np, stiffness_values, torch):
    axis = np.linspace(-1.8, 1.8, GRID_N)
    pos_grid, vel_grid = np.meshgrid(axis, axis)  # (G, G) each
    _grid = torch.tensor(
        np.stack([pos_grid.ravel(), vel_grid.ravel()], axis=1)
    )  # (G*G, 2)

    _K = len(stiffness_values)
    _x0_batch = _grid.repeat(_K, 1)  # (K*G*G, 2)
    _k_column = torch.tensor(np.repeat(stiffness_values, GRID_N * GRID_N)).reshape(-1, 1)

    _, _u0, _, _, _value = diff_mpc(x0=_x0_batch, params={"stiffness": _k_column})

    value_maps = _value.detach().numpy().reshape(_K, GRID_N, GRID_N)
    policy_maps = _u0.detach().numpy().reshape(_K, GRID_N, GRID_N)
    return policy_maps, pos_grid, value_maps, vel_grid


@app.cell
def _(mo, stiffness_values):
    k_slider = mo.ui.slider(
        start=0,
        stop=len(stiffness_values) - 1,
        step=1,
        value=len(stiffness_values) // 2,
        label="sweep index for the spring stiffness k",
        show_value=True,
    )
    return (k_slider,)


@app.cell
def _(
    k_slider,
    mo,
    np,
    plt,
    policy_maps,
    pos_grid,
    stiffness_values,
    value_maps,
    vel_grid,
):
    # No solve here — the slider picks one precomputed (value, policy) map pair.
    _i = k_slider.value

    # Shared contour levels across the whole sweep, so colors stay comparable
    # while dragging instead of rescaling on every step.
    _v_levels = np.linspace(value_maps.min(), value_maps.max(), 30)
    _p_levels = np.linspace(policy_maps.min(), policy_maps.max(), 30)

    map_fig, map_axes = plt.subplots(1, 2, figsize=(11, 4.5))

    _c0 = map_axes[0].contourf(
        pos_grid, vel_grid, value_maps[_i], levels=_v_levels, cmap="viridis"
    )
    map_fig.colorbar(_c0, ax=map_axes[0])
    map_axes[0].set_title("Value function V(x₀)")

    _c1 = map_axes[1].contourf(
        pos_grid, vel_grid, policy_maps[_i], levels=_p_levels, cmap="coolwarm"
    )
    map_fig.colorbar(_c1, ax=map_axes[1], label="Force F₀ [N]")
    map_axes[1].set_title("Policy π(x₀) = F₀*(x₀)")

    for ax_m in map_axes:
        ax_m.set_xlabel("Position [m]")
        ax_m.set_ylabel("Velocity [m/s]")
    map_fig.suptitle(f"Spring stiffness k = {stiffness_values[_i]:.2f}")
    map_fig.tight_layout()
    mo.vstack([k_slider, map_fig])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Reading the maps

    - The **value function** is a bowl centred at the origin: the further the
      initial state is from rest, the more cost it takes to get back.
    - The **policy** is a feedback law that pushes against displacement and
      velocity, saturating at the $\pm 0.5$ force limit — visible as the
      flat dark-red/dark-blue regions.
    - A **stiffer spring** (drag the slider right) does part of the
      controller's job for free — but also stores more energy at a given
      displacement, which reshapes both the bowl and the saturation regions.

    **Next:** `03_msd_sensitivities.py` differentiates these outputs with
    respect to the parameters.
    """)
    return


if __name__ == "__main__":
    app.run()
