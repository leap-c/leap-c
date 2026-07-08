"""Part 5 — batched solves and forecasts: one call, many problems.

The batch dimension is the workhorse of AcadosDiffMpcTorch: many initial
states, many parameter sets, many forecast windows — one solver call. This
notebook slides a synthetic day of weather/price forecasts through the R1C1
heating MPC, reads value and policy curves off a batched sweep, and takes
gradients with respect to a stagewise price forecast.
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
    # 05 — batched solves and forecasts

    Every input of `AcadosDiffMpcTorch` carries a leading **batch dimension**:
    `x0` is `(B, nx)`, a global override `(B, dim)`, a stagewise one
    `(B, N+1, dim)`. The `B` problems are solved together by an acados batch
    solver — so sweeps, forecast studies and training batches all cost one
    call.

    The second theme is **forecasts**, which enter through the two parameter
    interfaces of notebook 04:

    - **outdoor temperature** — non-differentiable stagewise values in
      `model.p`, passed per solve with shape `(B, N+1, 1)`,
    - **electricity price** — differentiable with `splits="stagewise"`, so
      it also takes per-stage values `(B, N+1, 1)`, but lives in
      `model.p_global` and supports gradients.

    A "forecast" is then nothing more than a length-`N+1` slice of a longer
    signal, handed to the solver through `params`.
    """)
    return


@app.cell
def _(mo):
    import sys

    sys.path.insert(0, str(mo.notebook_dir().parent))  # notebooks/ root -> nb_utils

    import matplotlib.pyplot as plt
    import numpy as np
    import torch

    from nb_utils.data import make_day_profiles, stack_forecast_windows
    from nb_utils.heating import build_heating_ocp

    from leap_c.torch import AcadosDiffMpcTorch

    return (
        AcadosDiffMpcTorch,
        build_heating_ocp,
        make_day_profiles,
        np,
        plt,
        stack_forecast_windows,
        torch,
    )


@app.cell
def _(AcadosDiffMpcTorch, build_heating_ocp, torch):
    N_FC = 32  # 32 stages of 15 min = an 8 h look-ahead
    DT_FC = 0.25  # time step [h]
    N_STARTS = 96  # one forecast window per quarter-hour of the day

    mpc_fc = AcadosDiffMpcTorch(
        *build_heating_ocp(
            N_FC, DT_FC, price_splits="stagewise", q_max=12.0, name="heating_forecast"
        ),
        dtype=torch.float64,
        n_batch_init=N_STARTS,
        verbose=False,
    )
    return DT_FC, N_FC, N_STARTS, mpc_fc


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Two constructor arguments matter once you batch:

    - `n_batch_init` sets how many solver instances the acados batch solver
      pre-allocates. Exceeding it later works, but triggers a solver
      re-creation — size it to your largest batch up front.
    - `num_threads_batch_solver` (default 4) parallelizes the batch across
      threads.
    - `dtype` fixes the dtype of all tensor inputs/outputs (the solver
      itself always computes in double precision); we pass
      `torch.float64` throughout the series so autograd checks are exact.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## A synthetic day

    `nb_utils.data.make_day_profiles` synthesizes a deterministic day:
    outdoor temperature follows a (slightly perturbed) daily sinusoid, and
    the price steps up during the morning and evening peaks. We generate 36
    hours so an 8-hour window can start at any quarter-hour of the day.
    """)
    return


@app.cell
def _(DT_FC, make_day_profiles, np, plt):
    t_day, outdoor_day, price_day = make_day_profiles(n_steps=144, dt_hours=DT_FC)

    data_fig, data_ax = plt.subplots(figsize=(9, 3.2))
    data_ax.plot(t_day, outdoor_day, color="tab:blue", label="outdoor temp [degC]")
    data_ax.set_xlabel("Time since midnight [h]")
    data_ax.set_ylabel("Outdoor temp [degC]", color="tab:blue")
    data_ax.set_xticks(np.arange(0, 37, 6))
    data_ax.grid(True, alpha=0.3)
    _ax_price = data_ax.twinx()
    _ax_price.step(t_day, price_day, where="post", color="tab:purple", label="price")
    _ax_price.set_ylabel("Price [EUR/kWh]", color="tab:purple")
    _ax_price.set_ylim(0.0, 0.5)
    data_fig.suptitle("Synthetic forecast data (36 h)")
    data_fig.tight_layout()
    data_fig
    return outdoor_day, price_day, t_day


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## One batched solve, 96 forecast windows

    Batch element $i$ receives the forecast window starting at quarter-hour
    $i$ — the whole day of open-loop plans is precomputed in **one** call
    (the room always starts at 21 °C). `stack_forecast_windows` does the
    slicing; the price is passed as a leaf tensor with `requires_grad=True`,
    so the same single solve also powers the gradient section at the end.
    """)
    return


@app.cell
def _(N_FC, N_STARTS, mpc_fc, outdoor_day, price_day, stack_forecast_windows, torch):
    outdoor_windows = stack_forecast_windows(outdoor_day, N_STARTS, N_FC)[
        ..., None
    ]  # (N_STARTS, N+1, 1), numpy -> non-differentiable interface
    price_windows = torch.tensor(
        stack_forecast_windows(price_day, N_STARTS, N_FC)[..., None],
        requires_grad=True,
    )  # (N_STARTS, N+1, 1), leaf tensor -> differentiable interface

    ctx_fc, _, x_fc, u_fc, value_fc = mpc_fc(
        x0=torch.full((N_STARTS, 1), 21.0),
        params={"outdoor_temp": outdoor_windows, "price": price_windows},
    )

    x_plans = x_fc.detach().numpy()  # (N_STARTS, N+1, 1)
    u_plans = u_fc.detach().numpy()  # (N_STARTS, N, 1)
    return price_windows, u_plans, value_fc, x_plans


@app.cell
def _(N_STARTS, mo):
    start_slider = mo.ui.slider(
        start=0,
        stop=N_STARTS - 1,
        step=1,
        value=20,  # 05:00 — just before the morning price peak
        label="forecast window start [quarter-hours since midnight]",
        show_value=True,
    )
    return (start_slider,)


@app.cell
def _(
    N_FC,
    mo,
    np,
    outdoor_day,
    plt,
    price_day,
    start_slider,
    t_day,
    u_plans,
    x_plans,
):
    _s = start_slider.value
    _t_win = t_day[_s : _s + N_FC + 1]

    fc_fig, fc_axes = plt.subplots(2, 1, figsize=(9, 6))

    # Top: the day profiles with the active horizon window shaded.
    fc_axes[0].plot(t_day[:129], outdoor_day[:129], color="tab:blue", label="outdoor temp")
    fc_axes[0].set_ylabel("Outdoor temp [degC]", color="tab:blue")
    fc_axes[0].axvspan(_t_win[0], _t_win[-1], color="gold", alpha=0.25, label="horizon window")
    fc_axes[0].set_xticks(np.arange(0, 33, 4))
    fc_axes[0].grid(True, alpha=0.3)
    fc_axes[0].legend(loc="upper left", fontsize=8)
    _ax_p = fc_axes[0].twinx()
    _ax_p.step(t_day[:129], price_day[:129], where="post", color="tab:purple")
    _ax_p.set_ylabel("Price [EUR/kWh]", color="tab:purple")
    _ax_p.set_ylim(0.0, 0.5)
    fc_axes[0].set_title(f"Forecast window starting at {_t_win[0]:.2f} h")

    # Bottom: the plan inside the window.
    fc_axes[1].plot(_t_win, x_plans[_s, :, 0], "-o", markersize=3, color="tab:blue",
                    label="planned room temp")
    fc_axes[1].axhline(21.0, ls="--", lw=0.8, color="gray")
    fc_axes[1].set_ylabel("Room temp [degC]", color="tab:blue")
    fc_axes[1].set_ylim(19.5, 22.5)
    fc_axes[1].set_xlabel("Time since midnight [h]")
    fc_axes[1].grid(True, alpha=0.3)
    fc_axes[1].legend(loc="upper left", fontsize=8)
    _ax_q = fc_axes[1].twinx()
    _ax_q.step(_t_win[:-1], u_plans[_s, :, 0], where="post", color="tab:orange")
    _ax_q.set_ylabel("Heating [kW]", color="tab:orange")
    _ax_q.set_ylim(0.0, 12.0)

    fc_fig.tight_layout()
    mo.vstack([start_slider, fc_fig])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Slide towards **05:00–06:00** and watch the plan pre-heat: the room is
    warmed *before* the 7–9 h price peak enters the window, then drifts
    through the expensive hours. The same happens again ahead of the evening
    peak. (Notebook 06 turns exactly this into a receding-horizon controller
    — solve, apply the first decision, step, repeat.)

    ## The MPC as a function: value and policy curves

    Solved as a batch over many initial states, the MPC *is* two functions
    of the state: the optimal cost $V(T_0)$ (a value function) and the first
    heating decision $q_0^\star(T_0)$ (a policy). One batched call sweeps 41
    initial temperatures under two constant prices.
    """)
    return


@app.cell
def _(N_FC, mpc_fc, np, torch):
    N_GRID = 41
    T0_grid = np.linspace(14.0, 24.0, N_GRID)
    price_levels = [0.15, 0.35]  # off-peak vs. peak

    # Stack the two sweeps into one batch of 82 problems.
    _x0 = torch.tensor(np.tile(T0_grid, len(price_levels)).reshape(-1, 1))
    _price = torch.tensor(
        np.concatenate(
            [np.full((N_GRID, N_FC + 1, 1), _p) for _p in price_levels]
        )
    )
    _outdoor = np.full((len(price_levels) * N_GRID, N_FC + 1, 1), 5.0)

    _, _, _, u_grid, value_grid = mpc_fc(
        x0=_x0, params={"outdoor_temp": _outdoor, "price": _price}
    )
    V_curves = value_grid.detach().numpy().reshape(len(price_levels), N_GRID)
    q0_curves = u_grid.detach().numpy()[:, 0, 0].reshape(len(price_levels), N_GRID)
    return N_GRID, T0_grid, V_curves, price_levels, q0_curves


@app.cell
def _(T0_grid, V_curves, plt, price_levels, q0_curves):
    vp_fig, vp_axes = plt.subplots(1, 2, figsize=(9.5, 3.6))
    for _i, _p in enumerate(price_levels):
        _c = "tab:green" if _i == 0 else "tab:red"
        vp_axes[0].plot(T0_grid, V_curves[_i], color=_c, label=f"price {_p:.2f}")
        vp_axes[1].plot(T0_grid, q0_curves[_i], color=_c, label=f"price {_p:.2f}")
    vp_axes[0].set_xlabel("Initial room temp $T_0$ [degC]")
    vp_axes[0].set_ylabel("Optimal cost $V(T_0)$")
    vp_axes[0].set_title("Value function")
    vp_axes[1].set_xlabel("Initial room temp $T_0$ [degC]")
    vp_axes[1].set_ylabel("First decision $q_0^\\star(T_0)$ [kW]")
    vp_axes[1].set_title("Policy")
    for _ax in vp_axes:
        _ax.grid(True, alpha=0.3)
        _ax.legend(fontsize=8)
    vp_fig.tight_layout()
    vp_fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The value function is a smooth bowl around the setpoint; starting cold
    costs more when energy is expensive. The policy saturates at
    $q_{\max}$ for cold starts — the flat segment where the actuator limit,
    not the optimizer, dictates the action (remember the vanishing gradients
    of notebook 03).

    ## Gradients with respect to a forecast

    Because the price is *differentiable stagewise*, every one of its
    `N+1` per-stage values is a separate parameter — so the gradient of the
    optimal cost tells us how one solve responds to **each future price
    individually**. One `value.sum().backward()` on the batched solve populates
    `price_windows.grad`, giving that gradient shaped exactly like the forecast.
    (The lower-level exact KKT sensitivity API exposes the same gradient
    directly, as `diff_mpc.diff_mpc_fun.sensitivity(ctx, "dvalue_dp_global")`.)
    """)
    return


@app.cell
def _(DT_FC, N_FC, mo, plt, price_windows, t_day, value_fc):
    _s = 20  # the 05:00 window — the morning price peak sits mid-horizon
    _t_win = t_day[_s : _s + N_FC + 1]

    value_fc.sum().backward()
    dV_dprice = price_windows.grad[_s, :, 0].numpy()

    sens_fig, sens_ax = plt.subplots(figsize=(9, 3.4))
    sens_ax.bar(_t_win, dV_dprice, width=0.8 * DT_FC, color="tab:purple")
    sens_ax.set_xlabel("Time since midnight [h]")
    sens_ax.set_ylabel("∂V/∂price_k  [kWh]")
    sens_ax.grid(True, alpha=0.3)
    sens_ax.set_title("Which future price is the 05:00 solve exposed to?")
    sens_fig.tight_layout()

    mo.vstack([
        sens_fig,
        mo.md(
            "Each bar is the planned energy bought at that stage (for a cost "
            "linear in the price, ∂V/∂price_k is exactly `q_k·Δt`)."
        ),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Recap

    - The batch dimension turns sweeps, forecast studies and training
      batches into single calls; size `n_batch_init` to your largest batch.
    - Forecasts are `(B, N+1, 1)` windows — numpy through `model.p` when
      you only need values, leaf tensors through `p_global` when you also
      need gradients.
    - A batched sweep over initial states reads the MPC's value function
      and policy directly off the solver.

    **Next:** `06_planner_interface.py` wraps the layer into a planner —
    `forward(obs) → action` — gives it a slacked, time-varying comfort band,
    and closes the loop against a house that does *not* match the model.
    """)
    return


if __name__ == "__main__":
    app.run()
