"""Part 5 — embedding forecasts in the MPC.

Forecasts are the flagship use case for stagewise parameters: every solve, a
window of predicted outdoor temperature and electricity price slides into the
horizon. This notebook feeds a synthetic day of weather/price data through the
R1C1 heating MPC from notebook 04, first open-loop (explore any start time
with a slider), then closed-loop (a receding-horizon day), and finally asks
the solver which future price the value function is most sensitive to.
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
    # 05 — embedding forecasts in the MPC

    Two forecast signals enter the heating OCP, through the two different
    parameter interfaces of the manager:

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
def _():
    import matplotlib.pyplot as plt
    import numpy as np
    import torch

    from nb_utils.data import make_day_profiles
    from nb_utils.heating import build_heating_ocp, step_room
    from nb_utils.params import p_global_slice

    from leap_c.ocp.acados.torch import AcadosDiffMpcTorch

    torch.set_default_dtype(torch.float64)
    return (
        AcadosDiffMpcTorch,
        build_heating_ocp,
        make_day_profiles,
        np,
        p_global_slice,
        plt,
        step_room,
        torch,
    )


@app.cell
def _(AcadosDiffMpcTorch, build_heating_ocp, torch):
    N_FC = 32  # 32 stages of 15 min = an 8 h look-ahead
    DT_FC = 0.25  # time step [h]
    N_STARTS = 96  # one forecast window per quarter-hour of the day

    # q_max is generous so the heater never saturates during the cold night —
    # every wiggle in the closed-loop trajectory is then driven by the price.
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
    (the room always starts at 21 °C). The price is passed as a leaf tensor
    with `requires_grad=True`, so the same solve also powers the gradient
    section at the end. Drag the slider through the day; nothing is
    re-solved.
    """)
    return


@app.cell
def _(N_FC, N_STARTS, mpc_fc, np, outdoor_day, price_day, torch):
    outdoor_windows = np.stack(
        [outdoor_day[s : s + N_FC + 1] for s in range(N_STARTS)]
    )[..., None]  # (N_STARTS, N+1, 1), numpy -> non-differentiable interface
    price_windows = torch.tensor(
        np.stack([price_day[s : s + N_FC + 1] for s in range(N_STARTS)])[..., None],
        requires_grad=True,
    )  # (N_STARTS, N+1, 1), leaf tensor -> differentiable interface

    ctx_fc, _, x_fc, u_fc, value_fc = mpc_fc(
        x0=torch.full((N_STARTS, 1), 21.0),
        params={"outdoor_temp": outdoor_windows, "price": price_windows},
    )

    x_plans = x_fc.detach().numpy()  # (N_STARTS, N+1, 1)
    u_plans = u_fc.detach().numpy()  # (N_STARTS, N, 1)
    return ctx_fc, price_windows, u_plans, value_fc, x_plans


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
    DT_FC,
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
    # No solve here — the slider picks one precomputed forecast window.
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
    peak.

    ## Closing the loop

    A receding-horizon controller repeats this every quarter-hour: solve with
    the current window, apply the first heating decision, step the (here:
    perfectly modelled) room, shift the window. This runs 96 sequential
    solves once at cell execution — not on interaction.
    """)
    return


@app.cell
def _(DT_FC, N_FC, N_STARTS, mpc_fc, np, outdoor_day, price_day, step_room, torch):
    T_sim = np.empty(N_STARTS + 1)
    q_sim = np.empty(N_STARTS)
    T_sim[0] = 21.0

    for _k in range(N_STARTS):
        _out_w = outdoor_day[_k : _k + N_FC + 1].reshape(1, -1, 1)
        _price_w = torch.tensor(price_day[_k : _k + N_FC + 1].reshape(1, -1, 1))
        _, _u0, _, _, _ = mpc_fc(
            x0=torch.tensor([[T_sim[_k]]]),
            params={"outdoor_temp": _out_w, "price": _price_w},
        )
        q_sim[_k] = float(_u0)
        T_sim[_k + 1] = step_room(T_sim[_k], q_sim[_k], outdoor_day[_k], dt=DT_FC)

    energy_cost = float(np.sum(price_day[:N_STARTS] * q_sim * DT_FC))
    return T_sim, energy_cost, q_sim


@app.cell
def _(DT_FC, N_STARTS, T_sim, energy_cost, np, plt, price_day, q_sim, t_day):
    _t_state = DT_FC * np.arange(N_STARTS + 1)
    _t_ctrl = DT_FC * np.arange(N_STARTS)

    cl_fig, cl_axes = plt.subplots(3, 1, figsize=(9, 7), sharex=True)
    cl_axes[0].step(t_day[:N_STARTS], price_day[:N_STARTS], where="post", color="tab:purple")
    cl_axes[0].set_ylabel("Price [EUR/kWh]")
    cl_axes[1].plot(_t_state, T_sim, color="tab:blue")
    cl_axes[1].axhline(21.0, ls="--", lw=0.8, color="gray")
    cl_axes[1].set_ylabel("Room temp [degC]")
    cl_axes[2].step(_t_ctrl, q_sim, where="post", color="tab:orange")
    cl_axes[2].set_ylabel("Heating [kW]")
    cl_axes[2].set_xlabel("Time since midnight [h]")
    for _ax_cl in cl_axes:
        _ax_cl.grid(True, alpha=0.3)
        for _lo, _hi in [(7.0, 9.0), (17.0, 20.0)]:
            _ax_cl.axvspan(_lo, _hi, color="tab:purple", alpha=0.08)
    cl_fig.suptitle(
        f"Receding-horizon day — realized energy cost {energy_cost:.2f} EUR"
    )
    cl_fig.tight_layout()
    cl_fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The purple bands mark the price peaks: the room is heated slightly above
    the setpoint just before them and allowed to sag through them — the
    forecast is worth money.

    ## Gradients with respect to a forecast

    Because the price is *differentiable stagewise*, every one of its
    `N+1` per-stage values is a column of `p_global` — so the exact KKT
    sensitivity `dvalue_dp_global` tells us how the optimal cost of one solve
    responds to **each future price individually**. We read it off the
    batched context from the window explorer above (no new solve) and
    cross-check it against autograd on the same solve.
    """)
    return


@app.cell
def _(
    DT_FC,
    N_FC,
    ctx_fc,
    mo,
    mpc_fc,
    np,
    p_global_slice,
    plt,
    price_windows,
    t_day,
    value_fc,
):
    _s = 20  # the 05:00 window — the morning price peak sits mid-horizon
    _t_win = t_day[_s : _s + N_FC + 1]

    # Exact KKT sensitivities for the whole batch: (B, 1, P). The price's
    # columns inside flat p_global are located from the registration order.
    _dV_dp = mpc_fc.diff_mpc_fun.sensitivity(ctx_fc, "dvalue_dp_global")
    _price_cols = p_global_slice(mpc_fc.parameter_manager, "price")
    dV_dprice = _dV_dp[_s, 0, _price_cols]  # (N+1,)

    # Cross-check against autograd: each batch element depends only on its own
    # price window, so one backward pass over the summed value recovers all
    # per-window gradients.
    value_fc.sum().backward()
    _dV_dprice_auto = price_windows.grad[_s, :, 0].numpy()
    assert np.allclose(dV_dprice, _dV_dprice_auto, rtol=1e-3, atol=1e-6)

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
            "linear in the price, ∂V/∂price_k is exactly `q_k·Δt`). The exact "
            "batched sensitivities agree with autograd to numerical precision."
        ),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Series wrap-up

    - **01** — register parameters, build the OCP, solve, plot the plan.
    - **02** — batched solves turn the MPC into value/policy maps.
    - **03** — gradients through the solver: autograd and exact KKT
      sensitivities.
    - **04** — the parameter design space: differentiable vs. not, global
      vs. splits vs. stagewise.
    - **05** — forecasts as stagewise parameters, closed loop, and
      gradients with respect to a forecast.

    Ideas that would extend the series (see the README roadmap): a cartpole
    notebook with stage-varying references (`leap_c/examples/cartpole`
    already supports `param_splits`), a one-state battery-arbitrage primer,
    and real weather data in place of the synthetic profiles.
    """)
    return


if __name__ == "__main__":
    app.run()
