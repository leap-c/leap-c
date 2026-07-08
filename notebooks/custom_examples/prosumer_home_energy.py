"""The prosumer: an economic MPC that buys, sells, heats and stores.

Notebook 06 traded with a single battery and one signed power. A real
prosumer has three levers behind one meter — a heat pump, a battery and a PV
panel — and faces asymmetric prices: it buys at a dynamic tariff and sells at
the much lower feed-in tariff. The 24-hour tariff is a stagewise
differentiable parameter, and the punchline is not just the planned grid
exchange but its full Jacobian: how every quarter-hour of the plan responds
to every quarter-hour of the price.
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
    # 08 — the prosumer: what is your plan worth, per price?

    Notebooks 04-06 each traded through **one** knob: a heater that only
    buys, a battery that buys and sells at the same price. A *prosumer*
    faces the real setting: an R1C1-heated building with a heat pump, a
    battery, a PV panel — and a grid connection where buying and selling
    are **not** symmetric. A household dynamic tariff swings between
    roughly 0.18 and 0.50 EUR/kWh over a day (spot price plus surcharges),
    while surplus PV is sold at the fixed feed-in tariff of about
    **0.079 EUR/kWh** — selling earns a quarter of what buying costs.

    The 24-hour buy-price profile at 15-minute resolution is a **stagewise
    differentiable parameter** (96 stages). The prosumer reports two
    things: the planned grid exchange over the horizon, and the
    **sensitivity of that plan to every price stage** — a 96×97 Jacobian
    telling the tariff-setter how one EUR/kWh more at 04:00 shifts the
    night's purchases — and where the battery has already decoupled the
    plan from the grid so completely that whole columns vanish.
    """)
    return


@app.cell
def _(mo):
    import sys

    sys.path.insert(0, str(mo.notebook_dir().parent))  # notebooks/ root -> nb_utils

    import matplotlib.pyplot as plt
    import numpy as np
    import torch

    from nb_utils.data import make_prosumer_profiles
    from nb_utils.params import p_global_slice
    from nb_utils.prosumer import (
        build_prosumer_ocp,
        gnet_price_jacobian,
        plot_gnet_price_jacobian_col,
        plot_gnet_price_jacobian_heatmap,
        plot_prosumer_day,
        plot_prosumer_plan,
        step_prosumer,
    )

    from leap_c.torch import AcadosDiffMpcTorch

    torch.set_default_dtype(torch.float64)
    return (
        AcadosDiffMpcTorch,
        build_prosumer_ocp,
        gnet_price_jacobian,
        make_prosumer_profiles,
        np,
        p_global_slice,
        plot_gnet_price_jacobian_col,
        plot_gnet_price_jacobian_heatmap,
        plot_prosumer_day,
        plot_prosumer_plan,
        plt,
        step_prosumer,
        torch,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The OCP

    States are the room temperature and the battery energy, controls are
    the heat-pump electric power $q$, the grid purchase $g_\text{buy}$ and
    the grid feed-in $g_\text{sell}$ (all $\geq 0$):

    $$T_{k+1} = T_k + \Delta t \left( \tfrac{T^\text{out}_k - T_k}{R\,C} + \tfrac{\mathrm{COP}\, q_k}{C} \right), \qquad E_{k+1} = E_k + \Delta t \, P^\text{bat}_k .$$

    The battery power is **not a control** — it is whatever balances the
    household bus:

    $$P^\text{bat}_k = g_{\text{buy},k} - g_{\text{sell},k} + p^\text{pv}_k - q_k .$$

    This elimination keeps every constraint a simple box (states:
    $T \in [19, 23]$, $E \in [0, E_{\max}]$; controls: three nonnegative
    boxes) — no equality constraints needed. The stage cost is purely money
    plus two regularizers:

    $$\ell_k = \text{price}^\text{buy}_k g_{\text{buy},k} - \text{price}^\text{sell} g_{\text{sell},k} + c_\text{wear} (P^\text{bat}_k)^2 + \varepsilon \left( q_k^2 + g_{\text{buy},k}^2 + g_{\text{sell},k}^2 \right),$$

    with the terminal credit $-\lambda E_N$ from `battery_arbitrage.py`. Three details are critical:

    - **The $\varepsilon$ term is mandatory, and $c_\text{wear}$ alone would
      not do.** The wear term's Hessian is rank **one** in the 3-D control
      space (it only curves along the direction that changes $P^\text{bat}$);
      the exact-Hessian SQP and the regularization-stripping sensitivity
      solver need positive curvature in *every* control direction.
      $\varepsilon = 10^{-3}$ distorts marginal prices by only
      $2\varepsilon g \approx 0.01$ EUR/kWh at 5 kW.
    - **Nobody buys and sells at once.** Doing both at 1 kW costs
      $\text{price}^\text{buy} - \text{price}^\text{sell} \geq 0.05$ EUR/kWh
      plus the $\varepsilon$ curvature — the optimizer never does it
      (asserted below).
    - **Comfort is soft, physics is hard.** The temperature band is a
      preference, enforced by slacks (`idxsbx` softens only the $T$ row —
      the same mechanism as the MSD notebook's soft state bounds); the battery box is
      physics and stays hard. The terminal value must satisfy
      $\text{price}^\text{sell} < \lambda < \min_k \text{price}^\text{buy}_k$
      — above the feed-in tariff or the plan dumps the battery into the
      grid at stage $N$, below the cheapest price or it hoard-buys for the
      terminal credit.

    The five parameters:

    | name | default | differentiable | splits | role |
    |---|---|---|---|---|
    | `price_buy` | 0.25 | yes | stagewise | dynamic tariff [EUR/kWh] — *the* parameter of this notebook |
    | `price_sell` | 0.079 | yes | global | feed-in tariff [EUR/kWh] |
    | `terminal_value` ($\lambda$) | 0.12 | yes | global | value of stored energy at stage $N$ |
    | `outdoor_temp` | 8.0 | no | — | weather forecast [degC] |
    | `p_pv` | 0.0 | no | — | PV forecast [kW] |

    The builder lives in `nb_utils/prosumer.py` — by now the pattern (a
    fresh manager and OCP built together, `EXTERNAL` cost, `DISCRETE`
    dynamics) is the one taught inline in the getting-started series.
    """)
    return


@app.cell
def _(AcadosDiffMpcTorch, build_prosumer_ocp, torch):
    N_HORIZON = 96  # 96 stages of 15 min = a full day
    DT = 0.25  # time step [h]
    E_MAX = 10.0  # battery capacity [kWh]
    Q_MAX = 4.0  # heat pump electric power limit [kW] (12 kW heat at COP 3)
    G_MAX = 10.0  # grid connection limit [kW]
    T0 = 21.0  # initial room temperature [degC]
    E0 = 5.0  # initial battery energy [kWh] — half full
    T_BAND = (19.0, 23.0)  # comfort band [degC]
    B_GRID = 48  # batch size of the precomputed slider grid (see below)

    mpc_pro = AcadosDiffMpcTorch(
        *build_prosumer_ocp(N_HORIZON, DT, e_max=E_MAX, q_max=Q_MAX, g_max=G_MAX, t_band=T_BAND),
        dtype=torch.float64,
        n_batch_init=B_GRID,
        verbose=False,
    )
    return B_GRID, DT, E0, E_MAX, N_HORIZON, T0, T_BAND, mpc_pro


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## A prosumer's day

    `nb_utils.data.make_prosumer_profiles` synthesizes a deterministic
    shoulder-season day: a dynamic tariff with a 0.22 EUR/kWh overnight
    base, a morning peak to 0.32, a midday solar dip to 0.18 and an evening
    peak whose height is the main experiment knob; a clear-sky PV bell
    peaking at 12:30; and an outdoor temperature between roughly 2 and
    14 degC, so heating is needed all day. The three profile knobs —
    evening-peak height, overall price level, PV size — can be manipulated through
    sliders further down.
    """)
    return


@app.cell
def _(DT, N_HORIZON, make_prosumer_profiles, plot_prosumer_day):
    t_day, outdoor_day, price_day, pv_day = make_prosumer_profiles(
        n_steps=N_HORIZON + 1, dt_hours=DT
    )
    PRICE_SELL = 0.079  # feed-in tariff [EUR/kWh]
    LAMBDA = 0.12  # terminal value of stored energy [EUR/kWh]

    plot_prosumer_day(t_day, outdoor_day, price_day, pv_day, PRICE_SELL)
    return LAMBDA, PRICE_SELL, outdoor_day, price_day, pv_day, t_day


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## One solve, read the plan

    A single solve over the default day. The stagewise buy price goes in as
    a `(1, 97, 1)` tensor, the two forecasts as numpy arrays of the same
    shape — the interface from getting_started notebook 05, just with more of
    everything.
    """)
    return


@app.cell
def _(
    E0,
    E_MAX,
    T0,
    T_BAND,
    mo,
    mpc_pro,
    np,
    outdoor_day,
    plot_prosumer_plan,
    price_day,
    pv_day,
    t_day,
    torch,
):
    _ctx, _, _x, _u, _value = mpc_pro(
        x0=torch.tensor([[T0, E0]]),
        params={
            "price_buy": torch.tensor(price_day.reshape(1, -1, 1)),
            "outdoor_temp": outdoor_day.reshape(1, -1, 1),
            "p_pv": pv_day.reshape(1, -1, 1),
        },
    )
    assert int(np.asarray(_ctx.status).item()) == 0

    _fig = plot_prosumer_plan(
        t_day,
        price_day,
        pv_day,
        _x.detach().numpy()[0],
        _u.detach().numpy()[0],
        E_MAX,
        T_BAND,
        "Open-loop plan for the default day",
    )
    mo.vstack([
        _fig,
        mo.md(
            f"Net cost of the day: `{float(_value.detach()):.2f}` EUR. The plan "
            "buys overnight and pre-heats toward the band edge before the morning "
            "peak, charges the battery from the midday PV surplus instead of "
            "selling it at 0.079 (stored energy is worth λ = 0.12), rides "
            "through the evening peak on the battery — and sells only what "
            "neither the house nor the battery can absorb."
        ),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## One batched solve, 48 scenarios

    Sliders never trigger solves in this series — they index precomputed
    batched results. Batch element $b$ is one combination of the three
    profile knobs: 4 evening-peak heights × 3 price levels × 4 PV sizes
    = 48 scenarios, solved in one call. The buy price goes in as a leaf
    tensor with `requires_grad=True`, as do the feed-in tariff and the
    terminal value.
    """)
    return


@app.cell
def _(
    B_GRID,
    DT,
    E0,
    LAMBDA,
    N_HORIZON,
    PRICE_SELL,
    T0,
    make_prosumer_profiles,
    mpc_pro,
    np,
    step_prosumer,
    torch,
):
    PEAK_GRID = np.array([0.25, 0.30, 0.40, 0.50])  # evening peak height [EUR/kWh]
    LEVEL_GRID = np.array([-0.05, 0.0, 0.05])  # uniform price shift [EUR/kWh]
    PV_GRID = np.array([0.0, 2.0, 4.0, 6.0])  # PV size [kWp]

    def combo_index(i_peak, i_level, i_pv):
        """Flat batch index of a knob combination (peak-major, PV-minor)."""
        return i_peak * len(LEVEL_GRID) * len(PV_GRID) + i_level * len(PV_GRID) + i_pv

    _prices, _pvs, _outs = [], [], []
    for _peak in PEAK_GRID:
        for _level in LEVEL_GRID:
            for _pv in PV_GRID:
                _, _outdoor, _price, _p_pv = make_prosumer_profiles(
                    n_steps=N_HORIZON + 1, peak_height=_peak, level_shift=_level, pv_kwp=_pv
                )
                _prices.append(_price)
                _pvs.append(_p_pv)
                _outs.append(_outdoor)

    price_buy_b = torch.tensor(np.stack(_prices)[..., None], requires_grad=True)
    price_sell_b = torch.full((B_GRID, 1), PRICE_SELL, requires_grad=True)
    lam_b = torch.full((B_GRID, 1), LAMBDA, requires_grad=True)
    outdoor_b = np.stack(_outs)[..., None]
    pv_b = np.stack(_pvs)[..., None]

    ctx_pro, _, x_pro, u_pro, value_pro = mpc_pro(
        x0=torch.tensor([[T0, E0]]).repeat(B_GRID, 1),
        params={
            "price_buy": price_buy_b,
            "price_sell": price_sell_b,
            "terminal_value": lam_b,
            "outdoor_temp": outdoor_b,
            "p_pv": pv_b,
        },
    )
    x_plans = x_pro.detach().numpy()  # (48, N+1, 2)
    u_plans = u_pro.detach().numpy()  # (48, N, 3)

    # The solves succeeded, and the solution behaves as argued above:
    assert (np.asarray(ctx_pro.status) == 0).all()
    # ... nobody buys and sells at once,
    assert (u_plans[:, :, 1] * u_plans[:, :, 2]).max() < 1e-4
    # ... the comfort band holds (up to a whisker of slack),
    assert x_plans[:, :, 0].min() > 19.0 - 1e-2 and x_plans[:, :, 0].max() < 23.0 + 1e-2
    # ... and rolling the plan through the true dynamics reproduces the states.
    _xk = np.array([T0, E0])
    for _k in range(N_HORIZON):
        _xk = step_prosumer(_xk, u_plans[0, _k], outdoor_b[0, _k, 0], pv_b[0, _k, 0], DT)
        assert np.allclose(_xk, x_plans[0, _k + 1], atol=1e-6)
    return (
        LEVEL_GRID,
        PEAK_GRID,
        PV_GRID,
        combo_index,
        ctx_pro,
        lam_b,
        price_buy_b,
        price_sell_b,
        pv_b,
        u_plans,
        u_pro,
        value_pro,
        x_plans,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The plan-vs-price Jacobian

    Row $k$ of the Jacobian is the gradient of the planned net exchange
    $g_{\text{net},k} = g_{\text{buy},k} - g_{\text{sell},k}$ with respect
    to the whole price profile. Reverse mode pays one backward pass per
    output (`advanced_sensitivities.py`'s lesson), so the $96 \times 97$ map costs 96
    backward passes over the batched solve — a couple of seconds for all
    48 scenarios at once.

    The low-level adjoint call `sensitivity(ctx, "du_dp_global")` returns
    the *stage-summed* control sensitivity $\partial (\sum_k u_k) /
    \partial p$ in a single shot — exactly the column sums of our Jacobian.
    We use it as an independent, machine-precision cross-check of every
    entry.
    """)
    return


@app.cell
def _(
    ctx_pro,
    gnet_price_jacobian,
    mpc_pro,
    np,
    p_global_slice,
    price_buy_b,
    u_pro,
):
    S_jac = gnet_price_jacobian(u_pro, price_buy_b)  # (48, 96, 97)

    # Independent cross-check: the one-call adjoint aggregate equals the
    # column sums of the 96 autograd rows.
    price_cols = p_global_slice(mpc_pro.parameter_manager, "price_buy")
    _du_dp = mpc_pro.diff_mpc_fun.sensitivity(ctx_pro, "du_dp_global")  # (48, nu, P)
    _agg = _du_dp[:, 1, price_cols] - _du_dp[:, 2, price_cols]
    assert np.allclose(S_jac.sum(axis=1), _agg, atol=1e-8)
    # The stage-N price multiplies nothing (stage N has only the terminal
    # cost), so its column is exactly zero.
    assert np.abs(S_jac[:, :, -1]).max() < 1e-10
    return S_jac, price_cols


@app.cell
def _(LEVEL_GRID, N_HORIZON, PEAK_GRID, PV_GRID, mo):
    peak_slider = mo.ui.slider(
        start=0, stop=len(PEAK_GRID) - 1, step=1, value=3,
        label="evening peak (0.25 / 0.30 / 0.40 / 0.50 EUR/kWh)", show_value=True,
    )
    level_slider = mo.ui.slider(
        start=0, stop=len(LEVEL_GRID) - 1, step=1, value=1,
        label="price level (-0.05 / 0.00 / +0.05 EUR/kWh)", show_value=True,
    )
    pv_slider = mo.ui.slider(
        start=0, stop=len(PV_GRID) - 1, step=1, value=2,
        label="PV size (0 / 2 / 4 / 6 kWp)", show_value=True,
    )
    stage_slider = mo.ui.slider(
        start=0, stop=N_HORIZON - 1, step=1, value=16,  # 04:00, overnight buying
        label="perturbed price stage j [quarter-hours since midnight]", show_value=True,
    )
    return level_slider, peak_slider, pv_slider, stage_slider


@app.cell
def _(
    E_MAX,
    LEVEL_GRID,
    PEAK_GRID,
    PV_GRID,
    S_jac,
    T_BAND,
    combo_index,
    level_slider,
    mo,
    peak_slider,
    plot_gnet_price_jacobian_col,
    plot_prosumer_plan,
    price_buy_b,
    pv_b,
    pv_slider,
    stage_slider,
    t_day,
    u_plans,
    x_plans,
):
    _b = combo_index(peak_slider.value, level_slider.value, pv_slider.value)
    _j = stage_slider.value
    _price = price_buy_b.detach().numpy()[_b, :, 0]

    _plan_fig = plot_prosumer_plan(
        t_day,
        _price,
        pv_b[_b, :, 0],
        x_plans[_b],
        u_plans[_b],
        E_MAX,
        T_BAND,
        f"peak {PEAK_GRID[peak_slider.value]:.2f}, "
        f"shift {LEVEL_GRID[level_slider.value]:+.2f}, "
        f"PV {PV_GRID[pv_slider.value]:.0f} kWp",
    )
    _jac_fig = plot_gnet_price_jacobian_col(t_day, S_jac[_b], _j, _price)

    mo.vstack([
        mo.hstack([peak_slider, level_slider, pv_slider], justify="start", wrap=True),
        stage_slider,
        mo.hstack([_plan_fig, _jac_fig]),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Things to look for while sliding:

    - **The negative spike at $k = j$** (default: 04:00) — one EUR/kWh
      more at stage $j$ and the plan buys several kW less right there. The
      **positive lobes around it** are where that energy goes instead: the
      battery (and the pre-heatable building) shift the purchase to
      neighboring cheap quarter-hours. This is intertemporal substitution,
      drawn by a solver.
    - **Columns are alive only where the plan buys.** Drag $j$ into the
      evening peak and the column drops to exactly zero: at the default
      settings the battery, the midday PV and a pre-heated building cover
      the whole peak, the plan touches the grid only at night — and a
      price the prosumer does not trade at has **no leverage** over it.
      Constraint-pinned controls have zero sensitivity.
    - **Set PV to zero** (a pure consumer) and daytime buying — and with
      it daytime sensitivity — reappears; the feed-in side goes quiet.
      Grow the evening peak at PV 0 and watch the overnight pre-charging
      intensify.
    - **Shift the whole price level** and almost nothing happens — with no
      place to avoid a uniform surcharge, the plan barely moves. Only
      *spreads* move plans.
    - Put $j$ on the stage-96 price and the column is exactly zero for a
      different reason — that price multiplies nothing.
    """)
    return


@app.cell
def _(S_jac, combo_index, mo, plot_gnet_price_jacobian_heatmap, t_day):
    _b = combo_index(3, 1, 2)  # the default scenario
    _heat_fig = plot_gnet_price_jacobian_heatmap(t_day, S_jac[_b])
    mo.vstack([
        _heat_fig,
        mo.md(
            "The full map for the default scenario: **two live blocks** — "
            "the overnight buying window and the late-evening re-entry "
            "after the peak — each with a blue own-price diagonal (buy less "
            "when *this* stage gets pricier; saturated, it is an order of "
            "magnitude stronger than the rest) inside a red substitution "
            "block (buy more at the other cheap stages instead). The "
            "substitution is stronger in the short late-evening block: "
            "fewer alternative hours to spread over. In between lies a "
            "large **dead region**: PV and the battery keep the daytime "
            "plan off the grid, and no price has any leverage there. The "
            "map is the prosumer's exposure, stage by stage."
        ),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Value gradients in EUR

    As in `battery_arbitrage.py`, the optimal cost $V$ is money and the envelope
    theorem makes its gradients legible:

    $$\frac{\partial V}{\partial \text{price}^\text{buy}_k}
    = g_{\text{buy},k} \Delta t, \qquad
    \frac{\partial V}{\partial \text{price}^\text{sell}}
    = -\Delta t \sum_k g_{\text{sell},k}, \qquad
    \frac{\partial V}{\partial \lambda} = -E_N .$$

    Energy *bought* per stage, energy *sold* over the day, energy *stored*
    at the end — all from one differentiable solve. We take the gradients
    with `torch.autograd.grad` (rather than `.backward()`) so the cell can
    rerun without accumulating, and cross-check them against the exact
    KKT value gradient `dvalue_dp_global` from `advanced_sensitivities.py`.
    """)
    return


@app.cell
def _(
    DT,
    combo_index,
    ctx_pro,
    lam_b,
    mo,
    mpc_pro,
    np,
    plt,
    price_buy_b,
    price_cols,
    price_sell_b,
    t_day,
    torch,
    u_plans,
    value_pro,
    x_plans,
):
    _dV_dbuy, _dV_dsell, _dV_dlam = torch.autograd.grad(
        value_pro.sum(), (price_buy_b, price_sell_b, lam_b), retain_graph=True
    )
    _dV_dbuy = _dV_dbuy[:, :, 0].numpy()
    _dV_dsell = _dV_dsell[:, 0].numpy()
    _dV_dlam = _dV_dlam[:, 0].numpy()

    # Envelope checks: the gradients are readable off the plan itself.
    assert np.allclose(_dV_dbuy[:, :-1], DT * u_plans[:, :, 1], atol=1e-6)
    assert np.abs(_dV_dbuy[:, -1]).max() < 1e-10  # stage-N price enters no cost
    assert np.allclose(_dV_dsell, -DT * u_plans[:, :, 2].sum(axis=1), atol=1e-6)
    assert np.allclose(_dV_dlam, -x_plans[:, -1, 1], atol=1e-6)

    # Exact KKT cross-check (advanced_sensitivities.py's API), all 48 scenarios at once.
    _dV_dp = mpc_pro.diff_mpc_fun.sensitivity(ctx_pro, "dvalue_dp_global")
    assert np.allclose(_dV_dp[:, 0, price_cols], _dV_dbuy, atol=1e-8)

    _b = combo_index(3, 1, 2)
    _fig, _ax = plt.subplots(figsize=(9, 3.2))
    _ax.bar(t_day, _dV_dbuy[_b], width=0.8 * 0.25, color="tab:purple")
    _ax.plot(t_day[:-1], DT * u_plans[_b, :, 1], ".", color="black", markersize=4,
             label=r"$g_{\mathrm{buy},k} \Delta t$ from the plan")
    _ax.set_xlabel("Time since midnight [h]")
    _ax.set_ylabel("∂V/∂price$_k$  [kWh]")
    _ax.grid(True, alpha=0.3)
    _ax.legend(loc="upper right", fontsize=8)
    _ax.set_title("Each bar is the energy bought at that stage")
    _fig.tight_layout()

    mo.vstack([
        _fig,
        mo.md(
            f"For the default scenario the day sells "
            f"{-_dV_dsell[_b]:.1f} kWh (∂V/∂price^sell = {_dV_dsell[_b]:.2f} "
            f"EUR per EUR/kWh) and ends with ∂V/∂λ = {_dV_dlam[_b]:.2f} — "
            f"{-_dV_dlam[_b]:.1f} kWh still in the battery. Autograd and the "
            "exact KKT gradient agree to machine precision."
        ),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Wrap-up

    On top of the series, this notebook added:

    - a **multi-input economic MPC** — heat pump, grid purchase and feed-in
      as controls, with the battery power *eliminated* through the power
      balance so that every constraint stays a box;
    - **asymmetric prices** — a stagewise differentiable dynamic tariff
      against a fixed feed-in tariff, with the price gap (plus a full-rank
      $\varepsilon$ regularizer) keeping buy and sell complementary;
    - **soft comfort, hard physics** — a slacked temperature band next to a
      hard battery box;
    - the **plan-vs-price Jacobian**: 96 backward passes give
      $\partial g_{\text{net},k} / \partial \text{price}_j$ for all 48
      scenarios, cross-checked to machine precision against the stage-summed
      adjoint call `du_dp_global` and the envelope theorem.
    """)
    return


if __name__ == "__main__":
    app.run()
