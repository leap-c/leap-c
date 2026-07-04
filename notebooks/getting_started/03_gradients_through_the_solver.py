"""Part 3 — every way to get a gradient out of the solver.

The heating MPC from notebook 02 is a differentiable function of its inputs.
This notebook maps the terrain: value vs. Q-function, the three autograd
routes (``.backward()``, ``torch.autograd.grad``,
``torch.autograd.functional.jacobian``), and the place where gradients die —
saturated constraints.
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
    # 03 — gradients through the solver

    The layer solves

    $$V(x_0, \theta) = \min_{x, u}\; \sum_k \ell_k(x_k, u_k; \theta)
    \quad \text{s.t.} \quad \text{dynamics}(\theta),\ \text{bounds},$$

    and everything it returns — `u0`, the trajectories, `value` — is
    differentiable with respect to $x_0$ and every **differentiable**
    parameter in $\theta$. The gradients are *exact*: leap-c reads them off
    the solver's KKT system rather than finite-differencing.

    We use the heating OCP of notebook 02 (imported from `nb_utils`), with
    its differentiable $R$, `comfort_setpoint`, and `price`.
    """)
    return


@app.cell
def _(mo):
    import sys

    sys.path.insert(0, str(mo.notebook_dir().parent))  # notebooks/ root -> nb_utils

    import matplotlib.pyplot as plt
    import numpy as np
    import torch

    from nb_utils.heating import build_heating_ocp

    from leap_c.torch import AcadosDiffMpcTorch

    return AcadosDiffMpcTorch, build_heating_ocp, np, plt, torch


@app.cell
def _(AcadosDiffMpcTorch, build_heating_ocp, torch):
    N_GRAD = 32  # 8 h horizon of 15 min steps
    DT_GRAD = 0.25
    Q_MAX = 12.0
    B_MAX = 41  # largest batch in this notebook

    mpc = AcadosDiffMpcTorch(
        *build_heating_ocp(N_GRAD, DT_GRAD, q_max=Q_MAX, name="heating_gradients"),
        dtype=torch.float64,
        n_batch_init=B_MAX,
        verbose=False,
    )
    return B_MAX, Q_MAX, mpc


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## V and Q: two calls, two functions

    - `diff_mpc(x0=...)` optimizes **all** controls → the value function
      $V(s)$,
    - `diff_mpc(x0=..., u0=...)` **pins the first control** and optimizes
      the rest → the Q-function $Q(s, a)$.

    Both come from the same solver. Sweeping $a$ at a fixed cold state
    (one batched call) traces $Q(s, \cdot)$; its minimum touches $V(s)$ at
    the optimal first action — the Bellman identity
    $V(s) = \min_a Q(s, a)$, read off the solver.
    """)
    return


@app.cell
def _(B_MAX, Q_MAX, mpc, np, plt, torch):
    T_COLD = 17.0  # a cold room, well below the 21 degC setpoint

    # Q(s, a): pin the first control at each of B_MAX levels.
    a_sweep = np.linspace(0.0, Q_MAX, B_MAX)
    _, _, _, _, q_values = mpc(
        x0=torch.full((B_MAX, 1), T_COLD),
        u0=torch.tensor(a_sweep).reshape(-1, 1),
    )
    # V(s): one free solve, plus its optimal first action.
    _, u0_star, _, _, v_value = mpc(x0=torch.tensor([[T_COLD]]))

    q_np = q_values.detach().numpy().ravel()
    vq_fig, vq_ax = plt.subplots(figsize=(8, 3.6))
    vq_ax.plot(a_sweep, q_np, color="tab:blue", label="Q(s, a)")
    vq_ax.axhline(v_value.item(), ls="--", lw=0.9, color="tab:green", label="V(s)")
    vq_ax.axvline(u0_star.item(), ls=":", lw=0.9, color="tab:orange",
                  label=f"optimal a = {u0_star.item():.2f} kW")
    vq_ax.set_xlabel("First heating decision a = q₀ [kW]")
    vq_ax.set_ylabel("Cost")
    vq_ax.grid(True, alpha=0.3)
    vq_ax.legend()
    vq_fig.suptitle("Q(s,·) touches V(s) at the optimal first action")
    vq_fig.tight_layout()
    vq_fig
    return (T_COLD,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Route 1: `.backward()` on a scalar

    The training-loop workhorse. Pass parameters as leaf tensors, call
    `.backward()` on the (summed) output, read `.grad` — gradients arrive
    shaped like the tensors you passed.

    (When checking gradients by hand, remember notebook 02's convention:
    acados scales stage costs by $\Delta t$, so $\partial V/\partial
    \mathrm{price}$ is the planned energy $\Delta t \sum_k q_k$ in kWh —
    not $\sum_k q_k$.)
    """)
    return


@app.cell
def _(T_COLD, mpc, torch):
    R_leaf = torch.tensor([[2.0]], dtype=torch.float64, requires_grad=True)
    price_leaf = torch.tensor([[0.15]], dtype=torch.float64, requires_grad=True)

    _, _, _, _, _value = mpc(
        x0=torch.tensor([[T_COLD]]), params={"R": R_leaf, "price": price_leaf}
    )
    _value.backward()

    dV_dR_bwd = R_leaf.grad.item()
    dV_dprice_bwd = price_leaf.grad.item()
    print(f"dV/dR     = {dV_dR_bwd:+.4f}   (insulation lowers the optimal cost)")
    print(f"dV/dprice = {dV_dprice_bwd:+.4f}   (= total energy bought [kWh], envelope theorem)")
    return R_leaf, dV_dR_bwd, dV_dprice_bwd, price_leaf


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Route 2: `torch.autograd.grad`

    The functional form of the same thing — no `.grad` side effects, several
    inputs at once, `retain_graph=True` to differentiate the same solve
    twice. We check it against route 1.
    """)
    return


@app.cell
def _(R_leaf, T_COLD, dV_dR_bwd, dV_dprice_bwd, mpc, np, price_leaf, torch):
    _, u0_g, _, _, value_g = mpc(
        x0=torch.tensor([[T_COLD]]), params={"R": R_leaf, "price": price_leaf}
    )

    (dV_dR, dV_dprice) = torch.autograd.grad(
        value_g.sum(), (R_leaf, price_leaf), retain_graph=True
    )
    # Same solve, different output: the first action's parameter gradients.
    (du0_dR, du0_dprice) = torch.autograd.grad(u0_g.sum(), (R_leaf, price_leaf))

    assert np.isclose(dV_dR.item(), dV_dR_bwd) and np.isclose(dV_dprice.item(), dV_dprice_bwd)
    print(f"dV/dR   = {dV_dR.item():+.4f}   (matches route 1)")
    print(f"du0/dR  = {du0_dR.item():+.4f},  du0/dprice = {du0_dprice.item():+.4f}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Route 3: `torch.autograd.functional.jacobian`

    For **vector-valued** questions. Here: the local feedback gain of the
    MPC policy, $\partial q_0^\star / \partial T_0$ — how the first heating
    decision reacts to the measured room temperature.
    """)
    return


@app.cell
def _(T_COLD, mpc, torch):
    _x0 = torch.tensor([[T_COLD]], dtype=torch.float64)
    feedback_gain = torch.autograd.functional.jacobian(
        lambda x0: mpc(x0=x0)[1], _x0  # [1] = u0
    )  # shape (1, nu, 1, nx)

    print(f"du0/dT0 = {feedback_gain.squeeze().item():+.3f} kW/K")
    print("negative: a warmer room needs less heating — the MPC *is* a feedback law")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    | route | use when |
    |---|---|
    | `.backward()` | training loops — scalar loss, `.grad` accumulation |
    | `torch.autograd.grad` | several inputs/outputs, no side effects |
    | `functional.jacobian` | full Jacobians of vector outputs |

    All three read the same exact KKT gradient. (There is also a lower-level
    API that returns Jacobian blocks straight off the solver context — see
    `custom_examples/advanced_sensitivities.py` once you need it.)

    ## Where gradients die: saturation

    Gradients of the *policy* only exist where the policy actually moves.
    Send the weather down until the heater pins at $q_{\max}$: from there,
    a slightly higher price changes nothing — the constraint, not the
    optimizer, dictates the action, and $\partial q_0 / \partial\mathrm{price}$
    collapses to zero.
    """)
    return


@app.cell
def _(B_MAX, Q_MAX, T_COLD, mpc, np, torch):
    outdoor_levels = np.linspace(-25.0, 15.0, B_MAX)

    _outdoor = np.tile(outdoor_levels.reshape(-1, 1, 1), (1, mpc.parameter_manager.N_horizon + 1, 1))
    _price = torch.full((B_MAX, 1), 0.15, dtype=torch.float64, requires_grad=True)

    _, u0_sat, _, _, _ = mpc(
        x0=torch.full((B_MAX, 1), T_COLD),
        params={"outdoor_temp": _outdoor, "price": _price},
    )
    # Each batch element depends only on its own price -> one backward pass
    # recovers all per-element gradients.
    (dq0_dprice,) = torch.autograd.grad(u0_sat.sum(), _price)

    q0_sat = u0_sat.detach().numpy().ravel()
    dq0_np = dq0_dprice.numpy().ravel()
    n_pinned = int((q0_sat > Q_MAX - 1e-6).sum())
    print(f"{n_pinned} of {B_MAX} solves pinned at q_max = {Q_MAX} kW")
    return dq0_np, outdoor_levels, q0_sat


@app.cell
def _(Q_MAX, dq0_np, outdoor_levels, plt, q0_sat):
    sat_fig, sat_axes = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
    sat_axes[0].plot(outdoor_levels, q0_sat, color="tab:orange")
    sat_axes[0].axhline(Q_MAX, ls="--", lw=0.8, color="gray", label="q_max")
    sat_axes[0].set_ylabel("q₀* [kW]")
    sat_axes[0].legend()
    sat_axes[1].plot(outdoor_levels, dq0_np, color="tab:purple")
    sat_axes[1].set_ylabel("∂q₀*/∂price")
    sat_axes[1].set_xlabel("Outdoor temperature [degC]")
    for _ax in sat_axes:
        _ax.grid(True, alpha=0.3)
    sat_fig.suptitle("The policy gradient dies where the actuator saturates")
    sat_fig.tight_layout()
    sat_fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    On the cold side the heater is pinned and the price gradient is exactly
    zero — a learning algorithm gets no signal from these samples. Keep
    this plot in mind whenever a training loss goes flat: check whether
    your policy is riding a constraint. (The *value* gradient does not die
    here — costs keep responding to prices even when the action cannot.)

    A related knob for learning setups: the constructor accepts
    `discount_factor=` to re-weight stage costs, which notebook 08 uses
    when it aligns the OCP with a reinforcement-learning objective.

    **Next:** `04_parameter_management.py` — the full parameter model:
    differentiable vs. not, and the stage structure of `splits`.
    """)
    return


if __name__ == "__main__":
    app.run()
