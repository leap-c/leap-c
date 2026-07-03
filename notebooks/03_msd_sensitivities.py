"""Part 3 — differentiating through the solver.

The "diff" in ``AcadosDiffMpcTorch``: the solver sits inside the autograd
graph, so its outputs can be differentiated with respect to any differentiable
parameter — via ``.backward()``, via ``torch.autograd.functional.jacobian``,
or by reading the exact KKT sensitivities straight off the solver context.

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
    # 03 — differentiating through the solver

    This notebook shows three routes to the same gradients:

    1. plain **autograd**: pass a parameter as a leaf tensor with
       `requires_grad=True` and call `.backward()`,
    2. `torch.autograd.functional.jacobian` — the MPC solve is just another
       differentiable op,
    3. the **exact KKT sensitivities** cached on the solver context, read via
       `sensitivity(ctx, ...)` — no extra backward pass needed.

    It ends with an interactive tangent line that slides along the policy
    and value curves using gradients precomputed in a single batched solve.
    """)
    return


@app.cell
def _():
    import matplotlib.pyplot as plt
    import numpy as np
    import torch

    from nb_utils.msd import build_msd_ocp
    from nb_utils.params import p_global_slice

    from leap_c.ocp.acados.torch import AcadosDiffMpcTorch

    torch.set_default_dtype(torch.float64)
    return AcadosDiffMpcTorch, build_msd_ocp, np, p_global_slice, plt, torch


@app.cell
def _(AcadosDiffMpcTorch, build_msd_ocp, torch):
    N_HORIZON = 50
    DT = 0.1
    N_SWEEP = 41  # the batched parameter sweep at the end of the notebook

    ocp, manager = build_msd_ocp(N_horizon=N_HORIZON, dt=DT)

    diff_mpc = AcadosDiffMpcTorch(
        ocp,
        manager,
        dtype=torch.float64,
        n_batch_init=N_SWEEP,
        verbose=False,
    )
    return N_SWEEP, diff_mpc


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Autograd through the MPC

    Pass the parameter as a leaf tensor with `requires_grad=True`, call
    `.backward()`, and read the gradient — for example $\partial V /
    \partial m$, how the optimal cost responds to the mass.
    """)
    return


@app.cell
def _(diff_mpc, mo, torch):
    x0 = torch.tensor([[0.5, 0.0]])  # displaced 0.5 m, at rest
    mass_param = torch.tensor([[1.5]], requires_grad=True)

    _, _, _, _, value_m = diff_mpc(x0=x0, params={"mass": mass_param})
    value_m.sum().backward()  # reverse-mode autograd populates mass_param.grad

    # The same gradient via torch.autograd.functional.jacobian: the MPC solve is
    # just another differentiable op, so any output (here the value) can be
    # differentiated with no hand-written sensitivity code.
    j_val = torch.autograd.functional.jacobian(
        lambda m: diff_mpc(x0=x0, params={"mass": m})[4],
        mass_param,
    )

    mo.md(
        f"""
        **Value at m = 1.5:** `{value_m.item():.4f}`

        **∂V/∂mass via `.backward()`:** `{mass_param.grad.item():.4f}`

        **∂V/∂mass via `autograd.functional.jacobian`:** `{j_val.reshape(-1).item():.4f}`

        Both routes agree — the gradient is exact (from the solver's KKT system),
        so the MPC can be dropped straight into a PyTorch training loop.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Policy gradient away from the constraints

    At $x_0 = [0.5,\,0]$ the optimal force rides the $\pm 0.5$ bound, so the
    policy is locally *flat* in the parameters and $\partial F_0^{*}/\partial p$
    is clipped towards zero.

    To see the gradient cleanly we move to a near-origin state
    $x_0 = [0.05,\,0]$, where the force stays strictly inside $(-0.5, 0.5)$, and
    sweep the (square-root) control-cost weight `r_diag_sqrt`. A larger weight
    penalises the force more, so both the policy $F_0^{*}$ and the value $V$
    vary **smoothly** with it.

    We solve the whole sweep in **one batched call** and read back the value, the
    policy, and both exact gradients (`du0_dp_global`, `dvalue_dp_global`) at every
    sweep point at once. Drag the slider below to move `r_diag_sqrt`: the tangent
    slides along both curves using those precomputed gradients.
    """)
    return


@app.cell
def _(N_SWEEP, diff_mpc, np, p_global_slice, torch):
    x0_far = torch.tensor([[0.05, 0.0]])  # near the origin — the force never saturates
    r_sweep = np.linspace(0.2, 1.2, N_SWEEP)  # one batched solve covers the sweep

    # One batched solve over the whole r-sweep. From its single context we read
    # back the value, the policy, AND both exact gradients at *every* sweep point
    # at once — so moving the slider below never triggers another solve.
    _x0_batch = x0_far.repeat(len(r_sweep), 1)
    _r_param = torch.tensor(r_sweep).reshape(-1, 1).requires_grad_(True)
    _ctx, _u0_sweep, _, _, _value_sweep = diff_mpc(
        x0=_x0_batch, params={"r_diag_sqrt": _r_param}
    )

    u0_curve = _u0_sweep.detach().numpy().ravel()  # policy F₀*(r), shape (B,)
    v_curve = _value_sweep.detach().numpy().ravel()  # value  V(r),   shape (B,)

    # Locate the r_diag_sqrt column in the flat p_global from the manager's
    # registration order (no hard-coded index).
    _col = p_global_slice(diff_mpc.parameter_manager, "r_diag_sqrt").start

    # Exact KKT sensitivities for the whole batch, straight off the single ctx:
    # du0_dp_global is (B, nu, P) and dvalue_dp_global is (B, 1, P).
    _du0_dp = diff_mpc.diff_mpc_fun.sensitivity(_ctx, "du0_dp_global")
    _dV_dp = diff_mpc.diff_mpc_fun.sensitivity(_ctx, "dvalue_dp_global")
    du0_dr_curve = _du0_dp[:, 0, _col]  # dF₀*/dr at each sweep point, (B,)
    dV_dr_curve = _dV_dp[:, 0, _col]  # dV/dr    at each sweep point, (B,)

    # Verify the batched sensitivities against batched autograd on the same solve.
    # Each batch element depends only on its own r_i, so differentiating the summed
    # output recovers the per-point gradients — one backward pass, no per-point loop.
    _du0_dr_auto = torch.autograd.grad(_u0_sweep.sum(), _r_param, retain_graph=True)[0]
    _dV_dr_auto = torch.autograd.grad(_value_sweep.sum(), _r_param)[0]
    assert np.allclose(du0_dr_curve, _du0_dr_auto.numpy().ravel(), rtol=1e-3, atol=1e-6)
    assert np.allclose(dV_dr_curve, _dV_dr_auto.numpy().ravel(), rtol=1e-3, atol=1e-6)

    # The force bound is inactive everywhere on this sweep.
    assert np.abs(u0_curve).max() < 0.5 - 1e-3
    return dV_dr_curve, du0_dr_curve, r_sweep, u0_curve, v_curve


@app.cell
def _(mo, r_sweep):
    # The slider snaps to the precomputed sweep points (gradients exist only there).
    r_slider = mo.ui.slider(
        start=0,
        stop=len(r_sweep) - 1,
        step=1,
        value=len(r_sweep) // 3,  # start mid-sweep, well away from the force bound
        label="sweep index for r_diag_sqrt",
        show_value=True,
    )
    return (r_slider,)


@app.cell
def _(
    dV_dr_curve,
    du0_dr_curve,
    mo,
    np,
    plt,
    r_slider,
    r_sweep,
    u0_curve,
    v_curve,
):
    _i = r_slider.value
    r0 = float(r_sweep[_i])
    u0_r0 = float(u0_curve[_i])
    v_r0 = float(v_curve[_i])
    du0_dr = float(du0_dr_curve[_i])
    dV_dr = float(dV_dr_curve[_i])

    grad_fig, grad_axes = plt.subplots(1, 2, figsize=(11, 4.5))
    _tan = np.array([r_sweep.min(), r_sweep.max()])

    grad_axes[0].plot(r_sweep, u0_curve, "-o", markersize=3, label="policy F₀*(r)")
    grad_axes[0].plot(
        _tan, u0_r0 + du0_dr * (_tan - r0), "--", color="tab:red",
        label=f"tangent  dF₀/dr = {du0_dr:.3f}",
    )
    grad_axes[0].plot([r0], [u0_r0], "s", color="tab:red")
    grad_axes[0].set_ylabel("First action F₀* [N]")
    grad_axes[0].set_title("Policy vs. control weight")

    grad_axes[1].plot(r_sweep, v_curve, "-o", markersize=3, color="tab:green", label="value V(r)")
    grad_axes[1].plot(
        _tan, v_r0 + dV_dr * (_tan - r0), "--", color="tab:red",
        label=f"tangent  dV/dr = {dV_dr:.3f}",
    )
    grad_axes[1].plot([r0], [v_r0], "s", color="tab:red")
    grad_axes[1].set_ylabel("Value V")
    grad_axes[1].set_title("Value vs. control weight")

    for ax_g in grad_axes:
        ax_g.set_xlabel("r_diag_sqrt")
        ax_g.grid(True, alpha=0.3)
        ax_g.legend()

    # Pin the y-axes to the data curves so the sliding tangent — which can shoot
    # far past the data where the gradient is steep — is clipped rather than
    # rescaling the axes on every drag.
    _u_lo, _u_hi = float(u0_curve.min()), float(u0_curve.max())
    _u_pad = (_u_hi - _u_lo) * 0.08
    grad_axes[0].set_ylim(_u_lo - _u_pad, _u_hi + _u_pad)

    _v_lo, _v_hi = float(v_curve.min()), float(v_curve.max())
    _v_pad = (_v_hi - _v_lo) * 0.08
    grad_axes[1].set_ylim(_v_lo - _v_pad, _v_hi + _v_pad)

    grad_fig.suptitle(f"Analytic tangent at x₀ = [0.05, 0], r_diag_sqrt = {r0:.3f}")
    grad_fig.tight_layout()
    mo.vstack([r_slider, grad_fig])
    return dV_dr, du0_dr, r0, u0_r0, v_r0


@app.cell(hide_code=True)
def _(dV_dr, du0_dr, mo, r0, u0_r0, v_r0):
    mo.md(f"""
    At `r_diag_sqrt = {r0:.3f}` (force `F₀* = {u0_r0:.4f}` N, strictly inside the
    ±0.5 bound, value `V = {v_r0:.4f}`) the exact gradients — read straight from
    the single batched solve — are:

    - **∂F₀/∂r:** `{du0_dr:.5f}`
    - **∂V/∂r:** `{dV_dr:.5f}`

    Drag the slider to move `r_diag_sqrt`: the tangent slides along both curves
    using gradients that were **all precomputed in one batched solve** (the batched
    `sensitivity(ctx, ...)` was cross-checked against batched autograd above), so
    every tangent is exact where no constraint binds. Contrast this with the
    saturated case above, where the force sits on the bound and its gradient would
    collapse to ≈ 0.

    **Next:** `04_heating_parameter_management.py` moves to a heating problem
    where parameters vary *across the horizon* — stagewise parameters, splits,
    and forecasts.
    """)
    return


if __name__ == "__main__":
    app.run()
