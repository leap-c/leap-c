"""Part 7 (advanced) — the exact KKT sensitivity API.

The recommended way to differentiate through ``AcadosDiffMpcTorch`` is plain
autograd (``.backward()`` / ``torch.autograd.functional.jacobian``), as shown in
``03_msd_sensitivities.py``. This notebook is the **advanced** counterpart: it
reaches past autograd into the solver's own exact KKT sensitivities via
``diff_mpc.diff_mpc_fun.sensitivity(ctx, ...)``.

We (A) show the low-level API, (B) prove it is a numerically exact match to
autograd, and (C) compare the timings — so you know when the extra machinery is
worth it. It reuses the mass-spring-damper OCP from ``nb_utils.msd``.
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
    # 07 — advanced: the exact KKT sensitivity API

    Autograd (notebook 03) is the recommended interface and should cover almost
    everything. Underneath it, the solver can hand back the **exact KKT
    sensitivities** directly:

    ```python
    diff_mpc.diff_mpc_fun.sensitivity(ctx, "du0_dp_global")   # (B, nu, P)
    diff_mpc.diff_mpc_fun.sensitivity(ctx, "dvalue_dp_global")  # (B, 1, P)
    ```

    This is a **lower-level** entry point. Two things make it "advanced":

    - it reaches into internals — the raw solver context `ctx` (element 0 of the
      `diff_mpc(...)` return) and the `.diff_mpc_fun` object;
    - it returns gradients against the **flat `p_global` vector** of length `P`,
      so you must locate a parameter's columns yourself. The notebook helper
      `p_global_slice(manager, name)` does that from the manager's registration
      order. Autograd, by contrast, hands the gradient straight back on the
      parameter tensor you passed (`param.grad`) — no column bookkeeping.

    So why reach for it? For a **full Jacobian of a vector output** (Section C):
    one call returns the whole block, whereas reverse-mode autograd pays one
    backward pass per output component.
    """)
    return


@app.cell
def _():
    import time

    import matplotlib.pyplot as plt
    import numpy as np
    import torch

    from nb_utils.msd import build_msd_ocp
    from nb_utils.params import p_global_slice

    from leap_c.torch import AcadosDiffMpcTorch

    torch.set_default_dtype(torch.float64)
    return AcadosDiffMpcTorch, build_msd_ocp, np, p_global_slice, plt, time, torch


@app.cell
def _(AcadosDiffMpcTorch, build_msd_ocp, torch):
    N_HORIZON = 20
    DT = 0.1
    B = 8  # a small batched r_diag_sqrt sweep

    ocp, manager = build_msd_ocp(N_horizon=N_HORIZON, dt=DT)

    diff_mpc = AcadosDiffMpcTorch(
        ocp,
        manager,
        dtype=torch.float64,
        n_batch_init=B,
        verbose=False,
    )
    return B, diff_mpc


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## A. Reading the exact sensitivities off the context

    One batched solve over a small `r_diag_sqrt` sweep, started near the origin
    (`x0 = [0.05, 0]`) so the force stays strictly inside its ±0.5 bound and the
    gradients are smooth. Element 0 of the return is the solver context `ctx`;
    we feed it straight to `sensitivity(ctx, ...)` and index the `r_diag_sqrt`
    column of the flat `p_global` with `p_global_slice`.
    """)
    return


@app.cell
def _(B, diff_mpc, mo, np, p_global_slice, torch):
    x0 = torch.tensor([[0.05, 0.0]]).repeat(B, 1)  # near the origin — force never saturates
    r_sweep = np.linspace(0.2, 1.2, B)
    r_param = torch.tensor(r_sweep).reshape(-1, 1).requires_grad_(True)

    # One batched solve. ctx (element 0) carries everything the sensitivity
    # solver needs; u0 and value are the outputs we will differentiate.
    ctx, u0, _, _, value = diff_mpc(x0=x0, params={"r_diag_sqrt": r_param})

    # Locate the r_diag_sqrt column in the flat p_global (registration order).
    col = p_global_slice(diff_mpc.parameter_manager, "r_diag_sqrt").start

    # Exact KKT sensitivities for the whole batch, straight off the single ctx.
    du0_dp = diff_mpc.diff_mpc_fun.sensitivity(ctx, "du0_dp_global")  # (B, nu, P)
    dV_dp = diff_mpc.diff_mpc_fun.sensitivity(ctx, "dvalue_dp_global")  # (B, 1, P)
    du0_dr_kkt = du0_dp[:, 0, col]  # dF0*/dr at each sweep point, (B,)
    dV_dr_kkt = dV_dp[:, 0, col]  # dV/dr    at each sweep point, (B,)

    mo.md(
        f"""
        Flat `p_global` has **P = {du0_dp.shape[-1]}** columns; `r_diag_sqrt`
        sits at column **{col}**.

        - `du0_dp_global` shape: `{tuple(du0_dp.shape)}`  (B, nu, P)
        - `dvalue_dp_global` shape: `{tuple(dV_dp.shape)}`  (B, 1, P)

        Slicing column {col} gives `dF0*/dr` and `dV/dr` at each of the {B} sweep
        points — the same curves notebook 03 plots, here read straight from `ctx`.
        """
    )
    return dV_dr_kkt, du0_dr_kkt, r_param, u0, value


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## B. Exact match against autograd

    The recommended autograd route gives the identical numbers. Each batch
    element depends only on its own `r_i`, so one backward pass over the summed
    output recovers all per-point gradients — no per-point loop, no
    `p_global_slice`.
    """)
    return


@app.cell
def _(dV_dr_kkt, du0_dr_kkt, mo, np, r_param, torch, u0, value):
    # Recommended route: autograd on the summed outputs. Gradients arrive shaped
    # like r_param — no flat-p_global indexing needed.
    du0_dr_auto = torch.autograd.grad(u0.sum(), r_param, retain_graph=True)[0].reshape(-1)
    dV_dr_auto = torch.autograd.grad(value.sum(), r_param)[0].reshape(-1)

    max_du0 = float(np.abs(du0_dr_kkt - du0_dr_auto.numpy()).max())
    max_dV = float(np.abs(dV_dr_kkt - dV_dr_auto.numpy()).max())

    # The two APIs are the same exact gradient from the KKT system.
    assert np.allclose(du0_dr_kkt, du0_dr_auto.numpy(), rtol=1e-3, atol=1e-6)
    assert np.allclose(dV_dr_kkt, dV_dr_auto.numpy(), rtol=1e-3, atol=1e-6)

    mo.md(
        f"""
        Max abs difference, KKT `sensitivity()` vs autograd:

        - `dF0*/dr`: `{max_du0:.2e}`
        - `dV/dr`: `{max_dV:.2e}`

        Numerically identical — both read the same exact gradient off the solver's
        KKT system. For scalar objectives like these, autograd is the simpler
        choice.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## C. Timing: when the low-level API pays off

    Two representative jobs, each timed end-to-end (solve + gradient), median
    over a few repeats. Because every repeat re-solves, `sensitivity()`
    recomputes rather than returning its per-`ctx` cache — a fair comparison.

    1. **Scalar value gradient** `dV/dp` — one adjoint solve either way, so the
       gradient is a small marginal cost on top of the solve. Autograd wins on
       simplicity here.
    2. **Full control-trajectory Jacobian** `du/dp` — a *vector* output.
       Reverse-mode autograd (`torch.autograd.functional.jacobian`) pays one
       backward pass per output component; the KKT solver returns the whole
       `(B, N·nu, P)` block in a single call.
    """)
    return


@app.cell
def _(B, diff_mpc, np, time, torch):
    _x0 = torch.tensor([[0.05, 0.0]]).repeat(B, 1)
    _r_np = np.linspace(0.2, 1.2, B).reshape(-1, 1)

    def _median(fn, repeats=5):
        samples = []
        for _ in range(repeats):
            _t0 = time.perf_counter()
            fn()
            samples.append(time.perf_counter() - _t0)
        return float(np.median(samples)) * 1e3  # ms

    def _solve_only():
        with torch.no_grad():
            diff_mpc(x0=_x0, params={"r_diag_sqrt": torch.tensor(_r_np)})

    def _val_autograd():
        r = torch.tensor(_r_np).requires_grad_(True)
        _, _, _, _, v = diff_mpc(x0=_x0, params={"r_diag_sqrt": r})
        v.sum().backward()

    def _val_kkt():
        c, _, _, _, _ = diff_mpc(x0=_x0, params={"r_diag_sqrt": torch.tensor(_r_np)})
        diff_mpc.diff_mpc_fun.sensitivity(c, "dvalue_dp_global")

    def _jac_autograd():
        r = torch.tensor(_r_np)
        torch.autograd.functional.jacobian(
            lambda p: diff_mpc(x0=_x0, params={"r_diag_sqrt": p})[3], r  # [3] = full control traj
        )

    def _jac_kkt():
        c, _, _, _, _ = diff_mpc(x0=_x0, params={"r_diag_sqrt": torch.tensor(_r_np)})
        diff_mpc.diff_mpc_fun.sensitivity(c, "du_dp_global")

    # Warm up once — the first acados calls take the slow path.
    for _f in (_solve_only, _val_autograd, _val_kkt, _jac_autograd, _jac_kkt):
        _f()

    t_solve = _median(_solve_only)
    t_val_auto = _median(_val_autograd)
    t_val_kkt = _median(_val_kkt)
    t_jac_auto = _median(_jac_autograd)
    t_jac_kkt = _median(_jac_kkt)
    return t_jac_auto, t_jac_kkt, t_solve, t_val_auto, t_val_kkt


@app.cell
def _(mo, plt, t_jac_auto, t_jac_kkt, t_solve, t_val_auto, t_val_kkt):
    time_fig, time_ax = plt.subplots(figsize=(8, 4))

    _labels = ["value gradient\n∂V/∂p", "full traj. Jacobian\n∂u/∂p"]
    _auto = [t_val_auto, t_jac_auto]
    _kkt = [t_val_kkt, t_jac_kkt]
    _x = range(len(_labels))
    _w = 0.38

    time_ax.bar([i - _w / 2 for i in _x], _auto, _w, label="autograd", color="tab:blue")
    time_ax.bar([i + _w / 2 for i in _x], _kkt, _w, label="KKT sensitivity()", color="tab:orange")
    time_ax.axhline(t_solve, ls="--", lw=1.0, color="gray", label=f"solve only ({t_solve:.1f} ms)")
    time_ax.set_xticks(list(_x))
    time_ax.set_xticklabels(_labels)
    time_ax.set_ylabel("median wall time [ms]")
    time_ax.set_title("Autograd vs. exact KKT sensitivity (solve + gradient)")
    time_ax.legend()
    time_ax.grid(True, axis="y", alpha=0.3)
    time_fig.tight_layout()

    _val_ratio = t_val_auto / t_val_kkt
    _jac_ratio = t_jac_auto / t_jac_kkt

    mo.vstack([
        time_fig,
        mo.md(
            f"""
            | job | autograd | KKT `sensitivity()` | autograd / KKT |
            |---|---|---|---|
            | value gradient ∂V/∂p | {t_val_auto:.1f} ms | {t_val_kkt:.1f} ms | {_val_ratio:.2f}× |
            | full traj. Jacobian ∂u/∂p | {t_jac_auto:.1f} ms | {t_jac_kkt:.1f} ms | {_jac_ratio:.2f}× |

            For the **scalar** value gradient the two are close and both are
            dominated by the solve — autograd's convenience is essentially free.
            For the **full Jacobian** the single KKT call is markedly faster,
            because autograd must run one backward pass per trajectory component.
            """
        ),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## When to reach for `sensitivity(ctx, ...)`

    Default to autograd. Drop to the low-level API when you specifically need a
    **full Jacobian of a vector output** (whole-trajectory sensitivities, or
    `du0/dp` across many controls), want to avoid building a torch backward
    graph, or want to precompute a batch of exact gradients off one solve. The
    cost is the coupling it exposes: the raw `ctx`, `.diff_mpc_fun`, and flat
    `p_global` column bookkeeping via `p_global_slice`.

    **Next:** notebook 08 puts both doors to work — a prosumer MPC reports
    the full Jacobian of its planned grid exchange with respect to a
    24-hour tariff, built from per-stage backward passes and cross-checked
    against the adjoint call.
    """)
    return


if __name__ == "__main__":
    app.run()
