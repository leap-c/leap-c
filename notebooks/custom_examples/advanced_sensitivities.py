"""Advanced — the exact KKT sensitivity API.

The recommended way to differentiate through ``AcadosDiffMpcTorch`` is plain
autograd (``.backward()`` / ``torch.autograd.functional.jacobian``), as shown in
``getting_started/03_gradients_through_the_solver.py``. This notebook is the
**advanced** counterpart: it reaches past autograd into the solver's own exact
KKT sensitivities via ``diff_mpc.diff_mpc_fun.sensitivity(ctx, ...)``.

We (A) show the low-level API, (B) prove it is a numerically exact match to
autograd, and (C) pin down what each field actually returns — including the
easy-to-misread stage-summed semantics of ``du_dp_global`` — and compare the
timings. It uses the mass-spring-damper OCP from ``nb_utils.msd``.
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
    # Advanced: the exact KKT sensitivity API

    Autograd (getting_started notebook 03) is the recommended interface and
    should cover almost everything. Underneath it, the solver can hand back
    the **exact KKT sensitivities** directly:

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

    So why reach for it? To read exact Jacobian *blocks* — like `du0/dp` for
    **all** parameters at once — straight off a solve, even one that never
    built a torch graph (e.g. inside `torch.no_grad()`). Section C also pins
    down what the whole-trajectory fields (`du_dp_global`, `dx_dp_global`)
    really return, which is easy to misread.
    """)
    return


@app.cell
def _(mo):
    import sys
    import time

    sys.path.insert(0, str(mo.notebook_dir().parent))  # notebooks/ root -> nb_utils

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
        points — the kind of curves getting_started notebook 03 plots, here
        read straight from `ctx`.
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
    ## C. What the trajectory fields return — and the timings

    **Read this before using `du_dp_global` or `dx_dp_global`:** they do
    **not** return the per-stage trajectory Jacobian their names suggest.
    `sensitivity(ctx, "du_dp_global")` is a single *adjoint* (backward) pass
    seeded with ones over all stages, so it returns the **stage-summed**
    sensitivity

    $$\texttt{du\_dp\_global}[b] \;=\; \frac{\partial \sum_k u_k}{\partial p}
    \in \mathbb{R}^{n_u \times P},$$

    of shape `(B, nu, P)` — not the `(B, N·nu, P)` per-stage block. The cell
    below verifies this by summing the true Jacobian over stages. If you need
    the full per-stage Jacobian, no single KKT call returns it; use
    `torch.autograd.functional.jacobian` (or one autograd pass per stage, as
    the prosumer example does).

    With that settled, three representative jobs, each timed end-to-end
    (solve + gradient), median over a few repeats. Because every repeat
    re-solves, `sensitivity()` recomputes rather than returning its per-`ctx`
    cache — a fair comparison.

    1. **Scalar value gradient** `dV/dp` — one adjoint solve either way.
    2. **Stage-summed control sensitivity** `d(Σₖ uₖ)/dp` — one autograd
       backward pass on `u.sum()` vs. one `du_dp_global` call: the same
       aggregate, computed both ways.
    3. **Full per-stage trajectory Jacobian** `duₖ/dp` — autograd only
       (`functional.jacobian`, one backward pass per output component).
    """)
    return


@app.cell
def _(B, diff_mpc, mo, np, p_global_slice, time, torch):
    _x0 = torch.tensor([[0.05, 0.0]]).repeat(B, 1)
    _r_np = np.linspace(0.2, 1.2, B).reshape(-1, 1)

    # --- First, verify the stage-summed semantics of du_dp_global. ---------
    _r = torch.tensor(_r_np)
    _ctx, _, _, _, _ = diff_mpc(x0=_x0, params={"r_diag_sqrt": _r})
    du_dp_sum = diff_mpc.diff_mpc_fun.sensitivity(_ctx, "du_dp_global")  # (B, nu, P)

    # The true per-stage Jacobian, summed over stages, must reproduce it.
    _jac_full = torch.autograd.functional.jacobian(
        lambda p: diff_mpc(x0=_x0, params={"r_diag_sqrt": p})[3], _r
    )  # (B, N, nu, B, 1) — block-diagonal in the batch
    _col = p_global_slice(diff_mpc.parameter_manager, "r_diag_sqrt").start
    _stage_summed_auto = np.array(
        [_jac_full[i].sum(dim=0)[0, i, 0].item() for i in range(B)]
    )
    assert np.allclose(du_dp_sum[:, 0, _col], _stage_summed_auto, rtol=1e-3, atol=1e-6)

    # --- Timings. -----------------------------------------------------------
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

    def _sum_autograd():
        # The same stage-summed aggregate du_dp_global returns: one backward pass.
        r = torch.tensor(_r_np).requires_grad_(True)
        _, _, _, u, _ = diff_mpc(x0=_x0, params={"r_diag_sqrt": r})
        u.sum().backward()

    def _sum_kkt():
        c, _, _, _, _ = diff_mpc(x0=_x0, params={"r_diag_sqrt": torch.tensor(_r_np)})
        diff_mpc.diff_mpc_fun.sensitivity(c, "du_dp_global")

    def _jac_autograd():
        # The true per-stage Jacobian — autograd only, no single KKT call.
        r = torch.tensor(_r_np)
        torch.autograd.functional.jacobian(
            lambda p: diff_mpc(x0=_x0, params={"r_diag_sqrt": p})[3], r
        )

    # Warm up once — the first acados calls take the slow path.
    for _f in (_solve_only, _val_autograd, _val_kkt, _sum_autograd, _sum_kkt, _jac_autograd):
        _f()

    t_solve = _median(_solve_only)
    t_val_auto = _median(_val_autograd)
    t_val_kkt = _median(_val_kkt)
    t_sum_auto = _median(_sum_autograd)
    t_sum_kkt = _median(_sum_kkt)
    t_jac_auto = _median(_jac_autograd)

    mo.md(
        r"Cross-check passed: summing the true per-stage Jacobian over stages "
        r"reproduces `du_dp_global` — confirming its **stage-summed** semantics."
    )
    return t_jac_auto, t_solve, t_sum_auto, t_sum_kkt, t_val_auto, t_val_kkt


@app.cell
def _(mo, plt, t_jac_auto, t_solve, t_sum_auto, t_sum_kkt, t_val_auto, t_val_kkt):
    time_fig, time_ax = plt.subplots(figsize=(8, 4))

    _labels = ["value gradient\n∂V/∂p", "stage-summed sens.\n∂(Σₖuₖ)/∂p"]
    _auto = [t_val_auto, t_sum_auto]
    _kkt = [t_val_kkt, t_sum_kkt]
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
    _sum_ratio = t_sum_auto / t_sum_kkt

    mo.vstack([
        time_fig,
        mo.md(
            f"""
            | job | autograd | KKT `sensitivity()` | autograd / KKT |
            |---|---|---|---|
            | value gradient ∂V/∂p | {t_val_auto:.1f} ms | {t_val_kkt:.1f} ms | {_val_ratio:.2f}× |
            | stage-summed ∂(Σₖuₖ)/∂p | {t_sum_auto:.1f} ms | {t_sum_kkt:.1f} ms | {_sum_ratio:.2f}× |
            | full per-stage Jacobian ∂uₖ/∂p | {t_jac_auto:.1f} ms | — (no single call) | — |

            For the two **adjoint-sized** jobs the routes are comparable — both
            are dominated by the solve, and autograd's convenience is essentially
            free. The **full per-stage Jacobian** is a genuinely bigger job:
            reverse mode pays one backward pass per trajectory component
            ({t_jac_auto / t_solve:.0f}× the bare solve here), and no single
            KKT call replaces it.
            """
        ),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## When to reach for `sensitivity(ctx, ...)`

    Default to autograd. Drop to the low-level API when you specifically need
    the **`du0/dp` Jacobian block for all parameters at once**, want gradients
    off a solve that never built a torch graph (`torch.no_grad()`), or want to
    precompute a batch of exact gradients off one `ctx`. Remember the
    semantics: `du_dp_global`/`dx_dp_global` are **stage-summed** adjoints
    `(B, nu, P)`/`(B, nx, P)` — a full per-stage trajectory Jacobian always
    goes through autograd. The cost of the low-level door is the coupling it
    exposes: the raw `ctx`, `.diff_mpc_fun`, and flat `p_global` column
    bookkeeping via `p_global_slice`.

    **Next:** `prosumer_home_energy.py` puts both doors to work — a prosumer
    MPC reports the full Jacobian of its planned grid exchange with respect
    to a 24-hour tariff, built from per-stage backward passes and
    cross-checked against the stage-summed adjoint call.
    """)
    return


if __name__ == "__main__":
    app.run()
