"""Part 6 — from layer to planner: comfort bounds and the closed loop.

``AcadosDiffMpcTorch`` solves one OCP; a *planner* is the thin class you write
around it that turns an observation into an action, carries the warm start
from step to step, and handles failures. This notebook builds one for the
heating problem — now with a time-varying, slacked comfort band — and closes
the loop against a house that does **not** match the model.

NOTE: the ``HeatingPlanner`` taught here is kept in sync with its importable
copy ``nb_utils.planner.HeatingPlanner`` used by notebooks 07 and 08.
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
    # 06 — the planner interface

    Nothing in leap-c is called "planner": the convention is a small class
    *you* own, wrapping the layer with exactly the interface your
    application needs — here `forward(obs) → heating power`. Three concerns
    live in it:

    1. **translation** — pull the forecast windows out of the observation
       and shape them into `params`,
    2. **warm starts** — feed the previous solve's `ctx` back in, so each
       receding-horizon step starts from an almost-correct iterate,
    3. **failure handling** — check `ctx.status`, and never carry a failed
       context forward.

    The occupant no longer tracks a setpoint but wants the temperature
    inside a **comfort band** that relaxes at night — a constraint, not a
    cost preference.
    """)
    return


@app.cell
def _(mo):
    import sys

    sys.path.insert(0, str(mo.notebook_dir().parent))  # notebooks/ root -> nb_utils

    import matplotlib.pyplot as plt
    import numpy as np
    import torch

    from nb_utils.env import HouseEnv
    from nb_utils.heating import R_THERMAL, build_heating_ocp_comfort_band

    from leap_c.torch import AcadosDiffMpcTorch

    return (
        AcadosDiffMpcTorch,
        HouseEnv,
        R_THERMAL,
        build_heating_ocp_comfort_band,
        np,
        plt,
        torch,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The comfort-band OCP

    The builder lives in `nb_utils.heating.build_heating_ocp_comfort_band`;
    it is the OCP of notebook 02 with the tracking cost replaced by a pure
    energy cost and these lines added — the part worth reading:

    ```python
    # Bounds are stagewise *parameters* -> time-varying per solve. leap-c
    # writes state bounds only at stage 0, so a slacked h-constraint is
    # the supported way to get time-varying, softly-enforced state bounds.
    t_lower = manager.register_parameter("t_lower", np.array([17.0]))   # model.p
    t_upper = manager.register_parameter("t_upper", np.array([21.0]))   # model.p
    R = manager.register_parameter(                                     # p_global
        "R", np.array([2.0]), differentiable=True
    )
    comfort_margin = manager.register_parameter(                        # p_global
        "comfort_margin", np.array([0.0]), differentiable=True
    )

    # T >= t_lower + comfort_margin  and  T <= t_upper, both softened.
    # (acados applies con_h_expr at stages 1..N-1; stage 0 gets no
    # h-constraint — x0 is fixed anyway. con_h_expr_0 would add one.)
    ocp.model.con_h_expr = ca.vertcat(T - t_lower - comfort_margin, t_upper - T)
    ocp.constraints.lh = np.zeros(2)
    ocp.constraints.uh = np.full(2, ACADOS_INFTY)
    ocp.constraints.idxsh = np.array([0, 1])          # slack both rows
    ocp.cost.Zl = ocp.cost.Zu = np.full(2, 1e2)       # quadratic violation penalty
    # ... mirrored with _e for the terminal stage, and
    # cost = price_weight * price * q + eps_reg * q**2
    ```

    Two design notes:

    - the slack weight `Zl` is a **fixed numeric** — acados cannot make it a
      parameter. That is why the learnable comfort knob is the bound
      *margin* (differentiable, in the constraint) rather than a bound
      *weight*,
    - the pure economic cost is linear in $q$, so a small `eps_reg * q**2`
      supplies the positive-definite curvature the solver needs — needed
      *here* because nothing else provides it: notebook 02's tracking cost
      had $(T - T_\mathrm{set})^2$ feeding curvature through the dynamics,
      while this cost has none whenever the band constraints are inactive
      (the battery example makes the full argument).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The planner

    A `torch.nn.Module` holding the layer plus the three differentiable
    knobs as `nn.Parameter`s — `R` (the planner's belief about the
    envelope), `price_weight` (how much the occupant cares about cost) and
    `comfort_margin` (how far above the scheduled bound they live).
    `forward` handles batches for training; `act` is the single-step
    closed-loop interface with warm starting and the failure rule.

    A warm-started solve that fails is automatically retried once from the
    default initializer inside leap-c (a custom `AcadosDiffMpcInitializer`
    can replace it). If the status is *still* nonzero, the one safe move is
    to **drop the stale context** so the next step starts fresh — that is
    the whole failure policy of `act`.
    """)
    return


@app.cell
def _(AcadosDiffMpcTorch, build_heating_ocp_comfort_band, np, torch):
    class HeatingPlanner(torch.nn.Module):
        """Wraps the comfort-band heating MPC into ``forward(obs) -> action``.

        NOTE: taught here, importable as ``nb_utils.planner.HeatingPlanner``
        (kept in sync).
        """

        def __init__(self, N_horizon=32, dt=0.25, q_max=12.0,
                     name="heating_planner", n_batch_init=256,
                     discount_factor=None):
            super().__init__()
            ocp, manager = build_heating_ocp_comfort_band(
                N_horizon, dt=dt, q_max=q_max, name=name
            )
            self.diff_mpc = AcadosDiffMpcTorch(
                ocp, manager, dtype=torch.float64,
                n_batch_init=n_batch_init,
                discount_factor=discount_factor, verbose=False,
            )
            self.R = torch.nn.Parameter(torch.tensor([2.0], dtype=torch.float64))
            self.price_weight = torch.nn.Parameter(
                torch.tensor([1.0], dtype=torch.float64)
            )
            self.comfort_margin = torch.nn.Parameter(
                torch.tensor([0.0], dtype=torch.float64)
            )
            self.ctx = None  # warm start carried across closed-loop steps

        def forward(self, obs, ctx=None):
            """Batched solve: obs holds "T" (B,) and the four (B, N+1) windows.

            "T" may also be a torch tensor — its autograd graph is preserved,
            which notebook 08 exploits to differentiate through rollouts.
            """
            x0 = torch.as_tensor(obs["T"], dtype=torch.float64).reshape(-1, 1)
            batch_size = x0.shape[0]
            params = {
                # Forecast windows: non-differentiable stagewise, (B, N+1, 1) numpy.
                "outdoor_temp": np.asarray(obs["outdoor"], dtype=np.float64)[..., None],
                "price": np.asarray(obs["price"], dtype=np.float64)[..., None],
                "t_lower": np.asarray(obs["t_lower"], dtype=np.float64)[..., None],
                "t_upper": np.asarray(obs["t_upper"], dtype=np.float64)[..., None],
                # Learnable knobs: one shared value, expanded over the batch
                # (backward sums the per-sample gradients — exactly the
                # dataset gradient a training loop wants).
                "R": self.R.expand(batch_size, 1),
                "price_weight": self.price_weight.expand(batch_size, 1),
                "comfort_margin": self.comfort_margin.expand(batch_size, 1),
            }
            ctx, u0, x, u, value = self.diff_mpc(x0, params=params, ctx=ctx)
            return u0, ctx, (x, u, value)

        def act(self, obs):
            """One warm-started closed-loop step for a single observation."""
            batched = {key: np.asarray(obs[key])[None] for key in
                       ("T", "outdoor", "price", "t_lower", "t_upper")}
            with torch.no_grad():
                u0, ctx, _ = self.forward(batched, ctx=self.ctx)
            self.ctx = ctx if ctx.status[0] == 0 else None
            return float(u0[0, 0])

    return (HeatingPlanner,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The world it will face

    `nb_utils.env.HouseEnv` is a plain `reset`/`step` class (no gym
    dependency): the **same R1C1 structure** the planner believes in, plus
    everything it does not know —

    - a leakier envelope: true $R = 1.4$ K/kW vs. the model's $2.0$,
    - occupancy heat gains between 18:00 and 22:00,
    - a small AR(1) thermal disturbance.

    Observations are exactly the planner's input dict; the reward is the
    negative of `price·q·Δt + w·(band violation)·Δt`. First, one solve from
    a cold Saturday 00:00 start:
    """)
    return


@app.cell
def _(HeatingPlanner, HouseEnv, np, plt):
    planner = HeatingPlanner(N_horizon=32, name="heating_planner_nb06", n_batch_init=1)
    env = HouseEnv(n_days=2, N_forecast=32, seed=0)

    obs0 = env.reset(start=0, T0=19.0)
    _u0, _ctx, (_x, _u, _value) = planner(
        {key: np.asarray(obs0[key])[None] for key in
         ("T", "outdoor", "price", "t_lower", "t_upper")}
    )

    _t = 0.25 * np.arange(33)
    plan_fig, plan_axes = plt.subplots(2, 1, figsize=(8, 4.8), sharex=True)
    plan_axes[0].fill_between(_t, obs0["t_lower"], obs0["t_upper"],
                              color="tab:green", alpha=0.12, label="comfort band")
    plan_axes[0].plot(_t, _x[0, :, 0].detach().numpy(), "-o", markersize=3,
                      color="tab:blue", label="planned T")
    plan_axes[0].set_ylabel("Room temp [degC]")
    plan_axes[0].legend(fontsize=8)
    plan_axes[1].step(_t[:-1], _u[0, :, 0].detach().numpy(), where="post",
                      color="tab:orange")
    plan_axes[1].set_ylabel("Heating [kW]")
    plan_axes[1].set_xlabel("Time since midnight [h]")
    for _ax in plan_axes:
        _ax.grid(True, alpha=0.3)
    plan_fig.suptitle(
        f"First plan (status {_ctx.status.tolist()}): ride the night band, "
        "pre-heat for the 08:00 step"
    )
    plan_fig.tight_layout()
    plan_fig
    return env, planner


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The plan hugs the *scheduled* lower bound — heating costs money, so the
    economic optimum is the cheapest temperature the band allows — and
    starts heating **before** the 08:00 night→day step it sees coming in
    its window.

    ## Closing the loop

    Receding horizon: observe, solve (warm-started), apply the first
    decision, step the world, repeat — 2 days, 192 solves. We run it twice:

    - against a **perfect** house (true $R$ = model $R$, no gains, no
      noise) — the sanity check,
    - against the **true** house — model mismatch and all.
    """)
    return


@app.cell
def _(HouseEnv, env, np, planner):
    def run_closed_loop(world, controller, n_steps=192, T0=19.0):
        controller.ctx = None  # fresh warm-start chain per run
        obs = world.reset(start=0, T0=T0)
        log = {"T": [obs["T"]], "q": [], "reward": [], "energy": [], "discomfort": []}
        for _ in range(n_steps):
            q = controller.act(obs)
            obs, reward, done, info = world.step(q)
            log["T"].append(obs["T"])
            log["q"].append(q)
            log["reward"].append(reward)
            log["energy"].append(info["energy_eur"])
            log["discomfort"].append(info["discomfort_kh"])
            if done:
                break
        return {key: np.asarray(vals) for key, vals in log.items()}

    env_perfect = HouseEnv(
        n_days=2, N_forecast=32, R_true=2.0, gain_kw=0.0, noise_std=0.0, seed=0
    )
    log_perfect = run_closed_loop(env_perfect, planner)
    log_true = run_closed_loop(env, planner)

    print(
        f"perfect model: {log_perfect['energy'].sum():6.2f} EUR energy, "
        f"{log_perfect['discomfort'].sum():5.2f} Kh discomfort\n"
        f"true house:    {log_true['energy'].sum():6.2f} EUR energy, "
        f"{log_true['discomfort'].sum():5.2f} Kh discomfort"
    )
    return log_perfect, log_true, run_closed_loop


@app.cell
def _(env, log_perfect, log_true, np, plt):
    _n = len(log_true["q"])
    _t_state = 0.25 * np.arange(_n + 1)
    _t_ctrl = 0.25 * np.arange(_n)

    cl_fig, cl_axes = plt.subplots(3, 1, figsize=(9, 7.5), sharex=True)
    cl_axes[0].fill_between(_t_state, env.t_lower[: _n + 1], env.t_upper[: _n + 1],
                            color="tab:green", alpha=0.12, label="comfort band")
    cl_axes[0].plot(_t_state, log_perfect["T"], color="tab:gray", lw=1.2,
                    label="perfect model")
    cl_axes[0].plot(_t_state, log_true["T"], color="tab:blue", lw=1.4,
                    label="true house (leakier + gains + noise)")
    cl_axes[0].set_ylabel("Room temp [degC]")
    cl_axes[0].legend(fontsize=8, loc="lower right")

    cl_axes[1].step(_t_ctrl, log_perfect["q"], where="post", color="tab:gray", lw=1.0)
    cl_axes[1].step(_t_ctrl, log_true["q"], where="post", color="tab:orange", lw=1.2)
    cl_axes[1].set_ylabel("Heating [kW]")
    _ax_price = cl_axes[1].twinx()
    _ax_price.step(_t_ctrl, env.price[:_n], where="post", color="tab:purple", alpha=0.5)
    _ax_price.set_ylabel("Price [EUR/kWh]", color="tab:purple")

    cl_axes[2].plot(_t_ctrl, np.cumsum(-log_perfect["reward"]), color="tab:gray",
                    label="perfect model")
    cl_axes[2].plot(_t_ctrl, np.cumsum(-log_true["reward"]), color="tab:red",
                    label="true house")
    cl_axes[2].set_ylabel("Cumulative cost [EUR]")
    cl_axes[2].set_xlabel("Time since midnight [h]")
    cl_axes[2].legend(fontsize=8)
    for _ax in cl_axes:
        _ax.grid(True, alpha=0.3)
    cl_fig.suptitle("Two days of receding-horizon control: perfect vs. true house")
    cl_fig.tight_layout()
    cl_fig
    return


@app.cell(hide_code=True)
def _(R_THERMAL, mo):
    mo.md(rf"""
    Against the perfect model the planner rides the band edge exactly — by
    design: the optimum of a pure energy cost is the coldest allowed
    temperature. Against the true house the same confidence backfires: the
    planner believes $R = {R_THERMAL}$, the house loses heat ~40 % faster,
    so between two solves the temperature falls *below* the band — a
    persistent sag the feedback only partially repairs (and the evening
    occupancy gains push the other way during the price peak). The cost gap
    between the two curves is the price of the model mismatch.

    Every solve in both runs converged (statuses were all zero — the OCP is
    a well-conditioned QP). The `act` failure rule exists for the day that
    stops being true: nonlinear models, tighter constraints, aggressive
    horizons.

    ## Recap

    - a planner = your thin class: obs → `params` → layer → action,
    - `ctx` warm starts make the receding horizon cheap; drop the context
      on failure instead of carrying it forward,
    - time-varying comfort bounds = stagewise bound *parameters* inside a
      slacked h-constraint,
    - a perfectly rational MPC with a wrong model rides its imagined band
      straight into real discomfort.

    **Next:** `07_imitation_learning.py` — the occupant is not a cost
    function but a *behavior*. Clone it: learn `price_weight` and
    `comfort_margin` from thermostat data, by backpropagating through the
    planner.
    """)
    return


if __name__ == "__main__":
    app.run()
