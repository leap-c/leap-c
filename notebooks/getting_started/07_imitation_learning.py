"""Part 7 — teaching the MPC by example (behavior cloning).

The occupant of notebook 06's house is not a cost function — they are a
thermostat habit. This notebook records their behavior and trains the
planner's two differentiable parameters so the MPC *imitates* them: a
regression whose model is an optimization solver, trained by
backpropagating through it.
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
    # 07 — imitation learning through the solver

    **Behavior cloning** is supervised learning on expert decisions: collect
    pairs $(o_t, u_t^\mathrm{expert})$, then fit a policy by minimizing

    $$\mathcal{L}(\theta) = \frac{1}{B}\sum_t
    \big\| u_0(o_t;\, \theta) - u_t^\mathrm{expert} \big\|^2 .$$

    Our policy class is the `HeatingPlanner` from notebook 06 — an MPC —
    and $\theta$ its three differentiable OCP parameters: `R` (the model's
    envelope belief), `price_weight` and `comfort_margin`. Because
    `u_0(o; \theta)` is differentiable through the solver, this is just
    PyTorch: forward, loss, `.backward()`, Adam.

    The payoff is interpretability: the fitted $\theta$ *is a statement
    about the occupant and their house* — how much they care about prices,
    how warm they actually live above the schedule, and how fast the house
    really loses heat.
    """)
    return


@app.cell
def _(mo):
    import sys

    sys.path.insert(0, str(mo.notebook_dir().parent))  # notebooks/ root -> nb_utils

    import matplotlib.pyplot as plt
    import numpy as np
    import torch

    from nb_utils.env import HouseEnv, collect_dataset
    from nb_utils.planner import HeatingPlanner

    from leap_c.utils.collate import collate_torch

    return HeatingPlanner, HouseEnv, collate_torch, collect_dataset, np, plt, torch


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The expert: an occupant who knows their house

    `nb_utils.env.thermostat_expert` is the occupant: a modulating
    thermostat holding the room `0.7` K above the *scheduled* lower bound.
    They have lived here long enough to know the house — the baseline power
    they set is the true steady-state demand `(T_pref − T_out) / R_true`,
    nudged proportionally when the room feels off. And they are completely
    **price-blind**. `collect_dataset` rolls this habit through two days in
    the true house and records `(observation, action)` pairs.
    """)
    return


@app.cell
def _(HouseEnv, collect_dataset, np, plt):
    env = HouseEnv(n_days=2, N_forecast=32, seed=0)
    data = collect_dataset(env, n_steps=192, start=0, T0=19.0)
    B_DATA = len(data["T"])

    _t = 0.25 * np.arange(B_DATA)
    exp_fig, exp_axes = plt.subplots(2, 1, figsize=(9, 5), sharex=True)
    exp_axes[0].fill_between(_t, data["t_lower"][:, 0], data["t_upper"][:, 0],
                             color="tab:green", alpha=0.12, label="comfort band")
    exp_axes[0].plot(_t, data["T"], color="tab:blue", lw=1.2, label="room temp")
    exp_axes[0].set_ylabel("Room temp [degC]")
    exp_axes[0].legend(fontsize=8, loc="lower right")
    exp_axes[1].step(_t, data["u_expert"], where="post", color="tab:orange")
    exp_axes[1].set_ylabel("Expert heating [kW]")
    exp_axes[1].set_xlabel("Time since midnight [h]")
    for _ax in exp_axes:
        _ax.grid(True, alpha=0.3)
    exp_fig.suptitle(f"The occupant's two days — {B_DATA} (obs, action) pairs")
    exp_fig.tight_layout()
    exp_fig
    return B_DATA, data, env


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The training loop

    Full-batch: all 192 observations go through **one batched solve per
    epoch** (this is why `n_batch_init` is sized to the dataset). Three
    details specific to differentiable MPC:

    - `assert (status == 0).all()` — a gradient read off a *failed* solve
      is garbage; on this convex QP a failure means a bug, so we stop hard,
    - the epoch's `ctx` warm-starts the next epoch: θ moves slowly, so the
      previous solution is an excellent initial iterate (warm starts change
      the speed, never the gradients),
    - clamps keep the OCP sane while θ moves: `price_weight` must stay
      non-negative, and `R` strictly positive (it divides the dynamics).
    """)
    return


@app.cell
def _(B_DATA, HeatingPlanner, data, np, torch):
    N_EPOCHS = 100

    planner_il = HeatingPlanner(
        N_horizon=32, name="heating_planner_il", n_batch_init=B_DATA
    )
    optimizer = torch.optim.Adam(planner_il.parameters(), lr=3e-2)
    u_expert = torch.tensor(data["u_expert"], dtype=torch.float64).reshape(-1, 1)

    losses, theta_path = [], []
    _ctx = None
    for _epoch in range(N_EPOCHS):
        u0_pred, _ctx_new, _ = planner_il(data, ctx=_ctx)
        assert np.all(_ctx_new.status == 0), "a solve failed during training"

        loss = ((u0_pred - u_expert) ** 2).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        planner_il.price_weight.data.clamp_(min=0.0)
        planner_il.R.data.clamp_(min=0.5)

        _ctx = _ctx_new  # warm-start the next epoch
        losses.append(float(loss))
        theta_path.append(
            (
                float(planner_il.R),
                float(planner_il.price_weight),
                float(planner_il.comfort_margin),
            )
        )

    theta_learned = theta_path[-1]
    assert losses[-1] < 0.5 * losses[0], "training failed to reduce the loss"
    print(
        f"loss {losses[0]:.3f} -> {losses[-1]:.3f}   |   "
        f"R -> {theta_learned[0]:.3f}, "
        f"price_weight -> {theta_learned[1]:.3f}, "
        f"comfort_margin -> {theta_learned[2]:.3f}"
    )
    return losses, planner_il, theta_learned, theta_path


@app.cell
def _(losses, np, plt, theta_path):
    il_fig, il_axes = plt.subplots(1, 2, figsize=(9.5, 3.6))
    il_axes[0].semilogy(losses, color="tab:blue")
    il_axes[0].set_xlabel("Epoch")
    il_axes[0].set_ylabel("BC loss [kW²]")
    il_axes[0].set_title("Loss")

    _theta = np.asarray(theta_path)
    il_axes[1].plot(_theta[:, 0], label="R [K/kW]", color="tab:red")
    il_axes[1].plot(_theta[:, 1], label="price_weight", color="tab:purple")
    il_axes[1].plot(_theta[:, 2], label="comfort_margin [K]", color="tab:green")
    il_axes[1].axhline(1.4, ls="--", lw=0.8, color="tab:red", alpha=0.6)
    il_axes[1].axhline(0.0, ls="--", lw=0.8, color="tab:purple", alpha=0.6)
    il_axes[1].axhline(0.7, ls="--", lw=0.8, color="tab:green", alpha=0.6)
    il_axes[1].set_xlabel("Epoch")
    il_axes[1].set_title("Parameter trajectories (dashed: ground truth)")
    il_axes[1].legend(fontsize=8)
    for _ax in il_axes:
        _ax.grid(True, alpha=0.3)
    il_fig.tight_layout()
    il_fig
    return


@app.cell(hide_code=True)
def _(mo, theta_learned):
    mo.md(rf"""
    The parameters converge to an interpretable portrait of the occupant
    *and* their house (dashed reference lines):

    - `R` → `{theta_learned[0]:.2f}` K/kW — the occupant's feedforward
      betrays the building physics: the clone discovers the envelope is
      leakier than the model's `2.0` (truth: `1.4`). Notebook 02 asked what
      insulation is *worth*; here the data says what the insulation *is*,
    - `price_weight` → `{theta_learned[1]:.2f}` — the thermostat never
      looks at the tariff, and the clone finds out: the learned MPC stops
      price-shifting,
    - `comfort_margin` → `{theta_learned[2]:.2f}` K — where the occupant
      actually lives above the schedule (truth: their `+0.7` K preference).

    The loss has a floor: the occupant's proportional term also reacts to
    the unobserved gains and noise, which no function of the observation
    can reproduce. Behavior cloning recovers the *policy*, not the
    disturbances.

    ## Did the clone learn the behavior, or just the actions?

    The real test is **closed-loop**: run the planner in the true house on
    an unseen day, before and after training, and compare against the
    occupant.
    """)
    return


@app.cell
def _(HouseEnv, env, np, planner_il, theta_learned, torch):
    from nb_utils.env import thermostat_expert

    def set_theta(theta):
        planner_il.R.data.fill_(theta[0])
        planner_il.price_weight.data.fill_(theta[1])
        planner_il.comfort_margin.data.fill_(theta[2])

    def rollout_planner(theta, world, n_steps=96):
        set_theta(theta)
        planner_il.ctx = None
        obs = world.reset(start=0, T0=19.0)
        log = {"T": [obs["T"]], "q": [], "energy": [], "discomfort": []}
        for _ in range(n_steps):
            q = planner_il.act(obs)
            obs, _, _, info = world.step(q)
            log["T"].append(obs["T"])
            log["q"].append(q)
            log["energy"].append(info["energy_eur"])
            log["discomfort"].append(info["discomfort_kh"])
        return {key: np.asarray(vals) for key, vals in log.items()}

    def rollout_expert(world, n_steps=96):
        obs = world.reset(start=0, T0=19.0)
        log = {"T": [obs["T"]], "q": [], "energy": [], "discomfort": []}
        for _ in range(n_steps):
            q = thermostat_expert(obs["T"], obs["t_lower"][0], obs["outdoor"][0])
            obs, _, _, info = world.step(q)
            log["T"].append(obs["T"])
            log["q"].append(q)
            log["energy"].append(info["energy_eur"])
            log["discomfort"].append(info["discomfort_kh"])
        return {key: np.asarray(vals) for key, vals in log.items()}

    env_eval = HouseEnv(n_days=1, N_forecast=32, seed=7)  # unseen disturbances
    log_expert = rollout_expert(env_eval)
    log_before = rollout_planner((2.0, 1.0, 0.0), env_eval)  # notebook 06 defaults
    log_after = rollout_planner(theta_learned, env_eval)

    # Leave the trained parameters on the module.
    set_theta(theta_learned)
    _ = env  # keep the training env alive in the dependency graph

    for _name, _log in [("expert", log_expert), ("before BC", log_before),
                        ("after BC", log_after)]:
        print(f"{_name:10s}: {_log['energy'].sum():5.2f} EUR, "
              f"{_log['discomfort'].sum():5.2f} Kh discomfort")
    return env_eval, log_after, log_before, log_expert


@app.cell
def _(env_eval, log_after, log_before, log_expert, np, plt):
    _n = len(log_expert["q"])
    _t_state = 0.25 * np.arange(_n + 1)

    cmp_fig, cmp_axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    cmp_axes[0].fill_between(_t_state, env_eval.t_lower[: _n + 1],
                             env_eval.t_upper[: _n + 1],
                             color="tab:green", alpha=0.12, label="comfort band")
    cmp_axes[0].plot(_t_state, log_expert["T"], color="tab:gray", lw=1.2,
                     label="occupant (thermostat)")
    cmp_axes[0].plot(_t_state, log_before["T"], color="tab:red", lw=1.2, alpha=0.8,
                     label="planner before BC")
    cmp_axes[0].plot(_t_state, log_after["T"], color="tab:blue", lw=1.4,
                     label="planner after BC")
    cmp_axes[0].set_ylabel("Room temp [degC]")
    cmp_axes[0].legend(fontsize=8, loc="lower right")

    cmp_axes[1].step(_t_state[:-1], log_expert["q"], where="post", color="tab:gray",
                     lw=1.0, label="occupant")
    cmp_axes[1].step(_t_state[:-1], log_after["q"], where="post", color="tab:blue",
                     lw=1.2, label="planner after BC")
    cmp_axes[1].set_ylabel("Heating [kW]")
    cmp_axes[1].set_xlabel("Time since midnight [h]")
    cmp_axes[1].legend(fontsize=8)
    for _ax in cmp_axes:
        _ax.grid(True, alpha=0.3)
    cmp_fig.suptitle("Unseen day in the true house: occupant vs. planner, before/after")
    cmp_fig.tight_layout()
    cmp_fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Before cloning, the planner rode the scheduled bound with the wrong
    envelope model and sagged below the band (notebook 06's mismatch
    problem). After cloning it lives where the occupant lives, with the
    occupant's knowledge of the house — the learned margin and the
    corrected `R` act as the robustness the demonstrations contained
    implicitly.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Aside: collating stored solver contexts

    Off-policy pipelines (replay buffers, `DataLoader`s) store *per-sample*
    records and later assemble random batches. leap-c ships
    `collate_torch` for exactly this: PyTorch's default collation plus a
    rule for `AcadosDiffMpcCtx`, so a stored warm start rides along with
    its sample and a batch of contexts becomes one batched context.
    """)
    return


@app.cell
def _(collate_torch, data, np, planner_il, theta_learned, torch):
    # A minimal replay buffer: three single-sample solves, each stored with
    # its own context...
    _buffer = []
    for _i in range(3):
        _single = {key: np.asarray(data[key][_i])[None]
                   for key in ("T", "outdoor", "price", "t_lower", "t_upper")}
        _, _ctx_i, _ = planner_il(_single)
        _buffer.append({"T": torch.tensor(data["T"][_i]), "ctx": _ctx_i})

    # ...collated into one batch: tensors stack, contexts merge.
    batch = collate_torch(_buffer)

    print(f"(learned theta: {theta_learned[0]:.2f}, {theta_learned[1]:.2f})")
    print(f"batched T: {batch['T'].shape},  "
          f"batched ctx: N_batch = {batch['ctx'].iterate.N_batch}")
    print("pass it back via planner(obs_batch, ctx=batch['ctx']) for a "
          "warm-started batched solve")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Recap

    - a differentiable MPC turns imitation learning into ordinary
      supervised PyTorch — the solver sits inside the loss like any layer,
    - one batched solve per epoch; warm-start it with last epoch's `ctx`;
      hard-stop on nonzero statuses,
    - the learned parameters are *legible*: a near-zero price weight, the
      occupant's comfort margin, and the house's true `R` — the clone
      **describes the occupant and the building**,
    - `collate_torch` batches stored samples *including* their solver
      contexts — the building block for replay-buffer training.

    **Next:** `08_rl_on_closed_loop_cost.py` — stop imitating, start
    improving: tune the same two parameters against the true closed-loop
    cost (energy + discomfort) by backpropagating through the rollout.
    """)
    return


if __name__ == "__main__":
    app.run()
