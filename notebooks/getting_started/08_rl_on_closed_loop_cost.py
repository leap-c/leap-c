"""Part 8 — tuning the planner against the true closed-loop cost.

Imitation stops at the occupant's skill. This notebook goes further: it tunes
the planner's parameters to minimize the *actual* closed-loop cost — energy
bill plus thermal discomfort — by backpropagating through an entire rollout,
chaining the solver's sensitivities from step to step. Differentiable
simulation as the simplest possible "RL" for a differentiable MPC.
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
    # 08 — reinforcement on the closed-loop cost

    Behavior cloning (notebook 07) can only be as good as the occupant.
    But we know what we actually want — small values of the running cost

    $$c_k = \underbrace{\mathrm{price}_k\, q_k\, \Delta t}_{\text{energy bill}}
    \;+\; w \cdot \underbrace{\big[(\mathrm{lb}_{k+1} - T_{k+1})_+ +
    (T_{k+1} - \mathrm{ub}_{k+1})_+\big] \Delta t}_{\text{discomfort [K·h]}}$$

    — the BOPTEST-style building objective (`cost + w · discomfort`, here
    $w = 1$ EUR per K·h). So minimize the discounted closed-loop sum
    directly:

    $$J(\theta) = \mathbb{E}\Big[\sum_{k=0}^{K-1} \gamma^k\,
    c_k\big(T_k,\, u_0(o_k; \theta)\big)\Big] .$$

    **What this is:** analytic policy search through a *differentiable
    simulation* — our true dynamics are a torch one-liner, the disturbances
    are known samples, and $u_0(o; \theta)$ is differentiable through the
    solver, so $\nabla_\theta J$ comes from one `backward()` through the
    whole rollout. **What this is not:** model-free RL. If all you have is
    a black-box environment, you would estimate the same gradient
    statistically — e.g. REINFORCE perturbs $\theta$ (or the actions),
    scores each perturbation by its return, and averages; dozens of
    rollouts buy a noisy version of what one backward pass gives us here
    exactly.
    """)
    return


@app.cell
def _(mo):
    import sys

    sys.path.insert(0, str(mo.notebook_dir().parent))  # notebooks/ root -> nb_utils

    import matplotlib.pyplot as plt
    import numpy as np
    import torch

    from nb_utils.env import HouseEnv, step_cost, true_step
    from nb_utils.planner import HeatingPlanner

    return HeatingPlanner, HouseEnv, np, plt, step_cost, torch, true_step


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Setup

    The planner starts from notebook 07's portrait of the occupant —
    $R = 1.6$, `price_weight` $= 0$, `comfort_margin` $= 0.8$ — so this
    notebook runs standalone but continues that story. Note
    `discount_factor=GAMMA` in the constructor: leap-c re-weights the OCP's
    stage costs by $\gamma^k$, aligning the planner's *internal* objective
    with the discounted return it is being trained on.

    The disturbances (occupancy gains, AR(1) noise) come from the same
    `HouseEnv` profiles as before — known to the training loop (that is
    what makes the rollout differentiable), never shown to the planner.
    """)
    return


@app.cell
def _(HeatingPlanner, HouseEnv, torch):
    GAMMA = 0.99
    K_STEPS = 48  # 12 h rollouts
    STARTS = [0, 24, 48, 72]  # midnight, 06:00, 12:00, 18:00 — B = 4
    THETA_IL = (1.6, 0.0, 0.8)  # notebook 07's result

    env_rl = HouseEnv(n_days=2, N_forecast=32, seed=0)

    planner_rl = HeatingPlanner(
        N_horizon=32,
        name="heating_planner_rl",
        n_batch_init=len(STARTS),
        discount_factor=GAMMA,
    )
    planner_rl.R.data.fill_(THETA_IL[0])
    planner_rl.price_weight.data.fill_(THETA_IL[1])
    planner_rl.comfort_margin.data.fill_(THETA_IL[2])

    # Profiles as tensors: constants of the differentiable rollout.
    outdoor_t = torch.tensor(env_rl.outdoor)
    price_t = torch.tensor(env_rl.price)
    lb_t = torch.tensor(env_rl.t_lower)
    ub_t = torch.tensor(env_rl.t_upper)
    dist_t = torch.tensor(env_rl.disturbance_kw)
    return (
        GAMMA,
        K_STEPS,
        STARTS,
        THETA_IL,
        dist_t,
        env_rl,
        lb_t,
        outdoor_t,
        planner_rl,
        price_t,
        ub_t,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The differentiable rollout

    Four parallel 12-hour rollouts, one batched solve per step. Gradients
    flow along **two** paths the layer exposes: directly from each action
    to θ ($\partial u_0/\partial \theta$), and through time — the action
    changes $T_{k+1}$, which changes the next solve
    ($\partial u_0/\partial x_0$), and so on down the chain. Two habits
    from the earlier notebooks carry over:

    - `ctx` warm-starting *inside* the differentiated rollout is exact:
      the context only sets the solver's initial iterate; gradients are
      computed at the converged solution and warm/cold solves agree,
    - a failed solve contributes garbage gradients, so its sample is
      **masked out** of the objective (with this QP, statuses stay 0 in
      practice — the mask is the pattern to copy, not a fix we need).
    """)
    return


@app.cell
def _(
    K_STEPS,
    STARTS,
    dist_t,
    env_rl,
    lb_t,
    np,
    outdoor_t,
    planner_rl,
    price_t,
    step_cost,
    torch,
    true_step,
    ub_t,
):
    def rollout(gamma):
        """Differentiable discounted cost of B parallel closed-loop rollouts."""
        starts = np.asarray(STARTS)
        n_roll = len(starts)
        T = torch.full((n_roll,), 19.0, dtype=torch.float64)
        ctx = None
        J = torch.zeros((), dtype=torch.float64)

        for k in range(K_STEPS):
            idx = starts + k
            obs = {
                "T": T,  # tensor -> the autograd graph flows through x0
                "outdoor": np.stack([env_rl.outdoor[i : i + 33] for i in idx]),
                "price": np.stack([env_rl.price[i : i + 33] for i in idx]),
                "t_lower": np.stack([env_rl.t_lower[i : i + 33] for i in idx]),
                "t_upper": np.stack([env_rl.t_upper[i : i + 33] for i in idx]),
            }
            u0, ctx, _ = planner_rl(obs, ctx=ctx)
            ok = torch.tensor(ctx.status == 0)

            q = u0[:, 0]
            T = true_step(T, q, outdoor_t[idx], gain=dist_t[idx])
            cost_k = step_cost(price_t[idx], q, T, lb_t[idx + 1], ub_t[idx + 1])
            J = J + gamma**k * torch.where(ok, cost_k, torch.zeros_like(cost_k)).sum()

        return J / n_roll

    return (rollout,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Before trusting a 48-step backward pass through 48 solver calls, check
    it: perturb one parameter, re-roll, compare against finite differences.
    """)
    return


@app.cell
def _(GAMMA, planner_rl, rollout, torch):
    J0 = rollout(GAMMA)
    J0.backward()
    grad_margin = float(planner_rl.comfort_margin.grad)

    # Central finite differences on comfort_margin (2 extra rollouts).
    _eps = 1e-3
    with torch.no_grad():
        planner_rl.comfort_margin += _eps
        J_plus = rollout(GAMMA)
        planner_rl.comfort_margin -= 2 * _eps
        J_minus = rollout(GAMMA)
        planner_rl.comfort_margin += _eps
    fd_margin = float((J_plus - J_minus) / (2 * _eps))

    _rel_err = abs(grad_margin - fd_margin) / max(abs(fd_margin), 1e-9)
    assert _rel_err < 0.3, f"rollout gradient check failed: {grad_margin} vs {fd_margin}"
    planner_rl.zero_grad()
    fd_checked = True  # gates the training cell on this check

    print(f"dJ/d comfort_margin: autograd {grad_margin:+.4f}, "
          f"finite differences {fd_margin:+.4f}  (rel. err. {_rel_err:.1%})")
    return (fd_checked,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The training loop

    Vanilla gradient descent on the rollout objective — Adam, ~25
    iterations, each one a full differentiable rollout. (`N_RL_ITERS` is
    the compute knob: each iteration costs `K_STEPS` batched solves plus
    the backward chain.)
    """)
    return


@app.cell
def _(GAMMA, fd_checked, planner_rl, rollout, torch):
    N_RL_ITERS = 25
    assert fd_checked  # run only after the gradient check above

    optimizer_rl = torch.optim.Adam(planner_rl.parameters(), lr=2e-2)
    J_path, theta_rl_path = [], []
    for _it in range(N_RL_ITERS):
        J = rollout(GAMMA)
        optimizer_rl.zero_grad()
        J.backward()
        optimizer_rl.step()
        planner_rl.price_weight.data.clamp_(min=0.0)
        planner_rl.R.data.clamp_(min=0.5)
        J_path.append(float(J))
        theta_rl_path.append(
            (
                float(planner_rl.R),
                float(planner_rl.price_weight),
                float(planner_rl.comfort_margin),
            )
        )

    theta_rl = theta_rl_path[-1]
    assert J_path[-1] < J_path[0], "training failed to improve the rollout cost"
    print(
        f"J {J_path[0]:.3f} -> {J_path[-1]:.3f} EUR   |   "
        f"R -> {theta_rl[0]:.3f}, price_weight -> {theta_rl[1]:.3f}, "
        f"comfort_margin -> {theta_rl[2]:.3f}"
    )
    return J_path, theta_rl, theta_rl_path


@app.cell
def _(J_path, np, plt, theta_rl_path):
    rl_fig, rl_axes = plt.subplots(1, 2, figsize=(9.5, 3.6))
    rl_axes[0].plot(J_path, color="tab:blue")
    rl_axes[0].set_xlabel("Iteration")
    rl_axes[0].set_ylabel("Discounted rollout cost J [EUR]")
    rl_axes[0].set_title("Objective")

    _theta = np.asarray(theta_rl_path)
    rl_axes[1].plot(_theta[:, 0], label="R [K/kW]", color="tab:red")
    rl_axes[1].plot(_theta[:, 1], label="price_weight", color="tab:purple")
    rl_axes[1].plot(_theta[:, 2], label="comfort_margin [K]", color="tab:green")
    rl_axes[1].set_xlabel("Iteration")
    rl_axes[1].set_title("Parameter trajectories")
    rl_axes[1].legend(fontsize=8)
    for _ax in rl_axes:
        _ax.grid(True, alpha=0.3)
    rl_fig.tight_layout()
    rl_fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The interesting move is `price_weight`: imitation drove it to zero
    (the occupant ignores the tariff), and the closed-loop cost pulls it
    **back up** — pre-heating before the price peaks saves real money that
    no demonstration contained. The margin settles where the marginal
    discomfort of sagging equals the marginal cost of heating earlier.

    ## Evaluation: occupant's skill vs. optimized

    Same unseen day as notebook 07, same stochastic house, warm-started
    receding horizon — the imitation parameters against the RL-tuned ones.
    """)
    return


@app.cell
def _(HouseEnv, THETA_IL, np, planner_rl, theta_rl):
    def evaluate(theta, world, n_steps=96):
        planner_rl.R.data.fill_(theta[0])
        planner_rl.price_weight.data.fill_(theta[1])
        planner_rl.comfort_margin.data.fill_(theta[2])
        planner_rl.ctx = None
        obs = world.reset(start=0, T0=19.0)
        log = {"T": [obs["T"]], "q": [], "energy": [], "discomfort": []}
        for _ in range(n_steps):
            q = planner_rl.act(obs)
            obs, _, _, info = world.step(q)
            log["T"].append(obs["T"])
            log["q"].append(q)
            log["energy"].append(info["energy_eur"])
            log["discomfort"].append(info["discomfort_kh"])
        return {key: np.asarray(vals) for key, vals in log.items()}

    env_eval_rl = HouseEnv(n_days=1, N_forecast=32, seed=7)
    eval_il = evaluate(THETA_IL, env_eval_rl)
    eval_rl = evaluate(theta_rl, env_eval_rl)

    # Leave the RL-tuned parameters on the module.
    planner_rl.R.data.fill_(theta_rl[0])
    planner_rl.price_weight.data.fill_(theta_rl[1])
    planner_rl.comfort_margin.data.fill_(theta_rl[2])

    for _name, _log in [("imitation θ", eval_il), ("RL-tuned θ", eval_rl)]:
        _total = _log["energy"].sum() + _log["discomfort"].sum()
        print(f"{_name}: {_log['energy'].sum():5.2f} EUR energy + "
              f"{_log['discomfort'].sum():5.2f} Kh discomfort = {_total:5.2f} total")
    return env_eval_rl, eval_il, eval_rl


@app.cell
def _(env_eval_rl, eval_il, eval_rl, np, plt):
    _n = len(eval_il["q"])
    _t_state = 0.25 * np.arange(_n + 1)
    _t_ctrl = 0.25 * np.arange(_n)

    ev_fig, ev_axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    ev_axes[0].fill_between(_t_state, env_eval_rl.t_lower[: _n + 1],
                            env_eval_rl.t_upper[: _n + 1],
                            color="tab:green", alpha=0.12, label="comfort band")
    ev_axes[0].plot(_t_state, eval_il["T"], color="tab:gray", lw=1.2,
                    label="imitation θ")
    ev_axes[0].plot(_t_state, eval_rl["T"], color="tab:blue", lw=1.4,
                    label="RL-tuned θ")
    ev_axes[0].set_ylabel("Room temp [degC]")
    ev_axes[0].legend(fontsize=8, loc="lower right")

    ev_axes[1].step(_t_ctrl, eval_il["q"], where="post", color="tab:gray", lw=1.0,
                    label="imitation θ")
    ev_axes[1].step(_t_ctrl, eval_rl["q"], where="post", color="tab:orange", lw=1.2,
                    label="RL-tuned θ")
    _ax_price = ev_axes[1].twinx()
    _ax_price.step(_t_ctrl, env_eval_rl.price[:_n], where="post",
                   color="tab:purple", alpha=0.5)
    _ax_price.set_ylabel("Price [EUR/kWh]", color="tab:purple")
    ev_axes[1].set_ylabel("Heating [kW]")
    ev_axes[1].set_xlabel("Time since midnight [h]")
    ev_axes[1].legend(fontsize=8)
    for _ax in ev_axes:
        _ax.grid(True, alpha=0.3)
    ev_fig.suptitle("Unseen day: imitated occupant vs. closed-loop-optimized planner")
    ev_fig.tight_layout()
    ev_fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Recap — and the end of the series

    - when the true dynamics are differentiable and the disturbances are
      recorded, policy improvement is one `backward()` through the rollout:
      the layer chains $\partial u_0/\partial \theta$ and
      $\partial u_0/\partial x_0$ across time steps,
    - warm starts stay exact inside a differentiated rollout; failed
      solves get masked out; a finite-difference spot check is cheap
      insurance on a long chain,
    - `discount_factor=` aligns the OCP's internal cost with a discounted
      return,
    - imitation gives a safe, human-informed starting point; the
      closed-loop objective then buys what no demonstration contains —
      here, price-awareness.

    **Where to go next:** the `custom_examples/` folder — economic MPC on
    a battery, the exact KKT sensitivity API, and a full prosumer (heat
    pump + PV + battery) with a plan-vs-tariff Jacobian.
    """)
    return


if __name__ == "__main__":
    app.run()
