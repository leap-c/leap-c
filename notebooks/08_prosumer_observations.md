# Observations from building notebook 08 (prosumer economic MPC)

Notes collected while developing `08_prosumer.py` (2026-07-04). Everything
below was verified numerically; the reproduction commands are at the end.
Setup: R1C1 building + heat pump (COP 3), 10 kWh battery, PV, asymmetric
grid prices (dynamic tariff vs. 0.079 EUR/kWh feed-in), N = 96 stages of
15 min, stagewise differentiable `price_buy`, 48-scenario batched solve.

## 1. `du_dp_global` returns the *stage-summed* sensitivity, not the trajectory Jacobian

The docstring of `AcadosDiffMpcCtx` ("sensitivity of the whole control
trajectory solution") and notebook 07's markdown ("the KKT solver returns the
whole `(B, N·nu, P)` block in a single call") both suggest that
`diff_mpc_fun.sensitivity(ctx, "du_dp_global")` returns the full Jacobian of
the control trajectory. It does not. The actual return shape is
`(B, nu, P)`, and its content is

    du_dp_global[b, i, :] = d( sum_k u_k[i] ) / d p_global ,

i.e. the adjoint of the *summed* trajectory. Cause: `_get_seed_seq`
(`leap_c/diff_mpc/function.py:402`) passes one identity seed per stage to
acados `eval_adjoint_solution_sensitivity`, which sums the adjoint
contributions across stages. `dx_dp_global` has the same behavior
(`(B, nx, P)`). Verified machine-exact against
`torch.autograd.grad(u[:, :, i].sum(), price)` (max err 0.0).

Consequences:

- Notebook 07's section C timing comparison ("full trajectory Jacobian:
  autograd vs. one KKT call") is apples-to-oranges — the autograd side
  computes the true `(B, N·nu, P)` Jacobian, the KKT side only the
  stage-summed `(B, nu, P)` aggregate. Its text has not been corrected yet.
- No test pins either behavior.
- A silent reshape of the `(B, nu, P)` result into per-stage blocks produces
  plausible-looking garbage.

Workaround used in 08: build the per-stage Jacobian with one
`torch.autograd.grad(..., retain_graph=True)` per stage
(`nb_utils/prosumer.py:gnet_price_jacobian`); 96 rows for all 48 scenarios
take about 2 s. The one-call aggregate then serves as an independent
cross-check: `S.sum(axis=1)` matches `du_dp_global` rows to ~1e-14.

A proper core fix would construct block-diagonal seeds (stage k carries the
identity in columns `k*nu:(k+1)*nu`, `n_seeds = N*nu`) so the call returns
the true `(B, N·nu, P)` Jacobian, at O(N^2) seed memory.

## 2. Sensitivities are *localized*: columns die exactly where the plan is constraint-pinned

The plan-vs-price Jacobian `S[k, j] = d g_net,k / d price_j` is exactly zero
in every column j where the plan does not trade (all three controls at their
bounds). This is not numerical noise — an active bound has zero local
sensitivity — and it dominates the structure of the map:

- At the default scenario (0.50 evening peak, 4 kWp PV, 10 kWh battery) the
  battery, the midday PV and a pre-heated building cover the *entire* evening
  peak: `g_buy = g_sell = 0` through 17-20 h, and the whole block of evening
  columns is zero. Raising the evening peak price further does not change the
  plan at all (equivalently `dV/d price_k = Δt · g_buy,k = 0` there). The
  prosumer is *decoupled* — the tariff-setter has no leverage where the
  prosumer does not trade.
- The live columns sit where the buying happens: the overnight/early-morning
  window (col-max ≈ 80 kW per EUR/kWh around 03:30-05:00) and a smaller,
  *stronger* late-evening block after the peak (≈ 10 vs. ≈ 3 in the wide
  night block — fewer alternative hours to spread the substitution over).
- This held across every parameterization tried (battery 2.5/5/10 kWh,
  outdoor day shifted 3-6 K colder): whenever the peak is worth avoiding, the
  optimizer avoids it *completely*, and the peak columns die. Only a barely
  raised "peak" (0.25 vs. 0.22 base) leaves evening buying interior.

Design consequence for the notebook: the default Jacobian slider stage is
04:00 (alive), not 18:30 (dead), and the dead evening is presented as the
punchline rather than fought with parameter tuning.

## 3. The regularizers set the Jacobian's magnitude

The economics are piecewise linear; all smooth curvature in the control
space comes from the two quadratic regularizers. The observed own-price
response ~ -80 kW per EUR/kWh matches the substitution slope implied by the
quadratic terms, 1/(2 (c_wear + ε)) = 1/(2 · 0.006) ≈ 83. So in this
LP-like economic MPC the *sign and support* of the Jacobian are economics,
but its *scale* is set by the regularization — worth remembering before
reading the numbers as physical demand elasticities. (In the plots the
diagonal is clipped/saturated for the same reason: it is one order of
magnitude above the substitution lobes, which carry the interesting
structure.)

Related, from the OCP formulation side:

- `c_wear · (P_bat)^2` alone leaves the control Hessian rank **one** in the
  3-D control space (it curves only along the direction that changes
  `P_bat`); the exact-Hessian SQP and the regularization-stripping
  sensitivity solver need the full-rank
  `ε (q^2 + g_buy^2 + g_sell^2)` term. With ε = 1e-3 the marginal-price
  distortion is ~0.01 EUR/kWh at 5 kW.
- Buy/sell complementarity needs no constraint: with
  `price_buy - price_sell >= 0.05` everywhere plus the ε curvature, the
  solver never buys and sells simultaneously
  (`max g_buy · g_sell ≈ 3e-6 kW²` across all 48 scenarios).

## 4. Level shifts do (almost) nothing; only spreads move plans

Shifting the whole tariff by ±0.05 EUR/kWh changes the planned net grid
exchange by at most ~0.13 kW: with no cheap hour to escape to, a uniform
surcharge is unavoidable and the plan barely reacts. All the behavior —
pre-charging, pre-heating, peak avoidance, selling — is driven by price
*spreads*. (The tiny residual response comes through the terminal value and
the feed-in tariff, which do not shift along.)

## 5. Free PV saturates the battery before the peak

For every PV size >= 2 kWp the battery hits `E_max` before 17:00 regardless
of the evening-peak height — midday PV is free energy, so "how much to
pre-charge" is not a price question once PV is present. Peak-height-dependent
pre-charging is only visible in the PV = 0 scenarios (where charging costs
money). The comfort band shows the same economics: the room rides the lower
band edge (19 degC) almost all day — the cheapest admissible comfort — with
one pre-heat bump to ~20.5 degC just before the evening peak.

## 6. Exactness of the sensitivity machinery

With `hessian_approx = "EXACT"` and the cost linear in the prices, the
envelope-theorem identities hold to machine precision (not just to solver
tolerance), because the backward pass evaluates the same expressions:

- `dV/d price_buy,k = Δt · g_buy,k` — max abs err 0.0 across 48 × 97 entries;
- `dV/d price_sell = -Δt · Σ_k g_sell,k` — ~7e-15;
- `dV/d λ = -E_N` — 0.0;
- the stage-N price column of the Jacobian is exactly zero (stage N carries
  only the terminal cost);
- autograd vs. `dvalue_dp_global` (exact KKT): 0.0;
- autograd Jacobian column-sums vs. `du_dp_global` aggregate: ~6e-14.

## 7. Guardrails that keep the problem well-posed

- Terminal value bracket: `price_sell < λ < min_k price_buy,k`
  (0.079 < 0.12 < 0.13). Below the feed-in tariff the plan dumps the battery
  into the grid at stage N; above the cheapest buy price it hoard-buys for
  the terminal credit. Both are finite-horizon artifacts, not economics.
- `q_max = 4 kW` electric (12 kW heat) is sized so the coldest night
  (~2 degC) can hold the band with headroom (~3.2 kW needed); with a tight
  `q_max` the comfort slack becomes permanently active and the "purely
  economic" reading of the sensitivities breaks.
- Terminal slack fields (`idxsbx_e`, `Zl_e`, ...) must mirror the stage
  slacks, otherwise the terminal temperature bound is *hard* and cold days
  can go infeasible.
- Grid codegen and solve cost stayed small at this size: p_global = 99
  (97 stagewise price blocks + 2 globals), build + codegen ~5 s, the
  48-scenario batched solve ~0.05 s, the 96-pass Jacobian ~2 s.

## Reproduction

All of section 6 and the complementarity/band checks run as asserts inside
`08_prosumer.py` on every headless execution:

```bash
cd notebooks && MPLBACKEND=Agg uv run --extra notebooks python -m marimo export html 08_prosumer.py -o /tmp/08.html
uv run --extra test --extra notebooks python -m pytest tests/test_notebooks.py -k 08 -v
```

The `du_dp_global` shape experiment (section 1) is a five-liner: solve any
batched problem, call `diff_mpc_fun.sensitivity(ctx, "du_dp_global")`, and
compare its rows with `torch.autograd.grad(u[:, :, i].sum(), price)`.
