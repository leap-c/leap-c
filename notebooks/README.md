# Notebooks

These are [marimo](https://marimo.io) notebooks (plain `.py` files, not `.ipynb`).

They form a short series — each notebook teaches one aspect of leap-c on a
small, visual system, with interactive sliders driving precomputed batched
solves. Recommended reading order:

| # | Notebook | System | Teaches | Key API |
|---|----------|--------|---------|---------|
| — | `intro.py` | CartPole | high-level planner in a gym loop | `create_planner` |
| 01 | `01_msd_build_and_solve.py` | mass-spring-damper | registering parameters, building a parametric OCP, solving, reading the plan | `AcadosParameterManager.register_parameter`, `AcadosDiffMpcTorch` |
| 02 | `02_msd_value_policy_maps.py` | mass-spring-damper | batched solves; the MPC as value function and policy over the state space | batching, `n_batch_init` |
| 03 | `03_msd_sensitivities.py` | mass-spring-damper | gradients through the solver | `.backward()`, `torch.autograd.functional.jacobian`, `sensitivity(ctx, ...)` |
| 04 | `04_heating_parameter_management.py` | R1C1 heating | differentiable vs. non-differentiable parameters; `splits`: global / blocks / stagewise | `splits=`, `model.p` vs. `model.p_global` |
| 05 | `05_heating_forecasts.py` | R1C1 heating | embedding weather/price forecasts, receding horizon, gradients w.r.t. a forecast | stagewise `(B, N+1, 1)` params, `dvalue_dp_global` |

Shared helpers live in `nb_utils/` (OCP builders, the RC-network diagram,
synthetic day profiles). The OCP builders are taught *inline* in 01
(mass-spring-damper) and 04 (heating); the copies in `nb_utils` exist so the
other notebooks can import them.

## Running

marimo is behind the `notebook` extra, so pass `--extra notebook` (or run
`uv sync --extra notebook` once and drop the flag afterwards):

```bash
# Interactive editor (edit + run cells) — usual choice
uv run --extra notebook marimo edit notebooks/01_msd_build_and_solve.py

# Read-only app view (runs it as a deployed app, cells hidden)
uv run --extra notebook marimo run notebooks/01_msd_build_and_solve.py
```

marimo prints a `http://localhost:2718` URL and normally opens it in your
browser automatically.

Note: the first run of each notebook generates and compiles an acados solver,
which can take a minute or two; subsequent runs are fast.

## Roadmap — future notebooks

- **Cartpole with stage-varying references** — `leap_c/examples/cartpole`
  already accepts `param_splits` for its reference; a notebook could morph the
  swing-up target over the horizon.
- **Battery arbitrage** — a one-state, one-input economic MPC (state of
  charge, charge rate, stagewise price): the smallest possible showcase of an
  economic cost.
- **Point mass with wind** — 2-D policy gradients `du0/dp` drawn as arrows.
- **Real forecast data** — swap the synthetic profiles in `nb_utils/data.py`
  for measured weather/price time series.
