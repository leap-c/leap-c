# Notebooks

These are [marimo](https://marimo.io) notebooks (plain `.py` files, not `.ipynb`).

They form a short series — each notebook teaches one aspect of leap-c on a
small, visual system, with interactive sliders driving precomputed batched
solves. Recommended reading order:

| # | Notebook | System | Teaches | Key API |
|---|----------|--------|---------|---------|
| — | `minimal_mpc.py` | scalar integrator | minimal end-to-end MPC; batched warm-start collation | `AcadosDiffMpcTorch`, `collate_torch` |
| 01 | `01_msd_build_and_solve.py` | mass-spring-damper | registering parameters, building a parametric OCP, solving, reading the plan | `AcadosParameterManager.register_parameter`, `AcadosDiffMpcTorch` |
| 02 | `02_msd_value_policy_maps.py` | mass-spring-damper | batched solves; the MPC as value function and policy over the state space | batching, `n_batch_init` |
| 03 | `03_msd_sensitivities.py` | mass-spring-damper | gradients through the solver with autograd | `.backward()`, `torch.autograd.functional.jacobian` |
| 04 | `04_heating_parameter_management.py` | R1C1 heating | differentiable vs. non-differentiable parameters; `splits`: global / blocks / stagewise | `splits=`, `model.p` vs. `model.p_global` |
| 05 | `05_heating_forecasts.py` | R1C1 heating | embedding weather/price forecasts, receding horizon, gradients w.r.t. a forecast | stagewise `(B, N+1, 1)` params, `.backward()` → `.grad` |
| 06 | `06_battery_arbitrage.py` | battery | economic MPC: a pure money cost, terminal energy value, signed price sensitivities | `EXTERNAL` economic cost, autograd `.grad` w.r.t. price and terminal value |
| 07 | `07_advanced_sensitivities.py` | mass-spring-damper | *advanced*: the exact KKT sensitivity API, exact-match validation vs. autograd, timing comparison | `diff_mpc_fun.sensitivity(ctx, ...)`, `p_global_slice` |

Shared helpers live in `nb_utils/` (OCP builders, the RC-network diagram,
synthetic day profiles). The OCP builders are taught *inline* in 01
(mass-spring-damper), 04 (heating) and 06 (battery); the copies of the first
two in `nb_utils` exist so the other notebooks can import them (the battery
OCP is used only in 06, so no copy exists).

## Running

marimo is behind the `notebooks` extra and the notebooks need `torch`, so pass
`--extra notebooks --extra torch` (or run `uv sync --extra notebooks --extra torch`
once and drop the flags afterwards):

```bash
# Interactive editor (edit + run cells) — usual choice
uv run --extra notebooks --extra torch marimo edit notebooks/01_msd_build_and_solve.py

# Read-only app view (runs it as a deployed app, cells hidden)
uv run --extra notebooks --extra torch marimo run notebooks/01_msd_build_and_solve.py
```

marimo prints a `http://localhost:2718` URL and normally opens it in your
browser automatically.

Note: the first run of each notebook generates and compiles an acados solver,
which can take a minute or two; subsequent runs are fast.

## Roadmap — future notebooks

- **Cartpole with stage-varying references** — build a swing-up OCP inline (as
  the MSD and heating notebooks do) and morph the target reference over the
  horizon using `splits`.
- **Point mass with wind** — 2-D policy gradients `du0/dp` drawn as arrows.
- **Real forecast data** — swap the synthetic profiles in `nb_utils/data.py`
  for measured weather/price time series.
