# Notebooks

These are [marimo](https://marimo.io) notebooks (plain `.py` files, not `.ipynb`).

**`getting_started/`** is a sequential course: eight notebooks, one focus each,
on two systems only (a mass-spring-damper and an R1C1 house-heating problem).
Read in order; afterwards you can formulate, batch, differentiate, and train
your own problem.

Shared helpers live in `nb_utils/` at this directory's root.

## getting_started — the course

The story: differentiable MPC in one page (01), then a house-heating problem
carried from a plain acados OCP (02) all the way to imitation and
reinforcement learning (07, 08).

| # | Notebook | System | Focus |
|---|----------|--------|-------|
| 01 | `01_intro_diff_mpc.py` | mass-spring-damper | end-to-end in five minutes: register a parameter, build, solve, read the 5-tuple, `.backward()` |
| 02 | `02_from_acados_to_diff_mpc.py` | R1C1 heating | converting an existing plain `AcadosOcp` + `AcadosOcpSolver`; what carries over, what changes; first `dV/dR` |
| 03 | `03_gradients_through_the_solver.py` | R1C1 heating | V vs. Q; the three autograd routes; where policy gradients die (saturation) |
| 04 | `04_parameter_management.py` | R1C1 heating | the full parameter model: `differentiable=`, `splits=`, override shapes, guard rails |
| 05 | `05_batched_solves_and_forecasts.py` | R1C1 heating | the batch dimension; forecasts through both parameter interfaces; value/policy curves; `∂V/∂price` |
| 06 | `06_planner_interface.py` | R1C1 heating | a `forward(obs) → action` planner; time-varying slacked comfort band; warm starts; closed loop vs. a mismatched house |
| 07 | `07_imitation_learning.py` | R1C1 heating | behavior cloning through the solver; the learned parameters as a portrait of occupant + building; `collate_torch` |
| 08 | `08_rl_on_closed_loop_cost.py` | R1C1 heating | backprop through a closed-loop rollout; BOPTEST-style cost + discomfort objective; `discount_factor=` |

## API coverage map

Where each piece of the core API is taught (the canonical index — if you are
looking for how to do X, start at the notebook named here):

| API surface | Notebook |
|---|---|
| `AcadosParameterManager`, `register_parameter(differentiable=)` | 01, 02 |
| `splits=` ("global" / int / list / "stagewise"), override shape contract | 04 |
| `model.p` vs. `model.p_global` (the interface matrix) | 04 (concept in 02) |
| guard rails (non-diff + `requires_grad`, late registration, bad splits, wrong shapes) | 04 |
| `AcadosDiffMpcTorch(ocp, manager)`, the 5-tuple `(ctx, u0, x, u, value)` | 01 |
| `verbose=`, `export_directory=`, module `repr` | 02 |
| `n_batch_init=`, `num_threads_batch_solver=`, `dtype=` | 05 |
| `discount_factor=` | 08 (noted in 03) |
| `forward(x0)` → V vs. `forward(x0, u0)` → Q | 03 |
| `params={...}` overrides (numpy / tensors, batched) | 01, 04, 05 |
| `.backward()` / `torch.autograd.grad` / `functional.jacobian` | 01, 03 |
| batched solves (sweeps, forecast windows, value/policy curves) | 05 |
| ctx warm starts (receding horizon, training epochs), `ctx.status` handling | 06, 07 |
| time-varying slacked bounds via parametric h-constraints | 06 (builder in `nb_utils/heating.py`) |
| training loops (behavior cloning; backprop-through-rollout) | 07, 08 |
| `collate_torch` (incl. collating stored `AcadosDiffMpcCtx`) | 07 |

Deliberately not covered in notebooks: the low-level
`diff_mpc_fun.sensitivity(ctx, field)` API that returns Jacobian blocks
straight off the solver context (named in 03 and 05), custom
`AcadosDiffMpcInitializer` (named in 06), `casadi_type="MX"` (note in 04),
and the manager's internal `combine_*` methods (never user-called).

## Conventions

- **Taught inline exactly once, imported everywhere else.** A builder or
  class whose construction *is* the lesson appears inline in one notebook
  and has a synced copy in `nb_utils/` (reciprocal `NOTE:` headers mark the
  pairs): the heating OCP builder ⇄ 02, `HeatingPlanner` ⇄ 06. Everything
  else lives only in `nb_utils/`.
- **Sliders never trigger solves** — where a notebook is interactive, the
  slider indexes a precomputed batched solve.
- **Bootstrap cell.** Notebooks live in subfolders, so each one inserts the
  notebooks root into `sys.path` at the top of its imports cell
  (`sys.path.insert(0, str(mo.notebook_dir().parent))`) before importing
  `nb_utils`.
- **Distinct `ocp.model.name` per notebook**, so generated solver code never
  collides.

## Running

marimo is behind the `notebooks` extra and the notebooks need `torch`, so pass
`--extra notebooks --extra torch` (or run `uv sync --extra notebooks --extra torch`
once and drop the flags afterwards):

```bash
# Interactive editor (edit + run cells) — usual choice
uv run --extra notebooks --extra torch marimo edit notebooks/getting_started/01_intro_diff_mpc.py

# Read-only app view (runs it as a deployed app, cells hidden)
uv run --extra notebooks --extra torch marimo run notebooks/getting_started/01_intro_diff_mpc.py
```

marimo prints a `http://localhost:2718` URL and normally opens it in your
browser automatically.

Note: the first run of each notebook generates and compiles an acados solver,
which can take a minute or two; subsequent runs are fast. Every notebook is
executed headlessly in CI by `tests/test_notebooks.py`.

## Roadmap — future notebooks

- **Cartpole with stage-varying references** — build a swing-up OCP inline
  and morph the target reference over the horizon using `splits`.
- **Point mass with wind** — 2-D policy gradients `du0/dp` drawn as arrows.
- **Real forecast data** — swap the synthetic profiles in `nb_utils/data.py`
  for measured weather/price time series.
