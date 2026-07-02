# Define a differentiable MPC

`leap-c` provides **differentiable model predictive control (MPC)**: it wraps a
state-of-the-art numerical optimal-control solver — acados' `AcadosOcpSolver` — so that a
whole MPC controller can live inside a PyTorch learning pipeline and be trained end to end.

The central interface is {py:class}`~leap_c.ocp.acados.torch.AcadosDiffMpcTorch`, a
`torch.nn.Module`. Its `forward` solves a (batched) optimal control problem with acados; its
`backward` returns *exact* gradients of the solution with respect to the problem's parameters.
Because it is an ordinary `nn.Module`, you can drop it into any model, compose it with neural
networks, and call `loss.backward()` straight through the solver.

To build one you need three things:

1. an `AcadosOcp` describing the optimal control problem (dynamics, cost, constraints);
2. an {py:class}`~leap_c.ocp.acados.parameters.AcadosParameterManager` declaring which numbers
   in that problem are **differentiable** (learnable, shared `p_global`) and which are
   **non-differentiable** (runtime-settable model values, per-stage `p`);
3. {py:class}`~leap_c.ocp.acados.torch.AcadosDiffMpcTorch`, which wraps the two.

```{mermaid}
flowchart LR
  OCP["AcadosOcp<br/>(dynamics, cost, constraints)"] --> DMPC
  PM["AcadosParameterManager<br/>(differentiable vs. runtime params)"] --> DMPC
  DMPC["AcadosDiffMpcTorch<br/>(torch.nn.Module)"] --> PIPE["your PyTorch model<br/>/ training loop"]
```

## Minimal example

The problem below is a `N_horizon`-step point mass: state `[position, velocity]`, control
`force`. `mass` is a non-differentiable model constant (set at runtime, not learned), while the
cost weights `Q` and `R` are differentiable (gradients flow back to them).

### Step 1 — Declare parameters

`register_parameter` returns a real CasADi symbol (not a placeholder) that you use directly in
the OCP expressions. The `differentiable` flag selects the interface.

```python
import casadi as ca
import numpy as np
import torch
from acados_template import AcadosOcp

from leap_c.ocp.acados.parameters import AcadosParameterManager
from leap_c.ocp.acados.torch import AcadosDiffMpcTorch

N_horizon = 20  # stages 0 .. N_horizon (inclusive)
dt = 0.05       # time step [s] — a plain constant, not a parameter

manager = AcadosParameterManager(N_horizon=N_horizon)

# Non-differentiable: a model constant set at runtime but not learned.
mass = manager.register_parameter(
    name="mass",
    default=np.array([1.0]),  # [kg]
    differentiable=False,
)

# Differentiable: cost weights a learner can tune (gradients flow to these).
Q = manager.register_parameter("Q", default=np.array([1.0, 1.0]), differentiable=True)
R = manager.register_parameter("R", default=np.array([0.1]), differentiable=True)
```

### Step 2 — Build the OCP with the returned symbols

Use the symbols from `register_parameter` directly in the dynamics and cost. (You can also
retrieve any registered symbol later with `manager.get(name)`.)

```python
ocp = AcadosOcp()
ocp.model.name = "point_mass"

x = ca.SX.sym("x", 2)  # [position, velocity]
u = ca.SX.sym("u", 1)  # force
ocp.model.x = x
ocp.model.u = u

# Discrete-time double integrator using the registered symbols.
ocp.model.disc_dyn_expr = ca.vertcat(x[0] + dt * x[1], x[1] + dt * u[0] / mass)

ocp.cost.cost_type = "EXTERNAL"
ocp.cost.cost_type_e = "EXTERNAL"
ocp.model.cost_expr_ext_cost = Q[0] * x[0] ** 2 + Q[1] * x[1] ** 2 + R[0] * u[0] ** 2
ocp.model.cost_expr_ext_cost_e = Q[0] * x[0] ** 2 + Q[1] * x[1] ** 2

# Nominal x0 so acados allocates the stage-0 bounds; the value is supplied per solve.
ocp.constraints.x0 = np.array([1.0, 0.0])
ocp.constraints.idxbu = np.array([0])
ocp.constraints.lbu = np.array([-10.0])
ocp.constraints.ubu = np.array([10.0])

ocp.solver_options.N_horizon = N_horizon
ocp.solver_options.tf = N_horizon * dt
ocp.solver_options.integrator_type = "DISCRETE"
ocp.solver_options.nlp_solver_type = "SQP"
ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
ocp.solver_options.hessian_approx = "EXACT"
```

### Step 3 — Wrap in `AcadosDiffMpcTorch`

Pass the OCP and its manager. The manager attaches the parameter symbols to the OCP and
generates the acados solvers. (The OCP must *not* already have `model.p` / `model.p_global`
set — the manager assigns them.)

```python
diff_mpc = AcadosDiffMpcTorch(ocp, manager)
```

### Step 4 — Forward and backward

Call the module with an initial state `x0` and a `params` dict of named overrides; any
parameter you omit keeps its registered default. `forward` returns the 5-tuple
`(ctx, u0, x, u, value)`: the solver context (for warm-starts), the first optimal action `u0`,
the full state and control trajectories `x`, `u`, and the optimal `value`. Then autograd works
as usual — `value.sum().backward()` populates `.grad` on the differentiable parameter tensors.

```python
x0 = torch.tensor([[1.0, 0.0]], dtype=torch.float64)  # shape (batch, x_dim)
Q_t = torch.tensor([[1.0, 1.0]], dtype=torch.float64, requires_grad=True)
R_t = torch.tensor([[0.1]], dtype=torch.float64, requires_grad=True)

ctx, u0, x, u, value = diff_mpc(x0, params={"Q": Q_t, "R": R_t})
# ctx.status == [0] means the solve succeeded.

value.sum().backward()
print(Q_t.grad, R_t.grad)
```

```{mermaid}
flowchart LR
  IN["x0, params<br/>(torch tensors)"] --> FWD["forward()<br/>batched acados solve"]
  FWD --> OUT["ctx, u0, x, u, value"]
  OUT --> LOSS["loss = value.sum()"]
  LOSS -. "backward()" .-> G["gradients w.r.t.<br/>differentiable params"]
```

Parameters are always passed **by name** inside the `params` dict — there is no top-level
`p_global=` argument. Passing `u0` in addition to `x0` evaluates a state-action value
`Q(x0, u0)`; omitting it (as above) evaluates a state value `V(x0)`.

## Differentiable and non-differentiable parameters

The `differentiable` flag chosen in Step 1 is the core concept of the parameter manager:

- **`differentiable=True`** — shared across the horizon as acados `p_global`, exposed to the
  learning interface with exact gradients / sensitivities (the `Q`, `R` weights above).
- **`differentiable=False`** — per-stage acados `p`, changeable at solve time but not
  differentiated (the `mass` above). At least one such parameter is required, so `p` is
  non-empty.

Differentiable parameters can also vary across the horizon (e.g. a time-of-use price) via the
`splits` argument. For the full treatment — stage-varying `splits`, runtime value packing, and
the lower-level `combine_*` helpers — see the
[parameter management guide](parameter_management.md).

## Building a planner for a custom environment

The examples wrap the pattern above into reusable **planners** for concrete
[gym](https://gymnasium.farama.org/) environments. Each package under
[`leap_c/examples/`](https://github.com/leap-c/leap-c/tree/main/leap_c/examples) bundles an
`env.py` (the environment), an `acados_ocp.py` (builds the `AcadosOcp` and its
`AcadosParameterManager`), and a `planner.py` (a `ParameterizedPlanner` that holds an
`AcadosDiffMpcTorch` and forwards observations to it).

You can instantiate a ready-made planner by name and query it like the module above:

```python
from leap_c.examples import create_planner

planner = create_planner("cartpole")
ctx, u0, x, u, value = planner.forward(obs=x0)
```

The `notebooks/intro.py` marimo notebook demonstrates this on the CartPole example. To *learn*
the parameters (reinforcement or imitation learning), wrap a planner in a
{py:class}`~leap_c.trainer.Trainer`.

## Next steps

- [Parameter management](parameter_management.md) — the full parameter API.
- [Core API](api/index.md) — the main user-facing classes.
- [Examples](https://github.com/leap-c/leap-c/tree/main/leap_c/examples) — complete environments
  and planners.
