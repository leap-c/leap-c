# Parameter Management in Acados OCPs

This tutorial walks through `AcadosParameterManager` — the class that controls how numerical
parameters flow into an acados Optimal Control Problem (OCP) and, optionally, into a learning
loop via `AcadosDiffMpcLayerTorch`.

You register each parameter once with `manager.register_parameter(...)`, which returns a CasADi
symbol you drop straight into your dynamics, cost, and constraint expressions. At solve time you
pass a dict of named parameter values to `AcadosDiffMpcLayerTorch`; the manager routes each value to the
right place (per-stage `p` or shared `p_global`) internally.

## Parameter interfaces at a glance

Every parameter has one of two interfaces, selected by the `differentiable` flag:

| `differentiable` | Interface | Symbolic in OCP | Changeable after solver creation | Exposed to learning |
|---|---|---|---|---|
| `False` | `"non-differentiable"` | Yes (`p`, per-stage) | Yes | No |
| `True` | `"differentiable"` | Yes (`p_global`, shared) | Yes | Yes (gradients / sensitivities) |

A value that never changes (e.g. the time step `dt`) does not need to be a parameter at all — use a
plain Python constant. There is no separate "fixed" interface.

Stage-varying differentiable parameters (e.g. an electricity price with two blocks) are declared
with the `splits` argument, covered in the "Stage-varying parameters and `splits`" section below.

## Minimal example

The scenario below is an `N_horizon`-step temperature-control problem with a non-differentiable
outdoor-temperature parameter, a differentiable comfort setpoint, and a stage-varying differentiable
price.

### Step 1 — Register parameters

`register_parameter` returns a real CasADi symbol (not a placeholder) that you can use immediately.

```python
import casadi as ca
import numpy as np
import torch
from acados_template import AcadosOcp

from leap_c.acados_torch import AcadosDiffMpcLayerTorch
from leap_c.parameters.base import AcadosParameterManager

N_horizon = 20  # stages 0 .. N_horizon (inclusive)

manager = AcadosParameterManager(N_horizon=N_horizon)

# Non-differentiable: ambient temperature, changed per call, no gradient.
outdoor_temp = manager.register_parameter(
    name="outdoor_temp",
    default=np.array([20.0]),   # ambient temperature [degC]
    differentiable=False,
)

# Differentiable, constant across stages: a single scalar the learner can adjust.
comfort_ref = manager.register_parameter(
    name="comfort_setpoint",
    default=np.array([21.0]),
    differentiable=True,
)

# Differentiable and stage-varying: two price blocks.
# With splits=[4, N_horizon]: block 0 covers stages 0-4, block 1 covers stages 5-N_horizon.
price = manager.register_parameter(
    name="price",
    default=np.array([0.15]),   # electricity price [EUR/kWh]
    differentiable=True,
    splits=[4, N_horizon],
)
```

### Step 2 — Build the OCP with the returned symbols

Use the symbols returned by `register_parameter` directly. (You can also retrieve any registered
symbol later with `manager.get(name)`, which is handy when parameter registration and OCP
construction live in separate functions.)

```python
R_THERMAL, C_THERMAL = 2.0, 1.5  # thermal resistance, capacitance
dt = 0.25                        # 15-minute time step [h] — a plain constant, not a parameter

ocp = AcadosOcp()
ocp.model.name = "temp_ctrl"

# State: room temperature [degC]
T = ca.SX.sym("T")
ocp.model.x = T

# Control: heating power [kW]
q = ca.SX.sym("q")
ocp.model.u = q

# Discrete-time RC dynamics using the registered symbols.
ocp.model.disc_dyn_expr = T + dt * ((outdoor_temp - T) / (R_THERMAL * C_THERMAL) + q / C_THERMAL)

# Stage cost: comfort deviation + price-weighted energy; terminal cost: comfort deviation only.
ocp.cost.cost_type = "EXTERNAL"
ocp.cost.cost_type_e = "EXTERNAL"
ocp.model.cost_expr_ext_cost = (T - comfort_ref) ** 2 + price * q
ocp.model.cost_expr_ext_cost_e = (T - comfort_ref) ** 2

# Provide a nominal x0 so acados allocates lbx/ubx at stage 0; the actual value is supplied per solve.
ocp.constraints.x0 = np.array([20.0])
ocp.constraints.lbu = np.array([-1.0])
ocp.constraints.ubu = np.array([1.0])
ocp.constraints.idxbu = np.array([0])

ocp.solver_options.tf = N_horizon * dt
ocp.solver_options.N_horizon = N_horizon
ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
ocp.solver_options.hessian_approx = "EXACT"
ocp.solver_options.integrator_type = "DISCRETE"
```

### Step 3 — Wrap in `AcadosDiffMpcLayerTorch` and solve

`AcadosDiffMpcLayerTorch` takes the OCP and its manager. Call it with the initial state and a `params`
dict of named overrides; any parameter you omit keeps its registered default.

```python
diff_mpc = AcadosDiffMpcLayerTorch(ocp, manager)

rng = np.random.default_rng(seed=0)
x0_batch = torch.tensor(rng.uniform(10.0, 30.0, size=(4, 1)))

# Stage-varying price override: shape (batch_size, n_segments); splits=[4, N_horizon] -> 2 segments.
price_tensor = torch.tensor(
    [[0.015, 0.020], [0.20, 0.25], [0.25, 0.30], [0.30, 0.35]],
    dtype=torch.float64,
    requires_grad=True,
)

ctx, u0, x, u, value = diff_mpc(x0_batch, params={"price": price_tensor})
# ctx.status == [0, 0, 0, 0] means all four solves succeeded.
```

### Step 4 — Gradients and sensitivities

Because `AcadosDiffMpcLayerTorch` is a differentiable `nn.Module`, standard PyTorch autograd works: back-
propagate through the solution, or use `torch.autograd.functional.jacobian` for full sensitivities.
No custom sensitivity API is needed.

```python
# Backpropagate the summed value through the solver to the price tensor.
value.sum().backward()
print(price_tensor.grad)

# du0/dprice: how the first action changes w.r.t. the price parameter.
# Re-solving warmstarted from ctx keeps this cheap.
j_u0 = torch.autograd.functional.jacobian(
    lambda p: diff_mpc(x0_batch, params={"price": p}, ctx=ctx)[1],
    price_tensor,
)
```

## Stage-varying parameters and `splits`

`splits` controls how a differentiable parameter varies across the horizon (it is only used for
differentiable parameters). It accepts:

- `list[int]`: sorted, ascending stage boundaries; the parameter takes one value per resulting
  segment. With `N_horizon = 9` and `splits=[4, 9]`, the parameter is constant over stages 0–4
  and takes a second value over stages 5–9. The last boundary must be `N_horizon` or `N_horizon - 1`.
- `int`: that many equal-sized segments.
- `"stagewise"`: one value per stage (equivalent to `list(range(N_horizon + 1))`).
- `"global"` (default): a single value shared across all stages.

Internally, stage variation is implemented with a one-hot **indicator** vector appended to the
non-differentiable parameters. The manager stores each block in `p_global` and `manager.get("price")`
returns a single indicator-gated expression that evaluates to the correct block value at every stage
— no conditional logic in the OCP. The indicator is set for you by
`combine_non_differentiable_parameters` (and by `AcadosDiffMpcLayerTorch` when it packs the per-stage
parameters). If the indicator were left all-zero, every stage would silently evaluate to zero for the
stage-varying parameters.

## Setting values at runtime

The primary path is the `params` dict passed to `AcadosDiffMpcLayerTorch`, keyed by parameter name. It
accepts both differentiable and non-differentiable overrides together; the manager routes each to the
right internal array:

```python
temp_forecast = rng.uniform(5.0, 25.0, size=(4, N_horizon + 1, 1))  # per-stage, non-differentiable
comfort_values = torch.tensor([[19.0], [21.0], [23.0], [22.5]], dtype=torch.float64)

ctx, u0, x, u, value = diff_mpc(
    x0=x0_batch,
    params={
        "outdoor_temp": temp_forecast,       # non-differentiable, per-stage forecast
        "comfort_setpoint": comfort_values,  # differentiable, per-batch scalar
        "price": price_tensor,               # differentiable, stage-varying
    },
)
```

If you need the packed arrays directly (for inspection or a custom solve loop), two lower-level
helpers build them explicitly:

- `combine_non_differentiable_parameters(batch_size=..., **overwrite)` returns the per-stage array of
  shape `(batch_size, N_horizon + 1, n_non_differentiable)`. Overwrites are `(batch_size,
  N_horizon + 1, pdim)` arrays; omitted parameters use their defaults, and the indicator is filled in
  automatically.
- `combine_differentiable_parameters_torch(batch_size=..., device=..., dtype=..., **overwrite)`
  returns the `p_global` tensor of shape `(batch_size, n_differentiable)`. For a stage-varying
  parameter the overwrite is `(batch_size, n_segments)` (one value per block).

```python
p_stagewise = manager.combine_non_differentiable_parameters(
    batch_size=4,
    outdoor_temp=temp_forecast,
)
p_global = manager.combine_differentiable_parameters_torch(
    batch_size=4,
    device=torch.device("cpu"),
    dtype=torch.float64,
    price=price_tensor,
)
```

A `combine_differentiable_parameters_jax` counterpart is available for JAX-based workflows.
