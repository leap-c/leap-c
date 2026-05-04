# Parameter Management in Acados OCPs

This tutorial walks through `AcadosParameter` and `AcadosParameterManager` - the two
classes that control how numerical parameters flow into an acados Optimal Control Problem
(OCP) and, optionally, into a learning loop via `AcadosPlanner`.

## Parameter types at a glance

Every parameter has an `interface` that determines its role:

| Interface | Symbolic in OCP? | Changeable after solver creation? | Exposed to learning? |
|---|---|---|---|
| `"fix"` | No (constant) | No | No |
| `"non-learnable"` | Yes (`p`, per-stage) | Yes | No |
| `"learnable"` | Yes (`p_global`, shared) | Yes | Yes (gradients available) |

## Minimal example

The scenario below is a 10-step temperature-control problem with one parameter of each type,
plus a stage-varying learnable parameter to illustrate the indicator mechanism.

### Step 1 - Define parameters

> **Canonical location:** `tutorial/parameter_manager/utils.py::make_params()`

```python
import gymnasium as gym
import numpy as np
from leap_c.ocp.acados.parameters import AcadosParameter, AcadosParameterManager

N_horizon = 10  # stages 0 .. 10 (inclusive)

params = [
    # Fixed: known constant, baked into the solver at compile time.
    AcadosParameter(
        name="dt",
        default=np.array([0.25]),   # 15-minute time step [h]
        interface="fix",
    ),

    # Non-learnable: changes every call (e.g. a weather forecast),
    # but is NOT differentiated through.
    AcadosParameter(
        name="outdoor_temp",
        default=np.array([20.0]),   # ambient temperature [degC]
        interface="non-learnable",
    ),

    # Learnable, constant across stages: a single scalar the learning
    # algorithm can adjust.
    AcadosParameter(
        name="comfort_setpoint",
        default=np.array([21.0]),
        space=gym.spaces.Box(low=np.array([15.0]), high=np.array([28.0]), dtype=np.float64),
        interface="learnable",
    ),

    # Learnable, stage-varying: two price blocks - one for stages 0-4,
    # another for stages 5-10.  The manager handles the indicator mechanism
    # automatically; you still call get("price") once in the OCP formulation.
    AcadosParameter(
        name="price",
        default=np.array([0.15]),   # electricity price [EUR/kWh]
        space=gym.spaces.Box(low=np.array([0.0]), high=np.array([1.0]), dtype=np.float64),
        interface="learnable",
        end_stages=[4, N_horizon],  # block 0: stages 0-4, block 1: stages 5-10
    ),
]

manager = AcadosParameterManager(params, N_horizon=N_horizon)
```

### Step 2 - Use symbolic variables in the OCP formulation

Inside the function that builds your `AcadosOcp`, retrieve each parameter's CasADi
expression (or constant numpy value for `"fix"`) via `manager.get()`.

> **Canonical location:** `tutorial/parameter_manager/utils.py::build_ocp()`

```python
import numpy as np
from acados_template import AcadosOcp, AcadosModel
import casadi as ca

def build_ocp(manager: AcadosParameterManager, N_horizon: int) -> AcadosOcp:
    ocp = AcadosOcp()
    model = AcadosModel()
    model.name = "temp_ctrl"

    # State: room temperature [degC]
    T = ca.SX.sym("T")
    model.x = T

    # Control: heating power [kW]
    q = ca.SX.sym("q")
    model.u = q

    # Retrieve parameters - "fix" returns a numpy array, others return CasADi expressions.
    dt             = manager.get("dt")[0]            # numpy scalar (unwrap from array)
    outdoor_temp   = manager.get("outdoor_temp")     # CasADi SX, from p (per-stage)
    comfort_ref    = manager.get("comfort_setpoint") # CasADi SX, from p_global
    price          = manager.get("price")            # CasADi SX expression, stage-aware
                                                     # (weighted sum over price_0_4 / price_5_10)

    # Discrete-time dynamics: simple RC model
    R, C = 2.0, 1.5   # thermal resistance, capacitance
    model.disc_dyn_expr = T + dt * ((outdoor_temp - T) / (R * C) + q / C)

    # Stage cost: comfort deviation + price-weighted energy
    model.cost_expr_ext_cost = (T - comfort_ref) ** 2 + price * q
    # Terminal cost: comfort deviation only (no control at terminal stage)
    model.cost_expr_ext_cost_e = (T - comfort_ref) ** 2

    ocp.cost.cost_type   = "EXTERNAL"
    ocp.cost.cost_type_e = "EXTERNAL"

    ocp.model = model
    manager.assign_to_ocp(ocp)   # wires p_global and p into the ocp object

    # Provide a nominal x0 so acados allocates lbx/ubx at stage 0.
    # The actual value is overwritten at each solve call by AcadosDiffMpcTorch.
    ocp.constraints.x0 = np.array([20.0])

    ocp.solver_options.tf               = N_horizon * dt
    ocp.solver_options.N_horizon        = N_horizon
    ocp.solver_options.qp_solver        = "PARTIAL_CONDENSING_HPIPM"
    ocp.solver_options.hessian_approx   = "EXACT"
    ocp.solver_options.integrator_type  = "DISCRETE"
    return ocp
```

`manager.assign_to_ocp(ocp)` flattens both parameter structs and writes them into:
- `ocp.model.p_global` - learnable parameters (shared across all stages)
- `ocp.model.p` - non-learnable parameters (a vector applied per stage)

### Step 3 - Set values at runtime

#### Non-learnable parameters: `combine_non_learnable_parameter_values`

Call this before each solver invocation to pack a batch of per-stage parameter arrays.
Pass stage-wise forecasts as `(batch_size, N_horizon + 1, param_dim)` arrays in `overwrite`.

> **See also:** `tutorial/parameter_manager/pm_tutorial.py` (lines that build `p_stagewise`)

```python
rng = np.random.default_rng(seed=0)

batch_size = 4
N_stages   = N_horizon + 1  # 11 stages (0 .. 10)

# Outdoor temperature forecast: shape (batch_size, N_stages, 1)
temp_forecast = rng.uniform(5.0, 25.0, size=(batch_size, N_stages, 1))

# Returns shape (batch_size, N_stages, n_nonlearnable)
p_stagewise = manager.combine_non_learnable_parameter_values(
    batch_size=batch_size,
    outdoor_temp=temp_forecast,
)
# p_stagewise[:,k,:] is the non-learnable parameter vector at stage k.
```

Without `overwrite`, all stages use the default value declared in `AcadosParameter`.

#### Learnable parameters: `combine_default_learnable_parameter_values`

Use this to create an initial parameter batch, optionally overwriting specific entries.
For stage-varying parameters the overwrite array must have shape
`(batch_size, N_horizon + 1, param_dim)` - the manager picks the value at the *start* of
each block.

> **See also:** `tutorial/parameter_manager/pm_tutorial_forecast.py` (non-default learnable params block)

```python
# Stage-wise price forecast: shape (batch_size, N_stages, 1)
price_forecast = rng.uniform(0.05, 0.40, size=(batch_size, N_stages, 1))

# Per-batch comfort setpoint: shape (batch_size, 1)
comfort_values = np.array([[19.0], [21.0], [23.0], [22.5]])

# Returns shape (batch_size, N_learnable).
# Starts from defaults and replaces the named entries.
param = manager.combine_default_learnable_parameter_values(
    comfort_setpoint=comfort_values,
    price=price_forecast,  # block 0 value taken from stage 0, block 1 from stage 5
)
```

## The indicator mechanism

`price` has two stage blocks (`end_stages=[4, 10]`).  Internally the manager stores
`price_0_4` and `price_5_10` in `p_global`, and appends an `indicator` vector of length
`N_horizon + 1` to `p`.  At stage `k`, `indicator[k] = 1` and all other entries are zero.

`manager.get("price")` returns:

```python
sum(indicator[0:5]) * price_0_4 + sum(indicator[5:11]) * price_5_10
```

This single CasADi expression evaluates to the correct block value at every stage without
any conditional logic in the OCP.

`combine_non_learnable_parameter_values` sets the indicator automatically - you never
have to touch it manually.  If you bypass this method and leave the indicator all-zero,
every stage silently evaluates to zero for all stage-varying learnable parameters.

## Using with AcadosPlanner

### Basic usage

`AcadosPlanner` wraps the above pattern.  Its `forward` method calls
`combine_non_learnable_parameter_values` internally using the default values for all
non-learnable parameters; the caller only needs to supply the initial state and the
learnable parameter tensor `param`.

> **Complete example:** `tutorial/parameter_manager/pm_tutorial.py`

```python
from leap_c.ocp.acados.planner import AcadosPlanner
from leap_c.ocp.acados.torch import AcadosDiffMpcTorch

ocp      = build_ocp(manager, N_horizon)
diff_mpc = AcadosDiffMpcTorch(ocp)
planner  = AcadosPlanner(param_manager=manager, diff_mpc=diff_mpc)

# p_global batch: shape (batch_size, N_learnable)
param = manager.combine_default_learnable_parameter_values(batch_size=batch_size)

# planner.forward calls combine_non_learnable_parameter_values internally
# (outdoor_temp stays at its default 20 degC for every stage)
ctx, u0, x, u, value = planner.forward(obs=x0_batch, param=param)
```

### Forecast-aware planner (subclassing `AcadosPlanner`)

When non-learnable parameters must be set from per-call data (e.g. a live weather
forecast), subclass `AcadosPlanner` and override `forward`.  The override should:

1. Extract the forecast from the observation dict.
2. Call `combine_non_learnable_parameter_values` with the overwrite to build `p_stagewise`.
3. Pass `p_stagewise` directly to `self.diff_mpc`.

> **Complete example:** `tutorial/parameter_manager/pm_tutorial_forecast.py`

```python
from typing import Any
import numpy as np
import torch
from leap_c.ocp.acados.diff_mpc import AcadosDiffMpcCtx
from leap_c.ocp.acados.planner import AcadosPlanner

class TempCtrlPlanner(AcadosPlanner):
    """Planner that injects an outdoor temperature forecast at every solve call."""

    def forward(
        self,
        obs: dict[str, Any],
        action: torch.Tensor | None = None,
        param: torch.Tensor | None = None,
        ctx: AcadosDiffMpcCtx | None = None,
    ):
        state         = obs["state"]                    # (batch_size, 1)
        temp_forecast = obs["outdoor_temp_forecast"]    # (batch_size, N_stages, 1)
        batch_size    = state.shape[0]

        # Build per-stage non-learnable params with the forecast injected.
        p_stagewise = self.param_manager.combine_non_learnable_parameter_values(
            batch_size=batch_size,
            outdoor_temp=temp_forecast,
        )

        return self.diff_mpc(state, action, param, p_stagewise, ctx=ctx)
```

The caller assembles the observation dict and non-default learnable params, then calls
`forward` as usual:

```python
planner = TempCtrlPlanner(param_manager=manager, diff_mpc=diff_mpc)

obs = {
    "state":                 x0_batch,       # torch.Tensor (batch_size, 1)
    "outdoor_temp_forecast": temp_forecast,  # np.ndarray  (batch_size, N_stages, 1)
}

# Non-default learnable params: per-batch comfort setpoint + stage-varying price
param = torch.tensor(
    manager.combine_default_learnable_parameter_values(
        comfort_setpoint=comfort_values,  # (batch_size, 1)
        price=price_forecast,             # (batch_size, N_stages, 1)
    )
)

ctx, u0, x, u, value = planner.forward(obs=obs, param=param)
```

## Complete implementations

The runnable scripts for this tutorial live in `tutorial/parameter_manager/`:

| File | What it shows |
|---|---|
| `tutorial/parameter_manager/utils.py` | Shared constants (`N_HORIZON`, `BATCH_SIZE`), `make_params()`, `build_ocp()` |
| `tutorial/parameter_manager/pm_tutorial.py` | Basic `AcadosPlanner` usage; default non-learnable params; `combine_*` illustration |
| `tutorial/parameter_manager/pm_tutorial_forecast.py` | `TempCtrlPlanner` subclass; forecast in obs dict; non-default learnable params |
