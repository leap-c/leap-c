# Getting Started

`leap-c` turns a model predictive controller into a differentiable PyTorch layer. It wraps
acados' state-of-the-art optimal-control solver `AcadosOcpSolver` into a layer whose
**forward** pass solves a (batched) optimal control problem and whose **backward** pass returns
*exact* gradients of the solution with respect to the problem's parameters. Because it is an
ordinary `torch.nn.Module`, you can drop it into any model, compose it with neural networks, and call
`loss.backward()` straight through the solver — the basis for learning MPC parameters with
reinforcement or imitation learning.

## The parametric optimal control problem

At each step leap-c solves a discrete-time **parametric optimal control problem (OCP)**: given an
initial state $\bar{x}_0$ and a parameter vector $p$, find the state and control trajectories that
minimize the accumulated cost subject to the system dynamics and constraints.

$$
\begin{aligned}
\min_{\substack{x_0, \dots, x_N \\ u_0, \dots, u_{N-1}}}\quad
  & \sum_{k=0}^{N-1} \ell(x_k, u_k;\, p) \;+\; \ell_N(x_N;\, p) \\[2pt]
\text{s.t.}\quad
  & x_0 = \bar{x}_0, \\
  & x_{k+1} = f(x_k, u_k;\, p), && k = 0, \dots, N-1, \\
  & h(x_k, u_k;\, p) \le 0,     && k = 0, \dots, N-1, \\
  & h_N(x_N;\, p) \le 0.
\end{aligned}
$$

- $x_k \in \mathbb{R}^{n_x}$ are the **states** and $u_k \in \mathbb{R}^{n_u}$ the **controls**,
  over a horizon of $N$ stages.
- $\bar{x}_0$ is the **initial state** (`x0`), fixed for each solve.
- $p$ collects the **parameters** — dynamics constants, cost weights, references — the quantities
  you set at runtime or *learn*.
- $\ell, \ell_N$ are the stage and terminal cost, $f$ the dynamics, and $h, h_N$ the path and
  terminal constraints.

Solving this problem defines a **solution map** from $(\bar{x}_0,\, p)$ to the first optimal
control $u_0^\star(\bar{x}_0,\, p)$ and the optimal value $V(\bar{x}_0;\, p)$. leap-c makes this
map differentiable in $p$.

## The differentiable-MPC interface

{py:class}`~leap_c.acados_torch.AcadosDiffMpcLayerTorch` implements the solution map above as a
`torch.nn.Module`. You build one from an `AcadosOcp` (the problem — dynamics, cost, constraints)
and an {py:class}`~leap_c.parameters.base.AcadosParameterManager` (which numbers are
learnable), then call it like any module:

```{mermaid}
flowchart LR
  OCP["AcadosOcp<br/>(dynamics, cost, constraints)"] --> DMPC
  PM["AcadosParameterManager<br/>(differentiable vs. runtime params)"] --> DMPC
  DMPC["AcadosDiffMpcLayerTorch<br/>(torch.nn.Module)"] --> PIPE["your PyTorch model<br/>/ training loop"]
```

Its public interface is just a constructor and a `forward`:

```python
class AcadosDiffMpcLayerTorch(torch.nn.Module):

    def __init__(
        self,
        ocp: AcadosOcp,                             # the OCP: dynamics, cost, constraints
        parameter_manager: AcadosParameterManager,  # which parameters are learnable
        # ... optional solver / code-generation options
    ): ...
```

Its public interface is just a constructor and a `forward`:

```python
class AcadosDiffMpcLayerTorch(torch.nn.Module):

    def __init__(
        self,
        ocp: AcadosOcp,                             # the OCP: dynamics, cost, constraints
        parameter_manager: AcadosParameterManager,  # which parameters are learnable
        # ... optional solver / code-generation options
    ): ...

    def forward(
        self,
        x0: torch.Tensor,                                # initial state, shape (B, n_x)
        u0: torch.Tensor | None = None,                  # optional fixed first action
        params: dict[str, torch.Tensor] | None = None,   # parameter overrides p, by name
        ctx: AcadosDiffMpcCtx | None = None,             # solver context, for warm starts
    ) -> tuple[AcadosDiffMpcCtx, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # returns (ctx, u0, x, u, value)
        ...
```

`forward` maps directly onto the OCP: it fixes $x_0 = \bar{x}_0$ from `x0`, applies the named
overrides in `params` as $p$, solves, and returns the first optimal control `u0` $= u_0^\star$,
the full state and control trajectories `x`, `u`, the optimal `value` $= V(\bar{x}_0;\, p)$, and a
`ctx` used to warm-start later solves. (Omitting `u0` evaluates the state value $V(\bar{x}_0)$;
passing it evaluates the state-action value $Q(\bar{x}_0, u_0)$.) Calling `value.sum().backward()`
then propagates **exact** gradients $\partial u_0^\star / \partial p$ and $\partial V / \partial p$
back to the learnable parameter tensors.

## Minimal example

A complete use of the layer — register the learnable parameters, build the module, solve, and
differentiate through the solve — reads like this:

```python
import numpy as np
import torch

from leap_c.acados_torch import AcadosDiffMpcLayerTorch
from leap_c.parameters.base import AcadosParameterManager

# 1. Register parameters. The manager returns CasADi symbols to use in the OCP.
#    differentiable=True  -> learnable, exact gradients (shared acados p_global)
#    differentiable=False -> runtime model value, not learned (acados p)
manager = AcadosParameterManager(N_horizon=20)
Q = manager.register_parameter("Q", np.array([1.0, 1.0]), differentiable=True)
R = manager.register_parameter("R", np.array([0.1]), differentiable=True)
mass = manager.register_parameter("mass", np.array([1.0]), differentiable=False)

# 2. Build the AcadosOcp from those symbols (dynamics, cost, constraints).
#    acados-specific; see "Define a differentiable MPC" for the full definition.
ocp = build_point_mass_ocp(manager, Q, R, mass)

# 3. Wrap the OCP and its parameters into a differentiable torch.nn.Module.
diff_mpc = AcadosDiffMpcLayerTorch(ocp, manager)

# 4. Solve a batch of OCPs for an initial state, overriding a parameter by name.
x0 = torch.tensor([[1.0, 0.0]], dtype=torch.float64)                  # (batch, n_x)
Q_t = torch.tensor([[1.0, 1.0]], dtype=torch.float64, requires_grad=True)
ctx, u0, x, u, value = diff_mpc(x0, params={"Q": Q_t})

# u0 is the first optimal action; differentiate the solution end to end.
value.sum().backward()          # exact gradients w.r.t. differentiable params
print(u0, Q_t.grad)
```

`build_point_mass_ocp` stands in for the acados OCP construction, which uses the symbols
returned above in the dynamics and cost. See
[Define a differentiable MPC](define_differentiable_mpc.md) for the full, runnable version
(the OCP body spelled out) and the accompanying explanation.

## Next steps

- [Define a differentiable MPC](define_differentiable_mpc.md) — a complete, runnable example that
  builds the `AcadosOcp`, wraps it, and differentiates through a solve.
- [Parameter management](parameter_management.md) — declaring learnable vs. runtime parameters and
  varying them across the horizon.
- [Core API](api/index.md) — the main user-facing classes.
