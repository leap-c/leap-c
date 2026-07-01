# Getting Started

`leap-c` turns a model predictive controller into a differentiable PyTorch layer. It wraps
acados' state-of-the-art optimal-control solver `AcadosOcpSolver` in a `torch.nn.Module` whose
**forward** pass solves a (batched) optimal control problem and whose **backward** pass returns
*exact* gradients of the solution with respect to the problem's parameters. Because it is an
ordinary `nn.Module`, you can drop it into any model, compose it with neural networks, and call
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

{py:class}`~leap_c.ocp.acados.torch.AcadosDiffMpcTorch` implements the solution map above as a
`torch.nn.Module`. You build one from an `AcadosOcp` (the problem — dynamics, cost, constraints)
and an {py:class}`~leap_c.ocp.acados.parameters.AcadosParameterManager` (which numbers are
learnable), then call it like any module:

```{mermaid}
flowchart LR
  OCP["AcadosOcp<br/>(dynamics, cost, constraints)"] --> DMPC
  PM["AcadosParameterManager<br/>(differentiable vs. runtime params)"] --> DMPC
  DMPC["AcadosDiffMpcTorch<br/>(torch.nn.Module)"] --> PIPE["your PyTorch model<br/>/ training loop"]
```

Its public interface is just a constructor and a `forward`:

```python
class AcadosDiffMpcTorch(torch.nn.Module):

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

## Next steps

- [Define a differentiable MPC](define_differentiable_mpc.md) — a complete, runnable example that
  builds the `AcadosOcp`, wraps it, and differentiates through a solve.
- [Parameter management](parameter_management.md) — declaring learnable vs. runtime parameters and
  varying them across the horizon.
- [Core API](api/index.md) — the main user-facing classes.
- [Examples](https://github.com/leap-c/leap-c/tree/main/leap_c/examples) — complete environments
  and planners.
