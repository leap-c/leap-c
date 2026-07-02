# Core API

The main user-facing interfaces of leap-c. Each entry links to its full,
type-annotated reference page.

- {py:class}`~leap_c.ocp.acados.torch.AcadosDiffMpcTorch` — the central interface:
  wraps an acados OCP solver as a differentiable PyTorch module.
- {py:class}`~leap_c.ocp.acados.parameters.AcadosParameterManager` — define and
  manage the parameters of an acados OCP without touching CasADi/acados internals.
- {py:class}`~leap_c.controller.ParameterizedController` — abstract base class for
  differentiable, parameterized controllers.
- {py:class}`~leap_c.trainer.Trainer` — base training loop for RL / imitation
  learning.

```{autoapisummary}
leap_c.ocp.acados.torch.AcadosDiffMpcTorch
leap_c.ocp.acados.parameters.AcadosParameterManager
leap_c.controller.ParameterizedController
leap_c.trainer.Trainer
```

The complete, auto-generated reference (including developer/internal modules) is
available under [API Reference](generated/leap_c/index).
