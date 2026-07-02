# Core API

The main user-facing interfaces of leap-c. Each entry links to its full,
type-annotated reference page.

- {py:class}`~leap_c.acados_torch.AcadosDiffMpcLayerTorch` — the central interface:
  wraps an acados OCP solver as a differentiable PyTorch module.
- {py:class}`~leap_c.parameters.base.AcadosParameterManager` — define and
  manage the parameters of an acados OCP without touching CasADi/acados internals.

```{autoapisummary}
leap_c.acados_torch.AcadosDiffMpcLayerTorch
leap_c.parameters.base.AcadosParameterManager
```

The complete, auto-generated reference (including developer/internal modules) is
available under [API Reference](generated/leap_c/index).
