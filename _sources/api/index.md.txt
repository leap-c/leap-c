# Core API

The main user-facing interfaces of leap-c. Each entry links to its full,
type-annotated reference page.

- {py:class}`~leap_c.torch.AcadosDiffMpcTorch` — the central interface:
  wraps an acados OCP solver as a differentiable PyTorch module.
- {py:class}`~leap_c.parameters.AcadosParameterManager` — define and
  manage the parameters of an acados OCP without touching CasADi/acados internals.
- {py:func}`~leap_c.utils.collate.collate_torch` — PyTorch default collation plus the leap-c
  context rule, useful when batching warm-start contexts.

```{autoapisummary}
leap_c.torch.AcadosDiffMpcTorch
leap_c.parameters.AcadosParameterManager
leap_c.utils.collate.collate_torch
```

The complete, auto-generated reference (including developer/internal modules) is
available under [API Reference](generated/leap_c/index).
