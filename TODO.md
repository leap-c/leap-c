# TODO

## Framework isomorphism for `AcadosParameterManager`

Currently `AcadosParameterManager` is a single concrete class with
framework-specific methods:

- `combine_differentiable_parameters_torch()` — PyTorch (hard dependency
  on `torch`)
- `combine_differentiable_parameters_jax()` — JAX placeholder
  (`NotImplementedError`)

This prevents clean separation of framework concerns and makes it hard to
add a new backend (e.g. a pure numpy version for testing without torch).

### Desired design

`AcadosParameterManager` should be abstract (`ABC, Generic[TensorType]`)
with a single abstract method `combine_differentiable_parameters()`. Each
framework (torch, jax, ...) would provide a concrete subclass implementing
it.

```python
class AcadosParameterManager(ABC, Generic[TensorType]):
    @abstractmethod
    def combine_differentiable_parameters(
        self, batch_size: int | None = None, **overwrites
    ) -> TensorType: ...


class AcadosParameterManagerTorch(AcadosParameterManager[torch.Tensor]):
    def combine_differentiable_parameters(self, ...): ...


class AcadosParameterManagerJax(AcadosParameterManager[Any]):
    def combine_differentiable_parameters(self, ...): ...
```

`combine_non_differentiable_parameters()` stays on the base class (it
returns `np.ndarray` — no framework dependency).

### Why not yet

Making this abstract requires splitting the `combine_*` method into a
torch-specific subclass and updating all call sites to reference the
concrete type. This should be done together with (or after) the JAX
implementation so both backends exercise the abstraction.
