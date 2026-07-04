[![Source Code License](https://img.shields.io/badge/license-BSD-blueviolet?style=for-the-badge)](https://github.com/leap-c/leap-c/blob/main/LICENSE)
![Python 3.11](https://img.shields.io/badge/python->=3.11-green.svg?style=for-the-badge)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json&style=for-the-badge)](https://docs.astral.sh/ruff/)
[![codecov](https://img.shields.io/codecov/c/github/leap-c/leap-c?token=TQE7RZ1O7M&style=for-the-badge)](https://codecov.io/github/leap-c/leap-c)

# leap-c

`leap-c` or (learning predictive control) provides a simple way how to wrap optimal-control solvers into modern deep learning pipelines like 
[PyTorch](https://pytorch.org/).

This repository provides a differentiable layer based on the very fast [acados](https://github.com/acados/acados) framework. Higher-level planners, environments, and RL training utilities live in downstream projects such as
[leap-c-lab](https://github.com/leap-c/leap-c-lab) and
[mpc-sac](https://github.com/leap-c/mpc-sac).

> [!NOTE]
> The repository recently moved toward a smaller core interface. The previous broader interface is
> available in [`v0.2.0-alpha`](https://github.com/leap-c/leap-c/tree/v0.2.0-alpha).

## Key Features

- A simple `torch` interface for `acados` called  `AcadosDiffMpcTorch`.
- Solve optimal control problems in parallel using multithreading.
- Backpropagate exact solution sensitivities through the MPC layer.
- Retrieve sensitivities of the optimal cost and the optimal control sequence with respect to problem parameters.
- Details as warm-starting, initialization and more are conveniently handled by the `AcadosDiffMpcTorch` interface.

## Installation

Follow the [installation guide](https://leap-c.github.io/leap-c/installation.html).

## Documentation

- [Getting started](https://leap-c.github.io/leap-c/getting_started.html)
- [Define a differentiable MPC](https://leap-c.github.io/leap-c/define_differentiable_mpc.html)
- [Parameter management](https://leap-c.github.io/leap-c/parameter_management.html)
- [Running notebooks](https://leap-c.github.io/leap-c/notebooks.html)
- [API reference](https://leap-c.github.io/leap-c/api/index.html)

## Minimal Interaction

```python
import torch

from leap_c.torch import AcadosDiffMpcTorch

# Build these with acados/CasADi and leap-c's parameter manager.
ocp = build_acados_ocp(...)
parameter_manager = build_parameter_manager(...)

diff_mpc = AcadosDiffMpcTorch(ocp, parameter_manager)

x0 = torch.tensor([[1.0, 0.0]], dtype=torch.float64)
Q = torch.tensor([[1.0, 1.0]], dtype=torch.float64, requires_grad=True)

ctx, u0, x, u, value = diff_mpc(x0, params={"Q": Q})
value.sum().backward()

print(u0)      # first optimal action
print(Q.grad)  # gradient through the MPC solve
```

The snippet omits the acados OCP construction. For a full runnable example, see
[Define a differentiable MPC](https://leap-c.github.io/leap-c/define_differentiable_mpc.html).

## Ecosystem

- [leap-c-lab](https://github.com/leap-c/leap-c-lab): planners, OCP definitions, gym
  environments, and controller/planner abstractions built on the core layer.
- [mpc-sac](https://github.com/leap-c/mpc-sac): SAC/CrossQ-style RL training utilities for
  learning MPC controllers with `leap-c` and `leap-c-lab`.


## Questions?

Open a new thread or browse existing ones on the
[GitHub discussions](https://github.com/leap-c/leap-c/discussions) page.

## Citing

If you use code from this repository in your work, please cite:

```bibtex
@misc{fichtner_leapc_2025,
  title = {Leap-c/Leap-c: V0.1.0-Alpha},
  author = {Fichtner, Leonard and Reinhardt, Dirk and Hoffmann, Jasper and Airaldi, Filippo and Frey, Jonathan and Kir Hromatko, Josip and Baumgaertner, Katrin and Amria, Mazen and Reiter, Rudolf and Sawant, Shambhuraj},
  year = 2025,
  month = oct,
  howpublished = {Zenodo}
}
```
