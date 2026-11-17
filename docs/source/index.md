# leap-c

## Learning Predictive Control

A framework for integrating optimal control solvers into deep learning pipelines.

### Key Features

- Combines model predictive control with reinforcement learning and imitation learning
- Built on top of community-standard tools [acados](https://docs.acados.org/) and [CasADi](https://web.casadi.org/)
- Seamless integration with [PyTorch](https://pytorch.org/) and [JAX](https://docs.jax.dev/) for deep learning. The OCP solver is wrapped into a PyTorch/JAX module and can be differentiated end-to-end.

### Development

leap-c is developed through a collaboration between:

- [Department of Engineering Cybernetics](https://www.ntnu.edu/itk) - Norwegian University of Science and Technology (Prof. Sebastien Gros)
- [Neurobotics Lab](https://nr.informatik.uni-freiburg.de/welcome) - University of Freiburg (Prof. Joschka Boedeker)  
- [Systems Control and Optimization Laboratory](https://www.syscop.de/) - University of Freiburg (Prof. Moritz Diehl)

[View on GitHub](https://github.com/leap-c/leap-c)

### Documentation

```{eval-rst}
Documentation latest build: |today|
```

```{toctree}
:maxdepth: 1
:caption: Contents

Home<self>
installation
getting_started
define_differentiable_mpc
parameter_management
notebooks
troubleshooting
```

```{toctree}
:maxdepth: 2
:caption: API Reference

api/index
api/generated/leap_c/index
```
