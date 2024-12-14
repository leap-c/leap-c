# SEAL

## Super Efficient acados Learning

A framework for integrating optimal control solvers into deep learning pipelines.

### Key Features

- Combines model predictive control with reinforcement learning and imitation learning
- Built on top of community-standard tools [acados](https://docs.acados.org/) and [casadi](https://web.casadi.org/)
- Seamless integration with [PyTorch](https://pytorch.org/) for deep learning. The ocp solver is wrapped into a PyTorch module and can be differentiated end-to-end.

### Development

SEAL is developed through collaboration between:

- [Department of Engineering Cybernetics](https://www.ntnu.edu/itk) - Norwegian University of Science and Technology (Prof. Sebastien Gros)
- [Neurobotics Lab](https://nr.informatik.uni-freiburg.de/welcome) - University of Freiburg (Prof. Joschka Boedeker)  
- [Systems Control and Optimization Laboratory](https://www.syscop.de/) - University of Freiburg (Prof. Moritz Diehl)

[View on GitHub](https://github.com/JasperHoffmann/seal)

### Documentation

```{eval-rst}
Documentation latest build: |today|
```

```{toctree}
:maxdepth: 1
:caption: Contents

Home<self>
installation
getting_started/index
api
```