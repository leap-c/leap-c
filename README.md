[![Source Code License](https://img.shields.io/badge/license-BSD-blueviolet?style=for-the-badge)](https://github.com/leap-c/leap-c/blob/main/LICENSE)
![Python 3.11](https://img.shields.io/badge/python->=3.11-green.svg?style=for-the-badge)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json&style=for-the-badge)](https://docs.astral.sh/ruff/)
[![codecov](https://img.shields.io/codecov/c/github/leap-c/leap-c?token=TQE7RZ1O7M&style=for-the-badge)](https://codecov.io/github/leap-c/leap-c)

# leap-c (Learning Predictive Control)

## Introduction

`leap-c` provides tools for learning optimal control policies using Imitation learning (IL) and Reinforcement Learning (RL) to enhance Model Predictive Control (MPC) algorithms. It is built on top of [CasADi](https://web.casadi.org/), [acados](https://docs.acados.org/index.html) and [PyTorch](https://pytorch.org/).

## Installation

`leap-c` can be set up by following the [installation guide](https://leap-c.github.io/leap-c/installation.html).

## Usage

Please see the [Getting started section](https://leap-c.github.io/leap-c/getting_started/index.html).

## Questions?

Open a new thread or browse the existing ones on the [GitHub discussions](https://github.com/leap-c/leap-c/discussions) page.

## Related Projects

The following projects follow similar ideas and might be intersting:

- [mpc.pytorch](https://github.com/locuslab/mpc.pytorch): Early work on embedding MPC in PyTorch for end-to-end learning, with a
    more restricted class of MPC problems
- [mpcrl](https://github.com/FilippoAiraldi/mpc-reinforcement-learning): A simpler codebase for using RL with MPC as function approximator
- [Neuromancer](https://github.com/pnnl/neuromancer): A differentiable programming library that allows to include parametric optimization layers
    (including MPC) in PyTorch computational graphs
- [ntnu-itk-autonomous-ship-lab/rlmpc](https://github.com/ntnu-itk-autonomous-ship-lab/rlmpc): A codebase tailored for marine vessel control using RL and MPC
