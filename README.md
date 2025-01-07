# leap-c (Learning Predictive Control)

## Introduction

leap-c provides tools for learning optimal control policies using learning
methodologies Imitation learning (IL) and Reinforcement Learning (RL) to enhance
Model Predictive Control (MPC) policies. It is built on top of
[acados](https://docs.acados.org/index.html) and [casadi](https://web.casadi.org/).

## Installation

### Dependencies

leap-c requires the following dependencies that need to be installed separately:

- [casadi](https://web.casadi.org/) for symbolic computations
- [acados](https://docs.acados.org/index.html) for generating OCP solvers

### Clone with recursive submodules

leap-c uses acados as a submodule. To clone the repository with the submodule, use the following command:

``` bash
    git clone --recurse-submodules git@github.com:leap-c/leap-c.git
```

### Installation steps

Create a python virtual environment using python version 3.11 and activate it:

``` bash
    sudo pip3 install virtualenv
    cd <PATH_TO_VENV_DIRECTORY>
    virtualenv .venv --python=/usr/bin/python3.11
    source .venv/bin/activate
```

Follow the instructions to build acados [here](https://docs.acados.org/installation/),
and also follow the instructions there to install acados' python interface.

Install the required casadi version with the installation script:

``` bash
    cd <PATH_TO_LEAP_C_DIRECTORY>
    ./install_new_casadi_py311_x86_64.sh
```

Install the minimum:

``` bash
    pip install -e .
```

or install with optional dependencies (e.g. for testing):

``` bash
    pip install -e .[test]
```

Install the desired pytorch version, see [here](https://pytorch.org/get-started/locally/).
E.g., to install cpu-only pytorch you can use

``` bash
    pip install torch --extra-index-url https://download.pytorch.org/whl/cpu
```

## Usage

Check out tests for the point mass system


``` bash
    python tests/leap_c/test_point_mass.py
```

## Questions/ Contact

Write to dirk.p.reinhardt@ntnu.no
