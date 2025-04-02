# Installation

## Linux/MacOS

### Prerequisites

- git
- Python 3.11 or higher
- [acados dependencies](https://docs.acados.org/installation/index.html)
- [CasADi nightly-se2 ](https://github.com/casadi/casadi/releases/tag/nightly-se2)

```bash
    git clone git@github.com:leap-c/leap-c.git leap_c
    cd leap_c
    git submodule update --init --recursive
```

### Python

We work with Python 3.11. A virtual environment is recommended. For example, to create a virtual environment called `.venv`
and activate it, run:

```bash
    pip3 install virtualenv
    virtualenv --python=/usr/bin/python3.11 .venv
    source .venv/bin/activate
```

The following steps assume that the virtual environment is activated.

#### CasADi nightly

We need the [nightly-se2 release of CasADi](https://github.com/casadi/casadi/releases/tag/nightly-se2) to work with the gradient functionality of acados. Install the correct wheel for your system, for example for Linux x86_64:

```bash
curl -L --remote-name https://github.com/casadi/casadi/releases/download/nightly-se2/casadi-3.6.6.dev+se2-cp311-none-manylinux2014_x86_64.whl
pip install casadi-3.6.6.dev+se2-cp311-none-manylinux2014_x86_64.whl
rm casadi-3.6.6.dev+se2-cp311-none-manylinux2014_x86_64.whl
```

#### acados

Then change into the acados directory 

```bash
cd external/acados
```

and build it as described in the [acados documentation](https://docs.acados.org/installation/index.html). When running the
`cmake` command, make sure to include the options `-DACADOS_WITH_OPENMP=ON` and `-DACADOS_PYTHON=ON`, `-DACADOS_NUM_THREADS=1`.

#### PyTorch

Install PyTorch as described on the [PyTorch website](https://pytorch.org/get-started/locally/).

### Install leap-c

To install the package in the root directory of the repository, run:

```bash
    pip install -e .
```

For development, you might want to install additional dependencies::

```bash
    pip install -e .[dev]
```

See the [pyproject.toml](https://github.com/leap-c/leap-c/blob/main/pyproject.toml) for more information on the installated packages.

## Testing

To run the tests, use:

```bash
    pytest
```
