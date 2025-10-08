# Installation

## Linux/MacOS

### Prerequisites

- git
- Python 3.11 or higher
- [acados dependencies](https://docs.acados.org/installation/index.html)

Clone the repository and recursively update submodules:
```bash
git clone https://github.com/leap-c/leap-c.git
cd leap-c
git submodule update --init --recursive
```

### Python

We work with Python 3.11. If it is not already installed on your system, you can obtain it using [deadsnakes](https://launchpad.net/~deadsnakes/+archive/ubuntu/ppa):
```bash
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt install python3.11
```

A virtual environment is recommended. For example, to create a virtual environment called `.venv`
and activate it, run:

```bash
pip3 install virtualenv
virtualenv --python=/usr/bin/python3.11 .venv
source .venv/bin/activate
```

The following steps assume that the virtual environment is activated.

#### acados

Then change into the acados directory

```bash
cd external/acados
```

and build it as described in the [acados documentation](https://docs.acados.org/installation/index.html).
When running the `cmake` command, include the options `-DACADOS_WITH_OPENMP=ON -DACADOS_NUM_THREADS=1`.
Afterwards, install the [python interface](https://docs.acados.org/python_interface/index.html) of acados.

#### PyTorch

Install PyTorch as described on the [PyTorch website](https://pytorch.org/get-started/locally/).

To install CPU-only PyTorch you can use:

``` bash
pip install torch --extra-index-url https://download.pytorch.org/whl/cpu
```

### Install leap-c

To install the package containing minimum dependencies in the root directory of the repository, run:

```bash
pip install -e .
```

For also enabling rendering in some of our examples use:

```bash
pip install -e ".[rendering]"
```

For development, you might want to install all additional dependencies:

```bash
pip install -e ".[dev]"
```

See the [pyproject.toml](https://github.com/leap-c/leap-c/blob/main/pyproject.toml) for more information on package configurations.

### Troubleshooting
In the [troubleshooting tab](https://leap-c.github.io/leap-c/troubleshooting.html),
we highlight how to fix common problems arising while using leap-c with VS Code.

## Windows
We recommend to use [WSL (Windows Subsystem for Linux)](https://ubuntu.com/desktop/wsl) and then following the guide above.
You can then conveniently program on your WSL, e.g.,
by using VS Code on Windows together with the "Remote Development" extension pack.

Note the [installation instructions for acados regarding WSL](https://docs.acados.org/installation/index.html#windows-10-wsl).
Also note the troubleshooting section for `plt.show()` in WSL below.

## Testing

To run the tests, use:

```bash
pytest tests -vv -s
```

## Linting and Formatting

Only relevant if you want to contribute to the repository.
For keeping our code style and our diffs consistent we use the [Ruff](https://docs.astral.sh/ruff/) linter and formatter.

To make this as easy as possible we also provide a [pre-commit](https://pre-commit.com/) config for running the linter and formatter automatically at every commit. For enabling pre-commit follow these steps:

1. Install pre-commit using pip (already done if you installed the additional "[dev]" dependencies of leap-c).
```bash
pip install pre-commit
```

2. In the leap-c root directory run
```bash
pre-commit install
```

Done! Now every commit will automatically be linted and formatted.