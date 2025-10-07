# Installation

## Linux/MacOS

### Prerequisites

- git
- Python 3.11 or higher

Clone the repository:
```bash
git clone git@github.com:leap-c/leap-c.git
cd leap-c
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

#### PyTorch

Install PyTorch as described on the [PyTorch website](https://pytorch.org/get-started/locally/).

To install CPU-only PyTorch you can use:

``` bash
pip install torch --extra-index-url https://download.pytorch.org/whl/cpu
```

### Install leap-c

To install the package containing minimum dependencies (including acados) in the root directory of the repository, run:

```bash
pip install -e .
```

**Note:** The `acados_template` Python package will be installed automatically as a dependency. If you encounter issues with the acados installation, please refer to the [acados installation documentation](https://docs.acados.org/installation/index.html) for system-specific requirements and troubleshooting.

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