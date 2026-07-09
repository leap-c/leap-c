# Installation

## Linux/MacOS

### Prerequisites

- git
- Python 3.11 or higher
- [acados dependencies](https://docs.acados.org/installation/index.html)

We recommend the [uv](https://docs.astral.sh/uv/) package manager. Install it with

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

(see the [uv installation guide](https://docs.astral.sh/uv/getting-started/installation/)
for other options). If you prefer, the classic `pip` + `venv` workflow is also fully
supported — the alternative commands are shown alongside each step below.

Clone the repository and recursively update submodules:

```bash
git clone https://github.com/leap-c/leap-c.git
cd leap-c
git submodule update --init --recursive
```

### Python

Create a virtual environment and activate it. Assuming a required Python version of 3.11 or newer, you can use the following commands.

Recommended (uv), which also fetches a matching Python if it is not installed:

```bash
uv sync
```

Alternatively (pip/venv):

```bash
python3 -m venv .venv
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

Then, export the following environment variables:

```bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:"$PWD/external/acados/lib"
export ACADOS_SOURCE_DIR="$PWD/external/acados"
```

## Install leap-c

> [!NOTE] If you're planning to use `pip`, we recommend you to manually create a virtual environment of your choice, with Python version of 3.11 or newer.

```bash
uv sync --extra torch        # minimal install with torch backend
```

Alternatively (pip/venv):

```bash
pip install -e .[torch]
```

See the [pyproject.toml](https://github.com/leap-c/leap-c/blob/main/pyproject.toml) for more information on package configurations.

## Backend

Main leap-c package does **not** auto-install a computational backend. Select one via its optional extra:

| Extra | Framework | Required for |
|-------|-----------|-------------|
| `torch` | PyTorch | `AcadosDiffMpcTorch` |
| `jax` | JAX | `AcadosDiffMpcJax` |

### PyTorch

PyTorch wheels default to GPU (CUDA).

**uv:**

```bash
uv sync --extra torch                                           # GPU (default)
uv sync --extra torch --index-url https://download.pytorch.org/whl/cpu  # CPU-only
```

**pip** (requires acados_template installed first — see [acados](#acados) above):

```bash
pip install -e ".[torch]"                                       # GPU (default)
pip install -e ".[torch]" --extra-index-url https://download.pytorch.org/whl/cpu  # CPU-only
```

The CPU choice above is a flag passed at install time, not a declared dependency, so a later
plain `uv sync`/`pip install -e .` (e.g. to pick up an unrelated change) re-resolves torch from
the default index and silently switches you back to the GPU build. Re-pass the CPU flag whenever
you resync after choosing CPU-only.

### JAX

JAX wheels default to **CPU**. For CUDA 12, use the `jax-cuda12` extra so the GPU build is
part of the locked, declared dependencies — not a manual post-install step that a later plain
`uv sync`/`pip install -e .` would silently revert back to CPU.

**uv** (CPU):

```bash
uv sync --extra jax
```

**uv** (GPU, CUDA 12):

```bash
uv sync --extra jax-cuda12
```

For other CUDA setups (local CUDA, ROCm, TPU), see the
[JAX installation guide](https://jax.readthedocs.io/en/latest/installation.html) and add the
matching extra (e.g. `jax[cuda12-local]`) to `pyproject.toml`, or install it manually with
`uv pip install` — keeping in mind a manual install like that does not survive a later plain
`uv sync`.

**pip** (CPU):

```bash
pip install -e ".[jax]"
```

**pip** (GPU, CUDA 12):

```bash
pip install -e ".[jax-cuda12]"
```

If no backend is installed and you try to import a backend-specific module,
leap-c raises a clear error telling you which backends are supported.

### Troubleshooting

In the [troubleshooting tab](https://leap-c.github.io/leap-c/troubleshooting.html),
we highlight how to fix common problems arising while using leap-c with VS Code.

## Docker (optional fallback)

Docker is an optional fallback for users who want a reproducible environment with
pre-built acados without compiling it locally. For regular development, we recommend
the native installation above.

Pre-built images are published to GitHub Container Registry. To run the interactive
marimo notebook server (no build required):

```bash
docker run -it --rm -p 7860:7860 ghcr.io/leap-c/leap-c:notebook
```

Open <http://localhost:7860> in your browser. For persisting notebook edits, using
the CPU shell image, and the full local-build/GPU/Dev Container instructions, see
[Running Notebooks](notebooks.md).

For advanced Docker configuration, Dockerfile stages, CI, and troubleshooting, see
[`docker/README.md`](https://github.com/leap-c/leap-c/blob/main/docker/README.md).

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

1. Install pre-commit (already done if you installed the additional "[dev]" dependencies of leap-c). Recommended (uv):

```bash
uv pip install pre-commit
```

Alternatively (pip):

```bash
pip install pre-commit
```

2. In the leap-c root directory run

```bash
pre-commit install
```

Done! Now every commit will automatically be linted and formatted.
