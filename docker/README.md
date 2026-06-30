# Docker

Detailed reference for the leap-c Docker images: pre-built images, local builds, GPU, Dev Containers, HuggingFace Spaces, Dockerfile stages, CI, and troubleshooting.

For running notebooks, see [Running Notebooks](https://leap-c.github.io/leap-c/notebooks.html) in the user documentation. This guide covers the advanced Docker internals below.

## Pre-built images

Images are published to GitHub Container Registry on release tags, nightly, and manual workflow dispatch:

```bash
# Notebook / marimo (x86_64 + ARM64) — also tagged as :latest
docker pull ghcr.io/leap-c/leap-c:notebook

# CPU shell (x86_64 + ARM64)
docker pull ghcr.io/leap-c/leap-c:cpu
```

The `:notebook` and `:cpu` images are multi-arch (linux/amd64 + linux/arm64). Docker automatically pulls the right image for your machine, including Apple Silicon Macs. No separate Mac image is needed.

Tags:

| Tag | Description |
|-----|-------------|
| `:latest` | Latest `notebook` target (nightly or release) |
| `:notebook` | Latest notebook build |
| `:cpu` | Latest CPU build (release or manual) |
| `:vX.Y.Z` | Release tag — points to `notebook` |
| `:vX.Y.Z-cpu`, `:vX.Y.Z-notebook` | Release tag per target |

To trigger a manual build: go to **Actions → Docker → Run workflow** on GitHub.

Pull and run — no build required:

```bash
docker run -it --rm -p 7860:7860 ghcr.io/leap-c/leap-c:notebook
```

## Quick start (local build)

### Prerequisites

Initialize the acados submodule before building:

```bash
git submodule update --init --recursive
```

### Notebook (marimo) — default target

```bash
docker build --target notebook -t leap-c:notebook .
docker run -it --rm -p 7860:7860 leap-c:notebook
```

`notebook` is the last (default) stage, so `--target notebook` is optional —
`docker build -t leap-c:notebook .` builds the same image. Passing it
explicitly keeps the build correct even if the Dockerfile stage order changes.

Open <http://localhost:7860> in your browser.

### CPU

```bash
docker build --target cpu -t leap-c:cpu .
docker run -it --rm -v "$(pwd):/workspace" -w /workspace leap-c:cpu
```

### GPU (local only)

GPU support requires an NVIDIA GPU, NVIDIA drivers, and the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html). Not available on macOS.

```bash
docker build --target gpu -t leap-c:gpu .
docker run -it --rm --gpus all -v "$(pwd):/workspace" -w /workspace leap-c:gpu
```

You can override the CUDA base image and PyTorch index with build args:

```bash
docker build --target gpu \
  --build-arg CUDA_IMAGE=nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04 \
  --build-arg TORCH_CUDA_INDEX=https://download.pytorch.org/whl/cu128 \
  -t leap-c:gpu .
```

### Persisting notebook changes (tutorials/workshops)

By default, notebook edits are saved inside the container and **lost when it stops**. To persist changes to the host, mount the notebooks directory:

```bash
docker run -it --rm -p 7860:7860 \
  -v "$(pwd)/notebooks:/home/leap/leap-c/notebooks" \
  leap-c:notebook
```

Edits made in the browser are saved directly to your local `notebooks/` folder.

## VS Code Dev Container

1. Install the [Dev Containers](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers) extension.
2. Open the leap-c repository in VS Code.
3. Press `Ctrl+Shift+P` → **Dev Containers: Reopen in Container**.

The dev container uses the `cpu` target by default. For GPU, change `"target": "cpu"` to `"target": "gpu"` in `.devcontainer/devcontainer.json`.

## HuggingFace Spaces

### Fast path (recommended)

Use the pre-built image so HuggingFace doesn't need to compile acados from scratch:

1. Create a new [HuggingFace Space](https://huggingface.co/new-space) with SDK set to **Docker**.
2. Add a `Dockerfile` to the Space repo with a single line:

```dockerfile
FROM ghcr.io/leap-c/leap-c:notebook
```

3. Add the following to the Space's `README.md` frontmatter:

```yaml
---
sdk: docker
app_port: 7860
---
```

HuggingFace pulls the pre-built image — startup takes seconds, not minutes.

### Build from source

Alternatively, push the full leap-c repository (including the initialized acados submodule) to the Space and HuggingFace will build the `notebook` target automatically.

## Dockerfile stages

```
base → acados-builder → runtime → cpu
                                   → notebook (default)
                          gpu (local/manual only)
```

| Stage | Base image | Description |
|-------|-----------|-------------|
| `base` | `ubuntu:24.04` | System deps, uv, non-root user `leap` |
| `acados-builder` | `base` | Builds acados from source, installs Tera renderer |
| `runtime` | `base` | Shared heavy layer: acados + Python 3.12 + PyTorch CPU + leap-c + marimo |
| `cpu` | `runtime` | Thin shell image for local development |
| `notebook` | `runtime` | marimo notebook server on port 7860 |
| `gpu` | `nvidia/cuda` | GPU image with PyTorch CUDA (local/manual only) |

## Environment variables

The following are set in all final images:

| Variable | Value |
|----------|-------|
| `ACADOS_SOURCE_DIR` | `/opt/acados` |
| `LD_LIBRARY_PATH` | includes `/opt/acados/lib` |
| `PATH` | includes `/opt/acados/bin` (Tera renderer) and venv |
| `VIRTUAL_ENV` | `/home/leap/.venv` |

## What's inside

- **acados** — pre-built from source with `-DACADOS_WITH_OPENMP=ON -DACADOS_NUM_THREADS=1`
- **Tera renderer** — pre-built binary on x86_64, built from source on ARM64 (Apple Silicon); auto-detected via `TARGETARCH`
- **acados_template** — Python interface to acados
- **PyTorch** — CPU (default images) or GPU (gpu target)
- **leap-c** — installed in editable mode with `[rendering,notebook]` extras
- **marimo** — reactive Python notebook server
- **uv** — fast Python package manager

## CI

The `.github/workflows/docker.yml` workflow builds and pushes images to `ghcr.io/leap-c/leap-c` on:
- Tag releases (`v*`) — builds `notebook` and `cpu`
- Manual dispatch — choose `notebook`, `cpu`, or `all`

The workflow builds `linux/amd64` and `linux/arm64` images in parallel on native GitHub runners, then creates a multi-arch manifest tag (for example `:notebook`). Docker automatically pulls the correct architecture for each user.

Each target/architecture pair uses its own registry cache (for example `buildcache-notebook-amd64`), so rebuilds can reuse expensive layers such as acados, PyTorch, and Python dependencies.

Notebook files are copied after the Python package installation layer. This means changes under `notebooks/` do not force acados, PyTorch, or leap-c dependencies to reinstall during Docker rebuilds.

## Troubleshooting

### GPU not detected

GPU Docker is **local/manual only** and requires Linux or WSL2 with an NVIDIA GPU.

Ensure the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) is installed:

```bash
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

Verify:

```bash
docker run --rm --gpus all nvidia/cuda:12.8.1-base-ubuntu24.04 nvidia-smi
```

### acados submodule not initialized

If the build fails at the acados compilation step:

```bash
git submodule update --init --recursive
```

### Library not found at runtime

If you get `libacados_cored.so not found` when running leap-c:

```bash
export LD_LIBRARY_PATH="/opt/acados/lib:$LD_LIBRARY_PATH"
```

This is set automatically in the Docker images but may need to be set manually if you copy binaries out of the container.

### ARM64 build is slow

Building `t_renderer` from source on ARM64 (Apple Silicon) adds ~1-2 minutes. This only happens on first build — subsequent builds use Docker layer cache. Pre-built ARM64 images are also available via `docker pull ghcr.io/leap-c/leap-c:notebook`.
