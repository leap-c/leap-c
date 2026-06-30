# Docker

Reproducible CPU/GPU containerized development environment for leap-c with pre-built acados.

## Pre-built images

Images are published to GitHub Container Registry on release tags, nightly, and manual workflow dispatch:

```bash
# CPU (x86_64 + ARM64)
docker pull ghcr.io/leap-c/leap-c:cpu

# GPU (x86_64 only)
docker pull ghcr.io/leap-c/leap-c:gpu

# Notebook / marimo (x86_64 + ARM64) — also tagged as :latest
docker pull ghcr.io/leap-c/leap-c:notebook
```

Tags:

| Tag | Description |
|-----|-------------|
| `:latest` | Latest `notebook` target (nightly or release) |
| `:cpu`, `:gpu`, `:notebook` | Latest build of each target |
| `:v2025.0.0` | Release tag — points to `notebook` |
| `:v2025.0.0-cpu`, `:v2025.0.0-gpu`, `:v2025.0.0-notebook` | Release tag per target |

To trigger a manual build: go to **Actions → Docker → Run workflow** on GitHub.

Pull and run — no build required:

```bash
docker run -it --rm -v "$(pwd):/workspace" -w /workspace ghcr.io/leap-c/leap-c:cpu
```

## Building from source

Only needed if you want to modify the Dockerfile or use a local acados fork.

### Prerequisites

Initialize the acados submodule before building:

```bash
git submodule update --init --recursive
```

### CPU

```bash
docker build --target cpu -t leap-c:cpu .
docker run -it --rm -v "$(pwd):/workspace" -w /workspace leap-c:cpu
```

### GPU

```bash
docker build --target gpu -t leap-c:gpu .
docker run -it --rm --gpus all -v "$(pwd):/workspace" -w /workspace leap-c:gpu
```

### Notebook (marimo)

The default build target launches a marimo notebook server on port 7860:

```bash
docker build -t leap-c:notebook .
docker run -it --rm -p 7860:7860 leap-c:notebook
```

Open <http://localhost:7860> in your browser.

#### Persisting notebook changes (tutorials/workshops)

By default, notebook edits are saved inside the container and **lost when it stops**. To persist changes to the host, mount the notebooks directory:

```bash
docker run -it --rm -p 7860:7860 \
  -v "$(pwd)/notebooks:/home/leap/leap-c/notebooks" \
  leap-c:notebook
```

Edits made in the browser are now saved directly to your local `notebooks/` folder.

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

| Stage | Base image | Description |
|-------|-----------|-------------|
| `base` | `ubuntu:24.04` | System deps, uv, non-root user `leap` |
| `acados-builder` | `base` | Builds acados from source, installs Tera renderer |
| `cpu` | `base` | PyTorch CPU, leap-c, acados Python interface |
| `gpu` | `nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04` | PyTorch GPU (CUDA 12.8) |
| `notebook` | `cpu` | marimo notebook server on port 7860 |

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
- **PyTorch** — CPU or GPU depending on target
- **leap-c** — installed in editable mode with `[rendering,notebook]` extras
- **marimo** — reactive Python notebook server
- **uv** — fast Python package manager

## CI

The `.github/workflows/docker.yml` workflow builds and pushes multi-arch images to `ghcr.io/leap-c/leap-c` on:
- Tag releases (`v*`)
- Nightly at 02:00 UTC (catches upstream acados/PyTorch changes)
- Manual dispatch via the Actions UI ("Run workflow")

Images are cached via GitHub Actions cache to speed up rebuilds.

## Troubleshooting

### GPU not detected

Ensure the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) is installed:

```bash
# Ubuntu/Debian
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
