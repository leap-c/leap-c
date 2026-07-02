# Running Notebooks

leap-c ships interactive [marimo](https://marimo.io) notebooks (see the `notebooks/` folder) that demonstrate how to formulate and solve optimal control problems with the differentiable acados layer.

You can run these notebooks either locally (recommended for development) or via a pre-built Docker image (a convenient fallback if you do not want to build acados yourself).

## Local setup

First complete the [native installation](installation.md), then install the notebook extras. Recommended (uv):

```bash
uv pip install -e ".[notebook]"
```

Alternatively (pip):

```bash
pip install -e ".[notebook]"
```

Launch the marimo server from the repository root:

```bash
marimo edit notebooks
```

Open the printed URL (default <http://localhost:8080>) in your browser. The example notebook `notebooks/intro.py` solves a CartPole MPC problem end-to-end.

## Docker fallback

If you do not want to build acados locally, you can use the pre-built notebook image published to GitHub Container Registry. No build step is required:

```bash
docker run -it --rm -p 7860:7860 ghcr.io/leap-c/leap-c:notebook
```

Open <http://localhost:7860> in your browser. The image is multi-arch (`linux/amd64` + `linux/arm64`), so it works on Apple Silicon Macs as well as x86_64 machines.

### Persisting notebook changes

By default, notebook edits are saved inside the container and **lost when it stops**. To persist changes to the host, mount the `notebooks/` directory:

```bash
docker run -it --rm -p 7860:7860 \
  -v "$(pwd)/notebooks:/home/leap/leap-c/notebooks" \
  ghcr.io/leap-c/leap-c:notebook
```

Edits made in the browser are then saved directly to your local `notebooks/` folder.

### CPU shell image

If you only need a reproducible shell environment (for example to run scripts or tests without installing acados), use the `cpu` image:

```bash
docker run -it --rm -v "$(pwd):/workspace" -w /workspace ghcr.io/leap-c/leap-c:cpu
```

## Advanced Docker usage

For local image builds, GPU support, VS Code Dev Containers, HuggingFace Spaces, Dockerfile stage details, CI behavior, and troubleshooting, see the detailed Docker reference:

[`docker/README.md`](https://github.com/leap-c/leap-c/blob/main/docker/README.md)
