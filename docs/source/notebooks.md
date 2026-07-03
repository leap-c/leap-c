# Running Notebooks

leap-c ships interactive [marimo](https://marimo.io) notebooks (see the `notebooks/` folder) that demonstrate how to formulate and solve optimal control problems with the differentiable acados layer.

Available notebooks:

- `notebooks/minimal_mpc.py`: scalar integrator MPC plus batched warm-start collation.

A guided series builds up from there on small, visual systems:

- `notebooks/01_msd_build_and_solve.py`: register parameters, build and solve a mass-spring-damper OCP, read the plan.
- `notebooks/02_msd_value_policy_maps.py`: batched solves as value-function and policy maps over the state space.
- `notebooks/03_msd_sensitivities.py`: gradients through the solver.
- `notebooks/04_heating_parameter_management.py`: differentiable vs. non-differentiable parameters and stage `splits` on an R1C1 heating model.
- `notebooks/05_heating_forecasts.py`: embedding weather/price forecasts over a receding horizon.
- `notebooks/06_battery_arbitrage.py`: economic MPC for battery arbitrage with signed price sensitivities.

You can run these notebooks either locally (recommended for development) or via a pre-built Docker image (a convenient fallback if you do not want to build acados yourself).

## Local setup

First complete the [native installation](installation.md), then install the notebook and torch extras. Recommended (uv):

```bash
uv pip install -e ".[notebooks,torch]"
```

Alternatively (pip):

```bash
pip install -e ".[notebooks,torch]"
```

Launch the marimo server from the repository root:

```bash
marimo edit notebooks
```

Open the printed URL (default <http://localhost:8080>) in your browser.

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

If you only need a reproducible shell environment (for example to run tests without installing acados), use the `cpu` image:

```bash
docker run -it --rm -v "$(pwd):/workspace" -w /workspace ghcr.io/leap-c/leap-c:cpu
```

## Advanced Docker usage

For local image builds, GPU support, VS Code Dev Containers, HuggingFace Spaces, Dockerfile stage details, CI behavior, and troubleshooting, see the detailed Docker reference:

[`docker/README.md`](https://github.com/leap-c/leap-c/blob/main/docker/README.md)
