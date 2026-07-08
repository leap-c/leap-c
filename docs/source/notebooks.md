# Running Notebooks

leap-c ships interactive [marimo](https://marimo.io) notebooks (see the `notebooks/` folder) that demonstrate how to formulate and solve optimal control problems with the differentiable acados layer.

**`notebooks/getting_started/`** is a sequential course on two small systems (a mass-spring-damper and an R1C1 house-heating problem):

- `01_intro_diff_mpc.py`: differentiable MPC end-to-end in five minutes.
- `02_from_acados_to_diff_mpc.py`: converting an existing plain `AcadosOcp` into a differentiable one.
- `03_gradients_through_the_solver.py`: V vs. Q, the three autograd routes, and where gradients die.
- `04_parameter_management.py`: differentiable vs. non-differentiable parameters, stage `splits`, override shapes, guard rails.
- `05_batched_solves_and_forecasts.py`: the batch dimension; weather/price forecasts through both parameter interfaces.
- `06_planner_interface.py`: an observation-to-action planner with a time-varying slacked comfort band, closed-loop against a mismatched house.
- `07_imitation_learning.py`: behavior cloning through the solver.
- `08_rl_on_closed_loop_cost.py`: tuning the planner by backpropagating through closed-loop rollouts.

See `notebooks/README.md` for the full API coverage map and reading order.

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
