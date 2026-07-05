# syntax=docker/dockerfile:1
# ==============================================================
# leap-c Dockerfile
#
# Multi-stage build using the pre-built acados-ci base image:
#
#   acados-ci → runtime → cpu / notebook
#                       ↘ gpu (local/manual only, COPY --from=acados-ci)
#
# Usage:
#   Notebook:     docker build -t leap-c:notebook .   (default target)
#   CPU dev:      docker build --target cpu -t leap-c:cpu .
#   GPU dev:      docker build --target gpu -t leap-c:gpu .
#
# Prerequisites:
#   No submodule init needed — acados comes from the acados-ci base image.
#
# For HuggingFace Spaces, the default (last) target is used automatically.
# ==============================================================

ARG CUDA_IMAGE=nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04
ARG TORCH_CUDA_INDEX=https://download.pytorch.org/whl/cu128
ARG ACADOS_CI_IMAGE=ghcr.io/leap-c/acados-ci:ubuntu24.04


# ----------------------------------------------------------
# Stage: runtime
# Heavy shared layer built on top of acados-ci.
# acados + system deps already present; we add the leap user,
# Python 3.12 venv, PyTorch CPU, acados_template, leap-c, marimo.
# Both `cpu` and `notebook` branch from this stage.
# ----------------------------------------------------------
FROM ${ACADOS_CI_IMAGE} AS runtime

# Install sudo (not in acados-ci) and create non-root user
RUN apt-get update && apt-get install -y --no-install-recommends sudo && \
    rm -rf /var/lib/apt/lists/* && \
    useradd -m -s /bin/bash leap && \
    echo "leap ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/leap && \
    chmod 0440 /etc/sudoers.d/leap

ENV HOME=/home/leap

USER leap

# Create Python 3.12 virtual environment
RUN uv venv --python 3.12 /home/leap/.venv

ENV VIRTUAL_ENV=/home/leap/.venv
ENV PATH="/home/leap/.venv/bin:${PATH}"

# --- Cache-efficient dependency installation ---
# Copy pyproject.toml first so dependency layers cache independently of source
COPY --chown=leap:leap pyproject.toml /home/leap/leap-c/pyproject.toml
WORKDIR /home/leap/leap-c

# Install acados_template from the acados-ci base image (already at /opt/acados)
RUN uv pip install /opt/acados/interfaces/acados_template

# Install PyTorch CPU (large download, cached independently of leap-c source)
RUN uv pip install torch --index-url https://download.pytorch.org/whl/cpu

# Install marimo (small, but needed for both targets)
RUN uv pip install "marimo>=0.13.0"

# Copy the Python package source needed for the editable install. Notebooks are
# copied later so notebook-only edits do not invalidate dependency layers.
COPY --chown=leap:leap README.md LICENSE /home/leap/leap-c/
COPY --chown=leap:leap leap_c /home/leap/leap-c/leap_c

# Install leap-c with rendering and notebook extras
RUN uv pip install -e ".[rendering,notebook]"

# Verify installation
RUN python -c "import acados_template; import leap_c; print('leap-c ready')"

# Copy notebooks after installation for fast notebook-only rebuilds.
COPY --chown=leap:leap notebooks /home/leap/leap-c/notebooks


# ----------------------------------------------------------
# Stage: cpu
# Thin shell image for local development.
# ----------------------------------------------------------
FROM runtime AS cpu

WORKDIR /workspace
CMD ["/bin/bash"]


# ----------------------------------------------------------
# Stage: gpu
# GPU image with CUDA and PyTorch GPU.
# Local/manual only — not built by CI by default.
# Copies acados artifacts from the published acados-ci image.
# ----------------------------------------------------------
FROM ${CUDA_IMAGE} AS gpu

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    sudo git cmake make build-essential gfortran pkg-config \
    wget curl unzip ca-certificates \
    libopenblas-dev liblapack-dev libblas-dev \
    swig python3 && \
    rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Create non-root user
RUN useradd -m -s /bin/bash leap && \
    echo "leap ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/leap && \
    chmod 0440 /etc/sudoers.d/leap

ENV HOME=/home/leap

# Copy acados build artifacts from the published acados-ci image
COPY --from=${ACADOS_CI_IMAGE} --chown=leap:leap /opt/acados/lib /opt/acados/lib
COPY --from=${ACADOS_CI_IMAGE} --chown=leap:leap /opt/acados/include /opt/acados/include
COPY --from=${ACADOS_CI_IMAGE} --chown=leap:leap /opt/acados/bin /opt/acados/bin

# Acados environment
ENV ACADOS_SOURCE_DIR=/opt/acados
ENV LD_LIBRARY_PATH="/opt/acados/lib"
ENV PATH="/opt/acados/bin:${PATH}"

USER leap

# Install Python 3.12 via uv and create a virtual environment
RUN uv python install 3.12 && \
    uv venv --python 3.12 /home/leap/.venv

ENV VIRTUAL_ENV=/home/leap/.venv
ENV PATH="/home/leap/.venv/bin:${PATH}"

# --- Cache-efficient dependency installation ---
COPY --chown=leap:leap pyproject.toml /home/leap/leap-c/pyproject.toml
WORKDIR /home/leap/leap-c

# Copy acados_template from the acados-ci image
COPY --from=${ACADOS_CI_IMAGE} --chown=leap:leap /opt/acados/interfaces/acados_template \
     /tmp/acados_template
RUN uv pip install /tmp/acados_template && rm -rf /tmp/acados_template

# Install PyTorch GPU (CUDA)
ARG TORCH_CUDA_INDEX
RUN uv pip install torch --index-url ${TORCH_CUDA_INDEX}

# Copy the rest of the leap-c source
COPY --chown=leap:leap . /home/leap/leap-c

# Install leap-c with rendering and notebook extras
RUN uv pip install -e ".[rendering,notebook]"

# Verify installation
RUN python -c "import acados_template; import leap_c; print('leap-c ready')"

WORKDIR /workspace
CMD ["/bin/bash"]


# ----------------------------------------------------------
# Stage: notebook
# Interactive marimo notebook server for tutorials and HuggingFace Spaces.
# This is the default build target — it must remain the LAST stage so
# that `docker build` (no --target) and HuggingFace Spaces select it.
# ----------------------------------------------------------
FROM runtime AS notebook

WORKDIR /home/leap/leap-c/notebooks

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=3s \
    CMD curl -f http://localhost:7860/health || exit 1

CMD ["marimo", "edit", "--host", "0.0.0.0", "-p", "7860", "--no-token"]
