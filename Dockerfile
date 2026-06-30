# syntax=docker/dockerfile:1
# ==============================================================
# leap-c Dockerfile
#
# Multi-stage build providing CPU, GPU, and notebook (marimo) targets.
#
# Usage:
#   CPU dev:      docker build --target cpu -t leap-c:cpu .
#   GPU dev:      docker build --target gpu -t leap-c:gpu .
#   Notebook:     docker build -t leap-c:notebook .   (default target)
#
# Prerequisites:
#   git submodule update --init --recursive
#
# For HuggingFace Spaces, the default (last) target is used automatically.
# ==============================================================


# ----------------------------------------------------------
# Stage: base
# Common system dependencies, uv, and non-root user.
# ----------------------------------------------------------
FROM ubuntu:24.04 AS base
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    sudo git cmake make build-essential gfortran pkg-config \
    wget curl unzip ca-certificates \
    libopenblas-dev liblapack-dev libblas-dev \
    swig python3 && \
    rm -rf /var/lib/apt/lists/*

# Install uv for fast Python package management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Create non-root user
RUN useradd -m -s /bin/bash leap && \
    echo "leap ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/leap && \
    chmod 0440 /etc/sudoers.d/leap

ENV HOME=/home/leap


# ----------------------------------------------------------
# Stage: acados-builder
# Builds acados from source and installs the Tera renderer.
# ----------------------------------------------------------
FROM base AS acados-builder

COPY --chown=leap:leap external/acados /opt/acados

USER leap
WORKDIR /opt/acados

# Build acados (matching CI flags from .github/actions/build-acados)
RUN cmake -S . -B build \
        -DACADOS_WITH_OPENMP=ON \
        -DACADOS_NUM_THREADS=1 && \
    cmake --build build --target install -- -j"$(nproc)"

# Install Tera renderer (required by acados_template for C code generation)
# On x86_64 we use the pre-built binary; on ARM64 (Apple Silicon) we build from source
ARG TARGETARCH
RUN mkdir -p /opt/acados/bin && \
    if [ "$TARGETARCH" = "amd64" ]; then \
        wget -q -O /opt/acados/bin/t_renderer \
            "https://github.com/acados/tera_renderer/releases/download/v0.2.0/t_renderer-v0.2.0-linux-amd64" && \
        chmod +x /opt/acados/bin/t_renderer; \
    else \
        curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y && \
        . "$HOME/.cargo/env" && \
        git clone --depth 1 --branch v0.2.0 \
            https://github.com/acados/tera_renderer.git /tmp/tera_renderer && \
        cd /tmp/tera_renderer && \
        cargo build --release --bin t_renderer && \
        cp /tmp/tera_renderer/target/release/t_renderer /opt/acados/bin/t_renderer && \
        rm -rf /tmp/tera_renderer "$HOME/.cargo"; \
    fi


# ----------------------------------------------------------
# Stage: cpu
# Final CPU image with PyTorch CPU and leap-c.
# ----------------------------------------------------------
FROM base AS cpu

# Copy acados build artifacts (libraries, headers, Tera renderer)
COPY --from=acados-builder --chown=leap:leap /opt/acados/lib /opt/acados/lib
COPY --from=acados-builder --chown=leap:leap /opt/acados/include /opt/acados/include
COPY --from=acados-builder --chown=leap:leap /opt/acados/bin /opt/acados/bin

# Acados environment
ENV ACADOS_SOURCE_DIR=/opt/acados
ENV LD_LIBRARY_PATH="/opt/acados/lib:${LD_LIBRARY_PATH}"
ENV PATH="/opt/acados/bin:${PATH}"

USER leap

# Install Python 3.11 via uv and create a virtual environment
RUN uv python install 3.11 && \
    uv venv --python 3.11 /home/leap/.venv

ENV VIRTUAL_ENV=/home/leap/.venv
ENV PATH="/home/leap/.venv/bin:${PATH}"

# --- Cache-efficient dependency installation ---
# Copy acados_template first (small, rarely changes) and install it.
# This pulls in numpy, scipy, etc. from PyPI.
COPY --chown=leap:leap external/acados/interfaces/acados_template \
     /home/leap/leap-c/external/acados/interfaces/acados_template
RUN uv pip install /home/leap/leap-c/external/acados/interfaces/acados_template

# Install PyTorch CPU (large download, cached independently of leap-c source)
RUN uv pip install torch --index-url https://download.pytorch.org/whl/cpu

# Copy the rest of the leap-c source
COPY --chown=leap:leap . /home/leap/leap-c
WORKDIR /home/leap/leap-c

# Install leap-c with rendering and notebook extras
RUN uv pip install -e ".[rendering,notebook]"

# Verify installation
RUN python -c "import acados_template; import leap_c; print('leap-c ready')"

WORKDIR /workspace
CMD ["/bin/bash"]


# ----------------------------------------------------------
# Stage: gpu
# Final GPU image with CUDA and PyTorch GPU.
# ----------------------------------------------------------
FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04 AS gpu

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

# Copy acados build artifacts
COPY --from=acados-builder --chown=leap:leap /opt/acados/lib /opt/acados/lib
COPY --from=acados-builder --chown=leap:leap /opt/acados/include /opt/acados/include
COPY --from=acados-builder --chown=leap:leap /opt/acados/bin /opt/acados/bin

# Acados environment
ENV ACADOS_SOURCE_DIR=/opt/acados
ENV LD_LIBRARY_PATH="/opt/acados/lib:${LD_LIBRARY_PATH}"
ENV PATH="/opt/acados/bin:${PATH}"

USER leap

# Install Python 3.11 via uv and create a virtual environment
RUN uv python install 3.11 && \
    uv venv --python 3.11 /home/leap/.venv

ENV VIRTUAL_ENV=/home/leap/.venv
ENV PATH="/home/leap/.venv/bin:${PATH}"

# --- Cache-efficient dependency installation ---
COPY --chown=leap:leap external/acados/interfaces/acados_template \
     /home/leap/leap-c/external/acados/interfaces/acados_template
RUN uv pip install /home/leap/leap-c/external/acados/interfaces/acados_template

# Install PyTorch GPU (CUDA 12.8)
RUN uv pip install torch --index-url https://download.pytorch.org/whl/cu128

# Copy the rest of the leap-c source
COPY --chown=leap:leap . /home/leap/leap-c
WORKDIR /home/leap/leap-c

# Install leap-c with rendering and notebook extras
RUN uv pip install -e ".[rendering,notebook]"

# Verify installation
RUN python -c "import acados_template; import leap_c; print('leap-c ready')"

WORKDIR /workspace
CMD ["/bin/bash"]


# ----------------------------------------------------------
# Stage: notebook
# Interactive marimo notebook server for tutorials and HuggingFace Spaces.
# Extends the CPU image — the default build target.
# ----------------------------------------------------------
FROM cpu AS notebook

USER leap
WORKDIR /home/leap/leap-c/notebooks

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=3s \
    CMD curl -f http://localhost:7860/health || exit 1

CMD ["marimo", "edit", "--host", "0.0.0.0", "-p", "7860", "--no-token"]
