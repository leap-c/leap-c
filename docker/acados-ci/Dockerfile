# syntax=docker/dockerfile:1
# Dedicated CI base image for leap-c downstream repositories.
#
# The image contains system dependencies, a prebuilt acados installation, uv,
# and Python 3.11/3.12/3.13. It intentionally does not install leap-c itself,
# so CI workflows can install and test exact repository refs.
#
# Multi-arch: linux/amd64 + linux/arm64.
# On amd64 the Tera renderer binary is downloaded; on arm64 it is built
# from source via Rust (adds ~1-2min to the build).

FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    cmake \
    curl \
    gfortran \
    git \
    libblas-dev \
    liblapack-dev \
    libopenblas-dev \
    make \
    pkg-config \
    python3 \
    swig \
    unzip \
    wget \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

COPY external/acados /opt/acados

WORKDIR /opt/acados

RUN cmake -S . -B build \
      -DACADOS_WITH_OPENMP=ON \
      -DACADOS_NUM_THREADS=1 \
    && cmake --build build --target install -- -j"$(nproc)" \
    && rm -rf build

# Install Tera renderer (required by acados_template for C code generation).
# On amd64: download pre-built binary. On arm64: build from source with Rust.
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

ENV ACADOS_SOURCE_DIR=/opt/acados
ENV ACADOS_INSTALL_DIR=/opt/acados
ENV LD_LIBRARY_PATH=/opt/acados/lib
ENV PATH=/opt/acados/bin:/root/.local/bin:${PATH}

RUN uv python install 3.11 3.12 3.13

WORKDIR /workspace

CMD ["bash"]
