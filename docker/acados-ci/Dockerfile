# syntax=docker/dockerfile:1
# Dedicated CI base image for leap-c downstream repositories.
#
# The image contains system dependencies, a prebuilt acados installation, uv,
# and Python 3.11/3.12/3.13. It intentionally does not install leap-c itself,
# so CI workflows can install and test exact repository refs.

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
    && .github/linux/install_tera.sh \
    && rm -rf build

ENV ACADOS_SOURCE_DIR=/opt/acados
ENV ACADOS_INSTALL_DIR=/opt/acados
ENV LD_LIBRARY_PATH=/opt/acados/lib
ENV PATH=/opt/acados/bin:/root/.local/bin:${PATH}

RUN uv python install 3.11 3.12 3.13

WORKDIR /workspace

CMD ["bash"]
