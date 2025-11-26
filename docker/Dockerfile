# ======================
# CPU IMAGE
# ======================
FROM ubuntu:24.04 AS cpu
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    sudo git cmake make build-essential gfortran pkg-config \
    wget curl unzip ca-certificates software-properties-common \
    python3-pip python3-venv python3-dev \
    libopenblas-dev liblapack-dev libblas-dev \
    swig fontconfig vim nano less && \
    rm -rf /var/lib/apt/lists/*

RUN useradd -m -s /bin/bash syscop && usermod -aG sudo syscop && \
    echo "syscop ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/90-syscop && chmod 0440 /etc/sudoers.d/90-syscop
ENV PIP_BREAK_SYSTEM_PACKAGES=1
USER syscop
ENV HOME=/home/syscop
WORKDIR /home/syscop
RUN python3 -m pip install --upgrade pip

# Copy leap-c repo to build acados
COPY --chown=syscop:syscop . /home/syscop/leap-c

# Build acados
WORKDIR /home/syscop/leap-c/external/acados
RUN mkdir -p build && cd build && \
    cmake .. -DACADOS_WITH_OPENMP=ON -DACADOS_WITH_QPOASES=ON && \
    make -j"$(nproc)"

# Install acados Python interface
WORKDIR /home/syscop/leap-c/external/acados/interfaces/acados_template
RUN pip install .

# Install PyTorch CPU
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision torchaudio

# Install leap-c in editable mode
WORKDIR /home/syscop/leap-c
RUN pip install -e .[rendering]

# Set up environment for pre-built acados
ENV ACADOS_SOURCE_DIR="/home/syscop/leap-c/external/acados"
ENV LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/home/syscop/leap-c/external/acados/lib"
ENV LIBGL_ALWAYS_SOFTWARE=1

WORKDIR /syscop_ws
CMD ["/bin/bash"]


# ======================
# GPU IMAGE
# ======================
FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04 AS gpu
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    sudo git cmake make build-essential gfortran pkg-config \
    wget curl unzip ca-certificates software-properties-common \
    python3-pip python3-venv python3-dev \
    libopenblas-dev liblapack-dev libblas-dev \
    swig fontconfig vim nano less && \
    rm -rf /var/lib/apt/lists/*

RUN useradd -m -s /bin/bash syscop && usermod -aG sudo syscop && \
    echo "syscop ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/90-syscop && chmod 0440 /etc/sudoers.d/90-syscop
ENV PIP_BREAK_SYSTEM_PACKAGES=1
USER syscop
ENV HOME=/home/syscop
WORKDIR /home/syscop
RUN python3 -m pip install --upgrade pip

# Copy leap-c repo to build acados
COPY --chown=syscop:syscop . /home/syscop/leap-c

# Build acados
WORKDIR /home/syscop/leap-c/external/acados
RUN mkdir -p build && cd build && \
    cmake .. -DACADOS_WITH_OPENMP=ON -DACADOS_WITH_QPOASES=ON && \
    make -j"$(nproc)"

# Install acados Python interface
WORKDIR /home/syscop/leap-c/external/acados/interfaces/acados_template
RUN pip install .

# Install PyTorch GPU
RUN pip install --index-url https://download.pytorch.org/whl/cu128 torch torchvision torchaudio

# Install leap-c in editable mode
WORKDIR /home/syscop/leap-c
RUN pip install -e .[rendering]

# Set up environment for pre-built acados
ENV ACADOS_SOURCE_DIR="/home/syscop/leap-c/external/acados"
ENV LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/home/syscop/leap-c/external/acados/lib"
ENV LIBGL_ALWAYS_SOFTWARE=1

WORKDIR /syscop_ws
CMD ["/bin/bash"]

