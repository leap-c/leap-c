# Docker Setup for leap-c

**Reproducible GPU/CPU containerized development environment for leap-c with acados.**

**Platform Support**: Windows | Linux (Ubuntu/Debian) | macOS  
**GPU Support**: NVIDIA CUDA (optional)

> **⚠️ IMPORTANT**: This repository must be placed in a `syscop_ws` directory. See [Directory Structure Requirements](#directory-structure-requirements) before starting!

---

## Table of Contents

1. [What is Docker?](#what-is-docker)
2. [Directory Structure Requirements](#directory-structure-requirements)
3. [Installation](#installation)
   - [Windows](#windows)
   - [Linux (Ubuntu/Debian)](#linux-ubuntudebian)
   - [VS Code Setup](#vs-code-setup)
4. [Quick Start](#quick-start)
   - [Method 1: VS Code Dev Container (Recommended)](#method-1-vs-code-dev-container-recommended)
   - [Method 2: Manual Scripts](#method-2-manual-scripts)
5. [Using Dev Containers](#using-dev-containers)
6. [Manual Docker Usage](#manual-docker-usage)
7. [Troubleshooting](#troubleshooting)
8. [File Structure](#file-structure)
9. [Advanced Topics](#advanced-topics)
10. [Summary](#summary)

---

## What is Docker?

**Docker** packages your application and dependencies into a **container** - like a lightweight, portable virtual machine.

### Key Concepts

- **Image**: Blueprint/template for a container
- **Container**: Running instance of an image
- **Volume**: Shared folder between host and container
- **Dev Container**: VS Code feature for developing inside containers

---

## Directory Structure Requirements

**IMPORTANT**: The repository must be placed in a specific directory structure for the setup to work correctly.

### Required Structure

```
syscop_ws/                    ← Your workspace root
└── leap-c/                   ← This repository
    ├── docker/               ← Docker configuration (you are here)
    │   ├── README.md
    │   ├── Dockerfile
    │   ├── build.sh
    │   ├── run.sh
    │   └── .devcontainer/
    ├── src/
    ├── tests/
    └── ...
```

### Setup Instructions

**Windows:**
```powershell
# Create workspace directory
mkdir C:\Users\YourName\syscop_ws
cd C:\Users\YourName\syscop_ws

# Clone or place the leap-c repository here
git clone <repository-url> leap-c
# OR move your existing leap-c folder here

# Your structure should now be:
# C:\Users\YourName\syscop_ws\leap-c\docker\
```

**Linux:**
```bash
# Create workspace directory
mkdir -p ~/syscop_ws
cd ~/syscop_ws

# Clone or place the leap-c repository here
git clone <repository-url> leap-c
# OR move your existing leap-c folder here

# Your structure should now be:
# ~/syscop_ws/leap-c/docker/
```

### Why This Structure?

- The Docker setup **mounts** `syscop_ws` as `/syscop_ws` inside the container
- This allows you to work on `leap-c` and potentially other projects in the same workspace
- All paths in scripts and configs expect this structure

### Verify Your Setup

Check that you have the correct structure:

**Windows:**
```powershell
# You should be able to run:
dir C:\Users\YourName\syscop_ws\leap-c\docker\Dockerfile
```

**Linux:**
```bash
# You should be able to run:
ls ~/syscop_ws/leap-c/docker/Dockerfile
```

If the file exists, you're ready to proceed!

---

## Installation

### Windows

#### 1. Install Docker Desktop

1. Download: https://docs.docker.com/desktop/install/windows-install/
2. Run installer, **enable "Use WSL 2 instead of Hyper-V"** 
3. Restart computer
4. Open Docker Desktop, wait for whale icon 🐳 in system tray
5. Verify:
   ```powershell
   docker --version
   docker run hello-world
   ```

### Linux (Ubuntu/Debian)

#### 1. Install Docker Engine

Full guide: https://docs.docker.com/engine/install/ubuntu/

#### 2. GPU Support (Optional - NVIDIA GPU only)

Manual: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html

### VS Code Setup

#### Install VS Code

- Guide: https://code.visualstudio.com/download

#### Install Required Extensions

**In VS Code** (`Ctrl+Shift+X`):

1. **Dev Containers** (ms-vscode-remote.remote-containers)
2. **Docker** (ms-azuretools.vscode-docker) - Recommended

Or via command line:
```bash
code --install-extension ms-vscode-remote.remote-containers
code --install-extension ms-azuretools.vscode-docker
```

---

## Quick Start

> **Prerequisites**: Ensure you have the [correct directory structure](#directory-structure-requirements) set up first!

### Method 1: VS Code Dev Container (Recommended)

**Easiest method - everything is automatic!**

#### Windows:
```powershell
# From your syscop_ws directory
cd C:\Users\YourName\syscop_ws\leap-c\docker
code .
```

#### Linux:
```bash
# From your syscop_ws directory
cd ~/syscop_ws/leap-c/docker
code .
```

#### In VS Code:

1. Press `Ctrl+Shift+P` or `F1` or bottom left >< icon
2. Type: **"Dev Containers: Reopen in Container"**
3. Select configuration:
   ```
   Select a dev container configuration:
       leap-c (auto-detect GPU/CPU)  - RECOMMENDED
       leap-c (force CPU)            - Testing
       leap-c (force GPU)            - Performance
   ```
4. Wait for build (~5-10 minutes first time) ☕
5. **Done!** You're inside the container!

Verify inside container:
```bash
python3 -c "from acados_template import AcadosOcp; print('acados works!')"
nvidia-smi  # If GPU mode
```

### Method 2: Manual Scripts

**Linux/macOS:**
```bash
# Navigate to docker folder from your syscop_ws
cd ~/syscop_ws/leap-c/docker

# Build image (auto-detects GPU)
./build.sh

# Run container
./run.sh
```

**Windows (PowerShell/Git Bash):**
```powershell
# Navigate to docker folder from your syscop_ws
cd C:\Users\YourName\syscop_ws\leap-c\docker

# Build image
bash build.sh

# Run container
bash run.sh
```

Your workspace (`syscop_ws` directory) is mounted at `/syscop_ws` inside the container.

---

## Using Dev Containers

### Opening the Container

**Method 1: Command Palette** (Easiest)
1. Open `docker` folder in VS Code: `code ~/syscop_ws/leap-c/docker` (Linux) or `code C:\Users\YourName\syscop_ws\leap-c\docker` (Windows)
2. `Ctrl+Shift+P` -- "Dev Containers: Reopen in Container"
3. Select your preferred configuration

**Method 2: Popup**
- VS Code may show a popup: "Folder contains a Dev Container configuration"
- Click **"Reopen in Container"**

**Method 3: Status Bar**
- Click the icon in bottom-left corner: `>< WSL` or similar
- Select "Reopen in Container"

### Inside the Container

Your terminal prompt changes:
```bash
syscop@container:/syscop_ws$  # You're in the container!
```

**Workspace location:**
- Host: `C:\Users\YourName\syscop_ws` (Windows) or `~/syscop_ws` (Linux)
- Container: `/syscop_ws`
- **Changes sync automatically!**

**Pre-installed:**
- Python, C++, CMake tools
- acados (pre-built)
- PyTorch
- VS Code extensions (Python, C++, Docker, Jupyter)

### Working in Container

```bash
# Your code
cd /syscop_ws/leap-c

# Edit files (saved to host)
code src/my_file.py

# Run Python
python3 my_script.py

# Run tests
pytest tests/
```

### Exiting the Container

1. `Ctrl+Shift+P`
2. "Dev Containers: Reopen Folder Locally"

Your files are **still on your host machine**.

### Switching Configurations

1. Reopen locally (see above)
2. Reopen in container
3. Select different configuration

---

## Manual Docker Usage

### Building Images

**Auto-detect (uses build.sh):**
```bash
# Navigate to docker folder
cd ~/syscop_ws/leap-c/docker  # Linux/macOS
# OR
cd C:\Users\YourName\syscop_ws\leap-c\docker  # Windows

# Build
./build.sh      # Linux/macOS
bash build.sh   # Windows
```

**Build specific target:**
```bash
# Navigate to leap-c repository root
cd ~/syscop_ws/leap-c  # Linux/macOS
# OR
cd C:\Users\YourName\syscop_ws\leap-c  # Windows

# GPU image
docker build -t syscop:gpu --target gpu -f docker/Dockerfile .

# CPU image
docker build -t syscop:cpu --target cpu -f docker/Dockerfile .
```

### Running Containers

**Using run.sh:**
```bash
# Navigate to docker folder
cd ~/syscop_ws/leap-c/docker           # Linux/macOS
cd C:\Users\YourName\syscop_ws\leap-c\docker  # Windows

# Run
./run.sh        # Linux/macOS
bash run.sh     # Windows
```

**Manual docker run (Linux):**
```bash
docker run -it \
  -v ~/syscop_ws:/syscop_ws \
  -w /syscop_ws \
  --gpus all \
  --ipc=host \
  --network=host \
  syscop:latest
```

**Manual docker run (Windows PowerShell):**
```powershell
docker run -it `
  -v C:\Users\YourName\syscop_ws:/syscop_ws `
  -w /syscop_ws `
  --gpus all `
  --ipc=host `
  syscop:latest
```

**Without GPU:**
```bash
docker run -it \
  -v ~/syscop_ws:/syscop_ws \
  -w /syscop_ws \
  --ipc=host \
  syscop:latest
```

### Useful Commands

```bash
# List images
docker images | grep syscop

# List containers
docker ps -a

# Stop container
docker stop syscop

# Remove container
docker rm syscop

# Remove image
docker rmi syscop:latest

# Clean up
docker system prune

# View logs
docker logs syscop

# Execute in running container
docker exec -it syscop bash
```

### Rebuild from Scratch

```bash
# Remove all syscop images
docker rmi syscop:latest syscop:gpu syscop:cpu

# Clear cache (optional)
docker builder prune

# Rebuild
cd ~/syscop_ws/leap-c/docker  # Linux/macOS
cd C:\Users\YourName\syscop_ws\leap-c\docker  # Windows

./build.sh      # Linux/macOS
bash build.sh   # Windows
```

---

## Troubleshooting

### Windows Issues

#### "Docker daemon is not running"
**Fix:**
1. Open Docker Desktop
2. Wait for whale icon 🐳 in system tray
3. Try again

#### "WSL 2 installation is incomplete"
**Fix:**
```powershell
# Open PowerShell as Administrator
wsl --update
# Restart computer
```

#### GPU not detected on Windows
**Fix:**
1. Windows 11 or Windows 10 21H2+ required
2. Install latest NVIDIA drivers: https://www.nvidia.com/download/index.aspx
3. Docker Desktop -- Settings:
   - General: "Use the WSL 2 based engine"
   - Resources - WSL Integration: Enable
4. Apply & Restart

See: https://docs.docker.com/desktop/gpu/

## File Structure

```
docker/
├── build.sh                 # Build Docker image (auto-detects GPU/CPU)
├── run.sh                   # Run container with workspace mounted
├── setup-ubuntu-gpu.sh      # Install NVIDIA Container Toolkit (Linux)
├── Dockerfile               # Multi-stage build (cpu/gpu targets)
├── README.md                # This file
└── .devcontainer/           # VS Code Dev Container configurations
    ├── leap-c-auto/         # Auto-detect GPU/CPU (default)
    │   └── devcontainer.json
    ├── leap-c-cpu/          # Force CPU mode
    │   └── devcontainer.json
    ├── leap-c-gpu/          # Force GPU mode
    │   └── devcontainer.json
    ├── select-target.sh     # OS & GPU detection script
    └── .gitignore
```

### Three Dev Container Configurations

When you open in VS Code, you can choose:

1. **leap-c (auto-detect GPU/CPU)**
   - Automatically detects hardware
   - Uses GPU if available, CPU otherwise
   - **Recommended for most users**

2. **leap-c (force CPU)**
   - Always uses CPU mode
   - Good for testing
   - Lighter resource usage

3. **leap-c (force GPU)**
   - Always uses GPU mode
   - Skips detection
   - Maximum performance

---

## Advanced Topics

### Architecture

- **acados** is pre-built in the image at `/home/syscop/leap-c/external/acados`
- **Your workspace** is mounted at `/syscop_ws` (live development)
- **Best of both worlds**: Fast startup + live code editing

### Two Code Locations

| Location | Purpose | Persistence |
|----------|---------|-------------|
| `/home/syscop/leap-c` | Pre-built acados/leap-c | Image (immutable) |
| `/syscop_ws/leap-c` | Your live workspace | Host filesystem (editable) |

**For development**: Work in `/syscop_ws/leap-c` - changes persist on your host.

### Environment Variables

Inside container:
- `ACADOS_SOURCE_DIR=/home/syscop/leap-c/external/acados`
- `LD_LIBRARY_PATH` includes acados libraries
- `LIBGL_ALWAYS_SOFTWARE=1` for rendering

### Dockerfile Multi-Stage Build

```dockerfile
# CPU stage
FROM ubuntu:24.04 AS cpu
# ... CPU setup

# GPU stage
FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04 AS gpu
# ... GPU setup (CUDA, cuDNN)
```

Build specific stage:
```bash
docker build --target cpu -f Dockerfile .
docker build --target gpu -f Dockerfile .
```

### Customizing Dev Container

Edit `.devcontainer/leap-c-auto/devcontainer.json`:

```json
{
  "customizations": {
    "vscode": {
      "extensions": [
        "ms-python.python",
        "your-extension-here"
      ],
      "settings": {
        "editor.fontSize": 14
      }
    }
  }
}
```

### Scripts Explained

**build.sh**
- Detects GPU availability
- Builds appropriate target (cpu/gpu)
- Tags as `syscop:latest`

**run.sh**
- Detects GPU
- Mounts workspace
- Configures X11 for GUI
- Starts container

**setup-ubuntu-gpu.sh** (Linux only)
- Installs NVIDIA Container Toolkit
- Configures Docker for GPU
- Restarts Docker daemon

**select-target.sh**
- Used by Dev Containers
- Detects OS (Windows/Linux/macOS)
- Detects GPU availability
- Builds image if needed
- Creates VS Code configuration

### Platform-Specific Notes

**Windows:**
- Docker Desktop required
- WSL 2 for best performance
- GPU: Windows 11 or 10 21H2+ with NVIDIA drivers
- Paths: Use `C:\Users\...`, VS Code handles conversion

**Linux:**
- Native Docker, best performance
- GPU: Automated via `./setup-ubuntu-gpu.sh`
- X11 GUI works out of the box

**macOS:**
- Docker Desktop required
- No GPU support (use CPU mode)
- X11: Requires XQuartz

---

## Summary

### For Beginners

1. Install Docker Desktop (Windows) or Docker Engine (Linux)
2. Install VS Code + Dev Containers extension
3. Open `docker` folder in VS Code
4. `Ctrl+Shift+P` -- "Reopen in Container"
5. Select "auto-detect GPU/CPU"
6. Wait for build
7. Start coding in `/syscop_ws/leap-c`!

### For Advanced Users

- Manual builds: `./build.sh`
- Manual run: `./run.sh`
- Customize `devcontainer.json`
- Force CPU/GPU modes for testing
- Use `docker build` directly for CI/CD

### Resources

- **Docker**: https://docs.docker.com/
- **VS Code Dev Containers**: https://code.visualstudio.com/docs/devcontainers/containers
- **NVIDIA Container Toolkit**: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/
- **Docker Desktop GPU**: https://docs.docker.com/desktop/gpu/

---

