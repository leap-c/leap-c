#!/usr/bin/env bash
set -euo pipefail

# Detect OS
detect_os() {
  if [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "win32" ]] || [[ "$OSTYPE" == "cygwin" ]]; then
    echo "windows"
  elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    echo "linux"
  elif [[ "$OSTYPE" == "darwin"* ]]; then
    echo "macos"
  else
    echo "unknown"
  fi
}

OS_TYPE=$(detect_os)
echo "[devcontainer] Detected OS: $OS_TYPE"

# Parse command line arguments
FORCE_MODE=""
while [[ $# -gt 0 ]]; do
  case $1 in
    --force-cpu)
      FORCE_MODE="cpu"
      shift
      ;;
    --force-gpu)
      FORCE_MODE="gpu"
      shift
      ;;
    --help|-h)
      echo "Usage: $0 [--force-cpu|--force-gpu]"
      echo "  --force-cpu    Force CPU build even if GPU is available"
      echo "  --force-gpu    Force GPU build (skip GPU detection)"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      echo "Use --help for usage information"
      exit 1
      ;;
  esac
done

echo "[devcontainer] Checking for GPU support..."

# Check for GPU support (platform-specific)
GPU_OK=0

if [[ "$FORCE_MODE" == "cpu" ]]; then
  echo "[devcontainer] --force-cpu detected - forcing CPU mode"
  GPU_OK=0
elif [[ "$FORCE_MODE" == "gpu" ]]; then
  echo "[devcontainer] --force-gpu detected - forcing GPU mode"
  GPU_OK=1
elif [[ "$OS_TYPE" == "windows" ]]; then
  # Windows: Check for NVIDIA GPU using nvidia-smi in Docker
  if docker run --rm --gpus all nvidia/cuda:12.8.1-base-ubuntu24.04 nvidia-smi >/dev/null 2>&1; then
    GPU_OK=1
    echo "[devcontainer] GPU detected (Windows)"
  else
    echo "[devcontainer] No GPU detected on Windows - using CPU mode"
    echo "[devcontainer] Note: Ensure Docker Desktop has GPU support enabled"
  fi
elif docker run --rm --gpus all nvidia/cuda:12.8.1-base-ubuntu24.04 nvidia-smi >/dev/null 2>&1; then
  GPU_OK=1
  echo "[devcontainer] GPU detected"
else
  echo "[devcontainer] No GPU detected - using CPU mode"
fi

# Determine which image to use/build
if [[ $GPU_OK -eq 1 ]]; then
  IMAGE_TAG="syscop:gpu"
  BUILD_TARGET="gpu"
else
  IMAGE_TAG="syscop:cpu"
  BUILD_TARGET="cpu"
fi

# Check if the required image exists
if ! docker image inspect "$IMAGE_TAG" >/dev/null 2>&1; then
  echo "[devcontainer] $IMAGE_TAG image not found!"
  echo "[devcontainer] Building $BUILD_TARGET target..."
  
  # Navigate to leap-c directory and build specific target
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  LEAP_C_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
  DF="$LEAP_C_DIR/docker/Dockerfile"
  
  docker build -t "$IMAGE_TAG" --target "$BUILD_TARGET" -f "$DF" "$LEAP_C_DIR"
  docker tag "$IMAGE_TAG" syscop:latest
  echo "[devcontainer] Built $IMAGE_TAG"
fi

# Create config override
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="$SCRIPT_DIR/devcontainer.local.json"

if [[ $GPU_OK -eq 1 ]]; then
  cat > "$CONFIG_FILE" <<JSON
{
  "image": "$IMAGE_TAG",
  "runArgs": [
    "--gpus", "all",
    "--ipc=host",
    "--network=host"
  ]
}
JSON
  echo "[devcontainer] Using GPU image: $IMAGE_TAG"
else
  cat > "$CONFIG_FILE" <<JSON
{
  "image": "$IMAGE_TAG",
  "runArgs": [
    "--ipc=host",
    "--network=host"
  ]
}
JSON
  echo "[devcontainer] Using CPU image: $IMAGE_TAG"
fi

echo "[devcontainer] Configuration ready!"
