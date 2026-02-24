#!/usr/bin/env bash
set -euo pipefail
# Get leap-c directory (build context includes leap-c repo for acados build)
LEAP_C_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../" && pwd)"
DF="$LEAP_C_DIR/docker/Dockerfile"

have_gpu=0
if docker run --rm --gpus all nvidia/cuda:12.8.1-base-ubuntu24.04 nvidia-smi >/dev/null 2>&1; then
  have_gpu=1
fi

if [[ $have_gpu -eq 0 ]] && command -v nvidia-smi >/dev/null 2>&1; then
  echo "[info] NVIDIA driver present. Trying to enable GPU runtime…"
  "$LEAP_C_DIR/docker/setup-ubuntu-gpu.sh" || true
  if docker run --rm --gpus all nvidia/cuda:12.8.1-base-ubuntu24.04 nvidia-smi >/dev/null 2>&1; then
    have_gpu=1
  fi
fi

if [[ $have_gpu -eq 1 ]]; then
  echo "[build] GPU available, building gpu image with pre-built acados"
  docker build -t syscop:gpu --target gpu -f "$DF" "$LEAP_C_DIR"
  docker tag syscop:gpu syscop:latest
else
  echo "[build] No GPU runtime, building cpu image with pre-built acados"
  docker build -t syscop:cpu --target cpu -f "$DF" "$LEAP_C_DIR"
  docker tag syscop:cpu syscop:latest
fi

echo "[done] Built: syscop:latest (and syscop:gpu or syscop:cpu)"
echo "[info] acados is pre-built in the image and ready to use!"


