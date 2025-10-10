#!/usr/bin/env bash
set -euo pipefail
NAME=${1:-syscop}

# Get workspace root (syscop_ws directory)
WORKSPACE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../" && pwd)"

gpu_args=()
if docker run --rm --gpus all nvidia/cuda:12.8.1-base-ubuntu24.04 nvidia-smi >/dev/null 2>&1; then
  gpu_args=(--gpus all)
fi

xbind=()
if [ -n "${DISPLAY:-}" ] && [ -S /tmp/.X11-unix/X0 ]; then
  xhost +local:docker >/dev/null 2>&1 || true
  xbind=(-v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY="$DISPLAY")
fi

echo "[run] Mounting workspace: $WORKSPACE_ROOT -> /syscop_ws"
echo "[run] Starting container: $NAME"

docker run -it \
  --entrypoint /bin/bash \
  -v "$WORKSPACE_ROOT:/syscop_ws" \
  -w /syscop_ws \
  "${xbind[@]}" \
  "${gpu_args[@]}" \
  --ipc=host \
  --network=host \
  --name "$NAME" \
  syscop:latest


