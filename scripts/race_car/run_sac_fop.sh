#!/usr/bin/env bash
# Launcher for SAC-FOP training on race_car (with plant/model mismatch).
#
# Override via env vars:
#   SEED=0              RNG seed.
#   OUT=output/...      Output path (logs, checkpoints, videos).
#   CONTROLLER=race_car { race_car | race_car_stagewise }.
#   VARIANT=fop         { fop | fopc | foa }.
#   DEVICE=cpu          torch device.
#   DTYPE=float32       torch dtype.
#
# Extra CLI flags are passed through.
set -euo pipefail

SEED="${SEED:-0}"
CONTROLLER="${CONTROLLER:-race_car}"
VARIANT="${VARIANT:-fop}"
OUT="${OUT:-output/race_car_sac_${VARIANT}_${CONTROLLER}_seed${SEED}}"
DEVICE="${DEVICE:-cpu}"
DTYPE="${DTYPE:-float32}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

python "${SCRIPT_DIR}/run_sac_fop.py" \
    --seed "${SEED}" \
    --controller "${CONTROLLER}" \
    --variant "${VARIANT}" \
    --output-path "${OUT}" \
    --with-val \
    --reuse-code \
    --device "${DEVICE}" \
    --dtype "${DTYPE}" \
    "$@"
