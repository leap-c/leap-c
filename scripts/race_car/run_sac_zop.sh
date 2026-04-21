#!/usr/bin/env bash
# Launcher for SAC-ZOP training on race_car (with plant/model mismatch).
#
# Override via env vars:
#   SEED=0              RNG seed.
#   OUT=output/...      Output path (logs, checkpoints, videos).
#   CONTROLLER=race_car { race_car | race_car_stagewise }.
#   DEVICE=cpu          torch device.
#   DTYPE=float32       torch dtype.
#
# Extra CLI flags are passed through, e.g.:
#   bash scripts/race_car/run_sac_zop.sh --mismatch-cm1 0.9
set -euo pipefail

SEED="${SEED:-0}"
CONTROLLER="${CONTROLLER:-race_car}"
OUT="${OUT:-output/race_car_sac_zop_${CONTROLLER}_seed${SEED}}"
DEVICE="${DEVICE:-cpu}"
DTYPE="${DTYPE:-float32}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

python "${SCRIPT_DIR}/run_sac_zop.py" \
    --seed "${SEED}" \
    --controller "${CONTROLLER}" \
    --output-path "${OUT}" \
    --with-val \
    --reuse-code \
    --device "${DEVICE}" \
    --dtype "${DTYPE}" \
    "$@"
