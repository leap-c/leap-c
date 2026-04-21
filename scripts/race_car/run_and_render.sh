#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../.."

OUTPUT_DIR="${1:-output/race_car_baseline}"
shift || true

python scripts/race_car/run_baseline.py --output-path "$OUTPUT_DIR" --overwrite "$@"
python scripts/race_car/render_baseline.py --run-dir "$OUTPUT_DIR" --save "$OUTPUT_DIR/lap.gif"
