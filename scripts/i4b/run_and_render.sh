#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../.."

python scripts/i4b/run_baseline.py --days 3 --output-path output/i4b_baseline --compute-sensitivities
python scripts/i4b/render_baseline.py --run-dir output/i4b_baseline
