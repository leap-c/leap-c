#!/usr/bin/env bash
set -euo pipefail
  
.venv/bin/python scripts/hvac/analyze_sensitivities.py \
    --output-dir outputs/sensitivity \
    --n-horizon 96 \
    --reuse-code-dir /tmp/hvac_code