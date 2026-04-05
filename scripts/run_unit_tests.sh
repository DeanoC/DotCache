#!/usr/bin/env bash
set -euo pipefail

.venv/bin/python -m pytest -q

if [[ "${RUN_GEMMA4_APPLE_SMOKE:-0}" == "1" ]]; then
  echo "Running opt-in Gemma 4 Apple regression lane..."
  .venv/bin/python -m pytest -q tests/test_gemma4_integration.py -k 'test_gemma4_generation_harness_runs_on_mps_tiny_random_model'
  .venv/bin/python scripts/run_gemma4_apple_smoke.py \
    --timeout-seconds "${GEMMA4_APPLE_SMOKE_TIMEOUT_SECONDS:-180}" \
    --output-dir "${GEMMA4_APPLE_SMOKE_OUTPUT_DIR:-benchmarks/results/gemma4_apple_smoke_$(date +%Y%m%d)}"
fi
