#!/usr/bin/env bash
set -euo pipefail

.venv/bin/python benchmarks/bench_decode_envelope_sweep.py --backend torch_mps "$@"
