#!/usr/bin/env bash
set -euo pipefail

.venv/bin/python benchmarks/bench_score.py
.venv/bin/python benchmarks/bench_decode.py
