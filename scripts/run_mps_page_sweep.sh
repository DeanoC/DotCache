#!/usr/bin/env bash
set -euo pipefail

.venv/bin/python benchmarks/bench_page_sweep.py "$@"
