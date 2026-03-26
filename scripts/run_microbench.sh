#!/usr/bin/env bash
set -euo pipefail

backend="${1:-auto}"

.venv/bin/python benchmarks/bench_encode.py --backend "${backend}"
.venv/bin/python benchmarks/bench_score.py --backend "${backend}"
.venv/bin/python benchmarks/bench_mix.py --backend "${backend}"
.venv/bin/python benchmarks/bench_decode.py --backend "${backend}"
.venv/bin/python benchmarks/bench_decode_reuse.py --backend "${backend}"
.venv/bin/python benchmarks/bench_decode_growth.py --backend "${backend}"
