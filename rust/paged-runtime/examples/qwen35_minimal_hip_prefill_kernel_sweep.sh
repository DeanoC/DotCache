#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT_DIR"

CARGO_BIN="${CARGO_BIN:-/home/deano/.cargo/bin/cargo}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen3.5-0.8B}"
DEVICE="${DEVICE:-hip:0}"
LAYER_ID="${LAYER_ID:-0}"
REPEATS="${REPEATS:-1}"
WARMUP_REPEATS="${WARMUP_REPEATS:-0}"
LINEAR_CHUNK_SIZE="${LINEAR_CHUNK_SIZE:-8}"
DELTA_SCAN_MODE="${DELTA_SCAN_MODE:-prebatched-local}"
DELTA_KERNEL_MIN_SEQUENCE="${DELTA_KERNEL_MIN_SEQUENCE:-1}"
PROMPT_TEXT="${PROMPT_TEXT:-Hello from DotCache}"
TARGETS="${TARGETS:-512 1024 2048}"

run_case() {
    local token_target="$1"

    echo "===== prompt_token_target=${token_target} ====="
    env \
        DOTCACHE_QWEN35_HIP_COMBINED_LINEAR_PREFILL=1 \
        CANDLE_QWEN35_LINEAR_CHUNK_SIZE="${LINEAR_CHUNK_SIZE}" \
        CANDLE_QWEN35_DELTA_SCAN_MODE="${DELTA_SCAN_MODE}" \
        DOTCACHE_QWEN35_DELTA_KERNEL_MIN_SEQUENCE="${DELTA_KERNEL_MIN_SEQUENCE}" \
        DOTCACHE_QWEN35_HIP_CHUNK_SINGLE_PREFILL=0 \
        DOTCACHE_QWEN35_HIP_MULTI_CHUNK_SCAN_PREFILL=0 \
        "$CARGO_BIN" run --release \
        --manifest-path rust/paged-runtime/Cargo.toml \
        --example hf_qwen35_minimal_linear_microbench \
        --features qwen35-minimal,qwen35-minimal-hip -- \
        "$MODEL_ID" "$PROMPT_TEXT" \
        --device "$DEVICE" \
        --layer-id "$LAYER_ID" \
        --prompt-token-target "$token_target" \
        --repeats "$REPEATS" \
        --warmup-repeats "$WARMUP_REPEATS"
    echo
}

for token_target in $TARGETS; do
    run_case "$token_target"
done
