#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

CARGO_BIN="${CARGO_BIN:-/home/deano/.cargo/bin/cargo}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen3.5-0.8B}"
DEVICE="${DEVICE:-hip:0}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1}"
LINEAR_CHUNK_SIZE="${LINEAR_CHUNK_SIZE:-8}"
DELTA_SCAN_MODE="${DELTA_SCAN_MODE:-prebatched-local}"
DELTA_KERNEL_MIN_SEQUENCE="${DELTA_KERNEL_MIN_SEQUENCE:-1}"

run_case() {
    local label="$1"
    local prompt="$2"
    shift 2

    echo "===== ${label} ====="
    env \
        DOTCACHE_QWEN35_PRINT_HIP_TRANSFERS=1 \
        CANDLE_QWEN35_LINEAR_CHUNK_SIZE="${LINEAR_CHUNK_SIZE}" \
        CANDLE_QWEN35_DELTA_SCAN_MODE="${DELTA_SCAN_MODE}" \
        DOTCACHE_QWEN35_DELTA_KERNEL_MIN_SEQUENCE="${DELTA_KERNEL_MIN_SEQUENCE}" \
        DOTCACHE_QWEN35_HIP_CHUNK_SINGLE_PREFILL=0 \
        DOTCACHE_QWEN35_HIP_MULTI_CHUNK_SCAN_PREFILL=0 \
        "$@" \
        "$CARGO_BIN" run --release \
        --manifest-path rust/paged-runtime/Cargo.toml \
        --example hf_qwen35_minimal \
        --features qwen35-minimal,qwen35-minimal-hip -- \
        "$MODEL_ID" "$prompt" "$MAX_NEW_TOKENS" --device "$DEVICE"
    echo
}

SHORT_PROMPT="${SHORT_PROMPT:-Hello from DotCache}"
MEDIUM_PROMPT="${MEDIUM_PROMPT:-The quick brown fox jumps over the lazy dog in a compact medium prompt for HIP validation.}"
LONG_PROMPT="${LONG_PROMPT:-The quick brown fox jumps over the lazy dog while the system processes several chunks of linear attention state on the HIP backend for this validation run.}"

run_case "short baseline" "$SHORT_PROMPT" \
    CANDLE_QWEN35_DELTA_STATE_SCAN_KERNEL=0 \
    CANDLE_QWEN35_DELTA_CHUNK_FUSED_KERNEL=0 \
    CANDLE_QWEN35_DELTA_FULL_KERNEL=0

run_case "short full-scan" "$SHORT_PROMPT" \
    CANDLE_QWEN35_DELTA_STATE_SCAN_KERNEL=0 \
    CANDLE_QWEN35_DELTA_CHUNK_FUSED_KERNEL=0 \
    CANDLE_QWEN35_DELTA_FULL_KERNEL=1

run_case "medium baseline" "$MEDIUM_PROMPT" \
    CANDLE_QWEN35_DELTA_STATE_SCAN_KERNEL=0 \
    CANDLE_QWEN35_DELTA_CHUNK_FUSED_KERNEL=0 \
    CANDLE_QWEN35_DELTA_FULL_KERNEL=0

run_case "medium full-scan" "$MEDIUM_PROMPT" \
    CANDLE_QWEN35_DELTA_STATE_SCAN_KERNEL=0 \
    CANDLE_QWEN35_DELTA_CHUNK_FUSED_KERNEL=0 \
    CANDLE_QWEN35_DELTA_FULL_KERNEL=1

run_case "long baseline" "$LONG_PROMPT" \
    CANDLE_QWEN35_DELTA_STATE_SCAN_KERNEL=0 \
    CANDLE_QWEN35_DELTA_CHUNK_FUSED_KERNEL=0 \
    CANDLE_QWEN35_DELTA_FULL_KERNEL=0

run_case "long full-scan" "$LONG_PROMPT" \
    CANDLE_QWEN35_DELTA_STATE_SCAN_KERNEL=0 \
    CANDLE_QWEN35_DELTA_CHUNK_FUSED_KERNEL=0 \
    CANDLE_QWEN35_DELTA_FULL_KERNEL=1
