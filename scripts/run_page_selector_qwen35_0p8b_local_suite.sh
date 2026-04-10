#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF' >&2
usage: bash scripts/run_page_selector_qwen35_0p8b_local_suite.sh <output-root> [options]

Runs a local Apple-box selector-oracle pipeline for Qwen/Qwen3.5-0.8B:
  1. capture attention-subset page traces on MPS
  2. generate oracle labels and selector datasets
  3. materialize the local smoke split suite
  4. run the compression-aware split bakeoff

Options:
  --device DEVICE                      Default: mps
  --torch-dtype DTYPE                  Default: float16
  --weight-quantization MODE           none or bnb_8bit (default: none)
  --tokens-per-page N                  Default: 16
  --group-size N                       Default: 32
  --max-traces N                       Optional cap on oracle-labeled traces
  --max-per-stage-kind N               Default: 128
  --prompt-family NAME                 Repeatable. Default: reasoning, instruction, retrieval
  --prompt-length N                    Repeatable. Default: 512, 1024
  --decode-steps N                     Repeatable. Default: 4
  --kind K|V                           Repeatable. Default: K, V
EOF
  exit 1
}

if [[ $# -lt 1 ]]; then
  usage
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="$REPO_ROOT/.venv/bin/python"
SUITE_CONFIG="$REPO_ROOT/configs/selector_split_suites/local_smoke_suite.json"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "python executable not found: $PYTHON_BIN" >&2
  exit 1
fi

OUTPUT_ROOT="$1"
shift

DEVICE="mps"
TORCH_DTYPE="float16"
WEIGHT_QUANTIZATION="none"
TOKENS_PER_PAGE="16"
GROUP_SIZE="32"
MAX_TRACES=""
MAX_PER_STAGE_KIND="128"

declare -a PROMPT_FAMILIES=()
declare -a PROMPT_LENGTHS=()
declare -a DECODE_STEPS=()
declare -a KINDS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --device)
      DEVICE="$2"
      shift 2
      ;;
    --torch-dtype)
      TORCH_DTYPE="$2"
      shift 2
      ;;
    --weight-quantization)
      WEIGHT_QUANTIZATION="$2"
      shift 2
      ;;
    --tokens-per-page)
      TOKENS_PER_PAGE="$2"
      shift 2
      ;;
    --group-size)
      GROUP_SIZE="$2"
      shift 2
      ;;
    --max-traces)
      MAX_TRACES="$2"
      shift 2
      ;;
    --max-per-stage-kind)
      MAX_PER_STAGE_KIND="$2"
      shift 2
      ;;
    --prompt-family)
      PROMPT_FAMILIES+=("$2")
      shift 2
      ;;
    --prompt-length)
      PROMPT_LENGTHS+=("$2")
      shift 2
      ;;
    --decode-steps)
      DECODE_STEPS+=("$2")
      shift 2
      ;;
    --kind)
      KINDS+=("$2")
      shift 2
      ;;
    --help|-h)
      usage
      ;;
    *)
      echo "unknown option: $1" >&2
      usage
      ;;
  esac
done

if [[ ${#PROMPT_FAMILIES[@]} -eq 0 ]]; then
  PROMPT_FAMILIES=("reasoning" "instruction" "retrieval")
fi
if [[ ${#PROMPT_LENGTHS[@]} -eq 0 ]]; then
  PROMPT_LENGTHS=("512" "1024")
fi
if [[ ${#DECODE_STEPS[@]} -eq 0 ]]; then
  DECODE_STEPS=("4")
fi
if [[ ${#KINDS[@]} -eq 0 ]]; then
  KINDS=("K" "V")
fi

CAPTURE_DIR="$OUTPUT_ROOT/capture"
LABELS_DIR="$OUTPUT_ROOT/labels"
SUITE_DIR="$OUTPUT_ROOT/suite"
BATCH_DIR="$OUTPUT_ROOT/batch_eval"
mkdir -p "$CAPTURE_DIR" "$LABELS_DIR" "$SUITE_DIR" "$BATCH_DIR"

CAPTURE_CMD=(
  "$PYTHON_BIN" "$REPO_ROOT/scripts/run_qwen35_page_trace_capture_sweep.py"
  --model-id "Qwen/Qwen3.5-0.8B"
  --device "$DEVICE"
  --torch-dtype "$TORCH_DTYPE"
  --weight-quantization "$WEIGHT_QUANTIZATION"
  --tokens-per-page "$TOKENS_PER_PAGE"
  --output-dir "$CAPTURE_DIR"
)
for family in "${PROMPT_FAMILIES[@]}"; do
  CAPTURE_CMD+=(--prompt-family "$family")
done
for length in "${PROMPT_LENGTHS[@]}"; do
  CAPTURE_CMD+=(--prompt-length "$length")
done
for steps in "${DECODE_STEPS[@]}"; do
  CAPTURE_CMD+=(--decode-steps "$steps")
done
for kind in "${KINDS[@]}"; do
  CAPTURE_CMD+=(--kind "$kind")
done

LABEL_CMD=(
  "$PYTHON_BIN" "$REPO_ROOT/scripts/generate_page_oracle_labels.py"
  --manifest "$CAPTURE_DIR/manifest.json"
  --output-dir "$LABELS_DIR"
  --group-size "$GROUP_SIZE"
  --tokens-per-page "$TOKENS_PER_PAGE"
  --max-per-stage-kind "$MAX_PER_STAGE_KIND"
)
if [[ -n "$MAX_TRACES" ]]; then
  LABEL_CMD+=(--max-traces "$MAX_TRACES")
fi
for kind in "${KINDS[@]}"; do
  LABEL_CMD+=(--kind "$kind")
done

SUITE_CMD=(
  "$PYTHON_BIN" "$REPO_ROOT/scripts/materialize_page_selector_split_suite.py"
  --input-dir "$LABELS_DIR"
  --output-root "$SUITE_DIR"
  --suite-config "$SUITE_CONFIG"
)

BATCH_CMD=(
  "$PYTHON_BIN" "$REPO_ROOT/scripts/train_page_selector_split_batch.py"
  --split-manifest "$SUITE_DIR/split_manifest.json"
  --output-dir "$BATCH_DIR"
)

printf 'Running Qwen3.5-0.8B trace capture into %s\n' "$CAPTURE_DIR" >&2
"${CAPTURE_CMD[@]}"

printf 'Generating oracle labels into %s\n' "$LABELS_DIR" >&2
"${LABEL_CMD[@]}"

printf 'Materializing local smoke suite into %s\n' "$SUITE_DIR" >&2
"${SUITE_CMD[@]}"

printf 'Running selector batch bakeoff into %s\n' "$BATCH_DIR" >&2
"${BATCH_CMD[@]}"

echo "$CAPTURE_DIR/manifest.json"
echo "$LABELS_DIR/summary.json"
echo "$LABELS_DIR/summary.md"
echo "$SUITE_DIR/split_manifest.json"
echo "$BATCH_DIR/selector_split_batch_summary.json"
echo "$BATCH_DIR/selector_split_batch_summary.md"
