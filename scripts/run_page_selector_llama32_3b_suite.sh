#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF' >&2
usage: bash scripts/run_page_selector_llama32_3b_suite.sh <output-root> [options]

Runs the full selector-oracle pipeline for meta-llama/Llama-3.2-3B-Instruct.
Requires a valid HF token in HF_TOKEN or HUGGINGFACE_HUB_TOKEN.
EOF
  exit 1
}

if [[ $# -lt 1 ]]; then
  usage
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="$REPO_ROOT/.venv/bin/python"
SUITE_CONFIG="$REPO_ROOT/configs/selector_split_suites/larger_machine_comprehensive_suite.json"

if [[ -z "${HF_TOKEN:-}" && -z "${HUGGINGFACE_HUB_TOKEN:-}" ]]; then
  echo "HF_TOKEN or HUGGINGFACE_HUB_TOKEN must be set for meta-llama/Llama-3.2-3B-Instruct" >&2
  exit 1
fi

OUTPUT_ROOT="$1"
shift

DEVICE="cuda"
TORCH_DTYPE="float16"
TOKENS_PER_PAGE="16"
GROUP_SIZE="32"
MAX_TRACES=""
MAX_PER_STAGE_KIND="256"
LINEAR_STEPS="400"
LINEAR_LEARNING_RATE="0.2"
LINEAR_L2="1e-3"

declare -a PROMPT_FAMILIES=()
declare -a PROMPT_LENGTHS=()
declare -a DECODE_STEPS=()
declare -a KINDS=()
declare -a CANDIDATES=()

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
    --candidate)
      CANDIDATES+=("$2")
      shift 2
      ;;
    --linear-steps)
      LINEAR_STEPS="$2"
      shift 2
      ;;
    --linear-learning-rate)
      LINEAR_LEARNING_RATE="$2"
      shift 2
      ;;
    --linear-l2)
      LINEAR_L2="$2"
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
  PROMPT_FAMILIES=("cache" "reasoning" "instruction" "retrieval")
fi
if [[ ${#PROMPT_LENGTHS[@]} -eq 0 ]]; then
  PROMPT_LENGTHS=("256" "512" "1024")
fi
if [[ ${#DECODE_STEPS[@]} -eq 0 ]]; then
  DECODE_STEPS=("4" "8")
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
  "$PYTHON_BIN" "$REPO_ROOT/scripts/run_llama_page_trace_capture_sweep.py"
  --model-id "meta-llama/Llama-3.2-3B-Instruct"
  --device "$DEVICE"
  --torch-dtype "$TORCH_DTYPE"
  --tokens-per-page "$TOKENS_PER_PAGE"
  --group-size "$GROUP_SIZE"
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
for candidate in "${CANDIDATES[@]}"; do
  LABEL_CMD+=(--candidate "$candidate")
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
  --linear-steps "$LINEAR_STEPS"
  --linear-learning-rate "$LINEAR_LEARNING_RATE"
  --linear-l2 "$LINEAR_L2"
)

printf 'Running Llama capture sweep into %s\n' "$CAPTURE_DIR" >&2
"${CAPTURE_CMD[@]}"

printf 'Generating oracle labels into %s\n' "$LABELS_DIR" >&2
"${LABEL_CMD[@]}"

printf 'Materializing comprehensive split suite into %s\n' "$SUITE_DIR" >&2
"${SUITE_CMD[@]}"

printf 'Running selector batch bakeoff into %s\n' "$BATCH_DIR" >&2
"${BATCH_CMD[@]}"

echo "$CAPTURE_DIR/manifest.json"
echo "$LABELS_DIR/summary.json"
echo "$LABELS_DIR/summary.md"
echo "$SUITE_DIR/split_manifest.json"
echo "$BATCH_DIR/selector_split_batch_summary.json"
echo "$BATCH_DIR/selector_split_batch_summary.md"
