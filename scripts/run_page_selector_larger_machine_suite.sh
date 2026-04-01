#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF' >&2
usage: bash scripts/run_page_selector_larger_machine_suite.sh <output-root> [options]

Runs the full larger-machine selector-oracle pipeline:
  1. capture attention-subset page traces
  2. generate oracle labels and selector datasets
  3. materialize the checked-in comprehensive split suite
  4. run the manifest-driven batch selector bakeoff

Options:
  --model-id MODEL_ID                  Hugging Face model id (default: Qwen/Qwen3.5-4B)
  --device DEVICE                      Accelerator device (default: cuda)
  --torch-dtype DTYPE                  Torch dtype passed to capture harness (default: float16)
  --weight-quantization MODE           none or bnb_8bit (default: none)
  --tokens-per-page N                  Tokens per page for trace capture (default: 16)
  --group-size N                       Oracle grouping size (default: 32)
  --max-traces N                       Optional cap on oracle-labeled traces
  --max-per-stage-kind N               Cap sampled traces per stage/kind bucket (default: 256)
  --prompt-family NAME                 Repeatable. Default families: cache, reasoning, instruction, retrieval
  --prompt-length N                    Repeatable. Default lengths: 128, 256, 512, 1024
  --decode-steps N                     Repeatable. Default decode steps: 4, 8
  --kind K|V                           Repeatable. Default kinds: K, V
  --candidate TOKEN                    Repeatable oracle candidate token. Unset uses default kind-aware menu.
  --linear-steps N                     Selector training steps (default: 400)
  --linear-learning-rate VALUE         Selector learning rate (default: 0.2)
  --linear-l2 VALUE                    Selector L2 penalty (default: 1e-3)

Output layout under <output-root>:
  capture/     merged capture manifest plus per-run trace dirs
  labels/      oracle labels, selector datasets, and label summary
  suite/       frozen split suite plus split manifest
  batch_eval/  aggregate and per-split selector bakeoff reports
EOF
  exit 1
}

if [[ $# -lt 1 ]]; then
  usage
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="$REPO_ROOT/.venv/bin/python"
SUITE_CONFIG="$REPO_ROOT/configs/selector_split_suites/larger_machine_comprehensive_suite.json"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "python executable not found: $PYTHON_BIN" >&2
  exit 1
fi

if [[ ! -f "$SUITE_CONFIG" ]]; then
  echo "suite config not found: $SUITE_CONFIG" >&2
  exit 1
fi

OUTPUT_ROOT="$1"
shift

MODEL_ID="Qwen/Qwen3.5-4B"
DEVICE="cuda"
TORCH_DTYPE="float16"
WEIGHT_QUANTIZATION="none"
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
    --model-id)
      MODEL_ID="$2"
      shift 2
      ;;
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
  PROMPT_LENGTHS=("128" "256" "512" "1024")
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
  "$PYTHON_BIN" "$REPO_ROOT/scripts/run_qwen35_attention_subset_capture_sweep.py"
  --model-id "$MODEL_ID"
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

printf 'Running capture sweep into %s\n' "$CAPTURE_DIR" >&2
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
