#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 5 ]]; then
  echo "usage: $0 <output-root> <pack-name> <prompt-pack> <shard-count> <point=artifact> [point=artifact ...]" >&2
  exit 1
fi

OUTPUT_ROOT="$1"
PACK_NAME="$2"
PROMPT_PACK="$3"
SHARD_COUNT="$4"
shift 4

POINT_MAPPINGS=("$@")
ACTIVE_POINT_MAPPINGS=()

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

mkdir -p "${OUTPUT_ROOT}"

for mapping in "${POINT_MAPPINGS[@]}"; do
  point_id="${mapping%%=*}"
  point_dir="${OUTPUT_ROOT}/${point_id}/${PACK_NAME}"
  if [[ -f "${point_dir}/longbench_selector_compare.json" ]] && [[ -f "${point_dir}/longbench_failure_workbook.json" ]]; then
    echo "skipping completed point for ${PACK_NAME}: ${point_id}" >&2
    continue
  fi
  shard_dir="${OUTPUT_ROOT}/${point_id}/${PACK_NAME}/shards"
  mkdir -p "${shard_dir}"
  rm -f "${shard_dir}"/shard_*.jsonl
  ACTIVE_POINT_MAPPINGS+=("${mapping}")
done

if [[ "${#ACTIVE_POINT_MAPPINGS[@]}" -eq 0 ]]; then
  echo "all requested points already complete for ${PACK_NAME}" >&2
  exit 0
fi

POINT_ARGS=()
for mapping in "${ACTIVE_POINT_MAPPINGS[@]}"; do
  POINT_ARGS+=(--point "${mapping}")
done

PIDS=()
cleanup() {
  local pid
  for pid in "${PIDS[@]:-}"; do
    kill "${pid}" 2>/dev/null || true
  done
  sleep 2
  for pid in "${PIDS[@]:-}"; do
    kill -9 "${pid}" 2>/dev/null || true
  done
}
trap cleanup EXIT INT TERM

for (( shard_index=0; shard_index<SHARD_COUNT; shard_index++ )); do
  ./.venv/bin/python scripts/run_qwen35_longbench_selector_operating_points.py \
    --model-id Qwen/Qwen3.5-9B \
    --backend torch_cuda \
    --device cuda \
    --torch-dtype float16 \
    --comparison-cases dense exact quality systems \
    --prompt-pack "${PROMPT_PACK}" \
    --pack-name "${PACK_NAME}" \
    --output-root "${OUTPUT_ROOT}" \
    --prompt-shard-count "${SHARD_COUNT}" \
    --prompt-shard-index "${shard_index}" \
    --max-prompt-tokens 4096 8192 \
    --warmup-runs 1 \
    --measured-runs 5 \
    --quality-check \
    "${POINT_ARGS[@]}" &
  PIDS+=("$!")
  if (( shard_index + 1 < SHARD_COUNT )); then
    sleep "${SHARD_START_STAGGER_SECONDS:-20}"
  fi
done

for pid in "${PIDS[@]}"; do
  wait "${pid}"
done

trap - EXIT INT TERM

for mapping in "${ACTIVE_POINT_MAPPINGS[@]}"; do
  point_id="${mapping%%=*}"
  point_dir="${OUTPUT_ROOT}/${point_id}/${PACK_NAME}"
  mkdir -p "${point_dir}"
  jsonl_path="${point_dir}/qwen3p5-9b_longbench_${PACK_NAME}.jsonl"
  for (( shard_index=0; shard_index<SHARD_COUNT; shard_index++ )); do
    shard_path="${point_dir}/shards/shard_$(printf '%02d' "${shard_index}").jsonl"
    if [[ ! -s "${shard_path}" ]]; then
      echo "missing shard output for ${point_id} ${PACK_NAME}: ${shard_path}" >&2
      exit 1
    fi
  done
  cat "${point_dir}"/shards/shard_*.jsonl > "${jsonl_path}"

  ./.venv/bin/python scripts/report_qwen35_longbench_selector_compare.py \
    --input "${jsonl_path}" \
    --expected-cases dense exact quality systems \
    --markdown-output "${point_dir}/longbench_selector_compare.md" \
    --json-output "${point_dir}/longbench_selector_compare.json" \
    --title "Qwen/Qwen3.5-9B LongBench ${PACK_NAME} ${point_id}"

  ./.venv/bin/python scripts/report_qwen35_longbench_failure_workbook.py \
    --input "${jsonl_path}" \
    --markdown-output "${point_dir}/longbench_failure_workbook.md" \
    --json-output "${point_dir}/longbench_failure_workbook.json" \
    --title "Qwen/Qwen3.5-9B LongBench ${PACK_NAME} ${point_id} Failure Workbook"
done
