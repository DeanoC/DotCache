#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: bash scripts/run_page_selector_qwen35_4b_suite.sh <output-root> [extra larger-machine options...]" >&2
  exit 1
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

exec bash "$REPO_ROOT/scripts/run_page_selector_larger_machine_suite.sh" \
  "$1" \
  --model-id "Qwen/Qwen3.5-4B" \
  "${@:2}"
