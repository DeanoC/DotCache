#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "usage: bash scripts/run_page_selector_local_smoke_suite.sh <oracle-bundle-dir> [output-root]" >&2
  exit 1
fi

ORACLE_BUNDLE_DIR="$1"
OUTPUT_ROOT="${2:-$ORACLE_BUNDLE_DIR/local_smoke_suite_run}"
SUITE_CONFIG="configs/selector_split_suites/local_smoke_suite.json"
SUITE_OUTPUT_ROOT="$OUTPUT_ROOT/suite"
BATCH_OUTPUT_ROOT="$OUTPUT_ROOT/batch_eval"

if [[ ! -d "$ORACLE_BUNDLE_DIR" ]]; then
  echo "oracle bundle directory does not exist: $ORACLE_BUNDLE_DIR" >&2
  exit 1
fi

if [[ ! -f "$ORACLE_BUNDLE_DIR/labels.jsonl" || ! -f "$ORACLE_BUNDLE_DIR/selector_dataset.jsonl" ]]; then
  echo "oracle bundle directory must contain labels.jsonl and selector_dataset.jsonl: $ORACLE_BUNDLE_DIR" >&2
  exit 1
fi

.venv/bin/python scripts/materialize_page_selector_split_suite.py \
  --input-dir "$ORACLE_BUNDLE_DIR" \
  --output-root "$SUITE_OUTPUT_ROOT" \
  --suite-config "$SUITE_CONFIG"

.venv/bin/python scripts/train_page_selector_split_batch.py \
  --split-manifest "$SUITE_OUTPUT_ROOT/split_manifest.json" \
  --output-dir "$BATCH_OUTPUT_ROOT"

echo "$BATCH_OUTPUT_ROOT/selector_split_batch_summary.json"
echo "$BATCH_OUTPUT_ROOT/selector_split_batch_summary.md"
