#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from dotcache.page_oracle import (
    OracleThresholds,
    build_selector_candidate_training_rows,
    build_selector_training_rows,
    run_oracle_labeling,
    save_oracle_labels,
    save_selector_candidate_training_rows,
    save_selector_training_rows,
)
from dotcache.planner import parse_page_mode_token


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate per-page oracle labels from a captured page trace manifest.")
    parser.add_argument("--manifest", required=True, help="Path to a page trace manifest.json emitted by the capture harness.")
    parser.add_argument("--output-dir", required=True, help="Directory where labels.jsonl and summary.json should be written.")
    parser.add_argument("--group-size", type=int, default=32)
    parser.add_argument("--tokens-per-page", type=int, default=None)
    parser.add_argument("--max-traces", type=int, default=None)
    parser.add_argument("--max-per-stage-kind", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--stage", action="append", default=[], help="Optional stage filter, e.g. prefill or decode.")
    parser.add_argument("--kind", action="append", choices=["K", "V"], default=[])
    parser.add_argument("--layer-id", type=int, action="append", default=[])
    parser.add_argument("--candidate", action="append", default=[])
    parser.add_argument("--max-mean-abs-error-ratio", type=float, default=0.10)
    parser.add_argument("--max-max-abs-error-ratio", type=float, default=1.00)
    parser.add_argument("--max-token-p95-error-ratio", type=float, default=0.25)
    parser.add_argument("--max-score-max-abs-error", type=float, default=None)
    parser.add_argument("--min-score-topk-agreement", type=float, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    thresholds = OracleThresholds(
        max_mean_abs_error_ratio=args.max_mean_abs_error_ratio,
        max_max_abs_error_ratio=args.max_max_abs_error_ratio,
        max_token_p95_error_ratio=args.max_token_p95_error_ratio,
        max_score_max_abs_error=args.max_score_max_abs_error,
        min_score_topk_agreement=args.min_score_topk_agreement,
    )
    candidates = tuple(parse_page_mode_token(token) for token in args.candidate) if args.candidate else None
    result = run_oracle_labeling(
        args.manifest,
        group_size=args.group_size,
        tokens_per_page=args.tokens_per_page,
        candidates=candidates,
        thresholds=thresholds,
        max_traces=args.max_traces,
        max_per_stage_kind=args.max_per_stage_kind,
        seed=args.seed,
        kinds=tuple(args.kind) if args.kind else None,
        stages=tuple(args.stage) if args.stage else None,
        layer_ids=tuple(args.layer_id) if args.layer_id else None,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    labels_path = output_dir / "labels.jsonl"
    summary_path = output_dir / "summary.json"
    markdown_path = output_dir / "summary.md"
    selector_dataset_path = output_dir / "selector_dataset.jsonl"
    selector_schema_path = output_dir / "selector_schema.json"
    selector_candidate_dataset_path = output_dir / "selector_candidate_dataset.jsonl"
    selector_candidate_schema_path = output_dir / "selector_candidate_schema.json"
    save_oracle_labels(result, labels_path=labels_path, summary_path=summary_path)
    markdown_path.write_text(str(result.summary["summary_table_markdown"]) + "\n", encoding="utf-8")
    selector_rows = build_selector_training_rows(result.labels)
    save_selector_training_rows(selector_rows, selector_dataset_path)
    selector_schema = {
        "row_count": len(selector_rows),
        "fields": sorted(selector_rows[0].to_dict().keys()) if selector_rows else [],
    }
    selector_schema_path.write_text(json.dumps(selector_schema, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    selector_candidate_rows = build_selector_candidate_training_rows(result.labels)
    save_selector_candidate_training_rows(selector_candidate_rows, selector_candidate_dataset_path)
    selector_candidate_schema = {
        "row_count": len(selector_candidate_rows),
        "fields": sorted(selector_candidate_rows[0].to_dict().keys()) if selector_candidate_rows else [],
    }
    selector_candidate_schema_path.write_text(json.dumps(selector_candidate_schema, sort_keys=True, indent=2) + "\n", encoding="utf-8")

    print(labels_path)
    print(summary_path)
    print(markdown_path)
    print(selector_dataset_path)
    print(selector_schema_path)
    print(selector_candidate_dataset_path)
    print(selector_candidate_schema_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
