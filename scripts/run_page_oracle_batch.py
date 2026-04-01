#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from dotcache.page_oracle import OracleThresholds, run_oracle_batch_replay
from dotcache.planner import parse_page_mode_token


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay a sampled batch of captured page traces against a candidate DotCache format menu.")
    parser.add_argument("--manifest", required=True, help="Path to a page trace manifest.json emitted by the capture harness.")
    parser.add_argument("--output", default=None, help="Optional JSON output path.")
    parser.add_argument("--markdown-output", default=None, help="Optional markdown table output path.")
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
    result = run_oracle_batch_replay(
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
    payload = result.to_dict()
    rendered = json.dumps(payload, sort_keys=True, indent=2)
    print(rendered)

    if args.output is not None:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered + "\n", encoding="utf-8")
    if args.markdown_output is not None:
        markdown_path = Path(args.markdown_output)
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        markdown_path.write_text(result.summary_table_markdown + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
