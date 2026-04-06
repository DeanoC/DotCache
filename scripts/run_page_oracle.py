#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from dotcache.config import DotCacheConfig
from dotcache.page_oracle import (
    OracleThresholds,
    PageTraceRecord,
    load_page_trace,
    run_oracle_replay,
    save_page_trace,
)
from dotcache.planner import parse_page_mode_token


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate one captured page against a candidate DotCache format menu.")
    parser.add_argument("--input", required=True, help="Path to a .npy tensor, a .npz with `values`, or a saved page trace .npz.")
    parser.add_argument("--output", default=None, help="Optional JSON output path.")
    parser.add_argument("--save-trace", default=None, help="Optional path to save the normalized page trace.")
    parser.add_argument("--source", default="manual")
    parser.add_argument("--kind", choices=["K", "V"], default=None)
    parser.add_argument("--layer-id", type=int, default=0)
    parser.add_argument("--kv-head-id", type=int, default=0)
    parser.add_argument("--token-start", type=int, default=0)
    parser.add_argument("--token-age", type=int, default=0)
    parser.add_argument("--query-npy", default=None, help="Optional .npy file containing one query vector for local score deltas.")
    parser.add_argument("--group-size", type=int, default=32)
    parser.add_argument("--tokens-per-page", type=int, default=None)
    parser.add_argument("--candidate", action="append", default=[])
    parser.add_argument("--max-mean-abs-error-ratio", type=float, default=0.10)
    parser.add_argument("--max-max-abs-error-ratio", type=float, default=1.00)
    parser.add_argument("--max-token-p95-error-ratio", type=float, default=0.25)
    parser.add_argument("--max-score-max-abs-error", type=float, default=None)
    parser.add_argument("--min-score-topk-agreement", type=float, default=None)
    return parser.parse_args()


def _load_values(path: Path) -> np.ndarray:
    if path.suffix == ".npy":
        return np.asarray(np.load(path), dtype=np.float32)
    with np.load(path, allow_pickle=False) as payload:
        if "values" not in payload:
            raise SystemExit(f"{path} does not contain a `values` array")
        return np.asarray(payload["values"], dtype=np.float32)


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)

    if input_path.suffix == ".npz":
        try:
            record = load_page_trace(input_path)
        except Exception:
            if args.kind is None:
                raise SystemExit("--kind is required when loading a generic tensor npz")
            values = _load_values(input_path)
            query = None if args.query_npy is None else np.asarray(np.load(args.query_npy), dtype=np.float32)
            record = PageTraceRecord(
                source=args.source,
                kind=args.kind,
                layer_id=args.layer_id,
                kv_head_id=args.kv_head_id,
                token_start=args.token_start,
                token_age=args.token_age,
                values=values,
                query=query,
            )
    else:
        if args.kind is None:
            raise SystemExit("--kind is required when loading a raw .npy tensor")
        values = _load_values(input_path)
        query = None if args.query_npy is None else np.asarray(np.load(args.query_npy), dtype=np.float32)
        record = PageTraceRecord(
            source=args.source,
            kind=args.kind,
            layer_id=args.layer_id,
            kv_head_id=args.kv_head_id,
            token_start=args.token_start,
            token_age=args.token_age,
            values=values,
            query=query,
        )

    if args.save_trace is not None:
        save_page_trace(record, args.save_trace)

    candidates = tuple(parse_page_mode_token(token) for token in args.candidate) if args.candidate else None
    thresholds = OracleThresholds(
        max_mean_abs_error_ratio=args.max_mean_abs_error_ratio,
        max_max_abs_error_ratio=args.max_max_abs_error_ratio,
        max_token_p95_error_ratio=args.max_token_p95_error_ratio,
        max_score_max_abs_error=args.max_score_max_abs_error,
        min_score_topk_agreement=args.min_score_topk_agreement,
    )
    base_config = DotCacheConfig(
        head_dim=record.head_dim,
        group_size=args.group_size,
        tokens_per_page=args.tokens_per_page or record.token_count,
    )
    replay = run_oracle_replay(
        record,
        base_config=base_config,
        candidates=candidates,
        thresholds=thresholds,
    )
    payload = replay.to_dict()
    rendered = json.dumps(payload, sort_keys=True, indent=2)
    print(rendered)
    if args.output is not None:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
