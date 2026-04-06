#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize grouped-batch fallback reasons from DotCache benchmark JSONL artifacts."
    )
    parser.add_argument("inputs", nargs="+", help="One or more JSONL result files.")
    return parser.parse_args()


def load_records(path: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if isinstance(payload, dict):
                records.append(payload)
    return records


def format_reason_counts(reason_counts: object) -> str:
    if not isinstance(reason_counts, dict) or not reason_counts:
        return "none"
    parts: list[str] = []
    for key in sorted(reason_counts):
        parts.append(f"{key}={reason_counts[key]}")
    return ", ".join(parts)


def summarize_record(record: dict[str, object]) -> str:
    prompt_length = record.get("prompt_length", "?")
    runner_case = record.get("runner_case", record.get("benchmark", "?"))
    decode_paths = record.get("decode_path_counts", {})
    shortlist_rejections = record.get("execution_shortlist_grouping_rejection_reason_counts", {})
    grouped_rejections = record.get("decode_grouped_batch_rejection_reason_counts", {})
    decode_ms = record.get("dotcache_decode_ms_per_step")
    if isinstance(decode_ms, float):
        decode_ms_text = f"{decode_ms:.2f}"
    else:
        decode_ms_text = str(decode_ms)
    return (
        f"context={prompt_length} case={runner_case} decode_ms_per_step={decode_ms_text} "
        f"paths={format_reason_counts(decode_paths)} "
        f"shortlist_group_reject={format_reason_counts(shortlist_rejections)} "
        f"grouped_fallback={format_reason_counts(grouped_rejections)}"
    )


def main() -> int:
    args = parse_args()
    for raw_input in args.inputs:
        path = Path(raw_input)
        records = load_records(path)
        print(f"# {path}")
        if not records:
            print("no records")
            continue
        for record in records:
            print(summarize_record(record))
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
