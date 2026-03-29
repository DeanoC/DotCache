#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize Qwen3.5 scorer-diagnostic JSONL artifacts for layer 23 and the worst layer."
    )
    parser.add_argument("--input", action="append", required=True, help="One or more scorer-diagnostic JSONL files.")
    return parser.parse_args()


def _load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        stripped = raw_line.strip()
        if not stripped.startswith("{"):
            continue
        candidate = json.loads(stripped)
        if isinstance(candidate, dict):
            rows.append(candidate)
    return rows


def _layer_records_by_id(row: dict[str, Any]) -> dict[int, dict[str, Any]]:
    records = row.get("scorer_layer_records")
    if not isinstance(records, list):
        return {}
    result: dict[int, dict[str, Any]] = {}
    for record in records:
        if isinstance(record, dict) and record.get("layer_id") is not None:
            result[int(record["layer_id"])] = record
    return result


def _group_mean(layer_record: dict[str, Any] | None, key: str) -> float | None:
    if not isinstance(layer_record, dict):
        return None
    groups = layer_record.get("groups")
    if not isinstance(groups, list) or not groups:
        return None
    values = [float(group[key]) for group in groups if isinstance(group, dict) and group.get(key) is not None]
    if not values:
        return None
    return float(mean(values))


def _format_float(value: float | None, digits: int = 3) -> str:
    if value is None:
        return "-"
    return f"{value:.{digits}f}"


def _summarize_row(path: Path, row: dict[str, Any]) -> list[str]:
    by_layer = _layer_records_by_id(row)
    worst_layer = row.get("scorer_worst_layer_id")
    worst_layer_id = int(worst_layer) if worst_layer is not None else None
    layer23 = by_layer.get(23)
    worst = by_layer.get(worst_layer_id) if worst_layer_id is not None else None
    return [
        path.name,
        str(row.get("prompt_length", "-")),
        str(worst_layer if worst_layer is not None else "-"),
        _format_float(_group_mean(layer23, "exact_top_recall")),
        _format_float(_group_mean(layer23, "approx_boundary_margin_normalized")),
        _format_float(_group_mean(layer23, "score_rank_correlation")),
        _format_float(_group_mean(layer23, "score_value_correlation")),
        _format_float(_group_mean(worst, "exact_top_recall")),
        _format_float(_group_mean(worst, "score_rank_correlation")),
        _format_float(_group_mean(worst, "score_value_correlation")),
    ]


def main() -> None:
    args = parse_args()
    rows_out: list[list[str]] = []
    for raw_path in args.input:
        path = Path(raw_path)
        for row in _load_rows(path):
            if row.get("scorer_diagnostic") is not True:
                continue
            rows_out.append(_summarize_row(path, row))
    print(
        "| File | Context | Worst Layer | Layer23 Exact Recall | Layer23 Margin Norm | Layer23 Rank Corr | Layer23 Value Corr | Worst Exact Recall | Worst Rank Corr | Worst Value Corr |"
    )
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in rows_out:
        print("| " + " | ".join(row) + " |")


if __name__ == "__main__":
    main()
