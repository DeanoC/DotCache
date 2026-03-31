#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any


CASE_ORDER = ["exact", "shortlist_base", "shortlist_l23_ctx", "shortlist_topk8", "shortlist_quality_profile"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize a hotpotqa-focused LongBench shortlist scorer diagnostic artifact."
    )
    parser.add_argument("input", help="JSONL artifact produced by the hotpot diagnostic wrapper.")
    parser.add_argument("--prompt-id", default="hotpot_case_order")
    parser.add_argument("--dataset", default="hotpotqa")
    parser.add_argument("--top-missed-pages", type=int, default=8)
    parser.add_argument("--markdown-output", default=None)
    return parser.parse_args()


def _load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        stripped = raw_line.strip()
        if not stripped:
            continue
        candidate = json.loads(stripped)
        if isinstance(candidate, dict):
            rows.append(candidate)
    return rows


def _case_sort_key(case: str) -> tuple[int, str]:
    try:
        return (CASE_ORDER.index(case), case)
    except ValueError:
        return (len(CASE_ORDER), case)


def _fmt_float(value: float | None, digits: int = 3) -> str:
    if value is None:
        return "-"
    return f"{value:.{digits}f}"


def _truncate_text(text: str, limit: int = 120) -> str:
    compact = " ".join(str(text).split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3] + "..."


def _layer_metric(row: dict[str, Any], key: str, layer_id: int) -> float | None:
    values = row.get(key)
    if not isinstance(values, dict):
        return None
    raw = values.get(str(layer_id))
    if raw is None:
        raw = values.get(layer_id)
    if raw is None:
        return None
    return float(raw)


def _scorer_layer_records(row: dict[str, Any]) -> list[dict[str, Any]]:
    records = row.get("scorer_layer_records")
    if not isinstance(records, list):
        return []
    return [record for record in records if isinstance(record, dict)]


def _collect_missed_pages(row: dict[str, Any]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for record in _scorer_layer_records(row):
        for group in record.get("groups", []):
            if not isinstance(group, dict):
                continue
            page_ranges = group.get("missed_exact_page_ranges") or group.get("scorer_missed_exact_page_ranges") or []
            for page_range in page_ranges:
                if not isinstance(page_range, dict):
                    continue
                start = page_range.get("token_start")
                end = page_range.get("token_end")
                if start is None or end is None:
                    continue
                counts[f"{int(start)}:{int(end)}"] += 1
    return counts


def _layer_rows(row: dict[str, Any]) -> list[list[str]]:
    rows: list[list[str]] = []
    for record in sorted(_scorer_layer_records(row), key=lambda item: int(item.get("layer_id", -1))):
        layer_id = int(record.get("layer_id", -1))
        groups = [group for group in record.get("groups", []) if isinstance(group, dict)]
        if not groups:
            continue
        exact_recall = [float(group["exact_top_recall"]) for group in groups if group.get("exact_top_recall") is not None]
        approx_recall = [
            float(group["approx_exact_top_recall"]) for group in groups if group.get("approx_exact_top_recall") is not None
        ]
        rank_corr = [float(group["score_rank_correlation"]) for group in groups if group.get("score_rank_correlation") is not None]
        value_corr = [
            float(group["score_value_correlation"]) for group in groups if group.get("score_value_correlation") is not None
        ]
        rows.append(
            [
                str(layer_id),
                str(len(groups)),
                _fmt_float(mean(exact_recall) if exact_recall else None),
                _fmt_float(mean(approx_recall) if approx_recall else None),
                _fmt_float(mean(rank_corr) if rank_corr else None),
                _fmt_float(mean(value_corr) if value_corr else None),
                str(sum(len(group.get("missed_exact_page_ranges") or group.get("scorer_missed_exact_page_ranges") or []) for group in groups)),
            ]
        )
    return rows


def _render(rows: list[dict[str, Any]], *, prompt_id: str, dataset: str, top_missed_pages: int) -> str:
    matching_rows = [
        row
        for row in rows
        if row.get("evaluation_prompt_id") == prompt_id and row.get("longbench_dataset") == dataset
    ]
    if not matching_rows:
        raise SystemExit(f"no rows found for prompt_id={prompt_id!r} dataset={dataset!r}")

    ordered_rows = sorted(matching_rows, key=lambda row: _case_sort_key(str(row.get("runner_case", ""))))
    prompt_row_index = ordered_rows[0].get("longbench_row_index", "-")

    lines = [
        "# Qwen3.5 CUDA LongBench Hotpot Diagnostic",
        "",
        f"Prompt id: `{prompt_id}`",
        f"Dataset: `{dataset}` row `{prompt_row_index}`",
        "",
        "## Case Summary",
        "",
        "| Case | QA F1 | Exact match | Decode ms/step | Worst layer | Layer 23 exact recall | Layer 23 rank corr | Generated text |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in ordered_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("runner_case", "unknown")),
                    _fmt_float(float(row.get("longbench_qa_f1_max", 0.0))),
                    _fmt_float(1.0 if row.get("longbench_answer_exact_match") else 0.0),
                    _fmt_float(float(row.get("dotcache_decode_ms_per_step", 0.0)), digits=2),
                    str(row.get("scorer_worst_layer_id", "-")),
                    _fmt_float(_layer_metric(row, "scorer_exact_top_recall_mean_by_layer", 23)),
                    _fmt_float(_layer_metric(row, "scorer_rank_correlation_mean_by_layer", 23)),
                    _truncate_text(row.get("longbench_generated_text", "")),
                ]
            )
            + " |"
        )

    for row in ordered_rows:
        case = str(row.get("runner_case", "unknown"))
        layer_rows = _layer_rows(row)
        if not layer_rows:
            continue
        lines.extend(
            [
                "",
                f"## {case} Layers",
                "",
                "| Layer | Groups | Mean exact recall | Mean approx recall | Mean rank corr | Mean value corr | Missed exact pages |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        lines.extend("| " + " | ".join(layer_row) + " |" for layer_row in layer_rows)

        missed_pages = _collect_missed_pages(row)
        if missed_pages:
            lines.extend(
                [
                    "",
                    f"Top repeatedly missed exact pages for `{case}`:",
                ]
            )
            for page_range, count in missed_pages.most_common(top_missed_pages):
                lines.append(f"- `{page_range}` missed `{count}` times")

    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    rows = _load_rows(Path(args.input))
    markdown = _render(rows, prompt_id=args.prompt_id, dataset=args.dataset, top_missed_pages=args.top_missed_pages)
    if args.markdown_output:
        out = Path(args.markdown_output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(markdown, encoding="utf-8")
    print(markdown, end="")


if __name__ == "__main__":
    main()
