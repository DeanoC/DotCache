#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a LongBench failure workbook for systems misses versus exact.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--markdown-output", required=True)
    parser.add_argument("--json-output", required=True)
    parser.add_argument("--title", default="Qwen LongBench Failure Workbook")
    return parser.parse_args()


def _load_rows(path: Path, *, measurement_kind: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        payload = json.loads(stripped)
        if payload.get("measurement_kind") == measurement_kind:
            rows.append(payload)
    return rows


def _fmt(value: object, digits: int = 3) -> str:
    if value is None:
        return "-"
    return f"{float(value):.{digits}f}"


def _markdown_table(rows: list[list[str]]) -> str:
    header = rows[0]
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows[1:])
    return "\n".join(lines)


def _effective_bytes_per_token(row: dict[str, Any]) -> float | None:
    direct = row.get("effective_bytes_per_token")
    if direct is not None:
        return float(direct)
    resident = row.get("resident_bytes")
    prompt_length = row.get("prompt_length")
    if resident is None or prompt_length in (None, 0):
        return None
    return float(resident) / float(prompt_length)


def _group_key(row: dict[str, Any]) -> tuple[int, str]:
    return (int(row["comparison_max_prompt_tokens"]), str(row["evaluation_prompt_id"]))


def _classify_failure(
    exact_row: dict[str, Any],
    systems_row: dict[str, Any],
) -> tuple[str, str]:
    exact_score = float(exact_row.get("longbench_official_score") or 0.0)
    systems_score = float(systems_row.get("longbench_official_score") or 0.0)
    raw_score = float(systems_row.get("longbench_official_score_raw") or systems_score)
    cleaning_delta = float(systems_row.get("longbench_official_score_cleaning_delta") or 0.0)
    exact_bytes = _effective_bytes_per_token(exact_row)
    systems_bytes = _effective_bytes_per_token(systems_row)
    score_gap = exact_score - systems_score

    if cleaning_delta >= 0.15 or (systems_row.get("longbench_chat_artifact_cleaned") and raw_score + 0.10 < systems_score):
        return (
            "write_format_damage",
            f"cleaning recovered {_fmt(cleaning_delta)} official-score points from raw output",
        )
    if exact_score >= 0.50 and score_gap >= 0.20:
        if exact_bytes is not None and systems_bytes is not None and systems_bytes + 0.25 < exact_bytes:
            return (
                "selection_miss",
                f"systems trails exact by {_fmt(score_gap)} while running at lower effective bytes/token",
            )
        return (
            "selection_miss",
            f"systems trails a strong exact row by {_fmt(score_gap)} without a formatting-only explanation",
        )
    return (
        "downstream_under_attention",
        f"systems still trails exact by {_fmt(score_gap)} after cleaning, but the miss is not explained by formatting alone",
    )


def build_report(rows: list[dict[str, Any]], *, title: str) -> tuple[dict[str, Any], str]:
    grouped: dict[tuple[int, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        grouped[_group_key(row)][str(row["comparison_case"])] = row

    workbook_rows: list[dict[str, Any]] = []
    for (max_prompt_tokens, prompt_id), case_rows in sorted(grouped.items()):
        exact_row = case_rows.get("exact")
        systems_row = case_rows.get("systems")
        if exact_row is None or systems_row is None:
            continue
        exact_score = float(exact_row.get("longbench_official_score") or 0.0)
        systems_score = float(systems_row.get("longbench_official_score") or 0.0)
        if systems_score >= exact_score:
            continue
        classification, reason = _classify_failure(exact_row, systems_row)
        workbook_rows.append(
            {
                "max_prompt_tokens": max_prompt_tokens,
                "evaluation_prompt_id": prompt_id,
                "longbench_dataset": systems_row.get("longbench_dataset"),
                "longbench_task_family": systems_row.get("longbench_task_family"),
                "classification": classification,
                "reason": reason,
                "exact_official_score": exact_score,
                "systems_official_score": systems_score,
                "systems_official_score_raw": systems_row.get("longbench_official_score_raw"),
                "score_gap_vs_exact": exact_score - systems_score,
                "systems_effective_bytes_per_token": _effective_bytes_per_token(systems_row),
                "exact_effective_bytes_per_token": _effective_bytes_per_token(exact_row),
                "systems_generated_text": systems_row.get("longbench_generated_text_cleaned")
                or systems_row.get("longbench_generated_text"),
                "exact_generated_text": exact_row.get("longbench_generated_text_cleaned")
                or exact_row.get("longbench_generated_text"),
            }
        )

    summary_counter = Counter(str(row["classification"]) for row in workbook_rows)
    family_counter: Counter[tuple[str, str]] = Counter(
        (str(row.get("longbench_task_family") or "unknown"), str(row["classification"]))
        for row in workbook_rows
    )
    summary_rows = [
        {"classification": classification, "n_rows": count}
        for classification, count in sorted(summary_counter.items())
    ]
    family_rows = [
        {"task_family": family, "classification": classification, "n_rows": count}
        for (family, classification), count in sorted(family_counter.items())
    ]

    workbook_table = [[
        "max_prompt_tokens",
        "prompt",
        "dataset",
        "task_family",
        "classification",
        "exact_score",
        "systems_score",
        "systems_raw_score",
        "gap_vs_exact",
        "reason",
    ]]
    for row in workbook_rows:
        workbook_table.append(
            [
                str(row["max_prompt_tokens"]),
                str(row["evaluation_prompt_id"]),
                str(row.get("longbench_dataset") or ""),
                str(row.get("longbench_task_family") or ""),
                str(row["classification"]),
                _fmt(row["exact_official_score"]),
                _fmt(row["systems_official_score"]),
                _fmt(row.get("systems_official_score_raw")),
                _fmt(row["score_gap_vs_exact"]),
                str(row["reason"]),
            ]
        )

    summary_table = [["classification", "n_rows"]]
    for row in summary_rows:
        summary_table.append([str(row["classification"]), str(row["n_rows"])])

    family_table = [["task_family", "classification", "n_rows"]]
    for row in family_rows:
        family_table.append(
            [str(row["task_family"]), str(row["classification"]), str(row["n_rows"])]
        )

    payload = {
        "rows": workbook_rows,
        "summary_rows": summary_rows,
        "task_family_rows": family_rows,
    }
    markdown = "\n".join(
        [
            f"# {title}",
            "",
            "## Summary",
            "",
            _markdown_table(summary_table),
            "",
            "## Task Family Breakdown",
            "",
            _markdown_table(family_table),
            "",
            "## Workbook",
            "",
            _markdown_table(workbook_table),
        ]
    )
    return payload, markdown


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)
    rows = _load_rows(input_path, measurement_kind="aggregate")
    payload, markdown = build_report(rows, title=str(args.title))
    Path(args.json_output).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    Path(args.markdown_output).write_text(markdown + "\n", encoding="utf-8")
    print(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
