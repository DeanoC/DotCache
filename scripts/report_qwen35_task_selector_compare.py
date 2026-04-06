#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize the compact Qwen task selector comparison suite.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--markdown-output", required=True)
    parser.add_argument("--json-output", required=True)
    parser.add_argument("--title", default="Qwen Task Selector Compare")
    return parser.parse_args()


def _load_aggregate_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        payload = json.loads(stripped)
        if payload.get("measurement_kind") == "aggregate":
            rows.append(payload)
    if not rows:
        raise SystemExit(f"no aggregate rows found in {path}")
    return rows


def _load_trial_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        payload = json.loads(stripped)
        if payload.get("measurement_kind") == "trial":
            rows.append(payload)
    return rows


def _fmt_float(value: object) -> str:
    if value is None:
        return "-"
    return f"{float(value):.3f}"


def _group_key(row: dict[str, Any]) -> tuple[str, int]:
    return (str(row["task_name"]), int(row["prompt_length_requested"]))


def _markdown_table(rows: list[list[str]]) -> str:
    header = rows[0]
    body = rows[1:]
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in body)
    return "\n".join(lines)


def _sample_key(row: dict[str, Any]) -> tuple[str, int, str]:
    return (
        str(row["task_name"]),
        int(row["prompt_length_requested"]),
        str(row["selector_profile"]),
    )


def _sample_output_table(rows: list[dict[str, Any]]) -> str:
    samples: dict[tuple[str, int, str], dict[str, Any]] = {}
    for row in rows:
        key = _sample_key(row)
        current = samples.get(key)
        if current is None or int(row.get("measurement_index", 10**9)) < int(current.get("measurement_index", 10**9)):
            samples[key] = row

    ordered = sorted(samples.values(), key=lambda row: (str(row["task_name"]), int(row["prompt_length_requested"]), str(row["selector_profile"])))
    table = [[
        "task",
        "prompt_length",
        "profile",
        "success",
        "expected",
        "generated_first_line",
        "generated_text",
    ]]
    for row in ordered:
        generated_text = str(row.get("task_generated_text", "")).replace("\n", "\\n")
        first_line = str(row.get("task_generated_first_line", generated_text.split("\\n", 1)[0] if generated_text else ""))
        table.append(
            [
                str(row["task_name"]),
                str(int(row["prompt_length_requested"])),
                str(row["selector_profile"]),
                _fmt_float(row.get("task_metric_value")),
                str(row.get("task_expected_answer", "")).replace("\n", "\\n"),
                first_line.replace("\n", "\\n"),
                generated_text,
            ]
        )
    return _markdown_table(table)


def build_report(
    rows: list[dict[str, Any]],
    *,
    title: str,
    trial_rows: list[dict[str, Any]] | None = None,
) -> tuple[dict[str, Any], str]:
    by_group: dict[tuple[str, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        by_group[_group_key(row)][str(row["selector_profile"])] = row

    summary_rows: list[dict[str, Any]] = []
    markdown_rows = [[
        "task",
        "prompt_length",
        "exact_success",
        "quality_success",
        "systems_success",
        "exact_decode_ms",
        "quality_decode_ms",
        "systems_decode_ms",
        "systems_vs_quality_speedup",
        "quality_ppl_ratio",
        "systems_ppl_ratio",
        "quality_rmse",
        "systems_rmse",
    ]]
    for (task_name, prompt_length), grouped in sorted(by_group.items(), key=lambda item: (item[0][0], item[0][1])):
        exact = grouped.get("exact", {})
        quality = grouped.get("quality", {})
        systems = grouped.get("systems", {})
        quality_decode = float(quality.get("dotcache_decode_ms_per_step", 0.0) or 0.0)
        systems_decode = float(systems.get("dotcache_decode_ms_per_step", 0.0) or 0.0)
        speedup = (quality_decode / systems_decode) if quality_decode > 0.0 and systems_decode > 0.0 else None
        row = {
            "task_name": task_name,
            "prompt_length": int(prompt_length),
            "exact_success": float(exact.get("task_metric_value", 0.0) or 0.0),
            "quality_success": float(quality.get("task_metric_value", 0.0) or 0.0),
            "systems_success": float(systems.get("task_metric_value", 0.0) or 0.0),
            "exact_decode_ms_per_step": float(exact.get("dotcache_decode_ms_per_step", 0.0) or 0.0),
            "quality_decode_ms_per_step": quality_decode,
            "systems_decode_ms_per_step": systems_decode,
            "systems_vs_quality_speedup": speedup,
            "quality_teacher_forced_perplexity_ratio": quality.get("teacher_forced_perplexity_ratio"),
            "systems_teacher_forced_perplexity_ratio": systems.get("teacher_forced_perplexity_ratio"),
            "quality_teacher_forced_logit_rmse": quality.get("teacher_forced_logit_rmse"),
            "systems_teacher_forced_logit_rmse": systems.get("teacher_forced_logit_rmse"),
        }
        summary_rows.append(row)
        markdown_rows.append(
            [
                task_name,
                str(prompt_length),
                _fmt_float(row["exact_success"]),
                _fmt_float(row["quality_success"]),
                _fmt_float(row["systems_success"]),
                _fmt_float(row["exact_decode_ms_per_step"]),
                _fmt_float(row["quality_decode_ms_per_step"]),
                _fmt_float(row["systems_decode_ms_per_step"]),
                _fmt_float(row["systems_vs_quality_speedup"]),
                _fmt_float(row["quality_teacher_forced_perplexity_ratio"]),
                _fmt_float(row["systems_teacher_forced_perplexity_ratio"]),
                _fmt_float(row["quality_teacher_forced_logit_rmse"]),
                _fmt_float(row["systems_teacher_forced_logit_rmse"]),
            ]
        )

    markdown_sections = [
        f"# {title}",
        "",
        _markdown_table(markdown_rows),
    ]
    if trial_rows:
        markdown_sections.extend(
            [
                "",
                "## Sample Outputs",
                "",
                _sample_output_table(trial_rows),
            ]
        )
    return {"rows": summary_rows}, "\n".join(markdown_sections)


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)
    rows = _load_aggregate_rows(input_path)
    trial_rows = _load_trial_rows(input_path)
    payload, markdown = build_report(rows, title=str(args.title), trial_rows=trial_rows)
    Path(args.json_output).write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    Path(args.markdown_output).write_text(markdown + "\n", encoding="utf-8")
    print(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
