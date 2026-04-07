#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize the compact Llama task selector comparison suite.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--markdown-output", required=True)
    parser.add_argument("--json-output", required=True)
    parser.add_argument("--title", default="Llama 3.2 3B Task Selector Compare")
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
    if measurement_kind == "aggregate" and not rows:
        raise SystemExit(f"no aggregate rows found in {path}")
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


def _preferred_profiles(rows: list[dict[str, Any]]) -> list[str]:
    preferred_order = {"dense": 0, "exact": 1, "quality": 2, "systems": 3}
    profile_names = {str(row.get("selector_profile", "")) for row in rows if row.get("selector_profile")}
    return sorted(profile_names, key=lambda name: (preferred_order.get(name, 99), name))


def _generated_token_ids(row: dict[str, Any]) -> list[int]:
    if row.get("task_generated_token_ids") is not None:
        return list(row.get("task_generated_token_ids") or [])
    if row.get("selector_profile") == "dense":
        return list(row.get("dense_generated_ids") or [])
    return list(row.get("dotcache_generated_ids") or [])


def _decode_ms_per_step(row: dict[str, Any]) -> object:
    if row.get("task_decode_ms_per_step") is not None:
        return row.get("task_decode_ms_per_step")
    if row.get("selector_profile") == "dense":
        return row.get("dense_decode_ms_per_step")
    return row.get("decode_ms_per_step")


def _dense_match_value(row: dict[str, Any]) -> str:
    cleaned = row.get("task_generated_text_cleaned", row.get("task_generated_text", ""))
    cleaned_text = str(cleaned).strip()
    if cleaned_text:
        return cleaned_text
    if row.get("task_generated_value") not in (None, ""):
        return str(row.get("task_generated_value")).strip()
    return ""


def _matches_dense_output(row: dict[str, Any], dense_row: dict[str, Any]) -> object:
    if not row or not dense_row:
        return None
    return float(_dense_match_value(row) == _dense_match_value(dense_row))


def _sample_output_table(rows: list[dict[str, Any]]) -> str:
    samples: dict[tuple[str, int, str], dict[str, Any]] = {}
    for row in rows:
        key = (str(row["task_name"]), int(row["prompt_length_requested"]), str(row["selector_profile"]))
        current = samples.get(key)
        if current is None or int(row.get("measurement_index", 10**9)) < int(current.get("measurement_index", 10**9)):
            samples[key] = row

    ordered = sorted(samples.values(), key=lambda row: (str(row["task_name"]), int(row["prompt_length_requested"]), str(row["selector_profile"])))
    dense_by_group = {
        _group_key(row): row
        for row in ordered
        if str(row.get("selector_profile", "")) == "dense"
    }
    table = [[
        "task",
        "prompt_length",
        "profile",
        "success",
        "matches_dense_output",
        "generated_tokens",
        "decode_steps",
        "expected",
        "generated_first_line_cleaned",
        "generated_text_cleaned",
    ]]
    for row in ordered:
        cleaned_text = str(row.get("task_generated_text_cleaned", row.get("task_generated_text", ""))).replace("\n", "\\n")
        first_line = cleaned_text.split("\\n", 1)[0] if cleaned_text else ""
        table.append(
            [
                str(row["task_name"]),
                str(int(row["prompt_length_requested"])),
                str(row["selector_profile"]),
                _fmt_float(row.get("task_metric_value")),
                _fmt_float(_matches_dense_output(row, dense_by_group.get(_group_key(row), {}))),
                str(len(_generated_token_ids(row))),
                str(int(row.get("decode_steps", 0) or 0)),
                str(row.get("task_expected_answer", "")).replace("\n", "\\n"),
                first_line,
                cleaned_text,
            ]
        )
    return _markdown_table(table)


def build_report(
    rows: list[dict[str, Any]],
    trial_rows: list[dict[str, Any]] | None = None,
    *,
    title: str = "Llama 3.2 3B Task Selector Compare",
) -> tuple[dict[str, Any], str]:
    by_group: dict[tuple[str, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        by_group[_group_key(row)][str(row["selector_profile"])] = row
    profiles = _preferred_profiles(rows)

    summary_rows: list[dict[str, Any]] = []
    markdown_header = ["task", "prompt_length"]
    for profile_name in profiles:
        markdown_header.extend([f"{profile_name}_success", f"{profile_name}_matches_dense_output", f"{profile_name}_decode_ms"])
    markdown_header.extend(
        [
            "quality_vs_dense_speedup",
            "systems_vs_quality_speedup",
            "quality_ppl_ratio",
            "systems_ppl_ratio",
            "quality_logit_max_abs",
            "systems_logit_max_abs",
        ]
    )
    markdown_rows = [markdown_header]
    for (task_name, prompt_length), grouped in sorted(by_group.items(), key=lambda item: (item[0][0], item[0][1])):
        dense = grouped.get("dense", {})
        quality = grouped.get("quality", {})
        systems = grouped.get("systems", {})
        dense_decode = float(_decode_ms_per_step(dense) or 0.0)
        quality_decode = float(_decode_ms_per_step(quality) or 0.0)
        systems_decode = float(_decode_ms_per_step(systems) or 0.0)
        quality_vs_dense = (dense_decode / quality_decode) if dense_decode > 0.0 and quality_decode > 0.0 else None
        systems_vs_quality = (quality_decode / systems_decode) if quality_decode > 0.0 and systems_decode > 0.0 else None
        row = {
            "task_name": task_name,
            "prompt_length": int(prompt_length),
            "quality_vs_dense_speedup": quality_vs_dense,
            "systems_vs_quality_speedup": systems_vs_quality,
            "quality_teacher_forced_perplexity_ratio": quality.get("teacher_forced_perplexity_ratio"),
            "systems_teacher_forced_perplexity_ratio": systems.get("teacher_forced_perplexity_ratio"),
            "quality_teacher_forced_logit_max_abs_error": quality.get("teacher_forced_logit_max_abs_error"),
            "systems_teacher_forced_logit_max_abs_error": systems.get("teacher_forced_logit_max_abs_error"),
        }
        for profile_name in profiles:
            profile_row = grouped.get(profile_name, {})
            row[f"{profile_name}_success"] = float(profile_row.get("task_metric_value", 0.0) or 0.0)
            row[f"{profile_name}_matches_dense_output"] = (
                1.0 if profile_name == "dense" and profile_row else _matches_dense_output(profile_row, dense)
            )
            row[f"{profile_name}_decode_ms_per_step"] = float(_decode_ms_per_step(profile_row) or 0.0)
        summary_rows.append(row)
        markdown_row = [task_name, str(prompt_length)]
        for profile_name in profiles:
            markdown_row.extend(
                [
                    _fmt_float(row.get(f"{profile_name}_success")),
                    _fmt_float(row.get(f"{profile_name}_matches_dense_output")),
                    _fmt_float(row.get(f"{profile_name}_decode_ms_per_step")),
                ]
            )
        markdown_row.extend(
            [
                _fmt_float(row["quality_vs_dense_speedup"]),
                _fmt_float(row["systems_vs_quality_speedup"]),
                _fmt_float(row["quality_teacher_forced_perplexity_ratio"]),
                _fmt_float(row["systems_teacher_forced_perplexity_ratio"]),
                _fmt_float(row["quality_teacher_forced_logit_max_abs_error"]),
                _fmt_float(row["systems_teacher_forced_logit_max_abs_error"]),
            ]
        )
        markdown_rows.append(markdown_row)

    markdown_sections = [
        f"# {title}",
        "",
        _markdown_table(markdown_rows),
    ]
    if trial_rows:
        markdown_sections.extend(["", "## Sample Outputs", "", _sample_output_table(trial_rows)])
    return {"rows": summary_rows}, "\n".join(markdown_sections)


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)
    rows = _load_rows(input_path, measurement_kind="aggregate")
    trial_rows = _load_rows(input_path, measurement_kind="trial")
    payload, markdown = build_report(rows, trial_rows, title=str(args.title))
    Path(args.json_output).write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    Path(args.markdown_output).write_text(markdown + "\n", encoding="utf-8")
    print(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
