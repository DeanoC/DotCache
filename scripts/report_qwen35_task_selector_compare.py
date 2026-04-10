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
    parser.add_argument(
        "--reference-input",
        default=None,
        help="Optional JSONL from a baseline run whose missing profiles should be merged into the report.",
    )
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


def _merge_missing_profile_rows(
    rows: list[dict[str, Any]],
    reference_rows: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    if not reference_rows:
        return list(rows)
    merged: dict[tuple[str, int, str], dict[str, Any]] = {
        _sample_key(row): row for row in reference_rows
    }
    merged.update({_sample_key(row): row for row in rows})
    return list(merged.values())


def _preferred_profiles(rows: list[dict[str, Any]]) -> list[str]:
    preferred_order = {"dense": 0, "exact": 1, "quality": 2, "systems": 3}
    profile_names = {str(row.get("selector_profile", "")) for row in rows if row.get("selector_profile")}
    return sorted(profile_names, key=lambda name: (preferred_order.get(name, 99), name))


def _generated_token_ids(row: dict[str, Any]) -> list[int]:
    if row.get("task_generated_token_ids") is not None:
        return list(row.get("task_generated_token_ids") or [])
    if row.get("dotcache_generated_ids") is not None:
        return list(row.get("dotcache_generated_ids") or [])
    return list(row.get("dense_generated_ids") or [])


def _decode_ms_per_step(row: dict[str, Any]) -> object:
    if row.get("task_decode_ms_per_step") is not None:
        return row.get("task_decode_ms_per_step")
    if row.get("dotcache_decode_ms_per_step") is not None:
        return row.get("dotcache_decode_ms_per_step")
    return row.get("dense_decode_ms_per_step")


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
        "matches_dense_output",
        "cap_hit",
        "generated_tokens",
        "decode_steps",
        "expected",
        "generated_first_line_cleaned",
        "generated_text_cleaned",
    ]]
    dense_by_group = {
        _group_key(row): row
        for row in ordered
        if str(row.get("selector_profile", "")) == "dense"
    }
    for row in ordered:
        cleaned_text = str(row.get("task_generated_text_cleaned", row.get("task_generated_text", ""))).replace("\n", "\\n")
        generated_tokens = len(_generated_token_ids(row))
        decode_steps = row.get("decode_steps")
        cap_hit = (
            decode_steps is not None
            and generated_tokens > 0
            and int(generated_tokens) >= int(decode_steps)
        )
        first_line = cleaned_text.split("\\n", 1)[0] if cleaned_text else ""
        matches_dense = _matches_dense_output(row, dense_by_group.get(_group_key(row), {}))
        table.append(
            [
                str(row["task_name"]),
                str(int(row["prompt_length_requested"])),
                str(row["selector_profile"]),
                _fmt_float(row.get("task_metric_value")),
                _fmt_float(matches_dense),
                "yes" if cap_hit else "no",
                str(int(generated_tokens)),
                "-" if decode_steps is None else str(int(decode_steps)),
                str(row.get("task_expected_answer", "")).replace("\n", "\\n"),
                first_line.replace("\n", "\\n"),
                cleaned_text,
            ]
        )
    return _markdown_table(table)


def build_report(
    rows: list[dict[str, Any]],
    *,
    title: str,
    trial_rows: list[dict[str, Any]] | None = None,
    reference_rows: list[dict[str, Any]] | None = None,
    reference_trial_rows: list[dict[str, Any]] | None = None,
) -> tuple[dict[str, Any], str]:
    rows = _merge_missing_profile_rows(rows, reference_rows)
    if trial_rows is not None or reference_trial_rows is not None:
        trial_rows = _merge_missing_profile_rows(trial_rows or [], reference_trial_rows)

    by_group: dict[tuple[str, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        by_group[_group_key(row)][str(row["selector_profile"])] = row
    profiles = _preferred_profiles(rows)

    summary_rows: list[dict[str, Any]] = []
    markdown_header = ["task", "prompt_length"]
    for profile_name in profiles:
        markdown_header.extend([f"{profile_name}_success", f"{profile_name}_matches_dense_output", f"{profile_name}_decode_ms"])
    markdown_header.extend(["quality_vs_dense_speedup", "systems_vs_quality_speedup", "quality_ppl_ratio", "systems_ppl_ratio", "quality_rmse", "systems_rmse"])
    markdown_rows = [markdown_header]
    for (task_name, prompt_length), grouped in sorted(by_group.items(), key=lambda item: (item[0][0], item[0][1])):
        dense = grouped.get("dense", {})
        exact = grouped.get("exact", {})
        quality = grouped.get("quality", {})
        systems = grouped.get("systems", {})
        dense_decode = float(_decode_ms_per_step(dense) or 0.0)
        quality_decode = float(_decode_ms_per_step(quality) or 0.0)
        systems_decode = float(_decode_ms_per_step(systems) or 0.0)
        quality_vs_dense_speedup = (dense_decode / quality_decode) if dense_decode > 0.0 and quality_decode > 0.0 else None
        speedup = (quality_decode / systems_decode) if quality_decode > 0.0 and systems_decode > 0.0 else None
        row = {
            "task_name": task_name,
            "prompt_length": int(prompt_length),
            "dense_success": float(dense.get("task_metric_value", 0.0) or 0.0),
            "dense_matches_dense_output": 1.0 if dense else None,
            "exact_success": float(exact.get("task_metric_value", 0.0) or 0.0),
            "exact_matches_dense_output": _matches_dense_output(exact, dense),
            "quality_success": float(quality.get("task_metric_value", 0.0) or 0.0),
            "quality_matches_dense_output": _matches_dense_output(quality, dense),
            "systems_success": float(systems.get("task_metric_value", 0.0) or 0.0),
            "systems_matches_dense_output": _matches_dense_output(systems, dense),
            "dense_decode_ms_per_step": dense_decode,
            "exact_decode_ms_per_step": float(_decode_ms_per_step(exact) or 0.0),
            "quality_decode_ms_per_step": quality_decode,
            "systems_decode_ms_per_step": systems_decode,
            "quality_vs_dense_speedup": quality_vs_dense_speedup,
            "systems_vs_quality_speedup": speedup,
            "quality_teacher_forced_perplexity_ratio": quality.get("teacher_forced_perplexity_ratio"),
            "systems_teacher_forced_perplexity_ratio": systems.get("teacher_forced_perplexity_ratio"),
            "quality_teacher_forced_logit_rmse": quality.get("teacher_forced_logit_rmse"),
            "systems_teacher_forced_logit_rmse": systems.get("teacher_forced_logit_rmse"),
        }
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
                _fmt_float(row["quality_teacher_forced_logit_rmse"]),
                _fmt_float(row["systems_teacher_forced_logit_rmse"]),
            ]
        )
        markdown_rows.append(markdown_row)

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
    reference_rows = None
    reference_trial_rows = None
    if args.reference_input:
        reference_path = Path(args.reference_input)
        reference_rows = _load_aggregate_rows(reference_path)
        reference_trial_rows = _load_trial_rows(reference_path)
    payload, markdown = build_report(
        rows,
        title=str(args.title),
        trial_rows=trial_rows,
        reference_rows=reference_rows,
        reference_trial_rows=reference_trial_rows,
    )
    Path(args.json_output).write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    Path(args.markdown_output).write_text(markdown + "\n", encoding="utf-8")
    print(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
