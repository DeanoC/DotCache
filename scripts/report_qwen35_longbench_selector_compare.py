#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

EXPECTED_CASES = ("exact", "quality", "systems", "streaming_sink_recent")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize Qwen LongBench selector comparison runs.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--markdown-output", required=True)
    parser.add_argument("--json-output", required=True)
    parser.add_argument("--title", default="Qwen LongBench Selector Compare")
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


def _mean_optional(values: list[object]) -> float | None:
    present = [float(value) for value in values if value is not None]
    if not present:
        return None
    return float(mean(present))


def _markdown_table(rows: list[list[str]]) -> str:
    header = rows[0]
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows[1:])
    return "\n".join(lines)


def _group_key(row: dict[str, Any]) -> tuple[int, str]:
    return (int(row["comparison_max_prompt_tokens"]), str(row["comparison_case"]))


def _prompt_key(row: dict[str, Any]) -> tuple[int, str, str]:
    return (
        int(row["comparison_max_prompt_tokens"]),
        str(row["evaluation_prompt_id"]),
        str(row["comparison_case"]),
    )


def _validate_expected_cases(rows: list[dict[str, Any]]) -> None:
    observed: dict[int, set[str]] = defaultdict(set)
    for row in rows:
        observed[int(row["comparison_max_prompt_tokens"])].add(str(row["comparison_case"]))
    missing_by_bucket: dict[int, list[str]] = {}
    for max_prompt_tokens, cases in sorted(observed.items()):
        missing = [case for case in EXPECTED_CASES if case not in cases]
        if missing:
            missing_by_bucket[max_prompt_tokens] = missing
    if missing_by_bucket:
        details = ", ".join(
            f"{max_prompt_tokens}: {', '.join(missing)}"
            for max_prompt_tokens, missing in missing_by_bucket.items()
        )
        raise SystemExit(f"incomplete aggregate coverage in report input; missing cases by context: {details}")


def _sample_output_table(rows: list[dict[str, Any]]) -> str:
    samples: dict[tuple[int, str, str], dict[str, Any]] = {}
    for row in rows:
        key = _prompt_key(row)
        current = samples.get(key)
        if current is None or int(row.get("measurement_index", 10**9)) < int(current.get("measurement_index", 10**9)):
            samples[key] = row

    ordered = sorted(
        samples.values(),
        key=lambda row: (
            int(row["comparison_max_prompt_tokens"]),
            str(row["evaluation_prompt_id"]),
            str(row["comparison_case"]),
        ),
    )
    table = [[
        "max_prompt_tokens",
        "prompt",
        "dataset",
        "case",
        "exact_match",
        "qa_f1",
        "generated",
    ]]
    for row in ordered:
        table.append(
            [
                str(int(row["comparison_max_prompt_tokens"])),
                str(row.get("evaluation_prompt_id", "")),
                str(row.get("longbench_dataset", "")),
                str(row.get("comparison_case", "")),
                _fmt_float(1.0 if row.get("longbench_answer_exact_match_cleaned") else 0.0),
                _fmt_float(row.get("longbench_qa_f1_max_cleaned")),
                str(row.get("longbench_generated_text_cleaned", "")).replace("\n", "\\n"),
            ]
        )
    return _markdown_table(table)


def build_report(
    rows: list[dict[str, Any]],
    *,
    title: str,
    trial_rows: list[dict[str, Any]] | None = None,
) -> tuple[dict[str, Any], str]:
    _validate_expected_cases(rows)
    grouped: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[_group_key(row)].append(row)

    summary_rows: list[dict[str, Any]] = []
    markdown_rows = [[
        "max_prompt_tokens",
        "case",
        "n_rows",
        "mean_exact_match",
        "mean_qa_f1",
        "mean_decode_ms",
        "p95_decode_ms",
        "mean_ppl_ratio",
        "mean_rmse",
    ]]
    for (max_prompt_tokens, case), case_rows in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1])):
        exact_match = [1.0 if row.get("longbench_answer_exact_match_cleaned") else 0.0 for row in case_rows]
        qa_f1 = [float(row.get("longbench_qa_f1_max_cleaned", 0.0) or 0.0) for row in case_rows]
        decode = [float(row.get("dotcache_decode_ms_per_step", 0.0) or 0.0) for row in case_rows]
        decode_p95 = [float(row.get("dotcache_decode_ms_per_step_p95", row.get("dotcache_decode_ms_per_step", 0.0)) or 0.0) for row in case_rows]
        ppl_ratio = [row.get("teacher_forced_perplexity_ratio") for row in case_rows]
        rmse = [row.get("teacher_forced_logit_rmse") for row in case_rows]
        summary = {
            "max_prompt_tokens": int(max_prompt_tokens),
            "comparison_case": case,
            "n_rows": len(case_rows),
            "mean_exact_match": float(mean(exact_match)),
            "mean_qa_f1": float(mean(qa_f1)),
            "mean_decode_ms_per_step": float(mean(decode)),
            "p95_decode_ms_per_step": float(mean(decode_p95)),
            "mean_teacher_forced_perplexity_ratio": _mean_optional(ppl_ratio),
            "mean_teacher_forced_logit_rmse": _mean_optional(rmse),
        }
        summary_rows.append(summary)
        markdown_rows.append(
            [
                str(summary["max_prompt_tokens"]),
                summary["comparison_case"],
                str(summary["n_rows"]),
                _fmt_float(summary["mean_exact_match"]),
                _fmt_float(summary["mean_qa_f1"]),
                _fmt_float(summary["mean_decode_ms_per_step"]),
                _fmt_float(summary["p95_decode_ms_per_step"]),
                _fmt_float(summary["mean_teacher_forced_perplexity_ratio"]),
                _fmt_float(summary["mean_teacher_forced_logit_rmse"]),
            ]
        )

    tradeoff_rows = [[
        "max_prompt_tokens",
        "quality_vs_exact_speedup",
        "systems_vs_exact_speedup",
        "systems_vs_quality_speedup",
        "streaming_vs_exact_speedup",
        "quality_minus_systems_exact_match",
        "quality_minus_systems_qa_f1",
    ]]
    summary_by_key = {(row["max_prompt_tokens"], row["comparison_case"]): row for row in summary_rows}
    for max_prompt_tokens in sorted({row["max_prompt_tokens"] for row in summary_rows}):
        exact = summary_by_key.get((max_prompt_tokens, "exact"))
        quality = summary_by_key.get((max_prompt_tokens, "quality"))
        systems = summary_by_key.get((max_prompt_tokens, "systems"))
        streaming = summary_by_key.get((max_prompt_tokens, "streaming_sink_recent"))
        tradeoff_rows.append(
            [
                str(max_prompt_tokens),
                _fmt_float(
                    (exact["mean_decode_ms_per_step"] / quality["mean_decode_ms_per_step"])
                    if exact and quality and quality["mean_decode_ms_per_step"] > 0.0
                    else None
                ),
                _fmt_float(
                    (exact["mean_decode_ms_per_step"] / systems["mean_decode_ms_per_step"])
                    if exact and systems and systems["mean_decode_ms_per_step"] > 0.0
                    else None
                ),
                _fmt_float(
                    (quality["mean_decode_ms_per_step"] / systems["mean_decode_ms_per_step"])
                    if quality and systems and systems["mean_decode_ms_per_step"] > 0.0
                    else None
                ),
                _fmt_float(
                    (exact["mean_decode_ms_per_step"] / streaming["mean_decode_ms_per_step"])
                    if exact and streaming and streaming["mean_decode_ms_per_step"] > 0.0
                    else None
                ),
                _fmt_float(
                    (quality["mean_exact_match"] - systems["mean_exact_match"])
                    if quality and systems
                    else None
                ),
                _fmt_float(
                    (quality["mean_qa_f1"] - systems["mean_qa_f1"])
                    if quality and systems
                    else None
                ),
            ]
        )

    markdown_sections = [
        f"# {title}",
        "",
        _markdown_table(markdown_rows),
        "",
        "## Tradeoff",
        "",
        _markdown_table(tradeoff_rows),
    ]
    if trial_rows:
        markdown_sections.extend(["", "## Sample Outputs", "", _sample_output_table(trial_rows)])
    return {"rows": summary_rows}, "\n".join(markdown_sections)


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)
    rows = _load_rows(input_path, measurement_kind="aggregate")
    trial_rows = _load_rows(input_path, measurement_kind="trial")
    payload, markdown = build_report(rows, title=str(args.title), trial_rows=trial_rows)
    Path(args.json_output).write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    Path(args.markdown_output).write_text(markdown + "\n", encoding="utf-8")
    print(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
