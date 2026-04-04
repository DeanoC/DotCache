#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize a manifest-driven Qwen results matrix run.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--markdown-output", required=True)
    parser.add_argument("--json-output", required=True)
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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


def _load_task_rows(output_dir: Path, model_key: str) -> list[dict[str, Any]]:
    path = output_dir / model_key / "task_compare" / "task_selector_compare.json"
    if not path.exists():
        return []
    payload = _load_json(path)
    rows: list[dict[str, Any]] = []
    for row in payload.get("rows", []):
        item = dict(row)
        item["model_key"] = model_key
        rows.append(item)
    return rows


def _load_longbench_rows(output_dir: Path, model_key: str) -> list[dict[str, Any]]:
    path = output_dir / model_key / "longbench" / "longbench_selector_compare.json"
    if not path.exists():
        return []
    payload = _load_json(path)
    rows: list[dict[str, Any]] = []
    for row in payload.get("rows", []):
        item = dict(row)
        item["model_key"] = model_key
        rows.append(item)
    return rows


def _load_backend_rows(output_dir: Path, model_key: str) -> list[dict[str, Any]]:
    path = output_dir / model_key / "backend_truth" / "backend_truth_report.json"
    if not path.exists():
        return []
    payload = _load_json(path)
    rows: list[dict[str, Any]] = []
    for row in payload.get("comparisons", []):
        item = dict(row)
        item["model_key"] = model_key
        rows.append(item)
    return rows


def _pretty_model_name(model: dict[str, Any]) -> str:
    return str(model.get("model_id") or model.get("model_key") or "")


def build_report(manifest: dict[str, Any], *, output_dir: Path) -> tuple[dict[str, Any], str]:
    models = list(manifest.get("models", []))
    task_rows: list[dict[str, Any]] = []
    longbench_rows: list[dict[str, Any]] = []
    backend_rows: list[dict[str, Any]] = []

    model_names = {str(model["model_key"]): _pretty_model_name(model) for model in models}
    for model in models:
        model_key = str(model["model_key"])
        task_rows.extend(_load_task_rows(output_dir, model_key))
        longbench_rows.extend(_load_longbench_rows(output_dir, model_key))
        backend_rows.extend(_load_backend_rows(output_dir, model_key))

    overview_table = [[
        "model",
        "task_rows",
        "longbench_rows",
        "backend_rows",
    ]]
    for model in models:
        model_key = str(model["model_key"])
        overview_table.append(
            [
                model_names[model_key],
                str(sum(1 for row in task_rows if row["model_key"] == model_key)),
                str(sum(1 for row in longbench_rows if row["model_key"] == model_key)),
                str(sum(1 for row in backend_rows if row["model_key"] == model_key)),
            ]
        )

    task_table = [[
        "model",
        "task",
        "context",
        "exact_success",
        "quality_success",
        "systems_success",
        "quality_decode_ms",
        "systems_decode_ms",
        "systems_vs_quality_speedup",
        "quality_rmse",
        "systems_rmse",
    ]]
    for row in sorted(task_rows, key=lambda item: (model_names[item["model_key"]], str(item["task_name"]), int(item["prompt_length"]))):
        task_table.append(
            [
                model_names[row["model_key"]],
                str(row["task_name"]),
                str(int(row["prompt_length"])),
                _fmt(row.get("exact_success")),
                _fmt(row.get("quality_success")),
                _fmt(row.get("systems_success")),
                _fmt(row.get("quality_decode_ms_per_step")),
                _fmt(row.get("systems_decode_ms_per_step")),
                _fmt(row.get("systems_vs_quality_speedup")),
                _fmt(row.get("quality_teacher_forced_logit_rmse")),
                _fmt(row.get("systems_teacher_forced_logit_rmse")),
            ]
        )

    longbench_table = [[
        "model",
        "context_cap",
        "case",
        "exact_match",
        "qa_f1",
        "decode_ms",
        "decode_p95_ms",
        "ppl_ratio",
        "rmse",
    ]]
    for row in sorted(longbench_rows, key=lambda item: (model_names[item["model_key"]], int(item["max_prompt_tokens"]), str(item["comparison_case"]))):
        longbench_table.append(
            [
                model_names[row["model_key"]],
                str(int(row["max_prompt_tokens"])),
                str(row["comparison_case"]),
                _fmt(row.get("mean_exact_match")),
                _fmt(row.get("mean_qa_f1")),
                _fmt(row.get("mean_decode_ms_per_step")),
                _fmt(row.get("p95_decode_ms_per_step")),
                _fmt(row.get("mean_teacher_forced_perplexity_ratio")),
                _fmt(row.get("mean_teacher_forced_logit_rmse")),
            ]
        )

    backend_table = [[
        "model",
        "context",
        "exact_decode_ms",
        "shortlist_decode_ms",
        "learned_decode_ms",
        "learned_vs_exact_speedup",
        "learned_vs_shortlist_speedup",
        "learned_m3_frac",
        "selector_us_per_invocation",
        "learned_score_ms",
        "learned_mix_ms",
    ]]
    for row in sorted(backend_rows, key=lambda item: (model_names[item["model_key"]], int(item["prompt_length"]))):
        exact = row["variants"]["exact"]
        shortlist = row["variants"]["shortlist_base"]
        learned = row["variants"]["learned_selector"]
        backend_table.append(
            [
                model_names[row["model_key"]],
                str(int(row["prompt_length"])),
                _fmt(exact.get("decode_ms_per_step")),
                _fmt(shortlist.get("decode_ms_per_step")),
                _fmt(learned.get("decode_ms_per_step")),
                _fmt(row["speedups"]["learned_selector"]["vs_exact"]),
                _fmt(row["speedups"]["learned_selector"]["vs_shortlist"]),
                _fmt(learned.get("m3_fraction")),
                _fmt(learned.get("selector_us_per_invocation")),
                _fmt(learned.get("score_ms_step")),
                _fmt(learned.get("mix_ms_step")),
            ]
        )

    payload = {
        "title": manifest.get("title", "Qwen Results Matrix"),
        "models": models,
        "task_rows": task_rows,
        "longbench_rows": longbench_rows,
        "backend_rows": backend_rows,
    }

    markdown = "\n".join(
        [
            f"# {payload['title']}",
            "",
            "## Coverage",
            "",
            _markdown_table(overview_table),
            "",
            "## Task Compare Matrix",
            "",
            _markdown_table(task_table),
            "",
            "## LongBench Matrix",
            "",
            _markdown_table(longbench_table),
            "",
            "## Backend Truth Matrix",
            "",
            _markdown_table(backend_table),
        ]
    )
    return payload, markdown


def main() -> int:
    args = parse_args()
    manifest = _load_json(Path(args.manifest))
    payload, markdown = build_report(manifest, output_dir=Path(args.output_dir).resolve())
    Path(args.markdown_output).write_text(markdown + "\n", encoding="utf-8")
    Path(args.json_output).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
