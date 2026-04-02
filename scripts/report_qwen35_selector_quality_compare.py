#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare exact, quality-profile, and systems-profile Qwen selector quality runs.")
    parser.add_argument("--exact", required=True, help="Path to exact JSONL artifact.")
    parser.add_argument("--quality", required=True, help="Path to learned quality-profile JSONL artifact.")
    parser.add_argument("--systems", required=True, help="Path to learned systems-profile JSONL artifact.")
    parser.add_argument("--markdown-output", default=None, help="Optional markdown output path.")
    parser.add_argument("--json-output", default=None, help="Optional json output path.")
    return parser.parse_args()


def _load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw_line in path.read_bytes().splitlines():
        stripped = raw_line.decode("utf-8", errors="ignore").replace("\x00", "").strip()
        if not stripped:
            continue
        payload = json.loads(stripped)
        if isinstance(payload, dict):
            rows.append(payload)
    exact_length_rows = [row for row in rows if row.get("prompt_mode") == "exact_length"]
    if exact_length_rows:
        rows = exact_length_rows
    aggregate_rows = [row for row in rows if row.get("measurement_kind") == "aggregate"]
    if aggregate_rows:
        rows = aggregate_rows
    return sorted(rows, key=lambda row: int(row.get("prompt_length") or 0))


def _fmt(value: float | None, digits: int = 3) -> str:
    if value is None:
        return "-"
    return f"{value:.{digits}f}"


def _row_summary(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "prompt_length": int(row.get("prompt_length") or 0),
        "prefill_ms": float(row.get("dotcache_prefill_ms") or 0.0),
        "decode_ms_per_step": float(row.get("dotcache_decode_ms_per_step") or 0.0),
        "decode_ms_per_step_p95": float(
            row.get("dotcache_decode_ms_per_step_p95") or row.get("dotcache_decode_ms_per_step") or 0.0
        ),
        "teacher_forced_token_agreement_rate": float(row.get("teacher_forced_token_agreement_rate") or 0.0),
        "teacher_forced_logit_rmse": float(row.get("teacher_forced_logit_rmse") or 0.0),
        "teacher_forced_logit_max_abs_error": float(row.get("teacher_forced_logit_max_abs_error") or 0.0),
        "replay_context_max_abs_error": float(row.get("replay_context_max_abs_error") or 0.0),
        "replay_output_max_abs_error": float(row.get("replay_output_max_abs_error") or 0.0),
        "learned_page_selector_profile": row.get("learned_page_selector_profile"),
        "learned_page_selector_logit_offset": float(row.get("learned_page_selector_logit_offset") or 0.0),
        "m3_fraction": float(
            (float(row.get("m3_pages") or 0.0) / max(float(row.get("total_static_pages") or 0.0), 1.0))
            if row.get("m3_pages") is not None
            else 0.0
        ),
    }


def _build_report(args: argparse.Namespace) -> dict[str, Any]:
    variants = {
        "exact": _load_rows(Path(args.exact)),
        "quality": _load_rows(Path(args.quality)),
        "systems": _load_rows(Path(args.systems)),
    }
    rows_by_context: dict[int, dict[str, dict[str, Any]]] = {}
    for variant_name, rows in variants.items():
        for row in rows:
            rows_by_context.setdefault(int(row.get("prompt_length") or 0), {})[variant_name] = _row_summary(row)
    comparisons: list[dict[str, Any]] = []
    for context in sorted(rows_by_context):
        variant_rows = rows_by_context[context]
        exact = variant_rows.get("exact")
        quality = variant_rows.get("quality")
        systems = variant_rows.get("systems")
        comparison: dict[str, Any] = {
            "prompt_length": context,
            "variants": variant_rows,
        }
        if exact and quality:
            comparison["quality_vs_exact_decode_speedup"] = float(exact["decode_ms_per_step"] / max(quality["decode_ms_per_step"], 1e-8))
        if exact and systems:
            comparison["systems_vs_exact_decode_speedup"] = float(exact["decode_ms_per_step"] / max(systems["decode_ms_per_step"], 1e-8))
        if quality and systems:
            comparison["systems_vs_quality_decode_speedup"] = float(quality["decode_ms_per_step"] / max(systems["decode_ms_per_step"], 1e-8))
            comparison["systems_minus_quality_token_agreement"] = float(
                systems["teacher_forced_token_agreement_rate"] - quality["teacher_forced_token_agreement_rate"]
            )
            comparison["systems_minus_quality_logit_rmse"] = float(
                systems["teacher_forced_logit_rmse"] - quality["teacher_forced_logit_rmse"]
            )
        comparisons.append(comparison)
    return {
        "inputs": {
            "exact": args.exact,
            "quality": args.quality,
            "systems": args.systems,
        },
        "comparisons": comparisons,
    }


def _render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Qwen3.5 Selector Quality Compare",
        "",
        "## Metrics",
        "",
        "| Context | Variant | Decode ms/step | Decode p95 | Prefill ms | Token agreement | Logit RMSE | Logit max abs | Replay ctx max abs | Replay out max abs | M3 frac | Profile | Logit offset |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |",
    ]
    for comparison in report["comparisons"]:
        context = int(comparison["prompt_length"])
        for variant_name in ("exact", "quality", "systems"):
            row = comparison["variants"].get(variant_name)
            if row is None:
                continue
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(context),
                        variant_name,
                        _fmt(row["decode_ms_per_step"]),
                        _fmt(row["decode_ms_per_step_p95"]),
                        _fmt(row["prefill_ms"]),
                        _fmt(row["teacher_forced_token_agreement_rate"]),
                        _fmt(row["teacher_forced_logit_rmse"]),
                        _fmt(row["teacher_forced_logit_max_abs_error"]),
                        _fmt(row["replay_context_max_abs_error"]),
                        _fmt(row["replay_output_max_abs_error"]),
                        _fmt(100.0 * row["m3_fraction"], digits=2),
                        str(row.get("learned_page_selector_profile") or "-"),
                        _fmt(row["learned_page_selector_logit_offset"]),
                    ]
                )
                + " |"
            )
    lines.extend(
        [
            "",
            "## Tradeoff",
            "",
            "| Context | Quality vs Exact speedup | Systems vs Exact speedup | Systems vs Quality speedup | Systems - Quality token agreement | Systems - Quality logit RMSE |",
            "| ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for comparison in report["comparisons"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(int(comparison["prompt_length"])),
                    _fmt(comparison.get("quality_vs_exact_decode_speedup")),
                    _fmt(comparison.get("systems_vs_exact_decode_speedup")),
                    _fmt(comparison.get("systems_vs_quality_decode_speedup")),
                    _fmt(comparison.get("systems_minus_quality_token_agreement")),
                    _fmt(comparison.get("systems_minus_quality_logit_rmse")),
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    report = _build_report(args)
    markdown = _render_markdown(report)
    if args.json_output:
        json_path = Path(args.json_output)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    if args.markdown_output:
        markdown_path = Path(args.markdown_output)
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        markdown_path.write_text(markdown, encoding="utf-8")
    print(markdown, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
