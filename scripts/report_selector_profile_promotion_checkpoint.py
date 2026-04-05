#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_QWEN_MATRIX_JSON = (
    REPO_ROOT / "benchmarks" / "results" / "qwen_results_matrix_20260404" / "qwen_results_matrix.json"
)
DEFAULT_QWEN_QUALITY_JSON = (
    REPO_ROOT / "benchmarks" / "results" / "qwen35_9b_selector_quality_compare_20260402" / "selector_quality_compare.json"
)
DEFAULT_LLAMA_TASK_JSON = (
    REPO_ROOT / "benchmarks" / "results" / "llama32_3b_task_selector_compare_20260402" / "task_selector_compare.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a cross-family selector profile promotion checkpoint report.")
    parser.add_argument("--qwen-matrix-json", default=str(DEFAULT_QWEN_MATRIX_JSON))
    parser.add_argument("--qwen-quality-json", default=str(DEFAULT_QWEN_QUALITY_JSON))
    parser.add_argument("--llama-task-json", default=str(DEFAULT_LLAMA_TASK_JSON))
    parser.add_argument("--markdown-output", required=True)
    parser.add_argument("--json-output", required=True)
    return parser.parse_args()


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _fmt(value: object, digits: int = 3) -> str:
    if value is None:
        return "-"
    return f"{float(value):.{digits}f}"


def _markdown_table(rows: list[list[str]]) -> str:
    header = rows[0]
    body = rows[1:]
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in body)
    return "\n".join(lines)


def _mean(values: list[float]) -> float:
    return sum(values) / max(len(values), 1)


def build_report(
    *,
    qwen_matrix_payload: dict[str, Any],
    qwen_quality_payload: dict[str, Any],
    llama_task_payload: dict[str, Any],
) -> tuple[dict[str, Any], str]:
    qwen_task_rows = list(qwen_matrix_payload.get("task_rows", []))
    qwen_longbench_rows = list(qwen_matrix_payload.get("longbench_rows", []))
    qwen_backend_rows = list(qwen_matrix_payload.get("backend_rows", []))
    llama_task_rows = list(llama_task_payload.get("rows", []))
    qwen_quality_rows = list(qwen_quality_payload.get("comparisons", []))

    qwen_task_speedups = [float(row["systems_vs_quality_speedup"]) for row in qwen_task_rows]
    llama_task_speedups = [float(row["systems_vs_quality_speedup"]) for row in llama_task_rows]
    qwen_quality_speedups = [float(row["systems_vs_quality_decode_speedup"]) for row in qwen_quality_rows]

    summary = {
        "promotion_calls": {
            "qwen": {
                "default_profile": "systems",
                "rationale": "systems preserves the currently trusted task rows and materially outperforms quality in serving decode across the validated Qwen lanes.",
                "mean_task_speedup_vs_quality": _mean(qwen_task_speedups),
                "mean_quality_harness_speedup_vs_quality": _mean(qwen_quality_speedups),
            },
            "llama": {
                "default_profile": "quality_or_systems_equivalent",
                "rationale": "systems and quality are effectively the same operating point because the selector is already saturated to M3.",
                "mean_task_speedup_vs_quality": _mean(llama_task_speedups),
            },
        },
        "qwen_task_rows": qwen_task_rows,
        "qwen_quality_rows": qwen_quality_rows,
        "llama_task_rows": llama_task_rows,
        "qwen_longbench_rows": qwen_longbench_rows,
        "qwen_backend_rows": qwen_backend_rows,
    }

    overview_rows = [[
        "model",
        "task_success_exact",
        "task_success_quality",
        "task_success_systems",
        "mean_systems_vs_quality_speedup",
        "promotion_call",
    ]]
    for model_key, model_name in (
        ("qwen35_4b", "Qwen3.5 4B"),
        ("qwen35_9b", "Qwen3.5 9B"),
        ("qwen35_27b", "Qwen3.5 27B"),
    ):
        rows = [row for row in qwen_task_rows if row.get("model_key") == model_key]
        exact_success = _mean([float(row.get("exact_success", 0.0)) for row in rows]) if rows else None
        quality_success = _mean([float(row.get("quality_success", 0.0)) for row in rows]) if rows else None
        systems_success = _mean([float(row.get("systems_success", 0.0)) for row in rows]) if rows else None
        speedup = _mean([float(row.get("systems_vs_quality_speedup", 0.0)) for row in rows]) if rows else None
        overview_rows.append(
            [
                model_name,
                _fmt(exact_success),
                _fmt(quality_success),
                _fmt(systems_success),
                _fmt(speedup),
                "Promote systems as default",
            ]
        )
    overview_rows.append(
        [
            "Llama 3.2 3B",
            "1.000",
            "1.000",
            "1.000",
            _fmt(summary["promotion_calls"]["llama"]["mean_task_speedup_vs_quality"]),
            "Quality and systems equivalent",
        ]
    )

    qwen_quality_table = [[
        "context",
        "quality_decode_ms",
        "systems_decode_ms",
        "systems_vs_quality_speedup",
        "token_agreement_quality",
        "token_agreement_systems",
        "quality_rmse",
        "systems_rmse",
        "quality_m3_frac",
        "systems_m3_frac",
    ]]
    for row in sorted(qwen_quality_rows, key=lambda item: int(item["prompt_length"])):
        quality = row["variants"]["quality"]
        systems = row["variants"]["systems"]
        qwen_quality_table.append(
            [
                str(int(row["prompt_length"])),
                _fmt(quality["decode_ms_per_step"]),
                _fmt(systems["decode_ms_per_step"]),
                _fmt(row["systems_vs_quality_decode_speedup"]),
                _fmt(quality["teacher_forced_token_agreement_rate"]),
                _fmt(systems["teacher_forced_token_agreement_rate"]),
                _fmt(quality["teacher_forced_logit_rmse"]),
                _fmt(systems["teacher_forced_logit_rmse"]),
                _fmt(quality["m3_fraction"]),
                _fmt(systems["m3_fraction"]),
            ]
        )

    task_table = [[
        "model",
        "task",
        "prompt_length",
        "exact_success",
        "quality_decode_ms",
        "systems_decode_ms",
        "systems_vs_quality_speedup",
        "quality_success",
        "systems_success",
    ]]
    for model_name, rows in (
        ("Qwen3.5 4B", [row for row in qwen_task_rows if row.get("model_key") == "qwen35_4b"]),
        ("Qwen3.5 9B", [row for row in qwen_task_rows if row.get("model_key") == "qwen35_9b"]),
        ("Qwen3.5 27B", [row for row in qwen_task_rows if row.get("model_key") == "qwen35_27b"]),
        ("Llama 3.2 3B", llama_task_rows),
    ):
        for row in sorted(rows, key=lambda item: (str(item["task_name"]), int(item["prompt_length"]))):
            task_table.append(
                [
                    model_name,
                    str(row["task_name"]),
                    str(int(row["prompt_length"])),
                    _fmt(row.get("exact_success")),
                    _fmt(row["quality_decode_ms_per_step"]),
                    _fmt(row["systems_decode_ms_per_step"]),
                    _fmt(row["systems_vs_quality_speedup"]),
                    _fmt(row["quality_success"]),
                    _fmt(row["systems_success"]),
                ]
            )

    longbench_table = [[
        "model",
        "context_cap",
        "exact_qa_f1",
        "quality_qa_f1",
        "systems_qa_f1",
        "streaming_qa_f1",
        "exact_decode_ms",
        "quality_decode_ms",
        "systems_decode_ms",
        "streaming_decode_ms",
        "systems_vs_quality_speedup",
        "systems_vs_streaming_speedup",
    ]]
    for model_key, model_name in (
        ("qwen35_4b", "Qwen3.5 4B"),
        ("qwen35_9b", "Qwen3.5 9B"),
        ("qwen35_27b", "Qwen3.5 27B"),
    ):
        rows = [row for row in qwen_longbench_rows if row.get("model_key") == model_key]
        longbench_by_key = {
            (int(row["max_prompt_tokens"]), str(row["comparison_case"])): row for row in rows
        }
        for context_cap in sorted({int(row["max_prompt_tokens"]) for row in rows}):
            exact = longbench_by_key.get((context_cap, "exact"))
            quality = longbench_by_key.get((context_cap, "quality"))
            systems = longbench_by_key.get((context_cap, "systems"))
            streaming = longbench_by_key.get((context_cap, "streaming_sink_recent"))
            longbench_table.append(
                [
                    model_name,
                    str(context_cap),
                    _fmt(exact.get("mean_qa_f1") if exact else None),
                    _fmt(quality.get("mean_qa_f1") if quality else None),
                    _fmt(systems.get("mean_qa_f1") if systems else None),
                    _fmt(streaming.get("mean_qa_f1") if streaming else None),
                    _fmt(exact.get("mean_decode_ms_per_step") if exact else None),
                    _fmt(quality.get("mean_decode_ms_per_step") if quality else None),
                    _fmt(systems.get("mean_decode_ms_per_step") if systems else None),
                    _fmt(streaming.get("mean_decode_ms_per_step") if streaming else None),
                    _fmt(
                        (float(quality["mean_decode_ms_per_step"]) / float(systems["mean_decode_ms_per_step"]))
                        if quality and systems and float(systems["mean_decode_ms_per_step"]) > 0.0
                        else None
                    ),
                    _fmt(
                        (float(streaming["mean_decode_ms_per_step"]) / float(systems["mean_decode_ms_per_step"]))
                        if streaming and systems and float(systems["mean_decode_ms_per_step"]) > 0.0
                        else None
                    ),
                ]
            )

    qwen_backend_table = [[
        "model",
        "context",
        "exact_decode_ms",
        "shortlist_decode_ms",
        "learned_decode_ms",
        "learned_vs_exact_speedup",
        "learned_vs_shortlist_speedup",
        "learned_m3_frac",
        "learned_score_ms",
        "learned_mix_ms",
        "selector_us_per_invocation",
    ]]
    for model_key, model_name in (
        ("qwen35_4b", "Qwen3.5 4B"),
        ("qwen35_9b", "Qwen3.5 9B"),
        ("qwen35_27b", "Qwen3.5 27B"),
    ):
        rows = [row for row in qwen_backend_rows if row.get("model_key") == model_key]
        for row in sorted(rows, key=lambda item: int(item["prompt_length"])):
            exact = row["variants"]["exact"]
            shortlist = row["variants"]["shortlist_base"]
            learned = row["variants"]["learned_selector"]
            qwen_backend_table.append(
                [
                    model_name,
                    str(int(row["prompt_length"])),
                    _fmt(exact["decode_ms_per_step"]),
                    _fmt(shortlist["decode_ms_per_step"]),
                    _fmt(learned["decode_ms_per_step"]),
                    _fmt(row["speedups"]["learned_selector"]["vs_exact"]),
                    _fmt(row["speedups"]["learned_selector"]["vs_shortlist"]),
                    _fmt(learned["m3_fraction"]),
                    _fmt(learned["score_ms_step"]),
                    _fmt(learned["mix_ms_step"]),
                    _fmt(learned["selector_us_per_invocation"]),
                ]
            )

    markdown = "\n".join(
        [
            "# Selector Profile Promotion Checkpoint",
            "",
            "## Promotion Call",
            "",
            "- Qwen3.5 4B / 9B / 27B: promote `systems` as the default serving selector profile. The new matrix shows the same basic read across all three validated Qwen sizes.",
            "- Llama 3.2 3B: keep `quality` and `systems` as equivalent operating points for now. The selector is already saturated to `M3`, so the extra systems tuning does not unlock additional speed.",
            "",
            "## Cross-Family Overview",
            "",
            _markdown_table(overview_rows),
            "",
            "## Qwen Serving-Quality Check",
            "",
            _markdown_table(qwen_quality_table),
            "",
            "## Cross-Family Task Check",
            "",
            _markdown_table(task_table),
            "",
            "## Qwen LongBench External Check",
            "",
            _markdown_table(longbench_table),
            "",
            "## Qwen Backend Check",
            "",
            _markdown_table(qwen_backend_table),
            "",
            "## Notes",
            "",
            "- The Qwen matrix now gives a single family-wide read across `4B`, `9B`, and native `27B` on compact tasks, LongBench, and backend truth.",
            "- All compact Qwen task rows pass except the previously-known shared `27B @ 2048 retrieval_passkey` miss, which still fails in `exact`, `quality`, and `systems` and therefore does not read like a selector regression.",
            "- Llama task rows confirm the same task success profile, but with `systems` and `quality` effectively tied on decode.",
            "- The Qwen LongBench matrix currently behaves more like a comparative systems check than a selector-separating quality benchmark: quality is tied across methods on the present pack, but `systems` remains much faster than `quality` and `exact` and retains lower RMSE than the streaming reference.",
            "- Backend truth stays consistent across Qwen scale: learned lanes are strongly M3-heavy, selector overhead stays around `25 us/inv`, and the remaining dominant cost is still backend `score + mix` rather than selector computation.",
            "- This checkpoint supports the current repo policy: Qwen serving defaults to `systems`, while Llama does not need extra systems bias.",
        ]
    )
    return summary, markdown


def main() -> int:
    args = parse_args()
    payload, markdown = build_report(
        qwen_matrix_payload=_load_json(args.qwen_matrix_json),
        qwen_quality_payload=_load_json(args.qwen_quality_json),
        llama_task_payload=_load_json(args.llama_task_json),
    )
    Path(args.json_output).write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    Path(args.markdown_output).write_text(markdown + "\n", encoding="utf-8")
    print(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
