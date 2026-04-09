#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


SNAPSHOT_NAME_RE = re.compile(r"prompt(?P<prompt>\d+)_layer(?P<layer>\d+)_kv(?P<kv>\d+)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize MPS paged-attention snapshot sweep results.")
    parser.add_argument("--input", required=True, help="Path to a sweep JSON payload from bench_mps_paged_attention_snapshot_sweep.py.")
    parser.add_argument("--markdown-output", required=True, help="Path to write the markdown report.")
    parser.add_argument("--json-output", required=True, help="Path to write the distilled JSON report.")
    parser.add_argument("--title", default="Qwen3.5 MPS Paged Attention Snapshot Sweep")
    return parser.parse_args()


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


def _parse_snapshot_name(snapshot_name: str) -> dict[str, int]:
    match = SNAPSHOT_NAME_RE.search(snapshot_name)
    if match is None:
        raise ValueError(f"could not parse snapshot name: {snapshot_name}")
    return {
        "prompt_length": int(match.group("prompt")),
        "layer_id": int(match.group("layer")),
        "kv_head_id": int(match.group("kv")),
    }


def _coverage_summary(candidate_records: list[dict[str, Any]]) -> dict[str, Any]:
    snapshots: dict[str, dict[str, Any]] = {}
    for record in candidate_records:
        snapshot_name = str(record["snapshot_name"])
        if snapshot_name in snapshots:
            continue
        parsed = _parse_snapshot_name(snapshot_name)
        snapshots[snapshot_name] = {
            "snapshot_name": snapshot_name,
            "snapshot_path": str(record["snapshot_path"]),
            "prompt_length": parsed["prompt_length"],
            "layer_id": parsed["layer_id"],
            "kv_head_id": parsed["kv_head_id"],
            "num_pages": int(record["num_pages"]),
            "tokens_per_page": int(record["tokens_per_page"]),
            "total_tokens": int(record["total_tokens"]),
        }

    ordered = sorted(
        snapshots.values(),
        key=lambda item: (item["prompt_length"], item["layer_id"], item["kv_head_id"]),
    )
    return {
        "snapshot_count": len(ordered),
        "prompt_lengths": sorted({int(item["prompt_length"]) for item in ordered}),
        "layer_ids": sorted({int(item["layer_id"]) for item in ordered}),
        "kv_head_ids": sorted({int(item["kv_head_id"]) for item in ordered}),
        "tokens_per_page_values": sorted({int(item["tokens_per_page"]) for item in ordered}),
        "num_pages_range": [min(int(item["num_pages"]) for item in ordered), max(int(item["num_pages"]) for item in ordered)],
        "total_tokens_range": [min(int(item["total_tokens"]) for item in ordered), max(int(item["total_tokens"]) for item in ordered)],
        "snapshots": ordered,
    }


def _recommendations_by_engine(recommendation_records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(record["engine"]): dict(record) for record in recommendation_records}


def _matched_speedups(aggregate_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_config: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for record in aggregate_records:
        by_config[str(record["config_key"])][str(record["engine"])] = record

    rows: list[dict[str, Any]] = []
    for config_key, engine_records in sorted(by_config.items()):
        baseline = engine_records.get("torch_mps_baseline")
        experimental = engine_records.get("mps_experimental")
        if baseline is None or experimental is None:
            continue
        rows.append(
            {
                "config_key": config_key,
                "baseline_avg_total_step_time_ms": float(baseline["avg_total_step_time_ms"]),
                "experimental_avg_total_step_time_ms": float(experimental["avg_total_step_time_ms"]),
                "speedup_ratio": float(baseline["avg_total_step_time_ms"]) / float(experimental["avg_total_step_time_ms"]),
                "baseline_avg_tokens_processed": float(baseline["avg_tokens_processed"]),
                "experimental_avg_tokens_processed": float(experimental["avg_tokens_processed"]),
                "experimental_max_abs_error": float(experimental["max_abs_error"]),
                "experimental_max_rel_error": float(experimental["max_rel_error"]),
            }
        )
    return sorted(rows, key=lambda row: row["speedup_ratio"], reverse=True)


def _prompt_breakdown(
    candidate_records: list[dict[str, Any]],
    *,
    recommendation_by_engine: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for record in candidate_records:
        engine = str(record["engine"])
        recommendation = recommendation_by_engine.get(engine)
        if recommendation is None or str(record["config_key"]) != str(recommendation["config_key"]):
            continue
        parsed = _parse_snapshot_name(str(record["snapshot_name"]))
        grouped[(parsed["prompt_length"], engine)].append(record)

    rows: list[dict[str, Any]] = []
    for (prompt_length, engine), records in sorted(grouped.items()):
        rows.append(
            {
                "prompt_length": int(prompt_length),
                "engine": engine,
                "config_key": str(recommendation_by_engine[engine]["config_key"]),
                "snapshot_count": len(records),
                "avg_total_step_time_ms": float(mean(float(record["total_step_time_ms"]) for record in records)),
                "avg_tokens_processed": float(mean(float(record["tokens_processed"]) for record in records)),
                "avg_processed_page_count": float(mean(float(record["processed_page_count"]) for record in records)),
                "max_abs_error": float(max(float(record["max_abs_error"]) for record in records)),
                "max_rel_error": float(max(float(record["max_rel_error"]) for record in records)),
            }
        )
    return rows


def _slice_spread(
    candidate_records: list[dict[str, Any]],
    *,
    engine: str,
    config_key: str,
) -> dict[str, Any]:
    matching = [record for record in candidate_records if str(record["engine"]) == engine and str(record["config_key"]) == config_key]
    ordered = []
    for record in matching:
        parsed = _parse_snapshot_name(str(record["snapshot_name"]))
        ordered.append(
            {
                "snapshot_name": str(record["snapshot_name"]),
                "prompt_length": parsed["prompt_length"],
                "layer_id": parsed["layer_id"],
                "kv_head_id": parsed["kv_head_id"],
                "total_step_time_ms": float(record["total_step_time_ms"]),
                "tokens_processed": float(record["tokens_processed"]),
            }
        )
    ordered.sort(key=lambda row: row["total_step_time_ms"])
    return {
        "engine": engine,
        "config_key": config_key,
        "snapshot_count": len(ordered),
        "fastest_slices": ordered[:5],
        "slowest_slices": list(reversed(ordered[-5:])),
    }


def _build_report(payload: dict[str, Any], *, title: str, input_path: str) -> dict[str, Any]:
    candidate_records = list(payload["candidate_records"])
    aggregate_records = list(payload["aggregate_records"])
    recommendation_records = list(payload["recommendation_records"])

    coverage = _coverage_summary(candidate_records)
    recommendation_by_engine = _recommendations_by_engine(recommendation_records)
    matched_speedups = _matched_speedups(aggregate_records)
    prompt_breakdown = _prompt_breakdown(candidate_records, recommendation_by_engine=recommendation_by_engine)

    experimental = recommendation_by_engine.get("mps_experimental")
    baseline = recommendation_by_engine.get("torch_mps_baseline")
    recommendation_comparison = None
    if experimental is not None and baseline is not None:
        recommendation_comparison = {
            "experimental_config_key": str(experimental["config_key"]),
            "baseline_config_key": str(baseline["config_key"]),
            "experimental_avg_total_step_time_ms": float(experimental["avg_total_step_time_ms"]),
            "baseline_avg_total_step_time_ms": float(baseline["avg_total_step_time_ms"]),
            "speedup_ratio": float(baseline["avg_total_step_time_ms"]) / float(experimental["avg_total_step_time_ms"]),
            "experimental_avg_tokens_processed": float(experimental["avg_tokens_processed"]),
            "baseline_avg_tokens_processed": float(baseline["avg_tokens_processed"]),
        }

    experimental_slice_spread = None
    if experimental is not None:
        experimental_slice_spread = _slice_spread(
            candidate_records,
            engine="mps_experimental",
            config_key=str(experimental["config_key"]),
        )

    return {
        "title": title,
        "inputs": {
            "input": input_path,
        },
        "coverage": coverage,
        "recommendations": recommendation_by_engine,
        "recommendation_comparison": recommendation_comparison,
        "matched_speedups": matched_speedups,
        "prompt_breakdown": prompt_breakdown,
        "experimental_slice_spread": experimental_slice_spread,
    }


def _render_markdown(report: dict[str, Any]) -> str:
    coverage = report["coverage"]
    experimental = report["recommendations"].get("mps_experimental")
    baseline = report["recommendations"].get("torch_mps_baseline")
    comparison = report.get("recommendation_comparison")

    lines = [
        f"# {report['title']}",
        "",
        "## Coverage",
        "",
        f"- snapshots: `{int(coverage['snapshot_count'])}`",
        f"- prompt lengths: `{', '.join(str(value) for value in coverage['prompt_lengths'])}`",
        f"- full-attention layers: `{', '.join(str(value) for value in coverage['layer_ids'])}`",
        f"- kv heads: `{', '.join(str(value) for value in coverage['kv_head_ids'])}`",
        f"- pages per snapshot: `{coverage['num_pages_range'][0]}-{coverage['num_pages_range'][1]}`",
        f"- total tokens per snapshot: `{coverage['total_tokens_range'][0]}-{coverage['total_tokens_range'][1]}`",
        "",
        "## Recommendation",
        "",
    ]

    recommendation_rows = [[
        "Engine",
        "Config",
        "Avg step ms",
        "Avg tokens",
        "Avg pages",
        "Pass rate",
        "Max abs err",
        "Max rel err",
    ]]
    for engine in ("mps_experimental", "torch_mps_baseline"):
        record = report["recommendations"].get(engine)
        if record is None:
            continue
        recommendation_rows.append(
            [
                engine,
                str(record["config_key"]),
                _fmt(record["avg_total_step_time_ms"]),
                _fmt(record["avg_tokens_processed"], digits=1),
                _fmt(record["avg_processed_page_count"], digits=1),
                _fmt(100.0 * float(record["pass_rate"]), digits=1) + "%",
                _fmt(record["max_abs_error"], digits=6),
                _fmt(record["max_rel_error"], digits=6),
            ]
        )
    lines.extend([_markdown_table(recommendation_rows), ""])

    if comparison is not None:
        speedup_ratio = float(comparison["speedup_ratio"])
        if speedup_ratio > 1.005:
            comparison_sentence = (
                "The current winning experimental config leads on the full replay corpus, "
                f"running at `{_fmt(comparison['experimental_avg_total_step_time_ms'])} ms` versus "
                f"`{_fmt(comparison['baseline_avg_total_step_time_ms'])} ms` for the best baseline, "
                f"which is a `{_fmt(speedup_ratio, digits=3)}x` speedup."
            )
        elif speedup_ratio < 0.995:
            comparison_sentence = (
                "The current winning experimental config trails the best baseline on the full replay corpus, "
                f"running at `{_fmt(comparison['experimental_avg_total_step_time_ms'])} ms` versus "
                f"`{_fmt(comparison['baseline_avg_total_step_time_ms'])} ms`, "
                f"or `{_fmt(speedup_ratio, digits=3)}x` of baseline speed."
            )
        else:
            comparison_sentence = (
                "The current winning experimental config is effectively at parity with the best baseline on the full replay corpus, "
                f"running at `{_fmt(comparison['experimental_avg_total_step_time_ms'])} ms` versus "
                f"`{_fmt(comparison['baseline_avg_total_step_time_ms'])} ms`, "
                f"with a `{_fmt(speedup_ratio, digits=3)}x` speed ratio."
            )
        lines.extend(
            [
                comparison_sentence
                + f" The tradeoff is a smaller active budget: `{_fmt(comparison['experimental_avg_tokens_processed'], digits=1)}` "
                f"average processed tokens for the experimental winner versus "
                f"`{_fmt(comparison['baseline_avg_tokens_processed'], digits=1)}` for the baseline winner.",
                "",
            ]
        )

    lines.extend(["## Matched Config Speedups", ""])
    speedup_rows = [[
        "Config",
        "Baseline ms",
        "Experimental ms",
        "Speedup",
        "Baseline tokens",
        "Experimental tokens",
    ]]
    for row in report["matched_speedups"][:8]:
        speedup_rows.append(
            [
                str(row["config_key"]),
                _fmt(row["baseline_avg_total_step_time_ms"]),
                _fmt(row["experimental_avg_total_step_time_ms"]),
                _fmt(row["speedup_ratio"], digits=3) + "x",
                _fmt(row["baseline_avg_tokens_processed"], digits=1),
                _fmt(row["experimental_avg_tokens_processed"], digits=1),
            ]
        )
    lines.extend([_markdown_table(speedup_rows), ""])

    lines.extend(["## Prompt Breakdown", ""])
    prompt_rows = [[
        "Prompt",
        "Engine",
        "Config",
        "Avg step ms",
        "Avg tokens",
        "Avg pages",
        "Max abs err",
    ]]
    for row in report["prompt_breakdown"]:
        prompt_rows.append(
            [
                str(int(row["prompt_length"])),
                str(row["engine"]),
                str(row["config_key"]),
                _fmt(row["avg_total_step_time_ms"]),
                _fmt(row["avg_tokens_processed"], digits=1),
                _fmt(row["avg_processed_page_count"], digits=1),
                _fmt(row["max_abs_error"], digits=6),
            ]
        )
    lines.extend([_markdown_table(prompt_rows), ""])

    spread = report.get("experimental_slice_spread")
    if spread is not None:
        lines.extend(["## Experimental Slice Spread", ""])
        spread_rows = [[
            "Slice",
            "Step ms",
            "Tokens",
        ]]
        for row in spread["slowest_slices"]:
            spread_rows.append(
                [
                    f"prompt{int(row['prompt_length'])}/layer{int(row['layer_id'])}/kv{int(row['kv_head_id'])}",
                    _fmt(row["total_step_time_ms"]),
                    _fmt(row["tokens_processed"], digits=1),
                ]
            )
        lines.extend(
            [
                "Slowest slices for the winning experimental config:",
                "",
                _markdown_table(spread_rows),
                "",
            ]
        )

    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    report = _build_report(payload, title=args.title, input_path=str(input_path))
    markdown = _render_markdown(report)

    json_output = Path(args.json_output)
    json_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8")

    markdown_output = Path(args.markdown_output)
    markdown_output.parent.mkdir(parents=True, exist_ok=True)
    markdown_output.write_text(markdown, encoding="utf-8")

    print(markdown, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
