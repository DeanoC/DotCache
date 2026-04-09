#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


LABEL_RE = re.compile(
    r"(?P<backend>baseline|experimental)_(?P<controller>robust|approx)_(?P<topk>\d+)_(?P<recent>\d+)_c(?P<chunk>\d+)"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize MPS paged-attention family-compare results.")
    parser.add_argument("--compare-input", required=True, help="Path to family-compare JSON.")
    parser.add_argument("--corpus-summary", required=True, help="Path to replay-corpus summary JSON.")
    parser.add_argument("--corpus-manifest", required=True, help="Path to replay-corpus manifest JSON.")
    parser.add_argument("--markdown-output", required=True, help="Path to write the markdown report.")
    parser.add_argument("--json-output", required=True, help="Path to write the distilled JSON report.")
    parser.add_argument("--title", default="Qwen3.5 MPS Paged Attention Real-Doc Family Compare")
    return parser.parse_args()


def _fmt(value: object, digits: int = 3) -> str:
    if value is None:
        return "-"
    return f"{float(value):.{digits}f}"


def _markdown_table(rows: list[list[str]]) -> str:
    def escape(cell: str) -> str:
        return str(cell).replace("|", "\\|")

    header = [escape(cell) for cell in rows[0]]
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
    ]
    lines.extend("| " + " | ".join(escape(cell) for cell in row) + " |" for row in rows[1:])
    return "\n".join(lines)


def _parse_label(label: str) -> dict[str, Any]:
    match = LABEL_RE.fullmatch(label)
    if match is None:
        raise ValueError(f"could not parse label: {label}")
    data = match.groupdict()
    return {
        "backend": data["backend"],
        "controller": data["controller"],
        "top_k": int(data["topk"]),
        "recent_window_tokens": int(data["recent"]),
        "chunk_size": int(data["chunk"]),
        "sink_window_tokens": 64,
        "approximate_mode": data["controller"] == "approx",
        "approximate_max_optional_blocks": 1 if data["controller"] == "approx" else 0,
    }


def _backend_label(backend: str) -> str:
    return {
        "baseline": "Baseline Backend",
        "experimental": "Experimental Backend",
    }[backend]


def _controller_label(controller: str) -> str:
    return {
        "robust": "Robust Full Pass",
        "approx": "Approx Budget",
    }[controller]


def _display_label(parsed: dict[str, Any]) -> str:
    return f"{_backend_label(str(parsed['backend']))} / {_controller_label(str(parsed['controller']))}"


def _config_key(parsed: dict[str, Any]) -> str:
    return (
        f"topk={parsed['top_k']}|recent={parsed['recent_window_tokens']}|"
        f"sink={parsed['sink_window_tokens']}|chunk={parsed['chunk_size']}|"
        f"approx={1 if parsed['approximate_mode'] else 0}|"
        f"approx_opt={parsed['approximate_max_optional_blocks']}"
    )


def _enrich_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    enriched = []
    for row in rows:
        parsed = _parse_label(str(row["label"]))
        enriched_row = dict(row)
        enriched_row.update(parsed)
        enriched_row["display_label"] = _display_label(parsed)
        enriched_row["config_key"] = _config_key(parsed)
        enriched_row["family_key"] = (
            f"{parsed['controller']}|topk={parsed['top_k']}|recent={parsed['recent_window_tokens']}"
        )
        enriched.append(enriched_row)
    return enriched


def _build_coverage(summary: dict[str, Any], manifest: dict[str, Any]) -> dict[str, Any]:
    case_rows = []
    for record in manifest.get("records", []):
        if record.get("prompt_mode") != "prompt_file":
            continue
        case_rows.append(
            {
                "case_tag": str(record["case_tag"]),
                "prompt_length": int(record["prompt_length"]),
                "prompt_file_path": str(record["prompt_file_path"]),
                "prefill_ms": float(record["prefill_ms"]),
                "dense_decode_ms_per_step": float(record["dense_decode_ms_per_step"]),
                "snapshot_count": int(record["paged_attention_snapshot_corpus_count"]),
            }
        )
    case_rows.sort(key=lambda row: row["case_tag"])
    return {
        "case_count": int(summary["case_count"]),
        "snapshot_count": int(summary["snapshot_count"]),
        "layer_ids": list(summary["layer_ids"]),
        "kv_head_ids": list(summary["kv_head_ids"]),
        "resolved_step_indices": list(summary["resolved_step_indices"]),
        "prompt_modes": dict(summary["counts_by_prompt_mode"]),
        "cases": case_rows,
    }


def _normalized_comparison(coverage: dict[str, Any], recommendations: dict[str, Any], manifest: dict[str, Any]) -> dict[str, Any]:
    case_records = [record for record in manifest.get("records", []) if record.get("prompt_mode") == "prompt_file"]
    if not case_records:
        return {}

    avg_prefill_ms = sum(float(record["prefill_ms"]) for record in case_records) / len(case_records)
    avg_dense_decode_ms_per_step = sum(float(record["dense_decode_ms_per_step"]) for record in case_records) / len(case_records)
    avg_decode_steps = sum(float(record["decode_steps"]) for record in case_records) / len(case_records)
    avg_exported_snapshots_per_prompt = sum(float(record["paged_attention_snapshot_corpus_count"]) for record in case_records) / len(case_records)
    avg_exported_steps = sum(len(record["paged_attention_snapshot_corpus_resolved_step_indices"]) for record in case_records) / len(case_records)
    avg_snapshots_per_exported_step = avg_exported_snapshots_per_prompt / avg_exported_steps if avg_exported_steps else None
    avg_total_capture_ms_per_prompt = avg_prefill_ms + avg_decode_steps * avg_dense_decode_ms_per_step
    dense_decode_ms_per_exported_snapshot = (
        avg_dense_decode_ms_per_step / avg_snapshots_per_exported_step if avg_snapshots_per_exported_step else None
    )
    total_capture_ms_per_exported_snapshot = (
        avg_total_capture_ms_per_prompt / avg_exported_snapshots_per_prompt if avg_exported_snapshots_per_prompt else None
    )

    experimental = recommendations.get("best_fully_passing_overall")
    fast_tradeoff = recommendations.get("best_fast_tradeoff_overall")
    normalized_rows = [
        {
            "name": "Original Dense Prefill",
            "scope": "One 4096-token prompt, full dense model",
            "avg_ms": avg_prefill_ms,
            "unit": "ms/prompt",
        },
        {
            "name": "Original Dense Decode",
            "scope": "One generated token, full dense model",
            "avg_ms": avg_dense_decode_ms_per_step,
            "unit": "ms/step",
        },
        {
            "name": "Replay Corpus Extraction",
            "scope": "One prompt capture: dense prefill + 4 dense decode steps, exporting 36 replay snapshots",
            "avg_ms": avg_total_capture_ms_per_prompt,
            "unit": "ms/prompt-capture",
        },
        {
            "name": "Dense Decode Amortized",
            "scope": "Decode-side cost per exported layer/head replay snapshot",
            "avg_ms": dense_decode_ms_per_exported_snapshot,
            "unit": "ms/exported-snapshot",
        },
        {
            "name": "Replay Extraction Amortized",
            "scope": "Full capture cost amortized over exported replay snapshots",
            "avg_ms": total_capture_ms_per_exported_snapshot,
            "unit": "ms/exported-snapshot",
        },
    ]
    if experimental is not None:
        normalized_rows.append(
            {
                "name": "Paged Replay Winner",
                "scope": str(experimental["display_label"]),
                "avg_ms": float(experimental["avg_total_step_time_ms"]),
                "unit": "ms/replay-snapshot",
                "ratio_vs_dense_decode_amortized": (
                    float(experimental["avg_total_step_time_ms"]) / dense_decode_ms_per_exported_snapshot
                    if dense_decode_ms_per_exported_snapshot
                    else None
                ),
            }
        )
    if fast_tradeoff is not None:
        normalized_rows.append(
            {
                "name": "Paged Replay Fast Tradeoff",
                "scope": str(fast_tradeoff["display_label"]),
                "avg_ms": float(fast_tradeoff["avg_total_step_time_ms"]),
                "unit": "ms/replay-snapshot",
                "ratio_vs_dense_decode_amortized": (
                    float(fast_tradeoff["avg_total_step_time_ms"]) / dense_decode_ms_per_exported_snapshot
                    if dense_decode_ms_per_exported_snapshot
                    else None
                ),
            }
        )

    return {
        "avg_prefill_ms": avg_prefill_ms,
        "avg_dense_decode_ms_per_step": avg_dense_decode_ms_per_step,
        "avg_decode_steps": avg_decode_steps,
        "avg_exported_snapshots_per_prompt": avg_exported_snapshots_per_prompt,
        "avg_exported_steps": avg_exported_steps,
        "avg_snapshots_per_exported_step": avg_snapshots_per_exported_step,
        "avg_total_capture_ms_per_prompt": avg_total_capture_ms_per_prompt,
        "dense_decode_ms_per_exported_snapshot": dense_decode_ms_per_exported_snapshot,
        "total_capture_ms_per_exported_snapshot": total_capture_ms_per_exported_snapshot,
        "rows": normalized_rows,
    }


def _recommendations(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_backend: dict[str, list[dict[str, Any]]] = {"baseline": [], "experimental": []}
    for row in rows:
        by_backend[str(row["backend"])].append(row)

    fully_passing = [row for row in rows if float(row["pass_rate"]) >= 1.0]
    near_perfect = [row for row in rows if float(row["pass_rate"]) >= 0.99]

    def best(items: list[dict[str, Any]]) -> dict[str, Any] | None:
        if not items:
            return None
        return sorted(items, key=lambda row: (-float(row["pass_rate"]), float(row["avg_total_step_time_ms"])))[0]

    best_fully_passing_overall = best(fully_passing)
    best_near_perfect_overall = best(near_perfect)
    best_fast_tradeoff_overall = None
    if best_fully_passing_overall is not None:
        faster_near_perfect = [
            row
            for row in near_perfect
            if float(row["avg_total_step_time_ms"]) < float(best_fully_passing_overall["avg_total_step_time_ms"])
        ]
        if faster_near_perfect:
            best_fast_tradeoff_overall = sorted(faster_near_perfect, key=lambda row: float(row["avg_total_step_time_ms"]))[0]

    return {
        "best_fully_passing_overall": best_fully_passing_overall,
        "best_near_perfect_overall": best_near_perfect_overall,
        "best_fast_tradeoff_overall": best_fast_tradeoff_overall,
        "best_fully_passing_by_backend": {
            backend: best([row for row in backend_rows if float(row["pass_rate"]) >= 1.0])
            for backend, backend_rows in by_backend.items()
        },
        "best_near_perfect_by_backend": {
            backend: best([row for row in backend_rows if float(row["pass_rate"]) >= 0.99])
            for backend, backend_rows in by_backend.items()
        },
    }


def _matched_families(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["family_key"]), {})[str(row["backend"])] = row

    matched = []
    for family_key, family_rows in sorted(grouped.items()):
        baseline = family_rows.get("baseline")
        experimental = family_rows.get("experimental")
        if baseline is None or experimental is None:
            continue
        matched.append(
            {
                "family_key": family_key,
                "baseline_label": str(baseline["label"]),
                "experimental_label": str(experimental["label"]),
                "baseline_avg_total_step_time_ms": float(baseline["avg_total_step_time_ms"]),
                "experimental_avg_total_step_time_ms": float(experimental["avg_total_step_time_ms"]),
                "speedup_ratio": float(baseline["avg_total_step_time_ms"]) / float(experimental["avg_total_step_time_ms"]),
                "baseline_avg_tokens_processed": float(baseline["avg_tokens_processed"]),
                "experimental_avg_tokens_processed": float(experimental["avg_tokens_processed"]),
                "baseline_pass_rate": float(baseline["pass_rate"]),
                "experimental_pass_rate": float(experimental["pass_rate"]),
            }
        )
    matched.sort(key=lambda row: row["speedup_ratio"], reverse=True)
    return matched


def _build_report(
    compare_rows: list[dict[str, Any]],
    *,
    corpus_summary: dict[str, Any],
    corpus_manifest: dict[str, Any],
    title: str,
    compare_input: str,
) -> dict[str, Any]:
    enriched = _enrich_rows(compare_rows)
    recommendations = _recommendations(enriched)
    matched = _matched_families(enriched)
    coverage = _build_coverage(corpus_summary, corpus_manifest)
    return {
        "title": title,
        "inputs": {
            "compare_input": compare_input,
            "corpus_output_dir": str(corpus_manifest.get("output_dir", "")),
        },
        "coverage": coverage,
        "normalized_comparison": _normalized_comparison(coverage, recommendations, corpus_manifest),
        "recommendations": recommendations,
        "matched_families": matched,
        "rows": enriched,
    }


def _render_markdown(report: dict[str, Any]) -> str:
    coverage = report["coverage"]
    normalized = report.get("normalized_comparison", {})
    best_full = report["recommendations"]["best_fully_passing_overall"]
    best_near = report["recommendations"]["best_near_perfect_overall"]
    best_tradeoff = report["recommendations"]["best_fast_tradeoff_overall"]
    matched = report["matched_families"]

    lines = [
        f"# {report['title']}",
        "",
        "## Coverage",
        "",
        f"- prompt-file cases: `{int(coverage['case_count'])}`",
        f"- replay snapshots: `{int(coverage['snapshot_count'])}`",
        f"- full-attention layers: `{', '.join(str(v) for v in coverage['layer_ids'])}`",
        f"- kv heads: `{', '.join(str(v) for v in coverage['kv_head_ids'])}`",
        f"- decode steps: `{', '.join(str(v) for v in coverage['resolved_step_indices'])}`",
        "",
    ]

    case_rows = [["Case", "Prompt", "Prefill ms", "Dense decode ms/step", "Snapshots"]]
    for row in coverage["cases"]:
        case_rows.append(
            [
                str(row["case_tag"]),
                str(row["prompt_file_path"]),
                _fmt(row["prefill_ms"]),
                _fmt(row["dense_decode_ms_per_step"]),
                str(int(row["snapshot_count"])),
            ]
        )
    lines.extend([_markdown_table(case_rows), ""])

    normalized_rows = normalized.get("rows", [])
    if normalized_rows:
        lines.extend(["## Normalized Comparison", ""])
        normalized_table = [["View", "Scope", "Avg ms", "Unit", "Vs dense decode amortized"]]
        for row in normalized_rows:
            ratio = row.get("ratio_vs_dense_decode_amortized")
            normalized_table.append(
                [
                    str(row["name"]),
                    str(row["scope"]),
                    _fmt(row["avg_ms"]),
                    str(row["unit"]),
                    (_fmt(ratio, 3) + "x") if ratio is not None else "-",
                ]
            )
        lines.extend([_markdown_table(normalized_table), ""])

    lines.extend(["## Recommendation", ""])
    if best_full is not None:
        lines.append(
            f"The best fully passing family is `{best_full['label']}` "
            f"({best_full['display_label']}) at `{_fmt(best_full['avg_total_step_time_ms'])} ms`, "
            f"processing `{_fmt(best_full['avg_tokens_processed'], 1)}` tokens with "
            f"`{_fmt(100.0 * float(best_full['pass_rate']), 1)}%` pass rate."
        )
        lines.append("")
    if best_tradeoff is not None:
        lines.append(
            f"The fastest near-perfect tradeoff is `{best_tradeoff['label']}` at "
            f"`{_fmt(best_tradeoff['avg_total_step_time_ms'])} ms` with "
            f"`{_fmt(100.0 * float(best_tradeoff['pass_rate']), 1)}%` pass rate."
        )
        lines.append("")

    recommendation_rows = [[
        "Family",
        "Backend / Controller",
        "Avg step ms",
        "Avg tokens",
        "Pass rate",
        "Max abs err",
        "Max rel err",
    ]]
    for key in ("best_fully_passing_overall", "best_fast_tradeoff_overall"):
        row = report["recommendations"][key]
        if row is None:
            continue
        recommendation_rows.append(
            [
                str(row["label"]),
                str(row["display_label"]),
                _fmt(row["avg_total_step_time_ms"]),
                _fmt(row["avg_tokens_processed"], 1),
                _fmt(100.0 * float(row["pass_rate"]), 1) + "%",
                _fmt(row["max_abs_error"], 6),
                _fmt(row["max_rel_error"], 6),
            ]
        )
    lines.extend([_markdown_table(recommendation_rows), ""])

    lines.extend(["## Matched Family Comparison", ""])
    matched_rows = [[
        "Family",
        "Baseline ms",
        "Experimental ms",
        "Speedup",
        "Baseline pass",
        "Experimental pass",
        "Tokens",
    ]]
    for row in matched:
        matched_rows.append(
            [
                str(row["family_key"]),
                _fmt(row["baseline_avg_total_step_time_ms"]),
                _fmt(row["experimental_avg_total_step_time_ms"]),
                _fmt(row["speedup_ratio"], 3) + "x",
                _fmt(100.0 * float(row["baseline_pass_rate"]), 1) + "%",
                _fmt(100.0 * float(row["experimental_pass_rate"]), 1) + "%",
                _fmt(row["experimental_avg_tokens_processed"], 1),
            ]
        )
    lines.extend([_markdown_table(matched_rows), ""])

    lines.extend(["## All Families", ""])
    all_rows = [[
        "Family",
        "Backend / Controller",
        "Avg step ms",
        "Avg tokens",
        "Avg pages",
        "Pass rate",
    ]]
    for row in report["rows"]:
        all_rows.append(
            [
                str(row["label"]),
                str(row["display_label"]),
                _fmt(row["avg_total_step_time_ms"]),
                _fmt(row["avg_tokens_processed"], 1),
                _fmt(row["avg_processed_page_count"], 1),
                _fmt(100.0 * float(row["pass_rate"]), 1) + "%",
            ]
        )
    lines.extend([_markdown_table(all_rows), ""])
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    compare_input = Path(args.compare_input)
    corpus_summary = json.loads(Path(args.corpus_summary).read_text(encoding="utf-8"))
    corpus_manifest = json.loads(Path(args.corpus_manifest).read_text(encoding="utf-8"))
    compare_rows = json.loads(compare_input.read_text(encoding="utf-8"))
    report = _build_report(
        compare_rows,
        corpus_summary=corpus_summary,
        corpus_manifest=corpus_manifest,
        title=args.title,
        compare_input=str(compare_input),
    )
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
