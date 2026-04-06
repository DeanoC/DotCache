#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare exact, shortlist, and learned-selector DotCache backend profiles.")
    parser.add_argument("--exact", required=True, help="Path to exact DotCache JSONL artifact.")
    parser.add_argument("--shortlist", required=True, help="Path to shortlist-base DotCache JSONL artifact.")
    parser.add_argument("--learned", required=True, help="Path to learned-selector KV-scope DotCache JSONL artifact.")
    parser.add_argument("--learned-k-only", default=None, help="Optional path to learned-selector K-only DotCache JSONL artifact.")
    parser.add_argument("--learned-v-only", default=None, help="Optional path to learned-selector V-only DotCache JSONL artifact.")
    parser.add_argument("--markdown-output", default=None, help="Optional markdown output path.")
    parser.add_argument("--json-output", default=None, help="Optional JSON output path.")
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


def _mib(value: Any) -> float | None:
    if value is None:
        return None
    return float(value) / (1024.0 * 1024.0)


def _mode_family_counts(mode_signature_counts: dict[str, int] | None) -> dict[str, int]:
    counts = {"M0": 0, "M1": 0, "M2": 0, "M3": 0, "M4": 0, "T3": 0}
    if not isinstance(mode_signature_counts, dict):
        return counts
    for signature, count in mode_signature_counts.items():
        mode = str(signature).split(":")[1] if ":" in str(signature) else ""
        if mode in counts:
            counts[mode] += int(count)
    return counts


def _row_summary(row: dict[str, Any]) -> dict[str, Any]:
    trace = row.get("decode_backend_trace") or {}
    steps = max(int(row.get("decode_steps") or 1), 1)
    mode_counts = _mode_family_counts(row.get("mode_signature_counts"))
    total_pages = max(sum(mode_counts.values()), 1)
    selector_invocations = int(row.get("learned_page_selector_invocations") or 0)
    selector_ms_total = float(row.get("learned_page_selector_ms_total") or 0.0)
    return {
        "prompt_length": int(row.get("prompt_length") or 0),
        "decode_ms_per_step": float(row.get("dotcache_decode_ms_per_step") or 0.0),
        "decode_ms_per_step_p95": float(
            row.get("dotcache_decode_ms_per_step_p95") or row.get("dotcache_decode_ms_per_step") or 0.0
        ),
        "prefill_ms": float(row.get("dotcache_prefill_ms") or 0.0),
        "resident_mib": _mib(row.get("resident_bytes")),
        "kv_resident_mib": _mib(row.get("kv_resident_bytes")),
        "m0_pages": int(mode_counts["M0"]),
        "m3_pages": int(mode_counts["M3"]),
        "m0_fraction": float(mode_counts["M0"] / total_pages),
        "m3_fraction": float(mode_counts["M3"] / total_pages),
        "shortlist_selected_pages": int(row.get("execution_shortlist_selected_pages") or 0),
        "shortlist_total_pages": int(row.get("execution_shortlist_total_pages") or 0),
        "selector_enabled": bool(row.get("learned_page_selector_enabled", False)),
        "selector_invocations": selector_invocations,
        "selector_ms_total": selector_ms_total,
        "selector_us_per_invocation": (selector_ms_total * 1000.0 / selector_invocations) if selector_invocations else None,
        "measurement_kind": row.get("measurement_kind"),
        "warmup_runs": int(row.get("warmup_runs") or 0),
        "measured_runs": int(row.get("measured_runs") or 1),
        "prepare_ms_step": float(trace.get("prepare_ms_total", 0.0)) / steps,
        "score_ms_step": float(trace.get("score_ms_total", 0.0)) / steps,
        "mix_ms_step": float(trace.get("mix_ms_total", 0.0)) / steps,
        "softmax_ms_step": float(trace.get("softmax_ms_total", 0.0)) / steps,
        "unpack_ms_step": float(trace.get("unpack_ms_total", 0.0)) / steps,
        "chunk_assembly_ms_step": float(trace.get("chunk_assembly_ms_total", 0.0)) / steps,
        "payload_read_mib_step": _mib(trace.get("payload_bytes_read")) / steps if trace.get("payload_bytes_read") is not None else None,
        "metadata_read_kib_step": float(trace.get("metadata_bytes_read", 0.0)) / (1024.0 * steps),
    }


def _fmt(value: float | None, digits: int = 2) -> str:
    if value is None:
        return "-"
    return f"{value:.{digits}f}"


def _build_report(args: argparse.Namespace) -> dict[str, Any]:
    variants = {
        "exact": _load_rows(Path(args.exact)),
        "shortlist_base": _load_rows(Path(args.shortlist)),
        "learned_selector": _load_rows(Path(args.learned)),
    }
    if args.learned_k_only:
        variants["learned_selector_k_only"] = _load_rows(Path(args.learned_k_only))
    if args.learned_v_only:
        variants["learned_selector_v_only"] = _load_rows(Path(args.learned_v_only))
    rows_by_context: dict[int, dict[str, dict[str, Any]]] = {}
    for variant_name, rows in variants.items():
        for row in rows:
            summary = _row_summary(row)
            rows_by_context.setdefault(int(summary["prompt_length"]), {})[variant_name] = summary
    comparisons: list[dict[str, Any]] = []
    for context in sorted(rows_by_context):
        context_rows = rows_by_context[context]
        learned = context_rows.get("learned_selector")
        exact = context_rows.get("exact")
        shortlist = context_rows.get("shortlist_base")
        comparison = {
            "prompt_length": context,
            "variants": context_rows,
        }
        speedups: dict[str, dict[str, float]] = {}
        for variant_name, variant_row in context_rows.items():
            if variant_name in {"exact", "shortlist_base"}:
                continue
            variant_speedups: dict[str, float] = {}
            if exact and variant_row:
                variant_speedups["vs_exact"] = float(exact["decode_ms_per_step"] / max(variant_row["decode_ms_per_step"], 1e-8))
            if shortlist and variant_row:
                variant_speedups["vs_shortlist"] = float(
                    shortlist["decode_ms_per_step"] / max(variant_row["decode_ms_per_step"], 1e-8)
                )
            if variant_speedups:
                speedups[variant_name] = variant_speedups
        if speedups:
            comparison["speedups"] = speedups
        comparisons.append(comparison)
    return {
        "inputs": {
            "exact": args.exact,
            "shortlist": args.shortlist,
            "learned": args.learned,
            "learned_k_only": args.learned_k_only,
            "learned_v_only": args.learned_v_only,
        },
        "comparisons": comparisons,
    }


def _render_markdown(report: dict[str, Any]) -> str:
    variant_order = [
        "exact",
        "shortlist_base",
        "learned_selector",
        "learned_selector_k_only",
        "learned_selector_v_only",
    ]
    lines = [
        "# Qwen3.5 Backend Truth Report",
        "",
        "## Decode And Memory",
        "",
        "| Context | Variant | Decode ms/step | Decode p95 | Prefill ms | Resident MiB | KV MiB | M0 frac | M3 frac | Shortlist | Selector ms | Selector us/inv |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for comparison in report["comparisons"]:
        context = int(comparison["prompt_length"])
        for variant_name in variant_order:
            row = comparison["variants"].get(variant_name)
            if row is None:
                continue
            shortlist_summary = (
                f"{int(row['shortlist_selected_pages'])}/{int(row['shortlist_total_pages'])}"
                if int(row["shortlist_total_pages"]) > 0
                else "-"
            )
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(context),
                        variant_name,
                        _fmt(row["decode_ms_per_step"]),
                        _fmt(row["decode_ms_per_step_p95"]),
                        _fmt(row["prefill_ms"]),
                        _fmt(row["resident_mib"]),
                        _fmt(row["kv_resident_mib"]),
                        _fmt(100.0 * float(row["m0_fraction"])),
                        _fmt(100.0 * float(row["m3_fraction"])),
                        shortlist_summary,
                        _fmt(row["selector_ms_total"]),
                        _fmt(row["selector_us_per_invocation"], digits=1),
                    ]
                )
                + " |"
            )
    lines.extend(
        [
            "",
            "## Backend Breakdown",
            "",
            "| Context | Variant | Prepare | Score | Mix | Softmax | Unpack | Chunk | Payload MiB/step | Metadata KiB/step |",
            "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for comparison in report["comparisons"]:
        context = int(comparison["prompt_length"])
        for variant_name in variant_order:
            row = comparison["variants"].get(variant_name)
            if row is None:
                continue
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(context),
                        variant_name,
                        _fmt(row["prepare_ms_step"]),
                        _fmt(row["score_ms_step"]),
                        _fmt(row["mix_ms_step"]),
                        _fmt(row["softmax_ms_step"]),
                        _fmt(row["unpack_ms_step"]),
                        _fmt(row["chunk_assembly_ms_step"]),
                        _fmt(row["payload_read_mib_step"]),
                        _fmt(row["metadata_read_kib_step"]),
                    ]
                )
                + " |"
            )
    lines.extend(["", "## Speedups", "", "| Context | Variant | Vs Exact | Vs Shortlist |", "| ---: | --- | ---: | ---: |"])
    for comparison in report["comparisons"]:
        for variant_name in ("learned_selector", "learned_selector_k_only", "learned_selector_v_only"):
            speedups = (comparison.get("speedups") or {}).get(variant_name)
            if not speedups:
                continue
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(int(comparison["prompt_length"])),
                        variant_name,
                        _fmt(speedups.get("vs_exact")),
                        _fmt(speedups.get("vs_shortlist")),
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
