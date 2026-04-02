#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize learned-selector page-mix sensitivity sweeps for Qwen3.5 serving.")
    parser.add_argument("--manifest", required=True, help="Path to selector_logit_sweep_manifest.json.")
    parser.add_argument("--results-dir", required=True, help="Directory containing per-variant JSONL benchmark outputs.")
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
        "resident_mib": _mib(row.get("resident_bytes")),
        "kv_resident_mib": _mib(row.get("kv_resident_bytes")),
        "m0_fraction": float(mode_counts["M0"] / total_pages),
        "m3_fraction": float(mode_counts["M3"] / total_pages),
        "selector_invocations": selector_invocations,
        "selector_us_per_invocation": (selector_ms_total * 1000.0 / selector_invocations) if selector_invocations else None,
        "score_ms_step": float(trace.get("score_ms_total", 0.0)) / steps,
        "mix_ms_step": float(trace.get("mix_ms_total", 0.0)) / steps,
        "payload_read_mib_step": _mib(trace.get("payload_bytes_read")) / steps if trace.get("payload_bytes_read") is not None else None,
    }


def _fmt(value: float | None, digits: int = 2) -> str:
    if value is None:
        return "-"
    return f"{value:.{digits}f}"


def _build_report(args: argparse.Namespace) -> dict[str, Any]:
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    results_dir = Path(args.results_dir)
    variants: list[dict[str, Any]] = []
    best_by_context: dict[int, float] = {}
    for entry in manifest.get("variants", []):
        variant = str(entry["variant"])
        rows = _load_rows(results_dir / f"{variant}.jsonl")
        summaries = [_row_summary(row) for row in rows]
        for summary in summaries:
            context = int(summary["prompt_length"])
            best_by_context[context] = min(best_by_context.get(context, float("inf")), float(summary["decode_ms_per_step"]))
        variants.append(
            {
                "variant": variant,
                "logit_offset": float(entry["logit_offset"]),
                "artifact_path": str(entry["artifact_path"]),
                "rows": summaries,
            }
        )

    for variant in variants:
        for row in variant["rows"]:
            best = best_by_context.get(int(row["prompt_length"]))
            row["decode_vs_best_ratio"] = float(row["decode_ms_per_step"] / best) if best and best > 0.0 else None
    return {
        "inputs": {
            "manifest": args.manifest,
            "results_dir": args.results_dir,
        },
        "target_candidate": manifest.get("target_candidate"),
        "variants": variants,
    }


def _render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Qwen3.5 Page Mix Sensitivity Report",
        "",
        f"- target candidate: `{report.get('target_candidate')}`",
        "",
        "| Offset | Context | Decode ms/step | Decode p95 | Decode / best | Resident MiB | KV MiB | M0 frac | M3 frac | Score | Mix | Payload MiB/step | Selector us/inv |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for variant in sorted(report["variants"], key=lambda item: float(item["logit_offset"])):
        offset = float(variant["logit_offset"])
        for row in variant["rows"]:
            lines.append(
                "| "
                + " | ".join(
                    [
                        _fmt(offset),
                        str(int(row["prompt_length"])),
                        _fmt(row["decode_ms_per_step"]),
                        _fmt(row["decode_ms_per_step_p95"]),
                        _fmt(row["decode_vs_best_ratio"], digits=3),
                        _fmt(row["resident_mib"]),
                        _fmt(row["kv_resident_mib"]),
                        _fmt(100.0 * float(row["m0_fraction"])),
                        _fmt(100.0 * float(row["m3_fraction"])),
                        _fmt(row["score_ms_step"]),
                        _fmt(row["mix_ms_step"]),
                        _fmt(row["payload_read_mib_step"]),
                        _fmt(row["selector_us_per_invocation"], digits=1),
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
