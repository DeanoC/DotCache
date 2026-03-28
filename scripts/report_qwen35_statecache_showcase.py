#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


BENCHMARKS = {
    "qwen35_deltanet_statecache_readout",
    "qwen35_deltanet_statecache_serving",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize Qwen3.5 StateCache compare-mode and serving-mode benchmark JSONL into a compact showcase table."
    )
    parser.add_argument(
        "--input",
        action="append",
        required=True,
        help="JSONL input file. Can be passed multiple times.",
    )
    parser.add_argument(
        "--format",
        choices=["markdown", "json"],
        default="markdown",
        help="Output format.",
    )
    return parser.parse_args()


def _load_jsonl_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if not path.exists():
        raise FileNotFoundError(f"input file not found: {path}")
    for raw_line in path.read_bytes().splitlines():
        stripped = raw_line.decode("utf-8", errors="ignore").replace("\x00", "").strip()
        if not stripped:
            continue
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and payload.get("benchmark") in BENCHMARKS:
            records.append(payload)
    return records


def _metric(record: dict[str, Any], key: str) -> float | None:
    value = record.get(key)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _tok_per_s(ms_per_step: float | None) -> float | None:
    if ms_per_step is None or ms_per_step <= 0:
        return None
    return 1000.0 / ms_per_step


def _pct(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator in (None, 0.0):
        return None
    return 100.0 * numerator / denominator


def _fmt_num(value: float | None, digits: int = 2) -> str:
    if value is None:
        return "-"
    return f"{value:.{digits}f}"


def _fmt_pct(value: float | None, digits: int = 2) -> str:
    if value is None:
        return "-"
    return f"{value:.{digits}f}%"


def _fmt_status(record: dict[str, Any]) -> str:
    if record.get("status") == "error":
        error_type = record.get("error_type") or "Error"
        return f"error:{error_type}"
    return "ok"


def _build_compare_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        if record.get("benchmark") != "qwen35_deltanet_statecache_readout":
            continue
        if record.get("status") == "error":
            rows.append(
                {
                    "mode": "compare",
                    "model_id": record.get("model_id"),
                    "prompt_length": record.get("prompt_length"),
                    "weight_quantization": record.get("weight_quantization"),
                    "status": _fmt_status(record),
                }
            )
            continue
        dense_ms = _metric(record, "dense_decode_ms_per_step")
        statecache_ms = _metric(record, "deltanet_statecache_decode_ms_per_step")
        fixed_dense = _metric(record, "deltanet_dense_fixed_resident_bytes")
        fixed_sc = _metric(record, "deltanet_statecache_fixed_resident_bytes")
        fixed_saved = None if fixed_dense is None or fixed_sc is None else fixed_dense - fixed_sc
        rows.append(
            {
                "mode": "compare",
                "model_id": record.get("model_id"),
                "prompt_length": int(record.get("prompt_length") or 0),
                "weight_quantization": record.get("weight_quantization"),
                "status": _fmt_status(record),
                "agreement": _metric(record, "deltanet_statecache_greedy_token_agreement_rate"),
                "dense_tps": _tok_per_s(dense_ms),
                "statecache_tps": _tok_per_s(statecache_ms),
                "speedup": None if dense_ms in (None, 0.0) or statecache_ms is None else dense_ms / statecache_ms,
                "fixed_saved_pct": _pct(fixed_saved, fixed_dense),
                "fixed_dense_mb": None if fixed_dense is None else fixed_dense / 1_000_000.0,
                "fixed_statecache_mb": None if fixed_sc is None else fixed_sc / 1_000_000.0,
                "output_max_abs_error": _metric(record, "deltanet_statecache_output_max_abs_error"),
            }
        )
    return sorted(rows, key=lambda row: (row["model_id"] or "", row.get("prompt_length") or 0))


def _build_serving_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        if record.get("benchmark") != "qwen35_deltanet_statecache_serving":
            continue
        if record.get("status") == "error":
            rows.append(
                {
                    "mode": "serving",
                    "model_id": record.get("model_id"),
                    "prompt_length": record.get("prompt_length"),
                    "weight_quantization": record.get("weight_quantization"),
                    "status": _fmt_status(record),
                }
            )
            continue
        statecache_ms = _metric(record, "deltanet_statecache_decode_ms_per_step")
        fixed_dense = _metric(record, "deltanet_dense_fixed_resident_bytes")
        fixed_sc = _metric(record, "deltanet_statecache_fixed_resident_bytes")
        total = _metric(record, "hybrid_state_total_bytes")
        token = _metric(record, "hybrid_token_growing_bytes")
        fixed_saved = None if fixed_dense is None or fixed_sc is None else fixed_dense - fixed_sc
        total_saved = None if total is None or token is None or fixed_sc is None else total - (token + fixed_sc)
        rows.append(
            {
                "mode": "serving",
                "model_id": record.get("model_id"),
                "prompt_length": int(record.get("prompt_length") or 0),
                "weight_quantization": record.get("weight_quantization"),
                "status": _fmt_status(record),
                "statecache_tps": _tok_per_s(statecache_ms),
                "statecache_ms": statecache_ms,
                "prefill_peak_alloc_gb": None
                if _metric(record, "deltanet_statecache_prefill_cuda_peak_memory_allocated_bytes") is None
                else _metric(record, "deltanet_statecache_prefill_cuda_peak_memory_allocated_bytes") / 1_000_000_000.0,
                "decode_peak_alloc_gb": None
                if _metric(record, "deltanet_statecache_decode_cuda_peak_memory_allocated_bytes") is None
                else _metric(record, "deltanet_statecache_decode_cuda_peak_memory_allocated_bytes") / 1_000_000_000.0,
                "fixed_saved_pct": _pct(fixed_saved, fixed_dense),
                "total_saved_pct": _pct(total_saved, total),
                "fixed_dense_mb": None if fixed_dense is None else fixed_dense / 1_000_000.0,
                "fixed_statecache_mb": None if fixed_sc is None else fixed_sc / 1_000_000.0,
                "token_mb": None if token is None else token / 1_000_000.0,
            }
        )
    return sorted(rows, key=lambda row: (row["model_id"] or "", row.get("prompt_length") or 0))


def _render_markdown(compare_rows: list[dict[str, Any]], serving_rows: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    lines.append("# Qwen3.5 StateCache Showcase")
    lines.append("")
    lines.append("## Compare-Mode")
    lines.append("")
    lines.append("| Model | Prompt | Weights | Status | Agreement | Dense tok/s | StateCache tok/s | Speedup | Fixed Saving | Fixed Dense MB | Fixed StateCache MB | Output Max Abs Error |")
    lines.append("|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in compare_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("model_id") or "-"),
                    str(row.get("prompt_length") or "-"),
                    str(row.get("weight_quantization") or "-"),
                    str(row.get("status") or "-"),
                    _fmt_num(row.get("agreement")),
                    _fmt_num(row.get("dense_tps")),
                    _fmt_num(row.get("statecache_tps")),
                    _fmt_num(row.get("speedup")),
                    _fmt_pct(row.get("fixed_saved_pct")),
                    _fmt_num(row.get("fixed_dense_mb")),
                    _fmt_num(row.get("fixed_statecache_mb")),
                    _fmt_num(row.get("output_max_abs_error")),
                ]
            )
            + " |"
        )
    lines.append("")
    lines.append("## Serving-Mode")
    lines.append("")
    lines.append("| Model | Prompt | Weights | Status | StateCache tok/s | Decode ms/step | Prefill Peak GB | Decode Peak GB | Fixed Saving | Total Saving | Fixed Dense MB | Fixed StateCache MB | Token-Growing MB |")
    lines.append("|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in serving_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("model_id") or "-"),
                    str(row.get("prompt_length") or "-"),
                    str(row.get("weight_quantization") or "-"),
                    str(row.get("status") or "-"),
                    _fmt_num(row.get("statecache_tps")),
                    _fmt_num(row.get("statecache_ms")),
                    _fmt_num(row.get("prefill_peak_alloc_gb")),
                    _fmt_num(row.get("decode_peak_alloc_gb")),
                    _fmt_pct(row.get("fixed_saved_pct")),
                    _fmt_pct(row.get("total_saved_pct")),
                    _fmt_num(row.get("fixed_dense_mb")),
                    _fmt_num(row.get("fixed_statecache_mb")),
                    _fmt_num(row.get("token_mb")),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    records: list[dict[str, Any]] = []
    for input_path in args.input:
        records.extend(_load_jsonl_records(Path(input_path)))
    compare_rows = _build_compare_rows(records)
    serving_rows = _build_serving_rows(records)
    payload = {
        "compare_mode": compare_rows,
        "serving_mode": serving_rows,
    }
    if args.format == "json":
        print(json.dumps(payload, indent=2, sort_keys=True))
        return
    print(_render_markdown(compare_rows, serving_rows))


if __name__ == "__main__":
    main()
