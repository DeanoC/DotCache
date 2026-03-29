#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize Qwen3.5 DotCache serving profile breakdowns.")
    parser.add_argument(
        "--input",
        default="benchmarks/results/qwen35_serving_sweep_20260329/qwen35_0p8b_dotcache_serving_sweep.jsonl",
        help="DotCache serving JSONL artifact.",
    )
    parser.add_argument("--format", choices=["markdown", "json"], default="markdown")
    return parser.parse_args()


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw_line in path.read_bytes().splitlines():
        stripped = raw_line.decode("utf-8", errors="ignore").replace("\x00", "").strip()
        if not stripped:
            continue
        payload = json.loads(stripped)
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def _fmt(value: float | None, digits: int = 2) -> str:
    if value is None:
        return "-"
    return f"{value:.{digits}f}"


def _mib(value: int | float | None) -> float | None:
    if value is None:
        return None
    return float(value) / (1024.0 * 1024.0)


def _build_summary(path: Path, records: list[dict[str, Any]]) -> dict[str, Any]:
    ok_records = [record for record in records if record.get("status", "ok") == "ok"]
    error_records = [record for record in records if record.get("status") == "error"]
    if any(record.get("prompt_mode") == "exact_length" for record in ok_records):
        ok_records = [record for record in ok_records if record.get("prompt_mode") == "exact_length"]
    if any(record.get("prompt_mode") == "exact_length" for record in error_records):
        error_records = [record for record in error_records if record.get("prompt_mode") == "exact_length"]
    ok_records.sort(key=lambda record: int(record.get("prompt_length") or 0))

    breakdown_rows: list[dict[str, Any]] = []
    backend_rows: list[dict[str, Any]] = []
    scaling_rows: list[dict[str, Any]] = []
    previous: dict[str, Any] | None = None
    previous_backend: dict[str, Any] | None = None
    for record in ok_records:
        steps = max(int(record.get("decode_steps") or 1), 1)
        step_ms = float(record["dotcache_decode_ms_per_step"])
        decode_runtime_ms_step = float(record.get("dotcache_decode_runtime_ms_total", 0.0)) / steps
        append_ms_step = float(record.get("dotcache_append_runtime_ms_total", 0.0)) / steps
        qkv_ms_step = float(record.get("dotcache_qkv_projection_ms_total", 0.0)) / steps
        output_ms_step = float(record.get("dotcache_output_projection_ms_total", 0.0)) / steps
        other_ms_step = step_ms - (decode_runtime_ms_step + append_ms_step + qkv_ms_step + output_ms_step)
        resident_mib = _mib(record.get("resident_bytes"))
        total_device_mib = _mib(
            max(
                int(record.get("dotcache_prefill_cuda_peak_memory_allocated_bytes") or 0),
                int(record.get("dotcache_decode_cuda_peak_memory_allocated_bytes") or 0),
            )
        )
        breakdown_row = {
            "context": int(record["prompt_length"]),
            "step_ms": step_ms,
            "decode_runtime_ms_step": decode_runtime_ms_step,
            "append_ms_step": append_ms_step,
            "qkv_ms_step": qkv_ms_step,
            "output_ms_step": output_ms_step,
            "other_ms_step": other_ms_step,
            "decode_runtime_share_pct": 100.0 * decode_runtime_ms_step / max(step_ms, 1e-8),
            "resident_mib": resident_mib,
            "total_device_mib": total_device_mib,
            "k_total_static_pages": int(record.get("k_total_static_pages") or 0),
            "v_total_static_pages": int(record.get("v_total_static_pages") or 0),
        }
        breakdown_rows.append(breakdown_row)
        trace = record.get("decode_backend_trace")
        if isinstance(trace, dict):
            prepare_ms_step = float(trace.get("prepare_ms_total", 0.0)) / steps
            score_ms_step = float(trace.get("score_ms_total", 0.0)) / steps
            softmax_ms_step = float(trace.get("softmax_ms_total", 0.0)) / steps
            mix_ms_step = float(trace.get("mix_ms_total", 0.0)) / steps
            chunk_assembly_ms_step = float(trace.get("chunk_assembly_ms_total", 0.0)) / steps
            unpack_ms_step = float(trace.get("unpack_ms_total", 0.0)) / steps
            fwht_ms_step = float(trace.get("fwht_ms_total", 0.0)) / steps
            backend_accounted_ms_step = (
                prepare_ms_step
                + score_ms_step
                + softmax_ms_step
                + mix_ms_step
                + chunk_assembly_ms_step
                + unpack_ms_step
                + fwht_ms_step
            )
            backend_rows.append(
                {
                    "context": int(record["prompt_length"]),
                    "prepare_ms_step": prepare_ms_step,
                    "score_ms_step": score_ms_step,
                    "softmax_ms_step": softmax_ms_step,
                    "mix_ms_step": mix_ms_step,
                    "chunk_assembly_ms_step": chunk_assembly_ms_step,
                    "unpack_ms_step": unpack_ms_step,
                    "fwht_ms_step": fwht_ms_step,
                    "backend_other_ms_step": max(decode_runtime_ms_step - backend_accounted_ms_step, 0.0),
                    "host_to_device_mib_step": _mib(trace.get("host_to_device_bytes")) / steps if trace.get("host_to_device_bytes") is not None else None,
                    "payload_read_mib_step": _mib(trace.get("payload_bytes_read")) / steps if trace.get("payload_bytes_read") is not None else None,
                    "metadata_read_kib_step": (
                        float(trace.get("metadata_bytes_read", 0.0)) / (1024.0 * steps)
                        if trace.get("metadata_bytes_read") is not None
                        else None
                    ),
                    "prepared_page_cache_hit_rate_pct": (
                        100.0
                        * float(trace.get("prepared_page_cache_hits", 0))
                        / max(float(trace.get("prepared_page_cache_hits", 0)) + float(trace.get("prepared_page_cache_misses", 0)), 1.0)
                    ),
                    "prepared_page_cache_resident_mib": _mib(trace.get("cache_resident_bytes")),
                    "max_temporary_mib": _mib(trace.get("max_temporary_bytes")),
                }
            )
        if previous is not None:
            scaling_rows.append(
                {
                    "from_context": int(previous["context"]),
                    "to_context": int(breakdown_row["context"]),
                    "context_scale": float(breakdown_row["context"] / previous["context"]),
                    "step_scale": float(breakdown_row["step_ms"] / previous["step_ms"]),
                    "decode_runtime_scale": float(
                        breakdown_row["decode_runtime_ms_step"] / max(previous["decode_runtime_ms_step"], 1e-8)
                    ),
                    "resident_scale": (
                        None
                        if previous["resident_mib"] in (None, 0.0) or breakdown_row["resident_mib"] is None
                        else float(breakdown_row["resident_mib"] / previous["resident_mib"])
                    ),
                }
            )
        previous = breakdown_row
        if backend_rows:
            current_backend = backend_rows[-1]
            if previous_backend is not None and current_backend["context"] != previous_backend["context"]:
                scaling_rows[-1].update(
                    {
                        "score_scale": float(current_backend["score_ms_step"] / max(previous_backend["score_ms_step"], 1e-8)),
                        "mix_scale": float(current_backend["mix_ms_step"] / max(previous_backend["mix_ms_step"], 1e-8)),
                        "prepare_scale": float(current_backend["prepare_ms_step"] / max(previous_backend["prepare_ms_step"], 1e-8)),
                    }
                )
            previous_backend = current_backend

    return {
        "input_path": str(path),
        "breakdown_rows": breakdown_rows,
        "backend_rows": backend_rows,
        "scaling_rows": scaling_rows,
        "error_contexts": [
            {
                "context": int(record.get("prompt_length") or 0),
                "error_type": record.get("error_type"),
                "error_message": record.get("error_message"),
            }
            for record in error_records
        ],
    }


def _render_markdown(summary: dict[str, Any]) -> str:
    lines = ["# Qwen3.5 DotCache Serving Profile", ""]
    lines.append("## Breakdown")
    lines.append("")
    lines.append(
        "| Context | Step ms | DotCache decode ms | Append ms | QKV ms | Output ms | Other ms | Decode share | Resident MiB | Total Device MiB | K Pages | V Pages |"
    )
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in summary["breakdown_rows"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["context"]),
                    _fmt(row["step_ms"]),
                    _fmt(row["decode_runtime_ms_step"]),
                    _fmt(row["append_ms_step"]),
                    _fmt(row["qkv_ms_step"]),
                    _fmt(row["output_ms_step"]),
                    _fmt(row["other_ms_step"]),
                    _fmt(row["decode_runtime_share_pct"]),
                    _fmt(row["resident_mib"]),
                    _fmt(row["total_device_mib"]),
                    str(row["k_total_static_pages"]),
                    str(row["v_total_static_pages"]),
                ]
            )
            + " |"
        )
    lines.append("")
    if summary["backend_rows"]:
        lines.append("## Backend Decode Stages")
        lines.append("")
        lines.append(
            "| Context | Prepare ms | Score ms | Softmax ms | Mix ms | Chunk Assembly ms | Unpack ms | FWHT ms | Backend Other ms | H2D MiB/step | Payload MiB/step | Metadata KiB/step | Cache Hit % | Cache Resident MiB | Max Temporary MiB |"
        )
        lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for row in summary["backend_rows"]:
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(row["context"]),
                        _fmt(row["prepare_ms_step"]),
                        _fmt(row["score_ms_step"]),
                        _fmt(row["softmax_ms_step"]),
                        _fmt(row["mix_ms_step"]),
                        _fmt(row["chunk_assembly_ms_step"]),
                        _fmt(row["unpack_ms_step"]),
                        _fmt(row["fwht_ms_step"]),
                        _fmt(row["backend_other_ms_step"]),
                        _fmt(row["host_to_device_mib_step"]),
                        _fmt(row["payload_read_mib_step"]),
                        _fmt(row["metadata_read_kib_step"]),
                        _fmt(row["prepared_page_cache_hit_rate_pct"]),
                        _fmt(row["prepared_page_cache_resident_mib"]),
                        _fmt(row["max_temporary_mib"]),
                    ]
                )
                + " |"
            )
        lines.append("")
    lines.append("## Scaling")
    lines.append("")
    lines.append("| Context Range | Context x | Step x | DotCache decode x | Score x | Mix x | Prepare x | Resident x |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for row in summary["scaling_rows"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    f"{row['from_context']} -> {row['to_context']}",
                    _fmt(row["context_scale"]),
                    _fmt(row["step_scale"]),
                    _fmt(row["decode_runtime_scale"]),
                    _fmt(row.get("score_scale")),
                    _fmt(row.get("mix_scale")),
                    _fmt(row.get("prepare_scale")),
                    _fmt(row["resident_scale"]),
                ]
            )
            + " |"
        )
    if summary["error_contexts"]:
        lines.append("")
        lines.append("## Errors")
        lines.append("")
        for row in summary["error_contexts"]:
            lines.append(f"- `{row['context']}`: `{row.get('error_type') or 'error'}`")
    return "\n".join(lines)


if __name__ == "__main__":
    args = parse_args()
    path = Path(args.input)
    summary = _build_summary(path, _load_jsonl(path))
    if args.format == "json":
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print(_render_markdown(summary))
