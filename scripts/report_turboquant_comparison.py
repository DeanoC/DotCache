#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


BENCHMARKS = {
    "turboquant_external",
    "llama_compare",
    "qwen35_text",
    "qwen35_deltanet_statecache_readout",
    "qwen35_attention_subset_statecache_dotcache",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize DotCache vs TurboQuant comparison JSONL.")
    parser.add_argument("--input", action="append", required=True, help="JSONL input file. Can be repeated.")
    parser.add_argument("--format", choices=["markdown", "json"], default="markdown")
    parser.add_argument(
        "--layout",
        choices=["rows", "context_matrix", "memory_matrix", "cache_memory_matrix", "device_memory_matrix"],
        default="rows",
    )
    return parser.parse_args()


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
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


def _fmt_num(value: float | None, digits: int = 2) -> str:
    if value is None:
        return "-"
    return f"{value:.{digits}f}"


def _fmt_pct(value: float | None, digits: int = 2) -> str:
    if value is None:
        return "-"
    return f"{value:.{digits}f}%"


def _fmt_mib(value: float | None, digits: int = 2) -> str:
    if value is None:
        return "-"
    return f"{value / (1024.0 * 1024.0):.{digits}f}"


def _error_label(error_type: str | None, error_message: str | None) -> str:
    error_type_text = (error_type or "").lower()
    error_message_text = (error_message or "").lower()
    if "timeout" in error_type_text or "timeout" in error_message_text:
        return "TIMEOUT"
    if "outofmemory" in error_type_text or "out of memory" in error_message_text or "oom" in error_message_text:
        return "OOM"
    if "assert" in error_message_text:
        return "ASSERT"
    if "quality_fail" in error_type_text or "quality fail" in error_message_text:
        return "QUALITY"
    return "ERROR"


def _metric_cell_text(row: dict[str, Any], metric_key: str, formatter) -> str:
    status = str(row.get("status") or "-")
    if status == "ok":
        metric_value = row.get(metric_key)
        if metric_value is not None:
            return formatter(metric_value)
        return "-"
    return _error_label(
        str(row.get("error_type") or ""),
        str(row.get("error_message") or ""),
    )


def _build_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        benchmark = record.get("benchmark")
        if benchmark == "turboquant_external":
            model_label = record.get("tokenizer_model_id") or record.get("model_id")
            context_value = int(record.get("requested_prompt_length") or record.get("prompt_length") or 0)
            mode = record.get("mode")
            if mode == "decode":
                rows.append(
                    {
                        "model": model_label,
                        "runtime": "llama.cpp_turboquant",
                        "config": record.get("config"),
                        "context": context_value,
                        "decode_ms": _metric(record, "decode_ms_per_step"),
                        "decode_tps": _metric(record, "eval_time_tokens_per_second"),
                        "ppl": None,
                        "agreement": None,
                        "status": record.get("status"),
                        "error_type": record.get("error_type"),
                        "error_message": record.get("error_message"),
                        "cache_memory_bytes": (
                            _metric(record, "memory_breakdown_all_devices_context_bytes")
                            or _metric(record, "memory_breakdown_device_context_bytes")
                        ),
                        "total_device_memory_bytes": (
                            _metric(record, "memory_breakdown_all_devices_self_bytes")
                            or _metric(record, "memory_breakdown_device_self_bytes")
                        ),
                    }
                )
            elif mode == "perplexity":
                rows.append(
                    {
                        "model": model_label,
                        "runtime": "llama.cpp_turboquant",
                        "config": f"{record.get('config')} ppl",
                        "context": int(record.get("perplexity_context") or 0),
                        "decode_ms": None,
                        "decode_tps": _metric(record, "eval_time_tokens_per_second"),
                        "ppl": _metric(record, "ppl"),
                        "agreement": None,
                        "status": record.get("status"),
                        "error_type": record.get("error_type"),
                        "error_message": record.get("error_message"),
                        "cache_memory_bytes": (
                            _metric(record, "memory_breakdown_all_devices_context_bytes")
                            or _metric(record, "memory_breakdown_device_context_bytes")
                        ),
                        "total_device_memory_bytes": (
                            _metric(record, "memory_breakdown_all_devices_self_bytes")
                            or _metric(record, "memory_breakdown_device_self_bytes")
                        ),
                    }
                )
        elif benchmark == "llama_compare":
            dense_decode_ms = _metric(record, "dense_decode_ms_per_step")
            rows.append(
                {
                    "model": record.get("model_id"),
                    "runtime": "hf_dense",
                    "config": "dense",
                    "context": int(record.get("prompt_length") or 0),
                    "decode_ms": dense_decode_ms,
                    "decode_tps": None if dense_decode_ms in (None, 0.0) else 1000.0 / dense_decode_ms,
                    "ppl": None,
                    "agreement": None,
                    "status": record.get("status", "ok"),
                    "error_type": record.get("error_type"),
                    "error_message": record.get("error_message"),
                    "cache_memory_bytes": _metric(record, "dense_final_kv_cache_bytes"),
                    "total_device_memory_bytes": None,
                }
            )
            default_mode_k = record.get("default_mode_k")
            default_mode_v = record.get("default_mode_v")
            config_label = "DotCache"
            if default_mode_k is not None and default_mode_v is not None:
                config_label = f"DotCache K={default_mode_k} / V={default_mode_v}"
            dotcache_decode_ms = _metric(record, "decode_ms_per_step")
            rows.append(
                {
                    "model": record.get("model_id"),
                    "runtime": "dotcache_hf",
                    "config": config_label,
                    "context": int(record.get("prompt_length") or 0),
                    "decode_ms": dotcache_decode_ms,
                    "decode_tps": None if dotcache_decode_ms in (None, 0.0) else 1000.0 / dotcache_decode_ms,
                    "ppl": None,
                    "agreement": _metric(record, "greedy_token_agreement_rate"),
                    "status": record.get("status", "ok"),
                    "error_type": record.get("error_type"),
                    "error_message": record.get("error_message"),
                    "cache_memory_bytes": _metric(record, "resident_bytes"),
                    "total_device_memory_bytes": None,
                }
            )
        elif benchmark == "qwen35_text":
            dense_decode_ms = _metric(record, "dense_decode_ms_per_step")
            rows.append(
                {
                    "model": record.get("model_id"),
                    "runtime": "hf_dense",
                    "config": "dense (shared harness)",
                    "context": int(record.get("prompt_length") or 0),
                    "decode_ms": dense_decode_ms,
                    "decode_tps": None if dense_decode_ms in (None, 0.0) else 1000.0 / dense_decode_ms,
                    "ppl": None,
                    "agreement": None,
                    "status": record.get("status", "ok"),
                    "error_type": record.get("error_type"),
                    "error_message": record.get("error_message"),
                    "cache_memory_bytes": _metric(record, "dense_final_cache_bytes"),
                    "total_device_memory_bytes": None,
                }
            )
        elif benchmark == "qwen35_deltanet_statecache_readout":
            dense_decode_ms = _metric(record, "dense_decode_ms_per_step")
            hybrid_state_total_bytes = _metric(record, "hybrid_state_total_bytes")
            rows.append(
                {
                    "model": record.get("model_id"),
                    "runtime": "hf_dense",
                    "config": "dense (statecache harness)",
                    "context": int(record.get("prompt_length") or 0),
                    "decode_ms": dense_decode_ms,
                    "decode_tps": None if dense_decode_ms in (None, 0.0) else 1000.0 / dense_decode_ms,
                    "ppl": None,
                    "agreement": None,
                    "status": record.get("status", "ok"),
                    "error_type": record.get("error_type"),
                    "error_message": record.get("error_message"),
                    "cache_memory_bytes": hybrid_state_total_bytes,
                    "total_device_memory_bytes": _metric(record, "deltanet_dense_capture_cuda_peak_memory_allocated_bytes"),
                }
            )
            statecache_decode_ms = _metric(record, "deltanet_statecache_decode_ms_per_step")
            bits = record.get("deltanet_statecache_bits")
            mode = record.get("deltanet_statecache_mode")
            config_label = "StateCache"
            if mode is not None and bits is not None:
                config_label = f"StateCache {mode} {bits}-bit"
            statecache_fixed_resident_bytes = _metric(record, "deltanet_statecache_fixed_resident_bytes")
            hybrid_token_growing_bytes = _metric(record, "hybrid_token_growing_bytes")
            rows.append(
                {
                    "model": record.get("model_id"),
                    "runtime": "statecache_hf",
                    "config": config_label,
                    "context": int(record.get("prompt_length") or 0),
                    "decode_ms": statecache_decode_ms,
                    "decode_tps": None if statecache_decode_ms in (None, 0.0) else 1000.0 / statecache_decode_ms,
                    "ppl": None,
                    "agreement": _metric(record, "deltanet_statecache_greedy_token_agreement_rate"),
                    "status": record.get("status", "ok") if record.get("deltanet_statecache_ready") is not False else "error",
                    "error_type": record.get("error_type"),
                    "error_message": record.get("error_message"),
                    "cache_memory_bytes": (
                        None
                        if statecache_fixed_resident_bytes is None or hybrid_token_growing_bytes is None
                        else statecache_fixed_resident_bytes + hybrid_token_growing_bytes
                    ),
                    "total_device_memory_bytes": max(
                        value
                        for value in (
                            _metric(record, "deltanet_statecache_prefill_cuda_peak_memory_allocated_bytes"),
                            _metric(record, "deltanet_statecache_decode_cuda_peak_memory_allocated_bytes"),
                        )
                        if value is not None
                    ) if any(
                        value is not None
                        for value in (
                            _metric(record, "deltanet_statecache_prefill_cuda_peak_memory_allocated_bytes"),
                            _metric(record, "deltanet_statecache_decode_cuda_peak_memory_allocated_bytes"),
                        )
                    ) else None,
                }
            )
        elif benchmark == "qwen35_attention_subset_statecache_dotcache":
            dense_decode_ms = _metric(record, "dense_decode_ms_per_step")
            rows.append(
                {
                    "model": record.get("model_id"),
                    "runtime": "hf_dense",
                    "config": "dense (hybrid harness)",
                    "context": int(record.get("prompt_length") or 0),
                    "decode_ms": dense_decode_ms,
                    "decode_tps": None if dense_decode_ms in (None, 0.0) else 1000.0 / dense_decode_ms,
                    "ppl": None,
                    "agreement": None,
                    "status": record.get("status", "ok"),
                    "error_type": record.get("error_type"),
                    "error_message": record.get("error_message"),
                    "cache_memory_bytes": None,
                    "total_device_memory_bytes": None,
                }
            )
            hybrid_decode_ms = _metric(record, "dotcache_decode_ms_per_step")
            bits_k = record.get("bits_k", record.get("default_bits_k"))
            bits_v = record.get("bits_v", record.get("default_bits_v"))
            state_bits = record.get("deltanet_statecache_bits")
            config_label = "Hybrid DotCache+StateCache"
            if bits_k is not None and bits_v is not None and state_bits is not None:
                config_label = f"Hybrid K={bits_k} / V={bits_v} + StateCache {state_bits}-bit"
            hybrid_resident_bytes = _metric(record, "resident_bytes")
            hybrid_fixed_resident_bytes = _metric(record, "deltanet_statecache_fixed_resident_bytes")
            rows.append(
                {
                    "model": record.get("model_id"),
                    "runtime": "hybrid_dotcache_statecache_hf",
                    "config": config_label,
                    "context": int(record.get("prompt_length") or 0),
                    "decode_ms": hybrid_decode_ms,
                    "decode_tps": None if hybrid_decode_ms in (None, 0.0) else 1000.0 / hybrid_decode_ms,
                    "ppl": None,
                    "agreement": None,
                    "status": record.get("status", "ok") if record.get("hybrid_dotcache_statecache_ready") is not False else "error",
                    "error_type": record.get("error_type"),
                    "error_message": record.get("error_message"),
                    "cache_memory_bytes": (
                        None
                        if hybrid_resident_bytes is None or hybrid_fixed_resident_bytes is None
                        else hybrid_resident_bytes + hybrid_fixed_resident_bytes
                    ),
                    "total_device_memory_bytes": None,
                }
            )
    return sorted(rows, key=lambda row: (str(row["model"]), str(row["runtime"]), int(row["context"]), str(row["config"])))


def _render_markdown(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# TurboQuant Comparison",
        "",
        "| Model | Runtime | Config | Context | Decode ms/step | Decode tok/s | Total MiB | PPL | Agreement | Status |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("model") or "-"),
                    str(row.get("runtime") or "-"),
                    str(row.get("config") or "-"),
                    str(row.get("context") or "-"),
                    _fmt_num(row.get("decode_ms")),
                    _fmt_num(row.get("decode_tps")),
                    _fmt_mib(row.get("cache_memory_bytes")),
                    _fmt_num(row.get("ppl"), 4),
                    _fmt_num(row.get("agreement")),
                    str(row.get("status") or "-"),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def _render_context_matrix(rows: list[dict[str, Any]], *, metric_key: str, formatter, title: str, unit_label: str) -> str:
    contexts = sorted({int(row.get("context") or 0) for row in rows if int(row.get("context") or 0) > 0})
    lines = [title, ""]
    header = ["Model", "Runtime", "Config", "Unit", *[str(context) for context in contexts]]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---", "---", "---", "---", *["---:" for _ in contexts]]) + "|")

    grouped: dict[tuple[str, str, str], dict[int, dict[str, Any]]] = {}
    for row in rows:
        key = (str(row.get("model") or "-"), str(row.get("runtime") or "-"), str(row.get("config") or "-"))
        grouped.setdefault(key, {})
        grouped[key][int(row.get("context") or 0)] = row

    # Prefer a single shared dense baseline over harness-specific dense baselines
    # when both are present for the same model.
    shared_dense_keys = {
        key
        for key in grouped
        if key[1] == "hf_dense" and key[2] == "dense (shared harness)"
    }
    dense_keys_to_remove: list[tuple[str, str, str]] = []
    for model, runtime, config in grouped:
        if runtime != "hf_dense":
            continue
        if config in {"dense (shared harness)", "dense"}:
            continue
        if (model, "hf_dense", "dense (shared harness)") in shared_dense_keys:
            dense_keys_to_remove.append((model, runtime, config))
    for key in dense_keys_to_remove:
        grouped.pop(key, None)

    # Some error-only sweep records omit detailed config metadata and surface as a generic
    # "StateCache" row. If there is exactly one specific StateCache config for the same model
    # and runtime, merge the generic contexts into that row instead of rendering a duplicate.
    generic_keys_to_remove: list[tuple[str, str, str]] = []
    for key, by_context in list(grouped.items()):
        model, runtime, config = key
        if runtime != "statecache_hf" or config != "StateCache":
            continue
        specific_candidates = [
            candidate_key
            for candidate_key in grouped
            if candidate_key[0] == model
            and candidate_key[1] == runtime
            and candidate_key[2].startswith("StateCache ")
        ]
        if len(specific_candidates) != 1:
            continue
        target = grouped[specific_candidates[0]]
        for context, row in by_context.items():
            target.setdefault(context, row)
        generic_keys_to_remove.append(key)
    for key in generic_keys_to_remove:
        grouped.pop(key, None)

    for (model, runtime, config), by_context in sorted(grouped.items()):
        cells = [model, runtime, config, unit_label]
        for context in contexts:
            row = by_context.get(context)
            cells.append("-" if row is None else _metric_cell_text(row, metric_key, formatter))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    records: list[dict[str, Any]] = []
    for input_path in args.input:
        records.extend(_load_jsonl(Path(input_path)))
    rows = _build_rows(records)
    if args.format == "json":
        print(json.dumps({"rows": rows}, indent=2, sort_keys=True))
        return
    if args.layout == "context_matrix":
        print(
            _render_context_matrix(
                rows,
                metric_key="decode_tps",
                formatter=_fmt_num,
                title="# TurboQuant Comparison",
                unit_label="tok/s",
            )
        )
        return
    if args.layout == "memory_matrix":
        print(
            _render_context_matrix(
                rows,
                metric_key="cache_memory_bytes",
                formatter=_fmt_mib,
                title="# TurboQuant Cache/State Memory Comparison",
                unit_label="MiB",
            )
        )
        return
    if args.layout == "cache_memory_matrix":
        print(
            _render_context_matrix(
                rows,
                metric_key="cache_memory_bytes",
                formatter=_fmt_mib,
                title="# TurboQuant Cache/State Memory Comparison",
                unit_label="MiB",
            )
        )
        return
    if args.layout == "device_memory_matrix":
        print(
            _render_context_matrix(
                rows,
                metric_key="total_device_memory_bytes",
                formatter=_fmt_mib,
                title="# TurboQuant Total Device Memory Comparison",
                unit_label="MiB",
            )
        )
        return
    print(_render_markdown(rows))


if __name__ == "__main__":
    main()
