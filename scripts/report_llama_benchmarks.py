#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize comparable Llama benchmark records across torch_mps and torch_cuda."
    )
    parser.add_argument(
        "--input",
        action="append",
        default=None,
        help="JSONL input. Supports benchmark history entries and raw benchmark records.",
    )
    parser.add_argument("--benchmark", choices=["llama_compare", "llama_loss"], required=True)
    parser.add_argument("--model-id", default=None, help="Optional exact model-id filter.")
    parser.add_argument(
        "--format",
        choices=["markdown", "json"],
        default="markdown",
        help="Output format.",
    )
    return parser.parse_args()


def _load_jsonl_records(path: Path) -> list[dict[str, Any]]:
    flattened: list[dict[str, Any]] = []
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
        if not isinstance(payload, dict):
            continue
        meta = {
            "entry_recorded_at": payload.get("recorded_at"),
            "entry_commit": payload.get("commit"),
            "entry_branch": payload.get("branch"),
            "entry_label": payload.get("label"),
            "entry_notes": payload.get("notes"),
            "entry_command": payload.get("command"),
        }
        if isinstance(payload.get("records"), list):
            for record in payload["records"]:
                if isinstance(record, dict):
                    merged = dict(record)
                    merged.update(meta)
                    flattened.append(merged)
        elif "benchmark" in payload:
            merged = dict(payload)
            merged.update(meta)
            flattened.append(merged)
    return flattened


def _normalized_record(record: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(record)
    defaults = {
        "default_mode_k": "M0",
        "default_mode_v": "M0",
        "quant_scheme_k": "affine",
        "quant_scheme_v": "affine",
        "tokens_per_page": 256,
        "torch_dtype": "float16",
    }
    for key, value in defaults.items():
        if normalized.get(key) is None:
            normalized[key] = value
    return normalized


def _case_key(record: dict[str, Any], benchmark: str) -> tuple[Any, ...]:
    common = (
        record.get("model_id"),
        record.get("default_mode_k"),
        record.get("quant_scheme_k"),
        record.get("default_mode_v"),
        record.get("quant_scheme_v"),
        record.get("tokens_per_page"),
    )
    if benchmark == "llama_compare":
        return (
            benchmark,
            *common,
            record.get("prompt_length"),
        )
    return (
        benchmark,
        *common,
        record.get("sequence_length"),
        record.get("prefix_length"),
        record.get("eval_steps"),
    )


def _case_label(record: dict[str, Any], benchmark: str) -> str:
    if benchmark == "llama_compare":
        return (
            f"prompt={record.get('prompt_length')} "
            f"K={record.get('default_mode_k')}/{record.get('quant_scheme_k')} "
            f"V={record.get('default_mode_v')}/{record.get('quant_scheme_v')}"
        )
    return (
        f"seq={record.get('sequence_length')} prefix={record.get('prefix_length')} eval={record.get('eval_steps')} "
        f"K={record.get('default_mode_k')}/{record.get('quant_scheme_k')} "
        f"V={record.get('default_mode_v')}/{record.get('quant_scheme_v')}"
    )


def _metric(record: dict[str, Any] | None, key: str) -> float | None:
    if record is None:
        return None
    value = record.get(key)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _format_number(value: float | None, digits: int = 2) -> str:
    if value is None:
        return "-"
    return f"{value:.{digits}f}"


def build_rows(records: list[dict[str, Any]], benchmark: str, model_id: str | None) -> list[dict[str, Any]]:
    filtered = [
        _normalized_record(record)
        for record in records
        if record.get("benchmark") == benchmark and (model_id is None or record.get("model_id") == model_id)
    ]
    grouped: dict[tuple[Any, ...], dict[str, dict[str, Any]]] = {}
    for record in filtered:
        backend = record.get("backend")
        if backend not in {"torch_mps", "torch_cuda"}:
            continue
        case = grouped.setdefault(_case_key(record, benchmark), {})
        current = case.get(backend)
        if current is None or (record.get("entry_recorded_at") or "") >= (current.get("entry_recorded_at") or ""):
            case[backend] = record

    rows: list[dict[str, Any]] = []
    for case_backends in grouped.values():
        sample = case_backends.get("torch_cuda") or case_backends.get("torch_mps")
        assert sample is not None
        mps = case_backends.get("torch_mps")
        cuda = case_backends.get("torch_cuda")
        if benchmark == "llama_compare":
            agreement_key = "greedy_token_agreement_rate"
            primary_key = "decode_ms_per_step"
            secondary_key = "dense_decode_ms_per_step"
            tertiary_key = "dotcache_vs_dense_kv_bytes_ratio"
            aux_key = "prefill_cache_ingest_ms"
        else:
            agreement_key = "teacher_forced_token_agreement_rate"
            primary_key = "dotcache_decode_ms_per_step"
            secondary_key = "dense_decode_ms_per_step"
            tertiary_key = "teacher_forced_loss_delta"
            aux_key = "prefill_ms"
        row = {
            "model_id": sample.get("model_id"),
            "case": _case_label(sample, benchmark),
            "mps_dotcache_ms": _metric(mps, primary_key),
            "mps_dense_ms": _metric(mps, secondary_key),
            "mps_agreement": _metric(mps, agreement_key),
            "mps_aux": _metric(mps, aux_key),
            "mps_tertiary": _metric(mps, tertiary_key),
            "mps_recorded_at": None if mps is None else mps.get("entry_recorded_at"),
            "cuda_dotcache_ms": _metric(cuda, primary_key),
            "cuda_dense_ms": _metric(cuda, secondary_key),
            "cuda_agreement": _metric(cuda, agreement_key),
            "cuda_aux": _metric(cuda, aux_key),
            "cuda_tertiary": _metric(cuda, tertiary_key),
            "cuda_recorded_at": None if cuda is None else cuda.get("entry_recorded_at"),
        }
        row["cuda_vs_mps_dotcache"] = (
            None
            if row["cuda_dotcache_ms"] is None or row["mps_dotcache_ms"] in (None, 0.0)
            else row["cuda_dotcache_ms"] / row["mps_dotcache_ms"]
        )
        rows.append(row)

    rows.sort(key=lambda row: (row["model_id"] or "", row["case"]))
    return rows


def render_markdown(rows: list[dict[str, Any]], benchmark: str) -> str:
    if benchmark == "llama_compare":
        aux_label = "Prefill Ingest ms"
        tertiary_label = "KV Ratio"
    else:
        aux_label = "Prefill ms"
        tertiary_label = "Loss Delta"
    header = (
        "| Model | Case | MPS Dense | MPS DotCache | CUDA Dense | CUDA DotCache | CUDA/MPS DotCache | "
        f"MPS {tertiary_label} | CUDA {tertiary_label} | MPS Agreement | CUDA Agreement | MPS {aux_label} | CUDA {aux_label} |\n"
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
    )
    lines = [header]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["model_id"]),
                    str(row["case"]),
                    _format_number(row["mps_dense_ms"]),
                    _format_number(row["mps_dotcache_ms"]),
                    _format_number(row["cuda_dense_ms"]),
                    _format_number(row["cuda_dotcache_ms"]),
                    _format_number(row["cuda_vs_mps_dotcache"]),
                    _format_number(row["mps_tertiary"], digits=3),
                    _format_number(row["cuda_tertiary"], digits=3),
                    _format_number(row["mps_agreement"], digits=3),
                    _format_number(row["cuda_agreement"], digits=3),
                    _format_number(row["mps_aux"]),
                    _format_number(row["cuda_aux"]),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    input_paths = args.input or ["benchmarks/results/history.jsonl"]
    records: list[dict[str, Any]] = []
    for raw_path in input_paths:
        records.extend(_load_jsonl_records(Path(raw_path)))
    rows = build_rows(records, benchmark=args.benchmark, model_id=args.model_id)
    if args.format == "json":
        print(json.dumps(rows, sort_keys=True))
    else:
        print(render_markdown(rows, benchmark=args.benchmark))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
