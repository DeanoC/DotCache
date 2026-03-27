#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from report_model_benchmarks import COMPARE_BENCHMARKS, _load_jsonl_records, _normalized_record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize model-specific KV key compressibility tolerance from recorded compare benchmarks."
        )
    )
    parser.add_argument(
        "--input",
        action="append",
        default=None,
        help="JSONL input. Supports benchmark history entries and raw benchmark records.",
    )
    parser.add_argument(
        "--benchmark",
        action="append",
        choices=sorted(COMPARE_BENCHMARKS),
        default=None,
        help="Optional compare benchmark filter. May be repeated.",
    )
    parser.add_argument("--backend", choices=["torch_cuda", "torch_mps"], default="torch_cuda")
    parser.add_argument("--model-id", default=None, help="Optional exact model-id filter.")
    parser.add_argument(
        "--agreement-threshold",
        type=float,
        default=0.99,
        help="Agreement threshold used for stability classification.",
    )
    parser.add_argument("--format", choices=["markdown", "json"], default="markdown")
    return parser.parse_args()


def _load_records(input_paths: list[str]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for raw_path in input_paths:
        records.extend(_load_jsonl_records(Path(raw_path)))
    return records


def _policy_kind(record: dict[str, Any]) -> str:
    key_overrides = list(record.get("key_mode_overrides") or [])
    value_overrides = list(record.get("value_mode_overrides") or [])
    if key_overrides or value_overrides:
        return "selective"
    if record.get("default_mode_k") == "M3":
        return "exact_k"
    if record.get("default_mode_k") == "M0" and record.get("default_mode_v") == "M0":
        return "all_m0"
    return "other"


def _policy_label(record: dict[str, Any]) -> str:
    key_overrides = list(record.get("key_mode_overrides") or [])
    value_overrides = list(record.get("value_mode_overrides") or [])
    if key_overrides or value_overrides:
        parts = ["selective"]
        if key_overrides:
            parts.append(f"K*={','.join(key_overrides)}")
        if value_overrides:
            parts.append(f"V*={','.join(value_overrides)}")
        return " ".join(parts)
    if record.get("default_mode_k") == "M3":
        return "exact K"
    if record.get("default_mode_k") == "M0" and record.get("default_mode_v") == "M0":
        return "all M0"
    return (
        f"K={record.get('default_mode_k')}/{record.get('quant_scheme_k')} "
        f"V={record.get('default_mode_v')}/{record.get('quant_scheme_v')}"
    )


def _metric(record: dict[str, Any] | None, key: str) -> float | None:
    if record is None or record.get("status") == "error":
        return None
    value = record.get(key)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _k_exact_fraction(record: dict[str, Any]) -> float | None:
    total_pages = record.get("k_total_static_pages")
    exact_pages = record.get("k_m3_pages")
    try:
        total_pages_int = int(total_pages)
    except (TypeError, ValueError):
        total_pages_int = 0
    try:
        exact_pages_int = int(exact_pages)
    except (TypeError, ValueError):
        exact_pages_int = 0
    if total_pages_int > 0:
        return float(exact_pages_int / total_pages_int)
    if record.get("status") == "error":
        return None
    if record.get("default_mode_k") == "M3":
        return 1.0
    if record.get("default_mode_k") == "M0":
        return 0.0
    return None


def _tok_per_sec(record: dict[str, Any]) -> float | None:
    decode_ms = _metric(record, "decode_ms_per_step")
    if decode_ms in (None, 0.0):
        return None
    return float(1000.0 / decode_ms)


def _status_label(record: dict[str, Any]) -> str:
    return "error" if record.get("status") == "error" else "ok"


def _classification_label(
    records_by_policy: dict[str, dict[str, Any]],
    *,
    agreement_threshold: float,
) -> tuple[str, dict[str, Any] | None]:
    all_m0 = records_by_policy.get("all_m0")
    selective = records_by_policy.get("selective")
    exact_k = records_by_policy.get("exact_k")
    all_m0_agreement = _metric(all_m0, "greedy_token_agreement_rate")
    selective_agreement = _metric(selective, "greedy_token_agreement_rate")
    exact_agreement = _metric(exact_k, "greedy_token_agreement_rate")

    if all_m0_agreement is not None and all_m0_agreement >= agreement_threshold:
        return "tolerates all-M0", all_m0
    if selective_agreement is not None and selective_agreement >= agreement_threshold:
        return "benefits from selective exact K", selective
    if (
        exact_agreement is not None
        and exact_agreement >= agreement_threshold
        and (all_m0 is not None or selective is not None)
    ):
        return "needs global exact K", exact_k
    return "unknown", selective or exact_k or all_m0


def build_report(
    records: list[dict[str, Any]],
    *,
    backend: str,
    benchmarks: list[str] | None,
    model_id: str | None,
    agreement_threshold: float,
) -> dict[str, Any]:
    benchmark_filter = set(benchmarks or COMPARE_BENCHMARKS)
    filtered = [
        _normalized_record(record)
        for record in records
        if record.get("benchmark") in benchmark_filter
        and record.get("backend") == backend
        and (model_id is None or record.get("model_id") == model_id)
    ]
    grouped: dict[tuple[Any, ...], dict[str, dict[str, Any]]] = {}
    for record in filtered:
        case_key = (record.get("benchmark"), record.get("model_id"), record.get("prompt_length"), record.get("backend"))
        kind = _policy_kind(record)
        case = grouped.setdefault(case_key, {})
        current = case.get(kind)
        if current is None or (record.get("entry_recorded_at") or "") >= (current.get("entry_recorded_at") or ""):
            case[kind] = record

    summary_rows: list[dict[str, Any]] = []
    policy_rows: list[dict[str, Any]] = []
    for (benchmark, case_model_id, prompt_length, case_backend), records_by_policy in sorted(grouped.items()):
        exact_record = records_by_policy.get("exact_k")
        exact_kv_resident = _metric(exact_record, "kv_resident_bytes")
        exact_kv_ratio = _metric(exact_record, "dotcache_vs_dense_kv_bytes_ratio")
        classification, best_record = _classification_label(
            records_by_policy,
            agreement_threshold=agreement_threshold,
        )

        if best_record is not None:
            best_k_exact = _k_exact_fraction(best_record)
            best_kv_resident = _metric(best_record, "kv_resident_bytes")
            best_kv_ratio = _metric(best_record, "dotcache_vs_dense_kv_bytes_ratio")
            best_vs_exact = None
            if best_kv_resident is not None and exact_kv_resident not in (None, 0.0):
                best_vs_exact = best_kv_resident / exact_kv_resident
            elif best_kv_ratio is not None and exact_kv_ratio not in (None, 0.0):
                best_vs_exact = best_kv_ratio / exact_kv_ratio
            summary_rows.append(
                {
                    "benchmark": benchmark,
                    "backend": case_backend,
                    "model_id": case_model_id,
                    "prompt_length": prompt_length,
                    "classification": classification,
                    "best_policy": _policy_label(best_record),
                    "k_exact_fraction": best_k_exact,
                    "agreement": _metric(best_record, "greedy_token_agreement_rate"),
                    "kv_vs_exact_k": best_vs_exact,
                    "kv_vs_dense": best_kv_ratio,
                    "decode_tok_per_s": _tok_per_sec(best_record),
                    "status": _status_label(best_record),
                }
            )

        for kind, record in sorted(records_by_policy.items(), key=lambda item: ("all_m0", "selective", "exact_k", "other").index(item[0]) if item[0] in {"all_m0", "selective", "exact_k", "other"} else 99):
            kv_resident = _metric(record, "kv_resident_bytes")
            kv_ratio = _metric(record, "dotcache_vs_dense_kv_bytes_ratio")
            kv_vs_exact = None
            if kv_resident is not None and exact_kv_resident not in (None, 0.0):
                kv_vs_exact = kv_resident / exact_kv_resident
            elif kv_ratio is not None and exact_kv_ratio not in (None, 0.0):
                kv_vs_exact = kv_ratio / exact_kv_ratio
            policy_rows.append(
                {
                    "benchmark": benchmark,
                    "backend": case_backend,
                    "model_id": case_model_id,
                    "prompt_length": prompt_length,
                    "policy_kind": kind,
                    "policy": _policy_label(record),
                    "k_exact_fraction": _k_exact_fraction(record),
                    "agreement": _metric(record, "greedy_token_agreement_rate"),
                    "kv_vs_exact_k": kv_vs_exact,
                    "kv_vs_dense": kv_ratio,
                    "decode_tok_per_s": _tok_per_sec(record),
                    "decode_ms_per_step": _metric(record, "decode_ms_per_step"),
                    "status": _status_label(record),
                }
            )

    return {
        "backend": backend,
        "agreement_threshold": agreement_threshold,
        "summary_rows": summary_rows,
        "policy_rows": policy_rows,
    }


def _fmt_pct(value: float | None, digits: int = 2) -> str:
    if value is None:
        return "-"
    return f"{value * 100:.{digits}f}%"


def _fmt_num(value: float | None, digits: int = 2) -> str:
    if value is None:
        return "-"
    return f"{value:.{digits}f}"


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        f"# Compressibility Profile ({report['backend']})",
        "",
        "## Classification",
        "| Model | Prompt | Classification | Best Policy | % K Exact | Agreement | KV vs Exact-K | KV vs Dense | Decode tok/s | Status |",
        "|---|---:|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in report["summary_rows"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["model_id"]),
                    str(row["prompt_length"]),
                    str(row["classification"]),
                    str(row["best_policy"]),
                    _fmt_pct(row["k_exact_fraction"]),
                    _fmt_num(row["agreement"], digits=3),
                    _fmt_num(row["kv_vs_exact_k"], digits=3) + "x" if row["kv_vs_exact_k"] is not None else "-",
                    _fmt_num(row["kv_vs_dense"], digits=3) + "x" if row["kv_vs_dense"] is not None else "-",
                    _fmt_num(row["decode_tok_per_s"], digits=2),
                    str(row["status"]),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Policy Rows",
            "| Model | Prompt | Policy | % K Exact | Agreement | KV vs Exact-K | KV vs Dense | Decode tok/s | Decode ms/step | Status |",
            "|---|---:|---|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in report["policy_rows"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["model_id"]),
                    str(row["prompt_length"]),
                    str(row["policy"]),
                    _fmt_pct(row["k_exact_fraction"]),
                    _fmt_num(row["agreement"], digits=3),
                    _fmt_num(row["kv_vs_exact_k"], digits=3) + "x" if row["kv_vs_exact_k"] is not None else "-",
                    _fmt_num(row["kv_vs_dense"], digits=3) + "x" if row["kv_vs_dense"] is not None else "-",
                    _fmt_num(row["decode_tok_per_s"], digits=2),
                    _fmt_num(row["decode_ms_per_step"], digits=2),
                    str(row["status"]),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    records = _load_records(args.input or ["benchmarks/results/history.jsonl"])
    report = build_report(
        records,
        backend=args.backend,
        benchmarks=args.benchmark,
        model_id=args.model_id,
        agreement_threshold=args.agreement_threshold,
    )
    if args.format == "json":
        print(json.dumps(report, sort_keys=True))
    else:
        print(render_markdown(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
