#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


BENCHMARKS = {"turboquant_external", "llama_compare"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize DotCache vs TurboQuant comparison JSONL.")
    parser.add_argument("--input", action="append", required=True, help="JSONL input file. Can be repeated.")
    parser.add_argument("--format", choices=["markdown", "json"], default="markdown")
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


def _build_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        benchmark = record.get("benchmark")
        if benchmark == "turboquant_external":
            mode = record.get("mode")
            if mode == "decode":
                rows.append(
                    {
                        "model": record.get("model_id"),
                        "runtime": "llama.cpp_turboquant",
                        "config": record.get("config"),
                        "context": int(record.get("prompt_length") or 0),
                        "decode_ms": _metric(record, "decode_ms_per_step"),
                        "decode_tps": _metric(record, "eval_time_tokens_per_second"),
                        "ppl": None,
                        "agreement": None,
                        "status": record.get("status"),
                    }
                )
            elif mode == "perplexity":
                rows.append(
                    {
                        "model": record.get("model_id"),
                        "runtime": "llama.cpp_turboquant",
                        "config": f"{record.get('config')} ppl",
                        "context": int(record.get("perplexity_context") or 0),
                        "decode_ms": None,
                        "decode_tps": _metric(record, "eval_time_tokens_per_second"),
                        "ppl": _metric(record, "ppl"),
                        "agreement": None,
                        "status": record.get("status"),
                    }
                )
        elif benchmark == "llama_compare":
            rows.append(
                {
                    "model": record.get("model_id"),
                    "runtime": "dotcache_hf",
                    "config": f"K={record.get('bits_k')} / V={record.get('bits_v')}",
                    "context": int(record.get("prompt_length") or 0),
                    "decode_ms": _metric(record, "dotcache_decode_ms_per_step"),
                    "decode_tps": None if _metric(record, "dotcache_decode_ms_per_step") in (None, 0.0) else 1000.0 / float(record["dotcache_decode_ms_per_step"]),
                    "ppl": None,
                    "agreement": _metric(record, "greedy_token_agreement_rate"),
                    "status": record.get("status", "ok"),
                }
            )
    return sorted(rows, key=lambda row: (str(row["model"]), str(row["runtime"]), int(row["context"]), str(row["config"])))


def _render_markdown(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# TurboQuant Comparison",
        "",
        "| Model | Runtime | Config | Context | Decode ms/step | Decode tok/s | PPL | Agreement | Status |",
        "|---|---|---|---:|---:|---:|---:|---:|---|",
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
                    _fmt_num(row.get("ppl"), 4),
                    _fmt_num(row.get("agreement")),
                    str(row.get("status") or "-"),
                ]
            )
            + " |"
        )
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
    print(_render_markdown(rows))


if __name__ == "__main__":
    main()
