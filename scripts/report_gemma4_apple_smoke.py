#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize a Gemma 4 Apple smoke runner artifact.")
    parser.add_argument(
        "--input",
        default=str(REPO_ROOT / "benchmarks" / "results" / f"gemma4_apple_smoke_{time.strftime('%Y%m%d')}" / "smoke_runner.json"),
        help="Path to smoke_runner.json produced by scripts/run_gemma4_apple_smoke.py",
    )
    parser.add_argument("--format", choices=["markdown", "json"], default="markdown")
    return parser.parse_args()


def _load_summary(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("smoke summary must be a JSON object")
    return payload


def _build_report(summary: dict[str, Any], *, input_path: Path) -> dict[str, Any]:
    probe_record = summary.get("probe_record") if isinstance(summary.get("probe_record"), dict) else {}
    return {
        "input_path": str(input_path),
        "status": summary.get("status"),
        "model_id": summary.get("model_id"),
        "prompt": summary.get("prompt"),
        "elapsed_s": float(summary.get("elapsed_s") or 0.0),
        "timeout_seconds": int(summary.get("timeout_seconds") or 0),
        "returncode": summary.get("returncode"),
        "runtime_device": probe_record.get("runtime_device"),
        "runtime_torch_dtype": probe_record.get("runtime_torch_dtype"),
        "greedy_token_agreement_rate": probe_record.get("greedy_token_agreement_rate"),
        "teacher_forced_logit_max_abs_error": probe_record.get("teacher_forced_logit_max_abs_error"),
        "m0_pages": probe_record.get("m0_pages"),
        "m3_pages": probe_record.get("m3_pages"),
        "resident_bytes": probe_record.get("resident_bytes"),
        "kv_resident_bytes": probe_record.get("kv_resident_bytes"),
        "dense_text": probe_record.get("dense_text"),
        "dotcache_text": probe_record.get("dotcache_text"),
        "error_type": summary.get("error_type"),
        "error_message": summary.get("error_message"),
    }


def _fmt_float(value: Any, digits: int = 2) -> str:
    if value is None:
        return "-"
    return f"{float(value):.{digits}f}"


def _fmt_mib(value: Any) -> str:
    if value is None:
        return "-"
    return f"{float(value) / (1024.0 * 1024.0):.2f}"


def _render_markdown(report: dict[str, Any]) -> str:
    lines = ["# Gemma 4 Apple Smoke", ""]
    lines.append("| Field | Value |")
    lines.append("|---|---|")
    lines.append(f"| Status | {report['status']} |")
    lines.append(f"| Model | {report['model_id']} |")
    lines.append(f"| Runtime | {report.get('runtime_device') or '-'} / {report.get('runtime_torch_dtype') or '-'} |")
    lines.append(f"| Prompt | {report['prompt']} |")
    lines.append(f"| Elapsed (s) | {_fmt_float(report['elapsed_s'])} |")
    lines.append(f"| Timeout (s) | {report['timeout_seconds']} |")
    lines.append(f"| Agreement | {_fmt_float(report.get('greedy_token_agreement_rate'), 3)} |")
    lines.append(f"| Max abs logit error | {_fmt_float(report.get('teacher_forced_logit_max_abs_error'), 6)} |")
    lines.append(f"| Resident MiB | {_fmt_mib(report.get('resident_bytes'))} |")
    lines.append(f"| KV Resident MiB | {_fmt_mib(report.get('kv_resident_bytes'))} |")
    lines.append(f"| M0 Pages | {report.get('m0_pages', '-')} |")
    lines.append(f"| M3 Pages | {report.get('m3_pages', '-')} |")
    lines.append(f"| Dense Text | {report.get('dense_text') or '-'} |")
    lines.append(f"| DotCache Text | {report.get('dotcache_text') or '-'} |")
    if report.get("error_type") or report.get("error_message"):
        lines.append(f"| Error | {(report.get('error_type') or 'error')}: {report.get('error_message') or '-'} |")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    summary = _load_summary(input_path)
    report = _build_report(summary, input_path=input_path)
    if args.format == "json":
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(_render_markdown(report))


if __name__ == "__main__":
    main()
