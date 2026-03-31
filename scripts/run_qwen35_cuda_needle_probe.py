#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run single-shot Qwen3.5 CUDA Needle-in-a-Haystack probes with standardized metadata."
    )
    parser.add_argument("--model-id", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--backend", default="torch_cuda")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch-dtype", default="float16")
    parser.add_argument(
        "--layer-profile",
        default=str(REPO_ROOT / "configs" / "layer_profiles" / "qwen35_0p8b_attention_subset_cuda_third_pass.yaml"),
    )
    parser.add_argument("--contexts", type=int, nargs="+", default=[32768, 49152])
    parser.add_argument(
        "--cases",
        nargs="+",
        choices=["exact", "shortlist_base", "shortlist_l23_ctx"],
        default=["exact", "shortlist_base", "shortlist_l23_ctx"],
    )
    parser.add_argument("--max-new-tokens", type=int, default=12)
    parser.add_argument("--needle-position-fraction", type=float, default=0.5)
    parser.add_argument("--needle-key", default="hidden passphrase")
    parser.add_argument("--needle-value", default="crimson-velvet-472")
    parser.add_argument("--timeout-seconds", type=int, default=900)
    parser.add_argument("--profile-backend", action="store_true")
    parser.add_argument("--quality-check", action="store_true")
    parser.add_argument("--evaluation-split", choices=["calibration", "held_out"], default="held_out")
    parser.add_argument("--evaluation-lane", choices=["systems", "quality", "diagnostic"], default="systems")
    parser.add_argument("--evaluation-prompt-family", default="needle_in_a_haystack")
    parser.add_argument("--evaluation-prompt-suite-name", default="qwen35_cuda_needle_in_a_haystack_v1")
    parser.add_argument("--evaluation-prompt-count", type=int, default=1)
    parser.add_argument("--evaluation-batch-size", type=int, default=1)
    parser.add_argument("--evaluation-protocol-version", default="2026-03-31")
    parser.add_argument("--evaluation-notes", default=None)
    parser.add_argument("--output", default=None, help="Optional JSONL path. Defaults to stdout only.")
    return parser.parse_args()


def _case_extra_args(case: str) -> list[str]:
    if case == "exact":
        return []
    if case == "shortlist_base":
        return [
            "--execution-recent-window",
            "1024",
            "--execution-sink-window",
            "256",
            "--execution-relevance-top-k",
            "4",
            "--execution-relevance-mode",
            "envelope",
        ]
    if case == "shortlist_l23_ctx":
        return [
            "--execution-recent-window",
            "1024",
            "--execution-sink-window",
            "256",
            "--execution-relevance-top-k",
            "4",
            "--execution-relevance-mode",
            "envelope",
            "--execution-relevance-top-k-context-layer",
            "layer:23:min_ctx:8192=8",
        ]
    raise ValueError(f"unsupported case: {case}")


def _append_record(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True))
        handle.write("\n")


def _benchmark_command(args: argparse.Namespace, *, context: int, case: str) -> list[str]:
    command = [
        sys.executable,
        str(REPO_ROOT / "benchmarks" / "bench_qwen35_attention_subset_dotcache_needle.py"),
        "--model-id",
        args.model_id,
        "--backend",
        args.backend,
        "--device",
        args.device,
        "--torch-dtype",
        args.torch_dtype,
        "--layer-profile",
        args.layer_profile,
        "--prompt-length",
        str(context),
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--needle-position-fraction",
        str(args.needle_position_fraction),
        "--needle-key",
        args.needle_key,
        "--needle-value",
        args.needle_value,
    ]
    if args.profile_backend:
        command.append("--profile-backend")
    if args.quality_check:
        command.append("--quality-check")
    command.extend(_case_extra_args(case))
    return command


def _run_single_probe(args: argparse.Namespace, *, context: int, case: str) -> dict[str, Any]:
    command = _benchmark_command(args, context=context, case=case)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT)

    started_at = time.monotonic()
    process = subprocess.Popen(
        command,
        cwd=str(REPO_ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    timed_out = False
    try:
        stdout_text, stderr_text = process.communicate(timeout=args.timeout_seconds)
    except subprocess.TimeoutExpired:
        timed_out = True
        os.killpg(process.pid, signal.SIGKILL)
        stdout_text, stderr_text = process.communicate()
    elapsed = time.monotonic() - started_at

    payload: dict[str, Any] | None = None
    for raw_line in stdout_text.splitlines():
        stripped = raw_line.strip().replace("\x00", "")
        if not stripped.startswith("{"):
            continue
        try:
            candidate = json.loads(stripped)
        except json.JSONDecodeError:
            continue
        if not isinstance(candidate, dict):
            continue
        if candidate.get("prompt_mode") == "needle_in_a_haystack" and int(candidate.get("prompt_length") or -1) == context:
            payload = candidate

    if payload is None:
        payload = {
            "benchmark": "qwen35_attention_subset_dotcache_needle",
            "benchmark_task": "needle_in_a_haystack",
            "model_id": args.model_id,
            "backend": args.backend,
            "device": args.device,
            "torch_dtype": args.torch_dtype,
            "layer_profile": args.layer_profile,
            "prompt_mode": "needle_in_a_haystack",
            "prompt_length": context,
            "status": "error",
            "error_type": "TimeoutExpired" if timed_out else "NoNeedleRow",
            "error_message": (
                f"timed out after {args.timeout_seconds}s"
                if timed_out
                else f"command exited {process.returncode} without needle row"
            ),
        }

    payload = dict(payload)
    payload["runner_case"] = case
    payload["runner_timeout_seconds"] = args.timeout_seconds
    payload["runner_wall_time_s"] = elapsed
    payload["runner_command"] = command
    payload["evaluation_split"] = args.evaluation_split
    payload["evaluation_lane"] = args.evaluation_lane
    payload["evaluation_prompt_family"] = args.evaluation_prompt_family
    payload["evaluation_prompt_suite_name"] = args.evaluation_prompt_suite_name
    payload["evaluation_prompt_count"] = args.evaluation_prompt_count
    payload["evaluation_batch_size"] = args.evaluation_batch_size
    payload["evaluation_protocol_version"] = args.evaluation_protocol_version
    if args.evaluation_notes:
        payload["evaluation_notes"] = args.evaluation_notes
    if timed_out:
        payload["runner_timed_out"] = True
    if stderr_text.strip():
        payload["runner_stderr_tail"] = stderr_text.strip()[-4000:]
    return payload


def main() -> None:
    args = parse_args()
    output_path = Path(args.output) if args.output else None
    if output_path is not None and output_path.exists():
        output_path.unlink()

    for case in args.cases:
        for context in args.contexts:
            record = _run_single_probe(args, context=context, case=case)
            if output_path is not None:
                _append_record(output_path, record)
            print(json.dumps(record, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
