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

from scripts.run_qwen35_cuda_needle_probe import _append_record, _case_extra_args


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run single-shot Qwen3.5 CUDA LongBench QA probes with standardized metadata."
    )
    parser.add_argument("--model-id", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--backend", default="torch_cuda")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch-dtype", default="float16")
    parser.add_argument(
        "--layer-profile",
        default=str(REPO_ROOT / "configs" / "layer_profiles" / "qwen35_0p8b_attention_subset_cuda_third_pass.yaml"),
    )
    parser.add_argument(
        "--cases",
        nargs="+",
        choices=["exact", "shortlist_base", "shortlist_l23_ctx", "shortlist_topk8", "shortlist_quality_profile"],
        default=["exact", "shortlist_base", "shortlist_l23_ctx"],
    )
    parser.add_argument("--prompt-pack", required=True, help="JSON file defining LongBench rows to run.")
    parser.add_argument(
        "--quality-layer-profile",
        default=str(
            REPO_ROOT / "configs" / "layer_profiles" / "qwen35_0p8b_attention_subset_cuda_shortlist_quality.yaml"
        ),
    )
    parser.add_argument("--timeout-seconds", type=int, default=1200)
    parser.add_argument("--profile-backend", action="store_true")
    parser.add_argument("--quality-check", action="store_true")
    parser.add_argument("--scorer-diagnostic", action="store_true")
    parser.add_argument("--evaluation-split", choices=["calibration", "held_out"], default="held_out")
    parser.add_argument("--evaluation-lane", choices=["systems", "quality", "diagnostic"], default="systems")
    parser.add_argument("--evaluation-prompt-family", default="longbench_qa")
    parser.add_argument("--evaluation-prompt-suite-name", default="qwen35_cuda_longbench_qa_pack_v1")
    parser.add_argument("--evaluation-prompt-count", type=int, default=4)
    parser.add_argument("--evaluation-batch-size", type=int, default=1)
    parser.add_argument("--evaluation-protocol-version", default="2026-03-31")
    parser.add_argument("--evaluation-notes", default=None)
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def _longbench_case_extra_args(case: str) -> list[str]:
    if case == "shortlist_topk8":
        return [
            "--execution-recent-window",
            "1024",
            "--execution-sink-window",
            "256",
            "--execution-relevance-top-k",
            "8",
            "--execution-relevance-mode",
            "envelope",
        ]
    if case == "shortlist_quality_profile":
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
    return _case_extra_args(case)


def _load_prompt_specs(path: str) -> list[dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not payload:
        raise SystemExit(f"prompt pack {path} must be a non-empty JSON list")
    prompt_specs: list[dict[str, Any]] = []
    for index, item in enumerate(payload, start=1):
        if not isinstance(item, dict):
            raise SystemExit(f"prompt pack item #{index} is not an object")
        prompt_id = str(item.get("prompt_id") or f"prompt_{index}")
        dataset = str(item.get("dataset") or "").strip()
        if not dataset:
            raise SystemExit(f"prompt pack item {prompt_id!r} must define dataset")
        row_index = int(item.get("row_index", -1))
        if row_index < 0:
            raise SystemExit(f"prompt pack item {prompt_id!r} must define a non-negative row_index")
        prompt_specs.append({"prompt_id": prompt_id, "dataset": dataset, "row_index": row_index})
    return prompt_specs


def _benchmark_command(args: argparse.Namespace, *, case: str, prompt_spec: dict[str, Any]) -> list[str]:
    layer_profile = args.layer_profile
    if case == "shortlist_quality_profile":
        layer_profile = args.quality_layer_profile
    command = [
        sys.executable,
        str(REPO_ROOT / "benchmarks" / "bench_qwen35_attention_subset_dotcache_longbench_qa.py"),
        "--model-id",
        args.model_id,
        "--backend",
        args.backend,
        "--device",
        args.device,
        "--torch-dtype",
        args.torch_dtype,
        "--layer-profile",
        layer_profile,
        "--longbench-dataset",
        prompt_spec["dataset"],
        "--longbench-row-index",
        str(prompt_spec["row_index"]),
    ]
    if args.profile_backend:
        command.append("--profile-backend")
    if args.quality_check:
        command.append("--quality-check")
    if args.scorer_diagnostic:
        command.append("--scorer-diagnostic")
    command.extend(_longbench_case_extra_args(case))
    return command


def _run_single_probe(args: argparse.Namespace, *, case: str, prompt_spec: dict[str, Any]) -> dict[str, Any]:
    command = _benchmark_command(args, case=case, prompt_spec=prompt_spec)
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
        if candidate.get("prompt_mode") == "longbench_qa" and candidate.get("longbench_dataset") == prompt_spec["dataset"]:
            row_index = candidate.get("longbench_row_index")
            if row_index is not None and int(row_index) == int(prompt_spec["row_index"]):
                payload = candidate

    if payload is None:
        payload = {
            "benchmark": "qwen35_attention_subset_dotcache_longbench_qa",
            "benchmark_task": "longbench_qa",
            "model_id": args.model_id,
            "backend": args.backend,
            "device": args.device,
            "torch_dtype": args.torch_dtype,
            "layer_profile": args.layer_profile,
            "prompt_mode": "longbench_qa",
            "longbench_dataset": prompt_spec["dataset"],
            "longbench_row_index": int(prompt_spec["row_index"]),
            "status": "error",
            "error_type": "TimeoutExpired" if timed_out else "NoLongBenchRow",
            "error_message": (
                f"timed out after {args.timeout_seconds}s"
                if timed_out
                else f"command exited {process.returncode} without LongBench row"
            ),
        }

    payload = dict(payload)
    payload["runner_case"] = case
    payload["evaluation_prompt_id"] = prompt_spec["prompt_id"]
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
    prompt_specs = _load_prompt_specs(args.prompt_pack)
    output_path = Path(args.output) if args.output else None
    if output_path is not None and output_path.exists():
        output_path.unlink()

    for prompt_spec in prompt_specs:
        for case in args.cases:
            record = _run_single_probe(args, case=case, prompt_spec=prompt_spec)
            record["evaluation_prompt_count"] = len(prompt_specs)
            record["evaluation_prompt_pack"] = args.prompt_pack
            if output_path is not None:
                _append_record(output_path, record)
            print(json.dumps(record, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
