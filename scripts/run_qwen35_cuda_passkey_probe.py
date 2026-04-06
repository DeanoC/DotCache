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

_PROMPT_MODE_REUSED = "needle_in_a_haystack"
_PASSKEY_BENCHMARK = "qwen35_attention_subset_dotcache_passkey"
_PASSKEY_TASK = "passkey_retrieval"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run single-shot Qwen3.5 CUDA passkey-retrieval probes with standardized metadata."
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
        choices=["exact", "shortlist_base", "shortlist_l23_ctx", "streaming_sink_recent"],
        default=["exact", "shortlist_base", "shortlist_l23_ctx"],
    )
    parser.add_argument("--max-new-tokens", type=int, default=12)
    parser.add_argument("--passkey-position-fraction", type=float, default=0.5)
    parser.add_argument("--passkey-key", default="five-digit passkey")
    parser.add_argument("--passkey-value", default="58142")
    parser.add_argument("--haystack-unit", default=None)
    parser.add_argument("--passkey-template", default=None)
    parser.add_argument("--question-template", default=None)
    parser.add_argument("--prompt-pack", default=None, help="Optional JSON file defining multiple passkey prompt variants.")
    parser.add_argument("--timeout-seconds", type=int, default=900)
    parser.add_argument("--profile-backend", action="store_true")
    parser.add_argument("--quality-check", action="store_true")
    parser.add_argument("--evaluation-split", choices=["calibration", "held_out"], default="held_out")
    parser.add_argument("--evaluation-lane", choices=["systems", "quality", "diagnostic"], default="systems")
    parser.add_argument("--evaluation-prompt-family", default="passkey_retrieval")
    parser.add_argument("--evaluation-prompt-suite-name", default="qwen35_cuda_passkey_retrieval_v1")
    parser.add_argument("--evaluation-prompt-count", type=int, default=1)
    parser.add_argument("--evaluation-batch-size", type=int, default=1)
    parser.add_argument("--evaluation-protocol-version", default="2026-03-31")
    parser.add_argument("--evaluation-notes", default=None)
    parser.add_argument("--output", default=None, help="Optional JSONL path. Defaults to stdout only.")
    return parser.parse_args(argv)


def _load_prompt_specs(args: argparse.Namespace) -> list[dict[str, Any]]:
    if not args.prompt_pack:
        return [
            {
                "prompt_id": "default",
                "passkey_key": args.passkey_key,
                "passkey_value": args.passkey_value,
                "passkey_position_fraction": args.passkey_position_fraction,
                **({"haystack_unit": args.haystack_unit} if args.haystack_unit is not None else {}),
                **({"passkey_template": args.passkey_template} if args.passkey_template is not None else {}),
                **({"question_template": args.question_template} if args.question_template is not None else {}),
            }
        ]

    pack_path = Path(args.prompt_pack)
    payload = json.loads(pack_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not payload:
        raise SystemExit(f"prompt pack {pack_path} must be a non-empty JSON list")

    prompt_specs: list[dict[str, Any]] = []
    for index, item in enumerate(payload, start=1):
        if not isinstance(item, dict):
            raise SystemExit(f"prompt pack item #{index} is not an object")
        prompt_id = str(item.get("prompt_id") or f"prompt_{index}")
        passkey_key = item.get("passkey_key") or item.get("needle_key")
        passkey_value = item.get("passkey_value") or item.get("needle_value")
        if not passkey_key or not passkey_value:
            raise SystemExit(f"prompt pack item {prompt_id!r} must define passkey_key/passkey_value")
        prompt_specs.append(
            {
                "prompt_id": prompt_id,
                "passkey_key": str(passkey_key),
                "passkey_value": str(passkey_value),
                "passkey_position_fraction": float(
                    item.get(
                        "passkey_position_fraction",
                        item.get("needle_position_fraction", args.passkey_position_fraction),
                    )
                ),
                **({"haystack_unit": str(item["haystack_unit"])} if "haystack_unit" in item else {}),
                **(
                    {"passkey_template": str(item["passkey_template"])}
                    if "passkey_template" in item
                    else (
                        {"passkey_template": str(item["needle_template"])}
                        if "needle_template" in item
                        else {}
                    )
                ),
                **({"question_template": str(item["question_template"])} if "question_template" in item else {}),
            }
        )
    return prompt_specs


def _benchmark_command(args: argparse.Namespace, *, context: int, case: str, prompt_spec: dict[str, Any]) -> list[str]:
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
        str(prompt_spec["passkey_position_fraction"]),
        "--needle-key",
        prompt_spec["passkey_key"],
        "--needle-value",
        prompt_spec["passkey_value"],
    ]
    if "haystack_unit" in prompt_spec:
        command.extend(["--haystack-unit", prompt_spec["haystack_unit"]])
    if "passkey_template" in prompt_spec:
        command.extend(["--needle-template", prompt_spec["passkey_template"]])
    if "question_template" in prompt_spec:
        command.extend(["--question-template", prompt_spec["question_template"]])
    if args.profile_backend:
        command.append("--profile-backend")
    if args.quality_check:
        command.append("--quality-check")
    command.extend(_case_extra_args(case))
    return command


def _retag_passkey_payload(payload: dict[str, Any], *, prompt_spec: dict[str, Any]) -> dict[str, Any]:
    record = dict(payload)
    record["benchmark"] = _PASSKEY_BENCHMARK
    record["benchmark_task"] = _PASSKEY_TASK
    record["prompt_mode"] = _PASSKEY_TASK
    record["passkey_key"] = record.get("needle_key", prompt_spec["passkey_key"])
    record["passkey_value"] = record.get("needle_value", prompt_spec["passkey_value"])
    record["passkey_position_fraction_requested"] = float(
        record.get("needle_position_fraction_requested", prompt_spec["passkey_position_fraction"])
    )
    if "needle_prompt_text" in record:
        record["passkey_prompt_text"] = record["needle_prompt_text"]
    if "needle_question_text" in record:
        record["passkey_question_text"] = record["needle_question_text"]
    if "needle_token_start" in record:
        record["passkey_token_start"] = record["needle_token_start"]
    if "needle_token_end" in record:
        record["passkey_token_end"] = record["needle_token_end"]
    if "needle_question_token_start" in record:
        record["passkey_question_token_start"] = record["needle_question_token_start"]
    if "needle_filler_before_tokens" in record:
        record["passkey_filler_before_tokens"] = record["needle_filler_before_tokens"]
    if "needle_filler_after_tokens" in record:
        record["passkey_filler_after_tokens"] = record["needle_filler_after_tokens"]
    if "needle_position_fraction_actual" in record:
        record["passkey_position_fraction_actual"] = record["needle_position_fraction_actual"]
    if "needle_expected_answer" in record:
        record["passkey_expected_answer"] = record["needle_expected_answer"]
    if "needle_generated_text" in record:
        record["passkey_generated_text"] = record["needle_generated_text"]
    if "needle_generated_first_line" in record:
        record["passkey_generated_first_line"] = record["needle_generated_first_line"]
    if "needle_answer_exact_match" in record:
        record["passkey_answer_exact_match"] = record["needle_answer_exact_match"]
    if "needle_answer_prefix_match" in record:
        record["passkey_answer_prefix_match"] = record["needle_answer_prefix_match"]
    if "needle_answer_contains_match" in record:
        record["passkey_answer_contains_match"] = record["needle_answer_contains_match"]
    if "needle_answer_correct" in record:
        record["passkey_answer_correct"] = record["needle_answer_correct"]
    record["passkey_runner_reused_prompt_builder"] = _PROMPT_MODE_REUSED
    return record


def _run_single_probe(args: argparse.Namespace, *, context: int, case: str, prompt_spec: dict[str, Any]) -> dict[str, Any]:
    command = _benchmark_command(args, context=context, case=case, prompt_spec=prompt_spec)
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
        if candidate.get("prompt_mode") == _PROMPT_MODE_REUSED and int(candidate.get("prompt_length") or -1) == context:
            payload = candidate

    if payload is None:
        payload = {
            "benchmark": _PASSKEY_BENCHMARK,
            "benchmark_task": _PASSKEY_TASK,
            "model_id": args.model_id,
            "backend": args.backend,
            "device": args.device,
            "torch_dtype": args.torch_dtype,
            "layer_profile": args.layer_profile,
            "prompt_mode": _PASSKEY_TASK,
            "prompt_length": context,
            "status": "error",
            "error_type": "TimeoutExpired" if timed_out else "NoPasskeyRow",
            "error_message": (
                f"timed out after {args.timeout_seconds}s"
                if timed_out
                else f"command exited {process.returncode} without passkey row"
            ),
        }

    payload = _retag_passkey_payload(payload, prompt_spec=prompt_spec)
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
    prompt_specs = _load_prompt_specs(args)
    output_path = Path(args.output) if args.output else None
    if output_path is not None and output_path.exists():
        output_path.unlink()

    for prompt_spec in prompt_specs:
        for case in args.cases:
            for context in args.contexts:
                record = _run_single_probe(args, context=context, case=case, prompt_spec=prompt_spec)
                record["evaluation_prompt_count"] = len(prompt_specs)
                record["evaluation_prompt_pack"] = str(args.prompt_pack) if args.prompt_pack else None
                if output_path is not None:
                    _append_record(output_path, record)
                print(json.dumps(record, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
