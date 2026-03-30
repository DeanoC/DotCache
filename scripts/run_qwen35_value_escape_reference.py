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
DEFAULT_0P8B_LAYER_PROFILE = (
    REPO_ROOT / "configs" / "layer_profiles" / "qwen35_0p8b_attention_subset_cuda_shortlist_context_aware.yaml"
)

PRESETS: dict[str, dict[str, Any]] = {
    "qwen35_0p8b_best": {
        "model_id": "Qwen/Qwen3.5-0.8B",
        "layer_profile": str(DEFAULT_0P8B_LAYER_PROFILE),
        "escape_layer": 23,
        "contexts": [32768, 49152, 65536],
        "prewarm": True,
        "prewarm_min_context": 49152,
    },
    "qwen35_4b_best": {
        "model_id": "Qwen/Qwen3.5-4B",
        "layer_profile": None,
        "escape_layer": 7,
        "contexts": [32768, 49152, 65536],
        "prewarm": False,
        "prewarm_min_context": 0,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the promoted Qwen3.5 value-escape benchmark reference lanes. "
            "These presets encode the current best benchmark-only operating points "
            "for 0.8B and 4B."
        )
    )
    parser.add_argument("--preset", choices=sorted(PRESETS), required=True)
    parser.add_argument("--backend", default="torch_cuda")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch-dtype", default="float16")
    parser.add_argument("--contexts", type=int, nargs="+", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=2)
    parser.add_argument("--tokens-per-page", type=int, default=16)
    parser.add_argument("--timeout-seconds", type=int, default=240)
    parser.add_argument("--blas-num-threads", type=int, default=1)
    parser.add_argument("--profile-backend", action="store_true")
    parser.add_argument("--quality-check", action="store_true")
    parser.add_argument("--scorer-diagnostic", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--output", default=None, help="Optional JSONL path. Defaults to stdout only.")
    return parser.parse_args()


def _preset(args: argparse.Namespace) -> dict[str, Any]:
    return PRESETS[str(args.preset)]


def _contexts(args: argparse.Namespace) -> list[int]:
    if args.contexts:
        return [int(context) for context in args.contexts]
    return [int(context) for context in _preset(args)["contexts"]]


def _append_record(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True))
        handle.write("\n")


def _benchmark_command(args: argparse.Namespace, *, context: int) -> list[str]:
    preset = _preset(args)
    escape_layer = int(preset["escape_layer"])
    command = [
        sys.executable,
        str(REPO_ROOT / "benchmarks" / "bench_qwen35_attention_subset_dotcache_serving.py"),
        "--model-id",
        str(preset["model_id"]),
        "--backend",
        args.backend,
        "--device",
        args.device,
        "--torch-dtype",
        args.torch_dtype,
        "--default-mode-k",
        "M0",
        "--default-mode-v",
        "M0",
        "--key-policy-tier",
        "exact",
        "--value-policy-tier",
        "exact",
        "--tokens-per-page",
        str(args.tokens_per_page),
        "--repeat-counts",
        "--target-prompt-lengths",
        str(context),
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--continue-on-error",
        "--blas-num-threads",
        str(args.blas_num_threads),
        "--execution-recent-window",
        "1024",
        "--execution-sink-window",
        "256",
        "--execution-relevance-top-k",
        "4",
        "--execution-relevance-top-k-context-layer",
        f"layer:{escape_layer}:min_ctx:8192=8",
        "--execution-relevance-mode",
        "envelope",
        "--execution-builtin-selector-cache",
        "--execution-builtin-selector-candidate-only",
        "--key-mode-override",
        f"layer:{escape_layer}=M0",
        "--value-mode-override",
        f"layer:{escape_layer}=M0",
        "--execution-value-escape-layer",
        str(escape_layer),
        "--execution-value-escape-mode",
        "M3",
    ]
    layer_profile = preset.get("layer_profile")
    if layer_profile:
        command.extend(["--layer-profile", str(layer_profile)])
    if bool(preset.get("prewarm", False)):
        command.append("--execution-value-escape-prewarm")
        min_context = int(preset.get("prewarm_min_context", 0))
        if min_context > 0:
            command.extend(["--execution-value-escape-prewarm-min-context", str(min_context)])
    if args.profile_backend:
        command.append("--profile-backend")
    if args.quality_check:
        command.append("--quality-check")
    if args.scorer_diagnostic:
        command.append("--scorer-diagnostic")
    return command


def _run_single_case(args: argparse.Namespace, *, context: int) -> dict[str, Any]:
    command = _benchmark_command(args, context=context)
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
        if candidate.get("prompt_mode") != "exact_length":
            continue
        if int(candidate.get("prompt_length") or -1) != context:
            continue
        payload = candidate

    preset = _preset(args)
    if payload is None:
        payload = {
            "benchmark": "qwen35_attention_subset_dotcache_serving",
            "model_id": str(preset["model_id"]),
            "backend": args.backend,
            "device": args.device,
            "torch_dtype": args.torch_dtype,
            "layer_profile": preset.get("layer_profile"),
            "prompt_mode": "exact_length",
            "prompt_length": context,
            "status": "error",
            "error_type": "TimeoutExpired" if timed_out else "NoExactRow",
            "error_message": (
                f"timed out after {args.timeout_seconds}s"
                if timed_out
                else f"command exited {process.returncode} without exact_length row"
            ),
        }

    payload = dict(payload)
    payload["runner_reference_preset"] = str(args.preset)
    payload["runner_escape_layer"] = int(preset["escape_layer"])
    payload["runner_prewarm_enabled"] = bool(preset.get("prewarm", False))
    payload["runner_prewarm_min_context"] = int(preset.get("prewarm_min_context", 0))
    payload["runner_timeout_seconds"] = int(args.timeout_seconds)
    payload["runner_wall_time_s"] = float(elapsed)
    payload["runner_command"] = command
    if timed_out:
        payload["runner_timed_out"] = True
    if stderr_text.strip():
        payload["runner_stderr_tail"] = stderr_text.strip()[-4000:]
    return payload


def main() -> None:
    args = parse_args()
    if not args.quality_check and not args.scorer_diagnostic:
        args.quality_check = True
    output_path = Path(args.output) if args.output else None
    if output_path is not None and output_path.exists():
        output_path.unlink()

    for context in _contexts(args):
        record = _run_single_case(args, context=context)
        if output_path is not None:
            _append_record(output_path, record)
        print(json.dumps(record, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
