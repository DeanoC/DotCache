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
DEFAULT_LAYER_PROFILE = (
    REPO_ROOT / "configs" / "layer_profiles" / "qwen35_0p8b_attention_subset_cuda_shortlist_context_aware.yaml"
)

SELECTOR_MODES = ("approx_shortlist", "layer23_full_context")
KV_MODES = (
    "exact_exact",
    "m0_exact",
    "exact_m0",
    "m0_m0",
    "m0_v_escape",
    "m0_v_escape_old",
    "m0_v_escape_top128",
    "m0_v_escape_top256",
    "m0_v_escape_top512",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the Qwen3.5 layer-23 selector/K/V ablation matrix. "
            "Note: selector_mode=layer23_full_context is the current stand-in for an exact layer-23 selector."
        )
    )
    parser.add_argument("--model-id", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--backend", default="torch_cuda")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch-dtype", default="float16")
    parser.add_argument("--layer-profile", default=str(DEFAULT_LAYER_PROFILE))
    parser.add_argument("--contexts", type=int, nargs="+", default=[32768, 49152])
    parser.add_argument("--selector-modes", nargs="+", choices=SELECTOR_MODES, default=list(SELECTOR_MODES))
    parser.add_argument("--kv-modes", nargs="+", choices=KV_MODES, default=list(KV_MODES))
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


def _selector_args(selector_mode: str) -> list[str]:
    args = [
        "--execution-recent-window",
        "1024",
        "--execution-sink-window",
        "256",
        "--execution-relevance-top-k",
        "4",
        "--execution-relevance-top-k-context-layer",
        "layer:23:min_ctx:8192=8",
        "--execution-relevance-mode",
        "envelope",
        "--execution-builtin-selector-cache",
        "--execution-builtin-selector-candidate-only",
    ]
    if selector_mode == "layer23_full_context":
        args.extend(["--execution-full-context-layer", "23"])
    elif selector_mode != "approx_shortlist":
        raise ValueError(f"unsupported selector mode: {selector_mode}")
    return args


def _kv_args(kv_mode: str) -> list[str]:
    if kv_mode == "exact_exact":
        return []
    if kv_mode == "m0_exact":
        return ["--key-mode-override", "layer:23=M0"]
    if kv_mode == "exact_m0":
        return ["--value-mode-override", "layer:23=M0"]
    if kv_mode == "m0_m0":
        return ["--key-mode-override", "layer:23=M0", "--value-mode-override", "layer:23=M0"]
    if kv_mode == "m0_v_escape":
        return [
            "--key-mode-override",
            "layer:23=M0",
            "--value-mode-override",
            "layer:23=M0",
            "--execution-value-escape-layer",
            "23",
            "--execution-value-escape-mode",
            "M3",
        ]
    if kv_mode == "m0_v_escape_old":
        return [
            "--key-mode-override",
            "layer:23=M0",
            "--value-mode-override",
            "layer:23=M0",
            "--execution-value-escape-layer",
            "23",
            "--execution-value-escape-mode",
            "M3",
            "--execution-value-escape-old-only",
        ]
    if kv_mode in {"m0_v_escape_top128", "m0_v_escape_top256", "m0_v_escape_top512"}:
        top_k = kv_mode.removeprefix("m0_v_escape_top")
        return [
            "--key-mode-override",
            "layer:23=M0",
            "--value-mode-override",
            "layer:23=M0",
            "--execution-value-escape-layer",
            "23",
            "--execution-value-escape-mode",
            "M3",
            "--execution-value-escape-top-k",
            top_k,
        ]
    raise ValueError(f"unsupported kv mode: {kv_mode}")


def _append_record(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True))
        handle.write("\n")


def _benchmark_command(
    args: argparse.Namespace,
    *,
    context: int,
    selector_mode: str,
    kv_mode: str,
) -> list[str]:
    command = [
        sys.executable,
        str(REPO_ROOT / "benchmarks" / "bench_qwen35_attention_subset_dotcache_serving.py"),
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
    ]
    if args.profile_backend:
        command.append("--profile-backend")
    if args.quality_check:
        command.append("--quality-check")
    if args.scorer_diagnostic:
        command.append("--scorer-diagnostic")
    command.extend(_selector_args(selector_mode))
    command.extend(_kv_args(kv_mode))
    return command


def _run_single_case(
    args: argparse.Namespace,
    *,
    context: int,
    selector_mode: str,
    kv_mode: str,
) -> dict[str, Any]:
    command = _benchmark_command(args, context=context, selector_mode=selector_mode, kv_mode=kv_mode)
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

    if payload is None:
        payload = {
            "benchmark": "qwen35_attention_subset_dotcache_serving",
            "model_id": args.model_id,
            "backend": args.backend,
            "device": args.device,
            "torch_dtype": args.torch_dtype,
            "layer_profile": args.layer_profile,
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
    payload["runner_selector_mode"] = selector_mode
    payload["runner_kv_mode"] = kv_mode
    payload["runner_timeout_seconds"] = int(args.timeout_seconds)
    payload["runner_wall_time_s"] = float(elapsed)
    payload["runner_command"] = command
    payload["runner_selector_exact_standin"] = bool(selector_mode == "layer23_full_context")
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

    for context in args.contexts:
        for selector_mode in args.selector_modes:
            for kv_mode in args.kv_modes:
                record = _run_single_case(
                    args,
                    context=context,
                    selector_mode=selector_mode,
                    kv_mode=kv_mode,
                )
                if output_path is not None:
                    _append_record(output_path, record)
                print(json.dumps(record, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
