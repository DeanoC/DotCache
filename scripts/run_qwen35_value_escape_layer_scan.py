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
DEFAULT_4B_ATTENTION_LAYERS = (3, 7, 11, 15, 19, 23)
SELECTOR_MODES = ("approx_shortlist", "layer_full_context")
KV_MODES = ("exact_m0", "m0_v_escape")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Scan candidate Qwen3.5 full-attention layers for value-side escape sensitivity. "
            "This is the cheap transfer/localization runner used before deeper per-layer tuning."
        )
    )
    parser.add_argument("--model-id", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--backend", default="torch_cuda")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch-dtype", default="float16")
    parser.add_argument("--layer-profile", default="none")
    parser.add_argument("--layers", type=int, nargs="+", default=list(DEFAULT_4B_ATTENTION_LAYERS))
    parser.add_argument("--contexts", type=int, nargs="+", default=[16384, 32768])
    parser.add_argument("--selector-modes", nargs="+", choices=SELECTOR_MODES, default=["approx_shortlist"])
    parser.add_argument("--kv-modes", nargs="+", choices=KV_MODES, default=list(KV_MODES))
    parser.add_argument("--max-new-tokens", type=int, default=2)
    parser.add_argument("--tokens-per-page", type=int, default=16)
    parser.add_argument("--timeout-seconds", type=int, default=240)
    parser.add_argument("--blas-num-threads", type=int, default=1)
    parser.add_argument("--profile-backend", action="store_true")
    parser.add_argument("--quality-check", action="store_true")
    parser.add_argument("--scorer-diagnostic", action="store_true")
    parser.add_argument("--output", default=None, help="Optional JSONL path. Defaults to stdout only.")
    return parser.parse_args()


def _normalized_layer_profile(layer_profile: str | None) -> str | None:
    if layer_profile is None:
        return None
    normalized = str(layer_profile).strip()
    if normalized.lower() in {"", "none", "null"}:
        return None
    return normalized


def _selector_args(selector_mode: str, *, layer_id: int) -> list[str]:
    args = [
        "--execution-recent-window",
        "1024",
        "--execution-sink-window",
        "256",
        "--execution-relevance-top-k",
        "4",
        "--execution-relevance-top-k-context-layer",
        f"layer:{layer_id}:min_ctx:8192=8",
        "--execution-relevance-mode",
        "envelope",
        "--execution-builtin-selector-cache",
        "--execution-builtin-selector-candidate-only",
    ]
    if selector_mode == "layer_full_context":
        args.extend(["--execution-full-context-layer", str(layer_id)])
    elif selector_mode != "approx_shortlist":
        raise ValueError(f"unsupported selector mode: {selector_mode}")
    return args


def _kv_args(kv_mode: str, *, layer_id: int) -> list[str]:
    if kv_mode == "exact_m0":
        return ["--value-mode-override", f"layer:{layer_id}=M0"]
    if kv_mode == "m0_v_escape":
        return [
            "--value-mode-override",
            f"layer:{layer_id}=M0",
            "--execution-value-escape-layer",
            str(layer_id),
            "--execution-value-escape-mode",
            "M3",
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
    layer_id: int,
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
    normalized_layer_profile = _normalized_layer_profile(args.layer_profile)
    if normalized_layer_profile is not None:
        command.extend(["--layer-profile", normalized_layer_profile])
    command.extend(_selector_args(selector_mode, layer_id=layer_id))
    command.extend(_kv_args(kv_mode, layer_id=layer_id))
    return command


def _run_single_case(
    args: argparse.Namespace,
    *,
    layer_id: int,
    context: int,
    selector_mode: str,
    kv_mode: str,
) -> dict[str, Any]:
    command = _benchmark_command(args, layer_id=layer_id, context=context, selector_mode=selector_mode, kv_mode=kv_mode)
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
            "layer_profile": _normalized_layer_profile(args.layer_profile),
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
    payload["runner_probe_layer_id"] = int(layer_id)
    payload["runner_selector_mode"] = selector_mode
    payload["runner_kv_mode"] = kv_mode
    payload["runner_timeout_seconds"] = int(args.timeout_seconds)
    payload["runner_wall_time_s"] = float(elapsed)
    payload["runner_command"] = command
    payload["runner_selector_exact_standin"] = bool(selector_mode == "layer_full_context")
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

    for layer_id in args.layers:
        for context in args.contexts:
            for selector_mode in args.selector_modes:
                for kv_mode in args.kv_modes:
                    record = _run_single_case(
                        args,
                        layer_id=int(layer_id),
                        context=context,
                        selector_mode=selector_mode,
                        kv_mode=kv_mode,
                    )
                    if output_path is not None:
                        _append_record(output_path, record)
                    print(json.dumps(record, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
