#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "benchmarks" / "results" / "qwen35_serving_sweep_20260329"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a serving-style Qwen3.5 native-vs-TurboQuant sweep.")
    parser.add_argument("--model-id", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--gguf-model-path", default="/workspace/models/gguf/qwen35_0p8b/Qwen3.5-0.8B-Q4_K_M.gguf")
    parser.add_argument("--backend", default="torch_cuda")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch-dtype", default="float16")
    parser.add_argument("--contexts", type=int, nargs="+", default=[448, 1024, 2048, 4096, 8192, 16384, 32768, 65536])
    parser.add_argument("--max-new-tokens", type=int, default=4)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument(
        "--dotcache-layer-profile",
        default=str(REPO_ROOT / "configs" / "layer_profiles" / "qwen35_0p8b_attention_subset_cuda_third_pass.yaml"),
    )
    parser.add_argument("--state-stage", default="readout_only_m0")
    parser.add_argument("--state-bits", type=int, default=8)
    parser.add_argument("--state-renorm-interval", type=int, default=0)
    parser.add_argument("--dense-timeout-seconds", type=int, default=900)
    parser.add_argument("--dotcache-timeout-seconds", type=int, default=900)
    parser.add_argument("--statecache-timeout-seconds", type=int, default=900)
    parser.add_argument("--turboquant-timeout-seconds", type=int, default=600)
    parser.add_argument("--turboquant-configs", nargs="+", default=["q8_0", "turbo3_uniform", "turbo3_la1"])
    parser.add_argument("--skip-dense", action="store_true")
    parser.add_argument("--skip-dotcache", action="store_true")
    parser.add_argument("--skip-statecache", action="store_true")
    parser.add_argument("--skip-turboquant", action="store_true")
    parser.add_argument("--dotcache-profile-backend", action="store_true")
    return parser.parse_args()


def _append_record(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True))
        handle.write("\n")


def _write_stdout(path: Path, stdout: str) -> None:
    if not stdout:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(stdout)
        if not stdout.endswith("\n"):
            handle.write("\n")


def _parsed_records_from_stdout(stdout: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for raw_line in stdout.splitlines():
        stripped = raw_line.strip().replace("\x00", "")
        if not stripped or not stripped.startswith("{"):
            continue
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            records.append(payload)
    return records


def _run_jsonl_command(
    *,
    command: list[str],
    output_path: Path,
    timeout_seconds: int,
    timeout_record_factory,
    missing_record_factories: list[tuple[Any, Any]],
) -> list[dict[str, Any]]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT)
    try:
        completed = subprocess.run(
            command,
            cwd=str(REPO_ROOT),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
        stdout_text = completed.stdout or ""
    except subprocess.TimeoutExpired as exc:
        stdout_text = exc.stdout or ""
        if isinstance(stdout_text, bytes):
            stdout_text = stdout_text.decode("utf-8", errors="ignore")
        _write_stdout(output_path, stdout_text)
        seen_records = _parsed_records_from_stdout(stdout_text)
        seen_keys = {key_fn(record) for key_fn, _ in missing_record_factories for record in seen_records if key_fn(record) is not None}
        for key_fn, record_factory in missing_record_factories:
            expected_key = record_factory("key_only")
            if expected_key in seen_keys:
                continue
            _append_record(output_path, record_factory("timeout"))
        return seen_records

    _write_stdout(output_path, stdout_text)
    seen_records = _parsed_records_from_stdout(stdout_text)
    if completed.returncode != 0 and not completed.stdout.strip():
        _append_record(
            output_path,
            timeout_record_factory(
                status="error",
                error_type="RuntimeError",
                error_message=(completed.stderr or completed.stdout or "").strip()[-4000:],
            ),
        )
    return seen_records


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dense_path = output_dir / "qwen35_0p8b_dense_serving_sweep.jsonl"
    dotcache_path = output_dir / "qwen35_0p8b_dotcache_serving_sweep.jsonl"
    statecache_path = output_dir / "qwen35_0p8b_statecache_serving_sweep.jsonl"
    turboquant_path = output_dir / "qwen35_0p8b_turboquant_serving_sweep.jsonl"
    for path, skip in (
        (dense_path, args.skip_dense),
        (dotcache_path, args.skip_dotcache),
        (statecache_path, args.skip_statecache),
        (turboquant_path, args.skip_turboquant),
    ):
        if not skip and path.exists():
            path.unlink()

    python_exe = sys.executable
    context_args = [str(context) for context in args.contexts]

    if not args.skip_dense:
        dense_command = [
            python_exe,
            str(REPO_ROOT / "benchmarks" / "bench_qwen35_text.py"),
            "--model-id",
            args.model_id,
            "--backend",
            args.backend,
            "--device",
            args.device,
            "--torch-dtype",
            args.torch_dtype,
            "--repeat-counts",
            "--target-prompt-lengths",
            *context_args,
            "--max-new-tokens",
            str(args.max_new_tokens),
            "--continue-on-error",
        ]
        _run_jsonl_command(
            command=dense_command,
            output_path=dense_path,
            timeout_seconds=args.dense_timeout_seconds,
            timeout_record_factory=lambda status="error", error_type="TimeoutExpired", error_message=None: {
                "benchmark": "qwen35_text",
                "model_id": args.model_id,
                "backend": args.backend,
                "device": args.device,
                "torch_dtype": args.torch_dtype,
                "prompt_mode": "exact_length",
                "max_new_tokens": args.max_new_tokens,
                "status": status,
                "error_type": error_type,
                "error_message": error_message or f"timed out after {args.dense_timeout_seconds}s",
            },
            missing_record_factories=[
                (
                    lambda record, context=context: int(record.get("prompt_length") or -1),
                    lambda mode, context=context: context if mode == "key_only" else {
                        "benchmark": "qwen35_text",
                        "model_id": args.model_id,
                        "backend": args.backend,
                        "device": args.device,
                        "torch_dtype": args.torch_dtype,
                        "prompt_mode": "exact_length",
                        "prompt_length": context,
                        "max_new_tokens": args.max_new_tokens,
                        "status": "error",
                        "error_type": "TimeoutExpired",
                        "error_message": f"timed out after {args.dense_timeout_seconds}s",
                    },
                )
                for context in args.contexts
            ],
        )

    if not args.skip_dotcache:
        dotcache_command = [
            python_exe,
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
            args.dotcache_layer_profile,
            "--repeat-counts",
            "--target-prompt-lengths",
            *context_args,
            "--max-new-tokens",
            str(args.max_new_tokens),
            "--continue-on-error",
        ]
        if args.dotcache_profile_backend:
            dotcache_command.append("--profile-backend")
        _run_jsonl_command(
            command=dotcache_command,
            output_path=dotcache_path,
            timeout_seconds=args.dotcache_timeout_seconds,
            timeout_record_factory=lambda status="error", error_type="TimeoutExpired", error_message=None: {
                "benchmark": "qwen35_attention_subset_dotcache_serving",
                "model_id": args.model_id,
                "backend": args.backend,
                "device": args.device,
                "torch_dtype": args.torch_dtype,
                "layer_profile": args.dotcache_layer_profile,
                "prompt_mode": "exact_length",
                "max_new_tokens": args.max_new_tokens,
                "status": status,
                "error_type": error_type,
                "error_message": error_message or f"timed out after {args.dotcache_timeout_seconds}s",
            },
            missing_record_factories=[
                (
                    lambda record, context=context: int(record.get("prompt_length") or -1),
                    lambda mode, context=context: context if mode == "key_only" else {
                        "benchmark": "qwen35_attention_subset_dotcache_serving",
                        "model_id": args.model_id,
                        "backend": args.backend,
                        "device": args.device,
                        "torch_dtype": args.torch_dtype,
                        "layer_profile": args.dotcache_layer_profile,
                        "prompt_mode": "exact_length",
                        "prompt_length": context,
                        "max_new_tokens": args.max_new_tokens,
                        "status": "error",
                        "error_type": "TimeoutExpired",
                        "error_message": f"timed out after {args.dotcache_timeout_seconds}s",
                    },
                )
                for context in args.contexts
            ],
        )

    if not args.skip_statecache:
        statecache_command = [
            python_exe,
            str(REPO_ROOT / "benchmarks" / "bench_qwen35_deltanet_statecache_serving.py"),
            "--model-id",
            args.model_id,
            "--backend",
            args.backend,
            "--device",
            args.device,
            "--torch-dtype",
            args.torch_dtype,
            "--state-stage",
            args.state_stage,
            "--bits",
            str(args.state_bits),
            "--renorm-interval",
            str(args.state_renorm_interval),
            "--repeat-counts",
            "--target-prompt-lengths",
            *context_args,
            "--max-new-tokens",
            str(args.max_new_tokens),
            "--continue-on-error",
        ]
        _run_jsonl_command(
            command=statecache_command,
            output_path=statecache_path,
            timeout_seconds=args.statecache_timeout_seconds,
            timeout_record_factory=lambda status="error", error_type="TimeoutExpired", error_message=None: {
                "benchmark": "qwen35_deltanet_statecache_serving",
                "model_id": args.model_id,
                "backend": args.backend,
                "device": args.device,
                "torch_dtype": args.torch_dtype,
                "prompt_mode": "exact_length",
                "max_new_tokens": args.max_new_tokens,
                "status": status,
                "error_type": error_type,
                "error_message": error_message or f"timed out after {args.statecache_timeout_seconds}s",
            },
            missing_record_factories=[
                (
                    lambda record, context=context: int(record.get("prompt_length") or -1),
                    lambda mode, context=context: context if mode == "key_only" else {
                        "benchmark": "qwen35_deltanet_statecache_serving",
                        "model_id": args.model_id,
                        "backend": args.backend,
                        "device": args.device,
                        "torch_dtype": args.torch_dtype,
                        "prompt_mode": "exact_length",
                        "prompt_length": context,
                        "max_new_tokens": args.max_new_tokens,
                        "status": "error",
                        "error_type": "TimeoutExpired",
                        "error_message": f"timed out after {args.statecache_timeout_seconds}s",
                    },
                )
                for context in args.contexts
            ],
        )

    if not args.skip_turboquant:
        for config in args.turboquant_configs:
            turboquant_command = [
                python_exe,
                str(REPO_ROOT / "benchmarks" / "bench_turboquant_external.py"),
                "--model-id",
                args.gguf_model_path,
                "--tokenizer-model-id",
                args.model_id,
                "--configs",
                config,
                "--target-prompt-lengths",
                *context_args,
                "--max-new-tokens",
                str(args.max_new_tokens),
                "--continue-on-error",
            ]
            _run_jsonl_command(
                command=turboquant_command,
                output_path=turboquant_path,
                timeout_seconds=args.turboquant_timeout_seconds,
                timeout_record_factory=lambda status="error", error_type="TimeoutExpired", error_message=None, config=config: {
                    "benchmark": "turboquant_external",
                    "mode": "decode",
                    "runtime": "llama.cpp_turboquant",
                    "config": config,
                    "model_id": args.gguf_model_path,
                    "tokenizer_model_id": args.model_id,
                    "prompt_mode": "exact_length",
                    "max_new_tokens": args.max_new_tokens,
                    "status": status,
                    "error_type": error_type,
                    "error_message": error_message or f"timed out after {args.turboquant_timeout_seconds}s",
                },
                missing_record_factories=[
                    (
                        lambda record, context=context, config=config: (record.get("config"), int(record.get("prompt_length") or -1)),
                        lambda mode, context=context, config=config: (config, context) if mode == "key_only" else {
                            "benchmark": "turboquant_external",
                            "mode": "decode",
                            "runtime": "llama.cpp_turboquant",
                            "config": config,
                            "model_id": args.gguf_model_path,
                            "tokenizer_model_id": args.model_id,
                            "prompt_mode": "exact_length",
                            "prompt_length": context,
                            "max_new_tokens": args.max_new_tokens,
                            "status": "error",
                            "error_type": "TimeoutExpired",
                            "error_message": f"timed out after {args.turboquant_timeout_seconds}s",
                        },
                    )
                    for context in args.contexts
                ],
            )

    print(json.dumps(
        {
            "output_dir": str(output_dir),
            "dense_jsonl": str(dense_path),
            "dotcache_jsonl": str(dotcache_path),
            "statecache_jsonl": str(statecache_path),
            "turboquant_jsonl": str(turboquant_path),
            "contexts": args.contexts,
            "max_new_tokens": args.max_new_tokens,
            "dotcache_profile_backend": bool(args.dotcache_profile_backend),
        },
        indent=2,
        sort_keys=True,
    ))


if __name__ == "__main__":
    main()
