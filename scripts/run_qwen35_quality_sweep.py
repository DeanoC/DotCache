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
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from transformers import AutoTokenizer

from benchmarks.bench_turboquant_external import _build_exact_prompt_text as _build_turboquant_exact_prompt_text
from dotcache.integrations.llama import resolve_hf_auth_kwargs

DEFAULT_OUTPUT_DIR = REPO_ROOT / "benchmarks" / "results" / "qwen35_quality_sweep_20260329"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run long-context quality checks for the Qwen3.5 TurboQuant comparison lane.")
    parser.add_argument("--model-id", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--gguf-model-path", default="/workspace/models/gguf/qwen35_0p8b/Qwen3.5-0.8B-Q4_K_M.gguf")
    parser.add_argument("--backend", default="torch_cuda")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch-dtype", default="float16")
    parser.add_argument("--prefix-lengths", type=int, nargs="+", default=[16384, 32768])
    parser.add_argument("--eval-steps", type=int, default=16)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--prompt-unit", default="Cache locality matters for fast decoding.")
    parser.add_argument(
        "--dotcache-layer-profile",
        default=str(REPO_ROOT / "configs" / "layer_profiles" / "qwen35_0p8b_attention_subset_cuda_third_pass.yaml"),
    )
    parser.add_argument("--state-stage", default="post_update_m0")
    parser.add_argument("--state-bits", type=int, default=8)
    parser.add_argument("--state-renorm-interval", type=int, default=0)
    parser.add_argument("--dense-timeout-seconds", type=int, default=1800)
    parser.add_argument("--dotcache-timeout-seconds", type=int, default=1800)
    parser.add_argument("--statecache-timeout-seconds", type=int, default=1800)
    parser.add_argument("--turboquant-timeout-seconds", type=int, default=1800)
    parser.add_argument("--turboquant-configs", nargs="+", default=["q8_0", "turbo3_uniform", "turbo3_la1"])
    parser.add_argument("--skip-dense", action="store_true")
    parser.add_argument("--skip-dotcache", action="store_true")
    parser.add_argument("--skip-statecache", action="store_true")
    parser.add_argument("--skip-turboquant", action="store_true")
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
    fallback_record: dict[str, Any],
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
        timeout_record = dict(fallback_record)
        timeout_record["status"] = "error"
        timeout_record["error_type"] = "TimeoutExpired"
        timeout_record["error_message"] = f"timed out after {timeout_seconds}s"
        _append_record(output_path, timeout_record)
        return _parsed_records_from_stdout(stdout_text)

    _write_stdout(output_path, stdout_text)
    seen_records = _parsed_records_from_stdout(stdout_text)
    if completed.returncode != 0 and not seen_records:
        error_record = dict(fallback_record)
        error_record["status"] = "error"
        error_record["error_type"] = "RuntimeError"
        error_record["error_message"] = (completed.stderr or completed.stdout or "").strip()[-4000:]
        _append_record(output_path, error_record)
    return seen_records


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dense_path = output_dir / "qwen35_0p8b_dense_quality.jsonl"
    dotcache_path = output_dir / "qwen35_0p8b_dotcache_quality.jsonl"
    statecache_path = output_dir / "qwen35_0p8b_statecache_quality.jsonl"
    turboquant_path = output_dir / "qwen35_0p8b_turboquant_quality.jsonl"
    if not args.skip_dense and dense_path.exists():
        dense_path.unlink()
    if not args.skip_dotcache and dotcache_path.exists():
        dotcache_path.unlink()
    if not args.skip_statecache and statecache_path.exists():
        statecache_path.unlink()
    if not args.skip_turboquant and turboquant_path.exists():
        turboquant_path.unlink()

    prompt_dir = output_dir / "prompts"
    prompt_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, **resolve_hf_auth_kwargs())
    python_exe = sys.executable

    for prefix_length in sorted(set(length for length in args.prefix_lengths if length > 0)):
        sequence_length = int(prefix_length + args.eval_steps)
        turboquant_prompt_length = int(sequence_length * 2)
        prompt_file = prompt_dir / f"qwen35_0p8b_ppl_sequence_{turboquant_prompt_length}.txt"
        prompt_text, _ = _build_turboquant_exact_prompt_text(
            tokenizer,
            prompt_unit=args.prompt_unit,
            prompt_length=turboquant_prompt_length,
        )
        prompt_file.write_text(prompt_text, encoding="utf-8")

        if not args.skip_dense:
            dense_command = [
                python_exe,
                str(REPO_ROOT / "benchmarks" / "bench_qwen35_loss.py"),
                "--model-id",
                args.model_id,
                "--backend",
                args.backend,
                "--device",
                args.device,
                "--torch-dtype",
                args.torch_dtype,
                "--sequence-length",
                str(sequence_length),
                "--prefix-length",
                str(prefix_length),
                "--eval-steps",
                str(args.eval_steps),
                "--prompt-unit",
                args.prompt_unit,
            ]
            _run_jsonl_command(
                command=dense_command,
                output_path=dense_path,
                timeout_seconds=args.dense_timeout_seconds,
                fallback_record={
                    "benchmark": "qwen35_loss",
                    "model_id": args.model_id,
                    "backend": args.backend,
                    "device": args.device,
                    "torch_dtype": args.torch_dtype,
                    "sequence_length": sequence_length,
                    "prefix_length": prefix_length,
                    "eval_steps": args.eval_steps,
                    "prompt_unit": args.prompt_unit,
                },
            )

        if not args.skip_dotcache:
            dotcache_command = [
                python_exe,
                str(REPO_ROOT / "benchmarks" / "bench_qwen35_attention_subset_dotcache_loss.py"),
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
                "--sequence-length",
                str(sequence_length),
                "--prefix-length",
                str(prefix_length),
                "--eval-steps",
                str(args.eval_steps),
                "--prompt-unit",
                args.prompt_unit,
            ]
            _run_jsonl_command(
                command=dotcache_command,
                output_path=dotcache_path,
                timeout_seconds=args.dotcache_timeout_seconds,
                fallback_record={
                    "benchmark": "qwen35_attention_subset_dotcache_loss",
                    "model_id": args.model_id,
                    "backend": args.backend,
                    "device": args.device,
                    "torch_dtype": args.torch_dtype,
                    "sequence_length": sequence_length,
                    "prefix_length": prefix_length,
                    "eval_steps": args.eval_steps,
                    "prompt_unit": args.prompt_unit,
                    "layer_profile": args.dotcache_layer_profile,
                },
            )

        if not args.skip_statecache:
            statecache_command = [
                python_exe,
                str(REPO_ROOT / "benchmarks" / "bench_qwen35_deltanet_statecache_loss.py"),
                "--model-id",
                args.model_id,
                "--backend",
                args.backend,
                "--device",
                args.device,
                "--torch-dtype",
                args.torch_dtype,
                "--sequence-length",
                str(sequence_length),
                "--prefix-length",
                str(prefix_length),
                "--eval-steps",
                str(args.eval_steps),
                "--group-size",
                "32",
                "--bits",
                str(args.state_bits),
                "--state-stage",
                args.state_stage,
                "--renorm-interval",
                str(args.state_renorm_interval),
                "--prompt-unit",
                args.prompt_unit,
            ]
            _run_jsonl_command(
                command=statecache_command,
                output_path=statecache_path,
                timeout_seconds=args.statecache_timeout_seconds,
                fallback_record={
                    "benchmark": "qwen35_deltanet_statecache_loss",
                    "model_id": args.model_id,
                    "backend": args.backend,
                    "device": args.device,
                    "torch_dtype": args.torch_dtype,
                    "sequence_length": sequence_length,
                    "prefix_length": prefix_length,
                    "eval_steps": args.eval_steps,
                    "prompt_unit": args.prompt_unit,
                    "deltanet_statecache_stage_name": args.state_stage,
                    "deltanet_statecache_bits": args.state_bits,
                    "deltanet_statecache_renorm_interval": args.state_renorm_interval,
                },
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
                    "--perplexity-file",
                    str(prompt_file),
                    "--perplexity-context",
                    str(sequence_length),
                    "--perplexity-chunks",
                    "1",
                    "--skip-decode",
                    "--continue-on-error",
                ]
                _run_jsonl_command(
                    command=turboquant_command,
                    output_path=turboquant_path,
                    timeout_seconds=args.turboquant_timeout_seconds,
                    fallback_record={
                        "benchmark": "turboquant_external",
                        "mode": "perplexity",
                        "runtime": "llama.cpp_turboquant",
                        "config": config,
                        "model_id": args.gguf_model_path,
                        "tokenizer_model_id": args.model_id,
                        "perplexity_context": sequence_length,
                        "perplexity_file": str(prompt_file),
                        "status": "error",
                    },
                )

    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "dense_jsonl": str(dense_path),
                "dotcache_jsonl": str(dotcache_path),
                "statecache_jsonl": str(statecache_path),
                "turboquant_jsonl": str(turboquant_path),
                "prefix_lengths": sorted(set(length for length in args.prefix_lengths if length > 0)),
                "eval_steps": args.eval_steps,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
