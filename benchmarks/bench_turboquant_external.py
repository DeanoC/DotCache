from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import time
from dataclasses import dataclass

from transformers import AutoTokenizer

from benchmarks.bench_gguf_external import _build_exact_prompt_text, _resolve_local_gguf_path
from dotcache.integrations.llama import resolve_hf_auth_kwargs


_TIMING_PATTERN = re.compile(
    r"(?P<label>prompt eval time|eval time|total time)\s*=\s*"
    r"(?P<ms>[0-9]+(?:\.[0-9]+)?)\s*ms\s*/\s*"
    r"(?P<count>[0-9]+)\s*"
    r"(?P<count_label>tokens|runs)"
    r"(?:\s*\(\s*(?P<ms_per_unit>[0-9]+(?:\.[0-9]+)?)\s*ms per token,\s*"
    r"(?P<tps>[0-9]+(?:\.[0-9]+)?)\s*tokens per second\s*\))?"
)
_PPL_PATTERN = re.compile(r"PPL\s*=\s*([0-9]+(?:\.[0-9]+)?)")
_DEFAULT_GGUF_MODELS_DIR = os.environ.get("GGUF_MODELS_DIR", "/workspace/models/gguf")
_DEFAULT_TURBOQUANT_CLI = os.environ.get(
    "TURBOQUANT_LLAMA_CLI",
    "/workspace/llama-cpp-turboquant-cuda/build/bin/llama-cli",
)
_DEFAULT_TURBOQUANT_PPL = os.environ.get(
    "TURBOQUANT_LLAMA_PERPLEXITY",
    "/workspace/llama-cpp-turboquant-cuda/build/bin/llama-perplexity",
)


@dataclass(frozen=True, slots=True)
class TurboConfig:
    key: str
    ctk: str
    ctv: str
    layer_adaptive: int | None = None


_CONFIGS: dict[str, TurboConfig] = {
    "q8_0": TurboConfig(key="q8_0", ctk="q8_0", ctv="q8_0"),
    "turbo3_uniform": TurboConfig(key="turbo3_uniform", ctk="turbo3", ctv="turbo3"),
    "turbo3_la1": TurboConfig(key="turbo3_la1", ctk="turbo3", ctv="turbo3", layer_adaptive=1),
    "turbo3_la5": TurboConfig(key="turbo3_la5", ctk="turbo3", ctv="turbo3", layer_adaptive=5),
    "turbo4_uniform": TurboConfig(key="turbo4_uniform", ctk="turbo4", ctv="turbo4"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run external TurboQuant / llama.cpp CUDA comparison benchmarks."
    )
    parser.add_argument("--model-id", required=True, help="GGUF repository or local .gguf path.")
    parser.add_argument("--tokenizer-model-id", required=True)
    parser.add_argument("--hf-file", default=None)
    parser.add_argument("--llama-cli", default=_DEFAULT_TURBOQUANT_CLI)
    parser.add_argument("--llama-perplexity", default=_DEFAULT_TURBOQUANT_PPL)
    parser.add_argument("--gguf-models-dir", default=_DEFAULT_GGUF_MODELS_DIR)
    parser.add_argument("--configs", nargs="+", default=["q8_0", "turbo3_uniform", "turbo3_la1", "turbo3_la5"])
    parser.add_argument("--target-prompt-lengths", type=int, nargs="+", default=[4096, 16384, 32768])
    parser.add_argument("--max-new-tokens", type=int, default=4)
    parser.add_argument("--prompt-unit", default="Cache locality matters for fast decoding.")
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--n-gpu-layers", type=int, default=99)
    parser.add_argument("--context-size", type=int, default=None)
    parser.add_argument("--perplexity-file", default=None, help="Optional plain text file for llama-perplexity.")
    parser.add_argument("--perplexity-context", type=int, default=2048)
    parser.add_argument("--perplexity-chunks", type=int, default=8)
    parser.add_argument("--continue-on-error", action="store_true")
    return parser.parse_args()


def _parse_timings(text: str) -> dict[str, float | int]:
    metrics: dict[str, float | int] = {}
    for match in _TIMING_PATTERN.finditer(text):
        label = match.group("label").replace(" ", "_")
        metrics[f"{label}_ms"] = float(match.group("ms"))
        metrics[f"{label}_count"] = int(match.group("count"))
        if match.group("ms_per_unit") is not None:
            metrics[f"{label}_ms_per_token"] = float(match.group("ms_per_unit"))
        if match.group("tps") is not None:
            metrics[f"{label}_tokens_per_second"] = float(match.group("tps"))
    if "eval_time_ms" in metrics and "eval_time_count" in metrics:
        count = int(metrics["eval_time_count"])
        if count > 0:
            metrics["decode_ms_per_step"] = float(metrics["eval_time_ms"]) / count
    return metrics


def _parse_ppl(text: str) -> float | None:
    match = _PPL_PATTERN.search(text)
    if match is None:
        return None
    return float(match.group(1))


def _resolve_model_arg(
    model_id: str,
    *,
    hf_file: str | None,
    gguf_models_dir: str,
) -> list[str]:
    local_model_path = _resolve_local_gguf_path(
        model_id,
        hf_file=hf_file,
        gguf_models_dir=gguf_models_dir,
    )
    if local_model_path is not None:
        return ["-m", local_model_path]
    args = ["-hf", model_id]
    if hf_file is not None:
        args.extend(["-hff", hf_file])
    return args


def _build_llama_cli_command(
    args: argparse.Namespace,
    *,
    config: TurboConfig,
    prompt_text: str,
) -> tuple[list[str], dict[str, str]]:
    command = [args.llama_cli, *_resolve_model_arg(args.model_id, hf_file=args.hf_file, gguf_models_dir=args.gguf_models_dir)]
    command.extend(
        [
            "-ctk",
            config.ctk,
            "-ctv",
            config.ctv,
            "-fa",
            "on",
            "-n",
            str(args.max_new_tokens),
            "-p",
            prompt_text,
            "--no-conversation",
            "--simple-io",
            "--temp",
            "0",
            "--seed",
            "0",
            "--no-display-prompt",
        ]
    )
    if args.threads is not None:
        command.extend(["-t", str(args.threads)])
    if args.n_gpu_layers is not None:
        command.extend(["-ngl", str(args.n_gpu_layers)])
    if args.context_size is not None:
        command.extend(["-c", str(args.context_size)])
    env = os.environ.copy()
    if config.layer_adaptive is not None:
        env["TURBO_LAYER_ADAPTIVE"] = str(config.layer_adaptive)
    else:
        env.pop("TURBO_LAYER_ADAPTIVE", None)
    return command, env


def _build_ppl_command(
    args: argparse.Namespace,
    *,
    config: TurboConfig,
) -> tuple[list[str], dict[str, str]]:
    command = [args.llama_perplexity, *_resolve_model_arg(args.model_id, hf_file=args.hf_file, gguf_models_dir=args.gguf_models_dir)]
    command.extend(
        [
            "-ctk",
            config.ctk,
            "-ctv",
            config.ctv,
            "-fa",
            "-c",
            str(args.perplexity_context),
            "--chunks",
            str(args.perplexity_chunks),
            "-f",
            str(args.perplexity_file),
        ]
    )
    if args.threads is not None:
        command.extend(["-t", str(args.threads)])
    if args.n_gpu_layers is not None:
        command.extend(["-ngl", str(args.n_gpu_layers)])
    env = os.environ.copy()
    if config.layer_adaptive is not None:
        env["TURBO_LAYER_ADAPTIVE"] = str(config.layer_adaptive)
    else:
        env.pop("TURBO_LAYER_ADAPTIVE", None)
    return command, env


def _emit(record: dict[str, object]) -> None:
    print(json.dumps(record, sort_keys=True), flush=True)


def _run_decode_case(
    args: argparse.Namespace,
    *,
    config: TurboConfig,
    prompt_text: str,
    prompt_length: int,
) -> None:
    command, env = _build_llama_cli_command(args, config=config, prompt_text=prompt_text)
    start = time.perf_counter()
    completed = subprocess.run(
        command,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    output_text = "\n".join(part.strip() for part in (completed.stdout, completed.stderr) if part.strip())
    record: dict[str, object] = {
        "benchmark": "turboquant_external",
        "mode": "decode",
        "runtime": "llama.cpp_turboquant",
        "config": config.key,
        "model_id": args.model_id,
        "tokenizer_model_id": args.tokenizer_model_id,
        "prompt_length": prompt_length,
        "max_new_tokens": args.max_new_tokens,
        "weight_format": "gguf",
        "status": "ok" if completed.returncode == 0 else "error",
        "returncode": completed.returncode,
        "wall_ms": elapsed_ms,
        "layer_adaptive": config.layer_adaptive,
        "ctk": config.ctk,
        "ctv": config.ctv,
    }
    record.update(_parse_timings(output_text))
    if completed.returncode != 0:
        record["error_type"] = "RuntimeError"
        record["error_message"] = output_text[-2000:]
    _emit(record)


def _run_ppl_case(args: argparse.Namespace, *, config: TurboConfig) -> None:
    if not args.perplexity_file:
        return
    command, env = _build_ppl_command(args, config=config)
    start = time.perf_counter()
    completed = subprocess.run(
        command,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    output_text = "\n".join(part.strip() for part in (completed.stdout, completed.stderr) if part.strip())
    record: dict[str, object] = {
        "benchmark": "turboquant_external",
        "mode": "perplexity",
        "runtime": "llama.cpp_turboquant",
        "config": config.key,
        "model_id": args.model_id,
        "tokenizer_model_id": args.tokenizer_model_id,
        "weight_format": "gguf",
        "status": "ok" if completed.returncode == 0 else "error",
        "returncode": completed.returncode,
        "wall_ms": elapsed_ms,
        "layer_adaptive": config.layer_adaptive,
        "ctk": config.ctk,
        "ctv": config.ctv,
        "perplexity_context": args.perplexity_context,
        "perplexity_chunks": args.perplexity_chunks,
        "perplexity_file": args.perplexity_file,
        "ppl": _parse_ppl(output_text),
    }
    record.update(_parse_timings(output_text))
    if completed.returncode != 0:
        record["error_type"] = "RuntimeError"
        record["error_message"] = output_text[-2000:]
    _emit(record)


def main() -> None:
    args = parse_args()
    if shutil.which(args.llama_cli) is None and not os.path.isfile(args.llama_cli):
        raise SystemExit(f"llama-cli not found: {args.llama_cli}")
    if args.perplexity_file and shutil.which(args.llama_perplexity) is None and not os.path.isfile(args.llama_perplexity):
        raise SystemExit(f"llama-perplexity not found: {args.llama_perplexity}")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_model_id, **resolve_hf_auth_kwargs())

    configs: list[TurboConfig] = []
    for key in args.configs:
        try:
            configs.append(_CONFIGS[key])
        except KeyError as exc:
            raise SystemExit(f"unknown turboquant config: {key}") from exc

    if args.perplexity_file:
        for config in configs:
            try:
                _run_ppl_case(args, config=config)
            except Exception as exc:
                if not args.continue_on_error:
                    raise
                _emit(
                    {
                        "benchmark": "turboquant_external",
                        "mode": "perplexity",
                        "runtime": "llama.cpp_turboquant",
                        "config": config.key,
                        "model_id": args.model_id,
                        "status": "error",
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                    }
                )

    for prompt_length in args.target_prompt_lengths:
        prompt_text, actual_prompt_length = _build_exact_prompt_text(
            tokenizer,
            prompt_unit=args.prompt_unit,
            prompt_length=prompt_length,
        )
        for config in configs:
            try:
                _run_decode_case(
                    args,
                    config=config,
                    prompt_text=prompt_text,
                    prompt_length=actual_prompt_length,
                )
            except Exception as exc:
                if not args.continue_on_error:
                    raise
                _emit(
                    {
                        "benchmark": "turboquant_external",
                        "mode": "decode",
                        "runtime": "llama.cpp_turboquant",
                        "config": config.key,
                        "model_id": args.model_id,
                        "prompt_length": actual_prompt_length,
                        "status": "error",
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                    }
                )


if __name__ == "__main__":
    main()
