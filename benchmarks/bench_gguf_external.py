from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import time

from transformers import AutoTokenizer

from dotcache.integrations.llama import resolve_hf_auth_kwargs


_TIMING_PATTERN = re.compile(
    r"(?P<label>prompt eval time|eval time|total time)\s*=\s*"
    r"(?P<ms>[0-9]+(?:\.[0-9]+)?)\s*ms\s*/\s*"
    r"(?P<count>[0-9]+)\s*"
    r"(?P<count_label>tokens|runs)"
    r"(?:\s*\(\s*(?P<ms_per_unit>[0-9]+(?:\.[0-9]+)?)\s*ms per token,\s*"
    r"(?P<tps>[0-9]+(?:\.[0-9]+)?)\s*tokens per second\s*\))?"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run an external llama.cpp / GGUF reference benchmark."
    )
    parser.add_argument("--model-id", required=True, help="Hugging Face GGUF repository for llama.cpp -hf.")
    parser.add_argument(
        "--tokenizer-model-id",
        required=True,
        help="Hugging Face tokenizer repo used to construct exact-length prompts.",
    )
    parser.add_argument("--llama-cli", default=os.environ.get("LLAMA_CPP_CLI", "llama-cli"))
    parser.add_argument("--max-new-tokens", type=int, default=4)
    parser.add_argument("--target-prompt-lengths", type=int, nargs="+", default=[1024, 2048])
    parser.add_argument("--repeat-counts", type=int, nargs="*", default=[])
    parser.add_argument("--prompt-unit", default="Cache locality matters for fast decoding.")
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--n-gpu-layers", type=int, default=None)
    parser.add_argument("--context-size", type=int, default=None)
    parser.add_argument("--continue-on-error", action="store_true")
    return parser.parse_args()


def _build_exact_prompt_text(
    tokenizer: AutoTokenizer,
    *,
    prompt_unit: str,
    prompt_length: int,
) -> tuple[str, int]:
    if prompt_length <= 0:
        raise ValueError("prompt_length must be positive")
    unit_ids = tokenizer(prompt_unit, add_special_tokens=False)["input_ids"]
    if not unit_ids:
        raise ValueError("prompt_unit tokenized to an empty sequence")

    token_ids: list[int] = []
    bos_token_id = getattr(tokenizer, "bos_token_id", None)
    if bos_token_id is not None:
        token_ids.append(int(bos_token_id))
    while len(token_ids) < prompt_length:
        token_ids.extend(int(token_id) for token_id in unit_ids)
    token_ids = token_ids[:prompt_length]
    prompt_text = tokenizer.decode(token_ids, skip_special_tokens=False)
    retokenized_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    return prompt_text, len(retokenized_ids)


def _build_repeat_prompt(prompt_unit: str, repeat_count: int) -> str:
    return " ".join([prompt_unit] * repeat_count)


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


def _llama_cli_command(
    args: argparse.Namespace,
    *,
    prompt_text: str,
) -> list[str]:
    command = [
        args.llama_cli,
        "-hf",
        args.model_id,
        "-n",
        str(args.max_new_tokens),
        "-p",
        prompt_text,
        "--temp",
        "0",
        "--seed",
        "0",
        "--no-display-prompt",
    ]
    if args.threads is not None:
        command.extend(["-t", str(args.threads)])
    if args.n_gpu_layers is not None:
        command.extend(["-ngl", str(args.n_gpu_layers)])
    if args.context_size is not None:
        command.extend(["-c", str(args.context_size)])
    return command


def _probe_llama_cli(executable: str) -> dict[str, object]:
    probe: dict[str, object] = {
        "llama_cli": executable,
        "llama_cli_found": True,
    }

    version_completed = subprocess.run(
        [executable, "--version"],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    version_text = "\n".join(
        part.strip() for part in (version_completed.stdout, version_completed.stderr) if part.strip()
    )
    probe["llama_cli_version_rc"] = version_completed.returncode
    probe["llama_cli_version_text"] = version_text[:500]

    help_completed = subprocess.run(
        [executable, "-h"],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    help_text = "\n".join(
        part.strip() for part in (help_completed.stdout, help_completed.stderr) if part.strip()
    )
    probe["llama_cli_help_rc"] = help_completed.returncode
    probe["llama_cli_supports_hf"] = "-hf" in help_text
    probe["llama_cli_help_excerpt"] = help_text[:1000]
    return probe


def _emit_error_record(
    *,
    args: argparse.Namespace,
    prompt_mode: str,
    requested_prompt_length: int | None,
    repeat_count: int | None,
    error_type: str,
    error_message: str,
    probe: dict[str, object] | None = None,
) -> None:
    record = {
        "benchmark": "gguf_external",
        "status": "error",
        "model_id": args.model_id,
        "tokenizer_model_id": args.tokenizer_model_id,
        "runtime": "llama_cpp",
        "prompt_mode": prompt_mode,
        "requested_prompt_length": requested_prompt_length,
        "repeat_count": repeat_count,
        "error_type": error_type,
        "error_message": error_message,
    }
    if probe is not None:
        record.update(probe)
    print(
        json.dumps(record, sort_keys=True),
        flush=True,
    )


def _run_case(
    args: argparse.Namespace,
    *,
    prompt_text: str,
    prompt_mode: str,
    requested_prompt_length: int | None,
    actual_prompt_length: int | None,
    repeat_count: int | None,
) -> None:
    executable = shutil.which(args.llama_cli)
    if executable is None:
        message = f"llama.cpp executable not found: {args.llama_cli}"
        if args.continue_on_error:
            _emit_error_record(
                args=args,
                prompt_mode=prompt_mode,
                requested_prompt_length=requested_prompt_length,
                repeat_count=repeat_count,
                error_type="MissingExecutable",
                error_message=message,
                probe={"llama_cli": args.llama_cli, "llama_cli_found": False},
            )
            return
        raise SystemExit(message)

    probe = _probe_llama_cli(executable)
    command = _llama_cli_command(args, prompt_text=prompt_text)
    started_at = time.perf_counter()
    completed = subprocess.run(
        command,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    wall_ms = (time.perf_counter() - started_at) * 1000.0

    combined_output = "\n".join(part for part in (completed.stdout, completed.stderr) if part)
    record: dict[str, object] = {
        "benchmark": "gguf_external",
        "status": "ok" if completed.returncode == 0 else "error",
        "runtime": "llama_cpp",
        "model_id": args.model_id,
        "tokenizer_model_id": args.tokenizer_model_id,
        "command": command,
        "prompt_mode": prompt_mode,
        "requested_prompt_length": requested_prompt_length,
        "prompt_length": actual_prompt_length,
        "repeat_count": repeat_count,
        "max_new_tokens": args.max_new_tokens,
        "wall_ms": wall_ms,
        "returncode": completed.returncode,
        "stdout_tail": completed.stdout[-1000:],
        "stderr_tail": completed.stderr[-1000:],
    }
    record.update(probe)
    record.update(_parse_timings(combined_output))
    print(json.dumps(record, sort_keys=True), flush=True)
    if completed.returncode != 0 and not args.continue_on_error:
        raise SystemExit(completed.returncode)


def main() -> None:
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_model_id, **resolve_hf_auth_kwargs())

    for repeat_count in args.repeat_counts:
        prompt_text = _build_repeat_prompt(args.prompt_unit, repeat_count)
        actual_prompt_length = len(tokenizer(prompt_text, add_special_tokens=True)["input_ids"])
        _run_case(
            args,
            prompt_text=prompt_text,
            prompt_mode="repeat_count",
            requested_prompt_length=None,
            actual_prompt_length=actual_prompt_length,
            repeat_count=repeat_count,
        )

    for prompt_length in sorted(set(length for length in args.target_prompt_lengths if length > 0)):
        prompt_text, actual_prompt_length = _build_exact_prompt_text(
            tokenizer,
            prompt_unit=args.prompt_unit,
            prompt_length=prompt_length,
        )
        _run_case(
            args,
            prompt_text=prompt_text,
            prompt_mode="exact_length",
            requested_prompt_length=prompt_length,
            actual_prompt_length=actual_prompt_length,
            repeat_count=None,
        )


if __name__ == "__main__":
    main()
