from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from dotcache.model_registry import ModelSpec, get_model_spec, list_model_specs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Emit or execute the shared DotCache model benchmark matrix."
    )
    parser.add_argument("--model-keys", nargs="*", default=[])
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--run-supported", action="store_true")
    parser.add_argument("--backend", choices=["torch_mps", "torch_cuda", "cpu_ref", "auto"], default="auto")
    parser.add_argument("--device", default=None)
    parser.add_argument("--torch-dtype", default="float16")
    parser.add_argument("--tokens-per-page", type=int, default=256)
    parser.add_argument("--max-new-tokens", type=int, default=4)
    parser.add_argument("--prompt-lengths", type=int, nargs="*", default=[])
    parser.add_argument(
        "--continue-on-error",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Pass --continue-on-error through to runnable compare harnesses.",
    )
    parser.add_argument("--output-format", choices=["jsonl", "pretty"], default="jsonl")
    return parser.parse_args()


def _selected_specs(model_keys: list[str]) -> tuple[ModelSpec, ...]:
    if not model_keys:
        return list_model_specs()
    return tuple(get_model_spec(key) for key in model_keys)


def _default_compare_command(
    spec: ModelSpec,
    *,
    backend: str,
    device: str | None,
    torch_dtype: str,
    tokens_per_page: int,
    max_new_tokens: int,
    prompt_lengths: tuple[int, ...],
    continue_on_error: bool,
) -> list[str] | None:
    if spec.benchmark_harness != "llama_compare" or not spec.dotcache_ready:
        return None
    root = Path(__file__).resolve().parent.parent
    command = [
        str(root / ".venv" / "bin" / "python"),
        str(root / "benchmarks" / "bench_llama_compare.py"),
        "--model-id",
        spec.model_id,
        "--backend",
        backend,
        "--torch-dtype",
        torch_dtype,
        "--tokens-per-page",
        str(tokens_per_page),
        "--max-new-tokens",
        str(max_new_tokens),
        "--target-prompt-lengths",
        *[str(length) for length in prompt_lengths],
    ]
    if continue_on_error:
        command.append("--continue-on-error")
    if device is not None:
        command.extend(["--device", device])
    return command


def _matrix_record(
    spec: ModelSpec,
    *,
    backend: str,
    device: str | None,
    torch_dtype: str,
    tokens_per_page: int,
    max_new_tokens: int,
    prompt_lengths_override: list[int],
    continue_on_error: bool,
) -> dict[str, object]:
    prompt_lengths = tuple(prompt_lengths_override) if prompt_lengths_override else spec.prompt_lengths
    command = _default_compare_command(
        spec,
        backend=backend,
        device=device,
        torch_dtype=torch_dtype,
        tokens_per_page=tokens_per_page,
        max_new_tokens=max_new_tokens,
        prompt_lengths=prompt_lengths,
        continue_on_error=continue_on_error,
    )
    return {
        **spec.to_dict(),
        "planned_prompt_lengths": prompt_lengths,
        "backend": backend,
        "device": device,
        "torch_dtype": torch_dtype,
        "tokens_per_page": tokens_per_page,
        "max_new_tokens": max_new_tokens,
        "continue_on_error": continue_on_error,
        "command": command,
        "status": "runnable" if command is not None else "scaffold_only",
    }


def _print_record(record: dict[str, object], *, output_format: str) -> None:
    if output_format == "pretty":
        print(json.dumps(record, indent=2, sort_keys=True))
        return
    print(json.dumps(record, sort_keys=True))


def main() -> None:
    args = parse_args()
    specs = _selected_specs(args.model_keys)
    if args.list:
        for spec in specs:
            _print_record(spec.to_dict(), output_format=args.output_format)
        return

    for spec in specs:
        record = _matrix_record(
            spec,
            backend=args.backend,
            device=args.device,
            torch_dtype=args.torch_dtype,
            tokens_per_page=args.tokens_per_page,
            max_new_tokens=args.max_new_tokens,
            prompt_lengths_override=args.prompt_lengths,
            continue_on_error=args.continue_on_error,
        )
        _print_record(record, output_format=args.output_format)
        if not args.run_supported or record["command"] is None:
            continue
        completed = subprocess.run(record["command"], check=False)
        if completed.returncode != 0:
            raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
