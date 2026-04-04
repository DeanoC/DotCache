#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SELECTOR_ARTIFACT = (
    REPO_ROOT
    / "benchmarks"
    / "results"
    / "qwen35_selector_qwen35_9b_suite_20260401"
    / "serving_selector_artifact"
    / "linear_selector_model.json"
)
PACK_PATHS = {
    "mini": REPO_ROOT / "configs" / "prompt_packs" / "qwen35_cuda_longbench_qa_pack_v1.json",
    "medium": REPO_ROOT / "configs" / "prompt_packs" / "qwen35_cuda_longbench_qa_pack_medium_v1.json",
    "full": REPO_ROOT / "configs" / "prompt_packs" / "qwen35_cuda_longbench_qa_pack_full_v1.json",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a named Qwen LongBench pack comparison.")
    parser.add_argument("--model-id", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--backend", default="torch_cuda")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch-dtype", default="float16")
    parser.add_argument("--selector-artifact", default=str(DEFAULT_SELECTOR_ARTIFACT))
    parser.add_argument("--pack", choices=sorted(PACK_PATHS), default="mini")
    parser.add_argument("--prompt-pack", default=None)
    parser.add_argument("--max-prompt-tokens", type=int, nargs="+", default=[4096, 8192])
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--measured-runs", type=int, default=5)
    parser.add_argument("--timeout-seconds", type=int, default=2400)
    parser.add_argument("--profile-backend", action="store_true")
    parser.add_argument("--trace-python-allocations", action="store_true")
    parser.add_argument("--quality-check", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def resolve_prompt_pack(args: argparse.Namespace) -> Path:
    if args.prompt_pack:
        return Path(args.prompt_pack).expanduser().resolve()
    return PACK_PATHS[str(args.pack)].resolve()


def model_slug(model_id: str) -> str:
    value = str(model_id).split("/")[-1].lower()
    return value.replace(".", "p")


def main() -> int:
    args = parse_args()
    prompt_pack = resolve_prompt_pack(args)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    slug = model_slug(args.model_id)
    jsonl_path = output_dir / f"{slug}_longbench_{args.pack}.jsonl"
    markdown_path = output_dir / "longbench_selector_compare.md"
    json_path = output_dir / "longbench_selector_compare.json"

    run_command = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "run_qwen35_longbench_selector_compare.py"),
        "--model-id",
        args.model_id,
        "--backend",
        args.backend,
        "--device",
        args.device,
        "--torch-dtype",
        args.torch_dtype,
        "--selector-artifact",
        args.selector_artifact,
        "--prompt-pack",
        str(prompt_pack),
        "--max-prompt-tokens",
        *[str(value) for value in args.max_prompt_tokens],
        "--warmup-runs",
        str(int(args.warmup_runs)),
        "--measured-runs",
        str(int(args.measured_runs)),
        "--timeout-seconds",
        str(int(args.timeout_seconds)),
        "--output",
        str(jsonl_path),
    ]
    if args.profile_backend:
        run_command.append("--profile-backend")
    if args.trace_python_allocations:
        run_command.append("--trace-python-allocations")
    if args.quality_check:
        run_command.append("--quality-check")

    report_command = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "report_qwen35_longbench_selector_compare.py"),
        "--input",
        str(jsonl_path),
        "--markdown-output",
        str(markdown_path),
        "--json-output",
        str(json_path),
        "--title",
        f"{args.model_id} LongBench {args.pack} Pack Compare",
    ]

    subprocess.run(run_command, cwd=str(REPO_ROOT), check=True)
    subprocess.run(report_command, cwd=str(REPO_ROOT), check=True)
    print(markdown_path)
    print(json_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
