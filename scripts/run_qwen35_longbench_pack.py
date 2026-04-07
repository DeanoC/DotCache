#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
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
PACK_PRESETS = {
    "original_suite": "original_full_suite",
    "stratified_16": "original_stratified_16_per_dataset",
    "stratified_32": "original_stratified_32_per_dataset",
}
COMPARISON_CASE_CHOICES = ["exact", "quality", "systems", "streaming_sink_recent", "quest_like"]
COMPARISON_CASE_PRESETS = {
    "full": COMPARISON_CASE_CHOICES,
    "paper_headline": ["exact", "systems", "streaming_sink_recent", "quest_like"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a named Qwen LongBench pack comparison.")
    parser.add_argument("--model-id", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--backend", default="torch_cuda")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch-dtype", default="float16")
    parser.add_argument("--selector-artifact", default=str(DEFAULT_SELECTOR_ARTIFACT))
    parser.add_argument("--pack", choices=sorted([*PACK_PATHS, *PACK_PRESETS]), default="original_suite")
    parser.add_argument("--prompt-pack", default=None)
    parser.add_argument("--comparison-cases", nargs="+", choices=COMPARISON_CASE_CHOICES, default=None)
    parser.add_argument("--comparison-case-preset", choices=sorted(COMPARISON_CASE_PRESETS), default=None)
    parser.add_argument("--prompt-shard-count", type=int, default=1)
    parser.add_argument("--prompt-shard-index", type=int, default=0)
    parser.add_argument("--max-prompt-tokens", type=int, nargs="+", default=[4096, 8192])
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--measured-runs", type=int, default=5)
    parser.add_argument("--timeout-seconds", type=int, default=2400)
    parser.add_argument("--profile-backend", action="store_true")
    parser.add_argument("--trace-python-allocations", action="store_true")
    parser.add_argument("--quality-check", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--skip-report", action="store_true")
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def resolve_prompt_pack(args: argparse.Namespace) -> Path:
    if args.prompt_pack:
        return Path(args.prompt_pack).expanduser().resolve()
    return PACK_PATHS[str(args.pack)].resolve()


def _sanitize_pack_label(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_") or "custom"


def resolve_pack_label(args: argparse.Namespace) -> str:
    if not args.prompt_pack:
        return str(args.pack)
    prompt_pack = resolve_prompt_pack(args)
    try:
        payload = json.loads(prompt_pack.read_text(encoding="utf-8"))
    except Exception:
        return _sanitize_pack_label(prompt_pack.stem)
    if isinstance(payload, dict):
        manifest_name = str(payload.get("manifest_name") or "").strip()
        phase = str(payload.get("phase") or "").strip()
        label_parts = [part for part in (_sanitize_pack_label(manifest_name), _sanitize_pack_label(phase)) if part]
        if label_parts:
            return "_".join(label_parts)
    return _sanitize_pack_label(prompt_pack.stem)


def model_slug(model_id: str) -> str:
    value = str(model_id).split("/")[-1].lower()
    return value.replace(".", "p")


def build_output_paths(
    *,
    output_dir: Path,
    model_id: str,
    pack: str,
    prompt_shard_count: int,
    prompt_shard_index: int,
) -> dict[str, Path]:
    slug = model_slug(model_id)
    shard_suffix = ""
    if int(prompt_shard_count) > 1:
        shard_suffix = f".shard{int(prompt_shard_index):02d}-of-{int(prompt_shard_count):02d}"
    jsonl_path = output_dir / f"{slug}_longbench_{pack}{shard_suffix}.jsonl"
    markdown_path = output_dir / f"longbench_selector_compare{shard_suffix}.md"
    json_path = output_dir / f"longbench_selector_compare{shard_suffix}.json"
    workbook_markdown_path = output_dir / f"longbench_failure_workbook{shard_suffix}.md"
    workbook_json_path = output_dir / f"longbench_failure_workbook{shard_suffix}.json"
    return {
        "jsonl": jsonl_path,
        "markdown": markdown_path,
        "json": json_path,
        "workbook_markdown": workbook_markdown_path,
        "workbook_json": workbook_json_path,
    }


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    pack_label = resolve_pack_label(args)
    output_paths = build_output_paths(
        output_dir=output_dir,
        model_id=args.model_id,
        pack=pack_label,
        prompt_shard_count=int(args.prompt_shard_count),
        prompt_shard_index=int(args.prompt_shard_index),
    )
    jsonl_path = output_paths["jsonl"]
    markdown_path = output_paths["markdown"]
    json_path = output_paths["json"]
    workbook_markdown_path = output_paths["workbook_markdown"]
    workbook_json_path = output_paths["workbook_json"]
    skip_report = bool(args.skip_report or int(args.prompt_shard_count) > 1)

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
    ]
    effective_comparison_cases = (
        list(args.comparison_cases)
        if args.comparison_cases
        else (
            list(COMPARISON_CASE_PRESETS[str(args.comparison_case_preset)])
            if args.comparison_case_preset
            else None
        )
    )
    if effective_comparison_cases:
        run_command.extend(["--comparison-cases", *effective_comparison_cases])
    run_command.extend(
        [
            "--prompt-shard-count",
            str(int(args.prompt_shard_count)),
            "--prompt-shard-index",
            str(int(args.prompt_shard_index)),
        ]
    )
    run_command.extend(
        [
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
    )
    if args.prompt_pack:
        run_command.extend(["--prompt-pack", str(resolve_prompt_pack(args))])
    elif args.pack in PACK_PATHS:
        run_command.extend(["--prompt-pack", str(resolve_prompt_pack(args))])
    else:
        run_command.extend(["--prompt-pack-preset", PACK_PRESETS[str(args.pack)]])
    if args.profile_backend:
        run_command.append("--profile-backend")
    if args.trace_python_allocations:
        run_command.append("--trace-python-allocations")
    if args.quality_check:
        run_command.append("--quality-check")
    else:
        run_command.append("--no-quality-check")

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
        f"{args.model_id} LongBench {pack_label} Pack Compare",
    ]
    workbook_command = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "report_qwen35_longbench_failure_workbook.py"),
        "--input",
        str(jsonl_path),
        "--markdown-output",
        str(workbook_markdown_path),
        "--json-output",
        str(workbook_json_path),
        "--title",
        f"{args.model_id} LongBench {pack_label} Failure Workbook",
    ]

    child_env = os.environ.copy()
    pythonpath_entries = [str(REPO_ROOT)]
    if child_env.get("PYTHONPATH"):
        pythonpath_entries.append(child_env["PYTHONPATH"])
    child_env["PYTHONPATH"] = os.pathsep.join(pythonpath_entries)

    subprocess.run(run_command, cwd=str(REPO_ROOT), env=child_env, check=True)
    if skip_report:
        print(jsonl_path)
        return 0
    subprocess.run(report_command, cwd=str(REPO_ROOT), env=child_env, check=True)
    subprocess.run(workbook_command, cwd=str(REPO_ROOT), env=child_env, check=True)
    print(markdown_path)
    print(json_path)
    print(workbook_markdown_path)
    print(workbook_json_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
